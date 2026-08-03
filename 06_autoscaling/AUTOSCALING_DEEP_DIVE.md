# Autoscaling LLM Inference

> Scaling GPU serving with demand — without wrecking latency or blowing the budget.

## Why This Matters

A web service scales cheaply: pods are small, start in seconds, and CPU utilization is a clean proxy for load. LLM inference breaks all three assumptions. A single replica pins one or more GPUs that cost more per hour than an entire fleet of CPU pods. A new replica must pull tens of gigabytes of weights and warm CUDA before it serves a single token, so "add capacity" is a **multi-minute** operation, not a multi-second one. And traffic is bursty — a Slack integration or a batch job can 10x your request rate in seconds.

Get autoscaling wrong and you fail in one of two expensive directions:

- **Under-provision:** the queue backs up, time-to-first-token (TTFT) climbs, requests time out, and by the time a new replica is ready the spike is over.
- **Over-provision:** you pay for idle H100s around the clock to hedge against a spike that comes twice a day.

This chapter is about threading that needle: scaling on the *right* signals, using the *right* mechanism (HPA, KEDA, Knative), and mitigating the cold-start tax that makes LLM autoscaling uniquely hard. For the metric definitions and the Prometheus/Grafana stack that feeds these controllers, see the [Monitoring chapter](../08_monitoring/README.md).

---

## Core Intuition: Why CPU-Based HPA Is Wrong for LLMs

The default Kubernetes HorizontalPodAutoscaler scales on CPU utilization. For a stateless web app that is a reasonable proxy: more requests → more CPU → scale up. For an LLM server it is actively misleading.

Consider what a vLLM or TGI process actually does. The heavy lifting happens on the **GPU**; the Python/host process spends most of its time waiting on CUDA kernels and shuffling tensors. So:

- A GPU that is **100% saturated** — KV cache full, requests queuing — can show **modest host CPU**. HPA sees "plenty of headroom" and refuses to scale while your p99 latency melts.
- A freshly loaded replica warming its cache can **spike CPU** while serving *nothing*, tricking HPA into scaling up when it shouldn't.

CPU utilization is decoupled from the thing you actually care about: **can I admit another request and still hit my latency SLO?** For LLMs the honest answer lives in queue depth, in-flight concurrency, KV-cache pressure, GPU utilization, and TTFT — never in host CPU.

The mental model: **scale on the length of the line, and on how long people wait in it — not on how busy the cashier's hands look.**

### The scaling control loop

Every mechanism below is the same loop with different parts swapped in. Keep it in your head:

```
  requests ──▶ [ vLLM replicas ] ──▶ metrics (queue depth, GPU util, TTFT)
                     ▲                          │
                     │                          ▼
              scale up / down          [ Prometheus / dcgm-exporter ]
                     ▲                          │
                     │                          ▼
             [ Deployment ] ◀── HPA / KEDA / KPA ◀── PromQL query vs target
```

The controller polls a metric, compares it to a target, and nudges the replica count. Everything interesting — which metric, which target, how fast to react, whether zero is allowed — is a knob on that loop.

---

## The Right Scaling Signals

There is no single perfect signal. Each trades responsiveness against noise and against how directly it maps to user-visible latency. Good production setups combine a **fast demand signal** (queue depth / concurrency) with an **SLO guardrail** (TTFT or p95 latency) so a breach of *either* forces a scale-up.

| Signal | Source | Pros | Cons |
|---|---|---|---|
| **Queue depth** (`vllm:num_requests_waiting`) | vLLM/TGI Prometheus metric | Directly reflects unmet demand; leads latency, so it's an *early* signal; cheap to compute | Zero when you're merely at capacity-but-coping; noisy for spiky traffic without smoothing |
| **In-flight / concurrency** (`vllm:num_requests_running`) | Engine metric or Knative KPA | Maps cleanly to a per-replica capacity target; stable | Saturates at the batch limit — can't tell "full" from "overwhelmed" alone |
| **GPU utilization** (`DCGM_FI_DEV_GPU_UTIL`) | NVIDIA `dcgm-exporter` | Hardware truth; catches non-vLLM workloads too | Lagging and coarse — 100% util can mean "efficiently batched" *or* "drowning"; poor sole signal |
| **KV-cache usage** (`vllm:gpu_cache_usage_perc`) | vLLM metric | Predicts imminent preemption/OOM before latency degrades | vLLM-specific; needs a sensible target (~90%) |
| **TTFT / p95 latency** (`vllm:e2e_request_latency_seconds`) | Engine histogram → `histogram_quantile` | Is the SLO — what users actually feel | **Lagging**: by the time it breaches, users are already hurting. Use as guardrail, not primary |
| **Requests per second** | Ingress / KPA | Simple, intuitive | Ignores request *size*; 10 long generations ≠ 10 short ones |
| **Batch/token throughput** | Engine metrics | Reflects real GPU work | Hard to set a stable target; varies with prompt length |

Rule of thumb: **lead with queue depth or concurrency, guard with a latency SLO, and treat GPU util as a sanity cross-check** — not as the trigger.

---

## Mechanism 1 — HPA with Custom / External Metrics

Kubernetes' HPA can scale on more than CPU. Since `autoscaling/v2` it supports three metric flavors:

- **Resource** — CPU/memory (the default; wrong for us).
- **Pods** — a custom per-pod metric averaged across pods (e.g. queue depth per replica).
- **Object / External** — a metric attached to another object or pulled from an external system (e.g. a Prometheus query).

To feed HPA a Prometheus metric you install the **Prometheus Adapter** (`prometheus-adapter`), which registers the `custom.metrics.k8s.io` / `external.metrics.k8s.io` APIs and translates HPA's metric requests into PromQL. GPU utilization itself comes from NVIDIA's **`dcgm-exporter`** (metric `DCGM_FI_DEV_GPU_UTIL`), scraped by Prometheus.

### Wiring the metric pipeline

HPA does not speak PromQL. The Prometheus Adapter bridges the gap: you give it a rule that maps a Kubernetes metric name to a query. A minimal rule exposing vLLM's queue depth as a per-pod custom metric looks like:

```yaml
# prometheus-adapter values.yaml (rules.custom[])
rules:
  custom:
    - seriesQuery: 'vllm:num_requests_waiting{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "vllm:num_requests_waiting"
        as: "vllm_num_requests_waiting"     # HPA-friendly name (no colon)
      metricsQuery: 'sum(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
```

Verify the metric is actually served before pointing HPA at it — a huge fraction of "HPA won't scale" incidents are just a missing or misnamed metric:

```bash
kubectl get --raw \
  "/apis/custom.metrics.k8s.io/v1beta1/namespaces/inference/pods/*/vllm_num_requests_waiting" | jq .
```

If that returns `no metrics returned from custom metrics API`, the problem is the pipeline (labels, series name, scrape) — not the HPA.

### The HPA algorithm

HPA computes desired replicas with a simple ratio:

\[
\text{desiredReplicas} = \left\lceil \text{currentReplicas} \times \frac{\text{currentMetricValue}}{\text{desiredMetricValue}} \right\rceil
\]

With multiple metrics, HPA computes a target for each and **takes the maximum** — the metric demanding the most replicas wins. That is exactly why a fast demand signal plus a latency guardrail composes well: whichever is more stressed drives the decision.

### Worked HPA manifest — scale on GPU utilization + queue depth

This assumes `dcgm-exporter` and `prometheus-adapter` are installed, and the adapter exposes `DCGM_FI_DEV_GPU_UTIL` as a Pods metric and `vllm_num_requests_waiting` as an External metric.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
  namespace: inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-llama3-8b
  minReplicas: 2            # provisioned floor — never cold-start the first request
  maxReplicas: 12           # capped by GPU quota (see pitfalls)
  metrics:
    # Primary demand signal: waiting requests per replica
    - type: Pods
      pods:
        metric:
          name: vllm_num_requests_waiting
        target:
          type: AverageValue
          averageValue: "5"        # aim to keep <5 queued per pod
    # Hardware cross-check: average GPU utilization
    - type: Pods
      pods:
        metric:
          name: DCGM_FI_DEV_GPU_UTIL
        target:
          type: AverageValue
          averageValue: "75"       # scale up past ~75% average util
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0     # react to spikes immediately
      policies:
        - type: Pods
          value: 4                      # add up to 4 pods...
          periodSeconds: 60             # ...per minute
        - type: Percent
          value: 100                    # ...or double, whichever is larger
          periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300   # wait 5 min of calm before shrinking
      policies:
        - type: Pods
          value: 1                      # remove at most 1 pod...
          periodSeconds: 120            # ...every 2 min — GPUs are expensive to churn
      selectPolicy: Min
```

**Tuning notes.**

- **`scaleUp.stabilizationWindowSeconds: 0`** — scale up on the freshest reading. Because cold starts are slow, hesitating on scale-up is the costliest mistake you can make.
- **`scaleDown.stabilizationWindowSeconds: 300`** — HPA takes the *highest* recommendation over the window before shrinking, so a 5-minute window prevents a brief traffic lull from tearing down a replica you'll re-pay a cold start to rebuild.
- **Asymmetric policies** — aggressive up, gentle down (one pod every two minutes). This is the opposite of a cost-first web-app config, and it's deliberate: for GPUs, *flapping is more expensive than a little idle*.
- **Target values are per-pod averages** — `averageValue: "5"` means HPA aims for 5 waiting requests per replica, so the ratio math scales linearly with fleet size.

---

## Mechanism 2 — KEDA for Event-Driven & Queue-Based Scaling

HPA + Prometheus Adapter works but is fiddly: you maintain adapter rules, and HPA alone **cannot scale to zero**. **KEDA** (Kubernetes Event-Driven Autoscaling) sits on top of HPA and fixes both. It ships 70+ **scalers** (Prometheus, Kafka, SQS, RabbitMQ, Redis, ...) and, crucially, can scale a deployment **from 0 → 1 and back to 0**.

KEDA introduces two ideas HPA lacks:

- **`activationThreshold`** — the value that flips a workload *from zero to one*. This is separate from the scaling `threshold` (which governs 1→N). It exists precisely so a single stray request doesn't wake a cold GPU, and so a trickle doesn't keep one warm.
- **`minReplicaCount: 0`** — legal in KEDA, impossible in raw HPA.

### Worked KEDA ScaledObject — queue depth + latency guardrail

This mirrors the pattern AWS documents for vLLM on EKS: a primary queue-depth trigger and a p95-latency guardrail, scaling to satisfy whichever demands more replicas.

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler
  namespace: inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-llama3-8b
  minReplicaCount: 1          # warm floor; set 0 only if cold starts are acceptable
  maxReplicaCount: 12
  pollingInterval: 15         # query Prometheus every 15s
  cooldownPeriod: 300         # after last trigger, wait 5 min before scaling toward min
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:               # KEDA passes this straight through to the HPA it manages
        scaleDown:
          stabilizationWindowSeconds: 300
        scaleUp:
          stabilizationWindowSeconds: 0
  triggers:
    # Primary: queue depth (waiting requests), averaged per replica
    - type: prometheus
      metricType: AverageValue
      metadata:
        serverAddress: http://kube-prometheus-stack-prometheus.monitoring.svc:9090
        query: sum(vllm:num_requests_waiting) or vector(0)
        threshold: "5"
        activationThreshold: "1"     # first waiting request wakes the deployment
    # Guardrail: p95 end-to-end latency in seconds
    - type: prometheus
      metricType: AverageValue
      metadata:
        serverAddress: http://kube-prometheus-stack-prometheus.monitoring.svc:9090
        query: |
          histogram_quantile(0.95,
            sum(rate(vllm:e2e_request_latency_seconds_bucket[1m])) by (le)) or vector(0)
        threshold: "5"               # scale up if p95 latency exceeds 5s
```

**Tuning notes.**

- **`or vector(0)`** is not decoration — if the query returns *no series* (e.g. the deployment is scaled to zero and exporting nothing), KEDA would otherwise error. `or vector(0)` yields a clean `0`, which is exactly the "no demand" reading you want.
- **`metricType: AverageValue`** divides the query result by the current replica count so the target is per-pod; use `Value` only when your query already returns a per-pod figure.
- **`activationThreshold: "1"` vs `threshold: "5"`** — one waiting request is enough to justify the *first* replica; you only add *more* replicas once the per-pod queue passes 5.
- **`cooldownPeriod`** governs the final ramp toward `minReplicaCount` (including the drop to zero); the HPA `behavior` block governs the 1→N steps.
- Prefer a **dedicated queue metric** (`num_requests_waiting`) over latency as the *primary* — latency lags, and by the time p95 breaches, the SLO is already violated. Latency is the seatbelt, not the accelerator.

---

## Mechanism 3 — Scale-to-Zero, Cold Starts, and Serverless GPU

Scale-to-zero is the dream: pay nothing when idle. For LLMs it collides head-on with the **cold-start problem**.

### Anatomy of an LLM cold start

When a scaled-to-zero deployment gets a request, the clock runs through:

1. **Scheduling** — Kubernetes finds a node with a free GPU (seconds → minutes if the cluster autoscaler must add a node).
2. **Image pull** — the container image is often 5–15 GB (CUDA, PyTorch, vLLM). Seconds to minutes if not cached on the node.
3. **Weight load** — read tens of GB of weights from disk/network into host RAM, then copy to VRAM. **This dominates** — often the largest single chunk.
4. **CUDA / engine warm-up** — initialize CUDA context, compile/capture CUDA graphs, allocate the KV cache.

For a mid-size model this is routinely **1–5 minutes**, and can be far worse if the cluster autoscaler has to boot a fresh GPU node first. That is an eternity for an interactive request. So true scale-to-zero is only acceptable for **latency-tolerant** workloads (batch, internal tools, dev). For anything user-facing, you keep a warm floor.

### Cold-start mitigation comparison

| Mitigation | How it works | Cold-start impact | Cost | Best for |
|---|---|---|---|---|
| **Provisioned min replicas** (`minReplicas`/`minReplicaCount` ≥ 1) | Never fully scale down; keep N warm | Eliminates it for the first N concurrent requests | Highest — you pay for idle GPUs | Interactive, SLA-bound traffic |
| **Warm pool / over-provision headroom** | Keep spare *ready* replicas ahead of demand (e.g. +1 buffer) | New traffic hits an already-warm pod | Medium — pay for the buffer only | Predictable spikes, autoscaling with slack |
| **Faster weight loading** (Run:ai Model Streamer, tensorizer, safetensors + fast storage) | Stream weights concurrently from object storage straight to GPU; skip slow deserialization | Cuts the dominant *load* phase (reported up to ~6x) | Low — engineering only | Every setup; stacks with others |
| **Node/image pre-pull & DaemonSet cache** | Pre-pull the container image and warm node caches | Removes image-pull phase | Low | Large images, node churn |
| **Snapshot / checkpoint-restore** (NVIDIA Dynamo snapshot, CUDA checkpoint/CRIU) | Snapshot a *warmed* process (CUDA context + weights in VRAM) and restore it | Can approach near-zero — skips load *and* warm-up | Medium; newer/less mature | Aggressive scale-to-zero without the latency tax |
| **Smaller/quantized model or smaller shards** | Fewer bytes to move and initialize | Proportionally shorter load | Free-ish (accuracy tradeoff) | When quality budget allows |

**Knative / serverless GPU.** Knative Serving offers request-driven autoscaling with native scale-to-zero. Its default **KPA (Knative Pod Autoscaler)** scales on **concurrency** or **RPS** rather than CPU — a much better fit for LLMs than raw HPA. Key pieces:

- **`containerConcurrency`** / target concurrency — the per-replica in-flight target KPA scales to maintain.
- **The activator** buffers requests while a scaled-to-zero service spins up, so requests aren't dropped — they're *held* (and pay the cold-start latency).
- **Panic mode / target-burst-capacity** — when traffic spikes sharply, KPA enters a short "panic" window and scales on a much shorter horizon to react fast, then relaxes.

Knative is elegant for bursty, latency-tolerant serving, but the activator's request buffering doesn't *erase* the cold start — it just prevents dropped requests. You still pay the minutes. Pair scale-to-zero with a snapshot/fast-load strategy, or keep `minScale ≥ 1` for interactive paths.

### Worked Knative Service — concurrency-driven with a warm floor

The same workload as a Knative Service, scaling on concurrency with KPA. Note `minScale: 1` — a warm floor that dodges the cold start on the interactive path while still capping cost with `maxScale`.

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: vllm-llama3-8b
  namespace: inference
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/class: "kpa.autoscaling.knative.dev"
        autoscaling.knative.dev/metric: "concurrency"
        autoscaling.knative.dev/target: "8"        # ~8 in-flight requests per replica
        autoscaling.knative.dev/target-utilization-percentage: "80"
        autoscaling.knative.dev/min-scale: "1"     # warm floor (set 0 for scale-to-zero)
        autoscaling.knative.dev/max-scale: "12"
        autoscaling.knative.dev/scale-down-delay: "5m"   # hold before shrinking
        autoscaling.knative.dev/target-burst-capacity: "200"  # activator buffers bursts
    spec:
      containerConcurrency: 16     # hard per-replica ceiling
      containers:
        - image: vllm/vllm-openai:latest
          resources:
            limits:
              nvidia.com/gpu: "1"
```

**Tuning notes.**

- **`target: 8` vs `containerConcurrency: 16`** — KPA aims to keep ~8 concurrent requests per replica (the soft target) while 16 is the hard cap. Keeping the soft target well below the hard cap leaves slack for the seconds before a new replica is ready.
- **`target-burst-capacity`** decides how much spike the activator absorbs (buffering requests) before KPA has scaled out; higher values route more traffic through the activator, trading a little steady-state latency for burst safety.
- **`scale-down-delay`** is Knative's answer to flapping — the KPA equivalent of HPA's scale-down stabilization window.
- Set **`min-scale: 0`** only when the workload tolerates the cold start; the activator will hold the request but the user still waits out the load.

### Which mechanism should I pick?

| Situation | Reach for | Why |
|---|---|---|
| Simple custom-metric scaling, floor ≥ 1, existing Prometheus | **HPA + Prometheus Adapter** | Fewest moving parts; native; no scale-to-zero needed |
| Queue/event-driven, scale-to-zero, many metric sources | **KEDA** | `activationThreshold`, 0→1, and 70+ scalers wrap HPA cleanly |
| Concurrency-driven serverless with request buffering | **Knative (KPA)** | Built-in scale-to-zero + activator; concurrency target fits LLMs |
| Bursty, latency-tolerant, want managed cold-start buffering | **Knative** | Activator holds requests during spin-up so nothing drops |
| Interactive, strict TTFT SLO | **Any + warm floor** | The mechanism matters less than never cold-starting the hot path |

These aren't mutually exclusive: a common production shape is **KEDA for the demand-driven scale (including 0→1) plus a provisioned floor for the interactive tier**, with all three fed by the same Prometheus pipeline.

---

## The Cost vs Latency Tradeoff

Every autoscaling decision is a bet on this tradeoff. Warm replicas cost money every second they're idle; cold replicas cost *latency* (and lost requests) every time you're caught short. You can make the tradeoff explicit.

### A worked calculation

Suppose one H100 replica costs about \( \$3.50 \) per hour and serves a steady \( 20 \) requests/second at your latency SLO. Your traffic is \( 20 \) req/s for 8 business hours and near-zero the other 16.

**Option A — always-on flat fleet.** Provision for peak, run 24/7:

\[
\text{Cost}_A = 1 \text{ replica} \times \$3.50/\text{hr} \times 24 \text{ hr} = \$84 \text{ per day}
\]

**Option B — autoscale with a warm floor.** Keep `minReplicas = 1` only during the 8 busy hours, scale to zero otherwise. Assume the 16 idle hours truly cost nothing:

\[
\text{Cost}_B = 1 \times \$3.50 \times 8 = \$28 \text{ per day}
\]

That's a **67% saving** — but it buys a cold start on the *first* request each morning and after any midday lull. If the SLO forbids a multi-minute first response, you instead keep a warm floor 24/7 and land back near Option A, or you spend engineering effort on snapshot/fast-load so scale-to-zero becomes safe.

**Headroom sizing.** To absorb bursts without waiting on a cold start, provision a buffer. If your scale-up (schedule + pull + load + warm) takes \( T_{cold} = 180 \) s and traffic can climb at \( 2 \) req/s², a replica in flight can't help for 3 minutes. Size the warm buffer to cover demand growth over \( T_{cold} \):

\[
\text{buffer replicas} = \left\lceil \frac{\Delta(\text{req/s over } T_{cold})}{\text{capacity per replica}} \right\rceil
\]

The general shape: **provisioned floor covers the baseline, headroom buffer covers what arrives during a cold start, and autoscaling handles the sustained ramp.** Tune the floor and buffer to the *cost of a missed SLO*, not to a generic utilization target.

---

## Failure Modes & Pitfalls

- **Flapping (thrashing).** Symmetric or twitchy thresholds scale up and down every few minutes, and each cycle pays a cold start. **Fix:** long `scaleDown.stabilizationWindowSeconds` (300s+), conservative scale-down policies, and a generous `cooldownPeriod`. For GPUs, err toward *stickiness*.
- **Scaling on a lagging metric.** If p95 latency or GPU utilization is your *primary* trigger, you scale *after* users are already hurting, and because cold starts are slow you stay behind the curve for minutes. **Fix:** lead with a *leading* signal (queue depth / concurrency); keep latency as a guardrail only.
- **Thundering herd on cold model load.** A big spike triggers many replicas at once; they *simultaneously* hammer the same weights bucket / registry, saturating network and disk, so **all** of them cold-start slower. **Fix:** cap `scaleUp` step size (`Pods: value` / `periodSeconds`), stagger with fast-load streaming, pre-warm images, and cache weights on nodes.
- **GPU quota / capacity ceilings.** `maxReplicas` is a wish; cloud GPU quota and actual availability are the reality. HPA will happily request 12 pods that sit `Pending` forever because there are no H100s to schedule them on. **Fix:** set `maxReplicas` to your *real* quota, alert on `Pending` GPU pods, and combine with cluster-autoscaler node pools sized to quota.
- **Metric pipeline is a hidden dependency.** If Prometheus, the adapter, or `dcgm-exporter` hiccups, HPA/KEDA see stale or missing metrics and may freeze or over-react. **Fix:** `or vector(0)` guards, `ignoreNullValues`, alert on the metrics pipeline itself, and a sane default replica count.
- **Per-pod vs total metric confusion.** Using `Value` where you meant `AverageValue` (or a raw `sum()` without dividing by replicas) makes the target scale wrong as the fleet grows — you either never scale or scale to the moon. **Fix:** be explicit about per-pod semantics and test with a load generator.
- **Scale-down mid-generation.** Terminating a pod that's mid-stream kills in-flight long generations. **Fix:** graceful termination with a drain period longer than your max generation time, and `terminationGracePeriodSeconds` sized accordingly.

---

## Production Checklist — What an Interviewer Probes

1. **"Why not CPU-based HPA for an LLM?"** — Host CPU is decoupled from GPU saturation; a full GPU can look idle to HPA. Name the real signals: queue depth, concurrency, KV-cache, GPU util, TTFT.
2. **"What's your primary scaling signal and why?"** — A *leading* demand signal (`vllm:num_requests_waiting` / concurrency), with a *lagging* latency SLO as a guardrail, and the max-of-metrics behavior that composes them.
3. **"How do you handle cold starts?"** — Quantify the phases (schedule, pull, load, warm), then name concrete mitigations: warm floor, headroom buffer, fast weight streaming, snapshot/checkpoint-restore, image pre-pull.
4. **"Scale to zero — yes or no?"** — "It depends on the SLO." Fine for batch/internal; dangerous for interactive unless snapshotting makes cold starts sub-second. Explain KEDA's `activationThreshold`.
5. **"How do you stop it flapping?"** — Asymmetric behavior: aggressive scale-up (0s window), conservative scale-down (300s+ window, one pod at a time), cooldown.
6. **"HPA vs KEDA vs Knative — when each?"** — HPA+adapter for simple custom-metric scaling; KEDA for event/queue-driven and scale-to-zero on any metric; Knative/KPA for concurrency-driven serverless with request buffering.
7. **"What breaks under a real traffic spike?"** — Thundering herd on weight load, GPU quota ceilings leaving pods `Pending`, and lagging-metric lock-step. Have a mitigation for each.
8. **"How do you pick the target value / threshold?"** — Derive it from load tests (see the [Load Testing chapter](../04_load_testing/README.md)): find the per-replica concurrency/queue depth at which TTFT just meets SLO, then set the target below it.

---

## Further Reading

- Kubernetes — [Horizontal Pod Autoscaler (v2, behavior & algorithm)](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- Kubernetes — [HPA Walkthrough with custom metrics](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/)
- KEDA — [Prometheus scaler](https://keda.sh/docs/latest/scalers/prometheus/) and [Scaling Deployments (activation vs threshold, scale-to-zero)](https://keda.sh/docs/latest/concepts/scaling-deployments/)
- AWS — [Autoscale AI inference with HPA and KEDA on EKS (vLLM)](https://docs.aws.amazon.com/eks/latest/userguide/ml-inference-autoscaling-hpa-keda.html)
- vLLM — [Autoscaling with KEDA (production-stack)](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html)
- Knative — [Autoscaling (KPA, concurrency, scale-to-zero, panic mode)](https://knative.dev/docs/serving/autoscaling/)
- Prometheus Adapter — [kubernetes-sigs/prometheus-adapter](https://github.com/kubernetes-sigs/prometheus-adapter)
- NVIDIA — [dcgm-exporter (GPU metrics for Prometheus)](https://github.com/NVIDIA/dcgm-exporter)
- NVIDIA — [Dynamo Snapshot: fast startup for inference on Kubernetes](https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/)
- Microsoft Azure — [Eliminate LLM cold starts: load models up to 6x faster with Run:ai Model Streamer](https://devblogs.microsoft.com/azure-sdk/eliminate-llm-cold-starts-load-models-up-to-6x-faster-with-azure-blob-storage-and-runai-model-streamer/)
- KServe — [Autoscaler for generative inference](https://kserve.github.io/website/docs/model-serving/generative-inference/autoscaling)

---

*Related chapters: [Load Testing](../04_load_testing/README.md) to derive your target thresholds, [vLLM Serving](../05_vllm_serving/README.md) for the engine metrics, and [Monitoring](../08_monitoring/README.md) for the Prometheus/Grafana pipeline that feeds every controller above.*
