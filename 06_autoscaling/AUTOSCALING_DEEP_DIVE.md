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

*(The GPU-memory pre-allocation and CPU/GPU decoupling claims above are corroborated by field reports on running vLLM under Kubernetes autoscaling — see the DEV Community write-up cited in War Story 1 and the Further Reading list.)*

> **The 60-second interview answer.** "CPU-based HPA scales on host CPU utilization, but an LLM server's real bottleneck is the GPU, not the host. vLLM pre-allocates most of its GPU memory for the KV cache at startup, so GPU memory looks flat whether you're idle or saturated, and the host process spends its time waiting on CUDA kernels, so CPU stays low even when every request is queuing. That means the metric HPA trusts by default — CPU — is decoupled from the thing that actually determines whether a new request gets served on time. Instead you scale on signals that reflect GPU-side demand directly: queue depth (`num_requests_waiting`), in-flight concurrency, KV-cache utilization, and GPU utilization from `dcgm-exporter`, with TTFT or p95 latency as a lagging guardrail rather than the primary trigger. The one-line version: scale on the length of the line, not on how busy the cashier's hands look."

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

**A note on continuous batching.** vLLM and similar engines use continuous (iteration-level) batching — new requests join a running batch between decode steps rather than waiting for the current batch to fully finish, and the batch composition changes token by token. This is *why* per-request metrics like "requests per second" undercount the real picture: two requests generating 20 tokens each and one request generating 2,000 tokens can produce an identical RPS reading while representing wildly different GPU-second costs. Prefer signals that reflect the batching engine's own internal state (queue depth, in-flight count, KV-cache usage) over signals computed purely from request *arrival*, since the engine's internal state is what continuous batching is actually managing.

**A note on multi-model and multi-LoRA serving.** A replica serving several LoRA adapters (or several small models) behind one endpoint doesn't have a single, fixed "capacity per replica" — it has one *per adapter/model combination currently loaded*, and swapping adapters costs time. Queue-depth and concurrency targets derived from a load test of one model in isolation can silently overstate real capacity once several are multiplexed onto the same GPU. If your fleet serves more than one model per replica, re-run the load-test-derived-threshold exercise above against the actual multi-model traffic mix, not a single-model benchmark.

**A note on GPU sharing and MIG.** NVIDIA's Multi-Instance GPU (MIG) partitions a single physical GPU into several smaller, hardware-isolated instances, and time-slicing shares a GPU across pods without hardware partitioning. Either changes the unit this entire chapter has been scaling: "a replica" no longer maps 1:1 to "a physical GPU," so `nvidia.com/gpu: 1` in a pod spec might mean a full H100 or a fraction of one depending on the node's MIG configuration. Before applying any of this chapter's cost or headroom math to a MIG-partitioned fleet, confirm which of "replica," "GPU instance," and "physical GPU" your `maxReplicas`/quota numbers actually refer to — the three are easy to conflate and the resulting error compounds directly into the warm-pool sizing calculation above.

---

### Signal composition patterns seen in production

Three combinations recur often enough to name:

1. **Queue depth (primary) + latency guardrail (safety net).** The default recommendation throughout this chapter. Queue depth reacts before users feel anything; latency confirms the SLO is actually being met and catches anything queue depth misses.
2. **Queue depth (primary) + KV-cache usage (early-warning) + latency (guardrail).** The three-signal composition War Story 1 argues for — necessary once long-context or highly variable-length requests are common enough that KV-cache pressure can spike faster than the queue does.
3. **Per-pool signals in disaggregated serving** — KV-cache utilization for decode, prefill-queue depth for prefill, scaled and alerted on independently (see the Landscape section's disaggregated-serving subsection). This is pattern 1 or 2, applied twice, once per pool.

The unifying rule is the same one this chapter opened with: pick signals close to the actual bottleneck, use the fastest one as the trigger, and keep the SLO itself as a guardrail rather than the primary control variable.

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

**Plugging in numbers.** Suppose a deployment currently runs 3 replicas, and the queue-depth metric averages 15 waiting requests per pod against a target of 5:

\[
\text{desiredReplicas} = \left\lceil 3 \times \frac{15}{5} \right\rceil = \left\lceil 9 \right\rceil = 9
\]

If, in the same cycle, the GPU-utilization metric only computes a desired count of 6, HPA takes the max of the two — 9 replicas, driven by queue depth — and that's the number that actually gets applied. This is the concrete mechanism behind "whichever signal is more stressed wins": it isn't a qualitative preference, it's this arithmetic, evaluated per metric, every sync.

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

### Troubleshooting — "my HPA won't scale"

In rough order of how often each is actually the culprit:

1. **The custom/external metric isn't being served at all.** Confirm with the `kubectl get --raw` check earlier in this section before touching the HPA object itself — most "HPA is broken" reports are a missing or misnamed metric one layer down.
2. **`AverageValue` vs `Value` mismatch.** A `sum()` query paired with `type: Value` (expecting a per-pod figure) makes the target drift as the fleet grows; verify which one your PromQL actually returns.
3. **The metric is real but the target is unreachable.** If `averageValue` is set far below what the workload can ever realistically achieve, HPA will happily recommend `maxReplicas` forever — sanity-check the target against a load test, not intuition.
4. **`minReplicas`/`maxReplicas` boundaries are silently constraining the recommendation** — `kubectl describe hpa` shows the computed vs. constrained value; a HPA "stuck" at `maxReplicas` is doing its job, the ceiling is just the bottleneck (see the GPU-quota pitfall and War Story 3).
5. **The `behavior` block's stabilization window is masking a real signal** — during initial rollout, temporarily set `scaleDown.stabilizationWindowSeconds` low to confirm the underlying metric-to-replica math works at all, then restore the production value.
6. **Multiple metrics disagree and the wrong one is winning** — remember HPA takes the *max* recommendation across metrics; if a stale or misconfigured secondary metric is pinned high, it can override a healthy primary signal.

## Mechanism 2 — KEDA for Event-Driven & Queue-Based Scaling

HPA + Prometheus Adapter works but is fiddly: you maintain adapter rules, and HPA alone **cannot scale to zero**. **KEDA** (Kubernetes Event-Driven Autoscaling) sits on top of HPA and fixes both. It ships 70+ **scalers** (Prometheus, Kafka, SQS, RabbitMQ, Redis, ...) and, crucially, can scale a deployment **from 0 → 1 and back to 0**.

KEDA introduces two ideas HPA lacks:

- **`activationThreshold`** — the value that flips a workload *from zero to one*. This is separate from the scaling `threshold` (which governs 1→N). It exists precisely so a single stray request doesn't wake a cold GPU, and so a trickle doesn't keep one warm.
- **`minReplicaCount: 0`** — legal in KEDA, impossible in raw HPA.

### `ScaledObject` vs `ScaledJob`

Everything above uses KEDA's `ScaledObject`, which manages an HPA behind a `Deployment` — the right shape for a long-lived pool of replicas serving live requests. KEDA also ships `ScaledJob`, which creates a Kubernetes `Job` per unit of work instead of managing replica count on a `Deployment` — the right shape for the batch-summarization system-design prompt later in this chapter: each queued item becomes its own Job, scaled by the same queue-depth-style triggers (SQS/Kafka/Redis length), with no notion of a "replica count" to tune stabilization windows for at all. Picking between them is really a question of workload shape, not a scaling-signal question: **`ScaledObject` for a pool of servers accepting live requests, `ScaledJob` for a stream of discrete, run-to-completion units of work.**

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

### Snapshot and checkpoint-restore, in more depth

The cold-start table above lists snapshot/checkpoint-restore as the mitigation that can approach near-zero cold starts by skipping *both* the weight-load and warm-up phases, not just one. Worth unpacking why: a "cold" replica pays for weight load (reading tens of GB into VRAM) and warm-up (CUDA context init, graph capture, KV-cache allocation) every single time it starts, even though both produce an *identical* end state for a given model and configuration. Snapshotting captures that end state once — a already-warmed process, weights resident in VRAM, CUDA context initialized — and restores it directly on subsequent starts, turning "redo the whole boot sequence" into "load a pre-made snapshot."

Two flavors show up in practice:

- **Process/CUDA-context snapshotting** (NVIDIA Dynamo Snapshot, CUDA checkpoint combined with CRIU-style process checkpointing) — snapshots the actual running process, GPU memory included, and restores it wholesale. Fastest in principle, since nothing is recomputed; newest and least battle-tested of the mitigations in this chapter, and typically tied to a specific engine/driver version.
- **Fast-load without full process snapshotting** (Run:ai Model Streamer and similar) — still runs a normal boot sequence, but makes the weight-load phase itself fast by streaming concurrently rather than sequentially. Less fragile than full snapshotting, and the mitigation with the clearest, most reproducible published numbers (the ~6–7.6x figures in the Landscape section above) — which is part of why it has become closer to a default than snapshotting has, as of this writing.

A practical stance: reach for fast weight loading first — it's lower-risk, stacks with every other mitigation in this chapter, and is now natively supported by vLLM. Reach for full process/CUDA snapshotting only when the SLO genuinely requires sub-few-second cold starts and the engineering cost of maintaining a more fragile, version-pinned snapshot pipeline is worth it — which is exactly the tradeoff the fastest serverless GPU platforms in the Landscape section's cold-start comparison table have already made on your behalf, if you'd rather not build it yourself.

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

## The 2025–2026 Landscape

Autoscaling for LLM serving moved from "borrow the web-app playbook" to a purpose-built discipline in 2025–2026. Four threads matter if you're building or defending a design today.

### KEDA's LLM-relevant scalers keep expanding

- The **Prometheus scaler** (used throughout this chapter) remains the workhorse; current docs are at KEDA v2.20 ([keda.sh/docs/2.20/scalers/prometheus](https://keda.sh/docs/2.20/scalers/prometheus/)).
- The **Cron scaler** ([keda.sh/docs/2.20/scalers/cron](https://keda.sh/docs/2.20/scalers/cron/), available since KEDA v1.5) lets you define named `start`/`end` cron windows with a `desiredReplicas` and IANA `timezone`. It does **not** run on a recurring implicit schedule — it only activates within the explicit windows you give it — which makes it the right tool for genuinely calendar-shaped demand (business hours, known batch windows). A `ScaledObject` can carry a Cron trigger *and* a Prometheus trigger simultaneously; KEDA scales to satisfy whichever trigger currently demands more, the same max-of-metrics idea as HPA's multi-metric behavior. See the worked example below.
- The KEDA community is actively debating going further. GitHub issue [kedacore/keda#6934](https://github.com/kedacore/keda/issues/6934) (opened 2026) proposes an "LLM Scaler" — nicknamed **Cognitive Scaling** — that would feed unstructured signals (news feeds, social sentiment, support-ticket volume) through an LLM to produce a scaling metric *before* the effect shows up in queue depth at all, e.g. scaling ahead of a product launch or a viral moment. KEDA maintainer JorTurFer's counter-proposal — "scaling modifiers" that blend an LLM's judgment with existing scaler outputs rather than shipping a bespoke scaler — is the more likely direction: augment reactive metrics with synthesized context, don't replace them. As of this writing it is a proposal under discussion, not a shipped feature — cite it as "where the conversation is going," not as production-ready.
- **Kedify** ([kedify.io](https://kedify.io)), founded by core KEDA maintainers, layers a proprietary **predictive autoscaling** feature on top of open-source KEDA: a `MetricPredictor` CRD forecasts a metric using Facebook Prophet, retrains on a configurable cadence (e.g. every six hours), validates itself against a held-out window using Mean Absolute Percentage Error (MAPE), and automatically falls back to the raw, non-predicted metric when forecast confidence drops ([kedify.io/resources/blog/predictive-autoscaling](https://kedify.io/resources/blog/predictive-autoscaling/), published Oct 23 2025). This is the productionized version of "look at yesterday's shape, not just this second's queue depth" — it augments, rather than replaces, the reactive Prometheus trigger.
- Azure's own AKS engineering team published a worked example of KEDA driving GPU inference autoscaling for KAITO-hosted models on AKS ([blog.aks.azure.com/2026/02/03/autoscale-inference-workloads-with-kaito](https://blog.aks.azure.com/2026/02/03/autoscale-inference-workloads-with-kaito), Feb 3 2026) — evidence that "KEDA + Prometheus + GPU inference" is now a documented, vendor-supported pattern, not just a DIY recipe.

### Serverless GPU platforms have converged on "seconds, not minutes"

An Aug 15 2025 comparison of serverless GPU inference platforms ranked them by cold-start latency ([beam.cloud/blog/top-serverless-gpu-providers](https://www.beam.cloud/blog/top-serverless-gpu-providers)):

| Platform | Reported cold start | Mechanism note |
|---|---|---|
| Beam | ~2–3 s | Custom `beta9` container runtime, not a generic Docker boot path |
| RunPod Serverless | 6–12 s | Snapshot-style "FlashBoot" fast-start path |
| Google Cloud Run (GPU) | 20–30 s | Standard container cold boot, GPU attach |
| Baseten | 16–60 s | Varies by model/deployment configuration |
| Replicate | instant for cached public models; 60+ s for custom deployments | Cache hit vs. cache miss |

The common thread at the fast end is **not using a generic Docker-then-CUDA-init boot sequence** — Beam's and RunPod's fast paths both skip large parts of the standard sequence that a vanilla Kubernetes pod pays for, which is exactly the "snapshot/checkpoint-restore" row in this chapter's cold-start table. It's no longer a research curiosity; it's how the fastest commercial platforms hit single-digit-second cold starts today. If you're building on raw Kubernetes rather than adopting one of these platforms outright, treat their numbers as the bar your users will implicitly compare you against.

### Model-weight streaming keeps shrinking the dominant cold-start phase

NVIDIA's **Run:ai Model Streamer** — an open-source Python SDK with a multi-threaded C++ backend — concurrently reads weight shards from storage while overlapping the CPU→GPU copy, instead of the default "read fully into host RAM, then copy" path. Its own published benchmarks on a 15 GB Llama 3 8B checkpoint ([developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer](https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer), Sept 16 2025):

| Source | Baseline loader | Model Streamer | Speedup |
|---|---|---|---|
| IO2 SSD, concurrency 8 | 47 s (Safetensors) | 7.53 s | ~6.2x |
| GP3 SSD, concurrency 16 | — | 14.34 s | — |
| Amazon S3, concurrency 32 | 37.36 s (Tensorizer) | 4.88 s | ~7.6x |

That "up to 6x" figure is the same one cited in Microsoft's Azure Blob Storage integration write-up ([devblogs.microsoft.com/azure-sdk/eliminate-llm-cold-starts-load-models-up-to-6x-faster-with-azure-blob-storage-and-runai-model-streamer](https://devblogs.microsoft.com/azure-sdk/eliminate-llm-cold-starts-load-models-up-to-6x-faster-with-azure-blob-storage-and-runai-model-streamer/)), and the mechanism is now wired natively into vLLM (`--load-format runai_streamer`, [docs.vllm.ai/en/stable/models/extensions/runai_model_streamer](https://docs.vllm.ai/en/stable/models/extensions/runai_model_streamer/)) and into GKE's model-loading path ([cloud.google.com/blog/products/containers-kubernetes/nvidia-runai-model-streamer-supports-cloud-storage](https://cloud.google.com/blog/products/containers-kubernetes/nvidia-runai-model-streamer-supports-cloud-storage)). Azure's AKS engineering blog (July 13 2026: [blog.aks.azure.com/2026/07/13/runai-streamer-vllm](https://blog.aks.azure.com/2026/07/13/runai-streamer-vllm)) walks through wiring the same streamer to Azure Blob for AKS-hosted vLLM. The practical takeaway for the cold-start table above: weight streaming is close to a default now, not a niche optimization — and it directly attacks the phase (weight load) that dominates the 1–5 minute cold-start estimate.

### Predictive and scheduled scaling for known traffic patterns

Two complementary tools have matured for demand that isn't a pure surprise:

1. **KEDA's Cron scaler** for genuinely scheduled patterns (business hours, batch windows, known regional peaks) — deterministic, no ML required, composable with a Prometheus trigger in the same `ScaledObject` (worked example below).
2. **Forecast-based predictive scaling** (Kedify's `MetricPredictor`, and the broader pattern described by observability vendors such as Sedai, [sedai.io/blog/predictive-autoscaling-in-kubernetes](https://sedai.io/blog/predictive-autoscaling-in-kubernetes)) for patterns that are *regular but not calendar-fixed* — a shape that correlates with a marketing calendar or a usage cadence that drifts week to week. These layer a forecast on top of, not instead of, the reactive queue-depth signal: the forecast pre-warms capacity, the reactive metric still governs the fine-grained ramp.

### Disaggregated serving and purpose-built LLM autoscalers

So far this chapter has treated "a replica" as the atomic scaling unit. Disaggregated serving — splitting the compute-bound **prefill** phase from the memory-bandwidth-bound **decode** phase onto separate GPU pools — breaks that assumption, and 2025–2026 tooling has started to bake autoscaling directly into the serving stack rather than leaving it entirely to HPA/KEDA:

- **NVIDIA Dynamo's Planner** makes *independent* scaling decisions for prefill and decode pools using phase-appropriate metrics: it monitors average KV-cache block utilization across decode GPUs, and separately tracks the depth of a global pending-request queue for prefill, comparing each against its own configurable threshold before shifting GPUs between pools or provisioning new ones from a shared pool (NVIDIA developer blog, May 20 2025: [developer.nvidia.com/blog/nvidia-dynamo-adds-gpu-autoscaling-kubernetes-automation-and-networking-optimizations](https://developer.nvidia.com/blog/nvidia-dynamo-adds-gpu-autoscaling-kubernetes-automation-and-networking-optimizations)). This is this chapter's "compose signals, don't trust one number" principle applied twice — once per pool, each with the metric that actually reflects that pool's bottleneck. NVIDIA has aligned Dynamo with the community **llm-d** project for large-scale distributed inference ([developer.nvidia.com/blog/nvidia-dynamo-accelerates-llm-d-community-initiatives-for-advancing-large-scale-distributed-inference](https://developer.nvidia.com/blog/nvidia-dynamo-accelerates-llm-d-community-initiatives-for-advancing-large-scale-distributed-inference)) and documented the Kubernetes deployment path directly ([developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/); Azure's AKS walkthrough for multi-node Dynamo on GB200 NVL72, Oct 24 2025: [blog.aks.azure.com/2025/10/24/dynamo-on-aks](https://blog.aks.azure.com/2025/10/24/dynamo-on-aks)). If you adopt disaggregated serving, the mechanisms in this chapter still apply — you apply them twice, once per pool, with pool-appropriate metrics (KV-cache percentage for decode, queue depth for prefill).

- **AIBrix** — originally released by ByteDance engineers and now hosted under the `vllm-project` GitHub organization ([github.com/vllm-project/aibrix](https://github.com/vllm-project/aibrix); announced Feb 21 2025: [vllm.ai/blog/2025-02-21-aibrix-release](https://vllm.ai/blog/2025-02-21-aibrix-release)) — ships a purpose-built `PodAutoscaler` CRD with three interchangeable algorithms: `HPA` (native CPU-style), `KPA` (Knative-style, with a stable window and a shorter panic window for sudden spikes), and `APA` — **Advanced Pod Autoscaler** — which scales like HPA's ratio-based formula but adds explicit `up-fluctuation-tolerance` / `down-fluctuation-tolerance` parameters as a built-in buffer against oscillation, functionally the same job this chapter's hand-tuned stabilization windows do, just expressed as a percentage tolerance band instead of a time window:

```yaml
apiVersion: autoscaling.aibrix.ai/v1alpha1
kind: PodAutoscaler
metadata:
  name: vllm-llama3-8b-apa
  annotations:
    autoscaling.aibrix.ai/up-fluctuation-tolerance: "0.1"     # 10% headroom before scaling up
    autoscaling.aibrix.ai/down-fluctuation-tolerance: "0.2"   # 20% headroom before scaling down
spec:
  scalingStrategy: APA
  minReplicas: 1
  maxReplicas: 8
  metricsSources:
    - metricSourceType: pod
      port: "8000"
      targetMetric: gpu_cache_usage_perc      # same KV-cache signal this chapter recommends as a leading indicator
      targetValue: "0.5"
  scaleTargetRef:
    kind: Deployment
    name: model-deployment
```

  Notice the target metric is `gpu_cache_usage_perc` — the same KV-cache-pressure signal this chapter's "Right Scaling Signals" table and War Story 1 recommend adding as an *earlier-than-queue-depth* warning — and the asymmetric-tolerance idea is the asymmetric-stabilization principle expressed through a different knob. The convergence is the point: whether you hand-roll HPA/KEDA `behavior` blocks or adopt a purpose-built controller like AIBrix's APA or Dynamo's Planner, the underlying lessons — lead with a signal close to the real bottleneck, dampen oscillation asymmetrically, and treat prefill/decode (or CPU/GPU) as distinct scaling domains when the architecture actually splits them — hold across all of them.

### Node-level GPU autoscaling closes the gap HPA/KEDA can't

Everything in this chapter so far scales **pods**. A new pod still needs a **node** with a free GPU, and if the cluster autoscaler can't provision one fast enough (or at all, against quota), `maxReplicas` is a wish, not a guarantee — exactly the "GPU quota / capacity ceilings" pitfall flagged earlier. AWS's EKS best-practices guide for AI/ML compute ([docs.aws.amazon.com/eks/latest/best-practices/aiml-compute](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html)) makes the pairing explicit: **Karpenter** for just-in-time node-level provisioning, **KEDA** for pod-level scaling on model performance metrics, working together rather than either alone:

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-inference
spec:
  template:
    spec:
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["on-demand"]
        - key: karpenter.k8s.aws/instance-category
          operator: In
          values: ["g"]          # GPU instance families
  limits:
    nvidia.com/gpu: "10"          # hard ceiling on GPUs this NodePool will provision
  disruption:
    consolidationPolicy: WhenEmpty
    consolidateAfter: 60m        # don't tear down nodes mid-spike — GPU nodes are as sticky as GPU pods
```

  AWS's own guidance for this pairing is worth internalizing verbatim: *"For real-time workloads, where scaling time is important and workloads take longer than two minutes for the application to be ready to serve traffic, consider optimizing container start-up and ML model loading times"* — the node-level Karpenter layer buys you a GPU to schedule onto, but it does nothing about the cold-start phases (image pull, weight load, warm-up) this chapter covers in depth; the two problems are solved by different mechanisms and both need solving. AWS also recommends **On-Demand Capacity Reservations (ODCRs)** — reserved GPU capacity with no long-term commitment, usable by Karpenter via `capacityReservationSelectorTerms` — as the fix for "HPA wants 12 replicas but there are no H100s to schedule them on," turning a `Pending`-pod incident into a capacity-planning line item instead. The `disruption.consolidateAfter: 60m` setting is the node-level analogue of this chapter's scale-down stabilization window, for exactly the same reason: a GPU node is as expensive to re-provision as a GPU pod is to cold-start, so err toward stickiness at both layers.

**What this means for the rest of the chapter.** None of the above replaces HPA/KEDA/Knative — it sits on top of or beside them. A representative 2026 production stack for a serious LLM product looks like: KEDA (a Prometheus trigger for reactive queue-depth scaling, plus a Cron trigger for known daily patterns, optionally a predictive layer for irregular-but-forecastable demand) driving the HPA that actually moves replica counts, with weight streaming cutting the cold-start tax for whichever replicas still have to boot cold.


---

## Build It in Practice — Extended

The manifests above are correct but abstract. This section tunes them against an actual traffic *shape* and turns the "headroom sizing" one-liner from the Cost section into a full worked calculation — including the moment the math tells you your `maxReplicas` cap is wrong.

### Worked walkthrough — tuning stabilization windows against a traffic trace

Microsoft's public **Azure LLM Inference Trace 2023** ([github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md)) records per-request arrival timestamps, input, and output token counts from production conversational and code-generation LLM services on Nov 11 2023, and was used to characterize workload shape in the Splitwise paper ([ISCA 2024](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/12/Splitwise_ISCA24.pdf)). The dataset itself doesn't ship a pre-aggregated "requests per minute" column, so rather than reprint raw rows, the walkthrough below uses an **illustrative 20-minute trace with the qualitative shape practitioners consistently report from it and from similar production traces**: a steady diurnal baseline punctuated by short (1–3 minute) bursts several times the baseline rate, plus occasional brief lulls. Treat the specific numbers as a teaching device, not a reproduction of the dataset.

Assume each replica sustainably serves at a per-pod queue-depth target of 5 waiting requests (matching the HPA/KEDA manifests above), so a "raw recommendation" column below is the replica count HPA's ratio formula would compute from the observed queue depth each minute, **before** any stabilization is applied:

| Minute | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Raw recommendation | 5 | 5 | 5 | 5 | 6 | 5 | 5 | 6 | 9 | 12 | 12 | 8 | 6 | 5 | 5 | 5 | 2 | 5 | 5 | 5 |

Minutes 8–10 model a real burst (a client retry storm, or a batch job landing); minute 16 models a one-minute lull that isn't a real trend. Now apply three `behavior` configurations and read off the **actual** replica count each minute — remember HPA's stabilization rule: over the trailing window it picks the *minimum* recent recommendation for scale-up decisions and the *maximum* recent recommendation for scale-down decisions, so a single window length changes the two directions asymmetrically only if you configure the two windows independently (as this chapter recommends).

**Config 1 — naive, symmetric (`scaleUp: 0s`, `scaleDown: 0s`)**: actual replicas track the raw recommendation exactly. The one-minute lull at minute 16 causes a real scale-down to 2 replicas, which then reverses back up to 5 the very next minute — a wasted cold start for a lull that was never a trend. Minor fluctuations (5→6→5 at minutes 3–5) also churn replicas even though nothing is actually wrong.

**Config 2 — recommended asymmetric (`scaleUp: 0s`, `scaleDown: 300s`)**: scale-up still reacts immediately (replicas hit 9 at minute 8 and 12 at minute 9 — no hesitation on real demand). Scale-down looks back 5 minutes and takes the **max** of that window. At minute 16, the trailing window (minutes 12–16) is `[6, 5, 5, 5, 2]`, so actual replicas hold at 6 — the lull is filtered out. By minute 19 the window `[5, 2, 5, 5, 5]` maxes at 5, and replicas settle exactly at the true steady state. This is the config used in the manifests above, and the trace shows *why*: it reacts to the burst immediately and ignores the false-alarm dip.

**Config 3 — overly conservative queue-based (`scaleUp: 120s`, `scaleDown: 600s`)**, mirroring the longer windows some queue-based guides recommend: with a 10-minute scale-down window, at minute 16 the trailing window (minutes 7–16) is `[6, 9, 12, 12, 8, 6, 5, 5, 5, 2]`, max = 12 — replicas stay **pinned at 12** for the full window even though real demand has been back at baseline (5) since minute 13. That's three extra minutes of paying for peak capacity nobody needs, on top of the five minutes Config 2 would have already smoothed.

The lesson generalizes into a sizing rule: **pick the scale-down window long enough to absorb the noise you actually observe (a single-minute lull, in this trace), but short enough that you're not bankrolling peak capacity long after a burst has visibly ended.** A useful starting point is a scale-down window of roughly \( 3\text{–}5 \times T_{cold} \) — long enough that you won't immediately need to re-pay a cold start if the burst repeats, short enough that idle GPU-minutes don't pile up. With \( T_{cold} \approx 180\text{s} \), that lands squarely on the 300s (5-minute) window this chapter recommends by default — the trace above is the empirical argument for that number, not just a rule of thumb.

### Worked warm-pool / min-replica sizing calculation

The Cost section's headroom formula is:

\[
\text{buffer replicas} = \left\lceil \frac{\Delta(\text{req/s over } T_{cold})}{\text{capacity per replica}} \right\rceil
\]

Run it against the *actual* burst from the trace above instead of a hypothetical rate. From minute 7 to minute 9 the raw recommendation climbed from 6 replicas to 12 replicas — a demand growth of 6 replicas in 2 minutes, or 3 replicas/minute. Over a cold-start time \( T_{cold} = 180\text{s} = 3\text{ min} \), if you were starting from a cold floor rather than already having 6 replicas warm, you'd need:

\[
\text{buffer replicas} = 3 \text{ replicas/min} \times 3 \text{ min} = 9
\]

Added to a steady-state floor of 5, that's a **warm floor of 14 replicas** to fully absorb this burst without any request ever waiting on a cold start — which exceeds the `maxReplicas: 12` ceiling used in every manifest earlier in this chapter. That's not a contrived result; it's the headroom math doing its job. It surfaces, *before* an incident, that the quota this chapter's example manifests assumed is actually undersized for the worst burst rate this traffic shape produces. The three honest responses, in order of preference:

1. **Raise the GPU quota** backing `maxReplicas`/`maxScale`, if capacity is available — the cheapest fix when it's possible.
2. **Accept bounded queueing** during the worst 1–2 minutes of a burst like this one, if your SLO has slack — quantify how much queue depth 12 replicas can absorb at the target of 5 per pod (60 requests) and compare against the observed peak.
3. **Shed or degrade** the excess — request queueing with a hard timeout, or routing overflow to a smaller/cheaper model — rather than silently violating the SLO for every request past replica 12.

### Little's Law: the theory behind the target values

Every target value picked in this chapter — 5 waiting requests per pod, 8 concurrent requests per replica — is an application of queueing theory's most useful identity, **Little's Law**:

\[
L = \lambda \times W
\]

where \(L\) is the average number of requests in the system (waiting plus in flight), \(\lambda\) is the arrival rate the system is sustaining (throughput), and \(W\) is the average time each request spends in the system — for a streaming LLM response, roughly TTFT plus generation time.

Rearranged, it tells you directly what a concurrency or queue-depth *target* should be once you've fixed a latency SLO. If a single replica can sustain throughput \(\lambda_{replica}\) at your target latency \(W_{SLO}\), the maximum in-flight population that replica can carry without breaching the SLO is

\[
L_{replica} = \lambda_{replica} \times W_{SLO}
\]

That is precisely what the Knative example's `containerConcurrency`/`target` pair, and the HPA/KEDA `averageValue` targets, are estimating — a defensible target isn't a round number, it falls out of \(\lambda_{replica} \times W_{SLO}\) measured from a load test. Concretely: a replica sustaining 8 requests/second of throughput at roughly a 1-second average time-in-system is, by Little's Law, carrying an average population of about 8 requests at any instant — which is exactly why `target: 8` shows up as this chapter's Knative concurrency target earlier. The number was never arbitrary; it falls out of the throughput and latency actually measured for the model and hardware in question.

### Deriving your threshold from a load test, not a guess

Interviewers routinely ask "how did you pick that number?" — here is the derivation, worked. Sweep concurrency per replica in a load test (see the [Load Testing chapter](../04_load_testing/README.md) for harness details) and record p95 TTFT at each level:

| Concurrency per replica | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|
| p95 TTFT (s) | 0.4 | 0.5 | 0.6 | 0.8 | 1.1 | 1.6 | 2.4 | 3.8 |

Against an SLO of p95 TTFT \( \le 1.5 \) s, concurrency 12 already breaches it (1.6 s) while concurrency 10 is still safe (1.1 s) — the curve's knee sits between the two, as it typically does once KV-cache pressure and batching contention start to dominate. Pick the target *below* the knee, not at it, to leave slack for the seconds between "queue starts growing" and "new replica is ready": a target of 8–10 gives roughly 20–35% headroom under the last safe measured point. This is the same derivation this chapter used to justify `target: 8` in the Knative manifest, and per Little's Law above, an 8-request target at roughly 0.8s average time-in-system corresponds to a sustained per-replica throughput of \(\lambda_{replica} = L / W \approx 8 / 0.8 = 10\) requests/second — a number you can now cross-check independently against the load test's own throughput measurement at that concurrency level, and treat any large mismatch as a sign the load test or the target needs a second look.

### Combining Cron and Prometheus triggers for a scheduled warm-up

If the diurnal baseline in the trace above repeats on a known daily cycle (e.g., an 8am ramp), pre-warm ahead of it rather than waiting for the reactive trigger to catch up mid-ramp. A single `ScaledObject` can carry both triggers; KEDA scales to satisfy whichever is higher:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler-scheduled
  namespace: inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-llama3-8b
  minReplicaCount: 1
  maxReplicaCount: 12
  pollingInterval: 15
  cooldownPeriod: 300
  triggers:
    # Reactive: same queue-depth trigger as before
    - type: prometheus
      metricType: AverageValue
      metadata:
        serverAddress: http://kube-prometheus-stack-prometheus.monitoring.svc:9090
        query: sum(vllm:num_requests_waiting) or vector(0)
        threshold: "5"
        activationThreshold: "1"
    # Scheduled: pre-warm to the known business-hours floor before the 8am ramp arrives
    - type: cron
      metadata:
        timezone: America/New_York
        start: "50 7 * * 1-5"    # 07:50 weekdays — T_cold (3 min) of margin before 08:00
        end: "0 20 * * 1-5"      # back to reactive-only floor at 20:00
        desiredReplicas: "5"     # matches the observed steady-state baseline
```

**Tuning notes.**

- The Cron trigger's `start` time is set **`T_cold` ahead of the real ramp**, not at the ramp itself — pre-warming only helps if the replicas are actually serving by the time demand arrives.
- Outside the `start`/`end` window, the Cron trigger contributes nothing and the reactive Prometheus trigger governs alone — including scaling further above the scheduled floor if the day's actual traffic exceeds the historical baseline.
- This is deterministic scheduling, not forecasting — it costs nothing to reason about and is the right first step before reaching for a forecasting layer like Kedify's `MetricPredictor` (see the Landscape section above), which is better suited to patterns that shift week to week rather than a fixed business-hours shape.

---

### Validating the config before it meets production traffic

None of the manifests above should meet real users untested. A short, concrete pre-production checklist:

1. **Replay a trace-shaped load test**, not just a flat ramp — use the traffic-shape methodology from the walkthrough above (steady baseline, a short sharp burst, a brief lull) so you exercise both the scale-up and scale-down paths, not only "more traffic forever." Tools from the [Load Testing chapter](../04_load_testing/README.md) apply directly.
2. **Watch replica count over time during the replay**, not just latency — a config that keeps latency fine by scaling up 20 times in 10 minutes has a flapping problem the latency graph alone won't show you.
3. **Kill the metrics pipeline mid-test** (stop `prometheus-adapter`, or block the KEDA-to-Prometheus network path) and confirm the autoscaler fails to a safe state — holds its last known replica count, or a configured safe default — rather than scaling to zero or erroring out. This is the pitfall table's "metric pipeline is a hidden dependency" row, tested rather than assumed.
4. **Force a scale-up from the actual floor** (not from an already-warm fleet) and measure real cold-start time end to end, including scheduling and image pull on a genuinely cold node — synthetic or vendor-published cold-start numbers are a starting estimate, not a substitute for your own measurement on your own image and node type.
5. **Trigger a simultaneous multi-replica scale-up** (the thundering-herd scenario from War Story 2) against your actual weight storage backend, and confirm the staggered `scaleUp` step policy is actually staggering it — a step cap configured but never load-tested is a step cap you're merely hoping works.
6. **Confirm graceful termination** by starting a long generation and triggering a scale-down mid-stream; verify `terminationGracePeriodSeconds` is long enough that the request completes rather than getting cut off.
7. **Alert on `Pending` GPU pods** during the test, deliberately capping node pool size below what the test will demand, to confirm the node-level ceiling from War Story 3 is visible to on-call rather than silent.

A config that has been through all seven steps once, deliberately, before its first real traffic spike is a materially different bet than one that has only been read and reasoned about.

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

The general shape: **provisioned floor covers the baseline, headroom buffer covers what arrives during a cold start, and autoscaling handles the sustained ramp.** Tune the floor and buffer to the *cost of a missed SLO*, not to a generic utilization target. See the fully worked version of this calculation, against an actual traffic shape, in "Build It in Practice — Extended" above.

**Measuring it, not just modeling it.** The worked calculation above assumes you already know your per-replica cost and utilization; in production, that visibility has to come from somewhere. Standard Kubernetes cost tooling reports at the node level, which is close to useless for a GPU fleet where the interesting question is *which deployment* is paying for idle capacity. **OpenCost** ([opencost.io](https://www.opencost.io)) has extended pod-level cost allocation to GPUs specifically by collecting `dcgm-exporter` metrics (`nvidia_gpu_utilization`, `nvidia_gpu_memory_used`) and correlating them with pod resource requests and cloud instance pricing, attributing cost down to the pod rather than the node — the tool coverage matches a problem this chapter's cost math otherwise leaves as an exercise for the reader: real-time inference workloads commonly run at only 20–40% GPU utilization due to sparse, bursty request patterns, while the bill reflects 100% of provisioned capacity regardless. **Kubecost**, OpenCost's commercially-supported counterpart, layers dashboards and alerting on the same underlying allocation model. Neither tool changes the autoscaling decision by itself, but both turn "we think our warm floor is costing too much" into a number you can actually put in front of whoever owns the budget — and into a feedback signal for re-running the headroom and warm-pool calculations above against real, current utilization rather than a one-time estimate.

---

## Failure Modes & Pitfalls

- **Flapping (thrashing).** Symmetric or twitchy thresholds scale up and down every few minutes, and each cycle pays a cold start. **Fix:** long `scaleDown.stabilizationWindowSeconds` (300s+), conservative scale-down policies, and a generous `cooldownPeriod`. For GPUs, err toward *stickiness*.
- **Scaling on a lagging metric.** If p95 latency or GPU utilization is your *primary* trigger, you scale *after* users are already hurting, and because cold starts are slow you stay behind the curve for minutes. **Fix:** lead with a *leading* signal (queue depth / concurrency); keep latency as a guardrail only.
- **Thundering herd on cold model load.** A big spike triggers many replicas at once; they *simultaneously* hammer the same weights bucket / registry, saturating network and disk, so **all** of them cold-start slower. **Fix:** cap `scaleUp` step size (`Pods: value` / `periodSeconds`), stagger with fast-load streaming, pre-warm images, and cache weights on nodes.
- **GPU quota / capacity ceilings.** `maxReplicas` is a wish; cloud GPU quota and actual availability are the reality. HPA will happily request 12 pods that sit `Pending` forever because there are no H100s to schedule them on. **Fix:** set `maxReplicas` to your *real* quota, alert on `Pending` GPU pods, and combine with cluster-autoscaler node pools sized to quota.
- **Metric pipeline is a hidden dependency.** If Prometheus, the adapter, or `dcgm-exporter` hiccups, HPA/KEDA see stale or missing metrics and may freeze or over-react. **Fix:** `or vector(0)` guards, `ignoreNullValues`, alert on the metrics pipeline itself, and a sane default replica count.
- **Per-pod vs total metric confusion.** Using `Value` where you meant `AverageValue` (or a raw `sum()` without dividing by replicas) makes the target scale wrong as the fleet grows — you either never scale or scale to the moon. **Fix:** be explicit about per-pod semantics and test with a load generator.
- **Scale-down mid-generation.** Terminating a pod that's mid-stream kills in-flight long generations. **Fix:** graceful termination with a drain period longer than your max generation time, and `terminationGracePeriodSeconds` sized accordingly.
- **MIG/GPU-sharing capacity confusion.** Sizing `maxReplicas` or warm-pool buffers against "physical GPUs" when the cluster is actually running MIG-partitioned or time-sliced GPUs (or vice versa) silently over- or under-provisions, since a "replica" no longer maps 1:1 to a full GPU. **Fix:** confirm what unit your quota numbers and `nvidia.com/gpu` requests actually refer to before reusing this chapter's cost/headroom formulas verbatim.
- **Regional/multi-cluster quota blindness.** For a globally-deployed product, GPU quota is granted per region/account, not globally — a `maxReplicas` ceiling sized against total fleet capacity can still leave one region's cluster starved of `Pending` pods while another region has idle headroom. **Fix:** size `maxReplicas`/node-pool ceilings and alerting per region, and treat cross-region traffic shifting (if your architecture supports it) as a capacity release valve distinct from autoscaling within a single region.

---

## Wiring Alerts for Autoscaler Health

Every failure mode above has a corresponding alert worth having before it fires in anger, not after. These pair with the Prometheus/Grafana stack from the [Monitoring chapter](../08_monitoring/README.md):

```promql
# Custom metrics API is not returning data — HPA/KEDA are flying blind
absent(vllm_num_requests_waiting{namespace="inference"})

# Replica churn rate — a proxy for flapping (tune the count/window to your traffic)
sum(changes(kube_deployment_status_replicas{deployment="vllm-llama3-8b"}[15m])) > 6

# GPU pods stuck unschedulable — maxReplicas has become aspirational (War Story 3)
sum(kube_pod_status_phase{phase="Pending", namespace="inference"}) > 0

# KV-cache pressure rising faster than queue depth — the War Story 1 blind spot
rate(vllm:num_preemptions_total[5m]) > 0

# Time-to-ready for a new replica exceeds the cold-start budget assumed by warm-pool sizing
histogram_quantile(0.95, rate(kube_pod_start_time_seconds_bucket[10m])) > 180
```

Each rule maps directly to a pitfall or war story earlier in this chapter: a missing metric, replica churn, unschedulable pods, silent preemption, and a cold start running longer than the sizing math assumed. Alerting on the autoscaler's own health, not just the application's, is what turns "we found out during the incident" into "we found out before it mattered."

## Production Case Studies & War Stories

### War story 1 — the queue looked fine while p99 spiked 8x

**Symptom.** A team scaling vLLM on `vllm:num_requests_waiting` as the sole trigger saw p99 latency spike roughly **8x** during a traffic-mix change (a burst of long-context requests), even though the queue-depth metric driving HPA never crossed its threshold — the autoscaler saw no problem and did nothing.

**Root cause.** vLLM pre-allocates most of its GPU memory for the KV cache at startup; when that cache fills — which long-context requests do far faster than short ones — the engine doesn't reject new work, it **silently preempts already-in-flight sequences** to make room. Preempted requests get re-queued and effectively restart, which is exactly the kind of latency cliff that a short, healthy-looking `num_requests_waiting` reading can completely miss: the queue is short precisely because the engine is quietly discarding progress on other requests rather than letting them wait. This pattern — and the specific "P99 latency can spike 8x" framing — is documented in a 2026 write-up on why vLLM autoscaling on Kubernetes breaks ([dev.to/soniarotglam/why-vllm-autoscaling-on-kubernetes-breaks-and-what-to-use-instead](https://dev.to/soniarotglam/why-vllm-autoscaling-on-kubernetes-breaks-and-what-to-use-instead-1231)).

**Fix.** Add `vllm:gpu_cache_usage_perc` (KV-cache utilization) and vLLM's preemption-count metric as an **earlier warning signal** alongside queue depth — KV-cache pressure and preemptions rise before queue depth does when the bottleneck is memory rather than raw arrival rate. The same source also flags a closely related tuning trap: running `--gpu-memory-utilization 0.95` leaves no slack and OOMs under concurrent load, while `0.85` provides the headroom that keeps preemption a rare event instead of the default behavior under load.

**Lesson.** A single "safe-looking" metric is a false sense of security if it isn't the metric closest to the actual failure mode. Queue depth is an excellent *arrival-rate* signal; it is not a *memory-pressure* signal. Compose signals that cover different failure modes rather than trusting one number to mean "everything is fine."

### War story 2 — a spike triggered a thundering herd of cold starts

**Symptom.** A traffic spike caused several replicas to scale up simultaneously. Each one took the better part of five minutes to become ready, so by the time the fleet caught up, the spike had already caused timeouts and dropped requests — the autoscaler technically "worked," but too slowly to matter.

**Root cause, quantified.** A concrete breakdown for a Llama 3.1 8B model on L40S GPUs, published by Tensorfuse ([tensorfuse.io/docs/blogs/reducing_gpu_cold_start](https://tensorfuse.io/docs/blogs/reducing_gpu_cold_start)), shows where the time actually goes in a naive cold start:

| Phase | Time (naive) |
|---|---|
| Model download | 61 s |
| Weight loading | 33 s |
| CUDA graph compilation | 42 s |
| CUDA graph capture | 54 s |
| **Total** | **294 s (4 min 54 s)** |

When several replicas cold-start at once, the *download* phase is the one that gets worse under contention — they're all pulling the same weights from the same registry or storage bucket at the same time, competing for the same network and disk bandwidth, so the herd's aggregate cold start is worse than any single one in isolation.

**Fix, with results.** The same source reports cutting the total to **82 seconds — a 70% reduction** — through a combination of: caching the model and compiled artifacts on a persistent volume so repeat cold starts skip the download entirely; restricting CUDA graph capture to the batch sizes actually used in production (e.g. `1,2,4,8,16,24,32,64`), which alone cut capture time from 54s to 7s; and leaning on `torch.compile`'s cross-instance compilation cache. Layer that with two scaling-side mitigations already in this chapter: **cap the scale-up step size** (`policies: [{type: Pods, value: 4, periodSeconds: 60}]`, as in the HPA manifest above) so the herd is staggered rather than simultaneous, and use **weight streaming** (Run:ai Model Streamer, see the Landscape section) to shrink the download/load phase itself rather than only caching around it.

**Lesson.** Thundering-herd cold starts are a *shared-resource contention* problem as much as a per-replica speed problem. Fixing per-replica cold-start time (caching, graph capture tuning, weight streaming) and fixing the stampede shape (staggered scale-up steps) are complementary — the trace-driven walkthrough above shows a real burst of 6 replicas arriving inside 2 minutes; without a capped scale-up policy, all 6 would hit the weights store at once.


---

### War story 3 — the pods scaled, the nodes didn't

**Symptom.** During a launch-day spike, HPA correctly computed a desired replica count of 10 (up from a floor of 3), and Kubernetes accepted all 10 pods — but 6 of them sat in `Pending` for the better part of 20 minutes. Users hitting the overflow saw timeouts, while `kubectl get hpa` showed the autoscaler doing exactly what it was configured to do.

**Root cause.** `maxReplicas` describes what the autoscaler is *allowed* to ask for; it says nothing about whether a **node** with a free GPU actually exists to run the new pod on. The cluster's GPU node pool had a static size, and provisioning a new GPU node from the cloud provider — capacity check, instance launch, driver/AMI boot, node join — took longer than the spike itself lasted. This is the scenario AWS's own EKS best-practices guidance for AI/ML compute is written to prevent: node-level provisioning and pod-level scaling are two different control loops with two different response times, and treating only the pod-level one (HPA/KEDA) as "the autoscaling system" leaves the slower loop as an invisible ceiling ([docs.aws.amazon.com/eks/latest/best-practices/aiml-compute](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html)).

**Fix.** Pair pod-level scaling with a node-level autoscaler purpose-built for just-in-time GPU provisioning (Karpenter on AWS, or the equivalent cluster-autoscaler node pool on other clouds), sized with real headroom rather than a static count that only covers steady state. Where the SLO can't tolerate even a Karpenter-speed node bring-up, use an **On-Demand Capacity Reservation** so the GPUs are already allocated to the account and Karpenter only has to attach a node to reservation, not queue for scarce on-demand capacity. Alert on `Pending` GPU pods as a first-class signal — it is the earliest, clearest evidence that `maxReplicas` has become aspirational rather than real.

**Lesson.** An autoscaling design isn't complete at the `Deployment`/`ScaledObject` layer. Ask, explicitly, "when HPA/KEDA asks for one more replica than the cluster currently has room for, what happens next, and how long does it take?" — and treat that answer as part of the SLO, not an infrastructure detail to hand-wave past. This is precisely the gap the Node-level GPU autoscaling subsection in the Landscape section above is describing.

### Quick reference — defaults to start from

A condensed version of every number this chapter derived, for when you're sketching a first config under time pressure:

| Knob | Starting default | Why |
|---|---|---|
| Primary signal | Queue depth (`num_requests_waiting`) or concurrency | Leading indicator, not lagging |
| Guardrail signal | p95/p99 TTFT or latency | Catches what the primary signal misses (see War Story 1) |
| `scaleUp.stabilizationWindowSeconds` | 0 | Never hesitate on real demand — cold starts already make you slow |
| `scaleDown.stabilizationWindowSeconds` | 300 (5 min) | ≈ \(3\text{–}5 \times T_{cold}\); absorbs noise without hoarding capacity (see trace walkthrough) |
| `scaleUp` step cap | +4 pods / 60s (or similar) | Prevents a thundering herd on shared weight storage |
| `scaleDown` step cap | 1 pod / 120s | GPUs are expensive to churn; err toward stickiness |
| KEDA `activationThreshold` | 1 | One real request is enough to justify the first replica |
| `minReplicas` / warm floor | steady-state baseline, sized from real traffic | Never cold-start the interactive hot path |
| Warm-pool buffer | \(\lceil \Delta(\text{req/s over } T_{cold}) / \text{capacity per replica} \rceil\) | Covers demand growth during the unavoidable cold-start window |
| `maxReplicas` / node pool ceiling | actual GPU quota, alerted on `Pending` | A cap you can't hit is worse than no cap — it fails silently |
| Concurrency/queue-depth target | \(\lambda_{replica} \times W_{SLO}\) from a load-test sweep, with ~20–35% margin below the SLO knee | Derived, not guessed (Little's Law) |
| KEDA workload shape | `ScaledObject` for live-request pools, `ScaledJob` for run-to-completion batch units | Matches the scaling primitive to the actual unit of work |
| Cost visibility | Pod-level GPU cost allocation (OpenCost/Kubecost) reviewed against actual utilization | Turns "the warm floor feels expensive" into a number, and a feedback loop for re-sizing it |

This table is a starting point for a design discussion, not a substitute for measuring your own workload's cold-start time, throughput-latency curve, and traffic shape — every number above was derived from a specific worked example earlier in this chapter, and yours will differ.

Use it as a checklist when reviewing someone else's config, too: for each row, ask "is this value here because it was measured, or because it was left at a default?" — the answer to that question is often the fastest way to find the next incident before it happens.

## Interview Mastery

### Core Q&A (1–8): the mechanics

1. **"Why not CPU-based HPA for an LLM?"** — Host CPU is decoupled from GPU saturation; a full GPU can look idle to HPA. Name the real signals: queue depth, concurrency, KV-cache, GPU util, TTFT. (See the 60-second answer callout earlier in this chapter.)
2. **"What's your primary scaling signal and why?"** — A *leading* demand signal (`vllm:num_requests_waiting` / concurrency), with a *lagging* latency SLO as a guardrail, and the max-of-metrics behavior that composes them.
3. **"How do you handle cold starts?"** — Quantify the phases (schedule, pull, load, warm), then name concrete mitigations: warm floor, headroom buffer, fast weight streaming, snapshot/checkpoint-restore, image pre-pull.
4. **"Scale to zero — yes or no?"** — "It depends on the SLO." Fine for batch/internal; dangerous for interactive unless snapshotting or streaming makes cold starts sub-second-to-low-seconds (see the serverless-GPU cold-start table in the Landscape section). Explain KEDA's `activationThreshold`.
5. **"How do you stop it flapping?"** — Asymmetric behavior: aggressive scale-up (0s window), conservative scale-down (300s+ window, one pod at a time), cooldown. Justify the window length against the actual cold-start time, not a default.
6. **"HPA vs KEDA vs Knative — when each?"** — HPA+adapter for simple custom-metric scaling; KEDA for event/queue-driven and scale-to-zero on any metric (plus Cron for scheduled floors); Knative/KPA for concurrency-driven serverless with request buffering.
7. **"What breaks under a real traffic spike?"** — Thundering herd on weight load, GPU quota ceilings leaving pods `Pending`, and lagging-metric lock-step. Have a mitigation for each — and be ready to cite the concrete cold-start breakdown (download/load/compile/capture) from the war stories above.
8. **"How do you pick the target value / threshold?"** — Derive it from load tests (see the [Load Testing chapter](../04_load_testing/README.md)): find the per-replica concurrency/queue depth at which TTFT just meets SLO, then set the target below it.

### Deeper Q&A (9–18): mechanism internals and judgment calls

9. **"Explain HPA's `AverageValue` vs `Value` metric types — why does the distinction matter for a growing fleet?"** — `AverageValue` divides the metric by current replica count before comparing to target, so the target stays meaningful as you scale (5 waiting requests *per pod*, regardless of fleet size). `Value` compares the raw number directly — use it only when the query already returns a per-pod figure, otherwise a `sum()` across a growing fleet will make HPA think demand is exploding (or never scale at all) purely because the denominator changed.
10. **"Walk me through HPA's desired-replicas formula."** — \( \text{desiredReplicas} = \lceil \text{currentReplicas} \times \frac{\text{currentMetricValue}}{\text{desiredMetricValue}} \rceil \), computed per metric, then HPA takes the **max** across all configured metrics — whichever metric wants the most replicas wins that cycle.
11. **"What is KEDA's `activationThreshold` and how does it differ from `threshold`?"** — `activationThreshold` gates the 0→1 transition (waking a scaled-to-zero deployment); `threshold` governs the 1→N scaling once at least one replica is running. Setting `activationThreshold` low (e.g. "1") means a single stray request can wake a cold GPU — intentional if you want zero missed requests, costly if bots or health-checks are the "stray" traffic.
12. **"How would you scale to zero safely for an interactive LLM endpoint?"** — Generally: don't, unless a snapshot/checkpoint-restore path gets you to low-single-digit-second restores (per the serverless-GPU comparison in the Landscape section). Otherwise keep `minReplicaCount ≥ 1` on the interactive path and reserve true scale-to-zero for batch/internal/dev traffic.
13. **"Explain Knative's panic mode and `target-burst-capacity`."** — Panic mode is a short window where KPA evaluates demand over a much shorter horizon than its normal window, so it reacts to sharp spikes faster than its steady-state smoothing would allow. `target-burst-capacity` sets how much traffic the activator is willing to buffer (holding requests, not dropping them) while new replicas spin up — higher values trade a little steady-state latency for burst safety.
14. **"How do you avoid a thundering herd when many replicas cold-start at once?"** — Cap the scale-up step size (`policies: [{type: Pods, value: N, periodSeconds: 60}]`) so replicas come up in waves rather than all at once, pre-pull/cache images and weights on nodes, and adopt weight streaming so each replica's load phase is short even under shared-storage contention. Reference the concrete before/after numbers (294s → 82s) from the cold-start war story.
15. **"How would you size a warm pool / min-replica floor mathematically, not by gut feel?"** — Take the observed worst-case demand growth rate (replicas or req/s per minute) from real traffic data, multiply by your cold-start time \( T_{cold} \) to get the buffer needed to fully absorb a burst with zero cold starts, and add it to your steady-state floor. Then sanity-check the result against your actual `maxReplicas`/GPU quota — the worked calculation above shows this can reveal the quota itself is undersized.
16. **"What's wrong with scaling on GPU utilization alone?"** — It's lagging and coarse: 100% utilization can mean "efficiently batched and totally healthy" or "drowning," and it doesn't distinguish the two. It's a good hardware-truth cross-check, a poor sole trigger.
17. **"If a `ScaledObject` has both a Cron trigger and a Prometheus trigger, which one wins?"** — Neither "wins" outright — KEDA scales to satisfy whichever trigger currently demands more replicas, same principle as HPA's multi-metric max. The Cron trigger guarantees a floor during its window; the Prometheus trigger can still scale above that floor if real demand exceeds the scheduled baseline.
18. **"What would you monitor to know your autoscaling configuration itself is broken — not just the app?"** — Alert on: HPA/KEDA unable to fetch the custom/external metric (stale or missing readings), GPU pods stuck `Pending` against `maxReplicas`, replica-count churn rate (a proxy for flapping), and time-to-ready per new replica (a proxy for whether your cold-start mitigations are actually working in production, not just in a benchmark).
19. **"How does continuous batching change what a 'per-replica capacity' number even means?"** — It means capacity isn't a fixed request count; it's shaped by the mix of prompt/generation lengths currently in the batch and, if multiple models or LoRA adapters share a replica, by which are currently loaded. A threshold derived from one workload mix can silently overstate capacity under a different mix — re-validate thresholds against your actual production traffic composition, not a single synthetic benchmark.
20. **"Give me Little's Law and tell me why it matters here."** — \( L = \lambda \times W \): average in-system population equals arrival rate times average time in system. It matters because it turns "what concurrency/queue-depth target should I set?" from a guess into a calculation — measure your per-replica throughput and target latency from a load test, multiply them, and that product is a defensible target, not a round number.

### System design prompt

> *"Design autoscaling for a bursty, cost-sensitive consumer LLM chat product: usage has a strong daily pattern, occasional viral spikes, and the business has a hard cost ceiling."*

A strong answer sketches a **tiered** system rather than a single mechanism:

```
                 ┌────────────────────────────────────────────┐
                 │            KEDA ScaledObject                │
                 │  ┌───────────────┐   ┌──────────────────┐   │
 known daily ───▶│  │ Cron trigger  │   │ Prometheus       │◀──│─── queue depth,
 pattern         │  │ (biz-hours    │   │ trigger          │   │    KV-cache %,
                 │  │  warm floor)  │   │ (reactive scale) │   │    p95 latency
                 │  └───────┬───────┘   └────────┬─────────┘   │    guardrail
                 │          └──────────┬──────────┘            │
                 │                 max(...)                    │
                 └───────────────────┬──────────────────────────┘
                                      ▼
                          [ HPA: replicas 5↔12, asym. behavior ]
                                      │
                     ┌────────────────┼────────────────────┐
                     ▼                ▼                    ▼
              warm floor (5)   staggered scale-up    overflow: request
              never cold        (+4 pods/min cap,     queue + timeout,
              starts             weight streaming)     or route to a
                                                        smaller model
```

Talking points an interviewer wants to hear, in rough priority order:

- **Warm floor sized off real data**, not a round number — derived from the steady-state baseline plus a headroom buffer covering \(T_{cold}\) of demand growth (the worked calculation above).
- **Cron-scheduled pre-warming** ahead of the known daily ramp, layered under a **reactive Prometheus trigger** for anything the schedule doesn't predict — including the viral spike case.
- **Composite scaling signals**: queue depth as the leading trigger, KV-cache usage as an earlier-than-queue warning for memory pressure, p95 latency as the final guardrail — not any single metric alone (see war story 1).
- **A capped, staggered scale-up** so a viral spike doesn't create a thundering herd on the weight store (see war story 2), paired with weight streaming to shrink whatever cold start still has to happen.
- **A cost ceiling enforced structurally**, not hoped for: `maxReplicas` tied to actual GPU quota (alerted on `Pending` pods), plus an explicit overflow path — bounded request queueing with a timeout, or degrading to a smaller/cheaper model — so the *system* fails predictably instead of silently blowing the latency SLO or the budget.
- **Asymmetric stabilization**: fast scale-up, slow scale-down, tuned against the cold-start time as shown in the trace walkthrough — not left at framework defaults.
- Explicitly naming the **tradeoff being made** — e.g. "we accept N minutes of degraded latency once or twice a year during an unprecedented spike, in exchange for not paying for peak capacity 24/7" — because a design with no acknowledged tradeoff is a design that hasn't been pressure-tested.

### A second system design prompt — the contrasting case

> *"Now design autoscaling for an internal batch-summarization pipeline: latency-tolerant (minutes are fine), but cost is the dominant concern, and traffic is extremely spiky (idle most of the day, large bursts overnight)."*

This is deliberately the mirror image of the consumer-chat prompt above, and a strong candidate notices that **the right answer changes almost every knob**, not just the numbers:

- **`minReplicas`/`minScale: 0` is now the right default**, not a risk to hedge against — there is no interactive user waiting on the first token, so the multi-minute cold start is an acceptable cost of being idle the rest of the time.
- **KEDA over HPA**, specifically for the 0→1 transition HPA cannot do, with `activationThreshold` tuned to avoid waking the fleet for a single stray job.
- **Queue-length-based triggers on the actual job queue** (SQS/Kafka/Redis depth via KEDA's native scalers) rather than an inference-engine metric — the unit of work is a batch job, not a live HTTP request, so the natural signal is upstream of the model server entirely.
- **No latency guardrail in the tight sense used for the chat product** — replace it with a *completion-time* SLO (e.g. "the overnight batch finishes by 6am"), which changes the guardrail from "p95 TTFT" to "queue drain rate given current replica count," a genuinely different metric.
- **Aggressive scale-up is still correct**, but for a different reason: cost-sensitivity argues for scaling up fast and back down fast, since every minute of an idle replica is pure waste with no offsetting latency benefit the way a warm floor provides for interactive traffic.
- **Spot/preemptible capacity becomes attractive** here in a way it wasn't for the interactive product — a batch job that gets preempted can simply be retried, so the cost savings (up to ~90% per AWS's own guidance cited in the Landscape section) are close to free.

The point of asking this as a follow-up is to check whether a candidate memorized "warm floor + asymmetric stabilization + composite signals" as a single template, or actually understands *why* each piece was chosen for the first scenario — and can correctly invert the ones that no longer apply.

### Common mistakes candidates make in this conversation

- Reciting "scale on GPU utilization" as the fix for CPU-based HPA being wrong — it's a better cross-check, not a better *primary* signal, for the same lagging-metric reason latency is a guardrail rather than a trigger.
- Proposing scale-to-zero for an interactive product without qualifying it against cold-start time, or without mentioning the fast-load/snapshot mitigations that make it viable.
- Treating `stabilizationWindowSeconds` as a single global knob rather than naming the deliberate asymmetry (fast up, slow down) and justifying the specific window against a cold-start time.
- Forgetting that `maxReplicas` needs a node-level counterpart — a strong answer proactively raises the Karpenter/cluster-autoscaler layer without being prompted.
- Picking round-number thresholds ("let's just use 70%") instead of deriving them from a load test's throughput/latency curve, per Little's Law.
- Presenting HPA, KEDA, and Knative as competitors rather than describing the common production pattern of layering them (KEDA driving the HPA it manages, Cron plus Prometheus triggers in one `ScaledObject`).
- Answering "how do you scale to zero" with a flat yes/no instead of naming the SLO-dependent tradeoff and the specific mitigations (fast weight loading, snapshotting) that make an aggressive answer defensible.
- Skipping straight to Kubernetes-native mechanisms without first naming *which signal* is correct for LLMs — a candidate who jumps to YAML before establishing why CPU is wrong has skipped the part of the answer that actually matters.



### Red flags vs. green flags

| Red flag | Green flag |
|---|---|
| Scales on CPU utilization | Scales on queue depth / concurrency, with a latency guardrail |
| Symmetric scale-up/scale-down windows | Aggressive scale-up (near-0s), conservative scale-down (300s+), justified against \(T_{cold}\) |
| A single scaling signal | Composite signals covering different failure modes (queue depth *and* KV-cache/preemption *and* latency) |
| `maxReplicas` picked arbitrarily ("seemed like enough") | `maxReplicas` tied to real GPU quota, with alerting on `Pending` GPU pods |
| Cold-start time unknown or unmeasured | Cold-start phases measured and broken down (schedule/pull/load/warm), with a named mitigation for the dominant phase |
| Scale-to-zero on an interactive SLO with no fast-load or snapshot story | Scale-to-zero reserved for batch/tolerant traffic, or paired with sub-second snapshot/streaming restore |
| Scale-up has no step cap — everything scales at once under a spike | Staggered scale-up (`Pods` policy with a step + period), avoiding a thundering herd on shared weight storage |
| No visibility into the metrics pipeline itself | Alerts on stale/missing custom or external metrics, with a safe default replica count as a fallback |
| Thresholds set by guesswork | Thresholds derived from load tests: the per-replica concurrency/queue depth at which TTFT just meets SLO |
| GPU-fraction/MIG replicas treated as full GPUs in capacity math | Quota, `maxReplicas`, and headroom formulas explicitly reference the correct unit (physical GPU vs. MIG slice vs. time-sliced share) |
| One global `maxReplicas` for a multi-region fleet | Per-region quota and `Pending`-pod alerting, since GPU capacity is granted regionally |

---

## Chapter Summary

1. **CPU is the wrong signal.** GPU saturation and host CPU are decoupled for an LLM server; scale on queue depth, concurrency, KV-cache pressure, and GPU utilization instead, with latency as a guardrail rather than the trigger.
2. **Cold starts are the defining constraint.** A 1–5 minute cold start (or worse) means hesitating on scale-up is the costliest mistake, and it means a warm floor or fast-load strategy is mandatory for any interactive SLO.
3. **Asymmetric behavior beats symmetric behavior.** Fast scale-up, slow scale-down — justified against your measured cold-start time, not a framework default — is the single highest-leverage tuning decision in this chapter.
4. **Composite signals beat a single signal.** Queue depth alone missed the KV-cache-preemption incident in War Story 1; the fix was adding a second, faster-moving signal, not replacing the first.
5. **`maxReplicas` needs a node-level counterpart.** Pod-level and node-level (Karpenter/cluster-autoscaler) scaling are two different control loops with two different response times — War Story 3 is what happens when only one is designed deliberately.
6. **Targets should be derived, not guessed.** Little's Law (\(L = \lambda \times W\)) turns a load-test sweep into a defensible concurrency or queue-depth target.
7. **2025–2026 tooling is converging on these same lessons**, just expressed through new knobs: AIBrix's fluctuation-tolerance parameters, Dynamo's per-pool prefill/decode signals, and KEDA's Cron and predictive layers are all restatements of "lead with a demand signal, dampen oscillation, and plan for known patterns" rather than replacements for understanding why.
8. **Validate the config before production traffic does it for you** — replay a trace-shaped load, kill the metrics pipeline on purpose, and force a real cold start from a real floor, rather than trusting the manifest alone.

## Glossary

| Term | Meaning |
|---|---|
| **TTFT** | Time-to-first-token — latency from request arrival to the first streamed token; the LLM-serving equivalent of "time to first byte" |
| **HPA** | Horizontal Pod Autoscaler — Kubernetes' native replica-count controller |
| **KEDA** | Kubernetes Event-Driven Autoscaling — wraps HPA, adds scale-to-zero and 70+ event-source scalers |
| **KPA** | Knative Pod Autoscaler — Knative Serving's default concurrency/RPS-based controller |
| **`ScaledObject`** | KEDA's CRD for scaling a `Deployment`/`StatefulSet` via one or more triggers |
| **`ScaledJob`** | KEDA's CRD for creating a Kubernetes `Job` per unit of work, for run-to-completion workloads |
| **`activationThreshold`** | KEDA's threshold governing the 0→1 transition, distinct from the 1→N scaling `threshold` |
| **`stabilizationWindowSeconds`** | The trailing window HPA looks back over before acting, to dampen flapping |
| **Cold start** | The end-to-end delay (schedule, pull, load, warm-up) before a newly started replica can serve a request |
| **Warm floor** | A `minReplicas`/`minScale` set ≥ 1 specifically to avoid ever paying a cold start on the interactive path |
| **Thundering herd** | Many replicas cold-starting simultaneously and contending for the same shared resource (weight storage, registry) |
| **KV cache** | The per-request key/value tensors an LLM engine caches to avoid recomputing attention over prior tokens; the dominant consumer of a replica's GPU memory |
| **Preemption** | vLLM evicting an in-flight request's KV-cache entries to free memory for others, effectively restarting that request |
| **Continuous batching** | Iteration-level batching where new requests join a running batch between decode steps, rather than waiting for the batch to fully drain |
| **Disaggregated serving** | Splitting prefill (compute-bound) and decode (memory-bandwidth-bound) onto separate GPU pools, scaled independently |
| **Little's Law** | \( L = \lambda \times W \) — ties average in-system population, throughput, and time-in-system together; the theoretical basis for concurrency/queue-depth targets |
| **Snapshot/checkpoint-restore** | Capturing an already-warmed process (CUDA context, weights resident in VRAM) and restoring it directly, skipping the load and warm-up phases on subsequent starts |
| **DCGM** | NVIDIA's Data Center GPU Manager — the source of `dcgm-exporter`'s Prometheus GPU metrics (utilization, memory) |
| **MIG** | NVIDIA Multi-Instance GPU — hardware partitioning of one physical GPU into several isolated instances, each schedulable as a separate `nvidia.com/gpu` unit |
| **Panic mode** | Knative KPA's short, fast-reacting evaluation window used when traffic spikes sharply, distinct from its normal steady-state smoothing window |
| **MAPE** | Mean Absolute Percentage Error — the accuracy check a predictive-scaling forecast (e.g. Kedify's Prophet-based `MetricPredictor`) validates itself against before trusting its own prediction |

## Further Reading

- Kubernetes — [Horizontal Pod Autoscaler (v2, behavior & algorithm)](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- Kubernetes — [HPA Walkthrough with custom metrics](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/)
- KEDA — [Prometheus scaler](https://keda.sh/docs/latest/scalers/prometheus/) and [Scaling Deployments (activation vs threshold, scale-to-zero)](https://keda.sh/docs/latest/concepts/scaling-deployments/)
- KEDA — [Cron scaler (scheduled scaling windows)](https://keda.sh/docs/2.20/scalers/cron/)
- KEDA — [GitHub issue #6934: LLM Scaler / "Cognitive Scaling" proposal](https://github.com/kedacore/keda/issues/6934)
- Kedify — [Predictive Autoscaling for Kubernetes (Prophet-based `MetricPredictor`)](https://kedify.io/resources/blog/predictive-autoscaling/)
- Kedify — [Kubernetes Autoscaling Use Cases: FinOps, AI/GPU, APIs](https://kedify.io/use-cases/)
- AWS — [Autoscale AI inference with HPA and KEDA on EKS (vLLM)](https://docs.aws.amazon.com/eks/latest/userguide/ml-inference-autoscaling-hpa-keda.html)
- vLLM — [Autoscaling with KEDA (production-stack)](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html)
- Knative — [Autoscaling (KPA, concurrency, scale-to-zero, panic mode)](https://knative.dev/docs/serving/autoscaling/)
- Prometheus Adapter — [kubernetes-sigs/prometheus-adapter](https://github.com/kubernetes-sigs/prometheus-adapter)
- NVIDIA — [dcgm-exporter (GPU metrics for Prometheus)](https://github.com/NVIDIA/dcgm-exporter)
- NVIDIA — [Dynamo Snapshot: fast startup for inference on Kubernetes](https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/)
- NVIDIA — [Reducing Cold Start Latency for LLM Inference with Run:ai Model Streamer (benchmarks)](https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer)
- vLLM docs — [Loading models with Run:ai Model Streamer](https://docs.vllm.ai/en/stable/models/extensions/runai_model_streamer/)
- Microsoft Azure — [Eliminate LLM cold starts: load models up to 6x faster with Run:ai Model Streamer](https://devblogs.microsoft.com/azure-sdk/eliminate-llm-cold-starts-load-models-up-to-6x-faster-with-azure-blob-storage-and-runai-model-streamer/)
- Azure AKS Engineering Blog — [Stream model weights to NVIDIA GPU (vLLM) from Azure Blob Storage using Run:ai Model Streamer](https://blog.aks.azure.com/2026/07/13/runai-streamer-vllm)
- Azure AKS Engineering Blog — [Autoscale KAITO inference workloads on AKS using KEDA](https://blog.aks.azure.com/2026/02/03/autoscale-inference-workloads-with-kaito)
- Google Cloud — [Accelerate model downloads on GKE with NVIDIA Run:ai Model Streamer](https://cloud.google.com/blog/products/containers-kubernetes/nvidia-runai-model-streamer-supports-cloud-storage)
- Beam — [The Top Serverless GPU Providers, Ranked by Cold Start (2025)](https://www.beam.cloud/blog/top-serverless-gpu-providers)
- Microsoft / Azure — [AzurePublicDataset: AzureLLMInferenceDataset2023 (real production traffic trace)](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md)
- Microsoft Research — [Splitwise: Efficient Generative LLM Inference Using Phase Splitting (ISCA 2024)](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/12/Splitwise_ISCA24.pdf)
- Tensorfuse — [Reducing GPU cold start time when using vLLM (294s → 82s breakdown)](https://tensorfuse.io/docs/blogs/reducing_gpu_cold_start)
- DEV Community — [Why vLLM autoscaling on Kubernetes breaks (and what to use instead)](https://dev.to/soniarotglam/why-vllm-autoscaling-on-kubernetes-breaks-and-what-to-use-instead-1231)
- OneUptime — [Using HPA stabilizationWindowSeconds to prevent scaling thrashing](https://oneuptime.com/blog/post/2026-02-09-hpa-stabilization-window-prevent-thrashing/view)
- KServe — [Autoscaler for generative inference](https://kserve.github.io/website/docs/model-serving/generative-inference/autoscaling)
- NVIDIA — [Dynamo adds GPU autoscaling, Kubernetes automation, and networking optimizations](https://developer.nvidia.com/blog/nvidia-dynamo-adds-gpu-autoscaling-kubernetes-automation-and-networking-optimizations)
- NVIDIA — [Dynamo accelerates llm-d community initiatives for large-scale distributed inference](https://developer.nvidia.com/blog/nvidia-dynamo-accelerates-llm-d-community-initiatives-for-advancing-large-scale-distributed-inference)
- NVIDIA — [Deploying disaggregated LLM inference workloads on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- Azure AKS Engineering Blog — [Scaling multi-node LLM inference with NVIDIA Dynamo and GB200 NVL72 GPUs on AKS](https://blog.aks.azure.com/2025/10/24/dynamo-on-aks)
- vLLM Blog — [Introducing AIBrix: a scalable, cost-effective control plane for vLLM](https://vllm.ai/blog/2025-02-21-aibrix-release)
- AIBrix — [Metric-based autoscaling docs (HPA/KPA/APA algorithms, PodAutoscaler CRD)](https://aibrix.readthedocs.io/latest/features/autoscaling/metric-based-autoscaling.html)
- AIBrix — [GitHub repository](https://github.com/vllm-project/aibrix)
- AWS — [EKS Best Practices: AI/ML compute autoscaling (Karpenter + KEDA, capacity reservations)](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html)
- OpenCost — [Kubernetes cost allocation, including GPU cost attribution](https://www.opencost.io)
- Kubecost — [Cost analyzer for Kubernetes (OpenCost-based, GPU/AI cost tracking)](https://kubecost.github.io/cost-analyzer/)

---

*Related chapters: [Load Testing](../04_load_testing/README.md) to derive your target thresholds, [vLLM Serving](../05_vllm_serving/README.md) for the engine metrics, and [Monitoring](../08_monitoring/README.md) for the Prometheus/Grafana pipeline that feeds every controller above.*
