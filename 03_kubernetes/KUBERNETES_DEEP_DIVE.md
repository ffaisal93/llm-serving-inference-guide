# Kubernetes for LLM Serving

> Deploying and operating GPU inference on Kubernetes — the control plane that most production LLM stacks run on.

## Why this matters

Once you have a working inference server (vLLM, TGI, Triton, SGLang), the next question is always the same: *how do I run twelve of these across a fleet of GPU nodes, upgrade them without dropping traffic, and not set money on fire?* Kubernetes is the de-facto answer. It gives you a declarative fleet, health-based traffic gating, rolling upgrades, and a scheduler that can place a pod on exactly the right GPU.

But LLM serving breaks several of Kubernetes' defaults. Models take **minutes** to load, so naive probes will kill pods mid-startup in an infinite crash loop. GPUs are **not overcommittable**, so the usual CPU/memory bin-packing intuition is wrong. Container images and model weights are **tens of gigabytes**, so image pulls and cold starts dominate. This chapter walks the mechanisms and the sharp edges.

Autoscaling (HPA, KEDA, custom metrics, scale-to-zero) is deep enough to deserve its own chapter — see *Autoscaling GPU Inference*. Here we reference it but focus on deployment, scheduling, health, weights, and lifecycle.

> **Saying it out loud.** Once you've got a working inference server, the question stops being "does it generate text" and becomes "how do I run twelve of these across a GPU fleet, upgrade them without dropping traffic, and not set money on fire." Kubernetes is the default answer because it gives you a declarative fleet, health-gated routing, rolling upgrades, and a scheduler that can place a pod on exactly the right GPU. But LLM serving breaks three Kubernetes defaults hard: models take *minutes* to load, so naive probes crash-loop them forever; GPUs can't be overcommitted, so your CPU bin-packing intuition is simply wrong; and images plus weights are tens of gigabytes, so cold start dominates everything. Most Kubernetes-for-LLM incidents are one of those three defaults biting.

---

## Core intuition

Three mental models carry most of the chapter:

1. **A GPU is an indivisible, non-overcommittable device.** Unlike CPU (compressible) and memory (overcommittable at your peril), a GPU is handed to exactly one container by the device plugin. You can *share* it deliberately (time-slicing, MPS, MIG), but the scheduler still treats each advertised unit as an integer resource. There is no "burst above your GPU limit."

2. **Health is a function of time, not just liveness.** A pod that has been alive for 40 seconds but hasn't loaded a 140 GB model is *not broken* — it's *starting*. Kubernetes has a dedicated primitive for exactly this distinction: the **startup probe**. Get this wrong and you get a crash loop that looks like a hardware failure.

3. **The weights are the workload.** For classic web services the container image *is* the app. For LLM serving the image is a runtime and the weights are the payload — often 10x larger than the image. Where the weights live and how they land on the node (baked into the image, PVC, `initContainer`, object storage, model cache) determines your cold-start time and your blast radius.

4. **Voluntary disruption is the one you control.** Nodes get drained for upgrades, spot reclamation, and autoscaler scale-down constantly. Kubernetes lets you *bound* that damage with a PodDisruptionBudget and a graceful-shutdown path. Unlike a crashed process (involuntary), these are scheduled, negotiable evictions — and the difference between a clean rolling drain and a full outage is a few lines of YAML you either wrote or didn't.

Keep these four in mind and the rest of the chapter is mostly detail: GPUs are exclusive integers, health is time-aware, weights are the real payload, and disruptions are bounded on purpose.

> **Saying it out loud.** Four mental models carry this whole chapter. One: a GPU is an indivisible, non-overcommittable device — unlike CPU which is compressible and memory which you can oversubscribe, a GPU goes to exactly one container as an integer, and there's no bursting above your limit. Two: health is a function of time, not just liveness — a pod that's been alive forty seconds without finishing a 140 GB model load isn't broken, it's starting, and Kubernetes has a dedicated primitive for that called the startup probe. Three: the weights are the workload — for a normal service the image *is* the app, here the image is just a runtime and the weights are ten times larger. Four: voluntary disruption is the one you control, and a PodDisruptionBudget is the few lines of YAML between a clean rolling drain and a full outage.

---

## Mechanisms in depth

### 1. The building blocks: Deployment, Service, Ingress

For a stateless replicated inference server the standard trio is:

- **Deployment** — declares *N* replicas of a pod template, handles rolling updates and self-healing.
- **Service** — a stable virtual IP + DNS name that load-balances across the ready pods (`ClusterIP` for in-cluster, `LoadBalancer` for cloud L4).
- **Ingress** (or Gateway API) — L7 routing, TLS termination, path/host rules, into the Service.

A subtlety unique to LLM serving: **long request durations and streaming**. Token-streaming responses (SSE) can run for tens of seconds to minutes. Make sure your Ingress/proxy timeouts (`proxy-read-timeout` on the NGINX ingress, backend request timeout on cloud LBs) are raised, and that buffering is disabled so tokens flush as they generate. Default 30–60s timeouts will cut long generations.

> **Saying it out loud.** The standard trio is boring and that's good: a Deployment declares N replicas and handles rolling updates and self-healing, a Service gives you a stable virtual IP that load-balances across the *ready* pods, and an Ingress or Gateway does L7 routing and TLS on the way in. The one thing genuinely different about LLM serving is request duration — a token-streaming response can run for minutes, and every proxy in the path defaults to a 30- or 60-second read timeout. So you raise `proxy-read-timeout` on the ingress and the backend timeout on any cloud load balancer, and you disable response buffering so tokens flush as they're generated. Miss that and long generations get truncated mid-sentence, and it'll look like a model bug rather than a proxy config.

### 2. GPU scheduling: the NVIDIA device plugin

Kubernetes has no native notion of a GPU. The **NVIDIA device plugin** is a DaemonSet that runs on every GPU node, discovers the GPUs, and *advertises* them to the kubelet as an **extended resource** named `nvidia.com/gpu`. The scheduler then treats that resource like any countable resource.

You request GPUs under `resources`:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1   # request one whole GPU
```

Key rules that trip people up:

- **Extended resources must be integers, and request must equal limit.** Kubernetes requires that for any extended resource, if you set it at all, the request and limit are equal. You cannot request 0.5 of a `nvidia.com/gpu` and burst to 1. Practically you only ever specify it under `limits` (Kubernetes copies it to `requests` for you).
- **GPUs are never overcommitted.** Two pods cannot each hold `nvidia.com/gpu: 1` on a node that advertises one GPU — the second stays `Pending`. This is by design: two processes fighting over one GPU's memory would OOM each other unpredictably.
- **Sharing is opt-in and explicit.** If you want to pack multiple pods on one GPU you enable **time-slicing** (the plugin advertises, say, 4 "replicas" of each GPU — but note this is *oversubscription with no memory isolation*, so proportional compute is **not** guaranteed), **MPS**, or hardware **MIG** partitions. Each mechanism changes what the plugin advertises; the scheduler math stays "integer units."

The device plugin is often installed as part of the **NVIDIA GPU Operator**, which additionally manages the driver, the container toolkit, DCGM metrics exporter, Node Feature Discovery, and MIG configuration — so you don't hand-install drivers on every node.

> **Saying it out loud.** Kubernetes has no built-in concept of a GPU at all. What makes it work is the NVIDIA device plugin — a DaemonSet on every GPU node that discovers the cards and advertises them to the kubelet as an extended resource called `nvidia.com/gpu`. From there the scheduler treats it like any countable resource, with two rules people trip on. Extended resources must be whole integers and request must equal limit, so there's no requesting half a GPU and bursting. And GPUs are never double-booked: two pods each asking for one GPU on a single-GPU node means the second sits `Pending`, by design, because two processes fighting over one card's VRAM would OOM each other unpredictably. Sharing exists, but it's explicit opt-in via MIG, MPS, or time-slicing.

### 2b. GPU sharing: MIG vs time-slicing vs MPS

One whole GPU per pod is wasteful for small models that use a few GB of a 80 GB card. Three mechanisms let you pack more, each changing what the device plugin advertises:

| Mechanism | Isolation | How it splits | Advertised as | Use when |
|-----------|-----------|---------------|---------------|----------|
| **MIG** (Multi-Instance GPU) | **Hardware** — separate memory + compute slices | Physically partitions an A100/H100 into up to 7 instances | `nvidia.com/mig-1g.10gb` etc. (or relabeled `nvidia.com/gpu`) | Strong isolation, predictable QoS, multi-tenant |
| **Time-slicing** | **None** — processes share memory, take turns on the SMs | Plugin advertises N "replicas" of each GPU; the driver context-switches | `nvidia.com/gpu` (inflated count) | Bursty/low-QPS dev workloads that tolerate contention |
| **MPS** (Multi-Process Service) | **Soft** — shared memory, concurrent kernels with optional compute caps | A daemon runs many clients' kernels concurrently | `nvidia.com/gpu` (configured slots) | Higher utilization than time-slicing, some control |

Critical caveat: **time-slicing gives no memory isolation** — two pods on one time-sliced GPU can OOM each other, and "requesting 2 shared GPUs" does **not** guarantee 2x the compute. For production multi-tenant serving, **MIG** is the safe choice; time-slicing/MPS are for dev or trusted, well-characterized co-tenancy. All three are configured via the device plugin / GPU Operator, not by the pod author.

> **Saying it out loud.** Giving a whole 80 GB card to a model that uses six gigabytes is wasteful, so there are three ways to pack more on — and they differ mainly in what guarantee they give you. MIG is hardware partitioning: an A100 or H100 splits into up to seven instances with genuinely separate memory and compute slices, so tenants can't touch each other. Time-slicing is pure oversubscription: the plugin just advertises four replicas of one card and the driver context-switches, with zero memory isolation. MPS sits in between — concurrent kernels with soft compute caps but still shared memory. The caveat that matters: time-slicing does *not* give you 4x compute and does *not* stop two pods from OOM-ing each other, so for anything multi-tenant you use MIG and treat the other two as dev-only.

### 3. Getting pods onto GPU nodes: selectors, taints, tolerations

You almost never want a random CPU workload landing on an expensive GPU node, and you want GPU pods to land *only* on GPU nodes. Two complementary mechanisms:

- **Node labels + `nodeSelector`/affinity (attraction).** GPU nodes carry labels — cloud pools add things like `cloud.google.com/gke-accelerator=nvidia-l4`, and the GPU Operator / NFD add labels such as `nvidia.com/gpu.product=NVIDIA-A100-SXM4-80GB`. You pull your pod toward them:

  ```yaml
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB
  ```

- **Taints + tolerations (repulsion).** You taint GPU nodes so *nothing* schedules there unless it explicitly tolerates the taint. Cloud GPU pools often auto-apply a taint like `nvidia.com/gpu=present:NoSchedule`. Your inference pod must tolerate it:

  ```yaml
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  ```

Use both: the taint keeps freeloaders off, the selector/affinity ensures your pod picks the right GPU SKU. A **GPU node pool** is simply a node group with a fixed instance type (all A100, or all L4), its own taint, and often its own cluster-autoscaler settings so you can scale GPU capacity independently of the CPU fleet.

> **Saying it out loud.** You need two complementary things, and people usually remember one. Taints are repulsion: you taint GPU nodes so nothing schedules there unless it explicitly tolerates the taint, which keeps random CPU workloads off your expensive hardware — cloud GPU pools often apply `nvidia.com/gpu=present:NoSchedule` for you. Node selectors and affinity are attraction: they pull your pod toward the right SKU using labels the GPU Operator and Node Feature Discovery add, like `nvidia.com/gpu.product`. Use both, because they solve different problems. The failure mode from getting it wrong is silent: a pod missing the toleration doesn't error, it just sits `Pending` forever, and the only place that tells you is `kubectl describe pod`.

### 4. Probes done right for multi-minute model loads

This is the single most common LLM-on-k8s bug. Kubernetes has three probes:

| Probe | Question it answers | Failure action |
|-------|--------------------|----------------|
| **startup** | "Has the container finished starting *yet*?" | Kill & restart the container (crash loop). Disables the other two until it first succeeds. |
| **readiness** | "Should this pod receive traffic *right now*?" | Remove pod from Service endpoints (no traffic), do **not** kill. |
| **liveness** | "Is this container wedged and needs a restart?" | Kill & restart the container. |

The classic failure: you set a **liveness** probe with a short `initialDelaySeconds`, the model takes 4 minutes to load, the liveness probe fails during load, the kubelet kills the container, it restarts, tries to load again, gets killed again — an infinite crash loop that looks like the model is broken.

**The fix is the startup probe.** While a startup probe is configured and not yet successful, liveness and readiness are *suppressed*. So you give the startup probe a generous budget:

```yaml
startupProbe:
  httpGet:
    path: /health
    port: 8000
  periodSeconds: 10
  failureThreshold: 60   # 10s * 60 = up to 600s (10 min) to become healthy
```

The effective grace window is `periodSeconds * failureThreshold`. Size it to your *worst-case* cold load (weights download + load into VRAM + CUDA graph capture / warmup), then add margin. Once the startup probe passes once, the fast liveness and readiness probes take over.

- **readiness** should reflect "can serve a request" — many servers expose `/health` (up) vs a readiness endpoint that only returns 200 once the model is loaded and warmup is done. Gate traffic on the latter.
- **liveness** should be *cheap and lenient* — it exists to recover a genuinely wedged process (deadlock, CUDA error), not to police slow loads. Use a modest `periodSeconds` and a `failureThreshold` of 3+ so a single slow health check doesn't kill a healthy pod mid-inference.

> **Saying it out loud.** This is the number-one LLM-on-Kubernetes bug, full stop. There are three probes: startup asks "has it finished starting yet," readiness asks "should traffic go here right now," and liveness asks "is this wedged and needs a restart." The classic disaster is copying a liveness probe from a CPU microservice — thirty seconds of initial delay, three failures allowed — onto a model that takes four minutes to load. The kubelet kills it at sixty seconds, it restarts, loads again, gets killed again, forever. The fix is the startup probe, because while it's pending it *suppresses* liveness and readiness entirely; give it `periodSeconds: 10` and `failureThreshold: 60` for a ten-minute budget sized to your worst-case cold load. Then keep liveness cheap and lenient — it exists to recover a deadlock, not to police a slow load.

### 5. Resource requests/limits and why GPUs aren't overcommitted

For CPU and memory you set requests (scheduling guarantee) and limits (cap). For LLM pods:

- **CPU:** set a request so the scheduler reserves headroom (tokenization, HTTP, scheduling loops are CPU-hungry), but be cautious with CPU *limits* — throttling the server's event loop can tank throughput. Many teams set a CPU request and no CPU limit.
- **Memory:** set request and limit close together and *generous*. Host RAM is used to stage weights before they hit VRAM; an OOMKill mid-load looks like a probe failure but is actually the kernel.
- **GPU:** `nvidia.com/gpu` request == limit == integer, always. There is no overcommit. The reason is physical: GPU memory (VRAM) has no swap and no soft limit the scheduler understands. If two pods both assumed they had the whole 80 GB, the second allocation would fail at CUDA-malloc time, not at schedule time — an ugly runtime crash instead of a clean `Pending`. So Kubernetes refuses to double-book.

Corollary — **GPU fragmentation**: because allocation is integer and per-node, a cluster with eight 8-GPU nodes and lots of single-GPU pods can end up unable to schedule a pod that needs 4 GPUs *on one node*, even though 20 GPUs are free cluster-wide. Multi-GPU / tensor-parallel pods need whole nodes or careful bin-packing (topology-aware scheduling, `podAffinity`, or a scheduler like the NVIDIA/Volcano gang scheduler).

> **Saying it out loud.** For CPU, set a request so the scheduler reserves headroom but be careful with limits, because throttling the event loop that handles tokenization and HTTP will tank your throughput — many teams set a CPU request and no limit at all. For memory, set request and limit close together and generous, because host RAM stages the weights before they land in VRAM, and an OOMKill mid-load looks exactly like a probe failure. For GPU, request equals limit equals an integer, always. The physical reason is that VRAM has no swap and no soft limit the scheduler understands: if two pods both assumed they had the whole 80 GB, the second would fail at CUDA-malloc time — an ugly runtime crash instead of a clean `Pending`. The corollary is fragmentation: twenty GPUs free cluster-wide can still mean zero schedulable for a pod that needs four on one node.

### 6. Serving the model weights

Where do the weights come from at pod start? Four common patterns, with tradeoffs:

| Approach | How | Pros | Cons |
|----------|-----|------|------|
| **Baked into image** | `COPY` weights into the Docker image | Simplest; immutable; no runtime fetch | Enormous images (30–100 GB+), slow pulls, registry bloat, rebuild to change weights |
| **initContainer download** | An `initContainer` pulls weights from object storage (S3/GCS) into a shared `emptyDir` | Small runtime image; weights versioned in bucket | Re-downloads on every cold start unless cached; needs credentials |
| **PVC (shared/RWX)** | Weights on a persistent volume (e.g. a network filesystem), mounted read-only | Download once, many pods share; fast pod start | Storage class must support RWX or you pre-populate; network FS bandwidth can bottleneck concurrent loads |
| **Node-local cache / model cache** | Cache weights on local NVMe, or use a model-cache layer (e.g. Run:ai model streamer, KServe modelcar, Fluid) | Fast warm starts; streams weights into VRAM | More moving parts; cache warmup / eviction to manage |

Rules of thumb: bake weights into the image only for small models or when immutability matters more than pull time. For large models, keep the runtime image lean and fetch weights via `initContainer` or a pre-populated read-only PVC, and *cache on the node* so replicas 2..N start fast. Beware the **thundering herd**: ten pods cold-starting simultaneously all pulling 140 GB from the same bucket will saturate egress and each other.

> **Saying it out loud.** Four ways to get weights onto a node, and the choice determines your cold-start time. Baked into the image is simplest and immutable, but you're pulling a 30-to-100-gigabyte image on every cold node. An initContainer downloading from S3 keeps the image lean but re-downloads on every cold start unless you cache it. A shared read-only PVC means you download once and many pods mount it — usually the right answer for large models — though your network filesystem bandwidth becomes the bottleneck when replicas load concurrently. Node-local NVMe cache is fastest for warm restarts but adds cache-warming and eviction to manage. The failure mode to name is the thundering herd: ten pods cold-starting at once, each pulling 140 GB from the same bucket, saturating egress and slowing each other down.

### 7. Rolling updates, PodDisruptionBudgets, graceful shutdown

- **Rolling updates:** the Deployment's `RollingUpdate` strategy with `maxUnavailable` / `maxSurge` controls how many pods are replaced at once. For GPU pods, `maxSurge` costs *real extra GPUs* — surging by 1 means the autoscaler must find another GPU node. Often you set `maxSurge: 0, maxUnavailable: 1` to avoid needing spare GPUs, accepting slightly reduced capacity during the rollout. And remember: each new pod pays the full multi-minute cold-start, so rollouts of GPU fleets are *slow*. Budget for it.

- **PodDisruptionBudget (PDB):** protects against *voluntary* disruptions (node drains, cluster-autoscaler scale-down, upgrades). Without a PDB, a node drain can evict all your replicas at once and take the service down. Set `minAvailable` (or `maxUnavailable`) so the eviction API refuses to take down too many at once:

  ```yaml
  apiVersion: policy/v1
  kind: PodDisruptionBudget
  spec:
    minAvailable: 2
    selector:
      matchLabels: { app: llm-inference }
  ```

- **Graceful shutdown:** on SIGTERM, a good inference server should stop accepting new requests, drain in-flight generations, then exit. Kubernetes gives it `terminationGracePeriodSeconds` (default 30s) before SIGKILL — raise this well above your longest expected generation (e.g. 120–300s) so streaming requests aren't cut off. Pair it with a `preStop` hook or a readiness flip so the pod is pulled from Service endpoints *before* it starts draining, avoiding races where traffic hits a shutting-down pod.

> **Saying it out loud.** Three lifecycle things, and each has a GPU-specific twist. On rolling updates, `maxSurge` costs *real extra GPUs* — surging by one means the autoscaler has to find another GPU node, which may not exist — so GPU fleets often run `maxSurge: 0, maxUnavailable: 1` and accept reduced capacity during the rollout. On PodDisruptionBudgets: without one, a routine node drain or an autoscaler scale-down can evict every replica simultaneously and take the whole service down, so you always ship a `minAvailable`. On graceful shutdown: the default `terminationGracePeriodSeconds` is thirty seconds, which will SIGKILL a pod in the middle of a two-minute generation, so raise it to a couple hundred and flip readiness first so traffic drains before the pod starts shutting down.

### 8. Serving frameworks & operators

You don't have to hand-roll Deployments. Higher-level tools add model-aware features (autoscaling on GPU/queue metrics, scale-to-zero, canary, standardized model formats):

- **KServe** — a CRD (`InferenceService`) on top of Knative/Kubernetes. Handles autoscaling (incl. scale-to-zero), canary rollout, and a standard prediction protocol. Has first-class support for LLM runtimes (vLLM) via `ServingRuntime`. Good when you want a platform abstraction over raw pods.
- **NVIDIA NIM / Triton on k8s** — NIM packages optimized model microservices as containers; the **NIM Operator** (and Triton) deploy them, and NIM integrates with **KServe** for the serving layer. Best when you're standardized on NVIDIA's optimized stack and want vendor-supported images.
- **KubeAI** — an open, k8s-native inference operator focused on OpenAI-compatible serving of LLMs (vLLM/Ollama), with built-in autoscaling and model management, no Istio/Knative dependency. Lighter-weight alternative to KServe.
- **Ray Serve (KubeRay)** — deploy via the `RayService` CRD. Shines for **multi-model, model-composition, and distributed (multi-node tensor/pipeline-parallel)** serving where a request fans across many actors/GPUs. More of a distributed compute framework than a thin serving layer.

> **Saying it out loud.** You don't have to hand-roll Deployments forever. KServe gives you an `InferenceService` CRD with autoscaling including scale-to-zero, canary rollouts, and first-class vLLM support — good when you're a platform team abstracting over many models. KubeAI is a lighter alternative focused on OpenAI-compatible LLM serving without dragging in Istio and Knative. Ray Serve via KubeRay is the one to reach for when a single request has to fan across many GPUs or nodes, or when you're composing models together — it's really a distributed compute framework, not a thin serving layer. And NVIDIA NIM plus the NIM Operator is the vendor-supported path if you're standardized on NVIDIA's optimized engines. The honest tradeoff: each of these buys you features and costs you a layer of abstraction to debug through.

---

### 9. Networking specifics: Gateway API, timeouts, affinity

- **Ingress vs Gateway API.** The classic `Ingress` resource works, but the newer **Gateway API** (`Gateway` + `HTTPRoute`) is the direction the ecosystem is moving and expresses timeouts, traffic splitting, and header routing more cleanly — useful for canarying model versions.
- **Streaming timeouts.** Token-by-token SSE/HTTP responses can run minutes. On the NGINX ingress set `nginx.ingress.kubernetes.io/proxy-read-timeout` and `proxy-send-timeout` to several hundred seconds and disable buffering (`proxy-buffering: "off"`) so tokens flush live. Cloud L7 LBs have their own backend timeout you must raise.
- **Session affinity for KV-cache reuse.** With prefix/KV caching, routing a follow-up request to the *same* replica that holds the cache boosts throughput. Basic `sessionAffinity: ClientIP` on the Service helps; smarter setups use a cache-aware router (e.g. the vLLM production stack / router) instead of round-robin.
- **Headless Services for multi-node.** Distributed (tensor/pipeline-parallel across nodes) runtimes often need pod-to-pod addressing; a headless Service (`clusterIP: None`) plus a StatefulSet gives stable per-pod DNS.

> **Saying it out loud.** Four networking things bite specifically for LLMs. Gateway API is where the ecosystem is going over classic Ingress, and it expresses timeouts and traffic splitting much more cleanly, which matters when you're canarying model versions. Streaming timeouts are the practical killer — set `proxy-read-timeout` to several hundred seconds and turn buffering off, or your SSE stream gets cut. Session affinity actually matters here in a way it doesn't for stateless services, because with prefix caching, routing a follow-up request back to the replica that already holds that KV cache is a real throughput win — `sessionAffinity: ClientIP` is the crude version, a cache-aware router is the good one. And for multi-node tensor parallelism you need a headless Service plus a StatefulSet for stable pod-to-pod DNS.

### 10. Observability: know when a GPU pod is unhealthy

Standard pod metrics miss the GPU. Add:

- **DCGM exporter** (shipped by the GPU Operator) → Prometheus: GPU utilization, memory used, temperature, ECC errors, throttling. Alert on sustained 0% utilization on a "ready" pod (stuck), on VRAM near 100% (OOM risk), and on XID/ECC errors (failing hardware).
- **Server-level metrics** from the runtime: queue depth, time-to-first-token, tokens/sec, running vs waiting requests. These drive autoscaling (see the autoscaling chapter) and tell you *why* latency moved.
- **Event/probe signals:** watch for `Unhealthy` probe events, `CrashLoopBackOff`, and `FailedScheduling` (usually taint/selector or capacity). `kubectl describe pod` and `kubectl get events` are your first stop.

A "ready" pod pinned at 0% GPU utilization with a growing request queue is the classic silent failure — the health endpoint returns 200 but inference is wedged. Alert on the *metric*, not just the probe.

> **Saying it out loud.** Standard pod metrics tell you nothing about the GPU, so you add two layers. DCGM exporter, which ships with the GPU Operator, feeds Prometheus GPU utilization, memory, temperature, ECC errors, and throttling. And server-level metrics from the runtime itself — queue depth, time-to-first-token, tokens per second, running versus waiting requests — which are what actually drive autoscaling and tell you *why* latency moved. The specific alert worth naming is the silent failure: a pod that passes its readiness probe, sits at 0% GPU utilization, and has a growing request queue. The HTTP endpoint returns 200, so Kubernetes thinks it's fine, but inference is wedged. Alert on the metric, not just the probe — the probe is exactly the thing that's lying to you.

### 11. Spot / preemptible GPUs and cost

GPU nodes are the dominant cost, so many teams run inference on **spot/preemptible** instances at a large discount — accepting that the cloud can reclaim the node with ~30–120s notice.

- Spread replicas across on-demand and spot with `topologySpreadConstraints` so a spot reclamation storm can't take the whole service down; keep a baseline of on-demand capacity protected by the PDB.
- The preemption signal arrives as a node drain → your **graceful shutdown** path (SIGTERM, drain, `terminationGracePeriodSeconds`) must fit inside the cloud's notice window, or in-flight requests are lost.
- Cold-start time is your enemy here: a reclaimed spot pod must re-download weights and reload the model before serving. Node-local weight caches and pre-pulled images shrink the recovery gap.
- Right-size the GPU: an 8B model on an 80 GB H100 wastes the card — MIG-slice it or pick a smaller SKU (L4/L40S) and let the *comparison table* of frameworks + autoscaling do the packing.

> **Saying it out loud.** GPU nodes dominate the bill, so running inference on spot or preemptible instances is tempting — big discount, but the cloud can reclaim the node on roughly 30 to 120 seconds notice. Three things make that survivable. Spread replicas across on-demand and spot with topology spread constraints, and keep an on-demand baseline protected by a PDB, so a reclamation storm can't take the whole service down. Make sure your graceful shutdown path — SIGTERM, drain, terminate — actually fits inside that notice window, because if it doesn't, every reclamation drops in-flight requests. And cold start is your real enemy: a reclaimed pod has to re-fetch weights and reload the model before it serves anything, so node-local weight caches and pre-pulled images are what shrink the recovery gap.

---

## The 2025–2026 landscape

The mechanisms above are stable, but the tooling around them moved fast between 2025 and 2026. Five developments matter most if you're standing up a new GPU-serving platform today.

> **Saying it out loud.** The mechanisms are stable but the tooling moved fast. GPU scheduling is heading toward Dynamic Resource Allocation, which went GA in Kubernetes 1.34 and replaces "advertise an integer count" with claim-based allocation that can express things like "two GPUs on the same NVLink island." KServe grew LLM-native features — autoscaling on vLLM queue metrics via KEDA, declarative multi-node tensor and pipeline parallelism. A standard traffic layer arrived in the Gateway API Inference Extension, so routing to the replica holding the right prefix cache is no longer bespoke per team. And Kueue finally solved quota arbitration between serving and batch. The throughline: GPUs are still exclusive integer units — DRA makes the *claims* richer, it doesn't make the scarcity go away.

### GPU sharing has matured: MIG, time-slicing, MPS — and DRA is coming for all of them

Through 2025 the **NVIDIA GPU Operator** (on the 24.9.x/25.x release line) remained the standard way to install the device plugin, driver, container toolkit, DCGM exporter, Node Feature Discovery, and MIG manager as one Helm-installed stack rather than hand-provisioning each piece. Its `ClusterPolicy` CRD is the single control surface: flip `mig.strategy` between `single` and `mixed`, and the operator relabels nodes and reconfigures the device plugin automatically. Time-slicing configuration is a plain `ConfigMap` the device plugin reads at startup (shown in the extended example below) — still oversubscription with no memory isolation, exactly as described above.

The structurally bigger change is **Dynamic Resource Allocation (DRA)**, which graduated to General Availability in **Kubernetes v1.34** (released September 1, 2025). DRA replaces the device plugin's "advertise an integer count" model with **claim-based allocation**: a `ResourceClaim` lets a pod ask for a *class* of device with structured parameters (a specific MIG profile, a specific interconnect topology, a set of GPUs that share an NVLink domain) instead of just a count of `nvidia.com/gpu`. NVIDIA and the major cloud Kubernetes offerings shipped early DRA driver support in 2025–2026 specifically to express MIG profiles and multi-GPU topology (e.g. "give me 2 GPUs on the same NVLink island") in ways the old device-plugin integer model could never express. DRA does not replace the device plugin overnight — most production clusters in 2026 still run the classic `nvidia.com/gpu` device plugin — but it is the direction the scheduler is heading, and it is the answer if an interviewer asks "how would Kubernetes GPU scheduling need to evolve to express topology?"

> **Saying it out loud.** The GPU Operator is still how you install the whole host-side stack — device plugin, driver, toolkit, DCGM, MIG manager — as one Helm chart, with `ClusterPolicy` as the single control surface. The structural change is DRA, Dynamic Resource Allocation, which went GA in Kubernetes 1.34 in September 2025. Instead of asking for a count of `nvidia.com/gpu`, a pod files a `ResourceClaim` for a *class* of device with structured parameters — a specific MIG profile, a driver version floor, or two GPUs sharing an NVLink domain. That last one is the killer example, because the old integer model literally cannot express topology, which is exactly why multi-GPU pods end up stranded by fragmentation. Most 2026 clusters still run the classic device plugin, but DRA is the right answer to "how would GPU scheduling need to evolve."

### KServe grew LLM-native primitives

**KServe v0.15** (released June 18, 2025) added serving-specific features that a raw Deployment has to hand-roll:

- **KEDA-based autoscaling on vLLM metrics** — instead of scaling on CPU%, KEDA can scale an `InferenceService` on `vllm:num_requests_running` or queue depth scraped straight from the vLLM Prometheus endpoint, which tracks actual serving pressure far better than CPU ever could.
- **Multi-node inference** — native `pipelineParallelSize` / `tensorParallelSize` fields on the `ServingRuntime` so a model too big for one node (e.g. Llama 3.1 405B) can be declared, not hand-orchestrated with StatefulSets and headless Services.
- **Distributed KV cache via LMCache** — KV-cache offload and cross-replica cache sharing, cutting time-to-first-token on multi-turn traffic.
- **Envoy AI Gateway integration** — token-aware rate limiting and model-routing policy at the gateway layer.
- The release also bumped the bundled vLLM backend to 0.8.5, adding support for newer model families and an OpenAI-compatible embeddings API.

> **Saying it out loud.** KServe 0.15, mid-2025, added the things a raw Deployment makes you hand-roll. KEDA-based autoscaling on actual vLLM metrics — `num_requests_running`, queue depth, scraped from the vLLM Prometheus endpoint — instead of guessing from CPU percent, which for a GPU workload is close to meaningless. Native `tensorParallelSize` and `pipelineParallelSize` fields, so a model too big for one node is a declaration rather than a hand-orchestrated StatefulSet plus headless Service. Distributed KV cache via LMCache, which offloads and shares cache across replicas to cut time-to-first-token on multi-turn traffic. And Envoy AI Gateway integration for token-aware rate limiting. The pattern worth noticing: every one of those is a serving-specific concern that generic Kubernetes primitives handle badly.

### A standard traffic layer arrived: the Gateway API Inference Extension

Before mid-2025, every team invented its own "route to the replica holding the right prefix cache" logic. The **Gateway API Inference Extension** (introduced June 5, 2025, `sigs.k8s.io/gateway-api-inference-extension`) standardizes this with two new resources sitting on top of the Gateway API: **`InferencePool`** groups model-server pods sharing hardware/model config, and **`InferenceModel`/`InferenceObjective`** declares model identity and request priority (interactive chat vs. batch) for the router to act on. Implementations shipped fast: **Istio** added support in 2025, as did **NGINX Gateway Fabric**, and Google's **GKE Inference Gateway** is built directly on it — using signals like KV-cache utilization (via `GCPBackendPolicy`) to route model-aware rather than round-robin. This is the closest thing the ecosystem has to a standard answer for "how do you route LLM traffic on Kubernetes" as of 2026.

> **Saying it out loud.** Before mid-2025, every team wrote their own logic for "route this request to the replica that already has the right prefix cache." The Gateway API Inference Extension standardizes that with two resources on top of Gateway API: an `InferencePool` groups model-server pods that share hardware and model config, and `InferenceModel` or `InferenceObjective` declares model identity and request priority — so interactive chat and batch traffic can be routed and prioritized differently. Istio, NGINX Gateway Fabric, and Google's GKE Inference Gateway all shipped support, with GKE routing on signals like KV-cache utilization rather than round-robin. Why it matters: round-robin across LLM replicas actively destroys prefix-cache hit rates, so cache-aware routing is a throughput win you get from the traffic layer, not the model.

### Kueue: batch/queue scheduling for GPU jobs

**Kueue** (`kueue.sigs.k8s.io`, a Kubernetes SIG project, on the v0.18/v0.19 release line as of early 2026) fills a gap raw Kubernetes scheduling never solved: **fair, quota-aware admission of batch and GPU jobs across teams**. Where the default scheduler will happily let one team's fine-tuning job or eval sweep grab every GPU in the cluster, Kueue adds `ClusterQueue`/`LocalQueue` objects with quotas, borrowing/lending between cohorts, priority-based FIFO admission, and **gang scheduling** (all-or-nothing pod admission — no more a distributed training job launching 7 of 8 needed pods and deadlocking). It integrates with the cluster-autoscaler via *provisioning requests*, so a queued job can trigger new GPU nodes only once it's actually about to be admitted. **MultiKueue** extends this across clusters, dispatching an admitted job's pods to whichever connected cluster has capacity — relevant for fine-tuning/eval batch workloads more than steady-state serving, but increasingly used to arbitrate GPU access between a serving fleet and a training/eval fleet sharing the same pool.

> **Saying it out loud.** Kueue fills a gap the default scheduler never addressed: fair, quota-aware admission of GPU jobs across teams. Out of the box, one team's eval sweep or fine-tuning run can grab every GPU in the cluster and there's nothing to stop it. Kueue adds `ClusterQueue` and `LocalQueue` objects with quotas, borrowing and lending between cohorts, and priority-based FIFO admission. The feature I'd single out is gang scheduling — all-or-nothing admission — which stops a distributed job from launching seven of the eight pods it needs and deadlocking capacity while it waits for the eighth. It also hooks into the cluster autoscaler via provisioning requests, so a queued job only triggers new GPU nodes once it's actually about to be admitted rather than speculatively.

### Multi-cluster and multi-region serving

Two 2025–2026 developments matter here. First, **llm-d** — a distributed inference stack founded by Red Hat, Google Cloud, IBM Research, CoreWeave, and NVIDIA, which joined the CNCF as a sandbox project in March 2026. It sits above vLLM/SGLang and adds **disaggregated prefill/decode** (splitting the compute-bound prefill phase from the memory-bandwidth-bound decode phase across different hardware), **prefix-cache-aware and predicted-latency routing**, and **tiered KV-cache offload** to CPU/disk — reporting up to 70% higher tokens/sec in disaggregated configurations and roughly 50k output tokens/sec on large clusters as of its v0.5 release (February 2026).

Second, **multi-cluster GKE Inference Gateway** (announced March 2026) extends the Gateway API Inference Extension across cluster and region boundaries: a dedicated **config cluster** holds the routing policy while multiple **target clusters** run the actual model pods, giving you cross-region failover, GPU/TPU capacity pooling (burst into whichever region has free accelerators), and model-aware routing globally instead of per-cluster. The pattern generalizes beyond GKE: the shared idea is *separate the routing control plane from the serving data plane* so a regional outage or a capacity crunch in one cluster doesn't take down the whole service — the multi-region equivalent of the PodDisruptionBudget mindset from single-cluster serving.

**What this means for the chapter's core intuitions:** GPUs are still exclusive, integer-scheduled units — DRA makes the *claims* richer, not the underlying scarcity softer. Health is still time-aware — KServe's KEDA integration scales on real serving pressure instead of guessing from CPU. The weights are still the workload — llm-d's disaggregation and KV-cache tiering are just more sophisticated answers to "where do the weights/cache live." And disruption is still bounded on purpose — multi-cluster routing is the PDB idea applied at the scale of a whole region.

> **Saying it out loud.** Two things worth knowing here. llm-d is a distributed inference stack — Red Hat, Google, IBM, CoreWeave, NVIDIA — that joined the CNCF as a sandbox project in March 2026, and it sits above vLLM and adds disaggregated prefill and decode, meaning you run the compute-bound prefill phase and the memory-bandwidth-bound decode phase on different hardware sized for each. They report up to 70% higher tokens per second in disaggregated configurations. Separately, multi-cluster GKE Inference Gateway extends the Inference Extension across regions, with a config cluster holding routing policy and target clusters running the pods. The generalizable idea in both: separate the routing control plane from the serving data plane, which is really the PodDisruptionBudget mindset applied at the scale of a region.

---

## Fully worked example: raw Deployment + Service

A complete, correct manifest for a vLLM server on a single A100, with GPU request, startup/readiness/liveness probes tuned for a slow load, weights fetched from object storage by an `initContainer` into a shared cache, a PDB, and graceful shutdown.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
  labels: { app: llm-inference }
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 0          # don't demand a spare GPU during rollout
      maxUnavailable: 1    # replace one pod at a time
  selector:
    matchLabels: { app: llm-inference }
  template:
    metadata:
      labels: { app: llm-inference }
    spec:
      terminationGracePeriodSeconds: 180   # let in-flight generations drain

      # --- placement: only land on the right GPU nodes ---
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule

      # --- weights: download once into a shared emptyDir cache ---
      volumes:
      - name: model-cache
        emptyDir:
          sizeLimit: 200Gi
      initContainers:
      - name: fetch-weights
        image: amazon/aws-cli:2.15.0
        command:
        - sh
        - -c
        - |
          if [ ! -f /models/.done ]; then
            aws s3 sync s3://my-models/llama-3.1-70b /models/llama-3.1-70b
            touch /models/.done
          fi
        volumeMounts:
        - { name: model-cache, mountPath: /models }

      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.6.3
        args:
        - --model=/models/llama-3.1-70b
        - --served-model-name=llama-3.1-70b
        - --port=8000
        ports:
        - containerPort: 8000
        volumeMounts:
        - { name: model-cache, mountPath: /models, readOnly: true }

        resources:
          limits:
            nvidia.com/gpu: 1          # one whole GPU, request==limit, integer
            memory: 96Gi               # host RAM to stage weights
          requests:
            cpu: "8"
            memory: 96Gi
            nvidia.com/gpu: 1

        # --- probes: startup guards the slow load, then liveness/readiness ---
        startupProbe:
          httpGet: { path: /health, port: 8000 }
          periodSeconds: 10
          failureThreshold: 60         # up to 600s to load 70B weights + warmup
        readinessProbe:
          httpGet: { path: /health, port: 8000 }
          periodSeconds: 10
          failureThreshold: 3          # pull from LB if it goes unhealthy
        livenessProbe:
          httpGet: { path: /health, port: 8000 }
          periodSeconds: 20
          failureThreshold: 3          # only restart a genuinely wedged process

        lifecycle:
          preStop:
            exec:
              # flip out of rotation, give the LB time to notice before drain
              command: ["sh", "-c", "sleep 15"]
---
apiVersion: v1
kind: Service
metadata:
  name: llm-inference
spec:
  selector: { app: llm-inference }
  ports:
  - name: http
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: llm-inference
spec:
  minAvailable: 2                      # keep >=2 replicas through drains/upgrades
  selector:
    matchLabels: { app: llm-inference }
```

Notes on the choices:

- `startupProbe` budget = `10s * 60 = 600s`. If your model loads in ~90s, this is generous headroom; shrink `failureThreshold` if you want faster crash detection, but never below your real worst-case load time.
- The `initContainer` idempotency (`.done` sentinel) means a restarted pod on a node whose `emptyDir` survived (it won't across reschedule) skips the re-download; for true cross-pod caching use a **read-only RWX PVC** or a node-local hostPath cache instead of `emptyDir`.
- `maxSurge: 0` trades a little capacity during rollout for *not needing an extra GPU*.
- The `preStop` sleep + 180s grace period gives streaming requests time to finish and the load balancer time to stop routing before the process exits.

> **Saying it out loud.** If I had to describe a correct GPU Deployment manifest out loud: one `nvidia.com/gpu` under limits, a toleration for the GPU node taint and a nodeSelector for the right SKU, a startup probe with a budget sized to worst-case cold load, a readiness probe that only passes once the model is actually loaded, a lenient liveness probe that only takes over afterward, an initContainer fetching weights into a shared cache volume, a PodDisruptionBudget with `minAvailable`, and a termination grace period well above your longest generation. Every single one of those exists because of a specific failure — probes crash-looping a healthy load, drains evicting all replicas, SIGKILL cutting a stream mid-token. It's not ceremony; it's a list of incidents somebody already had.

## Brief KServe example

The same intent, far less YAML, using KServe's `InferenceService`. KServe wires up autoscaling, routing, and the storage fetch for you.

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-31-70b
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 4
    model:
      modelFormat: { name: vLLM }
      storageUri: s3://my-models/llama-3.1-70b   # KServe fetches the weights
      resources:
        limits:
          nvidia.com/gpu: "1"
        requests:
          nvidia.com/gpu: "1"
          memory: 96Gi
```

KServe pulls the weights from `storageUri` (S3/GCS/PVC/HTTP), applies a `ServingRuntime` for the vLLM format, and manages the Deployment/Service/autoscaler behind the CRD. You still tune probes and node placement via the `ServingRuntime` or pod overrides.

> **Saying it out loud.** The KServe version of that same manifest is about fifteen lines: an `InferenceService` with min and max replicas, a model format of vLLM, a `storageUri` pointing at S3, and a GPU resource limit. KServe fetches the weights, applies a `ServingRuntime` for the format, and manages the Deployment, Service, and autoscaler behind the CRD for you. The honest tradeoff is that you've traded explicit control for less YAML — you still tune probes and node placement, just through the `ServingRuntime` or pod overrides rather than directly, and when something goes wrong you're now debugging through an abstraction layer. Which is fine, as long as you understand what the raw manifest would have looked like, because that's what the CRD is generating underneath.

---

## Build it in practice — extended: MIG, time-slicing & NetworkPolicy

The example above requests one whole GPU. In practice you'll often want a *mix*: a small model MIG-sliced for isolation, and a dev/low-QPS pool that's time-sliced for density. Both are configured at the GPU Operator / device-plugin layer, not in the pod spec — the pod spec just requests whatever resource name the plugin ends up advertising.

**1. Enable MIG via the GPU Operator's `ClusterPolicy`.** The operator's MIG manager reads a named profile from a `ConfigMap` and applies it to nodes carrying a matching label:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mig-parted-config
  namespace: gpu-operator
data:
  config.yaml: |
    version: v1
    mig-configs:
      all-1g.10gb:
        - devices: all
          mig-enabled: true
          mig-devices:
            "1g.10gb": 7   # slice each A100-80GB into 7 isolated instances
```

```
# Label the target node(s) to request that profile; the MIG manager
# cordons/drains GPU workloads on the node, repartitions, then relabels.
kubectl label node gpu-node-1 nvidia.com/mig.config=all-1g.10gb --overwrite
```

Once applied, the device plugin advertises `nvidia.com/mig-1g.10gb` on that node instead of (or alongside) `nvidia.com/gpu`, and a pod requests it exactly like any extended resource:

```yaml
resources:
  limits:
    nvidia.com/mig-1g.10gb: 1   # one hardware-isolated 10GB slice
```

**2. Enable time-slicing for a separate, lower-trust dev pool.** This is a plain `ConfigMap` the device plugin consumes, referenced from the node via a device-plugin config label — deliberately *not* the same nodes running MIG, since the two are different tradeoffs for different tenancy levels:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  a100-40gb: |-
    version: v1
    flags:
      migStrategy: none
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4     # advertise 4x the physical GPU count on this node
```

```
kubectl label node gpu-node-dev nvidia.com/device-plugin.config=a100-40gb --overwrite
```

Pods on `gpu-node-dev` still request `nvidia.com/gpu: 1` — the plugin is simply advertising 4 units per physical card, so 4 pods can land on one GPU with no memory isolation between them. This is the tradeoff called out earlier: fine for a bursty internal eval tool, wrong for a multi-tenant production pool.

**3. Lock down the network around the inference pods.** A `NetworkPolicy` limits which namespaces can reach the model server and what the pod itself can reach outbound — worth doing once you have more than one team on the cluster:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: llm-inference-netpol
  namespace: inference
spec:
  podSelector:
    matchLabels: { app: llm-inference }
  policyTypes: [Ingress, Egress]
  ingress:
  - from:
    - namespaceSelector:
        matchLabels: { kubernetes.io/metadata.name: ingress-system }   # gateway/ingress controller
    - namespaceSelector:
        matchLabels: { kubernetes.io/metadata.name: monitoring }       # Prometheus scraping /metrics
    ports:
    - { protocol: TCP, port: 8000 }
  egress:
  - to: []                            # DNS to any endpoint, restricted by port
    ports:
    - { protocol: UDP, port: 53 }
    - { protocol: TCP, port: 53 }
  - to:                                # object storage for weights, via cloud NAT
    - ipBlock: { cidr: 0.0.0.0/0 }
    ports:
    - { protocol: TCP, port: 443 }
```

In a real cluster you'd usually scope that last egress rule to your cloud provider's object-storage IP ranges rather than `0.0.0.0/0`; it's left broad here because most teams reach weights through a NAT gateway whose egress IP isn't something the pod's `NetworkPolicy` can pin down more tightly without also breaking other HTTPS egress (registry pulls, telemetry). The important part is the *shape*: ingress restricted to the gateway/ingress and monitoring namespaces only, egress restricted to DNS and HTTPS — nothing else in or out.

> **Saying it out loud.** The key thing to understand about MIG and time-slicing on Kubernetes is that neither is configured in the pod spec. They're set at the GPU Operator and device-plugin layer — a `ClusterPolicy` for MIG, a plain ConfigMap for time-slicing — and the pod just requests whatever resource name the plugin ends up advertising, like `nvidia.com/mig-1g.10gb` instead of `nvidia.com/gpu`. That separation matters because it means a MIG reconfiguration is a *node-level* operation that drains and repartitions the card, which is why nodes dip through a `Pending`-inducing window during a MIG change. Add a NetworkPolicy on top so the inference pods only accept traffic from the gateway namespace — on a shared GPU cluster the pods are multi-tenant neighbors, and default-allow networking is not what you want there.

---

## Debugging playbook: Pending and crash-looping GPU pods

Two symptoms cover most incidents. Work them like this:

**Pod stuck `Pending`.**

```
kubectl describe pod <pod> | sed -n '/Events/,$p'
```

Read the scheduler message:

- `0/12 nodes are available: 12 Insufficient nvidia.com/gpu` → no free GPUs. Is the cluster-autoscaler adding a GPU node? Is your node pool at max? Is another pod holding the GPU?
- `... node(s) had untolerated taint {nvidia.com/gpu: present}` → you're missing the **toleration**.
- `... didn't match Pod's node affinity/selector` → your `nodeSelector`/label is wrong (check exact label with `kubectl get nodes --show-labels`).
- `Insufficient cpu/memory` → GPU is free but the node can't fit your CPU/RAM request.

**Pod in `CrashLoopBackOff` during startup.**

```
kubectl logs <pod> -c vllm --previous     # logs from the killed attempt
kubectl get events --field-selector involvedObject.name=<pod>
```

- Repeated `Liveness probe failed` events at ~the same age → probe is killing the model mid-load; add/extend the **startup probe**.
- `OOMKilled` in the container's `lastState` → raise the **memory limit** (host RAM to stage weights).
- initContainer errors (S3 auth, disk full on `emptyDir` `sizeLimit`) → weights fetch is failing; the main container never starts.
- CUDA/driver errors in logs → driver/toolkit mismatch (a job for the **GPU Operator**), or the GPU was already claimed.

> **Saying it out loud.** Two symptoms cover almost every GPU pod incident, and both have a mechanical first move. If a pod is `Pending`, run `kubectl describe pod` and read the scheduler's own message — it names the exact constraint. `Insufficient nvidia.com/gpu` means no free GPUs, `untolerated taint` means you're missing a toleration, `didn't match node affinity` means your selector label is wrong, and `Insufficient cpu` means the GPU was free but the node couldn't fit your CPU or RAM request. If it's `CrashLoopBackOff` during startup, pull the logs from the *previous* attempt with `--previous` and check the events. Repeated liveness failures at suspiciously identical ages means a probe is killing the load; `OOMKilled` in `lastState` means host RAM, not VRAM. The scheduler is usually telling you the answer already.

### Walkthrough: a pod stuck `Pending` after a MIG config change

A concrete sequence, the way it actually unfolds on-call — this is the "20 free GPUs but nothing schedules" flavor, root-caused end to end:

1. **Notice.** A new replica from a rollout has sat in `Pending` for ten minutes.

   ```
   $ kubectl get pods -n inference
   NAME                             READY   STATUS    RESTARTS   AGE
   llm-inference-7d9c9b9f5c-4kxqp   0/1     Pending   0          10m
   ```

2. **Read the scheduler's reasoning** — always the first move, it usually names the exact constraint:

   ```
   $ kubectl describe pod llm-inference-7d9c9b9f5c-4kxqp -n inference | sed -n '/Events/,$p'
   Events:
     Type     Reason            Age    From               Message
     ----     ------            ----   ----               -------
     Warning  FailedScheduling  9m52s  default-scheduler   0/6 nodes are available: 4 Insufficient nvidia.com/gpu,
                                                             2 node(s) had untolerated taint {nvidia.com/gpu: present}.
   ```

3. **Cross-check node capacity vs. allocatable.** A MIG config change earlier that day relabeled two nodes, and the device plugin briefly advertised zero GPUs while it reconciled:

   ```
   $ kubectl get nodes -l nvidia.com/gpu.present=true \
       -o custom-columns=NAME:.metadata.name,ALLOC:.status.allocatable."nvidia\.com/gpu",CAP:.status.capacity."nvidia\.com/gpu"
   NAME        ALLOC   CAP
   gpu-node-1  0       8
   gpu-node-2  0       8
   gpu-node-3  8       8
   gpu-node-4  8       8
   ```

   `gpu-node-1`/`2` show `ALLOC=0` against `CAP=8` — the device plugin pod restarted mid-MIG-reconfiguration and hasn't re-registered the resource with the kubelet yet.

4. **Check the device plugin DaemonSet**, since it — not the kubelet — owns the extended resource:

   ```
   $ kubectl get pods -n gpu-operator -l app=nvidia-device-plugin-daemonset -o wide
   NAME                                READY   STATUS             RESTARTS   NODE
   nvidia-device-plugin-daemonset-a1   0/1     CrashLoopBackOff   6          gpu-node-1
   nvidia-device-plugin-daemonset-b2   0/1     CrashLoopBackOff   6          gpu-node-2
   nvidia-device-plugin-daemonset-c3   1/1     Running            0          gpu-node-3

   $ kubectl logs -n gpu-operator nvidia-device-plugin-daemonset-a1 --previous
   ... error: no MIG devices found matching profile "1g.10gb": mig-parted config "all-1g.10gb" not yet applied
   ```

5. **Root cause.** The `mig-parted-config` ConfigMap was updated to a new profile, the label `nvidia.com/mig.config` on `gpu-node-1`/`2` was flipped by the MIG manager, but the physical repartition (which requires the GPU to have no running processes) hadn't completed before the device plugin restarted and tried to enumerate MIG devices — a race between the MIG manager's drain/repartition step and the device plugin's own restart.

6. **Fix.** Let the MIG manager finish (it cordons and drains the node's GPU workloads before repartitioning — that's expected and is *why* nodes go through a `Pending`-inducing dip), or if it's stuck, restart the MIG manager pod on that node and re-check `kubectl get nodes ... ALLOC`. Once `ALLOC` matches `CAP` again the pending pod schedules within seconds — no change to the pod spec was ever needed.

**Lesson generalized:** `Insufficient nvidia.com/gpu` in the scheduler event is necessary but not sufficient diagnosis — always compare `allocatable` against `capacity` per node before assuming "the cluster is full." A gap between them almost always means the device plugin (or MIG manager, or driver container) is unhealthy on that specific node, not that GPUs are actually unavailable cluster-wide.

> **Saying it out loud.** Here's the shape of a real one. A pod sits `Pending` ten minutes; describe says `Insufficient nvidia.com/gpu` on four nodes. The move that cracks it is comparing each node's *allocatable* GPUs against its *capacity* — and two nodes show allocatable zero against capacity eight. That gap is never "the cluster is full," it's the device plugin being unhealthy on those specific nodes. Sure enough, the plugin pods are crash-looping because a MIG profile change flipped the node labels but the physical repartition hadn't finished, and the plugin restarted mid-reconfiguration trying to enumerate MIG devices that didn't exist yet. Nothing about the pod spec was ever wrong. The generalizable lesson: `Insufficient nvidia.com/gpu` is a necessary but not sufficient diagnosis — always check allocatable versus capacity per node before believing the cluster is out of GPUs.

### A note on cold-start math

Cold-start time for a scaled-up or rescheduled pod is roughly:

\[ T_{cold} = T_{provision} + T_{pull} + T_{weights} + T_{load} + T_{warmup} \]

where \( T_{provision} \) is node acquisition (0 if a node is warm, minutes if the cluster-autoscaler must boot a GPU VM), \( T_{pull} \) is image pull, \( T_{weights} \) is fetching weights to the node, \( T_{load} \) is loading them into VRAM, and \( T_{warmup} \) is CUDA-graph capture / first-token warmup. Your **startup probe** budget must exceed \( T_{weights} + T_{load} + T_{warmup} \) (the init container covers \( T_{weights} \) separately if you split it out), and your autoscaling responsiveness is gated by the whole sum — which is why node-local caches and pre-pulled images matter so much.

> **Saying it out loud.** Cold start for a GPU pod is a sum of five terms, and it's worth naming all of them: node provisioning if the autoscaler has to boot a VM, image pull, fetching the weights onto the node, loading them into VRAM, and warmup like CUDA graph capture. Two things fall out of that. Your startup probe budget has to exceed the load-plus-warmup portion, or you crash-loop a perfectly healthy pod. And your autoscaling responsiveness is gated by the entire sum — which is why a scale-up decision made in ten seconds can still take eight minutes to deliver capacity. That's the number that makes node-local weight caches and pre-pulled images worth the operational complexity: they're the only terms you can actually attack.

## Comparison: serving-on-Kubernetes options

| Option | What it is | Autoscaling / scale-to-zero | Best for | Cost |
|--------|-----------|------------------------------|----------|------|
| **Raw Deployment + Service** | You write the manifests | HPA only (you wire it); no scale-to-zero out of the box | Full control, simple single-model services, learning | High YAML/ops effort, most flexible |
| **KServe (InferenceService)** | CRD over Knative/k8s, standard model protocol | Yes, incl. scale-to-zero + canary | Platform teams wanting a model abstraction, many models, standardized rollout | Heavier install (Knative/Istio or raw-deploy mode), more concepts |
| **Ray Serve (KubeRay)** | Distributed serving on Ray via `RayService` | Yes, Ray-native autoscaling | Multi-model composition, distributed/multi-node tensor-parallel, complex pipelines | Ray cluster to operate; overkill for one small model |
| **Triton / NVIDIA NIM (+ NIM Operator)** | NVIDIA-optimized model servers/containers | Via KServe/HPA integration | NVIDIA-standardized stacks, optimized/quantized engines, vendor support | Vendor lock-in to NVIDIA images; excellent perf |
| **KubeAI** | Lightweight k8s-native LLM operator | Yes, incl. scale-from-zero | OpenAI-compatible serving without Istio/Knative | Younger ecosystem, smaller community |

Rule of thumb: start with a **raw Deployment** to understand the mechanics; graduate to **KServe** or **KubeAI** when you have many models and want autoscaling/canary for free; reach for **Ray Serve** when a single request must fan across multiple GPUs/nodes or you're composing models; adopt **NIM/Triton** when NVIDIA's optimized engines and support matter.

> **Saying it out loud.** My rule of thumb: start with a raw Deployment so you actually understand the mechanics — GPU requests, probe timing, PDBs, weight fetching. Graduate to KServe or KubeAI once you have many models and want autoscaling and canary rollouts for free rather than hand-wiring an HPA per service. Reach for Ray Serve when a single request has to fan across multiple GPUs or nodes, or when you're composing several models into a pipeline — that's genuinely a different problem shape. And adopt NIM or Triton when NVIDIA's optimized engines and vendor support matter more than portability. The tradeoff running through all of it is control versus ceremony: the raw path is the most flexible and the most YAML, and every abstraction above it is a layer you'll eventually have to debug through.

---

## Failure modes & pitfalls

- **Probes killing pods mid-load.** No startup probe (or a liveness probe with a too-short `initialDelaySeconds`) turns a 4-minute model load into an infinite `CrashLoopBackOff`. Always use a startup probe sized to worst-case load; keep liveness lenient. This is the number-one LLM-on-k8s bug.
- **GPU fragmentation.** Integer, per-node GPU allocation strands capacity: 20 GPUs free cluster-wide but no single node has the 4 your tensor-parallel pod needs. Use topology-aware / gang scheduling and design node pools around your parallelism.
- **Image pull of huge images.** A 60 GB image (weights baked in) can take many minutes to pull on a cold node, and the cluster-autoscaler's node-provision + pull time compounds it. Keep runtime images lean, pre-pull images to nodes, or use a node-local weight cache.
- **Thundering-herd weight downloads.** N pods cold-starting at once each pulling the full model from one bucket saturates egress and slows *all* of them. Pre-populate a read-only PVC or node cache; stagger scale-ups.
- **No PDB → full outage on drain.** A routine node upgrade or autoscaler scale-down can evict every replica simultaneously. Always ship a PodDisruptionBudget with `minAvailable`.
- **Missing tolerations / wrong selectors → Pending forever.** GPU nodes are tainted; a pod without the matching toleration silently stays `Pending`. Conversely, no selector and CPU pods squat on GPU nodes. `kubectl describe pod` shows the scheduling reason.
- **CPU/memory misconfig masquerading as GPU failure.** An OOMKill while staging weights into host RAM, or CPU throttling from a tight CPU limit, looks like a model/probe problem. Give generous memory limits; be careful with CPU limits.
- **Ingress/proxy timeouts cutting streams.** Default 30–60s proxy timeouts truncate long token streams. Raise read/backend timeouts and disable response buffering.
- **Rollouts assuming spare GPUs.** `maxSurge > 0` on a full GPU pool blocks the rollout waiting for GPUs that don't exist. Use `maxSurge: 0` or ensure headroom.
- **Ephemeral `emptyDir` cache re-downloads every reschedule.** An `emptyDir` dies with the pod, so a rescheduled pod re-fetches the whole model. If cold-start matters, back the cache with a read-only RWX PVC or a node-local `hostPath`/CSI volume that survives pod churn.
- **Graceful shutdown too short.** Default 30s `terminationGracePeriodSeconds` SIGKILLs pods mid-generation. Raise it above your longest generation and flip readiness first.
- **Time-slicing treated as a free lunch.** Advertising 4x replicas of a GPU does not give 4x compute or any memory isolation — two co-scheduled pods can OOM each other. Reach for MIG when tenants don't fully trust each other.
- **No quota between serving and batch/training on a shared cluster.** Without Kueue-style quotas, a large eval sweep or fine-tuning job can starve the serving fleet of GPUs with no admission control to stop it.

> **Saying it out loud.** The recurring failures on Kubernetes are pretty consistent. Probes killing pods mid-load — that's number one, and it's always a missing startup probe. GPU fragmentation, where twenty GPUs are free cluster-wide but no single node has the four your tensor-parallel pod needs. Huge image pulls compounding with node provisioning time. Thundering-herd weight downloads when N pods cold-start at once. No PDB, so a routine node drain takes the whole service down. Missing tolerations leaving pods `Pending` silently. An `emptyDir` cache that dies with the pod so every reschedule re-downloads the model. And a thirty-second grace period SIGKILL-ing pods mid-generation. The pattern: most of them look like something else — hardware failure, network outage, model bug — which is why the diagnosis discipline matters more than memorizing the list.

---

## Production case studies & war stories

### Case 1: the 3-minute model load that looked like a hardware failure

**Setup.** A team migrated a 34B model from a hand-run VM to a Kubernetes Deployment. They copied a probe config from an existing CPU microservice: `livenessProbe` with `initialDelaySeconds: 30`, `periodSeconds: 10`, `failureThreshold: 3` — no startup probe (this predates the startup-probe-first mindset now standard).

**Symptom.** Every rollout, every pod restart, every node replacement produced the same pattern: pod `Running`, then `CrashLoopBackOff` a few minutes later, forever. Logs showed the model server killed mid-`torch.load`. The on-call's first hypothesis was bad GPU hardware — they cordoned two "suspect" nodes before someone actually read the timeline.

**Root cause.** The model took roughly 3 minutes to load (weights from a network PVC plus CUDA graph capture). The liveness probe's math: `initialDelaySeconds(30) + periodSeconds(10) * failureThreshold(3) = 60s` grace window — 60 seconds against a 180-second load. The kubelet killed the container at roughly T+60s, every time, deterministically. It looked random only because different nodes had slightly different PVC read latency, shifting the exact restart timestamp.

**Fix.** Added a `startupProbe` with `periodSeconds: 10, failureThreshold: 30` (a 300s budget), and left liveness to only take over after startup succeeded — the pattern in the fully worked example above. Rollouts went from "always crash-loops for the first 10 minutes, eventually succeeds by luck when a fast node happens to finish in time" to clean on the first attempt.

**Lesson.** A crash loop with a *suspiciously consistent* time-to-first-crash is a probe timing bug, not hardware. Check the arithmetic (`initialDelaySeconds + periodSeconds * failureThreshold`) against your actual worst-case load time before blaming nodes. This is the single most common LLM-on-Kubernetes incident, and it is entirely self-inflicted — the fix is always a startup probe, never "replace the GPU."

> **Saying it out loud.** A team moved a 34B model onto Kubernetes and copied a probe config from an existing CPU microservice: liveness with thirty seconds initial delay, ten-second period, three failures. No startup probe. Every rollout produced the same thing — pod Running, then CrashLoopBackOff, forever — and on-call's first theory was bad GPU hardware, so they cordoned two nodes before anyone did the arithmetic. The math: 30 plus 10 times 3 is a sixty-second grace window against a 180-second load. The kubelet killed it at T+60 every single time, deterministically; it only *looked* random because PVC read latency varied slightly per node. The fix was a startup probe with a 300-second budget. The lesson worth stealing: a crash loop with a suspiciously *consistent* time-to-crash is a probe timing bug, never hardware.

### Case 2: 20 free GPUs, one pod stuck Pending — fragmentation across nodes

**Setup.** A cluster with five 8-GPU nodes ran a mix of single-GPU inference pods (many small models) alongside an occasional 4-GPU tensor-parallel deployment for a 70B model.

**Symptom.** The 4-GPU pod sat `Pending` for over an hour. `kubectl describe` reported plain `Insufficient nvidia.com/gpu` — no taint mismatch, no selector typo. Cluster-wide GPU utilization dashboards showed only 60% of GPUs in use — 20 of 40 GPUs "free."

**Root cause.** Single-GPU pods had been scheduled by bin-packing across all five nodes rather than filling nodes one at a time, leaving each node with 3–4 free GPUs but *no node* with 4 contiguous free GPUs together — because the default scheduler has no built-in notion of "keep this pod's GPUs on one node for NVLink locality" beyond the basic per-node integer count. The one 4-GPU tensor-parallel pod needed all 4 GPUs on a single node (for NVLink bandwidth between them) and there wasn't one.

**Fix, in order of effort:**

1. **Immediate:** manually cordon/drain single-GPU pods off one node to consolidate free capacity, unblocking the pending pod (a manual, one-time bin-pack).
2. **Short-term:** add a `PriorityClass` + preemption so multi-GPU pods can evict lower-priority single-GPU pods to consolidate space, and set `podAntiAffinity`/topology spread on the *small* pods to bias the scheduler toward filling nodes rather than spreading them evenly.
3. **Structural:** split the node pool — a pool of nodes reserved (via taint) for multi-GPU tensor-parallel workloads only, sized exactly to the parallelism degree, and a separate pool (optionally MIG- or time-sliced) for single-GPU/small-model traffic. This is the fix that actually scales: don't let the scheduler *discover* topology constraints at pending-time, encode them into pool shape up front.
4. **Longer-term:** adopt **Kueue** with gang scheduling for the multi-GPU workload class, so the 4-GPU pod's pods are admitted all-or-nothing against a quota that reserves capacity, and/or evaluate **DRA** once available on the platform, which can express "4 GPUs on the same NVLink domain" as a first-class claim instead of hoping bin-packing works out.

**Lesson.** "GPUs free cluster-wide" and "GPUs schedulable for this pod" are different numbers whenever a workload needs more than one GPU per node. Multi-GPU workloads need topology-aware placement designed in from the start (dedicated pools, gang scheduling, or DRA), not discovered as an incident.

> **Saying it out loud.** Five 8-GPU nodes, lots of single-GPU pods, and one 4-GPU tensor-parallel deployment that sat `Pending` for over an hour while dashboards cheerfully showed twenty of forty GPUs free. The cause is bin-packing: the scheduler had spread the small pods evenly across all five nodes, leaving three or four free GPUs on each — and no node with four free *together*, which is what the tensor-parallel pod needs for NVLink bandwidth between them. The immediate fix is manually consolidating; the structural fix is splitting the node pool so multi-GPU workloads get a tainted pool sized exactly to their parallelism degree. The lesson: "GPUs free cluster-wide" and "GPUs schedulable for this pod" are different numbers the moment a workload needs more than one GPU on one node, and you encode topology into pool shape up front instead of discovering it as an incident.

### Case 3: the thundering herd that looked like a network outage

**Setup.** An autoscaling event scaled a 70B-model service from 2 to 10 replicas in response to a traffic spike. All 8 new pods had cold `emptyDir` caches and started `initContainer`s pulling the same 140GB from one S3 bucket simultaneously.

**Symptom.** Network egress on the bucket's region saturated; all 8 pods' downloads slowed to a crawl, *and* the 2 already-healthy pods saw elevated latency as the shared NAT gateway's connection tracking maxed out. On-call initially chased it as a networking/NAT problem.

**Fix and lesson.** Pre-populate a read-only RWX PVC (or a node-local cache warmed ahead of the spike) so scale-up pods mount already-present weights instead of re-downloading; if that's not feasible, stagger initContainer starts (e.g. a prefetch step with concurrency limits, or node-level caching so only the *first* pod on a given node downloads). The autoscaling chapter covers pre-warming pools for exactly this reason — cold-start stampedes are an autoscaling problem wearing a networking costume.

> **Saying it out loud.** An autoscaler took a 70B service from two replicas to ten during a traffic spike. All eight new pods had cold `emptyDir` caches, so all eight initContainers started pulling the same 140 gigabytes from one S3 bucket at the same moment. Egress saturated, every download crawled, and the two already-healthy pods got slower too because the shared NAT gateway's connection tracking maxed out — so on-call chased it as a networking problem for a while. The fix is to make scale-up pods mount weights that are already there: a pre-populated read-only PVC, or a node-local cache so only the first pod on each node downloads. Failing that, stagger the initContainers with a concurrency limit. The framing I like: cold-start stampedes are an autoscaling problem wearing a networking costume.

---

## Interview mastery

### Explain GPU scheduling on Kubernetes in 60 seconds

> "Kubernetes has no native GPU concept — the NVIDIA device plugin, a DaemonSet on every GPU node, discovers GPUs and advertises them to the kubelet as the extended resource `nvidia.com/gpu`. Extended resources are integer and exclusive: request must equal limit, and the scheduler will never double-book one GPU across two pods, because VRAM has no swap and no safe overcommit. To keep GPU nodes for GPU workloads you taint the nodes and tolerate the taint on your pods, and to pick a specific SKU you add a `nodeSelector` on a GPU-product label. If you want to pack more than one workload onto a card you do it explicitly — MIG for hardware-isolated partitions, time-slicing or MPS for software-multiplexed sharing with weaker isolation — the device plugin just advertises different resource names or counts depending on which you pick. The scheduler math never changes: whatever unit you're advertising, it's still handed out as an exclusive integer per pod."

Practice saying that out loud in under a minute — it hits device plugin, extended resources, exclusivity/no-overcommit rationale, taints/tolerations, and the sharing mechanisms in the right order.

### System design prompt: "run 5 different model sizes on a shared GPU cluster"

**Prompt as typically asked:** *"You need to serve five models — say 1B, 8B, 34B, 70B, and 405B parameters — on one Kubernetes cluster, with mixed traffic (some interactive chat, some high-throughput batch). Sketch the architecture."*

A strong answer separates concerns by **parallelism degree** and **latency class**, not just "throw everything in one Deployment":

```
                        +-------------------------------+
                        |   Gateway API + Inference     |
                        |   Extension (InferencePool    |
                        |   per model, priority via      |
                        |   InferenceObjective)           |
                        +---------------+-----------------+
                                        | model-aware / prefix-cache-aware routing
        +---------------+--------------+--------------+---------------+
        v               v              v              v               v
  +-----------+   +-----------+  +-----------+  +-----------+  +----------------+
  | 1B pool   |   | 8B pool   |  | 34B pool  |  | 70B pool  |  | 405B pool      |
  | MIG-sliced|   | 1 GPU/pod |  | 1 GPU/pod |  | TP=4, one |  | TP=8/PP=2,     |
  | (7x/GPU)  |   | L4/L40S   |  | A100 80GB |  | node, A100|  | multi-node,    |
  | shared A100|  | node pool |  | node pool |  | node pool |  | dedicated pool |
  | pool      |   |           |  |           |  | (gang     |  | (gang          |
  |           |   |           |  |           |  | scheduled)|  | scheduled)     |
  +-----------+   +-----------+  +-----------+  +-----------+  +----------------+
        |               |              |              |               |
        +--- KEDA/HPA on vLLM queue-depth metrics, per pool, independently ---+
                                        |
                     Kueue ClusterQueues arbitrate GPU quota
                     between interactive (high priority, low
                     latency SLO) and batch (best-effort FIFO)
```

Key decisions to narrate:

1. **Pool by parallelism, not just by size.** The 1B and 8B models fit on fractional/single GPUs — MIG-slice the 1B model (it's small and latency-sensitive, so hardware isolation beats time-slicing) and give 8B a full GPU on a cheaper SKU. 70B and 405B need tensor/pipeline parallelism across multiple GPUs or nodes — give them dedicated, gang-scheduled pools sized exactly to their parallelism degree (this avoids the fragmentation war story above).
2. **Route by model identity and priority**, using the Gateway API Inference Extension's `InferencePool`/`InferenceObjective` (or a KV-cache-aware router like llm-d) — interactive chat traffic gets latency-priority routing and reserved capacity; batch/eval traffic runs best-effort and is the first to be preempted or queued.
3. **Autoscale on serving pressure, not CPU** — KEDA against each pool's vLLM `num_requests_running`/queue-depth metric, independently per model, so a spike in 8B traffic doesn't starve 405B capacity or vice versa.
4. **Arbitrate GPU quota with Kueue** if teams share the cluster for both serving and batch fine-tuning/eval — quotas prevent one workload class from starving another, and gang scheduling stops a partially-admitted multi-GPU job from deadlocking capacity.
5. **Bound blast radius per pool** — separate PDBs, separate node pools' taints, so a bad rollout or node drain on the 405B pool can't touch the 1B pool's availability.

Interviewers are usually grading whether you *separate concerns by parallelism/latency class* rather than proposing one Deployment-per-model with no shared reasoning about GPU topology — that's the signal that differentiates a strong answer.

> **Saying it out loud.** Serving 1B through 405B on one cluster with mixed interactive and batch traffic — the move is to separate concerns by *parallelism degree* and *latency class*, not to make one Deployment per model and hope. So: MIG-slice the 1B model since it's small and latency-sensitive and deserves hardware isolation; put 8B on a full but cheaper GPU like an L4; give 70B a gang-scheduled TP=4 pool sized exactly to its parallelism; give 405B a dedicated multi-node pool. Route by model identity and priority using `InferencePool` and `InferenceObjective` so interactive chat gets reserved capacity and batch runs best-effort. Autoscale each pool independently on vLLM queue depth, not CPU. And arbitrate quota with Kueue so an eval sweep can't starve the serving fleet. What's being graded is whether you separate by topology class at all.

### Red flags vs. green flags

| Signal | Red flag (weak answer) | Green flag (strong answer) |
|---|---|---|
| GPU overcommit | "You can set a GPU limit higher than request to burst" | "Extended resources require request == limit; GPUs are never overcommitted because VRAM has no swap" |
| Probe design | "Just increase `initialDelaySeconds` a lot" | "Use a startup probe sized to worst-case load; keep liveness fast and lenient once startup passes" |
| GPU sharing | Treats MIG/time-slicing/MPS as interchangeable | Distinguishes hardware isolation (MIG) from software multiplexing (time-slicing/MPS) and picks based on tenancy trust |
| Fragmentation | "Just add more nodes" | Explains topology-aware pooling, gang scheduling, or DRA as the structural fix |
| Rollouts | Doesn't mention `maxSurge` GPU cost | Explains `maxSurge: 0`/`maxUnavailable: 1` tradeoff for GPU-scarce rollouts |
| Disruption | Unaware of PDBs | Explains PDB + graceful shutdown + `terminationGracePeriodSeconds` together |
| Routing | Proposes plain round-robin for LLM traffic | Mentions prefix/KV-cache-aware routing (session affinity, llm-d, Gateway API Inference Extension) |
| Batch vs. serving | No answer for "how do you share GPUs between training and serving" | Mentions Kueue quotas/ClusterQueues and priority/preemption |
| Multi-region | Treats multi-region as "just another Ingress" | Explains config-cluster/target-cluster split and capacity bursting (or the general pattern even without naming a specific vendor) |
| Cold start | Doesn't account for weight fetch time in scaling decisions | Breaks cold start into provision + pull + weights + load + warmup and sizes probes/autoscaling around the sum |

### Q&A bank (18 questions)

1. **"A pod loads a 100 GB model in 5 minutes but keeps restarting. Why?"** — Missing/short **startup probe**; liveness kills it mid-load. Fix with a startup probe whose `periodSeconds * failureThreshold` exceeds worst-case load, and suppress liveness until then.
2. **"Why can't you overcommit GPUs like memory?"** — VRAM has no swap and the scheduler can't reason about it; the device plugin advertises integer, exclusive units (request == limit). Sharing requires explicit time-slicing/MPS/MIG.
3. **"How do you keep GPU pods on GPU nodes and everything else off?"** — Taint GPU nodes, add matching **tolerations**, plus a `nodeSelector`/affinity on the GPU SKU label. Explain the attraction-vs-repulsion split.
4. **"Where do the weights come from and how fast is a cold start?"** — Articulate baked-image vs initContainer vs read-only PVC vs node cache, the thundering-herd risk, and how you make replicas 2..N start fast.
5. **"How do you upgrade without an outage?"** — RollingUpdate with `maxSurge: 0`/`maxUnavailable: 1`, a **PDB** with `minAvailable`, graceful drain via `terminationGracePeriodSeconds` + `preStop` + readiness flip.
6. **"Readiness vs liveness vs startup — when does each fire and what does failure do?"** — Startup gates the others and restarts on failure; readiness gates *traffic* (no kill); liveness restarts a wedged process. Traffic should ride on a readiness endpoint that only passes after warmup.
7. **"When would you reach for KServe or Ray Serve over a raw Deployment?"** — KServe/KubeAI for many models + autoscaling/canary/scale-to-zero for free; Ray Serve for distributed multi-GPU/multi-node or model composition; raw Deployment for control/simplicity.
8. **"You have 20 free GPUs but a 4-GPU pod won't schedule — explain."** — **Fragmentation**: allocation is per-node and integer. Needs topology-aware/gang scheduling, dedicated pools sized to parallelism, or DRA.
9. **"What's the difference between MIG, time-slicing, and MPS, and when do you pick each?"** — MIG: hardware-isolated partitions, safe for untrusted multi-tenant workloads, fixed partition sizes. Time-slicing: no isolation, oversubscribed compute, fine for trusted/bursty dev traffic. MPS: shared memory but concurrent kernel execution with some compute control — a middle ground, still not hard-isolated.
10. **"How would you route requests so follow-up turns hit the replica with the warm KV cache?"** — Session affinity as a cheap first step; for real gains, a KV-cache-aware router (vLLM production stack router, llm-d, or the Gateway API Inference Extension's `InferencePool` with prefix-cache signals) rather than round-robin.
11. **"How do you share a GPU cluster fairly between serving and batch fine-tuning/eval jobs?"** — Kueue `ClusterQueue`/`LocalQueue` with quotas, borrowing/lending between cohorts, priority-based admission, and gang scheduling so partially-admitted multi-GPU jobs don't deadlock capacity.
12. **"What is Dynamic Resource Allocation and why does it matter for GPUs?"** — DRA (GA in Kubernetes 1.34, September 2025) replaces "advertise an integer count" with claim-based allocation, letting a pod request structured device properties (a MIG profile, GPUs on the same NVLink domain) that the old device-plugin model can't express.
13. **"How do you serve a model that doesn't fit on one node?"** — Tensor/pipeline parallelism across nodes (KServe's `tensorParallelSize`/`pipelineParallelSize`, or Ray Serve/KubeRay's distributed actors), a headless Service + StatefulSet for stable pod-to-pod addressing, and a dedicated, gang-scheduled node pool sized to the parallelism degree.
14. **"How do you avoid a regional outage taking down your inference service?"** — Multi-cluster/multi-region routing (e.g. the multi-cluster Inference Gateway pattern): a config cluster holds routing policy, multiple target clusters run pods across regions, giving automatic failover and cross-region capacity bursting.
15. **"Your rollout is stuck because `maxSurge` wants a GPU that doesn't exist. What do you do?"** — Set `maxSurge: 0, maxUnavailable: 1` to roll within existing capacity, or ensure headroom (spare GPU quota) before rolling; explain the tradeoff of reduced capacity during rollout vs. blocking entirely.
16. **"A 'ready' pod shows 0% GPU utilization but the request queue is growing. What's happening and how do you catch it?"** — The health endpoint is a shallow check that passed, but inference is wedged (deadlock, stuck CUDA context, dependency hang). Catch it with DCGM-exporter-based alerting on sustained 0% utilization on a ready pod, not just probe status — this is the "silent failure" that probes alone will never see.
17. **"How would you design probes and autoscaling differently for a model that takes 30 seconds to load vs. one that takes 10 minutes?"** — Both need a startup probe sized to worst case, but the 10-minute model changes the *autoscaling* answer too: scale-up latency of 10 minutes means you want to keep warm standby replicas or pre-provisioned capacity rather than relying on reactive HPA/KEDA scale-up alone.
18. **"What's the standard way to express LLM-aware traffic routing on Kubernetes as of 2026?"** — The Gateway API Inference Extension (`InferencePool` + `InferenceModel`/`InferenceObjective`), implemented by multiple gateway controllers and cloud-managed Inference Gateway offerings — model-aware, priority-aware routing standardized on top of the Gateway API instead of bespoke per-team logic.

---

## Appendix: DRA in concrete YAML, and a 2023-vs-2026 cheat sheet

Dynamic Resource Allocation is discussed above at the concept level; here is the shape of it in YAML so it's recognizable rather than abstract. This is illustrative of the pattern shipping across DRA-enabled clusters as of 2025–2026 (the exact field names can vary slightly by Kubernetes version and vendor driver, so treat this as "what to expect," not a copy-paste guarantee for your cluster's exact version):

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata:
  name: gpu-claim-template
spec:
  spec:
    devices:
      requests:
      - name: single-gpu
        exactly:
          deviceClassName: gpu.nvidia.com
          allocationMode: ExactCount
          count: 1
---
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
  - name: app
    image: vllm/vllm-openai:v0.6.3
    resources:
      claims:
      - name: single-gpu     # ties the container to the claim below
  resourceClaims:
  - name: single-gpu
    resourceClaimTemplateName: gpu-claim-template
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

Compare this to the device-plugin form used throughout this chapter (`resources.limits.nvidia.com/gpu: 1`). The device-plugin form asks for *a count*; the DRA form asks for *a claim against a device class*, and the claim's `spec.spec.devices.requests` block is where richer selection criteria (a specific MIG profile, GPUs sharing an NVLink domain, a minimum driver version) get expressed once your cluster's DRA driver supports them. Until then, the device-plugin model in the rest of this chapter remains the thing you'll actually operate day to day.

A quick before/after for orientation:

| Concern | Classic (device plugin, still dominant in 2026) | Emerging (DRA, GA since k8s 1.34) |
|---|---|---|
| How a pod asks for a GPU | `resources.limits: {nvidia.com/gpu: 1}` | `resourceClaims` referencing a `ResourceClaimTemplate`/`ResourceClaim` |
| What's expressible | An integer count of one resource name | Structured device selection (profile, topology, driver constraints) |
| Who advertises capacity | Device plugin DaemonSet per node | A DRA driver implementing the `resource.k8s.io` API |
| MIG / topology awareness | Encoded indirectly via distinct resource names (`nvidia.com/mig-1g.10gb`) | Expressed natively in the claim's device request |
| Maturity in production (2026) | Default, battle-tested | Early; vendor driver support still rolling out |

> **Saying it out loud.** The concrete difference between the two models is one sentence: the device-plugin form asks for *a count*, and the DRA form asks for *a claim against a device class*. In practice that means instead of `resources.limits: nvidia.com/gpu: 1`, you reference a `ResourceClaimTemplate` whose device request is where richer selection criteria live — a specific MIG profile, GPUs sharing an NVLink domain, a minimum driver version. That's the thing the integer model structurally cannot express, and it's why fragmentation and topology are so painful today. The honest state of play in 2026, though, is that the device plugin is still the default and battle-tested path, and DRA driver support is still rolling out — so know the shape of it, recognize it in YAML, and keep operating the classic model day to day.

### kubectl diagnostic cheat sheet

The commands used throughout the debugging playbook and war stories above, gathered in one place:

| Question | Command |
|---|---|
| Why is this pod `Pending`? | `kubectl describe pod <pod> \| sed -n '/Events/,$p'` |
| Does the node actually have free GPUs right now? | `kubectl get nodes -o custom-columns=NAME:.metadata.name,ALLOC:.status.allocatable."nvidia\.com/gpu",CAP:.status.capacity."nvidia\.com/gpu"` |
| Is the device plugin healthy on every GPU node? | `kubectl get pods -n gpu-operator -l app=nvidia-device-plugin-daemonset -o wide` |
| What did a crashed container log before it died? | `kubectl logs <pod> -c <container> --previous` |
| What exact labels does this node carry (for `nodeSelector` debugging)? | `kubectl get nodes --show-labels` |
| Is a probe repeatedly failing, and when? | `kubectl get events --field-selector involvedObject.name=<pod>` |
| Was the container OOMKilled? | `kubectl get pod <pod> -o jsonpath='{.status.containerStatuses[0].lastState.terminated.reason}'` |
| Is GPU utilization actually near zero on a "ready" pod? | DCGM exporter metric in Prometheus/Grafana, not kubectl — probes alone can't see this |
| Did a node drain evict more replicas than the PDB should have allowed? | `kubectl get pdb -n <namespace>` (check `ALLOWED DISRUPTIONS`) then `kubectl get events -A --field-selector reason=Killing` |

---

## Build it in practice — the combined manifest

Putting the MIG resource request, a `NetworkPolicy`, and the probe/PDB discipline from earlier together in one place, as you'd actually commit it for a small, latency-sensitive model running on a MIG-sliced, multi-tenant GPU pool:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: small-model-inference
  namespace: inference
  labels: { app: small-model-inference }
spec:
  replicas: 4
  strategy:
    rollingUpdate: { maxSurge: 0, maxUnavailable: 1 }
  selector:
    matchLabels: { app: small-model-inference }
  template:
    metadata:
      labels: { app: small-model-inference }
    spec:
      terminationGracePeriodSeconds: 60
      nodeSelector:
        nvidia.com/mig.config: all-1g.10gb        # only the MIG-partitioned pool
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.6.3
        args: [--model=/models/small-model, --port=8000]
        ports: [{ containerPort: 8000 }]
        resources:
          limits:
            nvidia.com/mig-1g.10gb: 1              # one hardware-isolated MIG slice
            memory: 16Gi
          requests:
            cpu: "2"
            memory: 16Gi
            nvidia.com/mig-1g.10gb: 1
        startupProbe:
          httpGet: { path: /health, port: 8000 }
          periodSeconds: 5
          failureThreshold: 24                     # 120s — small model, fast load
        readinessProbe:
          httpGet: { path: /health, port: 8000 }
          periodSeconds: 10
          failureThreshold: 3
        livenessProbe:
          httpGet: { path: /health, port: 8000 }
          periodSeconds: 20
          failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: small-model-inference
  namespace: inference
spec:
  selector: { app: small-model-inference }
  ports: [{ name: http, port: 80, targetPort: 8000 }]
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: small-model-inference
  namespace: inference
spec:
  minAvailable: 2
  selector:
    matchLabels: { app: small-model-inference }
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: small-model-inference-netpol
  namespace: inference
spec:
  podSelector:
    matchLabels: { app: small-model-inference }
  policyTypes: [Ingress, Egress]
  ingress:
  - from:
    - namespaceSelector:
        matchLabels: { kubernetes.io/metadata.name: ingress-system }
    - namespaceSelector:
        matchLabels: { kubernetes.io/metadata.name: monitoring }
    ports: [{ protocol: TCP, port: 8000 }]
  egress:
  - to: []
    ports: [{ protocol: UDP, port: 53 }, { protocol: TCP, port: 53 }]
```

Every piece here traces back to a mechanism explained earlier in the chapter: the MIG resource name (section 2b and the extended build example), `maxSurge: 0` for GPU-scarce rollouts (section 7), a startup probe sized in seconds appropriate to a *small* model's faster load (section 4 — contrast the 600s budget on the 70B example), a PDB (section 7), and the ingress/egress shape from the `NetworkPolicy` extended example above. Small models on shared MIG hardware still get the same disciplines as the 70B single-GPU deployment — just with numbers scaled down.

> **Saying it out loud.** The combined manifest is worth reading as a checklist rather than as YAML. It requests a MIG slice instead of a whole card, because this is a small latency-sensitive model on a multi-tenant pool where hardware isolation beats density. It has a startup probe sized to worst-case load, with liveness and readiness taking over only afterward. It has a PodDisruptionBudget so a node drain can't evict everything at once. It has a termination grace period longer than the longest generation, with readiness flipping first so traffic drains before shutdown begins. And it has a NetworkPolicy restricting ingress to the gateway namespace, because on a shared cluster your neighbors are other people's workloads. Every line traces to a specific failure mode from earlier in the chapter.

---

## Glossary — quick reference for interviews

| Term | One-line definition |
|---|---|
| **Device plugin** | DaemonSet that discovers GPUs and advertises them to the kubelet as an extended resource (`nvidia.com/gpu`) |
| **Extended resource** | A countable, non-CPU/memory resource type; must be integer, request must equal limit |
| **MIG** | Hardware partitioning of a GPU into isolated instances with separate memory/compute |
| **Time-slicing** | Software oversubscription of one GPU across N pods, no memory isolation |
| **MPS** | Multi-Process Service — concurrent kernel execution with shared memory, soft compute control |
| **DRA** | Dynamic Resource Allocation — claim-based device allocation (GA in k8s 1.34, Sept 2025), successor direction to the device-plugin model |
| **Startup probe** | Probe that gates liveness/readiness until it first succeeds; failure restarts the container |
| **Readiness probe** | Gates traffic (Service endpoints); failure does not kill the pod |
| **Liveness probe** | Detects a wedged process; failure restarts the container |
| **PDB** | PodDisruptionBudget — bounds voluntary disruption (drains, scale-down) via `minAvailable`/`maxUnavailable` |
| **GPU fragmentation** | Free GPUs exist cluster-wide but not co-located on one node for a multi-GPU pod |
| **Gang scheduling** | All-or-nothing pod admission so a multi-pod job never partially starts and deadlocks |
| **Kueue** | Kubernetes SIG project for quota-aware, gang-scheduled batch/GPU job admission across teams |
| **InferencePool / InferenceObjective** | Gateway API Inference Extension resources for model-aware, priority-aware LLM traffic routing |
| **llm-d** | CNCF sandbox (Mar 2026) distributed inference stack: disaggregated prefill/decode, KV-cache-aware routing |
| **Thundering herd (weights)** | Many pods cold-starting simultaneously saturate shared storage/network pulling the same weights |

---

## Bonus: Kueue quota for sharing GPUs between serving and batch

The landscape section above introduces Kueue conceptually; here is the minimal shape of the objects that implement "arbitrate GPU quota between interactive serving and best-effort batch" from the system-design sketch:

```yaml
apiVersion: kueue.x-k8s.io/v1beta2
kind: ResourceFlavor
metadata:
  name: gpu-a100
spec:
  nodeLabels:
    nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: serving-cluster-queue
spec:
  namespaceSelector: {}
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: gpu-a100
      resources:
      - { name: cpu, nominalQuota: 64 }
      - { name: memory, nominalQuota: 512Gi }
      - { name: "nvidia.com/gpu", nominalQuota: 16 }   # reserved baseline for serving
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: batch-eval-cluster-queue
spec:
  namespaceSelector: {}
  cohort: shared-gpu-cohort                            # can borrow idle quota from serving
  resourceGroups:
  - coveredResources: ["nvidia.com/gpu"]
    flavors:
    - name: gpu-a100
      resources:
      - { name: "nvidia.com/gpu", nominalQuota: 4, borrowingLimit: 12 }
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: LocalQueue
metadata:
  name: eval-jobs
  namespace: ml-eval
spec:
  clusterQueue: batch-eval-cluster-queue
```

The `serving-cluster-queue` guarantees the interactive fleet its 16-GPU baseline; the `batch-eval-cluster-queue` shares the same `cohort` and can *borrow* up to 12 more GPUs when serving isn't using its full quota, but never starves serving below its nominal reservation. This is the concrete mechanism behind bullet 4 of the system-design answer ("arbitrate GPU quota with Kueue") and the fix in war-story Case 2's "longer-term" remediation.

> **Saying it out loud.** The concrete shape of GPU quota arbitration is three objects. A `ResourceFlavor` that identifies the hardware class by node label. A `ClusterQueue` for serving with a nominal quota — say sixteen GPUs — that's a guaranteed baseline nobody can take. And a second `ClusterQueue` for batch and eval in the same cohort, with a small nominal quota but a borrowing limit, so it can expand into serving's idle capacity when serving isn't using it, and gets pushed back out when serving needs it. That's the whole idea: batch gets to use the expensive idle hardware without ever being able to starve the interactive fleet below its reservation. It's the mechanism behind "arbitrate quota with Kueue" and the long-term fix for the fragmentation war story.

---

## Further reading

- NVIDIA k8s device plugin (README, resource requests, time-slicing/MPS): <https://github.com/NVIDIA/k8s-device-plugin>
- NVIDIA GPU Operator (drivers, toolkit, NFD, MIG): <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html>
- NVIDIA GPU Operator — time-slicing GPUs in Kubernetes: <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html>
- Kubernetes — Configure Liveness, Readiness and Startup Probes: <https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/>
- Kubernetes — Probes concepts: <https://kubernetes.io/docs/concepts/workloads/pods/probes/>
- Kubernetes — Pod Disruption Budgets: <https://kubernetes.io/docs/concepts/workloads/pods/disruptions/>
- Kubernetes — Gateway API: <https://gateway-api.sigs.k8s.io/>
- Kubernetes — Schedule GPUs: <https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/>
- Kubernetes — Taints and Tolerations: <https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/>
- Kubernetes — Access DRA Device Metadata (task guide): <https://kubernetes.io/docs/tasks/configure-pod-container/assign-resources/access-dra-device-metadata/>
- Kubernetes blog — v1.34: Dynamic Resource Allocation graduates to GA (Sept 1, 2025): <https://kubernetes.io/blog/2025/09/01/kubernetes-v1-34-dra-updates/>
- Kubernetes blog — Introducing Gateway API Inference Extension (Jun 5, 2025): <https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/>
- Gateway API Inference Extension — docs: <https://gateway-api-inference-extension.sigs.k8s.io/>
- Gateway API Inference Extension — GitHub: <https://github.com/kubernetes-sigs/gateway-api-inference-extension>
- KServe documentation: <https://kserve.github.io/website/>
- KServe GitHub: <https://github.com/kserve/kserve>
- CNCF — Announcing KServe v0.15: Advancing Generative AI Model Serving (Jun 18, 2025): <https://www.cncf.io/blog/2025/06/18/announcing-kserve-v0-15-advancing-generative-ai-model-serving/>
- Red Hat Developer — How to set up KServe autoscaling for vLLM with KEDA (Sep 23, 2025): <https://developers.redhat.com/articles/2025/09/23/how-set-kserve-autoscaling-vllm-keda>
- Kueue — Kubernetes-native job queueing: <https://kueue.sigs.k8s.io/>
- Kueue — features overview: <https://kueue.sigs.k8s.io/docs/overview/>
- Kueue — ClusterQueue concept docs: <https://kueue.sigs.k8s.io/docs/concepts/cluster_queue/>
- GKE — Deploy a batch system using Kueue: <https://cloud.google.com/kubernetes-engine/docs/tutorials/kueue-intro>
- GKE — Allocate existing devices to workloads with DRA: <https://docs.cloud.google.com/kubernetes-engine/docs/how-to/deploy-dra-workloads>
- llm-d — distributed inference serving stack (CNCF sandbox, Mar 2026): <https://github.com/llm-d/llm-d>
- Google Cloud blog — Multi-cluster GKE Inference Gateway helps scale AI workloads (Mar 2026): <https://cloud.google.com/blog/products/containers-kubernetes/multi-cluster-gke-inference-gateway-helps-scale-ai-workloads>
- NVIDIA NIM — Kubernetes deployment & KServe: <https://docs.nvidia.com/nim/large-language-models/latest/deployment/kubernetes-deployment/kserve.html>
- NVIDIA NIM Operator: <https://docs.nvidia.com/nim-operator/latest/index.html>
- NVIDIA NIM Operator — Dynamic Resource Allocation support: <https://docs.nvidia.com/nim-operator/latest/dra.html>
- Ray Serve LLM on Kubernetes (KubeRay): <https://docs.ray.io/en/latest/cluster/kubernetes/examples/rayserve-llm-example.html>
- KubeAI (k8s-native LLM inference operator): <https://github.com/substratusai/kubeai>
- EKS — Manage NVIDIA GPU devices: <https://docs.aws.amazon.com/eks/latest/userguide/device-management-nvidia.html>

> **Next:** autoscaling these deployments — HPA on custom/GPU metrics, KEDA, queue-depth scaling, and scale-to-zero — is covered in *Autoscaling GPU Inference*.
