# Kubernetes for LLM Serving

> Deploying and operating GPU inference on Kubernetes — the control plane that most production LLM stacks run on.

## Why this matters

Once you have a working inference server (vLLM, TGI, Triton, SGLang), the next question is always the same: *how do I run twelve of these across a fleet of GPU nodes, upgrade them without dropping traffic, and not set money on fire?* Kubernetes is the de-facto answer. It gives you a declarative fleet, health-based traffic gating, rolling upgrades, and a scheduler that can place a pod on exactly the right GPU.

But LLM serving breaks several of Kubernetes' defaults. Models take **minutes** to load, so naive probes will kill pods mid-startup in an infinite crash loop. GPUs are **not overcommittable**, so the usual CPU/memory bin-packing intuition is wrong. Container images and model weights are **tens of gigabytes**, so image pulls and cold starts dominate. This chapter walks the mechanisms and the sharp edges.

Autoscaling (HPA, KEDA, custom metrics, scale-to-zero) is deep enough to deserve its own chapter — see *Autoscaling GPU Inference*. Here we reference it but focus on deployment, scheduling, health, weights, and lifecycle.

---

## Core intuition

Three mental models carry most of the chapter:

1. **A GPU is an indivisible, non-overcommittable device.** Unlike CPU (compressible) and memory (overcommittable at your peril), a GPU is handed to exactly one container by the device plugin. You can *share* it deliberately (time-slicing, MPS, MIG), but the scheduler still treats each advertised unit as an integer resource. There is no "burst above your GPU limit."

2. **Health is a function of time, not just liveness.** A pod that has been alive for 40 seconds but hasn't loaded a 140 GB model is *not broken* — it's *starting*. Kubernetes has a dedicated primitive for exactly this distinction: the **startup probe**. Get this wrong and you get a crash loop that looks like a hardware failure.

3. **The weights are the workload.** For classic web services the container image *is* the app. For LLM serving the image is a runtime and the weights are the payload — often 10x larger than the image. Where the weights live and how they land on the node (baked into the image, PVC, `initContainer`, object storage, model cache) determines your cold-start time and your blast radius.

4. **Voluntary disruption is the one you control.** Nodes get drained for upgrades, spot reclamation, and autoscaler scale-down constantly. Kubernetes lets you *bound* that damage with a PodDisruptionBudget and a graceful-shutdown path. Unlike a crashed process (involuntary), these are scheduled, negotiable evictions — and the difference between a clean rolling drain and a full outage is a few lines of YAML you either wrote or didn't.

Keep these four in mind and the rest of the chapter is mostly detail: GPUs are exclusive integers, health is time-aware, weights are the real payload, and disruptions are bounded on purpose.

---

## Mechanisms in depth

### 1. The building blocks: Deployment, Service, Ingress

For a stateless replicated inference server the standard trio is:

- **Deployment** — declares *N* replicas of a pod template, handles rolling updates and self-healing.
- **Service** — a stable virtual IP + DNS name that load-balances across the ready pods (`ClusterIP` for in-cluster, `LoadBalancer` for cloud L4).
- **Ingress** (or Gateway API) — L7 routing, TLS termination, path/host rules, into the Service.

A subtlety unique to LLM serving: **long request durations and streaming**. Token-streaming responses (SSE) can run for tens of seconds to minutes. Make sure your Ingress/proxy timeouts (`proxy-read-timeout` on the NGINX ingress, backend request timeout on cloud LBs) are raised, and that buffering is disabled so tokens flush as they generate. Default 30–60s timeouts will cut long generations.

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

### 2b. GPU sharing: MIG vs time-slicing vs MPS

One whole GPU per pod is wasteful for small models that use a few GB of a 80 GB card. Three mechanisms let you pack more, each changing what the device plugin advertises:

| Mechanism | Isolation | How it splits | Advertised as | Use when |
|-----------|-----------|---------------|---------------|----------|
| **MIG** (Multi-Instance GPU) | **Hardware** — separate memory + compute slices | Physically partitions an A100/H100 into up to 7 instances | `nvidia.com/mig-1g.10gb` etc. (or relabeled `nvidia.com/gpu`) | Strong isolation, predictable QoS, multi-tenant |
| **Time-slicing** | **None** — processes share memory, take turns on the SMs | Plugin advertises N "replicas" of each GPU; the driver context-switches | `nvidia.com/gpu` (inflated count) | Bursty/low-QPS dev workloads that tolerate contention |
| **MPS** (Multi-Process Service) | **Soft** — shared memory, concurrent kernels with optional compute caps | A daemon runs many clients' kernels concurrently | `nvidia.com/gpu` (configured slots) | Higher utilization than time-slicing, some control |

Critical caveat: **time-slicing gives no memory isolation** — two pods on one time-sliced GPU can OOM each other, and "requesting 2 shared GPUs" does **not** guarantee 2x the compute. For production multi-tenant serving, **MIG** is the safe choice; time-slicing/MPS are for dev or trusted, well-characterized co-tenancy. All three are configured via the device plugin / GPU Operator, not by the pod author.

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

### 5. Resource requests/limits and why GPUs aren't overcommitted

For CPU and memory you set requests (scheduling guarantee) and limits (cap). For LLM pods:

- **CPU:** set a request so the scheduler reserves headroom (tokenization, HTTP, scheduling loops are CPU-hungry), but be cautious with CPU *limits* — throttling the server's event loop can tank throughput. Many teams set a CPU request and no CPU limit.
- **Memory:** set request and limit close together and *generous*. Host RAM is used to stage weights before they hit VRAM; an OOMKill mid-load looks like a probe failure but is actually the kernel.
- **GPU:** `nvidia.com/gpu` request == limit == integer, always. There is no overcommit. The reason is physical: GPU memory (VRAM) has no swap and no soft limit the scheduler understands. If two pods both assumed they had the whole 80 GB, the second allocation would fail at CUDA-malloc time, not at schedule time — an ugly runtime crash instead of a clean `Pending`. So Kubernetes refuses to double-book.

Corollary — **GPU fragmentation**: because allocation is integer and per-node, a cluster with eight 8-GPU nodes and lots of single-GPU pods can end up unable to schedule a pod that needs 4 GPUs *on one node*, even though 20 GPUs are free cluster-wide. Multi-GPU / tensor-parallel pods need whole nodes or careful bin-packing (topology-aware scheduling, `podAffinity`, or a scheduler like the NVIDIA/Volcano gang scheduler).

### 6. Serving the model weights

Where do the weights come from at pod start? Four common patterns, with tradeoffs:

| Approach | How | Pros | Cons |
|----------|-----|------|------|
| **Baked into image** | `COPY` weights into the Docker image | Simplest; immutable; no runtime fetch | Enormous images (30–100 GB+), slow pulls, registry bloat, rebuild to change weights |
| **initContainer download** | An `initContainer` pulls weights from object storage (S3/GCS) into a shared `emptyDir` | Small runtime image; weights versioned in bucket | Re-downloads on every cold start unless cached; needs credentials |
| **PVC (shared/RWX)** | Weights on a persistent volume (e.g. a network filesystem), mounted read-only | Download once, many pods share; fast pod start | Storage class must support RWX or you pre-populate; network FS bandwidth can bottleneck concurrent loads |
| **Node-local cache / model cache** | Cache weights on local NVMe, or use a model-cache layer (e.g. Run:ai model streamer, KServe modelcar, Fluid) | Fast warm starts; streams weights into VRAM | More moving parts; cache warmup / eviction to manage |

Rules of thumb: bake weights into the image only for small models or when immutability matters more than pull time. For large models, keep the runtime image lean and fetch weights via `initContainer` or a pre-populated read-only PVC, and *cache on the node* so replicas 2..N start fast. Beware the **thundering herd**: ten pods cold-starting simultaneously all pulling 140 GB from the same bucket will saturate egress and each other.

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

### 8. Serving frameworks & operators

You don't have to hand-roll Deployments. Higher-level tools add model-aware features (autoscaling on GPU/queue metrics, scale-to-zero, canary, standardized model formats):

- **KServe** — a CRD (`InferenceService`) on top of Knative/Kubernetes. Handles autoscaling (incl. scale-to-zero), canary rollout, and a standard prediction protocol. Has first-class support for LLM runtimes (vLLM) via `ServingRuntime`. Good when you want a platform abstraction over raw pods.
- **NVIDIA NIM / Triton on k8s** — NIM packages optimized model microservices as containers; the **NIM Operator** (and Triton) deploy them, and NIM integrates with **KServe** for the serving layer. Best when you're standardized on NVIDIA's optimized stack and want vendor-supported images.
- **KubeAI** — an open, k8s-native inference operator focused on OpenAI-compatible serving of LLMs (vLLM/Ollama), with built-in autoscaling and model management, no Istio/Knative dependency. Lighter-weight alternative to KServe.
- **Ray Serve (KubeRay)** — deploy via the `RayService` CRD. Shines for **multi-model, model-composition, and distributed (multi-node tensor/pipeline-parallel)** serving where a request fans across many actors/GPUs. More of a distributed compute framework than a thin serving layer.

---

### 9. Networking specifics: Gateway API, timeouts, affinity

- **Ingress vs Gateway API.** The classic `Ingress` resource works, but the newer **Gateway API** (`Gateway` + `HTTPRoute`) is the direction the ecosystem is moving and expresses timeouts, traffic splitting, and header routing more cleanly — useful for canarying model versions.
- **Streaming timeouts.** Token-by-token SSE/HTTP responses can run minutes. On the NGINX ingress set `nginx.ingress.kubernetes.io/proxy-read-timeout` and `proxy-send-timeout` to several hundred seconds and disable buffering (`proxy-buffering: "off"`) so tokens flush live. Cloud L7 LBs have their own backend timeout you must raise.
- **Session affinity for KV-cache reuse.** With prefix/KV caching, routing a follow-up request to the *same* replica that holds the cache boosts throughput. Basic `sessionAffinity: ClientIP` on the Service helps; smarter setups use a cache-aware router (e.g. the vLLM production stack / router) instead of round-robin.
- **Headless Services for multi-node.** Distributed (tensor/pipeline-parallel across nodes) runtimes often need pod-to-pod addressing; a headless Service (`clusterIP: None`) plus a StatefulSet gives stable per-pod DNS.

### 10. Observability: know when a GPU pod is unhealthy

Standard pod metrics miss the GPU. Add:

- **DCGM exporter** (shipped by the GPU Operator) → Prometheus: GPU utilization, memory used, temperature, ECC errors, throttling. Alert on sustained 0% utilization on a "ready" pod (stuck), on VRAM near 100% (OOM risk), and on XID/ECC errors (failing hardware).
- **Server-level metrics** from the runtime: queue depth, time-to-first-token, tokens/sec, running vs waiting requests. These drive autoscaling (see the autoscaling chapter) and tell you *why* latency moved.
- **Event/probe signals:** watch for `Unhealthy` probe events, `CrashLoopBackOff`, and `FailedScheduling` (usually taint/selector or capacity). `kubectl describe pod` and `kubectl get events` are your first stop.

A "ready" pod pinned at 0% GPU utilization with a growing request queue is the classic silent failure — the health endpoint returns 200 but inference is wedged. Alert on the *metric*, not just the probe.

### 11. Spot / preemptible GPUs and cost

GPU nodes are the dominant cost, so many teams run inference on **spot/preemptible** instances at a large discount — accepting that the cloud can reclaim the node with ~30–120s notice.

- Spread replicas across on-demand and spot with `topologySpreadConstraints` so a spot reclamation storm can't take the whole service down; keep a baseline of on-demand capacity protected by the PDB.
- The preemption signal arrives as a node drain → your **graceful shutdown** path (SIGTERM, drain, `terminationGracePeriodSeconds`) must fit inside the cloud's notice window, or in-flight requests are lost.
- Cold-start time is your enemy here: a reclaimed spot pod must re-download weights and reload the model before serving. Node-local weight caches and pre-pulled images shrink the recovery gap.
- Right-size the GPU: an 8B model on an 80 GB H100 wastes the card — MIG-slice it or pick a smaller SKU (L4/L40S) and let the *comparison table* of frameworks + autoscaling do the packing.

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

### A note on cold-start math

Cold-start time for a scaled-up or rescheduled pod is roughly:

\[ T_{cold} = T_{provision} + T_{pull} + T_{weights} + T_{load} + T_{warmup} \]

where \( T_{provision} \) is node acquisition (0 if a node is warm, minutes if the cluster-autoscaler must boot a GPU VM), \( T_{pull} \) is image pull, \( T_{weights} \) is fetching weights to the node, \( T_{load} \) is loading them into VRAM, and \( T_{warmup} \) is CUDA-graph capture / first-token warmup. Your **startup probe** budget must exceed \( T_{weights} + T_{load} + T_{warmup} \) (the init container covers \( T_{weights} \) separately if you split it out), and your autoscaling responsiveness is gated by the whole sum — which is why node-local caches and pre-pulled images matter so much.

## Comparison: serving-on-Kubernetes options

| Option | What it is | Autoscaling / scale-to-zero | Best for | Cost |
|--------|-----------|------------------------------|----------|------|
| **Raw Deployment + Service** | You write the manifests | HPA only (you wire it); no scale-to-zero out of the box | Full control, simple single-model services, learning | High YAML/ops effort, most flexible |
| **KServe (InferenceService)** | CRD over Knative/k8s, standard model protocol | Yes, incl. scale-to-zero + canary | Platform teams wanting a model abstraction, many models, standardized rollout | Heavier install (Knative/Istio or raw-deploy mode), more concepts |
| **Ray Serve (KubeRay)** | Distributed serving on Ray via `RayService` | Yes, Ray-native autoscaling | Multi-model composition, distributed/multi-node tensor-parallel, complex pipelines | Ray cluster to operate; overkill for one small model |
| **Triton / NVIDIA NIM (+ NIM Operator)** | NVIDIA-optimized model servers/containers | Via KServe/HPA integration | NVIDIA-standardized stacks, optimized/quantized engines, vendor support | Vendor lock-in to NVIDIA images; excellent perf |
| **KubeAI** | Lightweight k8s-native LLM operator | Yes, incl. scale-from-zero | OpenAI-compatible serving without Istio/Knative | Younger ecosystem, smaller community |

Rule of thumb: start with a **raw Deployment** to understand the mechanics; graduate to **KServe** or **KubeAI** when you have many models and want autoscaling/canary for free; reach for **Ray Serve** when a single request must fan across multiple GPUs/nodes or you're composing models; adopt **NIM/Triton** when NVIDIA's optimized engines and support matter.

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

---

## Production checklist — what an interviewer probes

1. **"A pod loads a 100 GB model in 5 minutes but keeps restarting. Why?"** — Missing/short **startup probe**; liveness kills it mid-load. Fix with a startup probe whose `periodSeconds * failureThreshold` exceeds worst-case load, and suppress liveness until then.
2. **"Why can't you overcommit GPUs like memory?"** — VRAM has no swap and the scheduler can't reason about it; the device plugin advertises integer, exclusive units (request == limit). Sharing requires explicit time-slicing/MPS/MIG.
3. **"How do you keep GPU pods on GPU nodes and everything else off?"** — Taint GPU nodes, add matching **tolerations**, plus a `nodeSelector`/affinity on the GPU SKU label. Explain the attraction-vs-repulsion split.
4. **"Where do the weights come from and how fast is a cold start?"** — Articulate baked-image vs initContainer vs read-only PVC vs node cache, the thundering-herd risk, and how you make replicas 2..N start fast.
5. **"How do you upgrade without an outage?"** — RollingUpdate with `maxSurge: 0`/`maxUnavailable: 1`, a **PDB** with `minAvailable`, graceful drain via `terminationGracePeriodSeconds` + `preStop` + readiness flip.
6. **"Readiness vs liveness vs startup — when does each fire and what does failure do?"** — Startup gates the others and restarts on failure; readiness gates *traffic* (no kill); liveness restarts a wedged process. Traffic should ride on a readiness endpoint that only passes after warmup.
7. **"When would you reach for KServe or Ray Serve over a raw Deployment?"** — KServe/KubeAI for many models + autoscaling/canary/scale-to-zero for free; Ray Serve for distributed multi-GPU/multi-node or model composition; raw Deployment for control/simplicity.
8. **"You have 20 free GPUs but a 4-GPU pod won't schedule — explain."** — **Fragmentation**: allocation is per-node and integer. Needs topology-aware/gang scheduling or node pools sized to your parallelism.

---

## Further reading

- NVIDIA k8s device plugin (README, resource requests, time-slicing/MPS): <https://github.com/NVIDIA/k8s-device-plugin>
- NVIDIA GPU Operator (drivers, toolkit, NFD, MIG): <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html>
- Kubernetes — Configure Liveness, Readiness and Startup Probes: <https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/>
- Kubernetes — Probes concepts: <https://kubernetes.io/docs/concepts/workloads/pods/probes/>
- Kubernetes — Pod Disruption Budgets: <https://kubernetes.io/docs/concepts/workloads/pods/disruptions/>
- Kubernetes — Gateway API: <https://gateway-api.sigs.k8s.io/>
- Kubernetes — Schedule GPUs: <https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/>
- Kubernetes — Taints and Tolerations: <https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/>
- KServe documentation: <https://kserve.github.io/website/>
- KServe GitHub: <https://github.com/kserve/kserve>
- NVIDIA NIM — Kubernetes deployment & KServe: <https://docs.nvidia.com/nim/large-language-models/latest/deployment/kubernetes-deployment/kserve.html>
- NVIDIA NIM Operator: <https://docs.nvidia.com/nim-operator/latest/index.html>
- Ray Serve LLM on Kubernetes (KubeRay): <https://docs.ray.io/en/latest/cluster/kubernetes/examples/rayserve-llm-example.html>
- KubeAI (k8s-native LLM inference operator): <https://github.com/substratusai/kubeai>
- EKS — Manage NVIDIA GPU devices: <https://docs.aws.amazon.com/eks/latest/userguide/device-management-nvidia.html>

> **Next:** autoscaling these deployments — HPA on custom/GPU metrics, KEDA, queue-depth scaling, and scale-to-zero — is covered in *Autoscaling GPU Inference*.
