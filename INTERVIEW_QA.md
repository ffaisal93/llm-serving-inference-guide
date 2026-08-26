# LLM Serving & Inference Interview Q&A

Interview questions and answers for LLM serving, inference, and MLOps roles — written to help someone convince a **senior interviewer** they can build and operate real LLM inference infrastructure, not just call an API.

Each section maps to one of this guide's 11 hands-on chapters. Use the chapter deep-dives for implementation detail, the System Design section to rehearse whiteboard scenarios, and the Flashcards/Traps sections the night before an interview.

## Table of Contents

1. [LLM Inference Fundamentals](#llm-inference-fundamentals)
2. [Model Serving & Basic Serving Patterns (Ch. 01)](#model-serving--basic-serving-patterns-ch-01)
3. [Docker & Containerization (Ch. 02)](#docker--containerization-ch-02)
4. [Performance Optimization](#performance-optimization)
5. [Kubernetes & Deployment (Ch. 03)](#kubernetes--deployment-ch-03)
6. [Load Testing & Capacity Planning (Ch. 04)](#load-testing--capacity-planning-ch-04)
7. [vLLM Internals Deep Dive (Ch. 05)](#vllm-internals-deep-dive-ch-05)
8. [Autoscaling Deep Dive (Ch. 06)](#autoscaling-deep-dive-ch-06)
9. [Canary Deployments Deep Dive (Ch. 07)](#canary-deployments-deep-dive-ch-07)
10. [Monitoring & Observability (Ch. 08)](#monitoring--observability-ch-08)
11. [Model Versioning & Registry (Ch. 09)](#model-versioning--registry-ch-09)
12. [Drift Detection (Ch. 10)](#drift-detection-ch-10)
13. [Triton Inference Server (Ch. 11)](#triton-inference-server-ch-11)
14. [Production Best Practices](#production-best-practices)
15. [Additional Quick Questions](#additional-quick-questions)
16. [System Design Scenarios](#system-design-scenarios)
17. [2025-2026 Landscape Quiz](#2025-2026-landscape-quiz)
18. [Rapid-Fire Flashcards & Glossary](#rapid-fire-flashcards--glossary)
19. [Traps & How to Recover](#traps--how-to-recover)
20. [Tips for Interviews](#tips-for-interviews)
21. [Resources](#resources)

---

## LLM Inference Fundamentals

### Q1: Explain how LLM inference works step-by-step.

Tokenize to token IDs, embed to dense vectors, pass through the transformer layers (self-attention plus feed-forward per layer), project the final hidden states to vocabulary-size logits, sample the next token (greedy, top-k, top-p, temperature), append it, and repeat until EOS, max length, or a stop string. Each token depends on all previous ones; the KV cache stores past key/value tensors so they are not recomputed.

The split that matters: **prefill** processes the whole prompt at once and is compute-bound; **decode** produces one token per step and is memory-bandwidth-bound. Continuous batching, chunked prefill, and disaggregated prefill/decode all exist because of that split.

---

### Q2: What is KV caching and why is it important?

Prefill computes full Q, K, V for every prompt token. Decode computes Q only for the new token and reuses cached K and V. That turns an \( O(n^2) \) per-step cost into \( O(n) \): generating 100 tokens without a cache means 100 passes each reprocessing up to 100 tokens; with it, each pass processes 1 new token against the cached prefix.

KV cache size per token ≈ `2 (K and V) × num_layers × num_kv_heads × head_dim × dtype_bytes`. For a 70B-class dense model with grouped-query attention that is still megabytes per token at long context, which is why the KV cache — not the weights — is usually the binding memory constraint at high concurrency.

---

### Q3: What is the difference between training and inference?

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Mode** | Training mode (gradients computed) | Evaluation mode (no gradients) |
| **Batch** | Large, fixed batches (32-128+) | Small/variable batches, single requests |
| **Memory** | Stores activations for backprop + optimizer state | Only forward-pass activations + KV cache |
| **Speed** | Slower per step (backprop overhead) | Faster per step (forward only) |
| **Optimization target** | Loss minimization via gradient descent | Latency / throughput / cost per token |
| **Hardware** | Multi-GPU clusters, high-bandwidth interconnect | One GPU to multi-node, often shared/multi-tenant |

Training needs gradients plus optimizer state — 2-3x model memory in FP32 Adam state alone. Inference has fixed weights and is dominated by KV-cache growth and request-arrival variability instead, so there is no fixed batch shape to plan against.

---

### Q4: Explain the attention mechanism in the context of inference.

Q is what the current token is looking for, K describes what each past token offers, V is the information retrieved.

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) x V
```

Naive attention is \( O(n^2) \) in sequence length \( n \) for both compute and the attention matrix's memory, which is why long context is expensive. Production stacks never materialize the full \( n \times n \) matrix (FlashAttention-style fused kernels) and do not waste memory on it (PagedAttention-style block KV cache).

The variant ladder is MHA → MQA (one KV head) → GQA (a few KV heads shared across query heads; Llama 2/3, Mistral, most current models) → MLA (DeepSeek-V2/V3, compresses KV into a low-rank latent). GQA and MLA exist to shrink the KV cache — the real inference bottleneck — not to save training compute.

---

## Model Serving & Basic Serving Patterns (Ch. 01)

*Maps to Chapter 01 — Basic Serving (`app.py`, `model_loader.py`, `test_api.py`).*

### Q5: How would you design an LLM serving API?

```python
POST /v1/completions
{
  "prompt": "The future of AI is",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9
}
```

Validate the request (prompt length, parameter ranges) before touching the GPU, tokenize, generate, return text plus metadata — latency, token counts, finish reason. The model loads once at process startup, never per request.

Around that: async handling so concurrency does not block the event loop; streaming via SSE or chunked responses; typed errors (400 bad input, 429 overload, 503 draining); rate limiting that rejects early rather than queueing forever; and liveness separated from readiness so Kubernetes can tell "process up" from "model loaded".

---

### Q5b: Why must the model be loaded once at startup instead of per-request, and what does "model_loader.py as a singleton" actually buy you?

Loading a multi-GB to multi-tens-of-GB checkpoint takes seconds to minutes; per-request loading puts that on every request. A singleton loader — module-level global, or a class instantiated once in a startup hook — loads weights exactly once per process and shares them across every request that worker handles.

It also gives fail-fast startup: the pod crashes before it is marked ready instead of failing the first user request, which is why readiness must depend on "model loaded", not "HTTP server up". The trap is a model load left inside a request handler "for testing" with no guard, silently reloading every request.

---

### Q5c: Sync (Flask/gunicorn workers) vs async (FastAPI/uvicorn) for a basic LLM serving app — which do you pick and why?

Async, FastAPI on uvicorn. Most of a request's wall-clock time is awaiting the GPU through the engine's async generate call, so one process holds hundreds of in-flight requests without a thread each. It is what vLLM's OpenAI-compatible server and TGI actually do.

Sync loses twice. The GIL means CPU-bound work — tokenization, sampling, JSON parsing, validation — does not parallelize across threads in one process; only the forward pass releases it inside C/CUDA calls. And each gunicorn worker needs its own model copy on the GPU, which usually does not fit. The standard pattern is one async process per GPU (or per tensor-parallel GPU set), scaled by Kubernetes replicas rather than gunicorn workers.

---

### Q5d: How do you support streaming responses, and why does it matter for LLM UX?

Server-Sent Events (`text/event-stream`) or chunked transfer encoding: the engine yields tokens as generated and the HTTP layer flushes each chunk immediately instead of buffering.

Streaming makes TTFT the metric users feel. The user starts reading after prefill instead of after prefill plus full decode; on a 500-token answer that is the difference between "instant" and multi-second perceived latency.

The trap is buffering anywhere in the path — disable proxy buffering, watch gzip interacting badly with streaming, and set client timeouts long enough for slow generations.

---

### Q5e: Design the health check strategy for a basic LLM serving pod.

**Liveness** answers "is the process alive and not deadlocked" — a cheap `/healthz` that never touches the model; repeated failure restarts the container. **Readiness** answers "can this pod serve right now" — model loaded, engine not in an unrecoverable state; failure removes the pod from Service endpoints but does *not* restart it, which matters because restarting a temporarily overloaded pod is the wrong reaction. **Startup** covers the minutes a large model takes to load; a generous `failureThreshold`/`periodSeconds` stops liveness killing the pod mid-load, the most common cause of crash-looping large-model pods.

The anti-pattern is readiness that depends on GPU utilization or queue depth. That turns transient load into eviction, the remaining pods absorb the traffic, also fail readiness, and you get a thundering-herd death spiral.

---

### Q5f: How do you validate and sanitize LLM request inputs safely?

Enforce types and bounds with Pydantic: max prompt length, a `max_tokens` ceiling, valid ranges for temperature and top_p. The ceiling is the important one — without it a client can request 1,000,000 tokens and pin a GPU for minutes. Many production gateways cap `max_tokens` per plan or tier, because a huge generation inside a large batch multiplies memory pressure.

Reject before tokenization and generation; validation should be nearly free, and the point is to fail cheap requests cheaply rather than paying GPU cost to reject them. If user input is composed into a larger system prompt, escape or clearly delimit it rather than relying on the model to police itself.

---

### Q5g: What does "graceful shutdown" mean for an LLM serving pod, and why do naive implementations lose requests?

On `SIGTERM` — sent before a pod is killed during a rollout, scale-down, or preemption — the process stops accepting new requests, finishes or cleanly returns in-flight generations, then exits within `terminationGracePeriodSeconds`. Naive implementations die immediately, silently dropping mid-generation requests; with streaming that surfaces as a truncated answer.

Because a full generation takes a while, `terminationGracePeriodSeconds` usually needs to be far above the Kubernetes default of 30s — tens of seconds to a couple of minutes depending on `max_tokens` and concurrency. Pair it with readiness: flip readiness false the instant SIGTERM arrives, removing the pod from the Service, while draining existing connections.

---

### Q5h: Your `/v1/completions` endpoint works for one user in a demo but falls over with 50 concurrent users on one GPU. Diagnose it.

Diagnose by GPU utilization under load first. Low utilization with bad latency is a batching and scheduling problem, not a hardware problem — reaching for more GPUs before checking that is the wrong instinct.

The likely cause is a naive one-request-at-a-time generation loop, raw HuggingFace `.generate()` called synchronously per request, with no batching. At 50 concurrent users each waits a full serial turn, so latency scales linearly with load. The fix is a real inference engine — vLLM, TGI, or Triton — doing continuous batching. Secondary causes: no queueing or backpressure, so requests pile up in-process rather than being admitted or rejected predictably; no `max_tokens` cap, so a few huge requests starve everyone; and single-process serving with no path to a second replica.

---

## Docker & Containerization (Ch. 02)

*Maps to Chapter 02 — Docker (`Dockerfile.basic`, `Dockerfile.gpu`, `Dockerfile.optimized`, `docker-compose.yml`).*

### D1: Why does an LLM serving Dockerfile need to look different from a typical Python web app Dockerfile?

The base image must carry CUDA/cuDNN userspace libraries matching the host driver (`nvidia/cuda:...-runtime`, or `vllm/vllm-openai`), not `python:3.x-slim`, and the runtime must inject the device — `--gpus all` in Docker, the NVIDIA Container Toolkit and `nvidia.com/gpu` device plugin in Kubernetes. Without that the CUDA libraries are visible but the device is not.

Size is the other difference: CUDA plus PyTorch plus serving dependencies is easily 5-15GB, which drives pod start and node scale-up time during autoscaling bursts. Weights, often tens of GB, are not baked in — they are pulled at startup into a mounted volume, so the image stays small and a model swap needs no rebuild.

---

### D2: Explain the difference between `Dockerfile.basic`, `Dockerfile.gpu`, and `Dockerfile.optimized` patterns.

**Basic** is CPU-only on a plain Python base: fine for testing API logic without a GPU, useless for anything latency-sensitive. **GPU** uses a CUDA base with GPU-enabled wheels (CUDA-built `torch`, `vllm`) and pinned driver-compatible versions — the one that runs at production speed.

**Optimized** is the GPU image plus image engineering: multi-stage build so build dependencies never ship, layer ordering putting rarely-changing layers (CUDA base, system deps) before frequently-changing ones (app code) for cache hits, a `.dockerignore` so local caches and checkpoints are not shipped accidentally, and a runtime-only CUDA base instead of the full devel image.

---

### D3: Walk through multi-stage builds for an LLM serving image and why they matter here specifically.

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder
RUN pip install --user vllm torch
# compile any custom kernels here

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
COPY --from=builder /root/.local /root/.local
COPY app.py .
CMD ["python", "app.py"]
```

The `devel` image carries nvcc and headers and is only needed to build custom CUDA kernels or compile from source. It is roughly 2-3x larger than `runtime`. Shipping it costs registry storage, pull time, and attack surface for no benefit once the artifacts exist, so you compile with the heavy toolchain and ship only the runtime image with the artifacts copied across.

---

### D4: How do you handle GPU access and driver compatibility in Docker?

Install the NVIDIA Container Toolkit on the host so the runtime can expose GPU devices, then run with `docker run --gpus all` or `runtime: nvidia` in Compose.

Version compatibility is the number one real-world failure mode. The container's CUDA version must be compatible with the host's NVIDIA driver: drivers are backward-compatible with older CUDA runtimes but not forward-compatible with newer ones, so a container built against CUDA 12.4 fails on a host with only a CUDA-11-era driver. In Kubernetes the GPU Operator and device plugin abstract this and advertise `nvidia.com/gpu` as schedulable — but the compatibility problem does not disappear, it becomes "which node pool runs which driver version".

---

### D5: How should you manage model weights in a containerized serving setup — bake into the image or mount at runtime?

Baking in is simplest and fully immutable — the image *is* the version — but images become huge, every model update means a rebuild, push, and pull, and registry cost balloons. Mounting at runtime keeps the image small and generic: an init container or startup script pulls weights from S3/GCS or a registry into a volume or a node-local cache.

The common production answer is runtime mount with a warm node-local cache, because it decouples code version from model version. A model swap becomes a config change — an env var or ConfigMap pointing at a model URI — instead of a CI/CD rebuild, and that is what makes canary and rollback of *model* versions independent of *code* versions. Watch the cold-start penalty: pulling a 140GB checkpoint on every new pod is a major contributor to autoscaling lag.

---

### D6: What goes wrong if you don't pin exact versions of CUDA, PyTorch, and the inference engine in your image?

Silent ABI mismatches: a PyTorch wheel built against CUDA 12.1 loaded against a CUDA 12.4 runtime can work, half-work with a wrong kernel and a slow fallback path, or crash with `CUDA error: no kernel image is available for execution on the device`.

You also get drift. An unpinned `pip install vllm` today versus three weeks from now can pull a minor version with different default flags, different memory behavior, or a different OpenAI-API surface — nothing changes in the Dockerfile diff and production behavior changes anyway. Pin exact versions (`torch==2.4.0+cu124`, `vllm==0.6.3`), use a lockfile, and rebuild-and-test in CI before promoting a tag.

---

### D7: How would you use `docker-compose.yml` in local development for an LLM serving stack, and where does it stop being appropriate?

Good for the inner loop: serving container plus Prometheus, Grafana, and a mock registry in one command, GPU passthrough via `runtime: nvidia` or `deploy.resources.reservations.devices`, and a shared volume for weights so you do not re-download per `up`.

It stops at multi-node scheduling, autoscaling, rolling updates, secrets management at scale, and multi-tenant resource isolation — exactly the gap Kubernetes fills. Compose is single-host by design; production LLM serving is not.

---

### D8: How do you reduce image size and cold-start time for a GPU serving image without breaking reproducibility?

Multi-stage build to drop compiler toolchains, and prefer the framework's official runtime image (`vllm/vllm-openai:<pinned-tag>`) over assembling CUDA plus PyTorch plus engine yourself — it is already layer-optimized and tested upstream.

Order layers least-to-most frequently changing (base → system deps → Python deps → app code) so CI caches hit on app-code-only rebuilds. Keep weights out of the image and use a node-local cache. If cold node scale-up is on the autoscaling critical path, pre-pull images with a DaemonSet warm-up or node image pre-baking.

---

### D9: A container passes `nvidia-smi` inside `docker exec` but the serving process still reports "no CUDA devices". What do you check?

`nvidia-smi` working only confirms the toolkit and driver are visible to *that* process — not that the serving process, possibly a different user, cgroup, or one started before device injection, can use the device.

Check the container was started with the GPU flag for *this* invocation, easy to miss when you `docker exec` into a container originally run without `--gpus`. Check `CUDA_VISIBLE_DEVICES` is not empty or wrong. Then check the wheel: `torch.cuda.is_available()` returning False while `nvidia-smi` works almost always means a CPU-only wheel, because pip resolved plain `torch` instead of a CUDA-tagged build. In Kubernetes, confirm the pod sets `resources.limits."nvidia.com/gpu"` — GPUs are invisible without a limit — and that the device plugin DaemonSet is healthy on that node.

---

## Performance Optimization

### Q8: How would you optimize LLM inference latency?

Model level: quantization (FP16, FP8, INT8, INT4), pruning, distillation to a smaller student. Inference level: KV caching, batching, continuous batching (vLLM/TGI iteration-level scheduling), and speculative decoding, where a small draft model proposes several tokens and the target verifies them in one batched pass.

Hardware: a GPU rather than a CPU (10-100x for this workload), bf16/fp8 tensor-core kernel paths, and tensor/model parallelism for models that do not fit or need more aggregate bandwidth. System: pre-warm with a dummy forward pass before taking traffic, pool connections, cache responses or prefixes where semantically valid. Architecture: async handling, bounded request queues for bursts, and prefix- or KV-cache-aware load balancing rather than round-robin.

---

### Q9: What is PagedAttention and why does it matter?

Traditional KV caches pre-allocate a contiguous buffer sized for the *maximum* sequence length per request. When real generations are much shorter, that is severe internal fragmentation, and it makes long sequences at high concurrency impractical.

PagedAttention divides the cache into fixed-size blocks, like OS memory pages: allocated on demand, freed when sequences complete, reused for new sequences, and shared across sequences with an identical prefix — which is what prefix caching builds on. The result is near-zero fragmentation (only the last partial block is wasted), predictable support up to the model's max context, and more concurrent sequences in the same GPU memory — which is exactly what continuous batching needs in order to have work to schedule.

*(For deeper vLLM-specific internals — chunked prefill, scheduler design, disaggregated prefill/decode, speculative decoding, tensor/pipeline parallelism — see [vLLM Internals Deep Dive](#vllm-internals-deep-dive-ch-05).)*

---

### Q10: Explain the trade-offs between latency and throughput.

Latency is time for one request; throughput is requests or tokens per second across the system. Batching is the main lever: larger batches raise throughput and per-request latency through contention at each decode step, smaller batches do the reverse. Larger models raise quality and latency, lower precision is faster and smaller with accuracy risk, more GPUs buy throughput with cost.

For low latency: small batches, an optimized model, fast hardware, priority for interactive streaming. For high throughput: large batches, continuous batching, more GPUs, priority for offline work.

Do not optimize either in the abstract. Optimize against the actual SLO — for example P95 TTFT under 300ms and P95 inter-token latency under 50ms — and measure throughput *at that SLO*. Huge throughput bought by letting P99 blow up has moved the problem, not solved it.

---

## Kubernetes & Deployment (Ch. 03)

*Maps to Chapter 03 — Kubernetes (`deployment.yaml`, `deploy.sh`).*

### Q11: How would you deploy an LLM model to Kubernetes?

**1. Containerize** (see Ch. 02 above):
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

**2. Create a Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-serving
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: llm-serving
        image: llm-serving:v1.0
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
```

**3. Create a Service:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-serving
spec:
  selector:
    app: llm-serving
  ports:
  - port: 80
    targetPort: 8000
```

**4. Deploy:**
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

The parts needing judgment: GPUs are only requested as whole-unit `limits`, never fractional, without a sharing mechanism (K3); liveness, readiness, and startup probes need splitting (Q5e); the default rolling update strategy fits single-GPU-per-pod workloads badly (K2); configuration goes in ConfigMaps, API keys and tokens in Secrets.

---

### Q12: How does Horizontal Pod Autoscaling (HPA) work?

HPA checks metrics on an interval (15 seconds by default), compares the current value to the target, computes desired replicas, and updates the Deployment; Kubernetes creates or destroys pods to match.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

Built-in CPU and memory resource metrics are a poor signal for GPU-bound serving. The useful signals — requests per second, queue depth, latency — are custom metrics via the Prometheus Adapter or KEDA. A stabilization window prevents flapping.

*(For why CPU/memory-based HPA is usually the wrong tool for LLM serving, and what to use instead, see [Autoscaling Deep Dive](#autoscaling-deep-dive-ch-06).)*

---

### Q13: Explain canary deployments for model updates.

Deploy the new version alongside stable, split traffic (say 90/10), compare metrics between the two cohorts, step up if healthy (25%, 50%, 100%), and route everything back to stable if not.

Three implementations, coarse to fine: two Deployments behind one Service weighted by replica-count ratio; a service mesh or Gateway API splitting by percentage independent of replica counts; or application-level routing on headers or user-ID hashing for consistent bucketing.

Compare latency percentiles (P50/P95/P99) *between cohorts*, not in aggregate, plus error rates and quality metrics such as thumbs-down or hallucination-flag rate. The payoff is bounded blast radius, instant rollback by changing weights, and a real A/B comparison on live traffic.

*(For the full automated promotion/rollback pipeline, see [Canary Deployments Deep Dive](#canary-deployments-deep-dive-ch-07).)*

---

### K1: Why is `nvidia.com/gpu` treated so differently from CPU/memory by the Kubernetes scheduler?

GPUs are an **extended resource**: integer-only requests, no overcommit, and without MIG, time-slicing, or MPS configured, `requests` must equal `limits`. There is no fractional or burstable GPU by default, so the scheduler sees a node's GPUs as a small pool of indivisible slots — a pod asking for one either finds a node with a free GPU or does not schedule at all.

The consequence is that GPU bin-packing matters far more than CPU bin-packing. A cluster with plenty of *total* free GPU capacity spread as fractions across many nodes can still fail to schedule a pod because no single node has a whole GPU free, which makes node pool sizing and per-pod GPU granularity (1 vs 2 vs 8) a real design decision.

---

### K2: The default Kubernetes `RollingUpdate` strategy is a poor fit for a Deployment where each pod holds a 70B model on 4 GPUs. Why, and what would you change?

Default `RollingUpdate` with `maxSurge: 25%` brings up extra pods before removing old ones. For a pod needing 4 GPUs and minutes to load a huge checkpoint, that means spare GPUs sitting idle purely to support the rollout — usually unavailable in a GPU-constrained cluster.

Three fixes by situation: with no spare capacity and tolerance for a brief dip, set `maxSurge: 0, maxUnavailable: 1` to replace in place one at a time; for genuinely zero-downtime rollouts, explicitly over-provision by one replica's worth of GPUs; better still, move to a canary rollout (Ch. 07) so the old fleet stays fully up while a small new fleet is validated. Also tune `minReadySeconds` and the readiness probe so a pod is not counted available until the model has loaded and passed a warm-up check — otherwise the rollout outruns readiness and you get a real availability dip.

---

### K3: How do you share a single GPU across multiple pods in Kubernetes, and what are the trade-offs between the approaches?

| Approach | Isolation | Granularity | Notes |
|---|---|---|---|
| **Time-slicing** | None (software round-robin) | Coarse (N pods share one GPU's full memory + compute pool) | Simplest; risk of one pod OOM-ing or starving others; no memory isolation. |
| **MPS (Multi-Process Service)** | Weak (shared address space, but concurrent kernel execution) | Better compute overlap than time-slicing | Still no hard memory isolation; a misbehaving process can affect others. |
| **MIG (Multi-Instance GPU)** | Hardware-level (separate compute/memory partitions) | Fixed set of profiles (e.g., 7 slices on an A100/H100-class MIG-capable GPU) | True isolation, but partitions are fixed-size and set at node-configuration time, not per-pod-request time. |
| **DRA (Dynamic Resource Allocation)** | Depends on underlying tech (MIG, time-slice, or whole-device) | Flexible, expressed via `DeviceClass`/`ResourceClaim` with CEL-based selection | The modern (GA in Kubernetes v1.34) API replacing device-plugin extended resources for complex GPU allocation logic — verify exact version behavior before quoting in an interview, this area is moving fast. |
| **MPS/MIG combined with DRA** | Hardware isolation + flexible scheduling | Best of both, most complex to operate | Where the ecosystem (NVIDIA GPU Operator, KAI Scheduler) is heading as of 2026. |

Pick the isolation level from the blast radius you can tolerate: MIG or DRA-managed hard partitions for a platform shared across teams, time-slicing as a cost optimization for single-tenant batch or dev work.

---

### K4: Should an LLM serving Deployment use a `Deployment` or a `StatefulSet`?

A **Deployment** for most serving pods: they are stateless and interchangeable, with no per-pod identity or storage beyond shared model weights, and the Service load-balances across them.

A **StatefulSet** earns its place when pods need stable network identity or per-pod storage — a multi-node tensor-parallel or pipeline-parallel deployment where rank-0 needs a predictable address for the other ranks (common in multi-node vLLM/Triton or Ray-based clusters), or where each pod owns a distinct local NVMe cache worth preserving across restarts. The trap is reaching for it by default "because it's a model": the question is identity and storage, not whether the workload feels heavyweight.

---

### K5: Design node affinity / taints-and-tolerations for a mixed CPU+GPU cluster running LLM serving alongside other workloads.

Taint the GPU nodes (`nvidia.com/gpu=true:NoSchedule`) so only pods that tolerate the taint and request a GPU land there, keeping random CPU workloads off expensive hardware. Use `nodeAffinity`/`nodeSelector` to require the matching GPU node pool and, where it matters, a specific GPU generation — a pod assuming 80GB of HBM must not land on a 40GB node.

Add PriorityClasses so latency-sensitive serving outranks batch and offline jobs, letting the scheduler preempt lower-priority work under GPU pressure rather than leaving serving pods `Pending`. Use topology spread constraints across zones and nodes so one failure cannot take out every replica of a critical model.

---

### K6: How do you handle persistent storage for multi-tens-of-GB model weights across pod restarts and node scale-up events?

A **ReadOnlyMany PVC** shared across pods of the same model version, backed by a network filesystem or cloud RWX equivalent, stops every pod re-downloading the same weights. **Node-local caching** — hostPath or a DaemonSet-managed NVMe cache — trades simplicity for speed: the first pod on a node pays the download, later pods and restarts on that node are fast, but cold nodes from autoscaling still pay full cost, which lands on scale-up latency.

**Init containers** are the standard mechanism: download and verify the artifact before the main container starts, so readiness only checks "engine loaded". Watch for the race where multiple pods write the same node-local cache path during a scale-up burst — use a lock file or content-addressed paths keyed by model hash.

---

### K7: What's the "cold start" problem for GPU pods in Kubernetes, and how much of it is actually Kubernetes' fault?

Almost none of it — Kubernetes scheduling overhead is seconds. The time goes to provisioning a new GPU node if none is free (1-10+ minutes depending on cloud and instance type), pulling a multi-GB image, downloading multi-tens-of-GB weights, and engine warm-up: CUDA graph capture, kernel autotuning, vLLM's startup profiling pass.

So solving cold start means attacking image size (Ch. 02), weight caching (K6), and node pre-provisioning or warm pools, not tuning scheduler settings. It is also why naive reactive autoscaling, which scales up only after load has arrived, works poorly here.

---

### K8: How do you expose an LLM serving Service outside the cluster safely, and where does authentication/rate-limiting belong?

External load balancer or Ingress (or Gateway API) → API gateway for auth, rate limiting, request validation, and routing to version-specific backends → Kubernetes Service → serving pods.

Authentication and rate limiting belong at the gateway, not in the pod: the pod should be a dumb, fast, trusted-input component, and auth in every pod duplicates logic and makes the hot path slower and harder to change. TLS terminates at the Ingress or gateway too, so GPU-node CPU is not burned on crypto. Route model tiers and versions at the gateway by path or header rather than baking routing logic into clients.

---

## Load Testing & Capacity Planning (Ch. 04)

*Maps to Chapter 04 — Load Testing (`locust_test.py`, `measure_latency.py`).*

### LT1: What's wrong with load-testing an LLM API using the same request-per-second methodology you'd use for a REST CRUD API?

CRUD requests are roughly uniform in cost; LLM requests vary enormously with prompt length and `max_tokens`, so RPS alone is a poor load axis — two runs at identical RPS with different token distributions produce completely different GPU load. LLM serving also has two cost phases, prefill and decode, so a representative test must vary both input-length and output-length distributions, not just arrival rate.

Because of continuous batching, the right concurrency metric is usually the number of concurrent in-flight sequences. Throughput is a function of how many sequences the engine juggles, capped by KV-cache memory — not of arrival rate.

---

### LT2: Design a load test plan for a new vLLM deployment before it goes to production.

Define the traffic profile first — realistic prompt-length and output-length distributions from production logs if they exist, since chat, summarization, and RAG look nothing alike. Sweep concurrency rather than RPS, recording tokens/sec throughput and TTFT, inter-token latency, and end-to-end latency at each level.

Find the knee: throughput rises roughly linearly with concurrency until GPU or KV-cache saturation, then latency blows up while throughput plateaus or falls. That knee is the practical per-replica ceiling. Report max sustainable throughput *while P95 TTFT and P95 inter-token latency stay within SLO*, not raw peak.

Add a soak test at 70-80% of found capacity for hours, to catch memory leaks, KV-cache fragmentation, and slow degradation a burst test hides. Then test the failure path at 100%+ of capacity: do requests queue and eventually succeed, or does the engine error and OOM? That decides how you configure backpressure.

---

### LT3: What metrics do you pull out of a load test, beyond "requests per second"?

**TTFT**, dominated by prefill plus queueing, the metric users feel first. **Inter-token latency / TPOT**, dominated by decode step time and batch contention, which makes streaming feel fast or slow. **End-to-end latency**, TTFT + (TPOT × output tokens), what a non-streaming client sees.

**Throughput in tokens/sec** is the real capacity number. **GPU and KV-cache utilization** during the run tell you whether you are compute-bound, memory-bound, or scheduler-bound. **Error and timeout rate** as load rises shows where the system sheds, and whether it sheds gracefully with 429s or badly with timeouts and crashes.

---

### LT4: How do you use Locust (or similar) to generate realistic LLM traffic, and what's the catch with naive `locust_test.py`-style scripts?

Naive scripts send a fixed prompt repeatedly at a fixed rate. That misrepresents prefix-cache behavior — an identical prompt inflates hit rate, fully random prompts miss it entirely — and constant output length hides decode-bound bottlenecks. Better: sample prompts from a representative corpus with varying length, sample `max_tokens` from a realistic distribution, and ramp Locust's user count rather than a flat rate.

The catch is that Locust's own workers become the bottleneck before the server does at high concurrency, because of the Python GIL and single-machine limits. Distribute the workers, or use a purpose-built tool such as `vllm bench serve` or `genai-perf`, which model TTFT and TPOT correctly.

---

### LT5: `measure_latency.py`-style scripts show great P50 latency but the on-call gets paged for P99 timeouts in production. What's going on?

P50 tracks the common case; P99 is queueing effects, allocator or GC pauses, occasional very long prompts, cold caches, and batches temporarily saturated by a few outlier long generations.

A test at low or medium concurrency never enters the queueing regime that causes P99 blowups during production bursts, so it structurally cannot show you the tail. You have to test at and above expected *peak* concurrency. Also check whether the test hit a warm, pre-scaled fleet while production sometimes hits cold or just-scaled-up pods (K7) — that gap alone explains a lot of "P50 fine, P99 pages us".

---

### LT6: How does batch size / concurrency setting interact with load test results, and what would you tune based on what you see?

Throughput well below what GPU compute allows, with low GPU utilization, means the engine's max-concurrent-sequences or max-batched-tokens setting is too conservative — raise it, bounded by KV-cache memory.

Latency degrading sharply while throughput barely improves means you hit the KV-cache memory ceiling: the engine is queueing and evicting rather than adding sequences. The fix is more memory — a bigger GPU, a higher `gpu_memory_utilization` fraction, or a quantized KV cache — or accepting a lower concurrency ceiling. Chunked prefill matters here too: a long prompt hogging a full prefill step spikes inter-token latency for other in-flight decodes, and only a mix of very long and very short prompts reveals it.

---

### LT7: How would you capacity-plan "how many GPUs do I need to serve X req/s at Y SLO" from load test data?

From the load test, find the max concurrency or tokens/sec one replica sustains while meeting the latency SLO — call it `C_max`. Model expected traffic as *peak* concurrent requests, not average, using your actual length distributions: provision for peak, autoscale for the rest.

Required replicas ≈ `peak_concurrency / C_max`, plus headroom for rolling updates (K2), node or zone failure tolerance (N+1 or N+2), and burst above forecast. Multiply by GPUs-per-replica — the tensor-parallel degree — for total GPU count, then sanity-check against GPU memory needed for weights plus KV cache at that concurrency. Always re-validate on the target hardware, model, and quantization: theoretical FLOPs-based estimates are frequently off by 2-3x from measured reality, because decode is memory-bandwidth-bound.

---

### LT8: What's the difference between load testing "throughput mode" and "latency mode," and why would you run both?

**Throughput mode** fires as many concurrent requests as the client can generate, unconstrained by arrival timing, to find maximum sustainable tokens/sec — what you want for batch and offline workloads, and for locating the ceiling and the knee.

**Latency mode** sends requests at a fixed, realistic (roughly Poisson) arrival rate matching expected production traffic and measures the latency distribution there, validating the SLO under realistic rather than maximal load.

They answer different questions. A system can have an excellent throughput ceiling and still miss its latency SLO at normal traffic if it is misconfigured — for example batch size tuned for throughput rather than latency.

---

### LT9: How do you make load tests reproducible and comparable across engine versions or config changes?

Pin the exact prompt and output-length dataset with seeded sampling so two runs compare the same workload rather than different random draws, and hold hardware, instance type, driver and CUDA version, checkpoint, and quantization constant.

Report full latency distributions — P50/P90/P95/P99 — not averages, because averages hide tail regressions. Track results over time in a dashboard, or a CSV committed alongside config changes, so a vLLM version bump or scheduler-flag change gets an objective before-and-after instead of "it feels faster".

---

## vLLM Internals Deep Dive (Ch. 05)

*Maps to Chapter 05 — vLLM Serving (`vllm_server.py`). The two questions below are preserved from the original Model Serving section because they're really vLLM-specific.*

### Q6: What are the differences between HuggingFace Transformers and vLLM?

| Feature | HuggingFace `generate()` | vLLM |
|---------|-------------|------|
| **Batching** | Static batching | Continuous (iteration-level) batching |
| **Throughput** | Low (single-digit to low tens of req/s in naive setups) | Much higher — order(s) of magnitude, workload-dependent |
| **Memory** | Standard, often over-allocated KV cache | PagedAttention (efficient, near-zero fragmentation) |
| **GPU utilization** | Often 20-40% | Often 80-95% under load |
| **Ease of use** | Very easy, huge model/ecosystem coverage | Moderate — needs engine-specific configuration |
| **Flexibility** | High (arbitrary custom generation logic) | Moderate (optimized for the common serving path) |

Use `transformers` for development, research, one-off inference, and maximum flexibility — custom generation logic, unusual architectures. Use vLLM or a comparable engine for production, high-concurrency serving, and cost-sensitive GPU utilization.

*(Treat exact throughput multipliers as workload-dependent — always validate with your own load test per Ch. 04 rather than quoting a fixed number.)*

---

### Q7: How does continuous batching work in vLLM?

Static batching waits for a batch to fill, processes it, waits for all members to complete, then starts the next — so the batch runs at the speed of its slowest member and the GPU idles in the gap.

Continuous batching schedules at the decode-iteration level: new requests join as soon as there is KV-cache room, completed requests are removed immediately, and the rest continue without waiting for the cohort. The GPU never bubbles waiting for a batch to fill or drain.

```
Time 0: [Req1, Req2, Req3] -> Processing
Time 1: [Req1, Req2, Req3, Req4] -> Req4 added mid-flight
Time 2: [Req2, Req3, Req4] -> Req1 completed, removed
```

---

### V1: What is chunked prefill and why did vLLM add it?

Without it, a long prompt's prefill runs as one large step that can monopolize a full scheduler iteration, delaying the decode steps of every other in-flight request sharing it — one long prefill produces a visible latency spike for unrelated users.

Chunked prefill splits that prefill into smaller chunks processed across multiple iterations, interleaved with other requests' decode steps. The net effect is smoother, more predictable inter-token latency under mixed short/long-prompt traffic, at a small cost in total prefill throughput — a direct trade of throughput for tail-latency stability.

---

### V2: Explain vLLM's scheduler at a level that would satisfy a senior interviewer.

Each iteration the scheduler decides which requests run a step, subject to a KV-cache-block budget and a max-batched-tokens budget. It prioritizes continuing already-running decode sequences and admits new prefill requests from a waiting queue as capacity allows; with chunked prefill it can admit *partial* prefill work rather than making an all-or-nothing decision.

If KV-cache blocks run out it can **preempt** a running sequence — evicting it by swapping the cache out or recomputing it later — to make room. Which sequence is evicted, and the eviction policy, directly set fairness and tail latency. This scheduler is the actual engine behind continuous batching: it is a scheduling problem, not a batching trick.

---

### V3: What is speculative decoding and when does it actually help?

A small, cheap draft model proposes several candidate next tokens; the large target model verifies all of them in a single batched forward pass — cheaper than generating them one at a time — and accepts the longest correct prefix.

It helps most when decode is memory-bandwidth-bound, the usual case, and the draft's guesses are frequently right: low-temperature or near-deterministic generation, code completion, or drafts distilled to match the target's distribution. It helps less, or hurts, at high temperature and high output diversity, or at high concurrency where you are already GPU-compute-saturated — it trades extra compute per accepted token for fewer serial steps, and that trade is bad when compute is already the bottleneck.

Variants worth naming: draft-model speculative decoding, Medusa-style extra prediction heads on the target model itself, and n-gram or prompt-lookup decoding with no separate model, which works well where output repeats input literally, like code editing.

---

### V4: How does vLLM support tensor parallelism and pipeline parallelism, and when do you pick which?

**Tensor parallelism** shards each layer's weight matrices across GPUs — attention heads and MLP columns split across N GPUs — and needs an all-reduce or all-gather per layer, so it demands high-bandwidth interconnect and is usually kept within a node. **Pipeline parallelism** splits layers across GPUs or nodes sequentially, communicating only at stage boundaries, so it tolerates slower interconnect at the cost of pipeline bubbles unless you microbatch.

Rule of thumb: TP up to the GPU count in one NVLink domain, typically 8 GPUs on one node, to fit or speed up a model that does not fit on fewer; PP, or TP+PP together, when you must span nodes. Data parallelism — replicate the model, different requests per replica — is orthogonal and is what Kubernetes replicas give you. TP and PP set how big *one* replica is; replica count sets how many you run.

---

### V5: What quantization formats does a modern serving engine like vLLM support, and how do you choose one?

**FP16/BF16** is the baseline: minimal accuracy loss versus FP32, half the memory. **FP8** is native on Hopper and Blackwell-class tensor cores, roughly halves memory again, and can meaningfully speed up compute-bound prefill with small workload-dependent accuracy impact — increasingly the default fast path on new hardware as of 2025-2026 (verify the current support matrix before quoting). **INT8, AWQ, GPTQ, INT4** give aggressive memory reduction when weights rather than KV cache are the constraint, with larger accuracy risk, usually needing calibration data and a validation pass.

**KV-cache quantization** — FP8 KV cache — is a separate knob from weight quantization. It shrinks the cache specifically, which is often the real bottleneck at high concurrency and long context, independent of weight precision.

Decision process: baseline load test at FP16/BF16, try FP8 next since it is usually close to free on modern hardware, and only then reach for INT4 or AWQ. Validate task-specific quality, not just perplexity.

---

### V6: What is prefix caching (as distinct from PagedAttention itself) and why does it matter for RAG/chat workloads?

Prefix caching reuses KV-cache blocks across requests sharing an identical prompt prefix — the same system prompt, or the same long RAG context across follow-up questions — instead of recomputing prefill for the shared portion. It is a natural extension of PagedAttention, because that block-based cache is already non-contiguous and friendly to content addressing: hash blocks by content, look up, reuse hits.

The win is large for chat (repeated system prompt plus growing history) and RAG (same context, several questions), turning much prefill work into cache hits and dramatically improving TTFT. The design implication: your load balancer must be prefix- or cache-aware, routing requests likely to share a prefix to the same replica. Pure round-robin scatters hit opportunities across replicas sharing no cache state, and you lose most of the benefit.

---

### V7: What changed with vLLM's "V1" engine rewrite, and why should you know that as an interview talking point?

vLLM went through a significant core-architecture rewrite, alpha announced around early 2025, redesigning the scheduler and execution loop for lower CPU-side overhead, cleaner separation between scheduling and model execution, and better support for multimodal and newer architectures — while keeping continuous batching and PagedAttention as foundations.

Knowing vLLM is not a frozen design, and that its internals evolved while the user-facing API stayed largely stable, shows you track the ecosystem. Practically, exact V0-versus-V1 internals change fast: anchor on the invariants — continuous batching, paged block KV cache, chunked prefill, speculative decoding support — and flag version-specific internals as "verify against current docs".

---

### V8: What is disaggregated prefill/decode serving, and why would a senior team consider it?

Run prefill — compute-bound, bursty, parallelizable — and decode — memory-bandwidth-bound, needing sustained KV-cache residency — on separate GPU or node pools, transferring a request's KV cache from prefill worker to decode worker over fast interconnect or a shared store such as LMCache.

The motivation is that the phases have different optimal batch sizes and bottlenecks. Co-locating them means a long prefill can stall decode latency (mitigated but not eliminated by chunked prefill), and you cannot independently scale or tune each phase's hardware — prefill benefits from raw FLOPs, decode from memory bandwidth and cache capacity.

The cost is real complexity: a KV-cache transfer path, coordination between pools, and new failure modes such as the decode worker holding a request's cache disappearing. This is a very-large-scale optimization, not a default.

---

## Autoscaling Deep Dive (Ch. 06)

*Maps to Chapter 06 — Autoscaling (`hpa.yaml`). Builds on Q12 above.*

### AS1: Why is CPU utilization almost always the wrong HPA signal for a GPU-bound LLM serving pod?

The pod's CPU usage — tokenization, HTTP handling, orchestration — is largely decoupled from GPU load. A pod can sit at 90% GPU utilization and near KV-cache capacity while its CPU shows 10%, so a CPU-based target never fires when it should.

The correct signals are GPU- or engine-native: GPU utilization, KV-cache/block utilization, queue depth (pending requests waiting for a scheduler slot), or request latency directly. All of them need a custom or external metrics pipeline — the Prometheus Adapter, or a KEDA scaler reading Prometheus or the engine's metrics endpoint. HPA's built-in `Resource` metrics will not get you there.

---

### AS2: Design an HPA/KEDA config for a vLLM deployment using queue depth as the scaling signal.

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler
spec:
  scaleTargetRef:
    name: llm-serving
  minReplicaCount: 2
  maxReplicaCount: 20
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      query: avg(vllm_num_requests_waiting)
      threshold: "5"
```

`vllm_num_requests_waiting` rising means the running batch is already at capacity and work is backing up — a much earlier and more meaningful signal than GPU utilization plateaued at 95%.

Keep `minReplicaCount` above zero unless you have specifically solved cold start for scale-from-zero, or the first user after idle eats the full cold-start latency. Tune the scale-up stabilization window short and the scale-down window long, so a replica that just dipped below threshold is not torn down — GPU pods are expensive to bring back.

---

### AS3: Why is scale-to-zero for GPU inference pods so much harder than for a stateless web service?

A stateless web pod cold-starts in 1-5 seconds; a GPU serving pod's cold start — node provisioning, image pull, weight download, engine warm-up — is commonly 1-10+ minutes. Scale to zero and the first request after idle pays that entire cost, which almost never fits a reasonable latency SLO.

Mitigations: keep a warm minimum and never truly scale to zero on latency-sensitive paths; use keep-warm ping traffic to block scale-down during known-active hours; pre-provision a warm node pool so scale-up skips node provisioning; or accept scale-to-zero only for genuinely latency-insensitive batch workloads.

---

### AS4: How does cluster autoscaling (or Karpenter-style node autoscaling) interact with pod-level HPA for GPU workloads, and where does it commonly break?

HPA decides "we need N more pods". With no free GPU capacity those pods sit `Pending` until the cluster autoscaler or Karpenter notices and provisions a GPU node, adding full node-provisioning latency — often minutes — on top.

The common break is provisioning a node type lacking the GPU generation or count the pod's affinity and resource request demand, producing a provision-then-still-Pending loop; node pool definitions have to match exactly what the workload asks for. Scale-down breaks too: the autoscaler can drain a GPU node running a long-lived inference pod mid-generation, dropping or truncating requests, unless pod disruption budgets and graceful termination (Q5g) protect it.

GPU autoscaling is a two-layer problem and the layers' reaction latencies **stack** — total scale-up latency is the sum, not the max.

---

### AS5: What is predictive/scheduled autoscaling and when would you use it over purely reactive autoscaling?

Reactive autoscaling always lags demand by at least the metric-collection interval plus cold-start time — fine for smoothly varying load, poor for sharp predictable bursts.

Predictive or scheduled scaling pre-warms capacity ahead of known patterns — raising a minimum replica floor before a daily peak, a launch, or a marketing push — via a CronJob changing `minReplicaCount` or a dedicated controller. Use it whenever demand has a known or learnable pattern, because it attacks cold start directly by provisioning *before* the spike. Combine with reactive scaling for the unpredictable residual.

---

### AS6: A bursty consumer product (e.g., a viral social feature) spikes 20x in under a minute. Design the autoscaling response.

Pure reactive pod and node autoscaling cannot provision GPU capacity in under a minute, so the answer is not "scale faster" but "absorb the burst without needing to".

Four levers. **Admission control with backpressure**: bound the queue, return fast 429s beyond it, rather than degrading latency without limit for everyone. **Warm buffer capacity**: a standing buffer of replicas sized for the observed peak/burst ratio, funded as a deliberate cost trade-off. **Graceful degradation**: shed to a smaller model, cap `max_tokens`, or disable expensive features like long-context mode when queue depth crosses a threshold. **Per-user and per-tenant rate limits**, so one viral feature does not starve the platform.

Afterwards, use the observed burst shape to right-size the warm buffer and the predictive schedule (AS5).

---

### AS7: How do you avoid HPA "flapping" (rapid scale up/down) for a GPU workload, and why is flapping especially costly here vs. a CPU service?

The standard levers are `stabilizationWindowSeconds` on scale-down and rate limiting via HPA v2 `behavior` policies capping pods removed per period.

It is worse for GPU because a removed pod is not a process that restarts fast: bringing it back may mean re-provisioning a node (AS4) and re-downloading weights (K6/K7), so an unnecessary scale-down followed by a needed scale-up costs far more in latency and money than the same flap on a stateless service. Practical setting: asymmetric behaviour — short window for scale-up, minutes for scale-down — plus a `minReplicas` floor above the noise level of your traffic.

---

### AS8: How would you autoscale a multi-model platform where different models have very different load profiles and hardware needs?

Scale each model's deployment independently, with its own HPA/KEDA object and thresholds tuned to that model's capacity envelope — a 7B and a 70B saturate at completely different queue-depth and GPU-utilization numbers.

Use separate node pools per hardware profile — single-GPU nodes for small models, multi-GPU NVLink nodes for large tensor-parallel ones — so the cluster autoscaler provisions the right node shape, with taints and affinities (K5) preventing cross-contamination. For a long tail of low-traffic models, prefer a shared GPU pool with priority-based preemption over a dedicated autoscaling group each.

---

## Canary Deployments Deep Dive (Ch. 07)

*Maps to Chapter 07 — Canary Deployments (`canary-deployment.yaml`). Builds on Q13 above.*

### CD1: Walk through a fully automated canary pipeline for a new model version, end to end.

The version first passes an offline eval suite — quality benchmarks plus regression tests against known-bad outputs — before it is canary-eligible. It then deploys at 1-5% of traffic alongside stable, both behind the same gateway or mesh split.

An automated analysis window runs for a fixed period, typically 30-60 minutes, long enough for statistically meaningful samples, comparing canary against stable on error rate, P95 and P99 latency, and a proxy quality signal such as thumbs-down rate or a cheap automated eval on sampled outputs.

The decision is automatic: inside bounds steps traffic 5% → 25% → 50% → 100%; any breach rolls back to 0% immediately. At 100% the canary becomes the new stable, and the old stable is kept warm briefly for fast rollback, then scaled down. Argo Rollouts and Flagger implement this loop natively over Kubernetes, Prometheus, and a traffic-splitting layer.

---

### CD2: What metrics are *LLM-specific* red flags during a canary that a generic web-service canary pipeline wouldn't catch?

**Output quality regression**: a version can be faster and perfectly healthy on latency and errors while being worse — more hallucination, format violations, changed refusal rate. Infra metrics are blind to it. **Token-length distribution shift**: substantially longer or shorter outputs change cost and latency in ways an error-rate check never flags. **Safety/guardrail trigger rate**: a rise is a strong leading indicator of a behavior regression. **Structured-output and tool-call validity rate**: in agent pipelines a canary can look healthy while silently emitting malformed function calls downstream systems choke on.

So a real LLM canary needs an automated quality-eval step sampling live canary outputs before promotion — ideally a cheap LLM-as-judge or rule-based check running inline.

---

### CD3: Canary at the Kubernetes-Deployment-replica-ratio level vs. canary via a service mesh's percentage-based traffic split — what's the real difference, and when does it matter?

Replica-ratio canary — 9 stable pods plus 1 canary behind one Service — gives a split only as fine-grained as replica count allows, so 10 pods means 10% steps, and it couples "how much traffic the canary gets" to "how many pods it has", conflating blast radius with resource allocation.

Mesh or Gateway percentage split — Istio VirtualService weight, Gateway API HTTPRoute weight — decouples them entirely: exactly 1% of traffic to a canary running on a single full-sized pod. That matters most for large expensive models, where each 70B replica is a meaningful GPU cost and you want risk exposure decoupled from fleet sizing.

---

### CD4: Design the automated rollback trigger logic — what exactly should trip an automatic rollback, and how do you avoid both false positives and false negatives?

A small set of guardrail metrics with explicit thresholds evaluated over a rolling window rather than single data points: error rate above baseline + 2%, P99 latency above baseline × 1.5, guardrail-trigger rate above baseline plus an absolute threshold.

Require a **minimum sample size** before evaluating — a canary at 1% traffic for 2 minutes may have too few requests for the error-rate comparison to mean anything, and that is where false positives come from. Avoid false negatives by including the slower, sampled quality-eval signal (CD2), because a model can be infra-healthy and quality-broken at once.

Keep a manual kill switch alongside the automation: default to safe — roll back — on ambiguous or missing data, and let humans force a rollback regardless.

---

### CD5: How do you canary a change to *infrastructure* (e.g., a new vLLM version or a new GPU generation) rather than a new model weights version?

Same pipeline shape as CD1, but the canary variable is the serving stack or hardware. Deploy the new engine version or hardware behind a small traffic percentage running the *same* model weights as stable, so any regression is attributable to the infra change rather than confounded.

Watch for numerical and output-distribution differences an engine or hardware change introduces — different kernels, different default precision, different sampling implementation details — even with bit-for-bit identical weights. "Same model, different engine version, subtly different outputs" is a real, easy-to-miss failure mode. Load-test the new infra in isolation (Ch. 04) *before* canarying on live traffic; the live canary is the last safety net, not the first.

---

### CD6: What's the difference between canary, blue-green, and shadow (dark) traffic deployment strategies for model updates?

| Strategy | What happens | Risk exposure | Best for |
|---|---|---|---|
| **Canary** | Small % of *real* traffic gets *real* responses from the new version | Real users see new-version output, at small scale | Default choice — validates real behavior with bounded blast radius |
| **Blue-green** | Full traffic cutover from old to new at once (with fast rollback to "blue" if needed) | All users exposed simultaneously | When you need instant, all-or-nothing cutover and have high confidence from offline eval |
| **Shadow/dark traffic** | New version receives a *copy* of real traffic but its responses are discarded/logged, never shown to users | Zero user exposure | Validating latency/capacity/output-diffing on real traffic before any user ever sees the new version — great first step before a canary |

They are not mutually exclusive: a mature pipeline runs shadow first for zero-exposure validation, then a canary for bounded real exposure, then full promotion. Treating it as one all-or-nothing choice is itself the weaker answer.

---

### CD7: How do you handle stateful conversation context (multi-turn chat) correctly during a canary, so a user doesn't get inconsistent behavior mid-conversation?

Sticky routing keyed on session or conversation ID — consistent hashing at the gateway or mesh — so every turn of one conversation hits the same version for its duration. Otherwise a user gets wildly inconsistent behaviour, style, and quality switching between versions turn to turn.

That means canary percentage is measured in **sessions, not requests**: a 5% canary must mean 5% of conversations, not 5% of individual turns randomly assigned, which would corrupt every multi-turn conversation it touched. Record which version handled a session so post-hoc quality analysis can attribute a whole conversation to one version.

---

### CD8: Interviewer pushback: "Why not just A/B test in production analytics after full rollout instead of doing all this canary infrastructure?"

Full rollout exposes 100% of users to any regression immediately and simultaneously; bounding that blast radius *before* you have full-traffic data is the entire point.

Post-hoc analysis on a bad full rollout tells you how bad it was, not how to avoid most users experiencing it. By the time it surfaces, the bad responses have been shown, the cost incurred, and trust eroded at 100% scale.

They are also not alternatives: canary is the safe delivery mechanism, A/B analysis is the measurement layer. A canary literally *is* a small, automatically-managed live A/B test with a rollback trigger wired to its own results.

---

## Monitoring & Observability (Ch. 08)

*Maps to Chapter 08 — Monitoring (`prometheus_exporter.py`, `grafana_dashboard.json`, `prometheus.yml`).*

### Q14: What metrics should you monitor for LLM serving?

Application: request rate; latency at P50/P95/P99, with TTFT and inter-token latency tracked separately rather than only end-to-end; error rate; queue size, meaning pending requests waiting for a scheduler slot; throughput in input and output tokens/sec.

Model: generation time, average tokens per request, and the currently-running model version — essential for correlating a metric shift with a deploy. System: GPU utilization, GPU memory and KV-cache utilization used-versus-total, host CPU and memory pressure. Business: cost per request and per token, quality and feedback signals, API usage by endpoint, tenant, and model.

Three dashboards cover it: performance (latency, throughput, errors), resource (GPU, CPU, memory), and model (version-over-version comparison).

---

### M1: Design the Prometheus metric taxonomy for an LLM serving fleet — what should be a counter, gauge, or histogram?

**Counters**, monotonically increasing: total requests, total tokens generated, total errors by type — used to derive rates with `rate()`. **Gauges**, point-in-time: current queue depth, running and waiting sequence counts, GPU memory used, KV-cache block utilization. **Histograms**, bucketed distributions: request latency, TTFT, inter-token latency, tokens per request — anything you will want percentiles for.

Never store latency as a gauge or an average when you will want P95 or P99 later; you cannot recover percentiles from an average after the fact. Label carefully — model, version, tenant — but watch cardinality, because labelling by raw request ID or user ID explodes series count and can take Prometheus down.

---

### M2: What are the "four golden signals" and how do you adapt them specifically for LLM serving?

**Latency** splits into TTFT and inter-token latency rather than one end-to-end number, because their causes differ — queueing and prefill versus decode-step contention. **Traffic** becomes requests/sec *and* tokens/sec, since token volume rather than request count drives GPU load. **Errors** covers HTTP-level errors plus engine-level failures — OOM, timeout, unexpectedly truncated generation — that an HTTP-status-only view misses.

**Saturation** is GPU utilization *and* KV-cache/block utilization *and* queue depth. KV-cache saturation is frequently the real ceiling well before GPU compute hits 100%, so tracking GPU percentage alone hides the actual bottleneck.

---

### M3: Your Prometheus instance keeps falling over / running out of memory. What's the likely cause in an LLM serving context and how do you fix it?

Almost always cardinality explosion: an unbounded label — raw request ID, user ID, prompt hash — multiplies unique time series, and Prometheus memory scales with active series count.

The fix is a bounded label set (model, version, tenant-tier, status code), with per-request detail pushed to logs and traces instead of metrics. Also check scrape interval and retention, since a very short interval combined with high series count compounds the problem, and move long retention to remote-write storage — Thanos, Mimir, VictoriaMetrics — rather than a single local instance.

---

### M4: How do you set actionable alert thresholds for LLM latency without causing alert fatigue?

Alert on **SLO burn rate**, not raw threshold crossings — "we are consuming our latency budget X times faster than sustainable" — which adapts sensitivity so a blip does not page and a sustained regression does. Use multi-window, multi-burn-rate alerting: a fast short window for "page now" and a slower long one for "ticket, slow leak".

Alert on P95 and P99, but tune the threshold from your own load-test baseline (Ch. 04) rather than an industry number — "P95 < 500ms" is meaningless without your model size, hardware, and SLO context. Route by actionability: "GPU node down" pages on-call; "quality-eval score dipped 2%" is a ticket for the model team, not a 3am page.

---

### M5: How do you monitor and alert on cost, not just performance, for an LLM serving platform?

Track cost-per-request and cost-per-1K-tokens as first-class metrics, derived from GPU-hour cost divided by observed throughput. A regression that silently doubles cost-per-token — GPU utilization dropping from a bad batching config — is a real incident even when latency and errors look fine.

Alert on GPU utilization sustained well below baseline: "10 GPUs at 20% utilization for an hour" is wasted spend and an alertable condition. Where multi-tenant, attribute cost by tenant or model so a runaway customer does not hide inside an aggregate until the monthly bill arrives.

---

### M6: What's the role of distributed tracing (OpenTelemetry) in an LLM serving stack, beyond metrics and logs?

Metrics tell you *that* P99 latency is bad; tracing tells you *where* the time went on a specific slow request — gateway, queue wait for a scheduler slot, prefill, decode, or a downstream retrieval call for RAG.

It is most valuable for multi-hop architectures — gateway → router → serving pod → retrieval service or tool call — where one slow request could be slow at any hop and per-hop aggregates cannot correlate it. Propagate a trace ID from the gateway through the engine and downstream calls; most engines support OpenTelemetry or can be wrapped to emit spans. Sample at a rate balancing trace-storage cost against debuggability, sampling up on errors and high latency.

---

### M7: How do you build a monitoring setup that catches a *quality* regression (not just a latency/error regression) in production, continuously — not just during a canary window?

Continuously sample a small percentage of live production outputs — not only canary traffic — and run cheap automated quality signals over them: rule-based checks for format validity, length bounds, and refusal detection; a lightweight LLM-as-judge against rubrics; or comparison to reference outputs for a fixed regression-test prompt set replayed on a schedule.

Track them as time series with the same alerting discipline as latency (M4). A slow quality decay — from an unnoticed upstream data or prompt-template change, not just a bad deploy — needs burn-rate-style detection, because it never appears as a spike. This is the direct link to drift detection (Ch. 10): a quality-regression signal is often the first observable symptom of drift, well before a statistical test flags it.

---

## Model Versioning & Registry (Ch. 09)

*Maps to Chapter 09 — Model Versioning. Q19 is preserved from the original System Design section — versioning is really its own topic, not a system-design scenario.*

### Q19: How would you implement model versioning and rollback?

Version semantically — v1.0.0, v1.1.0, v2.0.0 — deciding explicitly what counts as major (architecture or behavior change), minor (fine-tune), and patch (config or quantization only). Store artifacts in a registry with metadata: training date, eval metrics, dataset version, base model lineage, and who approved promotion.

**1. Model registry layout:**
```
models/
  v1.0.0/
    model.bin
    tokenizer.json
    metadata.json
  v1.1.0/
    model.bin
    tokenizer.json
    metadata.json
```

**2. Deployment:**
```yaml
env:
- name: MODEL_VERSION
  value: "v1.0.0"
- name: MODEL_PATH
  value: "/models/v1.0.0"
```

**3. Rollback:**
```bash
# Update to previous version
kubectl set env deployment/llm-serving \
  MODEL_VERSION=v0.9.0

# Or use canary deployment to gradually
# route traffic back to the stable version
```

**4. API:**
```python
GET /api/v1/models/versions
POST /api/v1/models/rollback
{
  "target_version": "v1.0.0"
}
```

Validate offline before a version is canary-eligible, roll out gradually via canary (Ch. 07) rather than cutting over at once, track per-version performance continuously, and keep a changelog recording what data and eval each version was validated against.

---

### MV1: What actually belongs in a model registry's metadata, beyond "the weights file"?

**Lineage**: base model, fine-tuning or adapter data, training config, and the exact commit or config hash that produced the artifact. **Evaluation results**: the specific benchmark scores this version achieved, so any two versions are comparable on the same axes. **Serving compatibility**: which engine versions and quantization formats it is validated against — a checkpoint that behaves well in FP16 on vLLM 0.6.x is not automatically identical under a different engine or quantization.

**Approval and promotion metadata**: who approved it for canary or production, when, and against what criteria — this is what makes rollback and audit possible after the fact. **Content hash or checksum**, to catch silent corruption or a wrong-file upload before it reaches a serving pod.

---

### MV2: Should model artifacts be treated as immutable once published, and what does that discipline actually buy you?

Yes. Once `v1.2.0` is published it is never overwritten in place; a fix gets `v1.2.1`.

Immutability is what makes rollback trustworthy — "roll back to v1.1.0" only means something if v1.1.0 today is bit-for-bit what was validated and ran in production before. It also makes post-hoc debugging possible: with mutable artifacts, "which model produced this bad output six weeks ago" is unanswerable. Implement it with content-addressed storage or write-once buckets with versioned object keys, plus a registry API that rejects overwriting an existing tag.

---

### MV3: How do you decouple "code version" (serving engine/app) from "model version" (weights) in your deployment pipeline?

Treat serving image version and model version as independent axes with their own cadences: a fine-tune should not require an application rebuild, and an engine upgrade should not require touching the model artifact.

The mechanism: the serving image is generic and loads whatever `MODEL_PATH`/`MODEL_VERSION` it is told at startup, with model version injected by config — env var, ConfigMap, or a model-serving CRD. That is exactly why runtime weight-mounting beats baking weights in (D5). It is also what makes independent canarying possible: canary a new model on the same engine, or a new engine on the same weights (CD5), instead of changing both at once and losing attribution.

---

### MV4: Design the rollback SLA — how fast can/should a rollback actually happen, and what determines that floor?

The fastest rollback is traffic-routing only — flipping a canary weight back to 0% — and that is near-instant, seconds, *if* the stable version's pods are still warm. Keeping the previous version's pods alive for a fast-rollback window after promotion, rather than scaling them down immediately, is a deliberate cost trade-off.

If those pods are already gone, rollback requires re-provisioning capacity, so rollback time equals cold-start time (K7) — minutes, not seconds. That is the common gap between "we have rollback capability", true, and "our rollback SLA is under a minute", false unless you kept the old version warm. So the decision to communicate is the retention window: keep N-1 warm for X hours after promotion, sized against risk tolerance and budget.

---

### MV5: How do you version and roll back *prompts/system prompts* and *retrieval configuration*, not just model weights, given that both affect output just as much?

Treat prompt templates, system prompts, and RAG configs — index version, chunking parameters, retrieval top-k — as versioned artifacts with the same discipline as weights. A "model regression" investigation that checks weight version and ignores a same-day prompt-template change will miss the actual cause.

Store prompt and config versions alongside model version in registry metadata, so a production request maps to an exact (model, prompt template, retrieval config) triple. This matters for canary too: a system-prompt change deserves the same rigor as a model swap, and the common trap is treating it as "just a config tweak" that skips the deployment safety pipeline.

---

### MV6: What does "shadow deployment" buy you specifically for model *version* validation that offline eval doesn't?

Offline eval runs against a fixed historical dataset. It cannot tell you how the new version behaves on *today's* live traffic distribution, which may already have shifted away from what the eval set represents.

Shadow deployment runs the new version against live traffic in parallel with production, discarding its output but logging it for comparison against the production version's actual response on the same input. That catches distribution-shift-sensitive regressions offline eval structurally cannot see. The cost is doubled compute on the shadowed slice, so it is sampled and time-boxed before moving to a real traffic-splitting canary.

---

### MV7: Interviewer pushback: "Why do you need a formal model registry at all — why not just tag Docker images with the model baked in and use normal image-based deployment tooling?"

Baking weights into images couples model release cadence to image build, push, and pull cost, which is wasteful when fine-tunes are far more frequent than code changes — and wasteful in reverse too. A registry also carries model-specific metadata — eval scores, lineage, approval status — that a generic container registry has no concept of, so you end up building a parallel ad hoc tracker anyway, without the rollback API and promotion gates.

That said, it is a genuine trade-off. For a small team with infrequent updates and no multi-model complexity, baking weights into a versioned image *is* legitimate and simpler. The signal is recognizing *when* a registry earns its keep — frequent updates, multiple models, fast independent rollback — versus when it is over-engineering.

---

## Drift Detection (Ch. 10)

*Maps to Chapter 10 — Drift Detection (`drift_detector.py`). Q15 is preserved from the original Monitoring section — drift detection is really its own chapter/topic.*

### Q15: How would you detect model drift in production?

Three kinds. **Data drift**: the input distribution changes — users ask about topics or formats the model rarely saw. **Concept drift**: the input-output relationship changes — what counts as a correct answer shifts after a real-world event or policy change. **Prediction drift**: the output distribution changes — responses get longer, shorter, or differently structured.

Detection uses statistical tests — PSI (Population Stability Index) comparing reference against current, Kolmogorov-Smirnov for distribution differences, chi-square for categorical — surfaced through Evidently AI, custom Prometheus drift-score metrics, or direct comparison scripts.

```python
from evidently import Report, DataDriftTable

report = Report(metrics=[DataDriftTable()])
report.run(
    reference_data=train_data,
    current_data=production_data
)

if report.get_metric(DataDriftTable()).drift_detected:
    alert("Data drift detected!")
```

Set thresholds such as PSI > 0.2, monitor continuously rather than only at deploy time, then act: retrain or refresh if drift is significant and persistent, investigate *why* before reacting, and update the baseline if the shift is a legitimate new normal.

---

### DR1: For an LLM specifically (as opposed to a classic tabular ML model), what does "input drift" actually mean, and how do you measure it over free-form text?

You cannot run PSI or KS directly on raw text. You need a numeric representation first: prompt length distribution, embedding-space distance (embed prompts and track the centroid or distribution shift against a reference window), topic or intent classification distribution, or simple proxies such as detected language and keyword presence.

Embedding-based drift — Maximum Mean Discrepancy, or simply centroid distance over time — is the most common practical approach for free-form text, because it captures semantic shift without hand-built categorical features. The operational warning: embedding drift detectors need maintenance, because changing your embedding model invalidates the baseline and forces re-establishing it.

---

### DR2: What is "prediction drift" for a generative model, and why is it a *different* problem from input drift?

Prediction drift means the *output* distribution shifts — average response length, refusal rate, sentiment or tone, or the rate of a failure mode like repetition loops or format violations — even while inputs are stable.

It can happen with zero code or model change: an upstream prompt-template tweak, worse retrieved-context quality for RAG, or a subtle change in default sampling parameters. It matters separately because input drift says the world or the users changed, while prediction drift says the *system's behavior* changed — and a system can have a completely stable input distribution while output behavior silently regresses. That is often the first symptom of a bug.

---

### DR3: How would you set up automated drift-triggered retraining/refresh, and what are the dangers of doing this fully automatically?

Pipeline shape: a drift monitor scores on a schedule, the score crosses a threshold, that triggers retraining or fine-tuning on fresh data, and the new version goes through the normal eval and canary pipeline — never straight to production.

Danger one is **feedback loops**: retraining on production outputs, or on user reactions to them, without careful filtering bakes in and amplifies the drift that caused the trigger — the model drifts toward sycophancy, retrains on sycophantic outputs users engaged with more, and drifts further. Danger two is **false-positive churn**: an over-sensitive threshold retrains constantly on noise, burning compute and introducing instability, since each new version is itself a source of risk.

So drift detection should trigger an alert and investigation, with retraining as a human-gated decision. Full automation is a reasonable long-term goal only once eval, canary, and rollback are strong enough that a bad automated retrain cannot reach users.

---

### DR4: What's the difference between monitoring for drift and monitoring for outright *model degradation/failure* (e.g., repetition loops, refusals, garbage output)?

Drift is a *distributional* concept — comparing distributions over time against a reference — and can be entirely benign, because the world legitimately changed, and it is usually gradual.

Degradation detection catches acute, often binary bad behavior on individual outputs: repetition loops, empty outputs, malformed JSON when structured output was requested, a refusal on a benign request. That needs per-request rule-based checks, not distributional statistics.

They complement each other. Drift catches slow systemic shifts that per-request checks never notice, because each output looks fine while the aggregate moved. Per-request detection catches acute breakage that a distributional test averaged over thousands of requests would dilute into invisibility.

---

### DR5: How do you build a drift baseline/reference distribution in the first place, and how often should you refresh it?

Build the reference from a recent, validated window of production traffic — or the training distribution if traffic has not started — not an arbitrarily old snapshot. The reference must represent a period you are confident was healthy.

Refresh cadence is a two-sided trade-off. Refresh too often and you normalize away real slow drift, because comparing yesterday to today essentially never shows drift and gradual multi-week shifts stay hidden. Refresh too rarely and normal seasonal changes get flagged as drift forever. The common pattern: hold a fixed reference window — "traffic from the last validated release" — until a human explicitly promotes a new one, typically alongside a version bump. Refresh is a deliberate reviewed action, not a rolling automatic window, for the same reason immutable model versions matter (MV2).

---

### DR6: How does drift detection interact with multi-tenant serving, where different tenants have very different, legitimately-different traffic patterns?

A single global baseline shows "drift" constantly, simply because tenant mix shifts — one customer's usage differs from another's and relative volumes change week to week. At the aggregate level that is noise, not signal.

Better: maintain per-tenant or per-segment baselines and drift scores, and escalate globally only when drift appears *within* a tenant's own traffic over time, or when enough tenants show it simultaneously to suggest a shared cause such as a model or prompt-template change. This is directly analogous to why per-tenant cost and latency dashboards matter (M5): aggregates over a heterogeneous population hide exactly the signals you need.

---

### DR7: Interviewer pushback: "PSI and KS tests are from classical tabular ML monitoring — do they even make sense for LLMs, or is this cargo-culting a technique from a different problem?"

Fair pushback, and the honest answer is "partially". PSI and KS suit comparing distributions of *derived numeric features* — prompt length, embedding-distance scores, response length, latency — and remain genuinely useful there. They are not applicable to raw text without that extraction step.

What is LLM-specific and not borrowed: output-quality proxies such as guardrail trigger rate, refusal rate, and LLM-as-judge scores on sampled outputs. Those have no analogue in tabular drift detection and are arguably the more important signal for generative systems.

The best answer combines both — classical tests on derived numeric features, cheap and good for continuous monitoring, plus quality-eval sampling, more expensive but the ground-truth signal the statistical tests only proxy for.

---

## Triton Inference Server (Ch. 11)

*Maps to Chapter 11 — Triton.*

### T1: What problem does Triton Inference Server solve that a hand-rolled FastAPI + PyTorch server doesn't?

**Multi-framework, multi-model serving from one process**: PyTorch, TensorFlow, ONNX, TensorRT, and via the vLLM/TensorRT-LLM backends, LLM engines, all behind one gRPC/HTTP API. **Dynamic batching and concurrent model execution** built in and configurable per model, without hand-written scheduling logic. **Model repository abstraction**: models deploy by landing in a directory with a config file, and versioning, loading/unloading, and multi-version serving are server features rather than application code. **Production observability out of the box**: built-in Prometheus metrics, lifecycle events, per-model and per-version statistics.

The trade-off is more operational surface and a configuration model (`config.pbtxt`) to learn — worth it with multiple models, frameworks, or versions to manage; arguably overkill for a single model on a single serving path.

---

### T2: Describe the Triton model repository layout and what `config.pbtxt` controls.

```
model_repository/
  my_model/
    config.pbtxt
    1/
      model.onnx
    2/
      model.onnx
```

Each top-level directory is a model name; numbered subdirectories are versions, so Triton serves several versions of the same model simultaneously — directly useful for canary and A/B, with Ch. 07 concepts applying at the Triton layer.

`config.pbtxt` controls input and output tensor names, shapes, and datatypes; the backend (`onnxruntime`, `pytorch`, `tensorrt`, `vllm`, or `python` for custom); batching configuration including max batch size and dynamic batching parameters; instance groups, meaning how many copies run and on which GPUs; and version policy — latest only, all, or a specific set.

---

### T3: Explain Triton's dynamic batching vs. vLLM's continuous batching — are these the same idea?

No. **Triton dynamic batching** accumulates individual requests arriving within a short configurable window into one batch, then runs one forward pass — fitting models with a fixed-shape, single-pass profile: classification, embedding, non-autoregressive models.

**Continuous, iteration-level batching** is specific to autoregressive generation: it batches individual decode *steps*, letting requests join and leave mid-generation. Plain dynamic batching fits that badly, because requests finish at wildly different times depending on output length. So when serving LLMs *through* Triton you use the vLLM or TensorRT-LLM backend, which brings continuous-batching-style scheduling into Triton's process.

---

### T4: What are Triton's ensemble models, and when would you use one in an LLM serving pipeline?

An ensemble defines a pipeline of models and steps — preprocessing → embedding → retrieval → generation → postprocessing — as a single logical model that Triton executes as a DAG, handling data hand-off internally. It suits a RAG pipeline where one API call should trigger embed-query, retrieve, and generate, with Triton scheduling and batching each stage instead of the client making three round trips.

The trade-off: per-stage debugging is harder, since the client sees one opaque pipeline, and stage-level independent scaling is more constrained than separate Deployments. Good for tightly coupled, latency-sensitive pipelines; poor when stages have very different scaling and failure characteristics.

---

### T5: How does Triton fit into a Kubernetes deployment, and what does the "instance group" concept map to?

Triton runs as a normal container in a Deployment, typically requesting `nvidia.com/gpu` like any GPU workload; Kubernetes needs to know nothing Triton-specific.

**Instance groups** in `config.pbtxt` control how many copies of a *given model* Triton runs *within one Triton process or pod*, and on which GPUs — a Triton-internal concept distinct from and complementary to Kubernetes replica scaling, which runs multiple pods each potentially running several instance groups. So use instance groups to pack models onto the GPUs *within* a pod, and Kubernetes HPA and replicas to scale the *number of pods*. Conflating the layers is a common early mistake.

---

### T6: When would you choose Triton + a TensorRT-LLM backend over a plain vLLM deployment for LLM serving?

**TensorRT-LLM via Triton** is typically the highest raw throughput and lowest latency option on NVIDIA hardware, because it compiles model- and hardware-specific optimized kernels ahead of time via graph fusion and precision-specific kernel selection. The cost is an explicit build step per model + GPU-generation + precision combination, a steeper operational curve, and less flexibility for new or unusual architectures.

**vLLM**, standalone or via Triton's vLLM backend, gets a new model serving faster, has a simpler operational model, delivers strong out-of-the-box performance with no compile step, and moves faster on new architecture support.

Rule of thumb: TensorRT-LLM for a small number of stable, high-volume models where the extra investment pays back in throughput and cost; vLLM when iterating quickly across many changing models. The space moves fast — TensorRT-LLM's build ergonomics keep improving and NVIDIA NIM packages the trade-off as a pre-built container — so flag current capability claims as "verify before quoting".

---

### T7: How does Triton support A/B testing or canary between model versions natively?

The model repository natively supports multiple numbered versions of the same model name (T2) with a configurable version policy, so two versions can be loaded simultaneously and routed between.

Native Triton does not do traffic-percentage splitting itself — that decision lives in front, in an API gateway, a service mesh, or application logic. Triton's job is to have both versions loaded and ready to serve whichever is asked for. So the canary *mechanics* from Ch. 07 apply exactly as described; Triton only changes where "which version is loaded and serving" lives.

---

### T8: What monitoring does Triton expose natively, and how do you wire it into a Prometheus/Grafana stack (Ch. 08)?

A built-in `/metrics` endpoint in Prometheus format with per-model, per-version statistics: request count, inference duration broken out into queue time versus compute time, GPU utilization and memory, and success/failure counts.

Wire it in like any other Prometheus target — a `prometheus.yml` scrape config pointed at the Triton pod's metrics port. Not having to build this instrumentation yourself is one of Triton's practical advantages. Because it separates queue time from compute time by default, it directly supports the golden-signal breakdown from M2 with no custom work.

---

### T9: Interviewer pushback: "If vLLM alone already gives you continuous batching, PagedAttention, and an OpenAI-compatible API, what does adding Triton in front actually buy you — isn't it redundant infrastructure?"

For a single-model, single-framework, LLM-only deployment it genuinely can be redundant. Running vLLM's own OpenAI-compatible server directly is a perfectly reasonable, simpler answer, and it is worth saying so rather than defaulting to "always use Triton".

Triton earns its keep when you need one unified serving layer across heterogeneous models and frameworks — LLMs via vLLM/TensorRT-LLM, classical models via ONNX/TensorRT, embedding models, all behind one API and ops surface — or when you specifically need ensembles, fine-grained instance-group GPU packing, or the native multi-version model repository. Name the trade-off rather than a universal rule.

---

## Production Best Practices

### Q16: What are the key considerations for production LLM serving?

**Performance**: an explicit SLO such as P95 TTFT < 300ms rather than a generic "fast"; throughput validated by load testing (Ch. 04); autoscaling on demand (Ch. 06). **Reliability**: liveness, readiness, and startup probes (Q5e); graceful degradation instead of hard failure; circuit breakers against cascade failures; retries with exponential backoff and jitter, bounded so they cannot amplify an overload.

**Monitoring**: metrics (Ch. 08), structured logging with request and trace correlation, SLO-burn-rate alerting (M4), cross-hop tracing (M6). **Security**: API keys, OAuth, or mTLS between internal services; rate limiting to protect GPU capacity; input validation (Q5f); secrets in Kubernetes Secrets or a secrets manager, never baked into images. **Cost**: right-sized instances (Q20), scale-down when idle, efficient models or quantization where quality allows. **Model management**: versioning (Ch. 09), A/B comparison on real traffic (Ch. 07), fast rollback (MV4), drift detection (Ch. 10).

---

### Q17: How would you handle a sudden spike in traffic?

Immediately: HPA or KEDA scales up (Ch. 06), but that has real latency (AS4). Load-balance across pods, ideally prefix-aware (V6). Queue with a bounded queue, and rate limit to protect the backend — reject cheaply rather than letting everyone degrade.

While it happens, watch pod count and node provisioning progress, latency split TTFT versus inter-token, error and reject rates, and GPU plus KV-cache utilization. If overwhelmed, degrade gracefully — smaller model or capped `max_tokens` — rate limit with a fast 429 rather than a slow timeout, scale manually if autoscaling is too slow, and add capacity on a spot/on-demand mix.

Prevention is capacity planning from known per-replica capacity (LT7), load testing at and above expected load, properly configured autoscaling with predictive scaling for known patterns (AS5), and circuit breakers. Afterwards, use the burst as capacity-planning input (AS6), resize the warm buffer, raise baseline capacity if the level looks sustained, and update the runbook.

---

### Q20: Explain how you would optimize costs for LLM serving.

Compute — GPU instances — is usually the dominant cost by far, ahead of model and log storage, cross-region network transfer, and observability overhead.

Right-size first, from load-test data (LT7) rather than over-provisioning "just in case". Then autoscale down during low traffic, scale up predictively for known patterns, and use spot or preemptible instances for non-latency-critical batch work. Quantize (FP8/INT8/INT4 — see V5) or distill where quality allows. Caching is the highest-leverage lever for chat and RAG: prefix and KV-cache reuse (V6), plus response caching where semantically valid. Batch with continuous batching (Q7), because higher GPU utilization is directly lower cost per token — track it as a metric (M5), not just as "faster". Track cost per request and per token continuously, alert on waste, and commit to reserved capacity for predictable baseline load with spot or on-demand only for burst.

Worked example: 10 GPUs at 50% utilization versus 5 GPUs at 90% utilization via better batching and prefix caching, serving the same traffic. **Illustrative savings: ~50%** — validate with your own numbers, since exact ratios depend heavily on workload shape.

---

## Additional Quick Questions

### Q21: What is the difference between batch size and sequence length?

Batch size is the number of requests or sequences processed together; sequence length is the number of tokens in one request, prompt plus generated so far. Batch size 8 means eight requests processed simultaneously in traditional batching, or concurrently in flight under continuous batching; sequence length 512 means each request runs up to 512 tokens.

They hit memory differently. Larger batch size raises throughput and memory, because more KV cache is resident at once. Longer sequences raise both computation and memory, because KV cache grows linearly with sequence length per sequence.

---

### Q22: How does quantization affect model performance?

FP32 → FP16/BF16 is roughly 2x smaller and faster with minimal accuracy loss. FP16/BF16 → FP8 is roughly 2x smaller again, with meaningful speedup on Hopper and Blackwell-class tensor cores and small workload-dependent accuracy impact. FP16 → INT8 is roughly 2x smaller and faster with small accuracy loss, and may need calibration. INT8 → INT4 is roughly 2x smaller again with larger accuracy loss needing careful per-task validation.

The trade is faster inference, less memory, and lower cost against potential accuracy loss and the need for calibration data and per-task validation — not just perplexity. Use it in production when speed or cost matters and quality is validated as acceptable on your actual eval suite, and for memory-constrained deployments on smaller GPUs or at higher concurrency targets.

---

### Q23: What is the difference between model parallelism and data parallelism?

**Data parallelism** replicates the same model across GPUs or nodes with different data — different requests, in inference — on each replica. In training that needs gradient sync; in inference it is just "run N independent replicas", which is what Kubernetes replica scaling does.

**Model parallelism** splits the model across GPUs — tensor parallelism within a layer, pipeline parallelism across layers (V4) — so each GPU holds part of it. It is for models that do not fit on one GPU, or that need the combined memory bandwidth of several to hit a decode-latency target.

Concretely: data parallel inference is 8 GPUs each running an independent full copy of a 7B model on different requests — 8 replicas. Model parallel inference is 8 GPUs each holding 1/8 of a 70B+ model's weights, collectively serving one request's forward pass.

---

## System Design Scenarios

*This section replaces the original thin "System Design" section with 5 full worked scenarios. Q18 (the original quick-reference design question) is preserved below as a compact warm-up; use it if you only have two minutes, and use the full scenarios if you have twenty.*

### Q18: Design a system to serve LLMs at scale (quick reference).

**Architecture:**
```
[Load Balancer]
    |
[API Gateway] (Rate limiting, Auth)
    |
[Kubernetes Cluster]
    +-- [LLM Serving Pods] (vLLM)
    +-- [Monitoring] (Prometheus, Grafana)
    +-- [Model Registry] (S3/GCS)
```

**Components:** load balancer (traffic distribution, health checks, TLS termination) -> API gateway (auth, rate limiting, routing, versioning) -> serving layer (vLLM, HPA, GPU nodes) -> model storage (registry, versioning, node-local caching) -> monitoring (Prometheus/Grafana, centralized logging, alerting) -> data pipeline (request logging, drift detection, A/B testing).

**Scaling strategy:** horizontal (more pods via HPA), vertical (bigger GPUs for bigger models), multi-region (geographic distribution).

**Key metrics:** latency (P50/P95/P99, split TTFT/inter-token), throughput (tokens/sec), error rate, GPU/KV-cache utilization, cost per request.

*(The full scenarios below go through the actual reasoning — clarifying questions, trade-offs, and pushback — a senior interviewer expects for any one of these architecture pieces in depth.)*

---

### Scenario 1: Design serving for a 70B-parameter model at a defined cost/latency SLO

**Prompt as given in an interview:** "Design an inference platform to serve a 70B dense model. Target: P95 TTFT under 500ms, P95 end-to-end under 5s for a 500-token response, at the lowest cost per request you can justify."

**Clarifying questions to ask first:** expected volume and concurrency, peak and average, because 10 req/s and 10,000 req/s are different architectures; prompt-length distribution, short chat turns versus long RAG contexts, which drives prefill cost and chunked-prefill tuning; whether streaming is required, which affects the TTFT-versus-end-to-end split; whether a hard availability SLA constrains redundancy; and whether "lowest cost" is a preference or a hard number.

**Architecture:**
```
                     [ CDN / Edge ]
                          |
                 [ API Gateway / LB ]
                (auth, rate limit, routing)
                          |
        +-----------------+------------------+
        |                                    |
 [ K8s: 70B model pool ]            [ K8s: smaller fallback model pool ]
 (TP=4 or TP=8, per-replica)         (see Scenario 5 for fallback design)
        |
 [ Replica: 4x H100/H200 per pod, NVLink, TP=4 ]
 [ vLLM engine: continuous batching, chunked prefill, FP8 weights+KV ]
        |
 [ Prefix-cache-aware LB in front of replica pool ]
        |
 [ Model registry (S3) + node-local NVMe weight cache ]
        |
 [ Prometheus/Grafana, HPA on queue depth, cluster autoscaler on node pool ]
```

**Tensor parallelism degree is the single most important number here.** A 70B model in FP16 needs ~140GB for weights alone, so it does not fit on one 80GB GPU. TP=4 across four 80GB-class NVLink GPUs is a common fit; TP=2 with FP8 weights (~70GB) might fit on two with headroom for KV cache, halving GPU count per replica in exchange for quality risk — validate that with an eval suite before committing.

At 500-token outputs and target concurrency, KV cache is likely the binding constraint on concurrent sequences per replica: size the `gpu_memory_utilization` fraction accordingly and consider FP8 KV cache to roughly double effective concurrency. Enable chunked prefill so occasional long prompts do not spike inter-token latency, which directly protects P95 TTFT under mixed load. With a shared system prompt or repeated RAG context, prefix caching plus cache-aware routing is the highest-leverage cost and latency lever. Set replica count from load-test-derived per-replica capacity at the SLO (LT7), not a theoretical FLOPs calculation.

**Cost levers, ranked:** prefix caching (near-free once implemented); FP8 quantization of weights and KV cache (validate quality first); right-sizing TP degree against actual concurrency need; and spot/preemptible capacity for any tier that tolerates interruption, rarely the primary serving path.

**Defending against pushback.** *"Why not 8 GPUs per replica for headroom?"* — more GPUs per replica means more idle capacity most of the time and coarser scaling granularity, since each step adds 8 GPUs of cost rather than 4; right-size TP degree and use replica *count* as the load knob. *"Why not cache everything?"* — caching helps repeated content, not a genuinely novel prompt; for open-ended generation GPU cost is a floor you reduce, not eliminate. *"How do you know the load-test numbers hold?"* — they will not exactly, which is why the design carries N+1 headroom, capacity-based autoscaling as a second line, and continuous production latency monitoring.

---

### Scenario 2: Design a multi-tenant inference platform serving several model sizes to different internal teams

**Prompt as given in an interview:** "Several teams want to use a shared LLM platform: one needs a small 7B model for high-volume simple tasks, one needs a 70B model for complex reasoning, one is experimenting with a new fine-tune weekly. Design the platform."

**Clarifying questions to ask first:** whether tenants need hard isolation as a compliance boundary or just fair sharing; whether per-tenant cost attribution is required; how often new models appear, meaning a fixed catalog or a changing one; and whether latency SLOs differ per tenant.

**Architecture:**
```
                [ Gateway: auth, per-tenant rate limits, routing by model id ]
                                    |
        +---------------------------+---------------------------+
        |                           |                            |
 [ 7B model pool ]           [ 70B model pool ]         [ Experimental fine-tune pool ]
 (many small replicas,       (few large TP=4 replicas,   (small pool, DRA/MIG-shared GPUs,
  high concurrency,           priority: high)              priority: low, preemptible)
  priority: medium)
        |                           |                            |
        +---------------------------+---------------------------+
                                    |
                    [ Shared GPU node pools, PriorityClasses + taints ]
                    [ Per-tenant cost dashboards (Ch. 08, M5) ]
                    [ Per-model registry entries + independent canary pipelines (Ch. 07/09) ]
```

**Isolation**: namespaces per tenant for policy and quota isolation, plus MIG or DRA-managed GPU partitions (K3) where hard isolation matters — the experimental fine-tune must not be able to starve the 7B production pool of GPU memory on a shared node. For the 70B pool, dedicated whole-GPU nodes make more sense, since MIG-slicing a large model's TP group adds complexity for little benefit.

**Priority-based preemption**: production pools get a higher `PriorityClass` than the experimental pool, so under GPU pressure Kubernetes preempts experimental work first, letting the platform run experiments cheaply on spare capacity. **Independent scaling and cost**: each pool gets its own HPA/KEDA config tuned to its capacity envelope (AS8), and every request and metric is tagged with tenant ID at the gateway so cost-per-tenant dashboards work for chargeback and for catching runaway usage. The weekly-changing model gets its own registry entry and canary pipeline (Ch. 09), independent of the stable models' cadence.

**Defending against pushback.** *"Why not a dedicated cluster per tenant?"* — full dedication maximizes isolation but loses the cost benefit of sharing GPU capacity across tenants with complementary peaks; shared infrastructure with strong logical isolation gets most of the isolation at a fraction of the cost, short of a hard compliance requirement for physical separation. *"What if the experimental pool needs more than spare capacity?"* — that is what the priority design handles: its requests queue or degrade rather than starving production, and if the workload becomes important it graduates to its own provisioned pool as a deliberate decision.

---

### Scenario 3: Design the rollout pipeline for model updates with zero user-visible downtime

**Prompt as given in an interview:** "Design how a new model version goes from 'trained' to 'serving 100% of production traffic' with zero downtime and minimal risk."

**Clarifying questions to ask first:** full model swap or incremental fine-tune; acceptable time-to-full-rollout, hours or same-day; whether an offline eval suite exists or must be built; and whether multi-turn conversations are in play, which decides sticky-routing requirements (CD7).

**Pipeline:**
```
[New checkpoint] -> [Offline eval suite: benchmarks + regression tests]
       |  (fail -> stop here, never reaches serving)
       v
[Register in model registry, immutable version tag] (Ch. 09)
       |
       v
[Load onto a small pool, SHADOW traffic only] (CD6) -- validate latency/capacity + diff outputs vs. stable, zero user exposure
       |  (fail -> fix, re-shadow; never promote)
       v
[Canary: 1-5% real traffic, sticky by session] (CD1, CD7)
       |
   [Automated analysis window: latency, errors, guardrail-trigger rate,
    quality-eval sampling] (CD2, CD4)
       |
   pass -> step up (5% -> 25% -> 50% -> 100%)   fail -> automatic rollback to 0%
       |
       v
[100% traffic on new version; keep previous version warm for N hours] (MV4, fast-rollback window)
       |
       v
[Scale down previous version after fast-rollback window elapses]
```

**Zero downtime comes from traffic-weight shifting, never a hard cutover.** At every stage both versions are fully up and only the routing weight changes, so capacity never dips to serve the swap — in contrast with `RollingUpdate`'s pod-replacement churn (K2).

**Sticky-by-session routing** (CD7) is non-negotiable for multi-turn chat; without it, zero downtime at the infra level still produces a broken user experience. **Immutable versioning plus a fast-rollback warm window** (MV2, MV4) turns "we can roll back" into "we can roll back in seconds", which matters most when something breaks *after* full promotion. And **automated analysis must include quality signals** (CD2), because a pipeline checking only latency and errors will happily promote a model that is healthy and worse.

**Defending against pushback.** *"A lot of infrastructure for a one-line config change."* — the infrastructure cost is paid once; ad hoc rollouts pay a recurring risk cost on every update forever, and for an actively-used product those updates are constant. *"What if offline eval misses something?"* — that is exactly why there are three stages with different blind spots: offline eval is fast but historical, shadow catches live-distribution issues at zero exposure, canary catches the remainder with bounded exposure and an automatic kill switch.

---

### Scenario 4: Design monitoring and autoscaling for a bursty consumer product

**Prompt as given in an interview:** "A consumer-facing feature can go viral and spike traffic 10-20x within minutes, then fall back down. Design the monitoring and autoscaling approach."

**Clarifying questions to ask first:** the cost tolerance for a warm buffer versus the risk of degrading during a spike; whether bursts have any predictability or are fully organic; and the acceptable degradation mode if capacity is exceeded — slower responses, a smaller model, or hard rejection.

**Architecture:**
```
[ Gateway: per-user rate limit, bounded admission queue ]
       |
[ Warm buffer pool: sized for observed P99 burst ratio, always on ] (AS6)
       |
[ Reactive HPA/KEDA on queue depth ] --(fast scale-up policy)--> [ additional replicas ]
       |
[ Cluster autoscaler: pre-provisioned warm node pool ] (skips node-provisioning latency, K7/AS4)
       |
[ Degradation controller: watches queue depth ] --(threshold crossed)--> 
       [ shed to smaller fallback model / cap max_tokens / shed low-priority traffic ] (Scenario 5)
       |
[ Monitoring: real-time dashboards on queue depth, GPU util, TTFT P95/P99,
  reject rate -- alerts on SLO burn rate, not static thresholds ] (M4)
```

**Size the warm buffer off the observed burst ratio, not the average** — a deliberate cost-versus-resilience trade communicated explicitly: "we run at 40% baseline utilization specifically to absorb a 10x spike within 60 seconds". A thin margin plus pure reactive scaling guarantees degraded UX during exactly the highest-visibility traffic moments.

**Bounded admission queue with fast, honest rejection**: past a threshold, return fast 429s with retry-after rather than accepting requests into a growing queue that times out anyway. **Pre-provisioned warm node pool**: keep GPU nodes standing ready even without pods placed, so pod scale-up never waits on node provisioning (K7) — often the single biggest lever between detecting a spike and serving it. Layer **predictive scaling** (AS5) on top for known events, and keep **graceful degradation** (Scenario 5) as the tested last line.

**Defending against pushback.** *"Isn't a permanently warm buffer wasted spend?"* — partly; it is explicit insurance sized against the business cost of a bad viral moment, which for a consumer product can mean losing exactly the users the moment was meant to bring in. *"Why not autoscale hard and fast instead?"* — because GPU node provisioning and cold start (K7) cannot react in under a minute in most environments; the buffer compensates for a real operational floor, not a config mistake.

---

### Scenario 5: Design a fallback/degradation strategy when GPU capacity runs low

**Prompt as given in an interview:** "Your serving fleet is at capacity — demand exceeds what your GPUs can serve within SLO, and more capacity isn't available immediately (quota limit, spot capacity dried up, etc.). Design the degradation strategy."

**Clarifying questions to ask first:** whether a smaller fallback model exists and whether a quality drop is acceptable for some traffic; whether there is user tiering (paid versus free) that should decide who degrades first; and whether a slower response is acceptable or the product needs a hard latency ceiling even at reduced quality.

**Architecture:**
```
[ Gateway: tracks real-time capacity signal (queue depth / KV-cache util) ]
       |
       v
   capacity signal crosses threshold?
       |
   no -> normal routing to primary model pool
       |
   yes -> [ Degradation controller ] applies, in order of increasing severity:
       1. Cap max_tokens for new requests (reduces per-request cost/time)
       2. Reduce/disable expensive optional features (long-context mode, tool use, etc.)
       3. Route free/low-priority tier to a smaller fallback model pool
       4. Shed lowest-priority traffic entirely with a clear, fast error + retry-after
       |
       v
[ All degradation actions logged + alerted -- this is an incident, not silent behavior ]
       |
       v
[ Auto-recovery: as capacity signal drops below threshold, un-degrade in reverse order ]
```

**Degrade in graduated steps, not binary up-or-down**: capping `max_tokens` or disabling one expensive feature preserves service for far more users than an all-or-nothing shutoff, and each step sheds load at the smallest quality cost available.

**Tiered shedding by priority**: protect a paid tier longest and let free tiers absorb degradation first — a pre-agreed, documented business policy, not something improvised mid-incident, because whose requests get shed is a product decision. **Fall back to a smaller model rather than pure rejection where possible**, with product buy-in on which use cases tolerate a quality drop. **Make degradation visible internally**: every action fires an alert, because capacity running out is an operational incident worth investigating, not something that should silently self-heal. **Auto-recovery is symmetric**: un-degrade in reverse order as capacity recovers, with hysteresis so it does not flap at the threshold (AS7).

**Defending against pushback.** *"Isn't silently serving a worse model dishonest?"* — the alternative, everyone getting a slow or failing response, is worse for everyone including users who would have been fine with the smaller model; the fix is making the policy explicit and product-approved, such as surfacing "a faster, lighter response" in the UI. *"Why not just have enough capacity?"* — because "enough for the worst case at all times" is either prohibitively expensive or physically unavailable; quota limits and regional GPU shortages are real and recurring as of 2025-2026.

---

## 2025-2026 Landscape Quiz

*A senior interviewer will often probe whether you track the ecosystem or are reciting a 2023 blog post. The facts below were checked against current sources as of August 2026. Exact version numbers and benchmark ratios move fast — anything marked "verify before quoting" should be re-checked against current docs before you cite a specific number in an interview; the durable value here is the *shape* of the landscape, not the last digit of a version string.*

### LQ1: What are the main LLM inference engines in production use, and how do they position relative to each other?

**vLLM** is the dominant open-source general-purpose engine: continuous batching, PagedAttention, broad architecture coverage, an OpenAI-compatible server, and the "V1" engine architecture (rewritten scheduler and execution core, alpha from early 2025, matured through 2025-2026). It is the default for iterating across many model families.

**SGLang** is a fast-moving alternative with its own scheduler and runtime — RadixAttention for prefix caching — and strong results on structured-generation and high-concurrency workloads; now cited alongside vLLM rather than as a niche option. **TensorRT-LLM** gives compiled, hardware-specific kernels and is typically the throughput and latency ceiling on NVIDIA GPUs, at the cost of a build step and less architecture flexibility; increasingly consumed via Triton's backend or inside NVIDIA NIM. **Triton Inference Server** is the multi-framework serving layer (Ch. 11), hosting those engines behind one ops surface rather than competing as an engine. **NVIDIA NIM** is prebuilt containerized inference microservices packaging one of these behind a standard OpenAI-compatible API. *(Verify current NIM backend choices per model before quoting.)*

Version numbers and head-to-head benchmark ratios change roughly monthly: know the names and positioning, cite specifics as "as of my last check".

---

### LQ2: What's new about NVIDIA's current GPU generations relevant to inference (as of 2025-2026)?

**Hopper (H100/H200)** was widely deployed through 2024-2025; the H200 added significantly more HBM capacity and bandwidth than the H100, which directly helps KV-cache-bound serving. **Blackwell (B100/B200, GB200 NVL72)** is the current flagship, with major FP4/FP8 tensor-core throughput improvements and NVLink domain scale — GB200 NVL72 links many GPUs into one large NVLink domain.

**Blackwell Ultra (B300/GB300)** is a mid-cycle refresh with substantially more HBM3e per GPU, reported around the 288GB class. That matters because more per-GPU memory means fewer GPUs per replica for a given model plus KV-cache footprint, improving TP-degree economics. **Verify exact specs and availability before quoting** — this generation rolled out through late 2025 into 2026 and cloud availability was still expanding.

The trend: each generation's biggest inference-relevant win is memory capacity and bandwidth plus native low-precision (FP8/FP4) throughput, not raw FLOPs — both attack the two real bottlenecks, KV-cache memory and decode memory bandwidth.

---

### LQ3: What's Dynamic Resource Allocation (DRA) in Kubernetes, and what's its GA status?

DRA is the modern Kubernetes API for expressing complex device — especially GPU — allocation requirements: `DeviceClass` for what devices exist, `ResourceClaim`/`ResourceClaimTemplate` for what a pod needs, and CEL-based selection such as "a GPU with more than 40GB memory". It replaces the device-plugin model's all-or-nothing integer-count limitation and supports ranked fallback across GPU types, partitioned or shared access feeding MIG or time-slicing, and richer logic than a bare `nvidia.com/gpu: N` request.

**GA status**: DRA graduated to General Availability in Kubernetes v1.34, per the official Kubernetes project blog, with structured-parameter features and ecosystem tooling refined in following releases. **Verify the exact current minor version and feature maturity before quoting.** Interview framing: the old model, still very common in production, and DRA coexist — do not imply the old one is gone.

---

### LQ4: What are the current options for sharing one physical GPU across workloads on Kubernetes, ranked by isolation strength?

(See K3 above for the full comparison table.) Weakest to strongest: **time-slicing** (software round-robin, no memory isolation) → **MPS** (shared address space, better concurrent-kernel execution, still no hard memory isolation) → **MIG** (hardware-partitioned compute and memory slices, fixed profiles) → **DRA-orchestrated allocation** (flexible expression of any of the above, or whole-device claims, through a unified API).

The 2025-2026 trend consolidates around DRA as the *scheduling* layer on top of MIG, time-slicing, and MPS as the underlying *mechanisms*. DRA does not replace MIG; it makes MIG and the other mechanisms easier to request and compose with complex placement logic.

---

### LQ5: What's the current state of KV-cache offloading / cross-node cache sharing as a serving technique?

Tools like **LMCache** implement a KV-cache layer that offloads cache to CPU memory or NVMe/remote storage and shares it across multiple vLLM instances and nodes, extending prefix caching (V6) beyond one replica's GPU memory.

The motivation: prefix caching's benefit is capped by how much cache fits in one replica's GPU memory and how well the load balancer routes cache-sharing requests to the same replica. Offloading lets a much larger effective cache — system prompts, long-lived RAG contexts, long conversation histories — be reused across a whole fleet rather than staying replica-local and volatile.

This is an active area as of 2025-2026, with ongoing work standardizing KV-cache transfer and connector interfaces between prefill/decode-disaggregated setups (V8). Treat specific throughput multipliers as workload-dependent claims to verify.

---

### LQ6: What serving-platform / "inference-as-a-service" options exist besides self-hosting, and when would a senior engineer recommend one over self-hosting?

**NVIDIA NIM**: prebuilt optimized containers per model, self-hosted on your own GPUs with the engine-tuning done. **Managed inference endpoints** from clouds and model providers, dedicated or serverless: trade control and per-token cost optimization for operational simplicity. **Self-hosted on Kubernetes** with vLLM/SGLang/TensorRT-LLM/Triton: full control and best cost at scale, but you own everything in every chapter above.

Self-host when scale is large enough that infra cost and control outweigh the engineering investment — a rule of thumb some teams use is once GPU spend is large enough that a percentage point of utilization is a meaningful dollar figure. Use a managed platform when pre-scale, moving fast, or lacking platform-engineering capacity. It is a genuine build-versus-buy trade-off, not a purity contest.

---

### LQ7: Rapid-fire current-fact check — answer, then flag confidence.

GQA is standard across most current major open-weight model families — **high confidence, stable fact**. FP8 is a standard, well-supported inference precision on current NVIDIA data-center GPU generations — **high confidence, stable fact**.

Exact current vLLM/SGLang/TensorRT-LLM version numbers and head-to-head benchmark numbers — **verify before quoting, changes ~monthly**. Kubernetes DRA reached GA in v1.34 — **high confidence as of research date, but verify the Kubernetes version your target company actually runs, since a feature going GA does not mean every cluster has upgraded**. Specific GPU model availability and pricing, such as which cloud offers which Blackwell Ultra instance family — **verify before quoting, changes monthly**.

The general trend toward disaggregated prefill/decode serving and KV-cache-sharing infrastructure at the largest scale — **high confidence as a direction**, but treat any specific implementation's production-readiness claim as something to verify.

---
## Rapid-Fire Flashcards & Glossary

*One-liner Q->A pairs for the night before an interview. Organized by chapter. Skim top-to-bottom; if any answer doesn't come instantly, jump back to that chapter's deep-dive section above.*

### Fundamentals & Ch. 01 (Basic Serving)

| Q | A |
|---|---|
| Why is inference autoregressive? | Each new token depends on all previously generated tokens via attention. |
| What are the two inference phases? | Prefill (compute-bound, whole prompt) and decode (memory-bandwidth-bound, one token at a time). |
| Why cache K/V? | Avoids recomputing attention for every past token on every step. |
| What grows the KV cache? | Sequence length x batch size x layers x KV heads x head dim. |
| Why load the model once at startup? | Loading is slow (seconds-minutes); per-request loading would make every request pay that cost. |
| Sync or async for serving? | Async — most wall-clock time is spent awaiting the GPU, not doing CPU work. |
| What does streaming improve? | Perceived latency (TTFT) — user sees tokens as generated instead of waiting for the full response. |
| Liveness vs. readiness probe? | Liveness = process alive; readiness = model loaded and able to serve traffic. |
| Why can rollouts crash-loop on large models? | Liveness probe fires before a slow model load finishes — needs a startup probe. |
| What's the risk of a naive SIGTERM handler? | Drops in-flight/streaming requests instead of draining gracefully. |

### Ch. 02 (Docker)

| Q | A |
|---|---|
| Why not `python:slim` for GPU serving? | No CUDA/cuDNN userspace libraries matching the host driver. |
| devel vs. runtime CUDA image? | devel has compiler toolchain (nvcc); runtime is smaller, for running only. |
| Why multi-stage builds? | Compile with devel image, ship only the runtime image + artifacts. |
| Bake weights into the image or mount at runtime? | Usually mount at runtime — decouples model version from code/image version. |
| Top real-world Docker+GPU failure mode? | CUDA-version-vs-host-driver incompatibility. |
| Why pin exact dependency versions? | Unpinned installs cause silent ABI mismatches / behavior drift over time. |
| When does docker-compose stop being enough? | Multi-node scheduling, autoscaling, rolling updates, secrets at scale. |
| `nvidia-smi` works but `torch.cuda.is_available()` is False — likely cause? | CPU-only wheel installed instead of the CUDA-tagged build. |

### Performance Optimization

| Q | A |
|---|---|
| What is PagedAttention? | Block-based, non-contiguous KV cache allocation — like OS virtual memory paging. |
| What does continuous batching fix? | Static batching's "wait for slowest request" bubble; requests join/leave mid-batch. |
| FP16 -> FP8 -> INT4 quantization trend? | Each step roughly halves memory/increases speed, with increasing accuracy risk. |
| Latency vs. throughput lever? | Batch size — bigger batches raise throughput, raise per-request latency. |
| Right SLO framing? | Optimize throughput *at* a fixed latency SLO, not throughput in isolation. |

### Ch. 03 (Kubernetes)

| Q | A |
|---|---|
| Why must GPU `requests == limits`? | GPUs are an extended resource — no fractional/burstable allocation by default. |
| Default RollingUpdate problem for GPU pods? | Needs surge GPU capacity you may not have; slow, GPU-heavy pods make it worse. |
| Deployment or StatefulSet for serving pods? | Deployment, unless pods need stable identity/storage (e.g., multi-node TP rank-0). |
| Why taint GPU nodes? | Stops non-GPU workloads from occupying expensive GPU nodes. |
| What actually dominates GPU pod cold-start? | Node provisioning + image pull + weight download + engine warm-up, not scheduling. |
| MIG vs. time-slicing? | MIG = hardware-isolated partitions; time-slicing = software round-robin, no isolation. |
| What is DRA? | Dynamic Resource Allocation — flexible GPU allocation API, GA in Kubernetes v1.34. |
| Where should auth/rate-limiting live? | At the gateway layer, not inside the serving pod. |

### Ch. 04 (Load Testing)

| Q | A |
|---|---|
| Why is RPS a poor LLM load metric? | Requests vary hugely in cost by prompt/output length; concurrency matters more. |
| What's the "knee of the curve"? | The concurrency point where latency blows up while throughput plateaus. |
| TTFT vs. TPOT? | Time-to-first-token (prefill+queue) vs. time-per-output-token (decode step). |
| Why load test at concurrency, not just average load? | P99 tail latency only shows up under queueing pressure, not at low load. |
| What does a soak test catch that a burst test doesn't? | Memory leaks, KV-cache fragmentation, slow degradation over hours. |

### Ch. 05 (vLLM Internals)

| Q | A |
|---|---|
| What is chunked prefill for? | Splits long prompts' prefill into chunks so it doesn't stall other requests' decode. |
| What does the vLLM scheduler decide each iteration? | Which sequences run a step, within a KV-cache-block and token budget. |
| When does speculative decoding help most? | Memory-bandwidth-bound decode with a draft model that's frequently correct. |
| TP vs. PP? | TP shards each layer (needs NVLink, stays in-node); PP splits layers across stages (tolerates slower links, spans nodes). |
| What is prefix caching? | Reusing KV-cache blocks across requests sharing an identical prompt prefix. |
| What changed with vLLM's "V1" engine? | Core scheduler/execution rewrite (alpha ~early 2025) for lower overhead, broader model support. |
| What is disaggregated prefill/decode? | Running prefill and decode on separate GPU pools, transferring KV cache between them. |

### Ch. 06 (Autoscaling)

| Q | A |
|---|---|
| Why is CPU util a bad HPA signal here? | Decoupled from GPU/KV-cache load — a GPU-saturated pod can show 10% CPU. |
| Better HPA/KEDA signal? | Queue depth (requests waiting for a scheduler slot). |
| Why is GPU scale-to-zero hard? | Cold start (minutes) makes the first post-idle request violate any real SLO. |
| Two layers of GPU autoscaling latency? | Pod-level HPA + node-level cluster autoscaler — latencies stack, don't overlap. |
| What is predictive/scheduled scaling for? | Pre-warming capacity ahead of known/forecastable demand patterns. |
| Why is HPA flapping worse for GPU pods? | Removed replicas may need node re-provisioning + weight re-download to come back. |

### Ch. 07 (Canary Deployments)

| Q | A |
|---|---|
| Canary vs. blue-green vs. shadow? | Canary = small real-traffic %; blue-green = full cutover; shadow = copy traffic, discard output. |
| LLM-specific canary red flag? | Output-quality regression — invisible to latency/error metrics alone. |
| Why sticky-session routing during canary? | Multi-turn chats must not switch model versions mid-conversation. |
| Mesh-based split vs. replica-ratio split? | Mesh decouples traffic % from replica count; replica-ratio ties them together. |
| What should trip auto-rollback? | Guardrail metric breach over a rolling window with a minimum sample size. |

### Ch. 08 (Monitoring)

| Q | A |
|---|---|
| Four golden signals for LLM serving? | Latency (TTFT+inter-token), traffic (req/s+tokens/s), errors, saturation (GPU+KV-cache+queue). |
| #1 cause of Prometheus falling over? | Cardinality explosion from high-cardinality labels (e.g., raw request ID). |
| Better alerting than static thresholds? | SLO burn-rate, multi-window multi-burn-rate alerting. |
| What does tracing add beyond metrics? | Per-request breakdown of *where* time went across hops. |
| How do you catch quality regressions continuously? | Sample live traffic through automated quality checks / LLM-as-judge, not just at canary time. |

### Ch. 09 (Model Versioning)

| Q | A |
|---|---|
| Why immutable model artifacts? | Makes rollback and post-hoc debugging trustworthy. |
| What decouples code version from model version? | Runtime weight loading via config (env var/ConfigMap), not baked-in weights. |
| What determines real rollback speed? | Whether the previous version's pods are still warm, or need cold re-provisioning. |
| What else needs versioning besides weights? | Prompt templates and RAG retrieval config — they shift output just as much. |
| What does shadow deployment validate that offline eval can't? | Behavior on today's live traffic distribution, not a historical dataset. |

### Ch. 10 (Drift Detection)

| Q | A |
|---|---|
| Three types of drift? | Data drift (input), concept drift (input-output relationship), prediction drift (output). |
| How do you measure drift on free text? | Convert to numeric proxies first — embeddings, length, topic distribution — then PSI/KS/MMD. |
| Why is fully automatic drift-triggered retraining risky? | Feedback loops that amplify the very drift that triggered it. |
| Why maintain per-tenant drift baselines? | A global baseline shows false "drift" from normal tenant-mix shifts. |
| Biggest blind spot of PSI/KS alone for LLMs? | They don't capture output-quality regressions — need LLM-as-judge/guardrail signals too. |

### Ch. 11 (Triton)

| Q | A |
|---|---|
| What does Triton add over a hand-rolled server? | Multi-framework serving, dynamic batching, model repository/versioning, built-in metrics. |
| Dynamic batching vs. continuous batching? | Dynamic batches whole fixed-shape requests; continuous batches at the decode-step level for autoregressive generation. |
| What's an ensemble model? | A DAG of models/steps (e.g., embed->retrieve->generate) served as one logical model. |
| What do "instance groups" control? | How many copies of a model run, and on which GPU(s), within one Triton process. |
| When is Triton overkill? | Single model, single framework, no need for its ensemble/multi-version features. |

---

### Glossary

| Term | Definition |
|---|---|
| **Autoregressive generation** | Generating output one token at a time, each conditioned on all previous tokens. |
| **KV cache** | Stored attention key/value tensors for previously processed tokens, reused to avoid recomputation. |
| **Prefill** | The compute-bound phase processing the entire input prompt in one pass. |
| **Decode** | The memory-bandwidth-bound phase generating one output token per step. |
| **TTFT** | Time to first token — latency from request start to the first generated token. |
| **TPOT / inter-token latency** | Time per output token during decode. |
| **PagedAttention** | vLLM's block-based, non-contiguous KV-cache memory management technique. |
| **Continuous batching** | Iteration-level batching where requests join/leave a running batch mid-generation. |
| **Chunked prefill** | Splitting a long prompt's prefill across multiple scheduler iterations to avoid blocking decode. |
| **Prefix caching** | Reusing KV-cache blocks across requests sharing an identical prompt prefix. |
| **Speculative decoding** | Using a small draft model to propose multiple tokens, verified in one batched pass by the target model. |
| **Tensor parallelism (TP)** | Sharding each layer's weights across GPUs, needs high-bandwidth interconnect. |
| **Pipeline parallelism (PP)** | Splitting model layers sequentially across GPUs/nodes, tolerates slower interconnect. |
| **Data parallelism** | Replicating the full model across GPUs/nodes, each serving different requests. |
| **GQA (Grouped-Query Attention)** | Attention variant sharing a handful of KV heads across query heads to shrink KV cache. |
| **MQA (Multi-Query Attention)** | Attention variant with a single shared KV head. |
| **MLA (Multi-head Latent Attention)** | DeepSeek-style attention compressing KV into a low-rank latent to shrink cache further. |
| **Quantization** | Reducing numeric precision of weights/activations (FP16/FP8/INT8/INT4) to save memory/speed up compute. |
| **FP8** | 8-bit floating point precision, native on Hopper/Blackwell-class tensor cores. |
| **Disaggregated prefill/decode** | Running prefill and decode phases on separate GPU pools with KV-cache transfer between them. |
| **HPA (Horizontal Pod Autoscaler)** | Kubernetes controller that scales replica count based on metrics. |
| **KEDA** | Kubernetes Event-Driven Autoscaling — HPA extension supporting external/custom metric sources. |
| **DRA (Dynamic Resource Allocation)** | Kubernetes API (GA in v1.34) for flexible, claim-based device/GPU allocation. |
| **MIG (Multi-Instance GPU)** | NVIDIA hardware feature partitioning one GPU into isolated compute/memory slices. |
| **MPS (Multi-Process Service)** | NVIDIA feature allowing concurrent kernel execution from multiple processes on one GPU. |
| **Time-slicing** | Software round-robin sharing of one GPU across pods, no memory isolation. |
| **Cluster autoscaler / Karpenter** | Node-level autoscaling that provisions/removes nodes based on unschedulable pod pressure. |
| **PriorityClass / preemption** | Kubernetes mechanism to evict lower-priority pods to make room for higher-priority ones. |
| **Canary deployment** | Gradual traffic-weighted rollout of a new version alongside the stable one. |
| **Shadow (dark) traffic** | Sending a copy of live traffic to a new version without exposing its output to users. |
| **Blue-green deployment** | Full, instant traffic cutover between two fully-provisioned versions. |
| **Sticky routing** | Routing all requests of one session/conversation consistently to the same backend version. |
| **Model registry** | System of record for versioned model artifacts and their metadata/lineage/eval results. |
| **Immutable versioning** | Practice of never overwriting a published model version, only publishing new ones. |
| **Data drift** | Change in the distribution of production inputs vs. a reference distribution. |
| **Concept drift** | Change in the true input-output relationship over time. |
| **Prediction drift** | Change in the distribution of model outputs over time. |
| **PSI (Population Stability Index)** | Statistical measure comparing two distributions, common drift-detection metric. |
| **Triton model repository** | Directory-based structure where models and their versions/configs are deployed to Triton. |
| **Dynamic batching (Triton)** | Triton's general-purpose request-batching feature for fixed-shape, single-pass models. |
| **Ensemble model (Triton)** | A DAG-defined pipeline of models/steps served as one logical Triton model. |
| **Instance group (Triton)** | Configuration controlling how many copies of a model run, and on which GPU(s). |
| **TensorRT-LLM** | NVIDIA's compiled, hardware-optimized LLM inference engine. |
| **NVIDIA NIM** | Prebuilt containerized inference microservices packaging an optimized engine per model. |
| **SGLang** | An open-source LLM serving engine/runtime, notable for RadixAttention-based prefix caching. |
| **LMCache** | KV-cache offload/sharing layer extending prefix caching across replicas/nodes and to CPU/NVMe. |
| **Golden signals** | Latency, traffic, errors, saturation — the standard SRE monitoring framework. |
| **SLO burn rate** | Rate at which an error/latency budget is being consumed, used for smarter alerting. |
| **Cardinality (metrics)** | Number of unique label-value combinations for a metric; high cardinality can overload Prometheus. |
| **Cold start (GPU)** | Latency from "need capacity" to "capacity actually serving," dominated by node/image/weight provisioning. |
| **Warm buffer / warm pool** | Standing spare capacity kept ready to absorb bursts faster than reactive autoscaling can react. |
| **Graceful degradation** | Serving reduced-quality/reduced-cost responses under load rather than failing outright. |
| **LLM-as-judge** | Using a (usually cheaper) LLM to automatically score another model's outputs for quality/regressions. |

---

## Traps & How to Recover

*Common wrong answers/misconceptions that sound plausible but signal shallow experience to a senior interviewer — and the reframe that recovers the answer.*

### Trap 1: "We'd just add more GPUs to fix latency/throughput problems."

**Why it's wrong:** treats hardware as the first lever instead of the last one. A senior interviewer hears this as "hasn't actually diagnosed a real bottleneck before."
> **Say it.** "First I'd check GPU/KV-cache utilization and batching configuration — low utilization under load means a scheduling/batching problem, not a hardware problem. I'd only add GPUs after confirming the engine is already using the ones it has efficiently."

### Trap 2: "vLLM is always faster than everything else, so just use vLLM."

**Why it's wrong:** treats engine choice as a fixed ranking instead of a workload-dependent trade-off (TensorRT-LLM often wins on raw throughput for stable, high-volume models; Triton wins for multi-framework needs; SGLang is a real, current alternative).
> **Say it.** "vLLM is my default for iteration speed and broad model support, but for a small number of stable, extremely high-volume models I'd benchmark TensorRT-LLM, and if we're serving heterogeneous frameworks I'd put Triton in front regardless of engine choice."

### Trap 3: "We'll just use HPA on CPU utilization like any other service."

**Why it's wrong:** CPU utilization is decoupled from GPU/KV-cache load for LLM serving pods (AS1) — this is one of the fastest ways to reveal you haven't actually run GPU workloads in Kubernetes.
> **Say it.** "For GPU-bound serving I'd scale on a custom metric — queue depth or GPU/KV-cache utilization via the Prometheus Adapter or KEDA — CPU utilization on these pods is nearly meaningless."

### Trap 4: "Scale to zero when idle to save cost."

**Why it's wrong:** ignores GPU cold-start reality (minutes, not seconds) — the first request after scale-to-zero will badly violate almost any latency SLO.
> **Say it.** "For latency-sensitive paths I'd keep a warm minimum and rely on predictive/scheduled scaling for known low-traffic windows instead — true scale-to-zero only for genuinely latency-insensitive batch workloads."

### Trap 5: "Canary just means routing 10% of traffic to the new version and watching error rate."

**Why it's wrong:** misses that LLM canaries need output-quality signals (CD2) — a model can be infra-healthy and behaviorally regressed simultaneously, which pure error-rate/latency monitoring won't catch.
> **Say it.** "Infra metrics are necessary but not sufficient — I'd add an automated quality-eval sampling step on canary outputs (guardrail-trigger rate, structured-output validity, an LLM-as-judge score) before promoting."

### Trap 6: "Just quantize to INT4 everywhere for cost savings — precision doesn't really matter for chat."

**Why it's wrong:** asserts a blanket accuracy claim without task-specific validation; INT4 accuracy risk varies a lot by task (reasoning-heavy tasks degrade more than casual chat), and "doesn't matter" is exactly the unvalidated assumption that gets someone burned in production.
> **Say it.** "I'd start from FP8 as the likely sweet spot on current hardware, and only push to INT4 after validating on our actual eval suite for our actual task mix — not assume it's fine."

### Trap 7: "Model drift means we should just retrain automatically whenever drift is detected."

**Why it's wrong:** ignores feedback-loop risk (DR3) — automatic retraining on drifted/production data without human review can amplify the very drift it's reacting to.
> **Say it.** "I'd have drift detection trigger an alert and investigation, with retraining as a human-gated decision informed by that investigation — not a fully automatic pipeline straight to production."

### Trap 8: "Kubernetes handles GPU sharing out of the box, just request `nvidia.com/gpu: 0.5`."

**Why it's wrong:** factually incorrect — GPUs are an integer-only extended resource without MIG/time-slicing/MPS/DRA configured; a fractional request simply won't work (K1, K3).
> **Say it.** "GPU requests are whole-unit by default — to share one GPU across pods I'd configure MIG for hard isolation, or time-slicing/MPS for looser sharing, potentially orchestrated via DRA."

### Trap 9: "We don't need a model registry, we'll just tag Docker images with the model version."

**Why it's wrong:** conflates code-release cadence with model-release cadence, and loses model-specific metadata (eval scores, lineage, approval status) a registry provides natively (MV7).
> **Say it.** "For infrequent updates on one model, tagged images might be enough — but once we have frequent fine-tunes or multiple models needing independent rollback, I'd want an actual registry decoupled from the application image."

### Trap 10: "PSI/KL-divergence tests are all we need for drift detection on an LLM."

**Why it's wrong:** these classical statistical tests need a numeric feature to compare and say nothing about output *quality* directly (DR7) — a real blind spot for generative systems.
> **Say it.** "Statistical tests on derived features (embeddings, length) are useful for continuous monitoring, but I'd pair them with output-quality-eval sampling — that's the ground-truth signal the statistical tests are only ever a proxy for."

### Trap 11: "Just cache all the responses to save GPU cost."

**Why it's wrong:** treats response caching as a general LLM cost solution when it only helps for repeated/identical content — most production traffic (unique user prompts) won't hit a response cache at all; the real high-leverage cache is prefix/KV-cache reuse of shared context (V6), not full-response caching.
> **Say it.** "Full-response caching only helps for genuinely repeated queries. The bigger lever is prefix caching — reusing KV cache for shared system prompts or RAG context — which helps a much larger share of realistic traffic."

### Trap 12: "Zero downtime means using Kubernetes' rolling update strategy for model version swaps."

**Why it's wrong:** conflates a Deployment-level rolling update (for code/image changes) with a canary/traffic-weighted rollout (for model version changes, Scenario 3) — a rolling update replaces pods, it doesn't give you a controlled, metric-gated traffic ramp or an instant rollback via traffic-weight change.
> **Say it.** "For a model version change specifically, I'd use a canary pipeline with traffic-weight shifting rather than a pod-replacement rolling update — that gives instant rollback and gradual, metric-gated exposure, which a rolling update doesn't."

### Trap 13: "More replicas is always better for availability."

**Why it's wrong:** ignores that each GPU replica is expensive and that availability comes from the *right* redundancy (spread across zones/nodes, N+1 for failure tolerance) not raw count — over-provisioning replicas "for availability" without a specific failure scenario in mind is just wasted spend.
> **Say it.** "I'd size replica count from load-test-derived capacity plus explicit failure-tolerance headroom (N+1, spread across zones) — not an arbitrary 'more is safer' buffer."

### Trap 14: "Streaming and batching are in tension, so pick one."

**Why it's wrong:** conflates continuous batching (an engine-internal scheduling technique) with streaming (a client-facing response-delivery mechanism) — they're not in tension; every major serving engine streams tokens from a continuously-batched decode loop simultaneously for many requests.
> **Say it.** "They operate at different layers — continuous batching is how the engine schedules GPU work across concurrent requests; streaming is how each request's tokens are delivered to its client as they're produced. Production systems do both together."

---

### Red Flags vs. Green Flags — Master Table

| Topic | Red flag answer | Green flag answer |
|---|---|---|
| Latency problem | "Add more GPUs." | "Check GPU/KV-cache utilization and batching config first; add GPUs only if already utilization-bound." |
| Autoscaling signal | "Scale on CPU like any service." | "Scale on queue depth / GPU-KV-cache utilization via KEDA or the Prometheus Adapter." |
| Scale-to-zero | "Always scale to zero when idle." | "Keep a warm minimum for latency-sensitive paths; scale-to-zero only for batch/insensitive workloads." |
| Canary success criteria | "Error rate and latency look fine, ship it." | "Also check output-quality/guardrail signals before promoting." |
| Quantization | "INT4 everywhere, precision doesn't matter." | "Start from FP8, validate task-specific quality before going further." |
| Drift response | "Auto-retrain the moment drift is detected." | "Alert + investigate, human-gated retraining decision." |
| GPU sharing | "Request 0.5 GPU." | "Configure MIG/time-slicing/MPS/DRA explicitly — GPUs are integer-only by default." |
| Model release process | "Just tag a Docker image per model version." | "Weigh registry vs. tagged-images trade-off based on update frequency and rollback needs." |
| Cost optimization | "Cache all responses." | "Prioritize prefix/KV-cache reuse, quantization, and utilization — full-response caching only helps repeated queries." |
| Model version rollout | "Rolling update the Deployment." | "Traffic-weighted canary rollout with sticky sessions and a fast-rollback warm window." |
| Availability | "More replicas, always." | "Size from load-test capacity plus explicit N+1/zone-spread failure tolerance." |
| Streaming vs. batching | "Pick one." | "They're orthogonal — continuous batching (engine) and streaming (client delivery) work together." |
| Ecosystem knowledge | Asserts exact version numbers/benchmarks with false confidence. | Names current tools/trends correctly, flags fast-moving specifics as "verify before quoting." |
| Drift detection method | "PSI/KS on raw text." | "Convert to numeric proxies (embeddings, length) first; pair with output-quality-eval sampling." |
| GPU generation choice | "Biggest GPU count always wins." | "Right-size TP degree and quantization against actual concurrency/memory need." |

---

## Tips for Interviews

1. **Be specific**: use numbers and examples.
2. **Show trade-offs**: understand pros/cons, and say them out loud even when not asked.
3. **Think system-wide**: consider all components, not just the one the question named.
4. **Ask clarifying questions**: understand requirements (scale, SLO, budget) before designing.
5. **Draw diagrams**: visualize architecture — even a rough ASCII sketch on a whiteboard shows structured thinking.
6. **Discuss monitoring**: always mention observability — a design without it is incomplete to a senior interviewer.
7. **Talk about failures**: how to handle edge cases, degradation, and rollback, not just the happy path.
8. **Flag what's fast-moving**: for current tool/version/benchmark facts, it's a stronger answer to say "verify the exact number, but the shape is X" than to assert a specific figure with false confidence.
9. **Name the trade-off explicitly, don't just pick a side**: "I'd default to X, but Y is the better choice if Z" reads as more senior than a flat "always use X."

---

## Resources

- This repository's chapter READMEs and deep-dive docs (`01_basic_serving` through `11_triton`).
- [vLLM documentation and blog](https://docs.vllm.ai/) — engine internals, release notes, and architecture posts (e.g., the V1 engine and "Inside vLLM" posts).
- [Kubernetes documentation](https://kubernetes.io/docs/) — see the GPU/device-plugin and Dynamic Resource Allocation pages for current GPU scheduling capabilities.
- [NVIDIA Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/) and [GitHub repo](https://github.com/triton-inference-server/server).
- [NVIDIA developer documentation on NIM](https://developer.nvidia.com/nim) and the GPU Operator/MIG/DRA integration docs.
- Prometheus/Grafana guides — for the golden-signals and SLO-burn-rate alerting patterns referenced throughout Ch. 08.
- [Evidently AI documentation](https://docs.evidentlyai.com/) — drift detection tooling referenced in Ch. 10.
- [LMCache documentation](https://docs.lmcache.ai/) — KV-cache offloading/sharing referenced in the Landscape Quiz.
- Argo Rollouts / Flagger documentation — automated canary analysis and promotion referenced in Ch. 07.

**A note on currency:** the 2025-2026 Landscape Quiz section captures the state of the ecosystem as researched in August 2026. Inference engines, GPU generations, and Kubernetes GPU-scheduling features move fast — before an interview, spend 15 minutes checking the current release notes of whichever engine/platform the job description mentions by name.

Good luck with your interviews!
