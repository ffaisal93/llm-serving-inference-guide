# LLM Serving & Inference Interview Q&A

Comprehensive interview questions and answers for LLM serving, inference, and MLOps roles — written to help someone convince a **senior interviewer** they can build and operate real LLM inference infrastructure, not just call an API.

Each themed section below maps to one of this guide's 11 hands-on chapters. Use the chapter deep-dive sections to go deep on implementation details, the System Design section to rehearse whiteboard scenarios end-to-end, and the Flashcards/Traps sections the night before an interview for rapid review.

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

**Answer:**
1. **Tokenization**: Input text is converted to token IDs using the model's tokenizer.
2. **Embedding**: Token IDs are converted to dense vectors (embeddings).
3. **Forward Pass**:
   - Input passes through transformer layers.
   - Each layer applies self-attention and a feed-forward network.
   - Attention computes relationships between tokens.
4. **Output Projection**: The final layer projects hidden states to vocabulary-size logits.
5. **Sampling**: The next token is sampled from the resulting probability distribution (greedy, top-k, top-p/nucleus, temperature).
6. **Autoregressive Generation**: The process repeats with the new token appended until a stop condition (EOS token, max length, stop string) is hit.

**Key Points:**
- Inference is autoregressive: one token at a time, each depending on all previous tokens.
- KV cache stores attention key/value tensors to avoid recomputing them for every past token.
- There are two distinct phases with very different compute profiles: **prefill** (process the whole prompt, compute-bound, highly parallel) and **decode** (generate one token at a time, memory-bandwidth-bound). This split is the basis of most modern serving-engine design decisions (continuous batching, chunked prefill, disaggregated prefill/decode).

---

### Q2: What is KV caching and why is it important?

**Answer:**
KV (Key-Value) caching stores the attention key and value matrices for previously processed tokens so they don't need to be recomputed.

**How it works:**
- First token / prefill: compute full Q, K, V for every prompt token.
- Subsequent tokens / decode: compute Q only for the new token, reuse cached K, V for everything before it.

**Benefits:**
- **Speed**: avoids recomputing attention for every prior token on every step — turns an O(n²) per-step cost into an O(n) per-step cost.
- **Memory trade-off**: uses substantially more GPU memory (often the single largest consumer of GPU memory at high concurrency) but is dramatically faster.
- **Critical for throughput**: without it, generating each new token would require reprocessing the entire prefix.

**Example:**
- Without cache: generating 100 tokens = 100 forward passes, each reprocessing up to 100 tokens.
- With cache: generating 100 tokens = 100 forward passes, but each only processes 1 new token against the cached prefix.

**Sizing math a senior interviewer expects you to know:**
KV cache size per token ≈ `2 (K and V) × num_layers × num_kv_heads × head_dim × dtype_bytes`. For a 70B-class dense model with grouped-query attention this is still megabytes per token at long context, which is why KV cache — not model weights — is usually the binding memory constraint at high concurrency and long context.

---

### Q3: What is the difference between training and inference?

**Answer:**

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Mode** | Training mode (gradients computed) | Evaluation mode (no gradients) |
| **Batch** | Large, fixed batches (32-128+) | Small/variable batches, single requests |
| **Memory** | Stores activations for backprop + optimizer state | Only forward-pass activations + KV cache |
| **Speed** | Slower per step (backprop overhead) | Faster per step (forward only) |
| **Optimization target** | Loss minimization via gradient descent | Latency / throughput / cost per token |
| **Hardware** | Multi-GPU clusters, high-bandwidth interconnect | Anything from one GPU to multi-node clusters, but often shared/multi-tenant |

**Key differences:**
- Training updates weights and needs gradients + optimizer state (2-3x model memory in FP32 Adam state alone).
- Inference uses fixed weights and is dominated by KV cache growth and request-arrival variability rather than a fixed batch shape.

---

### Q4: Explain the attention mechanism in the context of inference.

**Answer:**
Attention determines which tokens to focus on when generating the next token.

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) x V
```

**Components:**
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What information do I have?"
- **V (Value)**: "What is the actual information?"

**In inference:**
- Computes relationships between the current token and all previous tokens.
- Allows the model to "attend" to relevant context.
- KV cache stores K and V to avoid recomputation.

**Complexity:**
- Naive attention is \( O(n^2) \) in sequence length \( n \) for compute and memory of the attention matrix; this is why long context is expensive.
- Modern serving stacks avoid materializing the full \( n \times n \) matrix (FlashAttention-style fused kernels) and avoid wasting memory on it (PagedAttention-style block-based KV cache — see the vLLM section).
- **Variants worth naming**: Multi-Head Attention (MHA) → Multi-Query Attention (MQA, one KV head) → Grouped-Query Attention (GQA, a handful of KV heads shared across query heads — used by Llama 2/3, Mistral, most current models) → Multi-head Latent Attention (MLA, DeepSeek-V2/V3, compresses KV into a low-rank latent to shrink cache further). GQA/MLA exist specifically to shrink the KV cache, which is the actual inference bottleneck, not to save training compute.

---

## Model Serving & Basic Serving Patterns (Ch. 01)

*Maps to Chapter 01 — Basic Serving (`app.py`, `model_loader.py`, `test_api.py`).*

### Q5: How would you design an LLM serving API?

**Answer:**

**API design:**
```python
POST /v1/completions
{
  "prompt": "The future of AI is",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Key components:**
1. **Request validation**: validate inputs (prompt length, parameter ranges) before touching the GPU.
2. **Model loading**: load the model once at process startup, not per request.
3. **Tokenization**: convert text to tokens.
4. **Generation**: run inference with the requested sampling parameters.
5. **Response**: return generated text plus metadata (latency, token counts, finish reason).

**Considerations:**
- **Async support**: handle concurrent requests without blocking the event loop on CPU-bound tokenization.
- **Streaming**: support Server-Sent Events / chunked responses so users see tokens as they're generated instead of waiting for the full completion.
- **Error handling**: graceful, typed error responses (400 for bad input, 429 for rate limiting/overload, 503 for draining).
- **Rate limiting**: protect the GPU from being overwhelmed; reject early and cheaply rather than queueing forever.
- **Health checks**: separate liveness (process is alive) from readiness (model loaded and able to serve) for Kubernetes probes.

---

### Q5b: Why must the model be loaded once at startup instead of per-request, and what does "model_loader.py as a singleton" actually buy you?

**Answer:**
- Loading a multi-GB/multi-tens-of-GB checkpoint from disk to GPU memory takes seconds to minutes; doing it per request would make every request pay that cost.
- A singleton loader (module-level global or a class instantiated once in an app lifespan/startup hook) guarantees the weights are loaded exactly once per process and shared across all requests handled by that worker.
- It also lets you fail fast at startup (crash the pod before it's marked ready) instead of failing on the first user request — this is exactly why the readiness probe should depend on "model loaded", not just "HTTP server up".
- Trap to avoid: loading the model inside a request handler "for testing" and forgetting to guard it — this silently reloads on every request in dev and can ship that way.

---

### Q5c: Sync (Flask/gunicorn workers) vs async (FastAPI/uvicorn) for a basic LLM serving app — which do you pick and why?

**Answer:**
- Python's GIL means CPU-bound work (tokenization, sampling) doesn't parallelize across threads within one process; the actual GPU forward pass releases the GIL while inside C/CUDA calls, but request handling (JSON parsing, validation, logging) doesn't.
- **Async (FastAPI + uvicorn/uvloop)** is preferred for LLM serving because most of a request's *wall-clock* time is spent awaiting the GPU (via an inference engine's async generate call), not doing Python work — async lets one process juggle hundreds of in-flight requests without one-thread-per-request overhead, and it's the model every major serving engine (vLLM's OpenAI-compatible server, TGI) actually uses.
- **Sync workers** (multiple gunicorn processes) are simpler to reason about but each process typically needs its own model copy loaded onto the GPU, which usually doesn't fit — so for anything beyond a toy demo, one async process per GPU (or per set of GPUs under tensor parallelism) talking to a single inference engine instance is the standard pattern, with process-level scaling handled by Kubernetes replicas, not gunicorn workers.

---

### Q5d: How do you support streaming responses, and why does it matter for LLM UX?

**Answer:**
- Use Server-Sent Events (`text/event-stream`) or chunked transfer encoding; the engine yields tokens as they're generated and the HTTP layer flushes each chunk immediately instead of buffering the full response.
- Time-to-first-token (TTFT) becomes the metric users actually feel — streaming lets a user start reading after ~prefill time instead of after prefill + full decode, which for a 500-token answer can be the difference between "instant" and multi-second perceived latency.
- Implementation detail: your framework and any reverse proxy/load balancer in front of it must not buffer the response (disable proxy buffering, watch out for gzip compression interacting badly with streaming, make sure client timeouts are long enough for slow generations).

---

### Q5e: Design the health check strategy for a basic LLM serving pod.

**Answer:**
- **Liveness probe**: "is the process alive and not deadlocked?" — a cheap endpoint (`/healthz`) that doesn't touch the model. If it fails repeatedly, Kubernetes restarts the container.
- **Readiness probe**: "can this pod actually serve traffic right now?" — checks the model is loaded, and ideally that the engine isn't in an unrecoverable state (e.g., OOM-crash-looping internally). If it fails, the pod is pulled from the Service's endpoints but *not* restarted — important, because restarting a pod that's just temporarily overloaded is the wrong reaction.
- **Startup probe**: for large models, loading can take minutes; a startup probe with a generous `failureThreshold`/`periodSeconds` prevents the liveness probe from killing the pod mid-load (a very common misconfiguration that causes crash-loop on large-model pods).
- Anti-pattern: making readiness depend on GPU utilization or queue depth — that turns transient load into pod eviction and can create a thundering-herd death spiral as remaining pods take the removed pod's traffic and also fail readiness.

---

### Q5f: How do you validate and sanitize LLM request inputs safely?

**Answer:**
- Use a schema library (Pydantic in Python) to enforce types and bounds: max prompt length, max_tokens ceiling (protects against a client requesting 1,000,000 tokens and pinning a GPU for minutes), valid ranges for temperature/top_p.
- Reject before tokenization/generation — validation should be nearly free; the whole point is to fail cheap requests cheaply instead of paying GPU cost to reject them.
- Sanitize for prompt-injection-adjacent concerns if the service composes user input into a larger system prompt (escape or clearly delimit user content; don't trust the model to self-police).
- Cap concurrent max_tokens exposure: a request asking for a huge generation with a huge batch multiplies memory pressure; many production gateways cap max_tokens per plan/tier.

---

### Q5g: What does "graceful shutdown" mean for an LLM serving pod, and why do naive implementations lose requests?

**Answer:**
- On `SIGTERM` (which Kubernetes sends before killing a pod during a rollout, scale-down, or preemption), the process should: stop accepting new requests, finish in-flight generations (or return them cleanly), then exit — within the `terminationGracePeriodSeconds` window.
- Naive implementations just die immediately on SIGTERM, silently dropping any request mid-generation — with streaming responses this is especially visible to users (a truncated answer).
- Because a single decode step can take a while and a full generation longer, `terminationGracePeriodSeconds` for LLM pods often needs to be much larger than the Kubernetes default (30s) — tens of seconds to a couple of minutes depending on max_tokens and concurrency.
- Combine with readiness: flip readiness to false the instant SIGTERM is received (removes the pod from the Service immediately) while continuing to drain existing connections — this is the standard "stop taking new work, finish existing work" pattern.

---

### Q5h: Your `/v1/completions` endpoint works for one user in a demo but falls over with 50 concurrent users on one GPU. Diagnose it.

**Answer:**
- Most likely cause: the demo used a naive one-request-at-a-time generation loop (e.g., raw HuggingFace `.generate()` synchronously per request) with no batching — at 50 concurrent users each waits for a full serial turn, so latency scales linearly with load instead of leveraging the GPU's actual parallel capacity.
- Fix path: put a real inference engine (vLLM, TGI, or Triton) in front that does continuous batching so the GPU processes many requests' decode steps together each iteration.
- Secondary causes worth checking: no queueing/backpressure (requests pile up in-process instead of being admitted/rejected predictably), no max_tokens cap (a few huge requests starve everyone else), and single-process serving with no way to add a second GPU/replica.
- What a senior interviewer wants to hear: you diagnose by looking at GPU utilization (low util under load = batching problem, not a hardware problem) before reaching for "add more GPUs."

---

## Docker & Containerization (Ch. 02)

*Maps to Chapter 02 — Docker (`Dockerfile.basic`, `Dockerfile.gpu`, `Dockerfile.optimized`, `docker-compose.yml`).*

### D1: Why does an LLM serving Dockerfile need to look different from a typical Python web app Dockerfile?

**Answer:**
- **Base image**: needs CUDA/cuDNN userspace libraries matching the driver on the host (`nvidia/cuda:...-runtime` or a framework image like `vllm/vllm-openai`), not a plain `python:3.x-slim`.
- **GPU runtime**: the container needs `--gpus all` (Docker) or the NVIDIA Container Toolkit / `nvidia.com/gpu` device plugin (Kubernetes) wired up — without it the container can see the CUDA libraries but not the physical device.
- **Image size**: CUDA + PyTorch + model-serving deps can easily be 5-15GB; this matters a lot more than for a typical web app because it directly affects pod start time and node scale-up time (large images take longer to pull on a fresh node — a real problem during autoscaling bursts).
- **Weights**: model weights (often tens of GB) are usually *not* baked into the image — they're pulled at startup from object storage/a model registry into a mounted volume, so the image itself stays reasonably sized and swapping model versions doesn't require a rebuild.

---

### D2: Explain the difference between `Dockerfile.basic`, `Dockerfile.gpu`, and `Dockerfile.optimized` patterns.

**Answer:**
- **Basic**: CPU-only, plain Python base image — good for local dev/testing of API logic without needing a GPU, bad for anything latency-sensitive.
- **GPU**: CUDA base image, GPU-enabled framework wheels (`torch` with CUDA build, `vllm`), correctly pins CUDA/driver-compatible versions — this is the one that actually runs the model at production speed.
- **Optimized**: same as GPU but applies image-engineering best practices: multi-stage build (build deps discarded from final image), layer ordering so rarely-changing layers (CUDA base, system deps) come before frequently-changing ones (app code) to maximize cache hits, `.dockerignore` to avoid shipping local caches/checkpoints accidentally, and often a slimmer runtime-only CUDA base instead of the full devel image.

---

### D3: Walk through multi-stage builds for an LLM serving image and why they matter here specifically.

**Answer:**
```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder
RUN pip install --user vllm torch
# compile any custom kernels here

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
COPY --from=builder /root/.local /root/.local
COPY app.py .
CMD ["python", "app.py"]
```
- The `devel` image (has nvcc, headers) is only needed to *build* custom CUDA kernels or compile packages from source; it's ~2-3x larger than the `runtime` image.
- Shipping the `devel` image to production wastes registry storage, pull time, and attack surface for no benefit once the build artifacts exist.
- Multi-stage builds let you compile with the heavy toolchain and ship only the runtime image with the compiled artifacts copied over.

---

### D4: How do you handle GPU access and driver compatibility in Docker?

**Answer:**
- Install the **NVIDIA Container Toolkit** on the host so the container runtime can expose GPU devices.
- Run with `docker run --gpus all ...` (or `runtime: nvidia` in `docker-compose.yml`).
- **Version compatibility is the #1 real-world failure mode**: the CUDA version inside the container must be compatible with the *host's* NVIDIA driver (drivers are backward-compatible with older CUDA runtimes, not forward-compatible with newer ones) — a container built against CUDA 12.4 will fail on a host with only a CUDA-11-era driver.
- In Kubernetes, this is abstracted via the **NVIDIA GPU Operator** / device plugin, which advertises `nvidia.com/gpu` as a schedulable resource and handles driver/toolkit installation on nodes — but the CUDA-version-vs-driver compatibility problem doesn't go away, it just moves to "which node pool has which driver version."

---

### D5: How should you manage model weights in a containerized serving setup — bake into the image or mount at runtime?

**Answer:**
- **Bake into image**: simplest, fully reproducible/immutable deployment (the image *is* the version), but images become huge, every model update means a full image rebuild+push+pull, and registry costs/pull times balloon.
- **Mount/download at runtime**: image stays small and generic; an init container or startup script pulls weights from S3/GCS/a model registry into a volume (or a `hostPath`/node-local cache to avoid re-downloading on every pod restart). This decouples "code version" from "model version" — you can roll a new model without rebuilding the serving image.
- **Common production answer**: runtime mount with a warm node-local cache, because it lets you swap models via a config change (env var / ConfigMap pointing at a model URI) rather than a full CI/CD image rebuild — this is also what makes canary/rollback of *model* versions independent from *code* versions (see Model Versioning, Ch. 09).
- Watch for: cold-start penalty — pulling a 140GB checkpoint over the network on every new pod is a major contributor to autoscaling lag (see Autoscaling, Ch. 06).

---

### D6: What goes wrong if you don't pin exact versions of CUDA, PyTorch, and the inference engine in your image?

**Answer:**
- Silent ABI mismatches: a PyTorch wheel built against CUDA 12.1 loaded against a CUDA 12.4 runtime can work, half-work (wrong kernel selected, slower fallback path), or crash with a cryptic `CUDA error: no kernel image is available for execution on the device`.
- "Works on my machine" drift: an unpinned `pip install vllm` today vs. three weeks from now can pull a new vLLM minor version with different default flags, different memory behavior, or a different OpenAI-API compatibility surface — a change that looks like nothing in the Dockerfile diff can change production behavior.
- Best practice: pin exact versions (`torch==2.4.0+cu124`, `vllm==0.6.3`), use a lockfile, and rebuild-and-test in CI before promoting an image tag — treat the image as the unit of reproducibility, same discipline as pinning application dependencies.

---

### D7: How would you use `docker-compose.yml` in local development for an LLM serving stack, and where does it stop being appropriate?

**Answer:**
- Good for local/dev: bring up the serving container plus supporting services (Prometheus, Grafana, a mock model registry) with one command, wire GPU passthrough (`runtime: nvidia`/`deploy.resources.reservations.devices`), share a volume for model weights so you don't re-download them per `up`.
- Stops being appropriate at: multi-node scheduling, autoscaling, rolling updates, secrets management at scale, and multi-tenant resource isolation — that's exactly the gap Kubernetes fills (Ch. 03). Compose is single-host by design; production LLM serving is not.
- Good interview answer pattern: "Compose for inner-loop dev parity, Kubernetes for anything that needs to scale past one box or survive a node failure."

---

### D8: How do you reduce image size and cold-start time for a GPU serving image without breaking reproducibility?

**Answer:**
- Multi-stage build (Q8) to drop compiler toolchains from the final image.
- Prefer the framework's official runtime image (e.g., `vllm/vllm-openai:<pinned-tag>`) over building CUDA + PyTorch + engine from scratch — it's already been layer-optimized and tested by the maintainers.
- Order Dockerfile layers from least-to-most frequently changing (base image → system deps → Python deps → app code) so CI cache hits on rebuilds that only touch app code.
- Don't bake model weights into the image (Q10); use a node-local weights cache instead so image pulls stay fast even as models grow.
- Pre-pull images onto nodes (`DaemonSet` warm-up, or cluster-autoscaler node image pre-baking) if cold node scale-up time is on your critical path for autoscaling.

---

### D9: A container passes `nvidia-smi` inside `docker exec` but the serving process still reports "no CUDA devices". What do you check?

**Answer:**
- `nvidia-smi` working just confirms the toolkit/driver is visible to the container process itself — it doesn't confirm the *specific process* (e.g., a different user, a different cgroup, or a process started before device injection) can actually use it.
- Check: was the container started with the GPU flag at all for *this* invocation (easy to forget on an `docker exec` into a container that was run without `--gpus`); is `CUDA_VISIBLE_DEVICES` set to something empty or wrong; does the installed PyTorch/vLLM wheel actually have CUDA support (`torch.cuda.is_available()` returning False despite `nvidia-smi` working almost always means a CPU-only wheel got installed by mistake, e.g., pip resolved to `torch` instead of a CUDA-tagged build).
- In Kubernetes specifically: check the pod actually has `resources.limits."nvidia.com/gpu"` set (GPUs aren't visible at all without a limit — they're not a compressible/requestable-only resource) and that the device plugin/GPU Operator DaemonSet is healthy on that node.

---

## Performance Optimization

### Q8: How would you optimize LLM inference latency?

**Answer:**

**1. Model optimization:**
- **Quantization**: FP16, INT8, INT4, FP8 (trade accuracy for speed and memory).
- **Model pruning**: remove unnecessary weights.
- **Knowledge distillation**: use a smaller model trained to mimic the larger one.

**2. Inference optimization:**
- **KV caching**: cache attention key-values.
- **Batching**: process multiple requests together.
- **Continuous batching**: vLLM/TGI-style iteration-level scheduling.
- **Speculative decoding**: use a small draft model to propose multiple tokens, verify them in one batched pass with the target model.

**3. Hardware:**
- **GPU**: use a GPU instead of CPU (10-100x faster for this workload).
- **Tensor cores**: use specialized hardware paths (bf16/fp8 tensor core kernels).
- **Model/tensor parallelism**: split across multiple GPUs for models that don't fit or need more bandwidth.

**4. System:**
- **Pre-warming**: load the model and run a dummy forward pass before accepting real traffic.
- **Connection pooling**: reuse connections.
- **Response caching**: cache responses/prefixes when semantically valid to do so.

**5. Architecture:**
- **Async processing**: don't block on I/O.
- **Request queuing**: handle bursts gracefully with bounded queues.
- **Load balancing**: distribute requests, ideally KV-cache- or prefix-aware.

---

### Q9: What is PagedAttention and why does it matter?

**Answer:**
PagedAttention is vLLM's memory management technique for the KV cache.

**Problem it solves:**
- Traditional KV cache implementations pre-allocate a contiguous buffer sized for the maximum sequence length per request, causing severe internal fragmentation when actual generations are much shorter.
- Wastes memory when sequences have different lengths, and can't support very long sequences efficiently at high concurrency.

**How it works:**
1. Divide the KV cache into fixed-size "blocks" (like OS memory pages).
2. Allocate blocks on-demand as tokens are generated.
3. Free blocks when sequences complete.
4. Reuse freed blocks for new sequences, and share blocks across sequences that have an identical prefix (prefix caching builds directly on this).

**Benefits:**
- **Efficient memory**: near-zero fragmentation (only fragmentation is within the last partially-filled block).
- **Longer sequences**: supports sequences up to the model's max context more predictably.
- **Higher throughput**: more concurrent sequences fit in the same GPU memory, which is what continuous batching needs to have work to schedule.

**Analogy:** like virtual memory in operating systems — pages allocated as needed, non-contiguous physically but presented as one logical KV cache to the attention kernel.

*(For deeper vLLM-specific internals — chunked prefill, scheduler design, disaggregated prefill/decode, speculative decoding, tensor/pipeline parallelism — see [vLLM Internals Deep Dive](#vllm-internals-deep-dive-ch-05).)*

---

### Q10: Explain the trade-offs between latency and throughput.

**Answer:**

**Latency**: time for one request (ms). **Throughput**: requests (or tokens) processed per second across the whole system.

**Trade-offs:**

1. **Batching:**
   - Larger batches → higher throughput, higher per-request latency (more contention for each decode step).
   - Smaller batches → lower latency, lower throughput.

2. **Model size:**
   - Larger model → higher quality, higher latency.
   - Smaller model → lower latency, potentially lower quality.

3. **Quantization:**
   - Lower precision → faster, less memory, potential accuracy loss.
   - Higher precision → slower, more memory, better accuracy.

4. **Hardware:**
   - More GPUs → higher throughput, higher cost.
   - Fewer GPUs → lower cost, lower throughput.

**Optimization strategy:**
- **Low latency**: small batches, optimized model, fast hardware, prioritize interactive/streaming traffic.
- **High throughput**: large batches, continuous batching, multiple GPUs, prioritize offline/batch traffic.
- **Senior framing**: don't optimize "latency" or "throughput" in the abstract — optimize for the actual SLO (e.g., "P95 TTFT < 300ms, P95 inter-token latency < 50ms") and measure throughput *at that SLO*, not throughput in isolation. A system that hits huge throughput numbers by letting P99 latency blow up has usually just moved the problem, not solved it.

---

## Kubernetes & Deployment (Ch. 03)

*Maps to Chapter 03 — Kubernetes (`deployment.yaml`, `deploy.sh`).*

### Q11: How would you deploy an LLM model to Kubernetes?

**Answer:**

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

**Key considerations:**
- Resource limits (GPU, memory) — GPUs are only ever requested as whole-unit `limits`, never fractional, without a sharing mechanism (see K3 below).
- Health checks (liveness/readiness/startup — see Q5e).
- Rolling update strategy (see K2 below — the default doesn't fit single-GPU-per-pod workloads well).
- ConfigMaps for configuration, Secrets for API keys/tokens.

---

### Q12: How does Horizontal Pod Autoscaling (HPA) work?

**Answer:**
HPA automatically scales the number of pod replicas based on metrics.

**How it works:**
1. HPA checks metrics every 15 seconds (default, via the metrics pipeline).
2. Compares the current metric value to the target.
3. Calculates the desired number of replicas.
4. Updates the Deployment's replica count.
5. Kubernetes creates/destroys pods to match.

**Example:**
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

**Metrics:**
- **CPU/Memory**: built-in resource metrics — a poor signal for GPU-bound LLM serving.
- **Custom metrics**: requests-per-second, queue depth, or latency, via the Prometheus Adapter or KEDA.

**Scaling behavior:**
- **Scale up**: when the metric exceeds target (add pods).
- **Scale down**: when the metric is below target (remove pods).
- **Stabilization window**: prevents flapping (rapid scale up/down cycles).

*(For why CPU/memory-based HPA is usually the wrong tool for LLM serving, and what to use instead, see [Autoscaling Deep Dive](#autoscaling-deep-dive-ch-06).)*

---

### Q13: Explain canary deployments for model updates.

**Answer:**
A canary deployment gradually rolls out a new model version to a small percentage of traffic.

**Process:**
1. **Deploy canary**: deploy the new version alongside the stable one.
2. **Route traffic**: split traffic (e.g., 90% stable, 10% canary).
3. **Monitor**: compare metrics (latency, errors, output quality).
4. **Gradual increase**: if healthy, increase canary traffic (25%, 50%, 100%).
5. **Rollback**: if issues appear, route all traffic back to stable.

**Implementation:**
- **Kubernetes-native**: two Deployments behind one Service, weighted by replica-count ratio (coarse-grained).
- **Service mesh (Istio) / Gateway API**: fine-grained percentage-based traffic splitting independent of replica counts.
- **Application-level**: route based on headers/user-ID hashing for consistent bucketing.

**Benefits:**
- **Risk reduction**: validate on a small blast radius first.
- **Quick rollback**: revert traffic weights instantly if issues are detected.
- **A/B testing**: compare model versions on real traffic.

**Monitoring:**
- Compare latency (P50/P95/P99) between canary and stable cohorts, not just in aggregate.
- Error rates.
- Business/quality metrics (e.g., thumbs-down rate, hallucination-flag rate).

*(For the full automated promotion/rollback pipeline, see [Canary Deployments Deep Dive](#canary-deployments-deep-dive-ch-07).)*

---

### K1: Why is `nvidia.com/gpu` treated so differently from CPU/memory by the Kubernetes scheduler?

**Answer:**
- GPUs are an **extended resource**: they can only be requested as integers, cannot be overcommitted, and (without MIG/time-slicing/MPS configured) `requests` must equal `limits` — Kubernetes doesn't support fractional or burstable GPU by default.
- This means the scheduler treats a GPU node's GPUs like a small pool of indivisible slots; a pod asking for 1 GPU either fits on a node with a free GPU or it doesn't schedule at all — there's no "squeeze it in" the way there is for CPU.
- Practical consequence: GPU **bin-packing** matters a lot more than CPU bin-packing. A cluster with plenty of *total* free GPU capacity spread as fractions across many nodes can still fail to schedule a pod if no single node has a whole GPU free — this is why node pool sizing and pod GPU-count granularity (1 vs 2 vs 8 GPUs per pod) is a real design decision, not an afterthought.

---

### K2: The default Kubernetes `RollingUpdate` strategy is a poor fit for a Deployment where each pod holds a 70B model on 4 GPUs. Why, and what would you change?

**Answer:**
- Default `RollingUpdate` with `maxSurge: 25%` tries to bring up *extra* pods before removing old ones — for a pod that needs 4 GPUs and takes minutes to load a huge checkpoint, "surge" capacity means you need spare GPU capacity sitting idle just to support the rollout, which is often not available in a GPU-constrained cluster.
- Fixes: set `maxSurge: 0, maxUnavailable: 1` (replace-in-place, one at a time) if you have zero spare capacity and can tolerate a brief capacity dip; or explicitly provision surge capacity (over-provision by one replica's worth of GPUs) if you need truly zero-downtime rollouts; or move to a **canary** rollout (Ch. 07) so the "old" fleet stays fully up while a small new fleet is validated, rather than doing an in-place blue/green swap of the whole Deployment at once.
- Also tune `minReadySeconds` and the readiness probe (Q5e) so a pod isn't counted "available" until the model has actually finished loading and passed a real warm-up check — otherwise the rollout can proceed faster than pods are actually ready, causing a real availability dip.

---

### K3: How do you share a single GPU across multiple pods in Kubernetes, and what are the trade-offs between the approaches?

**Answer:**
| Approach | Isolation | Granularity | Notes |
|---|---|---|---|
| **Time-slicing** | None (software round-robin) | Coarse (N pods share one GPU's full memory + compute pool) | Simplest; risk of one pod OOM-ing or starving others; no memory isolation. |
| **MPS (Multi-Process Service)** | Weak (shared address space, but concurrent kernel execution) | Better compute overlap than time-slicing | Still no hard memory isolation; a misbehaving process can affect others. |
| **MIG (Multi-Instance GPU)** | Hardware-level (separate compute/memory partitions) | Fixed set of profiles (e.g., 7 slices on an A100/H100-class MIG-capable GPU) | True isolation, but partitions are fixed-size and set at node-configuration time, not per-pod-request time. |
| **DRA (Dynamic Resource Allocation)** | Depends on underlying tech (MIG, time-slice, or whole-device) | Flexible, expressed via `DeviceClass`/`ResourceClaim` with CEL-based selection | The modern (GA in Kubernetes v1.34) API replacing device-plugin extended resources for complex GPU allocation logic — verify exact version behavior before quoting in an interview, this area is moving fast. |
| **MPS/MIG combined with DRA** | Hardware isolation + flexible scheduling | Best of both, most complex to operate | Where the ecosystem (NVIDIA GPU Operator, KAI Scheduler) is heading as of 2026. |

**Senior framing**: pick isolation level based on the blast radius you can tolerate. For a shared inference platform serving multiple internal teams, prefer MIG or DRA-managed hard partitions; for single-tenant batch/dev workloads, time-slicing is an acceptable cost optimization.

---

### K4: Should an LLM serving Deployment use a `Deployment` or a `StatefulSet`?

**Answer:**
- Most stateless serving pods (no per-pod persistent identity or storage beyond the shared model weights) belong in a **Deployment** — pods are interchangeable, and the Service load-balances across them.
- **StatefulSet** becomes relevant when pods need stable network identity or per-pod storage — e.g., a multi-node tensor-parallel or pipeline-parallel deployment where rank-0 needs a predictable address for the other ranks to connect to (common with vLLM/Triton multi-node setups, or a Ray-based serving cluster), or when each pod owns a distinct local NVMe cache you want to preserve across restarts.
- Don't reach for StatefulSet by default "because it's a model" — it's about *identity and storage*, not about "this workload feels heavyweight."

---

### K5: Design node affinity / taints-and-tolerations for a mixed CPU+GPU cluster running LLM serving alongside other workloads.

**Answer:**
- Taint GPU nodes (`nvidia.com/gpu=true:NoSchedule`) so only pods that explicitly tolerate the taint (and request a GPU) land there — prevents random CPU workloads from occupying expensive GPU nodes.
- Use `nodeAffinity`/`nodeSelector` on the serving pods to require the matching GPU node pool (and, if relevant, a specific GPU generation — you don't want a pod that assumes 80GB of HBM landing on a 40GB node).
- Combine with **PriorityClasses**: give latency-sensitive serving pods higher priority than batch/offline jobs so the scheduler preempts lower-priority work under GPU pressure rather than leaving serving pods `Pending`.
- Use topology spread constraints across zones/nodes so a single node or zone failure doesn't take out all replicas of a critical model at once.

---

### K6: How do you handle persistent storage for multi-tens-of-GB model weights across pod restarts and node scale-up events?

**Answer:**
- **ReadOnlyMany PVC** (e.g., backed by a network filesystem or a cloud "block volume with RWX" equivalent) shared across pods of the same model version avoids every pod independently re-downloading the same weights.
- **Node-local caching** (hostPath or a local NVMe-backed cache managed by a DaemonSet) trades simplicity for speed: first pod on a node pays the download cost, subsequent pods/restarts on that node are fast — but cold nodes (new from autoscaling) still pay full cost, which directly affects scale-up latency (see Ch. 06).
- **Init containers** are the standard mechanism: an init container downloads/verifies the model artifact from the registry before the main container starts, so the main container's readiness probe only has to check "engine loaded," not "and also, is the download done."
- Watch for: race conditions if multiple pods write to the same node-local cache path concurrently during a scale-up burst — use a lock file or content-addressed paths (e.g., keyed by model hash) to make the cache safe for concurrent writers.

---

### K7: What's the "cold start" problem for GPU pods in Kubernetes, and how much of it is actually Kubernetes' fault?

**Answer:**
- Most of a GPU pod's cold-start time is *not* Kubernetes scheduling overhead (which is seconds) — it's: (a) provisioning a new GPU node if none is free (can be 1-10+ minutes depending on cloud/instance type), (b) pulling a multi-GB container image, (c) downloading multi-tens-of-GB model weights, (d) engine warm-up (CUDA graph capture, kernel autotuning, e.g. vLLM's startup profiling pass).
- Kubernetes itself contributes almost none of this — which is why solving cold start means attacking image size (Ch. 02), weight caching (K6), and node pre-provisioning/warm pools, not tuning scheduler settings.
- This is precisely why naive reactive autoscaling (scale up only after load already arrived) works poorly for LLM serving — see Autoscaling Deep Dive (Ch. 06) for predictive/pre-warmed approaches.

---

### K8: How do you expose an LLM serving Service outside the cluster safely, and where does authentication/rate-limiting belong?

**Answer:**
- Typical chain: external Load Balancer / Ingress (or Gateway API) → API Gateway (auth, rate limiting, request validation, routing to model-version-specific backends) → Kubernetes Service → serving pods.
- Authentication and rate limiting belong at the gateway layer, not inside the serving pod — the serving pod should be a "dumb," fast, trusted-input component; pushing auth into every pod duplicates logic and makes the hot path slower and harder to change.
- TLS termination typically happens at the Ingress/gateway, not per-pod, to avoid burning GPU-node CPU on crypto.
- If different model tiers/versions need different backends, route at the gateway/Ingress layer by path or header (`/v1/completions` with `model: "..."` in the body, or a dedicated path per model) rather than baking routing logic into the client.

---

## Load Testing & Capacity Planning (Ch. 04)

*Maps to Chapter 04 — Load Testing (`locust_test.py`, `measure_latency.py`).*

### LT1: What's wrong with load-testing an LLM API using the same request-per-second methodology you'd use for a REST CRUD API?

**Answer:**
- A CRUD API's requests are roughly uniform in cost; LLM requests vary enormously by prompt length and `max_tokens`, so "requests/sec" alone is a poor load axis — two runs at the same RPS but different token distributions produce completely different GPU load.
- LLM serving has two distinct cost phases (prefill vs decode — see Q1); a load test needs to vary both **input length** and **output length** distributions to be representative, not just request rate.
- Because of continuous batching, the *right* concurrency metric is often "number of concurrent in-flight sequences," not "requests per second" — throughput is fundamentally a function of how many sequences the engine is juggling, capped by KV-cache memory, not just arrival rate.

---

### LT2: Design a load test plan for a new vLLM deployment before it goes to production.

**Answer:**
1. **Define the traffic profile**: realistic prompt-length and output-length distributions (pull from production logs if you have any, or from expected use case — chat vs summarization vs RAG have very different profiles).
2. **Sweep concurrency**, not just RPS: run at increasing numbers of concurrent requests/users and record throughput (tokens/sec) and latency (TTFT, inter-token latency, end-to-end) at each level.
3. **Find the knee of the curve**: throughput rises roughly linearly with concurrency until GPU/KV-cache saturation, then latency blows up while throughput plateaus (or falls) — that knee is your practical capacity ceiling per replica.
4. **Test at your actual SLO, not "as fast as possible"**: report the max sustainable throughput *while P95 TTFT and P95 inter-token latency stay within SLO*, not raw peak throughput.
5. **Include a soak test**: run at ~70-80% of found capacity for an extended period (hours) to catch memory leaks, KV-cache fragmentation over time, and slow degradation that a short burst test won't reveal.
6. **Test failure/recovery paths**: what happens at 100%+ of capacity — do requests queue and eventually succeed, or does the engine start erroring/OOMing? Both matter for how you configure backpressure.

---

### LT3: What metrics do you pull out of a load test, beyond "requests per second"?

**Answer:**
- **TTFT (time to first token)** — dominated by prefill + queueing time; the metric users feel first.
- **Inter-token latency / TPOT (time per output token)** — dominated by decode step time and batch contention; determines how "fast" streaming feels.
- **End-to-end latency** — TTFT + (TPOT × output tokens); the metric a non-streaming client cares about.
- **Throughput in tokens/sec** (input + output, or output-only depending on what you're optimizing) — the real capacity number, more meaningful than requests/sec.
- **GPU utilization and KV-cache utilization** during the run — tells you whether you're compute-bound, memory-bound, or scheduler-bound.
- **Error/timeout rate** as load increases — where does the system start shedding load, and does it do so gracefully (429s) or badly (timeouts/crashes)?

---

### LT4: How do you use Locust (or similar) to generate realistic LLM traffic, and what's the catch with naive `locust_test.py`-style scripts?

**Answer:**
- Naive scripts often send a fixed prompt repeatedly at a fixed rate — this doesn't exercise prefix-cache behavior realistically (either artificially inflating hit rate with an identical prompt, or missing it entirely with fully random prompts) and doesn't vary output length, hiding decode-bound bottlenecks.
- Better approach: sample prompts from a representative corpus (varying length), sample `max_tokens`/expected output length from a realistic distribution, and use Locust's user-count ramp-up to simulate concurrency growth rather than a flat rate.
- Catch: Locust's own workers can become the bottleneck (Python GIL, single-machine limits) before the server does at high concurrency — for serious load testing distribute Locust workers, or use a purpose-built LLM load tool (e.g., `vllm bench serve`, `genai-perf`) that already models TTFT/TPOT correctly and doesn't fight Python overhead.

---

### LT5: `measure_latency.py`-style scripts show great P50 latency but the on-call gets paged for P99 timeouts in production. What's going on?

**Answer:**
- Mean/P50 latency is dominated by the common case; tail latency (P99) is dominated by queueing effects, GC/allocator pauses, occasional very long prompts, cold caches, or a batch temporarily saturated by a few outlier long-generation requests.
- A load test run at low/medium concurrency will show good P50 and simply not exercise the queueing regime that causes P99 blowups in production bursts — you have to load test at (and above) your expected peak concurrency, not just average concurrency, to see P99 behavior.
- Also check: is the load test hitting a warm, pre-scaled fleet, while production traffic sometimes hits cold/just-scaled-up pods (see K7)? That gap alone can explain a P50-looks-fine-but-P99-pages-us pattern.

---

### LT6: How does batch size / concurrency setting interact with load test results, and what would you tune based on what you see?

**Answer:**
- If throughput is well below what GPU compute should allow and GPU utilization is low, the engine's max-concurrent-sequences or max-batched-tokens setting is probably too conservative — raise it (bounded by KV-cache memory).
- If latency degrades sharply while throughput barely improves, you've likely hit the KV-cache memory ceiling — the engine is now queueing/evicting rather than adding new sequences to the running batch; the fix is more memory (bigger GPU, higher `gpu_memory_utilization` fraction, quantized KV cache) or accepting a lower concurrency ceiling.
- Chunked prefill settings matter here too: a long prompt hogging a full prefill step can spike inter-token latency for other in-flight decode requests sharing that step; load testing with a mix of very long and very short prompts reveals this in a way a uniform-length test won't.

---

### LT7: How would you capacity-plan "how many GPUs do I need to serve X req/s at Y SLO" from load test data?

**Answer:**
1. From the load test, find the max concurrency (or max tokens/sec) one replica sustains while meeting the latency SLO — call it `C_max`.
2. Model expected traffic: peak concurrent requests (not average — provision for peak, autoscale for the rest), using your actual prompt/output-length distribution.
3. Required replicas ≈ `peak_concurrency / C_max`, plus headroom for: rolling updates (K2), node/zone failure tolerance (N+1 or N+2), and burst above forecast.
4. Multiply by GPUs-per-replica (tensor-parallel degree) to get total GPU count, then sanity-check against actual GPU memory needed for weights + KV cache at that concurrency.
5. Always re-validate with a real load test on the target hardware/model/quantization combination — theoretical FLOPs-based capacity estimates are frequently off by 2-3x from measured reality because of memory-bandwidth bottlenecks in decode.

---

### LT8: What's the difference between load testing "throughput mode" and "latency mode," and why would you run both?

**Answer:**
- **Throughput mode**: fire as many concurrent requests as the client can generate, unconstrained by arrival timing, to find the engine's maximum sustainable tokens/sec — useful for batch/offline workloads and for finding the theoretical ceiling.
- **Latency mode**: send requests at a fixed, realistic arrival rate (Poisson-ish) matching expected production traffic, and measure latency distribution at that specific load — useful for validating the SLO under realistic (not maximal) load.
- You need both: throughput mode tells you the ceiling (and where the knee is); latency mode tells you whether your *planned* capacity, running at *planned* traffic, actually meets the SLO you promised — a system can have a great throughput ceiling and still miss its latency SLO at normal traffic if it's misconfigured (e.g., batch size tuned for throughput, not latency).

---

### LT9: How do you make load tests reproducible and comparable across engine versions or config changes?

**Answer:**
- Pin the exact prompt/output-length dataset (seeded sampling) so two runs are comparing the same workload, not different random draws.
- Run on the same hardware/instance type, same driver/CUDA version, same model checkpoint and quantization.
- Report full latency distributions (P50/P90/P95/P99), not just averages — averages hide regressions that show up only in the tail.
- Track load test results over time (a simple dashboard or even a CSV committed alongside config changes) so a vLLM version bump or a scheduler-flag change has an objective before/after comparison instead of "it feels faster."

---

## vLLM Internals Deep Dive (Ch. 05)

*Maps to Chapter 05 — vLLM Serving (`vllm_server.py`). The two questions below are preserved from the original Model Serving section because they're really vLLM-specific.*

### Q6: What are the differences between HuggingFace Transformers and vLLM?

**Answer:**

| Feature | HuggingFace `generate()` | vLLM |
|---------|-------------|------|
| **Batching** | Static batching | Continuous (iteration-level) batching |
| **Throughput** | Low (single-digit to low tens of req/s in naive setups) | Much higher — order(s) of magnitude, workload-dependent |
| **Memory** | Standard, often over-allocated KV cache | PagedAttention (efficient, near-zero fragmentation) |
| **GPU utilization** | Often 20-40% | Often 80-95% under load |
| **Ease of use** | Very easy, huge model/ecosystem coverage | Moderate — needs engine-specific configuration |
| **Flexibility** | High (arbitrary custom generation logic) | Moderate (optimized for the common serving path) |

**When to use HuggingFace `transformers`:** development/prototyping, research, small-scale or one-off inference, need for maximum flexibility (custom generation logic, unusual architectures).

**When to use vLLM (or a comparable engine):** production, high-throughput or high-concurrency serving, cost-sensitive GPU utilization.

*(Treat exact throughput multipliers as workload-dependent — always validate with your own load test per Ch. 04 rather than quoting a fixed number.)*

---

### Q7: How does continuous batching work in vLLM?

**Answer:**
Continuous batching allows adding/removing requests dynamically at each decode iteration, rather than batching only at request boundaries.

**Traditional (static) batching:**
1. Wait for a batch to fill (e.g., 8 requests).
2. Process the entire batch.
3. Wait for all to complete.
4. Start the next batch — the whole batch is only as fast as its slowest member.

**Continuous batching:**
1. Start processing a batch.
2. Add new requests as they arrive, as soon as there's KV-cache room.
3. Remove completed requests from the batch immediately.
4. Continue processing remaining requests without waiting for the whole cohort.
5. GPU stays busy — no bubble waiting for a batch to fill or fully drain.

**Benefits:** higher GPU utilization, lower latency (no waiting for a batch to fill), better throughput.

**Example:**
```
Time 0: [Req1, Req2, Req3] -> Processing
Time 1: [Req1, Req2, Req3, Req4] -> Req4 added mid-flight
Time 2: [Req2, Req3, Req4] -> Req1 completed, removed
```

---

### V1: What is chunked prefill and why did vLLM add it?

**Answer:**
- Without it, a long prompt's prefill runs as one large step that can monopolize a full scheduler iteration, delaying the decode steps of every other in-flight request sharing that iteration — a long prefill can cause a visible latency spike for unrelated users.
- Chunked prefill splits a long prompt's prefill into smaller chunks processed across multiple iterations, interleaved with other requests' decode steps, so no single request's prefill blocks everyone else's inter-token latency.
- Net effect: smoother, more predictable inter-token latency under mixed short/long-prompt traffic, at a small cost in total prefill throughput — a direct example of trading a bit of throughput for tail-latency stability.

---

### V2: Explain vLLM's scheduler at a level that would satisfy a senior interviewer.

**Answer:**
- Each iteration, the scheduler decides which requests to run a step for, subject to a KV-cache-block budget and a max-batched-tokens budget.
- It prioritizes continuing already-running (decode) sequences, and admits new (prefill) requests from a waiting queue as capacity allows — with chunked prefill, it can also admit *partial* prefill work for a new/long request rather than an all-or-nothing decision.
- If KV-cache blocks run out, the scheduler can **preempt** a running sequence (evict it, either by swapping its KV cache out or by recomputing it later) to make room for others — which one gets evicted (and eviction policy) directly affects fairness and tail latency.
- This scheduler is the actual "engine" behind continuous batching (Q7) — it's what makes continuous batching a scheduling problem, not just a batching trick.

---

### V3: What is speculative decoding and when does it actually help?

**Answer:**
- A small/cheap "draft" model proposes several candidate next tokens; the large "target" model verifies all of them in a single batched forward pass (cheaper than generating them one at a time autoregressively) and accepts the longest correct prefix.
- Helps most when decode is memory-bandwidth-bound (the usual case) and the draft model's guesses are frequently correct — e.g., low-temperature/deterministic-ish generation, code completion, or draft models specifically trained/distilled to match the target's distribution.
- Helps less (or can hurt) at high sampling temperature/high diversity output, or when running at high concurrency where you're already GPU-compute-saturated by other sequences' decode steps — speculative decoding trades some extra compute per accepted token for fewer serial steps, and that trade is worse when compute is already the bottleneck.
- Variants worth naming: draft-model speculative decoding, "Medusa"/multi-head style (extra prediction heads on the target model itself), and n-gram/prompt-lookup decoding (no separate model at all, works well for tasks with lots of literal repetition like code editing).

---

### V4: How does vLLM support tensor parallelism and pipeline parallelism, and when do you pick which?

**Answer:**
- **Tensor parallelism (TP)**: shards each layer's weight matrices across GPUs (e.g., attention heads and MLP columns split across N GPUs); requires an all-reduce/all-gather communication step per layer, so it needs high-bandwidth interconnect (NVLink) between the GPUs involved — this is why TP is usually kept *within* a node.
- **Pipeline parallelism (PP)**: splits the model's layers across GPUs/nodes sequentially; communication is only at stage boundaries (much less than TP), so it tolerates slower interconnect (e.g., across nodes over Ethernet/InfiniBand) better, at the cost of pipeline bubbles unless microbatching is used.
- **Rule of thumb**: use TP up to the GPU count within one NVLink domain (e.g., 8 GPUs on one node) to fit/speed up a model that doesn't fit on fewer GPUs; use PP (or TP+PP together) when you need to span multiple nodes, since cross-node TP communication is usually too slow relative to compute.
- Data parallelism (replicate the whole model, different requests per replica) is orthogonal and is what you scale with Kubernetes replicas/HPA — TP/PP determine how big *one* replica is; replica count determines how many of those you run.

---

### V5: What quantization formats does a modern serving engine like vLLM support, and how do you choose one?

**Answer:**
- **FP16/BF16**: baseline for most serving, minimal accuracy loss vs FP32, halves memory vs FP32.
- **FP8**: supported natively on Hopper/Blackwell-class tensor cores; roughly halves memory and can meaningfully speed up compute-bound phases (prefill) with small, workload-dependent accuracy impact — increasingly the default "fast path" on new hardware generations as of 2025-2026 (verify current default support matrix before quoting for a specific model/engine).
- **INT8/AWQ/GPTQ/INT4**: aggressive memory reduction (useful when weights, not KV cache, are the constraint, or on memory-constrained GPUs), larger accuracy risk, usually needs calibration data and a validation pass before shipping.
- **KV-cache quantization** (e.g., FP8 KV cache) is a *separate* knob from weight quantization — it shrinks the KV cache specifically, which is often the actual bottleneck at high concurrency/long context, independent of what precision the weights are stored in.
- Decision process: start from FP16/BF16 baseline load test, then try FP8 (usually close to free quality-wise on modern hardware) before reaching for INT4/AWQ, and always validate task-specific quality (not just perplexity) before shipping a quantized model.

---

### V6: What is prefix caching (as distinct from PagedAttention itself) and why does it matter for RAG/chat workloads?

**Answer:**
- Prefix caching reuses KV-cache blocks across requests that share an identical prompt prefix (e.g., the same system prompt, or the same long RAG context reused across follow-up questions) instead of recomputing prefill for the shared portion every time.
- Because PagedAttention's block-based KV cache is already non-contiguous and content-addressable-friendly, this is a relatively natural extension — cache blocks by content hash, look up on new requests, reuse hits.
- Huge win for chat (repeated system prompt + growing conversation history) and RAG (same retrieved context, multiple questions) workloads — can turn a large fraction of "prefill" work into cache hits, dramatically improving TTFT.
- Design implication: your load balancer ideally should be **prefix/cache-aware** (route requests likely to share a prefix to the same replica) rather than purely round-robin, or you lose most of the benefit by scattering cache-hit opportunities across replicas that don't share cache state.

---

### V7: What changed with vLLM's "V1" engine rewrite, and why should you know that as an interview talking point?

**Answer:**
- vLLM underwent a significant core-architecture rewrite (alpha announced ~early 2025) redesigning the scheduler and execution loop for lower CPU-side overhead, cleaner separation of scheduling/model-execution, and better support for the multimodal/newer-model-architecture surface, while keeping continuous batching and PagedAttention as foundational ideas.
- Interview signal: knowing that "vLLM" isn't a single frozen design — it has actively evolved its internals (scheduler, execution engine) even while its user-facing API stayed largely stable — shows you track the ecosystem rather than reciting a single blog post from a couple of years ago.
- Practical takeaway: exact internal architecture details (V0 vs V1 engine specifics) change fast; anchor your answer on the *invariants* (continuous batching, paged/block KV cache, chunked prefill, speculative decoding support) and flag version-specific internals as "verify against current docs" rather than asserting a specific implementation detail with false confidence.

---

### V8: What is disaggregated prefill/decode serving, and why would a senior team consider it?

**Answer:**
- Idea: run prefill (compute-bound, bursty, parallelizable) and decode (memory-bandwidth-bound, needs sustained KV-cache residency) on **separate pools of GPUs/nodes**, with the KV cache for a request transferred from a prefill worker to a decode worker once prefill completes (over fast interconnect, or via a shared KV-cache store like LMCache).
- Motivation: prefill and decode have different optimal batch sizes and different bottlenecks; co-locating them on the same GPU means a long prefill can stall decode's latency (mitigated but not eliminated by chunked prefill), and it means you can't independently scale/tune each phase's hardware (e.g., prefill benefits more from raw FLOPs, decode benefits more from memory bandwidth and cache capacity).
- Cost: real engineering complexity — a KV-cache transfer path, coordination between two pools, and new failure modes (what happens if the decode worker for a request's KV cache goes away?). This is an advanced/large-scale optimization (the kind large model-serving platforms adopt), not a default for most production deployments — good interviewees flag it as "a technique for squeezing out the next increment at very large scale," not something to reach for on day one.

---

## Autoscaling Deep Dive (Ch. 06)

*Maps to Chapter 06 — Autoscaling (`hpa.yaml`). Builds on Q12 above.*

### AS1: Why is CPU utilization almost always the wrong HPA signal for a GPU-bound LLM serving pod?

**Answer:**
- The pod's CPU usage (tokenization, HTTP handling, orchestration) is largely decoupled from how loaded the GPU actually is — a pod can be at 90% GPU utilization / near KV-cache capacity while its CPU sits at 10%, so a CPU-based HPA target never fires when it should.
- The correct signals are GPU-native or engine-native: GPU utilization, KV-cache/block utilization, queue depth (pending requests waiting for a scheduler slot), or request latency directly.
- These require a **custom/external metrics** pipeline (Prometheus Adapter, or KEDA scaler reading from Prometheus/the engine's own metrics endpoint) — plain built-in `Resource` metrics in HPA won't get you there.

---

### AS2: Design an HPA/KEDA config for a vLLM deployment using queue depth as the scaling signal.

**Answer:**
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
- `vllm_num_requests_waiting` (or equivalent queued-request gauge) rising means the running batch is already at capacity and new work is backing up — a much earlier and more meaningful signal than GPU-utilization-plateaued-at-95%.
- Set `minReplicaCount` above zero unless you've specifically solved the cold-start problem (K7) for scale-from-zero; otherwise the first user during a scale-from-zero event eats the full cold-start latency.
- Tune the scale-up stabilization window short (react fast to load) and the scale-down window long (don't tear down a replica that just barely dipped below threshold — GPU pods are expensive to bring back).

---

### AS3: Why is scale-to-zero for GPU inference pods so much harder than for a stateless web service?

**Answer:**
- A stateless web pod might cold-start in 1-5 seconds; a GPU serving pod's cold start (K7) — node provisioning, image pull, weight download, engine warm-up — is commonly 1-10+ minutes.
- If you scale to zero, the *first* request after idle pays that entire cold-start cost, which almost never fits any reasonable latency SLO — this is the central tension of GPU scale-to-zero.
- Mitigations: keep a warm minimum (never truly scale to zero for latency-sensitive paths), use a "keep warm" ping/synthetic traffic to prevent scale-down during known-active hours, pre-provision a warm node pool (nodes exist and are GPU-ready, but pods aren't yet placed) so scale-up skips the node-provisioning step, or accept scale-to-zero only for genuinely latency-insensitive/batch workloads where a cold-start delay is acceptable.

---

### AS4: How does cluster autoscaling (or Karpenter-style node autoscaling) interact with pod-level HPA for GPU workloads, and where does it commonly break?

**Answer:**
- HPA decides "we need N more pods"; if no node has free GPU capacity, those pods sit `Pending` until the cluster/node autoscaler notices unschedulable pods and provisions a new GPU node — this handoff adds the full node-provisioning latency (often minutes) on top of pod scheduling.
- Common break: cluster autoscaler / Karpenter provisioning a node type that doesn't actually have the GPU generation or count the pod's affinity/resource request demands, causing a provision-then-still-Pending loop — GPU node pool definitions need to precisely match what the workload asks for.
- Also watch scale-down: cluster autoscaler can try to drain a GPU node that's running a long-lived inference pod mid-generation, causing dropped or truncated requests unless pod disruption budgets and graceful termination (Q5g) are configured to protect it.
- Senior framing: GPU autoscaling is a two-layer problem (pod-level HPA + node-level cluster autoscaling) and both layers' reaction latency stack on top of each other — total scale-up latency is the sum, not the max, of the two.

---

### AS5: What is predictive/scheduled autoscaling and when would you use it over purely reactive autoscaling?

**Answer:**
- Reactive autoscaling (HPA/KEDA responding to current metrics) always lags real demand by at least the metric-collection interval plus cold-start time — fine for smoothly varying load, poor for sharp, predictable bursts.
- Predictive/scheduled scaling pre-warms capacity ahead of known patterns: e.g., scale a minimum replica floor up before a known daily traffic peak (business hours, a product launch, a marketing push), using a CronJob-driven change to `minReplicaCount` or a dedicated predictive-scaling controller.
- Use it whenever demand has a known/learnable pattern (time-of-day, day-of-week, scheduled events) — it directly attacks the cold-start problem (AS3) by moving capacity provisioning *before* the load spike instead of reacting to it.
- Still combine with reactive scaling for the unpredictable residual — predictive scaling handles "the shape you can forecast," reactive handles "the surprise on top of it."

---

### AS6: A bursty consumer product (e.g., a viral social feature) spikes 20x in under a minute. Design the autoscaling response.

**Answer:**
- Pure reactive pod+node autoscaling cannot react in under a minute for GPU capacity — the answer isn't "scale faster," it's "absorb the burst without needing to scale that fast":
  1. **Admission control / queueing with backpressure**: bound the queue, return fast 429s beyond it rather than letting latency degrade unboundedly for everyone.
  2. **Warm buffer capacity**: keep a standing buffer of replicas sized for your observed peak/burst ratio, not just average load, funded as a deliberate cost trade-off for burst resilience.
  3. **Degrade gracefully under pressure**: shed to a smaller/cheaper model, cap `max_tokens`, or disable expensive features (e.g., long-context mode) automatically when queue depth crosses a threshold — see the Fallback/Degradation system design scenario.
  4. **Rate limit per-user/per-tenant** so one viral feature doesn't starve every other user of the platform.
- Post-incident: use the observed burst shape to right-size the warm buffer and predictive-scaling schedule for next time (AS5) — every real burst is data for the next capacity plan.

---

### AS7: How do you avoid HPA "flapping" (rapid scale up/down) for a GPU workload, and why is flapping especially costly here vs. a CPU service?

**Answer:**
- Standard HPA levers: `stabilizationWindowSeconds` on scale-down (wait before removing replicas after load drops), scale-down/up rate limiting (`policies` in HPA v2 behavior spec — cap how many pods can be removed per period).
- Why it's worse for GPU: a removed GPU pod isn't just "a process that restarts fast" — bringing it back may mean re-provisioning a node (AS4) and re-downloading weights (K6/K7), so an unnecessary scale-down followed by a needed scale-up shortly after is far more expensive in both latency and cost (repeated cold starts, repeated image/weight pulls) than the equivalent flap on a stateless web service.
- Practical setting: asymmetric behavior — fast/short stabilization window for scale-up, long (minutes) stabilization window for scale-down, and a floor (`minReplicas`) sized above the noise level of your traffic so you're not scaling in and out of your own baseline load.

---

### AS8: How would you autoscale a multi-model platform where different models have very different load profiles and hardware needs?

**Answer:**
- Scale each model's deployment independently, each with its own HPA/KEDA object and its own metric thresholds tuned to that model's actual capacity envelope (a 7B model and a 70B model saturate at very different queue-depth/GPU-utilization numbers).
- Use separate node pools per hardware profile (e.g., single-GPU nodes for small models, multi-GPU NVLink nodes for large tensor-parallel models) so the cluster autoscaler provisions the right shape of node for the right model's pods, and taints/affinities (K5) keep them from cross-contaminating.
- Consider a shared "GPU pool with priority-based preemption" model for cost efficiency across many low-traffic models (small/rare models share capacity, evicted under pressure by higher-priority traffic) rather than dedicating a full autoscaling group per rarely-used model — this is exactly the multi-tenant platform system-design scenario below.

---

## Canary Deployments Deep Dive (Ch. 07)

*Maps to Chapter 07 — Canary Deployments (`canary-deployment.yaml`). Builds on Q13 above.*

### CD1: Walk through a fully automated canary pipeline for a new model version, end to end.

**Answer:**
1. **Build & validate offline**: new model version passes an offline eval suite (quality benchmarks, regression tests against known-bad outputs) before it's even eligible for canary.
2. **Deploy canary at low weight** (e.g., 1-5% of traffic) alongside the stable version, both behind the same gateway/mesh traffic split.
3. **Automated analysis window**: run for a fixed window (e.g., 30-60 minutes, long enough to get statistically meaningful samples) comparing canary vs. stable on a fixed metric set (error rate, P95/P99 latency, and — critically for LLMs — a proxy quality signal like thumbs-down rate or a cheap automated eval on sampled outputs).
4. **Automated promote/rollback decision**: if metrics stay within defined bounds, automatically step traffic up (5% → 25% → 50% → 100%); if any metric breaches a threshold, automatically roll back to 0% canary traffic immediately.
5. **Full promotion**: once at 100%, the "canary" becomes the new "stable," and the old stable is either kept warm briefly (fast-rollback window) or scaled down.
6. **Tooling**: Argo Rollouts or Flagger implement this analysis-and-promotion loop natively on top of Kubernetes + a metrics provider (Prometheus) + a traffic-splitting layer (Istio/Gateway API/SMI).

---

### CD2: What metrics are *LLM-specific* red flags during a canary that a generic web-service canary pipeline wouldn't catch?

**Answer:**
- **Output quality regression**: a new model version can be faster and equally "healthy" on latency/error metrics while being *worse* in output quality (more hallucination, format violations, refusal-rate changes) — generic infra metrics are blind to this entirely.
- **Token-length distribution shift**: a new version generating substantially longer or shorter outputs than expected changes cost and latency in ways that a simple error-rate check won't flag.
- **Safety/guardrail trigger rate**: an increase in the rate your safety filters/guardrails fire on the new version's outputs is a strong leading indicator of a behavior regression.
- **Structured-output/tool-call validity rate**: for models used in agent/tool-calling pipelines, a canary can look "healthy" on latency/errors while silently producing malformed function calls at a higher rate downstream systems choke on.
- Practical answer: a real LLM canary pipeline needs an **automated quality-eval step** sampling live canary outputs (not just infra metrics) before promoting, ideally with a cheap LLM-as-judge or rule-based check running inline.

---

### CD3: Canary at the Kubernetes-Deployment-replica-ratio level vs. canary via a service mesh's percentage-based traffic split — what's the real difference, and when does it matter?

**Answer:**
- **Replica-ratio canary** (e.g., 9 stable pods + 1 canary pod behind one Service): traffic split is only as fine-grained as your replica count allows (with 10 pods you get 10% steps), and it couples "how much traffic the canary gets" to "how many pods it has," which conflates two different concerns (blast radius vs. resource allocation).
- **Mesh/Gateway percentage-based split** (Istio VirtualService weight, Gateway API HTTPRoute weight): decouples traffic percentage from replica count entirely — you can send exactly 1% of traffic to a canary running on a single full-sized pod, giving you fine control over blast radius independent of infrastructure sizing.
- Matters most for large/expensive models: with a 70B model where each replica is a meaningful GPU cost, you want the canary's traffic percentage (risk exposure) decoupled from how many expensive replicas you're forced to run for it.

---

### CD4: Design the automated rollback trigger logic — what exactly should trip an automatic rollback, and how do you avoid both false positives and false negatives?

**Answer:**
- Define a small set of **guardrail metrics** with explicit thresholds evaluated over a rolling window (not single data points, to avoid noise-driven false rollbacks): e.g., error rate > baseline + 2%, P99 latency > baseline × 1.5, guardrail-trigger rate > baseline + absolute threshold.
- Require **minimum sample size** before evaluating (a canary at 1% traffic for 2 minutes may have too few requests for the error-rate comparison to be statistically meaningful) — this avoids false positives from small-sample noise.
- Avoid false negatives by not relying solely on fast-moving infra metrics — include the slower, sampled quality-eval signal (CD2) even though it updates less frequently, since a model can be infra-healthy and quality-broken simultaneously.
- Always have a **manual override** (kill switch) in addition to automation — automated systems should default to safe (rollback) on ambiguous/missing data, and humans should be able to force a rollback regardless of what the automation says.

---

### CD5: How do you canary a change to *infrastructure* (e.g., a new vLLM version or a new GPU generation) rather than a new model weights version?

**Answer:**
- Same pipeline shape (CD1) applies, but the "canary" variable is the serving stack/hardware, not the model — deploy the new engine version/hardware behind a small traffic percentage running the *same* model weights as stable, so any regression is attributable to the infra change, not a confounded model change.
- Pay special attention to numerical/output-distribution differences an engine or hardware change can introduce (different kernel implementations, different default precision, different sampling implementation details) even with bit-for-bit identical weights — this is a real, easy-to-miss failure mode: "same model, different engine version, subtly different outputs."
- Load-test (Ch. 04) the new infra in isolation *before* canarying it on live traffic — infra canaries should catch capacity/config regressions that a synthetic load test already should have caught; live canary is the last safety net, not the first one.

---

### CD6: What's the difference between canary, blue-green, and shadow (dark) traffic deployment strategies for model updates?

**Answer:**
| Strategy | What happens | Risk exposure | Best for |
|---|---|---|---|
| **Canary** | Small % of *real* traffic gets *real* responses from the new version | Real users see new-version output, at small scale | Default choice — validates real behavior with bounded blast radius |
| **Blue-green** | Full traffic cutover from old to new at once (with fast rollback to "blue" if needed) | All users exposed simultaneously | When you need instant, all-or-nothing cutover and have high confidence from offline eval |
| **Shadow/dark traffic** | New version receives a *copy* of real traffic but its responses are discarded/logged, never shown to users | Zero user exposure | Validating latency/capacity/output-diffing on real traffic before any user ever sees the new version — great first step before a canary |

- Senior answer: these aren't mutually exclusive — a mature pipeline often does shadow traffic first (validate safely with zero exposure), then canary (validate with bounded real exposure), then full promotion — treating it as one all-or-nothing choice is itself a weaker answer.

---

### CD7: How do you handle stateful conversation context (multi-turn chat) correctly during a canary, so a user doesn't get inconsistent behavior mid-conversation?

**Answer:**
- Use **sticky routing** keyed on a session/conversation ID (consistent hashing at the gateway/mesh layer) so all turns of one conversation are routed to the same version (stable or canary) for the conversation's duration — otherwise a user can get wildly inconsistent behavior/style/quality switching between model versions turn-to-turn.
- This means canary traffic percentage is measured in *sessions*, not raw requests — a 5% canary should mean "5% of conversations," not "5% of individual turns randomly assigned," which would corrupt every multi-turn conversation it touches.
- Store which version handled a session (even just implicitly via consistent hashing) so post-hoc analysis of quality metrics can correctly attribute a whole conversation's outcome to one version.

---

### CD8: Interviewer pushback: "Why not just A/B test in production analytics after full rollout instead of doing all this canary infrastructure?"

**Answer:**
- Full rollout means 100% of users are exposed to any regression immediately and simultaneously — a canary's entire point is bounding blast radius *before* you have the full-traffic A/B data to know if there's a problem.
- Post-hoc A/B analysis on a bad full rollout tells you "how bad it was," not "let's avoid most users experiencing it" — by the time the analysis surfaces a problem, the damage (bad responses shown, cost incurred, trust eroded) already happened at 100% scale.
- The two aren't actually alternatives: canary is the *safe delivery mechanism*, A/B analysis is the *measurement layer* — you want both, canary literally *is* a (small, automatically-managed) live A/B test with a rollback trigger wired to its own results.

---

## Monitoring & Observability (Ch. 08)

*Maps to Chapter 08 — Monitoring (`prometheus_exporter.py`, `grafana_dashboard.json`, `prometheus.yml`).*

### Q14: What metrics should you monitor for LLM serving?

**Answer:**

**Application metrics:**
- **Request rate**: requests per second.
- **Latency**: P50, P95, P99 response times — and for LLMs specifically, TTFT and inter-token latency separately, not just end-to-end.
- **Error rate**: failed requests percentage.
- **Queue size**: pending requests waiting for a scheduler slot.
- **Throughput**: tokens per second (input and output).

**Model metrics:**
- **Generation time**: time to generate tokens.
- **Tokens generated**: average tokens per request.
- **Model version**: which model/version is currently running (essential for correlating any metric shift with a deploy).

**System metrics:**
- **GPU utilization**: GPU usage percentage.
- **GPU memory / KV-cache utilization**: used vs. total.
- **CPU usage / memory usage**: host-side resource pressure.

**Business metrics:**
- **Cost per request/token**: compute cost.
- **User satisfaction**: quality/feedback signals.
- **API usage**: requests by endpoint/tenant/model.

**Key dashboards:**
1. **Performance**: latency, throughput, errors.
2. **Resource**: GPU, CPU, memory usage.
3. **Model**: version-over-version performance comparison.

---

### M1: Design the Prometheus metric taxonomy for an LLM serving fleet — what should be a counter, gauge, or histogram?

**Answer:**
- **Counters** (monotonically increasing): total requests, total tokens generated, total errors by type — used to derive rates via `rate()`.
- **Gauges** (point-in-time value): current queue depth, current number of running/waiting sequences, GPU memory used, KV-cache block utilization — these represent instantaneous state, not cumulative totals.
- **Histograms** (bucketed distributions): request latency, TTFT, inter-token latency, tokens-per-request — anything you need percentiles for. Never use a gauge/average for latency where you'll want P95/P99 later; you can't recover percentiles from an average after the fact.
- Label carefully (model name, version, tenant) but watch cardinality (see M3) — labeling every metric by raw request ID or user ID would explode cardinality and can take down Prometheus itself.

---

### M2: What are the "four golden signals" and how do you adapt them specifically for LLM serving?

**Answer:**
- **Latency** → split into TTFT and inter-token latency, not just one end-to-end number, because they have different root causes (queueing/prefill vs. decode-step contention).
- **Traffic** → requests/sec *and* tokens/sec, since token volume (not just request count) is what actually drives GPU load.
- **Errors** → HTTP-level errors, plus engine-level failures (OOM, timeout, generation truncated unexpectedly) that a naive HTTP-status-only view would miss.
- **Saturation** → GPU utilization *and* KV-cache/block utilization *and* queue depth — for LLM serving, KV-cache saturation is frequently the real ceiling well before raw GPU compute utilization hits 100%, so tracking only GPU% can hide the actual bottleneck.

---

### M3: Your Prometheus instance keeps falling over / running out of memory. What's the likely cause in an LLM serving context and how do you fix it?

**Answer:**
- Almost always **cardinality explosion**: a label with unbounded/high-cardinality values (raw request ID, raw user ID, raw prompt hash) attached to a metric multiplies the number of unique time series, and Prometheus memory usage scales with active series count.
- Fix: never label by request ID; aggregate to a bounded label set (model, version, tenant-tier, status code) and push per-request detail (including request IDs) to logs/traces instead of metrics.
- Also check scrape interval and retention — a very short scrape interval combined with high series count compounds the problem; and consider a remote-write/long-term-storage backend (Thanos, Mimir, VictoriaMetrics) if retention needs exceed what a single Prometheus instance should hold locally.

---

### M4: How do you set actionable alert thresholds for LLM latency without causing alert fatigue?

**Answer:**
- Alert on **SLO burn rate**, not raw threshold crossings — e.g., "we're consuming our monthly error/latency budget X times faster than sustainable," which naturally adapts sensitivity (a brief blip doesn't page, a sustained regression does).
- Use multi-window, multi-burn-rate alerting (a fast/short window for "page now, this is severe" and a slower/longer window for "ticket, this is a slow leak") — this is the standard SRE pattern for exactly this problem.
- Alert on P99/P95, but tune the threshold from *your own* measured baseline (from load testing, Ch. 04) rather than an arbitrary industry number — "P95 < 500ms" is meaningless without knowing your model size/hardware/SLO context.
- Route alerts by actual actionability: "GPU node down" pages on-call; "quality-eval score dipped 2%" goes to a ticket/dashboard for the model team, not a 3am page.

---

### M5: How do you monitor and alert on cost, not just performance, for an LLM serving platform?

**Answer:**
- Track cost-per-request and cost-per-1K-tokens as first-class metrics, derived from GPU-hour cost divided by observed throughput — a performance regression that silently doubles cost-per-token (e.g., GPU utilization dropped due to a bad batching config) is a real incident even if latency/error metrics look fine.
- Alert on GPU utilization *sustained* well below expected baseline (e.g., "10 GPUs running at 20% utilization for an hour") — that's wasted spend, and it's an alertable condition just like an error-rate spike, just on a different axis.
- Tag/attribute cost by tenant or model where multi-tenant, so a runaway single customer/feature doesn't hide inside an aggregate cost number until the monthly bill arrives.

---

### M6: What's the role of distributed tracing (OpenTelemetry) in an LLM serving stack, beyond metrics and logs?

**Answer:**
- Metrics tell you *that* P99 latency is bad; tracing tells you *where* the time went for a specific slow request — e.g., time in the gateway, time queued waiting for a scheduler slot, time in prefill, time in decode, time in a downstream retrieval call for RAG.
- Especially valuable for multi-hop architectures (gateway → router → serving pod → possibly a retrieval service/tool call for agentic use cases) where "one slow request" could be slow at any of several hops, and aggregate metrics per-hop won't tell you which specific requests correlate a slowdown across hops.
- Practical setup: propagate a trace ID from the gateway through to the serving engine and any downstream calls (most serving engines support OpenTelemetry instrumentation or can be wrapped to emit spans), and sample at a rate that balances trace-storage cost against debuggability (100% sampling is rarely affordable at scale; sample up on errors/high latency).

---

### M7: How do you build a monitoring setup that catches a *quality* regression (not just a latency/error regression) in production, continuously — not just during a canary window?

**Answer:**
- Continuously sample a small percentage of live production outputs (not just canary-window traffic) and run them through cheap automated quality signals: rule-based checks (format validity, length bounds, refusal detection), a lightweight LLM-as-judge scoring a sample against rubrics, or comparison against known reference outputs for a fixed regression-test prompt set replayed on a schedule.
- Track these as time series just like latency/error rate, with the same alerting discipline (M4) — a slow quality decay (e.g., from an unnoticed upstream data/prompt-template change, not just a bad deploy) needs the same "burn rate" style detection as a latency SLO, because it won't show up as a sudden spike.
- This is the direct link to Drift Detection (Ch. 10): a quality-regression signal *is* often the first observable symptom of underlying input or behavior drift, well before a formal statistical drift test flags it.

---

## Model Versioning & Registry (Ch. 09)

*Maps to Chapter 09 — Model Versioning. Q19 is preserved from the original System Design section — versioning is really its own topic, not a system-design scenario.*

### Q19: How would you implement model versioning and rollback?

**Answer:**

**Versioning strategy:**
1. **Semantic versioning**: v1.0.0, v1.1.0, v2.0.0 — decide what counts as major (architecture/behavior change) vs. minor (fine-tune update) vs. patch (config/quantization change only).
2. **Model registry**: store models with metadata (training date, eval metrics, dataset version, base model lineage).
3. **Metadata tracking**: training date, metrics, dataset, who approved promotion.

**Implementation:**

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

**Best practices:**
- **Test before deploy**: validate the new version offline before it's even canary-eligible.
- **Gradual rollout**: use canary deployment (Ch. 07), don't cut over 100% at once.
- **Monitor**: track version performance continuously, not just at rollout time.
- **Document**: maintain a changelog per version, including what data/eval it was validated against.

---

### MV1: What actually belongs in a model registry's metadata, beyond "the weights file"?

**Answer:**
- **Lineage**: base model + fine-tuning/adapter data used, training config, exact commit/config hash that produced this artifact.
- **Evaluation results**: the specific benchmark/eval-suite scores this version achieved, so any two versions are comparable on the same axes.
- **Serving compatibility**: which engine version(s)/quantization formats this checkpoint is validated against — a checkpoint that works fine in FP16 on vLLM 0.6.x isn't automatically guaranteed to behave identically under a different engine or quantization scheme.
- **Approval/promotion metadata**: who approved it for canary/production, when, and against what criteria — this is what makes rollback and audit possible after the fact, not just "we have the file."
- **Content hash/checksum**: to detect silent corruption or an accidental wrong-file upload before it ever reaches a serving pod.

---

### MV2: Should model artifacts be treated as immutable once published, and what does that discipline actually buy you?

**Answer:**
- Yes — once `v1.2.0` is published to the registry, it should never be overwritten in place. Need a fix? Publish `v1.2.1`.
- Immutability is what makes rollback trustworthy: "roll back to v1.1.0" only means something reliable if v1.1.0 today is bit-for-bit the same thing that was validated and ran in production before.
- It also makes debugging possible after the fact — if artifacts are mutable, "which model actually produced this bad output six weeks ago" becomes unanswerable.
- Implementation: content-addressed storage or write-once buckets with versioned object keys, registry API that rejects overwrite of an existing version tag.

---

### MV3: How do you decouple "code version" (serving engine/app) from "model version" (weights) in your deployment pipeline?

**Answer:**
- Serving image version and model version should be independent axes, each with its own release cadence — a model fine-tune shouldn't require a full application rebuild/redeploy, and an engine upgrade shouldn't require touching the model artifact.
- Practical mechanism: the serving image is generic (loads whatever `MODEL_PATH`/`MODEL_VERSION` it's told to at startup, per Q10/Q19), and model version is injected via config (env var, ConfigMap, or a dedicated model-serving CRD if using an operator) — this is exactly why Q10 recommended runtime weight-mounting over baking weights into the image.
- This decoupling is also what makes independent canarying possible: you can canary a new model version on the same engine version, or canary a new engine version on the same model weights (CD5), rather than being forced to change both at once and losing the ability to attribute a regression to either.

---

### MV4: Design the rollback SLA — how fast can/should a rollback actually happen, and what determines that floor?

**Answer:**
- Fastest possible rollback (traffic-routing only, e.g., flipping a canary weight back to 0% or a router config) can be near-instant (seconds) if the "stable" version's pods are still warm and running — this is why keeping the previous version's pods alive for a fast-rollback window after a promotion (rather than immediately scaling them down) is a deliberate, worthwhile cost trade-off.
- If the previous version's pods have already been scaled down, rollback now requires re-provisioning capacity — meaning rollback time equals cold-start time (K7), which can be minutes, not seconds. This is a common gap between "we have rollback capability" (true) and "our rollback SLA is under a minute" (false, unless you kept the old version warm).
- Decision to communicate to an interviewer: pick a fast-rollback retention window (e.g., keep N-1 warm for X hours/days after a promotion) sized against your risk tolerance and cost budget — it's an explicit trade-off, not a free win.

---

### MV5: How do you version and roll back *prompts/system prompts* and *retrieval configuration*, not just model weights, given that both affect output just as much?

**Answer:**
- Treat prompt templates, system prompts, and RAG retrieval configs (index version, chunking parameters, retrieval top-k) as versioned artifacts with the same discipline as model weights — a "model regression" investigation that only checks weight version and ignores a same-day prompt-template change will miss the actual cause.
- Store prompt/config versions alongside model version in the registry metadata so a given production request can be attributed to an exact (model, prompt template, retrieval config) triple, not just a model version.
- This matters for canary too: changing a system prompt is a change worth canarying with the same rigor as a model swap (CD1-CD8 apply equally), since it can shift output behavior just as much as a weight change — a common trap is treating prompt changes as "just a config tweak" that skips the deployment safety pipeline entirely.

---

### MV6: What does "shadow deployment" buy you specifically for model *version* validation that offline eval doesn't?

**Answer:**
- Offline eval runs against a fixed, historical dataset — it can't tell you how the new version behaves on *today's* real, live traffic distribution, which may have already shifted from whatever the offline eval set represents.
- Shadow deployment (CD6) runs the new version against live traffic in parallel with production, discarding its output (never shown to users) but logging it for comparison against the current production version's actual response on the same input — this catches distribution-shift-sensitive regressions offline eval structurally can't see.
- Cost: doubles compute for the shadowed traffic slice (both versions run the same requests) — usually done on a sampled percentage of traffic, not all of it, and only for a bounded validation window before moving to a real (traffic-splitting) canary.

---

### MV7: Interviewer pushback: "Why do you need a formal model registry at all — why not just tag Docker images with the model baked in and use normal image-based deployment tooling?"

**Answer:**
- Baking weights into images (Q10) couples model release cadence to image build/push/pull cost and time, which is wasteful when model updates (e.g., frequent fine-tunes) are much more frequent than code changes, and vice versa.
- A registry gives you model-specific metadata (eval scores, lineage, approval status) that a generic container registry has no concept of — you'd end up building a parallel ad hoc tracking system anyway, just without the tooling (rollback API, promotion gates) built around it.
- That said, this is a real trade-off, not a dogma: for a small team with infrequent model updates and no multi-tenant/multi-model complexity, baking weights into a versioned image *is* a legitimate, simpler answer — the senior signal is recognizing *when* the added complexity of a registry earns its keep (frequent updates, multiple models, need for fast independent rollback) versus when it's over-engineering.

---

## Drift Detection (Ch. 10)

*Maps to Chapter 10 — Drift Detection (`drift_detector.py`). Q15 is preserved from the original Monitoring section — drift detection is really its own chapter/topic.*

### Q15: How would you detect model drift in production?

**Answer:**

**Types of drift:**
1. **Data drift**: input distribution changes (e.g., users start asking about topics/formats the model rarely saw before).
2. **Concept drift**: the input-output relationship changes (what counts as a "correct" answer shifts, e.g., because of a real-world event or policy change).
3. **Prediction drift**: the output distribution changes (e.g., the model starts generating longer, shorter, or differently-structured responses).

**Detection methods:**

**1. Statistical tests:**
- **PSI (Population Stability Index)**: compare reference vs. current distributions.
- **Kolmogorov-Smirnov**: test for distribution differences.
- **Chi-square**: test categorical distribution differences.

**2. Monitoring tools:**
- **Evidently AI**: open-source drift detection.
- **Prometheus**: custom drift-score metrics.
- **Custom scripts**: compare reference vs. current data directly.

**3. Implementation:**
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

**4. Alerting:**
- Set thresholds (e.g., PSI > 0.2).
- Monitor continuously, not just at deploy time.
- Alert on drift detection and investigate causes.

**Actions:**
- **Retrain/refresh the model**: if drift is significant and persistent.
- **Investigate**: understand *why* drift occurred before reacting.
- **Update the baseline**: update reference data if the shift represents a legitimate new normal, not a problem.

---

### DR1: For an LLM specifically (as opposed to a classic tabular ML model), what does "input drift" actually mean, and how do you measure it over free-form text?

**Answer:**
- You can't run PSI/KS tests directly on raw text the way you would on a numeric feature — you need a numeric representation first: prompt length distribution, embedding-space distance (embed prompts, track the centroid/distribution shift of the embedding cloud over time vs. a reference window), topic/intent classification distribution (if you have a classifier or clustering over intents), or simple proxy features (language detected, presence of certain keywords/domains).
- Embedding-based drift (compare embedding distributions via a distance metric like Maximum Mean Discrepancy, or simpler centroid-distance-over-time) is the most common practical approach for free-form text drift because it captures semantic shift without needing hand-built categorical features.
- Practical warning: embedding drift detectors need their own maintenance — if you change your embedding model, your drift baseline is invalidated and needs to be re-established, which is an easy-to-miss operational dependency.

---

### DR2: What is "prediction drift" for a generative model, and why is it a *different* problem from input drift?

**Answer:**
- Prediction drift here means the *output* distribution shifts — e.g., average response length changes, refusal rate changes, sentiment/tone changes, or the rate of a particular failure mode (repetition loops, format violations) changes — even if the input distribution looks stable.
- This can happen with **zero code or model change**: an upstream prompt-template tweak, a change in retrieved-context quality for RAG, or even a subtle change in default sampling parameters can shift output distribution independent of any input shift.
- Why it matters separately: input drift tells you "the world/users changed"; prediction drift tells you "the system's behavior changed" — you need both signals because a system can have a completely stable input distribution while output behavior silently regresses (this is often the *first* symptom of a bug, well before anyone notices via input analysis).

---

### DR3: How would you set up automated drift-triggered retraining/refresh, and what are the dangers of doing this fully automatically?

**Answer:**
- Pipeline shape: drift monitor computes a score on a schedule → score crosses threshold → triggers a retraining/fine-tuning job on fresh data → new model version goes through the normal eval + canary pipeline (never skips straight to production, drift-triggered or not).
- Danger #1: **feedback loops** — if you retrain on production outputs (or worse, on user reactions to production outputs) without careful filtering, you can bake in and amplify whatever drift/bias caused the trigger in the first place ("model drifts toward sycophancy → retrains on sycophantic outputs users engaged with more → drifts further").
- Danger #2: **false-positive-triggered churn** — an overly sensitive drift threshold causes constant retraining on noise, burning compute and, worse, introducing instability (each new version is itself a source of behavior change/risk) for no real benefit.
- Senior answer: drift detection should trigger an **alert/investigation**, and retraining should be a *human-gated* decision informed by that investigation, not a fully automatic trigger straight to production, precisely because of the above failure modes — full automation is a good long-term goal only once you have strong enough guardrails (eval suites, canary, rollback) that a bad automated retrain can't actually reach users.

---

### DR4: What's the difference between monitoring for drift and monitoring for outright *model degradation/failure* (e.g., repetition loops, refusals, garbage output)?

**Answer:**
- Drift is a *distributional* concept — comparing distributions over time/reference — and can be entirely benign (the world legitimately changed) or gradual.
- Degradation/failure detection is about catching acute, often binary bad behavior on individual outputs — repetition loops, empty outputs, malformed JSON when structured output was requested, refusal on a benign request — which needs per-request rule-based checks, not distributional statistics.
- Both matter, and they complement each other: drift detection catches slow, systemic shifts that individual per-request checks won't notice (each individual output looks "fine" but the aggregate has moved); per-request failure detection catches acute breakage that a distributional test averaged over thousands of requests might dilute into invisibility.

---

### DR5: How do you build a drift baseline/reference distribution in the first place, and how often should you refresh it?

**Answer:**
- Reference distribution is typically built from a recent, validated window of production traffic (or the training/fine-tuning data distribution, if traffic hasn't started yet) — not from an arbitrarily old snapshot, since the "reference" needs to represent a period you're confident was healthy.
- Refresh cadence is a trade-off: refresh too often and you'll normalize away real, slow drift (comparing "yesterday to today" will basically never show drift, because it re-baselines constantly and hides gradual multi-week shifts); refresh too rarely and normal seasonal/legitimate distribution changes get flagged as false-positive "drift" forever.
- Common pattern: maintain a fixed reference window (e.g., "the traffic from the last validated release") until a human explicitly promotes a new reference window (typically alongside a model/config version bump) — refresh is a deliberate, reviewed action, not a rolling automatic window, for the same reason immutable model versions matter (MV2).

---

### DR6: How does drift detection interact with multi-tenant serving, where different tenants have very different, legitimately-different traffic patterns?

**Answer:**
- A single global drift baseline across all tenants will constantly show "drift" simply because tenant mix shifts (a big customer's usage pattern differs from another's, and their relative traffic volume changes week to week) — this is noise, not signal, at the aggregate level.
- Better: maintain per-tenant (or per-segment) baselines and drift scores, and only escalate a global alert if drift appears *within* a tenant's own traffic over time, or if a large enough fraction of tenants show it simultaneously (which suggests a genuinely shared cause — e.g., a shared model/prompt-template change — rather than tenant-mix noise).
- This is directly analogous to why per-tenant cost/latency dashboards matter (M5) — aggregate metrics across a heterogeneous multi-tenant population hide exactly the signals you need.

---

### DR7: Interviewer pushback: "PSI and KS tests are from classical tabular ML monitoring — do they even make sense for LLMs, or is this cargo-culting a technique from a different problem?"

**Answer:**
- Fair pushback, and the honest answer is "partially" — PSI/KS are well-suited to comparing distributions of *derived numeric features* (prompt length, embedding-distance scores, response length, latency) and remain genuinely useful there; they're not directly applicable to raw text without that feature-extraction step first.
- What's LLM-specific and *not* borrowed from classical ML monitoring: output-quality proxies (guardrail trigger rate, refusal rate, LLM-as-judge scores on sampled outputs), which have no real analogue in classic tabular drift detection and are arguably the more important signal for generative systems.
- Best answer combines both: classical statistical tests on well-chosen derived numeric features (cheap, well-understood, good for continuous monitoring) *plus* LLM-specific quality-eval sampling (more expensive per-sample, but the ground truth signal that PSI/KS on embeddings is only ever a proxy for) — leaning on only one of the two leaves a real blind spot.

---

## Triton Inference Server (Ch. 11)

*Maps to Chapter 11 — Triton.*

### T1: What problem does Triton Inference Server solve that a hand-rolled FastAPI + PyTorch server doesn't?

**Answer:**
- **Multi-framework, multi-model serving from one process**: Triton can serve PyTorch, TensorFlow, ONNX, TensorRT, and (via the vLLM/TensorRT-LLM backends) LLM-specific engines side by side, behind one unified gRPC/HTTP API — useful when your platform serves more than just LLMs, or multiple LLM engines.
- **Dynamic batching and concurrent model execution** built into the server itself, configurable per-model without hand-writing scheduling logic.
- **Model repository abstraction**: models are deployed by dropping them into a directory structure with a config file — versioning, loading/unloading, and multi-version serving (for canary/A-B) are handled by the server's model-management API rather than custom application code.
- **Production-grade observability out of the box**: built-in Prometheus metrics, model-load/unload lifecycle events, per-model and per-version statistics.
- Trade-off: more operational surface area and a specific configuration model to learn (`config.pbtxt`) vs. the simplicity of a hand-written server — worth it once you have multiple models/frameworks/versions to manage consistently, arguably overkill for a single model with a single simple serving path.

---

### T2: Describe the Triton model repository layout and what `config.pbtxt` controls.

**Answer:**
```
model_repository/
  my_model/
    config.pbtxt
    1/
      model.onnx
    2/
      model.onnx
```
- Each top-level directory under the repository is a model name; numbered subdirectories are versions, letting Triton serve multiple versions of the same model simultaneously (directly useful for canary/A-B — Ch. 07 concepts apply at the Triton layer too).
- `config.pbtxt` controls: input/output tensor names, shapes, and datatypes; the backend to use (e.g., `onnxruntime`, `pytorch`, `tensorrt`, `vllm`, `python` for a custom backend); batching configuration (max batch size, dynamic batching parameters); instance groups (how many copies of the model to run, on which GPU(s)); and version policy (serve latest only, serve all, serve a specific set).

---

### T3: Explain Triton's dynamic batching vs. vLLM's continuous batching — are these the same idea?

**Answer:**
- **Triton dynamic batching** (the general-purpose feature) accumulates individual inference requests arriving within a short configurable window into a single batch before running one forward pass — this works well for models with a fixed-shape, single-forward-pass execution profile (classification, embedding, non-autoregressive models).
- **Continuous/iteration-level batching** (vLLM's core idea, also implemented via Triton's dedicated LLM backends) is specific to *autoregressive* generation: it batches at the level of individual decode *steps*, not whole requests, letting requests join/leave the batch mid-generation (Q7) — plain dynamic batching, which waits for a batch window then runs one fixed pass, doesn't fit autoregressive generation well because requests finish at wildly different times (different output lengths).
- Practical implication: when serving LLMs *through* Triton, you use the **vLLM backend** or **TensorRT-LLM backend** for Triton, which brings continuous-batching-style scheduling into Triton's process — plain generic dynamic batching is the wrong tool for autoregressive LLM decode.

---

### T4: What are Triton's ensemble models, and when would you use one in an LLM serving pipeline?

**Answer:**
- An ensemble model defines a **pipeline of models/steps** (e.g., preprocessing → embedding model → retrieval step → LLM generation → postprocessing) as a single logical model that Triton executes as a DAG, handling data hand-off between steps internally.
- Useful for a RAG pipeline where you want a single API call to trigger embed-query → retrieve → generate, with Triton managing the scheduling/batching at each stage rather than the client making three separate round-trip calls.
- Trade-off: debugging an ensemble's per-stage behavior is harder than debugging separate services (all stages are one opaque pipeline from the client's view), and stage-level independent scaling is more constrained than if each stage were its own separately-scaled service/Deployment — good for tightly coupled, latency-sensitive pipelines; a poor fit if stages have very different scaling/failure characteristics that benefit from independent operational treatment.

---

### T5: How does Triton fit into a Kubernetes deployment, and what does the "instance group" concept map to?

**Answer:**
- Triton itself runs as a normal container in a Kubernetes Deployment/pod, typically with `nvidia.com/gpu` requested like any other GPU workload (Q11/K1) — Kubernetes doesn't need to know anything Triton-specific; Triton just presents as a gRPC/HTTP service.
- **Instance groups** in `config.pbtxt` control how many copies of *a given model* Triton runs *within one Triton process/pod*, and on which specific GPU(s) — this is a Triton-internal multi-model/multi-instance concept, distinct from and complementary to Kubernetes-level replica scaling (which runs multiple *pods*, each potentially running multiple Triton instance groups).
- Practical design: use instance groups to pack multiple smaller models (or multiple copies of one model) efficiently onto the GPUs available *within* a pod, and use Kubernetes HPA/replicas (Ch. 06) to scale the *number of pods* for overall fleet capacity — conflating the two layers is a common early mistake ("just add instance groups" when you actually need more pods, or vice versa).

---

### T6: When would you choose Triton + a TensorRT-LLM backend over a plain vLLM deployment for LLM serving?

**Answer:**
- **TensorRT-LLM via Triton**: typically the highest raw throughput/lowest latency option on NVIDIA hardware because TensorRT-LLM compiles model-and-hardware-specific optimized kernels (graph fusion, precision-specific kernel selection) ahead of time — but it requires an explicit build/compile step per model+GPU-generation+precision combination, has a steeper operational learning curve, and is less flexible for rapidly iterating on new/unusual model architectures.
- **vLLM** (standalone, or via Triton's vLLM backend): faster to get a new/custom model serving, simpler operational model, very strong out-of-the-box performance without a separate compile step, and a faster-moving open-source ecosystem for new model-architecture support.
- **Rule of thumb**: reach for TensorRT-LLM when you have a small number of well-known, stable, high-volume models where the extra compile/ops investment pays for itself in throughput/cost at scale; default to vLLM (with or without Triton in front) when you need to iterate quickly across many/changing models — this exact "raw performance vs. iteration speed" trade-off is a good thing to name explicitly in an interview.
- Note this space moves fast (TensorRT-LLM's own build/deploy ergonomics have been improving, and NVIDIA NIM packages this whole trade-off as a pre-built container) — flag specific current capability claims as "verify before quoting."

---

### T7: How does Triton support A/B testing or canary between model versions natively?

**Answer:**
- Because Triton's model repository natively supports multiple numbered versions of the same model name (T2) with a configurable version policy, you can load two versions simultaneously and route between them.
- Native Triton doesn't do traffic-percentage splitting itself — that routing decision typically happens at a layer in front of Triton (an API gateway, a service mesh, or application logic choosing which version to call) — Triton's job is to have both versions loaded and ready to serve whichever one is called, not to decide the split itself.
- This means the canary *mechanics* (Ch. 07: traffic weighting, automated analysis, rollback triggers) still apply exactly as described — Triton just changes where "which version is currently loaded and serving" lives (inside Triton's model repository) versus (in a non-Triton setup) as separate Kubernetes Deployments.

---

### T8: What monitoring does Triton expose natively, and how do you wire it into a Prometheus/Grafana stack (Ch. 08)?

**Answer:**
- Triton exposes a built-in `/metrics` endpoint in Prometheus format with per-model, per-version statistics: request count, inference duration (queue time vs. compute time, broken out — directly useful for diagnosing whether a slowdown is scheduling/queueing or actual compute), GPU utilization/memory, and success/failure counts.
- Wire it in exactly like any other Prometheus target (`prometheus.yml` scrape config pointed at the Triton pod's metrics port) — this is one of Triton's practical advantages over a fully hand-rolled server: you don't have to build this instrumentation yourself, it's part of the platform.
- Because Triton separates queue time from compute time in its metrics by default, it directly supports the golden-signal breakdown described in M2 (saturation vs. latency) without custom instrumentation work.

---

### T9: Interviewer pushback: "If vLLM alone already gives you continuous batching, PagedAttention, and an OpenAI-compatible API, what does adding Triton in front actually buy you — isn't it redundant infrastructure?"

**Answer:**
- For a single-model, single-framework, LLM-only deployment, it can genuinely be redundant — running vLLM's own OpenAI-compatible server directly (no Triton) is a perfectly reasonable, simpler answer, and a senior engineer should say so rather than defaulting to "always use Triton."
- Triton earns its keep when you need **one unified serving layer across heterogeneous models/frameworks** (LLMs via vLLM/TensorRT-LLM backend, plus classical models via ONNX/TensorRT, plus embedding models, all behind one API/ops surface), or when you specifically need Triton-level features (ensembles, fine-grained instance-group GPU packing, native multi-version model repository) that a standalone engine's server doesn't provide.
- Good interview answer: name the trade-off explicitly ("Triton adds an abstraction/ops layer that pays for itself with model/framework diversity, and is unnecessary overhead for a single homogeneous LLM deployment") rather than treating either "always Triton" or "never Triton" as a universal rule.

---

## Production Best Practices

### Q16: What are the key considerations for production LLM serving?

**Answer:**

**1. Performance:**
- **Latency**: define an explicit SLO (e.g., P95 TTFT < 300ms) rather than a generic "fast" target.
- **Throughput**: handle expected load, validated via load testing (Ch. 04).
- **Scalability**: auto-scale based on demand (Ch. 06).

**2. Reliability:**
- **Health checks**: liveness, readiness, and startup probes (Q5e).
- **Error handling**: graceful degradation, not hard failure.
- **Circuit breakers**: prevent cascade failures into downstream/upstream systems.
- **Retries**: with exponential backoff and jitter, bounded so retries don't amplify an overload.

**3. Monitoring:**
- **Metrics**: comprehensive observability (Ch. 08).
- **Logging**: structured logging with request/trace correlation.
- **Alerting**: proactive, actionable, SLO-burn-rate-based (M4).
- **Tracing**: request tracing across hops (M6).

**4. Security:**
- **Authentication**: API keys, OAuth, mTLS between internal services.
- **Rate limiting**: prevent abuse and protect GPU capacity.
- **Input validation**: sanitize inputs (Q5f).
- **Secrets management**: never bake API keys into images; use Kubernetes Secrets/a secrets manager.

**5. Cost:**
- **Resource optimization**: right-size instances (Q20 below).
- **Auto-scaling**: scale down when not needed.
- **Model optimization**: use efficient models/quantization where quality allows.

**6. Model management:**
- **Versioning**: track model versions (Ch. 09).
- **A/B testing**: compare model performance on real traffic (Ch. 07).
- **Rollback**: quick revert capability (MV4).
- **Drift detection**: monitor model degradation (Ch. 10).

---

### Q17: How would you handle a sudden spike in traffic?

**Answer:**

**Immediate actions:**
1. **Auto-scaling**: HPA/KEDA should scale up automatically (Ch. 06) — but recognize this has real latency (AS4).
2. **Load balancing**: distribute across pods, ideally prefix/cache-aware (V6).
3. **Queue management**: queue requests with a bounded queue if needed.
4. **Rate limiting**: protect the backend from overload — reject cheaply rather than let everyone degrade.

**Prevention:**
1. **Capacity planning**: understand max sustainable capacity per replica (LT7).
2. **Load testing**: test under expected and above-expected load (Ch. 04).
3. **Auto-scaling**: configure HPA/KEDA properly, with predictive scaling for known patterns (AS5).
4. **Circuit breakers**: prevent cascade failures.

**Monitoring during a spike:**
- Watch pod count and node provisioning progress.
- Monitor latency (P95, P99) split by TTFT vs. inter-token.
- Check error/reject rates.
- GPU and KV-cache utilization.

**If overwhelmed:**
1. **Degrade gracefully**: shed to a smaller model or cap `max_tokens` automatically.
2. **Rate limit**: reject excess requests with a clear, fast 429, not a slow timeout.
3. **Scale manually**: if auto-scaling is insufficient or too slow.
4. **Add capacity**: more nodes/GPUs, potentially spot/on-demand mix.

**Post-incident:**
- Analyze what happened (use the burst as capacity-planning input, per AS6).
- Improve auto-scaling config and warm-buffer sizing.
- Increase baseline capacity if the new traffic level looks sustained, not a one-off.
- Document learnings and update the runbook.

---

### Q20: Explain how you would optimize costs for LLM serving.

**Answer:**

**Cost components:**
1. **Compute**: GPU/CPU instances — usually the dominant cost by far for LLM serving.
2. **Storage**: model storage, registry, logs/traces.
3. **Network**: data transfer, especially cross-region/cross-cloud.
4. **Monitoring**: observability tooling overhead.

**Optimization strategies:**

**1. Right-sizing:**
- Use appropriate instance/GPU types for the actual model size and concurrency needed.
- Don't over-provision "just in case" — use load-test data (LT7) to size deliberately.

**2. Auto-scaling:**
- Scale down during low traffic.
- Scale up only when needed, with predictive scaling for known patterns.
- Use spot/preemptible instances for non-latency-critical or batch workloads.

**3. Model optimization:**
- **Quantization**: FP8/INT8/INT4 (smaller, faster — see V5 for the current landscape).
- **Distillation**: use a smaller model where quality allows.

**4. Caching:**
- Prefix/KV-cache reuse for repeated context (V6) — often the single highest-leverage cost lever for chat/RAG workloads.
- Cache common responses where semantically valid.

**5. Batch processing:**
- Batch requests when possible; use continuous batching (Q7).
- Higher GPU utilization = lower cost per request/token — track this as a metric (M5), not just "faster."

**6. Monitoring:**
- Track cost per request/token continuously.
- Identify and alert on expensive/wasteful operations (M5).

**7. Reserved/committed capacity:**
- Commit to usage for discounts on predictable baseline load, use spot/on-demand only for burst.

**Example:**
- Before: 10 GPUs, 50% utilization = a given monthly cost.
- After: 5 GPUs, 90% utilization (via better batching/prefix caching) = roughly half that cost for the same served traffic.
- **Illustrative savings: ~50%** — validate with your own numbers; exact ratios depend heavily on workload shape.

---

## Additional Quick Questions

### Q21: What is the difference between batch size and sequence length?

**Answer:**
- **Batch size**: number of requests/sequences processed together.
- **Sequence length**: number of tokens in a single request (prompt + generated so far).

**Example:**
- Batch size = 8: process 8 requests simultaneously (in traditional batching) or concurrently in-flight (continuous batching).
- Sequence length = 512: each request has up to 512 tokens.

**Impact:**
- Larger batch size → higher throughput, more memory (more KV cache resident at once).
- Longer sequence length → more computation and more memory (KV cache grows linearly with sequence length per sequence).

---

### Q22: How does quantization affect model performance?

**Answer:**
Quantization reduces model precision to save memory and speed up inference.

**Types:**
- **FP32 → FP16/BF16**: ~2x smaller, faster, minimal accuracy loss.
- **FP16/BF16 → FP8**: ~2x smaller again, meaningful speedup on Hopper/Blackwell-class tensor cores, small workload-dependent accuracy impact.
- **FP16 → INT8**: ~2x smaller, faster, small accuracy loss, may need calibration.
- **INT8 → INT4**: ~2x smaller again, faster, larger accuracy loss — validate carefully per task.

**Trade-offs:**
- **Pros**: faster inference, less memory, lower cost.
- **Cons**: potential accuracy loss, may need calibration data and per-task validation, not just perplexity.

**When to use:**
- Production, when speed/cost matters and quality has been validated as acceptable for the target task.
- After validating accuracy on your actual eval suite, not just a generic benchmark.
- For memory-constrained deployment (smaller GPUs, higher concurrency targets).

---

### Q23: What is the difference between model parallelism and data parallelism?

**Answer:**

**Data parallelism:**
- Same model replicated on multiple GPUs/nodes.
- Different data (different requests, in inference) on each replica.
- Used heavily in training (with gradient sync); in inference, this is just "run N independent replicas" — which is what Kubernetes replica scaling (Ch. 06) does.

**Model parallelism:**
- Split model across multiple GPUs (tensor parallelism within a layer, pipeline parallelism across layers — V4).
- Each GPU holds part of the model.
- Used for large models that don't fit on one GPU, or that need the combined memory bandwidth of multiple GPUs to hit a decode-latency target.

**Example:**
- **Data parallel (inference)**: 8 GPUs, each running an independent full copy of a 7B model, serving different requests — this is just "8 replicas."
- **Model parallel (inference)**: 8 GPUs, each holding 1/8 of a 70B+ model's weights (tensor-parallel), collectively serving one request's forward pass together.

---

## System Design Scenarios

*This section replaces the original thin "System Design" section with 5 full worked scenarios. Q18 (the original quick-reference design question) is preserved below as a compact warm-up; use it if you only have two minutes, and use the full scenarios if you have twenty.*

### Q18: Design a system to serve LLMs at scale (quick reference).

**Answer:**

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

**Clarifying questions to ask first:**
- What's the expected request volume/concurrency (peak and average)? This determines whether we're capacity-planning for 10 req/s or 10,000 req/s — completely different architectures.
- What's the prompt-length distribution (short chat turns vs. long RAG contexts)? Drives prefill cost and chunked-prefill tuning.
- Is streaming required? (Almost always yes for chat UX — affects the TTFT-vs-end-to-end SLO split.)
- Is there a hard availability SLA (e.g., 99.9%) that constrains how much redundancy we need to build in?
- Budget ceiling — is "lowest cost" a soft preference or a hard number we're solving against?

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

**Key design decisions and trade-offs:**
- **Tensor parallelism degree**: a 70B model in FP16 needs ~140GB just for weights — doesn't fit on one 80GB GPU. TP=4 across 4x 80GB-class GPUs (NVLink) is a common fit; TP=2 with FP8 weights (~70GB) might fit on 2 GPUs with headroom for KV cache, trading some quality risk for half the GPU count per replica — this exact trade-off (TP degree vs. quantization vs. cost) is the single most important number in this scenario, and I'd validate the FP8 quality trade with an eval suite before committing.
- **KV cache budget**: at 500-token outputs and target concurrency, KV cache is likely the binding constraint on how many concurrent sequences one replica can hold — size GPU memory allocation (`gpu_memory_utilization` fraction) and consider FP8 KV cache to roughly double effective concurrency per replica.
- **Chunked prefill**: with mixed prompt lengths, enable it so occasional long prompts don't spike inter-token latency for concurrent shorter-output users — directly protects the P95 TTFT SLO under mixed load.
- **Prefix caching + cache-aware routing**: if there's a shared system prompt or repeated RAG context, this can be the single highest-leverage cost/latency lever — cuts prefill cost for a large fraction of requests.
- **Replica count**: from load-test-derived per-replica capacity (LT7) at the target SLO, not from a theoretical FLOPs calculation — always validate the SLO-constrained concurrency ceiling empirically before sizing the fleet.

**Cost levers, ranked:**
1. Prefix caching (near-free once implemented, can cut a large fraction of prefill cost for chat/RAG).
2. FP8 quantization of both weights and KV cache (validate quality first).
3. Right-sizing TP degree against actual concurrency need, not "the biggest GPU count available."
4. Spot/preemptible capacity for any part of the fleet that can tolerate interruption (rarely the primary serving path itself, but useful for a batch/overflow tier).

**Defending against pushback:**
- *"Why not just use 8 GPUs per replica for maximum headroom?"* — More GPUs per replica means more idle capacity most of the time (over-provisioned for average load) and coarser scaling granularity (each scale-up step adds 8 GPUs' worth of cost, not 4) — right-size TP degree to what the model and target concurrency actually need, and use replica *count* (not TP degree) as the scaling knob for load.
- *"Why not just cache everything to avoid the GPU cost entirely?"* — Caching helps for repeated content (shared prefixes) but doesn't help a genuinely novel prompt; for open-ended chat/generation with unique-per-request content, GPU inference cost is a floor you can reduce (quantization, batching efficiency) but not eliminate via caching alone.
- *"How do you know your load-test numbers will hold in production?"* — They won't exactly, which is why the design includes headroom (N+1 replicas), real-time capacity-based autoscaling as a second line of defense, and continuous production latency monitoring to catch drift from the load-test assumptions early.

---

### Scenario 2: Design a multi-tenant inference platform serving several model sizes to different internal teams

**Prompt as given in an interview:** "Several teams want to use a shared LLM platform: one needs a small 7B model for high-volume simple tasks, one needs a 70B model for complex reasoning, one is experimenting with a new fine-tune weekly. Design the platform."

**Clarifying questions to ask first:**
- Do tenants need hard isolation (compliance/security boundary between teams) or just fair resource sharing?
- Is cost attribution per-tenant required (chargeback/showback)?
- How often do new models/fine-tunes get added — is this a fixed catalog or a constantly-changing one?
- Are there different latency SLOs per tenant/use case, or one shared SLO?

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

**Key design decisions and trade-offs:**
- **Isolation strategy**: use Kubernetes namespaces per tenant for policy/quota isolation, and MIG or DRA-managed GPU partitions (K3) where hard resource isolation matters (e.g., the experimental fine-tune shouldn't be able to starve the 7B production pool of GPU memory on a shared node) — for the 70B pool, dedicated whole-GPU nodes make more sense since MIG-slicing a large model's TP group adds complexity for little benefit.
- **Priority-based preemption**: give production-traffic pools (7B, 70B) higher `PriorityClass` than the experimental pool, so under GPU pressure, Kubernetes preempts the experimental workload first — this lets the platform run the experimental pool cheaply on "spare" capacity without a permanent dedicated allocation.
- **Independent scaling per model**: each model pool gets its own HPA/KEDA config tuned to its own capacity envelope (AS8) — a 7B model's queue-depth threshold means something completely different from a 70B model's.
- **Cost attribution**: tag every request/metric with tenant ID at the gateway layer and roll cost-per-tenant dashboards off of GPU-second-per-tenant estimates — needed both for chargeback and for catching one tenant's runaway usage before it silently degrades everyone else's experience.
- **Model catalog management**: the experimental/weekly-changing model needs its own registry + canary pipeline (Ch. 09) independent of the stable models' release cadence — don't force all three model pools through one shared release process just because they share infrastructure.

**Defending against pushback:**
- *"Why not just give every tenant their own dedicated cluster?"* — Fully dedicated clusters maximize isolation but lose all the cost benefits of sharing GPU capacity across tenants with different, often complementary, peak-usage times (the 7B pool's off-peak hours can absorb the experimental pool's burst, etc.) — shared infrastructure with strong logical isolation (namespaces, priority classes, MIG/DRA) gets most of the isolation benefit at a fraction of the cost, which is why it's the more senior answer for anything short of a hard compliance requirement for physical separation.
- *"What happens when the experimental pool needs more GPUs than 'spare capacity' provides?"* — That's exactly what the priority/preemption design handles: its requests queue or degrade gracefully (Scenario 5) rather than the platform silently starving production traffic — and if the experimental workload becomes a stable, important use case, it graduates from "shares spare capacity" to "gets its own provisioned pool," a deliberate, visible decision rather than an accidental resource fight.

---

### Scenario 3: Design the rollout pipeline for model updates with zero user-visible downtime

**Prompt as given in an interview:** "Design how a new model version goes from 'trained' to 'serving 100% of production traffic' with zero downtime and minimal risk."

**Clarifying questions to ask first:**
- Is this a full model swap (new weights/architecture) or an incremental fine-tune of the current production model?
- What's the acceptable time-to-full-rollout — hours, or does it need to complete same-day?
- Is there an existing offline eval suite, or does one need to be built as part of this?
- Multi-turn conversations in play (affects sticky-routing requirements, CD7)?

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

**Key design decisions and trade-offs:**
- **Zero downtime comes from traffic-weight shifting, never a hard cutover**: at every stage both versions are fully "up" — the only thing changing is the routing weight, so there's never a moment where capacity actually drops to serve the swap (contrast with `RollingUpdate`'s pod-replacement churn, K2 — for a model-version swap, this scenario is describing canary-based rollout, not a Deployment rolling update).
- **Sticky-by-session routing** (CD7) is non-negotiable for multi-turn chat — otherwise "zero downtime" at the infra level still produces a broken user experience (inconsistent model behavior mid-conversation).
- **Immutable versioning + fast-rollback warm window** (MV2, MV4) is what turns "we can roll back" into "we can roll back in seconds," which is the actual zero-downtime guarantee that matters when something goes wrong *after* full promotion, not just during canary.
- **Automated analysis must include quality signals, not just infra health** (CD2) — a rollout pipeline that only checks latency/error rate will happily promote a model that's healthy but worse.

**Defending against pushback:**
- *"This seems like a lot of infrastructure for what could be a one-line config change."* — The infrastructure cost is paid once (build the pipeline); the alternative (manual, ad hoc rollouts) pays a recurring risk cost on every single model update indefinitely — and model updates for an actively-used LLM product are rarely a one-time event, they're a recurring operational reality, which is exactly when investing in the pipeline is justified.
- *"What if the offline eval suite doesn't catch something the shadow/canary stages would?"* — That's precisely why there are three independent validation stages (offline eval, shadow, canary) with different blind spots — offline eval is fast but uses historical data; shadow catches live-distribution issues with zero user exposure; canary catches anything remaining with bounded exposure and an automatic kill switch. No single stage is trusted alone.

---

### Scenario 4: Design monitoring and autoscaling for a bursty consumer product

**Prompt as given in an interview:** "A consumer-facing feature can go viral and spike traffic 10-20x within minutes, then fall back down. Design the monitoring and autoscaling approach."

**Clarifying questions to ask first:**
- What's the actual cost tolerance for over-provisioning a warm buffer vs. the cost/reputational risk of degrading during a spike?
- Is there any predictability to bursts (time-of-day patterns, known launch events) or are they fully unpredictable (organic virality)?
- What's the acceptable degradation mode if capacity is truly exceeded — slower responses, a smaller model, or hard rejection?

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

**Key design decisions and trade-offs:**
- **Warm buffer sized off the observed burst ratio, not the average** — this is a deliberate cost-vs-resilience trade-off communicated explicitly to stakeholders (e.g., "we run at 40% baseline utilization specifically to absorb a 10x spike within 60 seconds"); the alternative (thin margin, pure reactive scaling) guarantees degraded UX during exactly the highest-visibility traffic moments (viral moments are also PR-visible moments).
- **Bounded admission queue with fast, honest rejection**: past a queue-depth threshold, return fast 429s (with retry-after guidance) rather than accepting requests into an ever-growing queue that eventually times out anyway — a fast, clear rejection is a better user experience and a better system-health outcome than a slow, silent failure.
- **Pre-provisioned warm node pool**: keep a standing set of GPU nodes ready (even if pods aren't placed on them yet) so pod-level scale-up doesn't have to wait on node-level provisioning (K7) — this is often the single biggest lever for closing the gap between "detected the spike" and "capacity actually serving it."
- **Predictive scaling for known patterns** (AS5) layered on top of reactive scaling for the unpredictable residual — if there's a known launch event, pre-scale ahead of it rather than relying on reactive detection alone.
- **Graceful degradation as the last line of defense** (Scenario 5) — even with a warm buffer and fast autoscaling, plan for the case where the burst still exceeds capacity, and have an explicit, tested fallback behavior rather than an undefined failure mode.

**Defending against pushback:**
- *"Isn't a permanently warm buffer just wasted spend most of the time?"* — Yes, partially — it's an explicit insurance cost, sized against the business cost of a bad viral-moment experience (which, for a consumer product, can mean losing exactly the users a viral moment was supposed to bring in) — the right size is a business trade-off, not a purely technical one, and should be revisited periodically against actual observed burst frequency/severity.
- *"Why not just autoscale hard and fast instead of paying for a warm buffer?"* — Because GPU node provisioning and cold-start (K7) fundamentally cannot react in under a minute in most environments — no amount of HPA tuning changes that physical/operational floor, so a warm buffer is compensating for a real latency floor, not a config mistake.

---

### Scenario 5: Design a fallback/degradation strategy when GPU capacity runs low

**Prompt as given in an interview:** "Your serving fleet is at capacity — demand exceeds what your GPUs can serve within SLO, and more capacity isn't available immediately (quota limit, spot capacity dried up, etc.). Design the degradation strategy."

**Clarifying questions to ask first:**
- Is there a smaller/cheaper model available that can serve as a fallback, and is a quality drop acceptable for some fraction of traffic?
- Are all users/requests equal priority, or is there a tiering (paid vs. free, internal vs. external) that should inform who gets degraded first?
- Is a slower response acceptable, or does the product need a hard latency ceiling even at reduced quality?

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

**Key design decisions and trade-offs:**
- **Degrade in graduated steps, not binary "up or down"**: capping `max_tokens` or disabling one expensive feature preserves service for far more users than an all-or-nothing shutoff, and each step is chosen to shed load with the smallest quality/UX cost first.
- **Tiered shedding by priority**: if there's a paid tier, protect it longest; free/best-effort tiers absorb degradation first — this needs to be a pre-agreed business policy (documented, not improvised mid-incident), because "whose requests get shed first" is a product decision, not just an engineering one.
- **Fallback to a smaller model rather than pure rejection where possible**: a lower-quality-but-real answer is often a better user experience than an error, especially for use cases where "good enough" is acceptable (this needs product buy-in on which use cases tolerate a quality drop and which don't — not every use case should silently get a worse model).
- **Make degradation visible internally**: every degradation action should fire an alert/dashboard signal — this is capacity running out, which is an operational incident worth investigating and following up on (was it a genuine unexpected spike, or a capacity-planning miss?), not something that should silently self-heal without anyone noticing it happened.
- **Auto-recovery symmetric to degradation**: un-degrade in the reverse order as capacity signal recovers, with hysteresis (don't flap back and forth right at the threshold) — same stabilization-window thinking as HPA (AS7).

**Defending against pushback:**
- *"Isn't silently serving a worse model to some users dishonest to the product's users?"* — The alternative — everyone gets a slow/failing response during overload — is worse for everyone, including the users who'd have been fine with a smaller model's answer; the right answer is making the policy explicit and product-approved (e.g., surfacing "a faster, lighter response" in the UI when degraded) rather than hiding it, not avoiding degradation entirely.
- *"Why build a whole degradation system instead of just... having enough capacity?"* — Because "enough capacity for the absolute worst case, at all times" is either prohibitively expensive or physically unavailable on demand (quota limits, regional GPU shortages are real and recurring as of 2025-2026) — a degradation strategy is the practical answer to a real, recurring constraint, not a workaround for bad capacity planning; even well-capacity-planned systems hit true exogenous shortages sometimes.

---

## 2025-2026 Landscape Quiz

*A senior interviewer will often probe whether you track the ecosystem or are reciting a 2023 blog post. The facts below were checked against current sources as of August 2026. Exact version numbers and benchmark ratios move fast — anything marked "verify before quoting" should be re-checked against current docs before you cite a specific number in an interview; the durable value here is the *shape* of the landscape, not the last digit of a version string.*

### LQ1: What are the main LLM inference engines in production use, and how do they position relative to each other?

**Answer:**
- **vLLM**: the dominant open-source general-purpose engine — continuous batching, PagedAttention, broad model-architecture coverage, OpenAI-compatible server, active "V1" engine architecture (rewritten scheduler/execution core, alpha from early 2025, matured through 2025-2026). Default choice for iterating quickly across many model families.
- **SGLang**: a fast-moving alternative engine, notable for its own scheduler/runtime design (RadixAttention for prefix caching) and strong benchmark results on structured-generation and high-concurrency workloads; increasingly cited alongside vLLM in 2025-2026 comparisons rather than as a niche alternative.
- **TensorRT-LLM (NVIDIA)**: compiled, hardware-specific optimized kernels, typically the throughput/latency ceiling on NVIDIA GPUs at the cost of a build/compile step and less architecture flexibility; increasingly consumed *via* Triton's TensorRT-LLM backend or packaged inside NVIDIA NIM rather than used standalone.
- **Triton Inference Server**: the multi-framework/multi-backend serving layer (Ch. 11) — can host vLLM, TensorRT-LLM, or other backends behind one unified serving/ops surface, rather than being a competing "engine" per se.
- **NVIDIA NIM**: prebuilt, containerized "inference microservices" that package one of the above engines (TensorRT-LLM, vLLM, or SGLang, depending on the model) behind a standard OpenAI-compatible API — positioned as the fastest path to a supported, optimized deployment without hand-tuning engine internals yourself. *(Verify current NIM backend choices per model before quoting — this has changed over time.)*
- Exact version numbers and head-to-head benchmark ratios between these engines change roughly monthly — verify before quoting a specific number in an interview; know the *names and positioning*, cite specifics as "as of my last check."

---

### LQ2: What's new about NVIDIA's current GPU generations relevant to inference (as of 2025-2026)?

**Answer:**
- **Hopper (H100/H200)**: widely deployed generation through 2024-2025; H200 added significantly more HBM capacity/bandwidth over H100, directly helping KV-cache-bound serving.
- **Blackwell (B100/B200, GB200 NVL72)**: current-generation flagship, with major FP4/FP8 tensor-core throughput improvements and NVLink domain scale (GB200 NVL72 links many GPUs into one large NVLink domain) aimed squarely at large-model training *and* inference at scale.
- **Blackwell Ultra (B300/GB300)**: a mid-cycle refresh with substantially more HBM3e memory per GPU (reported around 288GB class) — directly relevant to inference because more per-GPU memory means fewer GPUs needed per replica for a given model+KV-cache footprint, improving TP-degree economics. **Verify exact specs/availability before quoting** — this generation was rolling out through late 2025 into 2026 and cloud availability (e.g., new EC2 instance families) was still expanding.
- **Practical inference implication of the whole trend**: each generation's biggest inference-relevant win isn't just raw FLOPs, it's memory capacity/bandwidth and native low-precision (FP8/FP4) tensor-core throughput — both directly attack the two real bottlenecks (KV-cache memory, decode memory-bandwidth) discussed throughout this guide, more than raw compute does.

---

### LQ3: What's Dynamic Resource Allocation (DRA) in Kubernetes, and what's its GA status?

**Answer:**
- DRA is the modern Kubernetes API for expressing complex device (especially GPU) allocation requirements — `DeviceClass` (what kinds of devices exist), `ResourceClaim`/`ResourceClaimTemplate` (what a pod needs), and CEL-based selection (e.g., "a GPU with >40GB memory") — replacing the older device-plugin/extended-resource model's all-or-nothing integer-count limitation.
- It supports use cases the old model handled poorly: ranked fallback preferences across GPU types, partitioned/shared access (feeding into MIG or time-slicing configuration), and richer selection logic than a bare `nvidia.com/gpu: N` request.
- **GA status**: DRA graduated to General Availability in Kubernetes v1.34 (per the official Kubernetes project blog). Continued refinement of structured-parameter features and ecosystem tooling (device plugins, the NVIDIA GPU Operator's DRA integration, schedulers like the KAI Scheduler) continued through the following releases. **Verify the exact current minor version and feature maturity before quoting specifics** — this is one of the fastest-moving parts of the Kubernetes GPU story right now.
- Interview framing: know that the *old* model (integer extended resources via device plugins, still very common in production today) and the *new* model (DRA) coexist, and that DRA is what unlocks fine-grained/flexible GPU sharing without hand-rolled workarounds — don't imply the old model is gone; most existing clusters still use it.

---

### LQ4: What are the current options for sharing one physical GPU across workloads on Kubernetes, ranked by isolation strength?

**Answer:**
(See K3 above for the full comparison table.) In brief, from weakest to strongest isolation: **time-slicing** (software round-robin, no memory isolation) → **MPS** (shared address space, better concurrent-kernel execution, still no hard memory isolation) → **MIG** (hardware-partitioned compute+memory slices, fixed profiles) → **DRA-orchestrated allocation** (flexible expression of any of the above, or whole-device claims, through a unified API). The 2025-2026 trend is consolidating around DRA as the *scheduling* layer on top of MIG/time-slicing/MPS as the underlying *mechanism* — DRA doesn't replace MIG, it makes MIG (and other sharing mechanisms) easier to request and compose with complex placement logic.

---

### LQ5: What's the current state of KV-cache offloading / cross-node cache sharing as a serving technique?

**Answer:**
- Tools like **LMCache** implement a KV-cache layer that can offload cache to CPU memory or even NVMe/remote storage and share cache across multiple vLLM instances/nodes — extending prefix caching (V6) beyond a single replica's GPU memory.
- Motivation: prefix caching's benefit is capped by how much of the cache fits in one replica's GPU memory and how well the load balancer routes cache-sharing requests to the same replica; offloading/sharing lets a much larger effective cache (system prompts, long-lived RAG contexts, long conversation histories) be reused across a whole fleet rather than being replica-local and volatile.
- This is an active, fast-evolving area as of 2025-2026 (there's active work on standardizing KV-cache transfer/connector interfaces between prefill/decode-disaggregated setups, V8) — good to know it exists and why it matters, but treat specific throughput multipliers as workload-dependent claims to verify, not universal constants.

---

### LQ6: What serving-platform / "inference-as-a-service" options exist besides self-hosting, and when would a senior engineer recommend one over self-hosting?

**Answer:**
- **NVIDIA NIM**: prebuilt optimized containers per model, self-hosted on your own GPUs but with the engine-tuning work done for you.
- **Managed inference endpoints** from major clouds and model providers (dedicated-capacity or serverless GPU endpoints): trade control and per-token cost optimization for operational simplicity — no cluster to run, no autoscaling/monitoring stack to build.
- **Self-hosted on Kubernetes with vLLM/SGLang/TensorRT-LLM/Triton**: full control (this whole guide), best cost-at-scale, but you own everything described in every chapter above.
- **Senior recommendation framing**: self-host when scale is large enough that infra cost/control genuinely outweighs the engineering investment (rule of thumb some teams use: once GPU spend is large enough that a percentage-point of utilization improvement is a meaningful dollar figure, self-hosting/optimization work pays for itself); use a managed platform when you're pre-scale, need to move fast, or don't have (and don't want to build) platform engineering capacity — this is a genuine build-vs-buy trade-off, not a "real engineers self-host" purity contest.

---

### LQ7: Rapid-fire current-fact check — answer, then flag confidence.

**Answer (with confidence flags):**
- GQA (grouped-query attention) is standard across most current major open-weight model families — **high confidence, stable fact**.
- FP8 is a standard, well-supported inference precision on current NVIDIA data-center GPU generations — **high confidence, stable fact**.
- Exact current vLLM/SGLang/TensorRT-LLM version numbers and head-to-head benchmark numbers — **verify before quoting, changes ~monthly**.
- Kubernetes DRA reached GA in v1.34 — **high confidence as of research date, but verify the current Kubernetes version your target company runs, since GA status of a *feature* doesn't mean every cluster has upgraded to use it**.
- Specific GPU model availability/pricing (e.g., which cloud offers which Blackwell Ultra instance family) — **verify before quoting, changes monthly with new instance-family launches**.
- The general trend toward disaggregated prefill/decode serving and KV-cache-sharing infrastructure at the largest scale — **high confidence as a direction**, but treat any specific implementation's maturity/production-readiness claim as something to verify.

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
**Recovery:** "First I'd check GPU/KV-cache utilization and batching configuration — low utilization under load means a scheduling/batching problem, not a hardware problem. I'd only add GPUs after confirming the engine is already using the ones it has efficiently."

### Trap 2: "vLLM is always faster than everything else, so just use vLLM."

**Why it's wrong:** treats engine choice as a fixed ranking instead of a workload-dependent trade-off (TensorRT-LLM often wins on raw throughput for stable, high-volume models; Triton wins for multi-framework needs; SGLang is a real, current alternative).
**Recovery:** "vLLM is my default for iteration speed and broad model support, but for a small number of stable, extremely high-volume models I'd benchmark TensorRT-LLM, and if we're serving heterogeneous frameworks I'd put Triton in front regardless of engine choice."

### Trap 3: "We'll just use HPA on CPU utilization like any other service."

**Why it's wrong:** CPU utilization is decoupled from GPU/KV-cache load for LLM serving pods (AS1) — this is one of the fastest ways to reveal you haven't actually run GPU workloads in Kubernetes.
**Recovery:** "For GPU-bound serving I'd scale on a custom metric — queue depth or GPU/KV-cache utilization via the Prometheus Adapter or KEDA — CPU utilization on these pods is nearly meaningless."

### Trap 4: "Scale to zero when idle to save cost."

**Why it's wrong:** ignores GPU cold-start reality (minutes, not seconds) — the first request after scale-to-zero will badly violate almost any latency SLO.
**Recovery:** "For latency-sensitive paths I'd keep a warm minimum and rely on predictive/scheduled scaling for known low-traffic windows instead — true scale-to-zero only for genuinely latency-insensitive batch workloads."

### Trap 5: "Canary just means routing 10% of traffic to the new version and watching error rate."

**Why it's wrong:** misses that LLM canaries need output-quality signals (CD2) — a model can be infra-healthy and behaviorally regressed simultaneously, which pure error-rate/latency monitoring won't catch.
**Recovery:** "Infra metrics are necessary but not sufficient — I'd add an automated quality-eval sampling step on canary outputs (guardrail-trigger rate, structured-output validity, an LLM-as-judge score) before promoting."

### Trap 6: "Just quantize to INT4 everywhere for cost savings — precision doesn't really matter for chat."

**Why it's wrong:** asserts a blanket accuracy claim without task-specific validation; INT4 accuracy risk varies a lot by task (reasoning-heavy tasks degrade more than casual chat), and "doesn't matter" is exactly the unvalidated assumption that gets someone burned in production.
**Recovery:** "I'd start from FP8 as the likely sweet spot on current hardware, and only push to INT4 after validating on our actual eval suite for our actual task mix — not assume it's fine."

### Trap 7: "Model drift means we should just retrain automatically whenever drift is detected."

**Why it's wrong:** ignores feedback-loop risk (DR3) — automatic retraining on drifted/production data without human review can amplify the very drift it's reacting to.
**Recovery:** "I'd have drift detection trigger an alert and investigation, with retraining as a human-gated decision informed by that investigation — not a fully automatic pipeline straight to production."

### Trap 8: "Kubernetes handles GPU sharing out of the box, just request `nvidia.com/gpu: 0.5`."

**Why it's wrong:** factually incorrect — GPUs are an integer-only extended resource without MIG/time-slicing/MPS/DRA configured; a fractional request simply won't work (K1, K3).
**Recovery:** "GPU requests are whole-unit by default — to share one GPU across pods I'd configure MIG for hard isolation, or time-slicing/MPS for looser sharing, potentially orchestrated via DRA."

### Trap 9: "We don't need a model registry, we'll just tag Docker images with the model version."

**Why it's wrong:** conflates code-release cadence with model-release cadence, and loses model-specific metadata (eval scores, lineage, approval status) a registry provides natively (MV7).
**Recovery:** "For infrequent updates on one model, tagged images might be enough — but once we have frequent fine-tunes or multiple models needing independent rollback, I'd want an actual registry decoupled from the application image."

### Trap 10: "PSI/KL-divergence tests are all we need for drift detection on an LLM."

**Why it's wrong:** these classical statistical tests need a numeric feature to compare and say nothing about output *quality* directly (DR7) — a real blind spot for generative systems.
**Recovery:** "Statistical tests on derived features (embeddings, length) are useful for continuous monitoring, but I'd pair them with output-quality-eval sampling — that's the ground-truth signal the statistical tests are only ever a proxy for."

### Trap 11: "Just cache all the responses to save GPU cost."

**Why it's wrong:** treats response caching as a general LLM cost solution when it only helps for repeated/identical content — most production traffic (unique user prompts) won't hit a response cache at all; the real high-leverage cache is prefix/KV-cache reuse of shared context (V6), not full-response caching.
**Recovery:** "Full-response caching only helps for genuinely repeated queries. The bigger lever is prefix caching — reusing KV cache for shared system prompts or RAG context — which helps a much larger share of realistic traffic."

### Trap 12: "Zero downtime means using Kubernetes' rolling update strategy for model version swaps."

**Why it's wrong:** conflates a Deployment-level rolling update (for code/image changes) with a canary/traffic-weighted rollout (for model version changes, Scenario 3) — a rolling update replaces pods, it doesn't give you a controlled, metric-gated traffic ramp or an instant rollback via traffic-weight change.
**Recovery:** "For a model version change specifically, I'd use a canary pipeline with traffic-weight shifting rather than a pod-replacement rolling update — that gives instant rollback and gradual, metric-gated exposure, which a rolling update doesn't."

### Trap 13: "More replicas is always better for availability."

**Why it's wrong:** ignores that each GPU replica is expensive and that availability comes from the *right* redundancy (spread across zones/nodes, N+1 for failure tolerance) not raw count — over-provisioning replicas "for availability" without a specific failure scenario in mind is just wasted spend.
**Recovery:** "I'd size replica count from load-test-derived capacity plus explicit failure-tolerance headroom (N+1, spread across zones) — not an arbitrary 'more is safer' buffer."

### Trap 14: "Streaming and batching are in tension, so pick one."

**Why it's wrong:** conflates continuous batching (an engine-internal scheduling technique) with streaming (a client-facing response-delivery mechanism) — they're not in tension; every major serving engine streams tokens from a continuously-batched decode loop simultaneously for many requests.
**Recovery:** "They operate at different layers — continuous batching is how the engine schedules GPU work across concurrent requests; streaming is how each request's tokens are delivered to its client as they're produced. Production systems do both together."

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
