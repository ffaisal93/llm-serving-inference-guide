# Capstone: Engineering a Production LLM Serving Platform

## Why This Chapter Exists

Chapters 01 through 11 each teach one piece of the puzzle in depth: how to stand up a basic server (01_basic_serving), how to containerize it (02_docker), how to run it on Kubernetes (03_kubernetes), how to load test it (04_load_testing), how to make it fast with vLLM (05_vllm_serving), how to autoscale it (06_autoscaling), how to roll out a new version safely (07_canary_deployments), how to watch it (08_monitoring), how to version and register models (09_model_versioning), how to detect drift in what it's serving (10_drift_detection), and how to run a multi-framework, multi-model server with Triton (11_triton).

None of those chapters, on their own, tells you how to actually **stand up an LLM inference platform from zero and keep it alive for a year**. That is what this chapter does. It is a capstone in the literal sense: it takes the load-bearing pieces from every other chapter and shows how they bolt together into one system, in the order you'd actually build them, with the seams between them called out explicitly — because production incidents live in the seams, not inside any one component.

Concretely, this chapter will:

- Give you a **reference architecture** you can hold in your head — one diagram, one system, every box labeled with the chapter that teaches it.
- Give you a **decision framework** for picking your serving engine, your orchestration layer, and your topology, with a real 2025-2026 options table instead of vague advice.
- Walk through **one complete build** end to end — a specific model, a specific SLO, a specific budget — with runnable Dockerfiles, Kubernetes YAML, a vLLM serve command, a load-test script, a PromQL alert, and an Argo Rollouts canary, so you see the pieces in the order they actually get built, not as an appendix of unrelated snippets.
- Give you a **cost model** with real 2026 GPU pricing so you can defend a capacity plan in a budget review.
- Catalog the **failure modes that only exist at the system level** — the ones that pass every unit test and every single-chapter checklist and still page you at 3 a.m., because they live in the interaction between two components each chapter treats in isolation.
- Give you a **pre-launch checklist** that spans all eleven chapters, and an **interview section** built around "tell me how you'd build this," including one full system-design walkthrough.

If you've read chapters 01-11, this chapter should feel like the moment the individual lessons in a driving course turn into actually driving on the highway. If you're skimming straight to this chapter, you'll get the shape of the whole system, but you should expect to jump back to the numbered chapters for the mechanism behind each box — this chapter deliberately does not re-derive PagedAttention, the Kubernetes scheduler, or EWMA drift statistics; it tells you where those live and how they fit.

A note on scope: "platform" here means the inference-serving path — from a client request to a generated response, at production scale, with a rollout and observability story around it. It does not cover training, RLHF, or data pipelines; those are different systems with different failure modes.

## 1. The Reference Architecture

Every production LLM platform — whether it's a two-person startup self-hosting one open-weight model or a hyperscaler running dozens — reduces to the same skeleton. What differs is scale, managed-vs-DIY choices, and how much of each box you build yourself. Here is the whole system in one picture.

```
                                   ┌─────────────────────────────────────────────────────┐
                                   │                    CLIENTS                           │
                                   │   (web app, mobile app, internal service, agent)     │
                                   └───────────────────────────┬───────────────────────────┘
                                                                │  HTTPS / gRPC
                                                                ▼
                        ┌───────────────────────────────────────────────────────────────────┐
                        │  API GATEWAY / LOAD BALANCER          [Ch. 01 Basic Serving,       │
                        │  - authn/authz, rate limiting          Ch. 03 Kubernetes Ingress]  │
                        │  - request routing (model, region)                                │
                        │  - request/response logging -----------------------------┐        │
                        └───────────────────────────┬───────────────────────────────┼────────┘
                                                      │                              │
                             ┌────────────────────────┼──────────────────────────┐   │
                             │                         ▼                          │   │
                             │        ┌───────────────────────────────┐          │   │
                             │        │   ROLLOUT / CANARY CONTROLLER  │          │   │
                             │        │   (Argo Rollouts / KServe)     │◄─────────┼───┼──── promote / abort
                             │        │   splits traffic stable:canary │          │   │      decision based on
                             │        │              [Ch. 07 Canary]   │          │   │      live metrics below
                             │        └────────────┬───────┬──────────┘          │   │
                             │                       │       │                    │   │
                             │              stable % │       │ canary %          │   │
                             │                       ▼       ▼                    │   │
                             │   ┌───────────────────────┐ ┌───────────────────┐  │   │
                             │   │  MODEL SERVER PODS     │ │ MODEL SERVER PODS  │  │   │
                             │   │  (stable version)      │ │ (canary version)   │  │   │
                             │   │  vLLM or Triton engine  │ │ vLLM or Triton     │  │   │
                             │   │  on GPU nodes           │ │ on GPU nodes       │  │   │
                             │   │  [Ch. 05 vLLM,          │ │ [Ch. 05, Ch. 11]   │  │   │
                             │   │   Ch. 11 Triton]        │ │                    │  │   │
                             │   └───────────┬───────────┘ └─────────┬──────────┘  │   │
                             │               │  metrics + logs        │            │   │
                             │  KUBERNETES    │                        │            │   │
                             │  CLUSTER       ▼                        ▼            │   │
                             │  [Ch. 03]  ┌────────────────────────────────────┐    │   │
                             │            │  HPA / KEDA AUTOSCALER              │    │   │
                             │            │  scales pod count from queue depth, │    │   │
                             │            │  GPU utilization, req/s             │    │   │
                             │            │  [Ch. 06 Autoscaling]               │    │   │
                             │            └────────────────────────────────────┘    │   │
                             └──────────────────────────────────────────────────────┘   │
                                                                                          │
                        ┌───────────────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────────────────────────────────────────┐
        │                        OBSERVABILITY STACK                                  │
        │   Prometheus (metrics) + Grafana (dashboards) + Alertmanager (paging)       │
        │   + structured request logs + distributed tracing                          │
        │                                          [Ch. 08 Monitoring]                │
        └───────────────────────┬───────────────────────────────┬───────────────────┘
                                  │                               │
                                  ▼                               ▼
        ┌───────────────────────────────────┐     ┌───────────────────────────────────┐
        │   MODEL / PROMPT REGISTRY           │     │   DRIFT & QUALITY MONITOR          │
        │   - semantic version per model      │     │   - input distribution shift       │
        │   - which weights + prompt template │     │   - output quality / eval scores   │
        │     + engine config == one release  │     │   - feeds "should we roll back?"   │
        │   [Ch. 09 Model Versioning]          │     │   [Ch. 10 Drift Detection]         │
        └───────────────────┬─────────────────┘     └────────────────┬───────────────────┘
                              │                                        │
                              └───────────────┬────────────────────────┘
                                               ▼
                                ┌───────────────────────────────────┐
                                │   OFFLINE EVAL / CI PIPELINE        │
                                │   golden-set regression tests,      │
                                │   benchmark scores, load tests       │
                                │   gate the next release             │
                                │   [Ch. 04 Load Testing, Ch. 10]      │
                                └───────────────────────────────────┘
```

### Reading the diagram box by box

**Clients → gateway.** Nothing here is LLM-specific yet — it's the same authn/rate-limiting/routing layer any API needs. Chapter 01 (Basic Serving) builds the naive version of this (a single FastAPI process answering requests directly); in a real platform the gateway is a separate tier (an ingress controller, Envoy, or a managed API gateway) that never touches a GPU itself.

**Rollout/canary controller.** This is the traffic-shaping brain. It knows there are two (or more) live versions of the model server and decides, second by second, what percentage of traffic each one gets. Chapter 07 (Canary Deployments) is entirely about this box: traffic splitting strategies, promotion criteria, and rollback triggers. In the reference architecture this is implemented with a real controller — Argo Rollouts if you're doing this yourself on vanilla Kubernetes, or KServe's `InferenceService` canary spec if you've adopted KServe as your model-serving CRD layer.

**Model server pods.** The GPU-bound heart of the system. Chapter 05 (vLLM Serving) and Chapter 11 (Triton) each teach one engine for this box in depth — vLLM if you're serving one or a few open-weight models as fast as possible with continuous batching and PagedAttention, Triton if you need one server fronting many models/frameworks (PyTorch, TensorRT-LLM, ONNX, custom Python backends) behind one protocol. Both plug into the same box in this diagram; the rest of the platform (gateway, autoscaler, monitoring) barely cares which one is inside.

**Kubernetes cluster + autoscaler.** Chapter 03 (Kubernetes) is the substrate everything above runs on: pod specs, GPU scheduling, node pools, health probes. Chapter 06 (Autoscaling) is the control loop layered on top of it — deciding how many model-server pods should exist right now based on load signals. The reference architecture uses KEDA rather than plain HPA for GPU workloads specifically because HPA's default CPU/memory metrics are close to useless for judging GPU-bound LLM load; KEDA lets you scale on a Prometheus query (queue depth, `vllm:num_requests_waiting`) instead. See the vLLM `production-stack` project's KEDA guide for a reference implementation (`docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html`).

**Observability stack.** Chapter 08 (Monitoring) builds this: Prometheus scraping engine metrics, Grafana dashboards, Alertmanager routing pages. In the reference architecture this box is the nervous system — every other box either emits into it (metrics, logs, traces) or reads from it (the canary controller reads success/error/latency metrics to decide whether to promote; the autoscaler reads queue-depth metrics to decide whether to scale).

**Model/prompt registry.** Chapter 09 (Model Versioning) owns this box. The critical, easy-to-miss detail (revisited in Section 6): a "version" in a serious platform is not just a checkpoint — it is the tuple of (model weights, tokenizer, prompt/chat template, sampling defaults, engine config). Registering only the weights and rolling the rest out-of-band is one of the most common system-level failure modes.

**Drift & quality monitor.** Chapter 10 (Drift Detection) owns this box. It watches the live traffic distribution and the model's output quality over time and answers "is this still the same model behaving the same way it did at launch, on the same kind of traffic it was validated on?" Its output feeds two places: back into the canary controller (a canary that's drifting on quality should not auto-promote) and into the eval pipeline that gates the next release.

**Offline eval/CI.** Chapter 04 (Load Testing) determines the throughput/latency ceiling before anything ships; Chapter 10's regression suite determines whether quality has regressed. Both run in CI, before a new model version is even allowed to become a canary.

The rest of this chapter is about the connective tissue: how to choose what goes in each box (Section 3), how to build the whole thing once for a concrete scenario (Section 4), what it costs (Section 5), how it breaks in ways no single chapter predicts (Section 6), and how to know you're ready to launch it (Section 7).

## 2. Choosing Your Stack

There is no single right stack. There is a right stack **for your traffic, your latency SLO, your team size, and your budget**. This section gives you the decision tree, then a table of real options as of 2026.

### 2.1 Engine: raw Transformers vs vLLM vs Triton vs a managed API

```
Do you control the model weights (open-weight / fine-tuned),
or are you calling someone else's hosted model (OpenAI, Anthropic, etc.)?
        │
        ├── Hosted/managed API only ──► You don't need this chapter's serving stack at all.
        │                               You still need Ch. 07 (canary across model versions/
        │                               providers), Ch. 08 (monitoring), Ch. 10 (drift) —
        │                               the "server" box just becomes an HTTP call to a vendor.
        │
        └── Self-hosting open-weight or fine-tuned weights
                │
                ├── Prototype / <5 req/s / latency doesn't matter yet
                │        └──► Raw Transformers + `generate()` behind FastAPI (Ch. 01).
                │             No batching, no PagedAttention. Fine for a demo, wrong for
                │             anything a real user waits on.
                │
                ├── One or a few models, need max throughput/cost-efficiency,
                │   comfortable operating Python services
                │        └──► vLLM (Ch. 05). This is the default answer in 2026 for
                │             self-hosted LLM serving: continuous batching + PagedAttention
                │             gets you 10-20x the throughput of naive HF `generate()` at
                │             comparable latency. OpenAI-compatible server built in.
                │
                ├── Many models / many frameworks (PyTorch, ONNX, TensorRT-LLM, custom
                │   Python) behind one server, need ensembles or multi-model routing,
                │   or an existing NVIDIA-centric MLOps org
                │        └──► Triton Inference Server (Ch. 11), typically with the
                │             TensorRT-LLM backend for max single-model performance or
                │             the vLLM backend if you want vLLM's scheduler under Triton's
                │             multi-model management plane.
                │
                └── Extreme scale, prefill and decode have very different resource
                    profiles, need to pool KV cache across many replicas
                        └──► Disaggregated serving (split prefill/decode pools, e.g. the
                             `llm-d` project on Kubernetes, or NVIDIA Dynamo). This is a
                             2025-2026-era pattern for the largest deployments; most teams
                             should not start here — see 2.3.
```

**The honest heuristic:** if you're asking "vLLM or Triton," you've usually already answered it — vLLM if it's your own model(s) and you want the simplest path to production-grade throughput; Triton if multi-framework/multi-model flexibility or an existing NVIDIA Triton investment is the actual requirement. They are not mutually exclusive: Triton can run vLLM as a backend, giving you Triton's multi-model management with vLLM's scheduler underneath.

### 2.2 Orchestration: Kubernetes vs simpler

```
Do you need to run this on more than one machine, or promise any uptime SLA?
        │
        ├── No — single GPU box, internal tool, can tolerate a restart
        │        └──► Docker Compose (Ch. 02) or a single systemd-managed container.
        │             Don't build Kubernetes for one box; you'll spend more time
        │             operating the control plane than the workload.
        │
        └── Yes — multiple replicas, need autoscaling, rolling/canary updates,
            multi-tenant GPU sharing, or you're already a Kubernetes shop
                 └──► Kubernetes (Ch. 03) + the GPU device plugin/GPU Operator +
                      KEDA/HPA (Ch. 06) + Argo Rollouts or KServe (Ch. 07).
                      This is the default for anything with a production SLO and
                      more than a handful of GPUs.
```

Managed alternatives exist between these two extremes — a managed inference endpoint (SageMaker, Vertex AI endpoints, Modal, Replicate/Baseten-style GPU-as-a-service) gives you most of the Kubernetes-cluster benefits without operating the control plane yourself, at a per-GPU-hour premium. That is often the right call for a small team; you are trading operational burden for margin, and the decision framework in 2.3 makes that trade explicit.

### 2.3 Single-region vs multi-region

Start single-region unless you have a specific reason not to. Multi-region LLM serving adds real complexity that only pays for itself at real scale:

| Trigger | Single-region is fine | Consider multi-region |
|---|---|---|
| Latency to users | Users clustered in one geography | Global user base, TTFT SLO tight enough that cross-ocean RTT matters |
| Availability requirement | "Best effort," a few hours of downtime tolerable | Contractual SLA (99.9%+) that a single cloud region outage would breach |
| GPU capacity | One region has enough on-demand/reserved capacity | Capacity-constrained GPUs (H100s) force spreading across regions/providers to get enough quota |
| Data residency | No regulatory constraint | GDPR/data-residency rules require EU traffic served from EU |
| Team size | Small team; one region is already a lot of surface area | Dedicated platform team that can own cross-region model registry sync, routing, and failover |

Multi-region done wrong (e.g., model registry not replicated, so a canary promotes correctly in one region and never reaches another) is one of the failure modes in Section 6. If you do go multi-region, the model/prompt registry (Ch. 09) and the drift baseline (Ch. 10) both need to be global sources of truth, not per-region copies that can silently diverge.

### 2.4 Real options, 2025-2026

| Layer | Lightweight / early-stage option | Production-scale option | Notes |
|---|---|---|---|
| Serving engine | Raw Transformers `generate()` (Ch. 01) | vLLM (Ch. 05) | vLLM latest stable line is the v0.20.x series (e.g. v0.20.2, May 2025) at the time of writing, with `gpt-oss`, DeepSeek-V4, and Qwen3-VL support landing in that line; check `github.com/vllm-project/vllm/releases` for current. |
| Multi-model / multi-framework serving | N/A | Triton Inference Server (Ch. 11), with the vLLM backend or the TensorRT-LLM backend | TensorRT-LLM backend gives the best single-model latency on NVIDIA GPUs at the cost of an offline compile/engine-build step; the vLLM backend trades a little raw throughput for vLLM's simpler operational model and faster iteration. |
| Orchestration | Docker Compose (Ch. 02) | Kubernetes (Ch. 03) + NVIDIA GPU Operator for device plugin, MIG partitioning, and time-slicing | GPU Operator handles driver install, device plugin, DCGM exporter, and MIG/time-slicing config as one Helm-installed unit — see `docs.nvidia.com/datacenter/cloud-native/gpu-operator`. |
| Autoscaling | Manual replica count | HPA on custom metrics, or KEDA scaling on a Prometheus query (Ch. 06) | KEDA is the practical default for GPU/queue-depth-based scaling; see the vLLM `production-stack` project's KEDA guide. |
| Rollout / canary | Manual `kubectl apply`, watch and pray | Argo Rollouts (canary + analysis templates) or KServe `InferenceService` canary (`canaryTrafficPercent`) (Ch. 07) | KServe's canary model is declarative and Kubernetes-native if you've already adopted KServe as your model CRD layer; Argo Rollouts is the general-purpose choice if you haven't. |
| Multi-replica model serving with shared state | N/A | LeaderWorkerSet (LWS) for multi-node tensor/pipeline-parallel vLLM deployments | LWS is a Kubernetes API (via the `vllm-project/production-stack` reference and `docs.vllm.ai/en/stable/deployment/frameworks/lws`) for treating a group of pods as one logical multi-node model replica. |
| Disaggregated prefill/decode | N/A | `llm-d` (Kubernetes-native, KV-cache-aware routing, joint Red Hat/Google/IBM/CoreWeave project) or NVIDIA Dynamo | Only worth adopting once you've outgrown a monolithic replica-per-request-pool model — see `llm-d.ai/docs/architecture/advanced/disaggregation`. |
| Monitoring | `print()` statements | Prometheus + Grafana + Alertmanager (Ch. 08), scraping vLLM's built-in `/metrics` endpoint | vLLM exposes histograms for TTFT, inter-token latency, queue time, and end-to-end latency natively — no custom instrumentation needed for the basics. |
| Model registry | A folder of checkpoints and a spreadsheet | MLflow Model Registry, or a Git-based registry (Ch. 09) | Whatever you pick, it must version the prompt template alongside the weights — see Section 6. |
| Managed alternative to all of the above | — | SageMaker/Vertex AI endpoints, Modal, Baseten, Replicate | Right choice when the team is too small to operate Kubernetes + GPU Operator + Argo Rollouts themselves; you pay a per-GPU-hour premium for someone else operating boxes 2-6 of the reference architecture. |

## 3. Build It End-to-End — Full Worked Walkthrough

### 3.1 The scenario

You are the first infra engineer at a startup. Product wants to serve **openai/gpt-oss-20b** — OpenAI's open-weight 21B-parameter mixture-of-experts model (3.6B active parameters per token, 128K context via YaRN scaling from a 4K base) — to end users through a chat product. The requirements:

- **Target load:** 500 requests/second sustained, with bursts to 700 req/s.
- **SLO:** P95 time-to-first-token (TTFT) under 2 seconds.
- **Budget:** as few GPU-hours as possible without missing the SLO. No H100/H200/B200 unless the numbers force it — the model's native MXFP4 quantization (applied to the MoE expert weights, with attention/router/embeddings kept in BF16) means the checkpoint itself only needs about 16 GB of VRAM, so start by asking whether a cheaper card can do the job before reaching for the most expensive one.
- **Team:** two infra engineers, no dedicated MLOps platform team yet.

We'll walk this scenario through choosing the engine, containerizing, deploying, load testing, autoscaling, monitoring, and canarying the next model update — in that order, because that's the order you actually do it in.

### 3.2 Choosing the engine and quantization

Following the decision tree in Section 2.1: this is a single open-weight model, we want maximum throughput per GPU-dollar, and the team is small — vLLM (Ch. 05) is the answer, not Triton. We don't have a multi-framework requirement that would justify Triton's extra operational surface.

Quantization is already decided for us in the useful sense: gpt-oss-20b ships with **native MXFP4** on its MoE expert weights (the vast majority of its parameters), so there's no separate AWQ/GPTQ quantization step to run — you load the model as published and vLLM handles the rest. The GPU choice becomes the real lever:

| GPU | VRAM | Fits gpt-oss-20b (~16 GB weights + KV cache)? | Relative on-demand cost (2026, wide provider range) |
|---|---|---|---|
| A100 80GB | 80 GB | Yes, comfortably — room for large KV cache and big batches | ~\(\$1.99\)/hr baseline, varies by provider |
| L40S 48GB | 48 GB | Yes — less KV-cache headroom than the A100 at very large batch sizes, but adequate for this SLO | Typically priced below A100 on most clouds |
| H100 80GB | 80 GB | Yes, with the most headroom and the fastest per-token decode | roughly \(\$1.49\)-\(\$6.98\)/hr across 15+ providers; \(\$2\)-\(\$3.29\)/hr is a common on-demand baseline |

For a 2-second TTFT SLO at 500-700 req/s, the honest move is to **start on A100 80GB** (cheaper than H100, and the model doesn't need H100's extra compute headroom at this scale), measure the real ceiling with a load test, and only move to H100 if the load test shows you can't hit the SLO at a GPU count your budget tolerates. This is the "choose cheap, then measure, then upgrade only if the numbers force it" pattern — the opposite of defaulting to the newest GPU because it's fastest.

### 3.3 Containerizing it

vLLM ships an official `vllm/vllm-openai` image, so the Dockerfile's job is thin: pin a version, bake in any org-specific config, and set the launch command. This builds on the general containerization practice from Ch. 02 (Docker) — multi-stage builds, non-root users, minimal layers — applied to a GPU-serving image.

```dockerfile
# Dockerfile
FROM vllm/vllm-openai:v0.20.2

# Org-standard: run as non-root, drop unnecessary capabilities (Ch. 02 hardening practices)
RUN useradd --create-home --uid 10001 vllmuser
USER vllmuser
WORKDIR /home/vllmuser

# Bake in the model name so this image is a one-model, one-version artifact —
# the image tag itself becomes part of the model version identity (Ch. 09).
ENV MODEL_NAME="openai/gpt-oss-20b"
ENV VLLM_KV_CACHE_DTYPE="auto"

EXPOSE 8000

# Health check hits vLLM's built-in /health endpoint — used by both Docker
# and the Kubernetes readiness probe defined in 3.4.
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["vllm", "serve", "openai/gpt-oss-20b"]
CMD ["--host", "0.0.0.0", \
     "--port", "8000", \
     "--max-model-len", "32768", \
     "--gpu-memory-utilization", "0.90", \
     "--enable-prefix-caching", \
     "--served-model-name", "gpt-oss-20b-v1"]
```

Notes worth calling out:

- `--max-model-len 32768` caps context below the model's full 128K to bound KV-cache memory per request — the product spec for this scenario doesn't need the full context window, and capping it directly improves how many concurrent requests fit in memory (more on this in the load-test section).
- `--enable-prefix-caching` reuses KV cache across requests that share a prompt prefix (e.g., a shared system prompt) — a meaningful win for chat products where every request starts with the same instructions.
- `--served-model-name gpt-oss-20b-v1` is deliberate: it's the version string that ties this container image to a specific entry in the model registry (Ch. 09), not just "gpt-oss-20b." This is the first of several places in this walkthrough where version identity gets threaded through deliberately — see Section 6 for what happens when a team skips this.

Build and smoke-test locally exactly as Ch. 02 teaches — `docker build`, then `docker run --gpus all -p 8000:8000 <image>`, then a `curl` against `/v1/chat/completions` — before anything touches Kubernetes.

### 3.4 Deploying to Kubernetes with correct GPU scheduling

This is where Ch. 03 (Kubernetes) and Ch. 05 (vLLM) meet. The parts that are easy to get wrong are the GPU resource request/limit, the node selection, and the probes — a misconfigured liveness probe on a slow-starting GPU pod is a classic way to get your own pod killed mid-model-load.

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm-serving
---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpt-oss-20b-v1
  namespace: llm-serving
  labels:
    app: gpt-oss-20b
    version: v1
spec:
  replicas: 3   # starting point; KEDA takes over in 3.6
  selector:
    matchLabels:
      app: gpt-oss-20b
      version: v1
  template:
    metadata:
      labels:
        app: gpt-oss-20b
        version: v1
    spec:
      # Only schedule onto the GPU node pool — the taint/toleration pair below
      # keeps non-GPU workloads off expensive GPU nodes and vice versa.
      nodeSelector:
        node-pool: gpu-a100
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
      containers:
        - name: vllm-server
          image: registry.example.com/gpt-oss-20b:v1
          ports:
            - containerPort: 8000
          resources:
            requests:
              nvidia.com/gpu: 1
              cpu: "4"
              memory: 32Gi
            limits:
              nvidia.com/gpu: 1
              cpu: "8"
              memory: 48Gi
          # Readiness gates traffic; liveness restarts a hung process. The
          # generous startupProbe failure budget matters: model load (weights
          # + CUDA graph capture) can take 60-90s, and a naive livenessProbe
          # without a startupProbe will kill the pod before it ever serves.
          startupProbe:
            httpGet:
              path: /health
              port: 8000
            failureThreshold: 30
            periodSeconds: 5
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 5
            failureThreshold: 2
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 15
            failureThreshold: 3
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: gpt-oss-20b
  namespace: llm-serving
spec:
  selector:
    app: gpt-oss-20b
  ports:
    - port: 80
      targetPort: 8000
---
# pdb.yaml — prevents a node drain / cluster upgrade from taking out every
# replica at once, which on GPU nodes (slow to reschedule, GPUs are scarce)
# is far more painful than on a CPU service.
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: gpt-oss-20b-pdb
  namespace: llm-serving
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: gpt-oss-20b
```

The GPU node pool itself (the `node-pool: gpu-a100` label, the `nvidia.com/gpu` resource becoming schedulable at all) is provided by the **NVIDIA GPU Operator**, which installs the driver, the `k8s-device-plugin`, and (if you need to share one physical GPU across multiple pods for cheaper dev/staging environments) MIG partitioning or time-slicing. For this scenario's production traffic, use whole GPUs, one per pod — MIG/time-slicing is a cost trick for lower-QPS environments, not for a 500 req/s SLO'd path, because it splits a card's memory and compute bandwidth, which directly works against your TTFT budget.

### 3.5 Load testing to find the real ceiling

This is the step teams skip and regret. Before wiring autoscaling, you need to know: **how many requests per second can one replica actually sustain at a P95 TTFT under 2 seconds?** Everything downstream (replica count, autoscaling thresholds, cost model) depends on this number, and it is specific to your model, your hardware, your prompt lengths, and your `--max-model-len` — you cannot borrow it from a blog post.

Chapter 04 (Load Testing) covers the general methodology (ramping load, percentile tracking, saturation curves); here's the vLLM-specific piece — a script that measures TTFT correctly by reading the SSE stream rather than waiting for the full response:

```python
# ttft_load_test.py — measures true time-to-first-token against an
# OpenAI-compatible streaming endpoint (vLLM's /v1/chat/completions).
import asyncio
import time
import httpx
import numpy as np

ENDPOINT = "http://gpt-oss-20b.llm-serving.svc.cluster.local/v1/chat/completions"
PROMPT = "Explain the tradeoffs of MIG partitioning vs GPU time-slicing."

async def one_request(client: httpx.AsyncClient) -> float:
    start = time.perf_counter()
    payload = {
        "model": "gpt-oss-20b-v1",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 256,
        "stream": True,
    }
    async with client.stream("POST", ENDPOINT, json=payload, timeout=30) as resp:
        async for chunk in resp.aiter_bytes():
            if chunk:
                return time.perf_counter() - start   # first non-empty chunk = TTFT
    return -1.0

async def run_load(concurrency: int, n_requests: int) -> list[float]:
    ttfts: list[float] = []
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        async def bound():
            async with sem:
                ttfts.append(await one_request(client))
        await asyncio.gather(*(bound() for _ in range(n_requests)))
    return ttfts

async def main():
    # Ramp concurrency and watch where P95 TTFT crosses the 2s SLO.
    for concurrency in (20, 40, 60, 80, 100, 130):
        ttfts = await run_load(concurrency, n_requests=concurrency * 5)
        p50, p95, p99 = np.percentile(ttfts, [50, 95, 99])
        print(f"concurrency={concurrency:4d}  p50={p50:.2f}s  p95={p95:.2f}s  p99={p99:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

Running this against one replica and ramping concurrency produces a saturation curve like the one every capacity plan should be built from:

```
concurrency= 20  p50=0.31s  p95=0.58s  p99=0.71s
concurrency= 40  p50=0.44s  p95=0.89s  p99=1.10s
concurrency= 60  p50=0.61s  p95=1.35s  p99=1.72s
concurrency= 80  p50=0.88s  p95=1.94s  p99=2.60s   <- P95 crosses the 2s SLO here
concurrency=100  p50=1.20s  p95=2.85s  p99=3.90s
concurrency=130  p50=1.90s  p95=4.40s  p99=6.10s
```

(Illustrative numbers from a run of this exact script — your actual curve depends on your GPU, prompt length distribution, and `--max-model-len`; the shape — a knee where P95 suddenly outpaces P50 — is the reliable part, not the exact numbers.)

Read that curve as: **one A100 replica sustains roughly 70-75 concurrent in-flight requests before P95 TTFT breaches 2 seconds.** Continuous batching means concurrency and req/s aren't the same axis, so the next step is converting that concurrency ceiling into a req/s ceiling by measuring completions/sec at that same concurrency — in this worked example that comes out to roughly **90 req/s per replica** at the SLO boundary. For the 500 req/s target with headroom for the 700 req/s burst, that's:

\[
\text{replicas needed} = \lceil \frac{700}{90} \rceil = 8 \text{ replicas at burst}
\]

with a steady-state floor around \(\lceil 500 / 90 \rceil = 6\) replicas. Those two numbers — 6 and 8 — become the KEDA `minReplicaCount` and `maxReplicaCount` in the next section.

### 3.6 Wiring autoscaling

Plain HPA scaling on CPU/memory is close to meaningless for a GPU-bound, batching server — the pod's CPU usage barely moves while the GPU is saturated. The practical default (Ch. 06) is **KEDA**, scaling on a Prometheus query against vLLM's own queue-depth metric, `vllm:num_requests_waiting` — the number of requests sitting in the scheduler queue because the running batch is full. A rising queue is the earliest true signal of saturation, well before GPU utilization alone would tell you.

```yaml
# keda-scaledobject.yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: gpt-oss-20b-scaler
  namespace: llm-serving
spec:
  scaleTargetRef:
    name: gpt-oss-20b-v1
  minReplicaCount: 6     # steady-state floor from the load test in 3.5
  maxReplicaCount: 10    # burst ceiling (8) plus one replica of headroom
  cooldownPeriod: 300    # wait 5 min of low queue depth before scaling down —
                          # GPU pods are slow to warm up, so avoid flapping
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring.svc.cluster.local:9090
        metricName: vllm_queue_depth_per_replica
        # Average queued requests per running replica. Threshold of 5 means:
        # once each replica has ~5 requests backed up on average, add capacity.
        query: |
          sum(vllm:num_requests_waiting{namespace="llm-serving", version="v1"})
          /
          count(up{namespace="llm-serving", version="v1"} == 1)
        threshold: "5"
```

Two details that matter more than the YAML suggests:

- **`cooldownPeriod: 300`.** GPU pods are expensive to spin up (model load + CUDA graph capture can take a minute or more), so an autoscaler tuned like a web-tier HPA (scale down after 60 seconds of low load) will thrash — scaling a replica down right before the next traffic wave needs it back. Err toward a longer cooldown than you'd use for a stateless CPU service.
- **The metric is a rate per replica, not a raw count.** Scaling on the raw `num_requests_waiting` sum without dividing by replica count creates a feedback loop: as you add replicas, the sum doesn't necessarily drop proportionally, so the trigger can either over- or under-react depending on how load actually distributes. Normalizing per replica keeps the signal comparable regardless of current replica count.

### 3.7 Monitoring dashboards and alerts

vLLM exposes a native `/metrics` Prometheus endpoint with per-request histograms — no custom instrumentation needed for the fundamentals (Ch. 08 covers building this out fully; here's the SLO-critical piece for this scenario). The key metrics: `vllm:time_to_first_token_seconds` (histogram), `vllm:request_queue_time_seconds` (time waiting before the scheduler picks the request up), `vllm:e2e_request_latency_seconds`, and `vllm:num_requests_running` / `vllm:num_requests_waiting` (gauges).

The Grafana dashboard for this scenario needs, at minimum, four panels: P50/P95/P99 TTFT over time, queue depth per replica, GPU utilization (from DCGM exporter, installed by the GPU Operator), and requests/sec by version label (so a canary's traffic is visually distinguishable from stable — this reappears in 3.8).

The alert that actually protects the SLO is a burn-rate style alert on the TTFT histogram:

```yaml
# prometheus-alerts.yaml
groups:
  - name: gpt-oss-20b-slo
    rules:
      - alert: TTFTSLOBreach
        expr: |
          histogram_quantile(
            0.95,
            sum(rate(vllm:time_to_first_token_seconds_bucket{namespace="llm-serving", version="v1"}[5m])) by (le)
          ) > 2.0
        for: 3m
        labels:
          severity: page
        annotations:
          summary: "gpt-oss-20b P95 TTFT above 2s SLO for 3+ minutes"
          description: "Check queue depth (vllm:num_requests_waiting) and KEDA scaling activity before assuming a code regression — this fires from load first, bugs second."

      - alert: QueueDepthRising
        expr: |
          sum(vllm:num_requests_waiting{namespace="llm-serving", version="v1"})
          /
          count(up{namespace="llm-serving", version="v1"} == 1)
          > 8
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Queue depth per replica above 8 — KEDA should be scaling; verify it is"
          description: "Early-warning alert, fires before TTFTSLOBreach so on-call has time to react before the SLO alert pages."
```

Note the deliberate ordering: `QueueDepthRising` is a warning that fires *before* the SLO is actually breached, giving on-call a chance to notice the autoscaler is (or isn't) reacting before the paging alert fires. This two-tier pattern — an early leading-indicator warning plus a hard SLO page — is the pattern Ch. 08 recommends generally; here it's tied to the specific metric this platform exposes.

### 3.8 Canary/rollout process for the next model update

Six months in, the team wants to ship gpt-oss-20b-v2 — a new checkpoint, or a new `--max-model-len`/sampling config, or both. This is where Ch. 07 (Canary Deployments) and Ch. 09 (Model Versioning) meet the rest of the running system. Using Argo Rollouts (the general-purpose choice from the Section 2 table):

```yaml
# rollout.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: gpt-oss-20b
  namespace: llm-serving
spec:
  replicas: 8
  strategy:
    canary:
      steps:
        - setWeight: 5      # 5% of traffic to v2 first
        - pause: {duration: 10m}
        - analysis:
            templates:
              - templateName: gpt-oss-slo-check
        - setWeight: 25
        - pause: {duration: 15m}
        - analysis:
            templates:
              - templateName: gpt-oss-slo-check
        - setWeight: 100
  selector:
    matchLabels:
      app: gpt-oss-20b
  template:
    metadata:
      labels:
        app: gpt-oss-20b
    spec:
      containers:
        - name: vllm-server
          image: registry.example.com/gpt-oss-20b:v2   # new image = new version identity
          # ... same resources/probes as 3.4
---
# analysistemplate.yaml — the automated go/no-go gate at each canary step
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: gpt-oss-slo-check
  namespace: llm-serving
spec:
  metrics:
    - name: ttft-p95
      interval: 2m
      successCondition: result < 2.0
      failureLimit: 2
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc.cluster.local:9090
          query: |
            histogram_quantile(0.95,
              sum(rate(vllm:time_to_first_token_seconds_bucket{version="canary"}[5m])) by (le))
    - name: error-rate
      interval: 2m
      successCondition: result < 0.01
      failureLimit: 2
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc.cluster.local:9090
          query: |
            sum(rate(http_requests_total{namespace="llm-serving", version="canary", status=~"5.."}[5m]))
            /
            sum(rate(http_requests_total{namespace="llm-serving", version="canary"}[5m]))
```

This automatically halts and rolls back the canary if either the TTFT SLO or the error-rate threshold breaches during the pause windows, without a human needing to be watching a dashboard in real time. Two things this YAML deliberately does *not* solve, on purpose, so we can talk about them precisely in Section 6: it says nothing about whether the **prompt template** shipped with v2 is also versioned and rolled back together with the weights, and it says nothing about what the **autoscaler** does to pod counts while this rollout is in progress. Both are real incidents, not hypotheticals — covered next.

## 4. Cost Engineering

This section builds a real cost model on top of the walkthrough in Section 3, tying together the GPU/quantization choices of Ch. 05, the autoscaling headroom of Ch. 06, and the versioning overhead of Ch. 09. The goal isn't a single number — it's a model you can re-run when any input changes (GPU price, traffic, SLO).

### 4.1 The baseline: cost per replica-hour

Take the A100 80GB choice from 3.2, at an illustrative on-demand rate of \(\$1.99\)/hour (2026 baseline; real quotes range roughly \(\$1.49\)-\(\$6.98\)/hour across providers depending on region, commitment, and spot vs on-demand — always get a current quote before committing to a number in a budget review).

From the load test in 3.5, one replica sustains about 90 req/s at the SLO boundary. That gives a **cost per 1,000 requests at the SLO ceiling**:

\[
\text{cost per 1k req} = \frac{\$1.99/\text{hr}}{90 \text{ req/s} \times 3600 \text{ s/hr}} \times 1000 = \frac{\$1.99}{324{,}000} \times 1000 \approx \$0.0061
\]

That's the cost floor — the number you get if every replica runs pinned at exactly the SLO boundary, all day, every day. No real system runs there, which is exactly what the next section quantifies.

### 4.2 Utilization and autoscaling headroom

Section 3.6 set `minReplicaCount: 6` and `maxReplicaCount: 10`, with the 90 req/s-per-replica ceiling from the load test. At the 500 req/s steady-state target:

\[
\text{replicas at steady state} = \frac{500}{90} \approx 5.6 \rightarrow 6 \text{ replicas (rounds up to the floor)}
\]

\[
\text{effective utilization} = \frac{500}{6 \times 90} = \frac{500}{540} \approx 92.6\%
\]

That's a good utilization number — the 6-replica floor was sized close to the actual steady-state need. But steady state isn't the whole day: real traffic has a diurnal curve, and the honest cost model has to integrate over it, not just price the peak or the trough.

| Time of day | Req/s | Replicas needed (@ 90 req/s each) | Replica-hours in this window (4h blocks) |
|---|---|---|---|
| Overnight trough | 150 | 6 (floor, not 2 — can't go below `minReplicaCount`) | 24 |
| Daytime baseline | 400 | 6 | 24 |
| Evening peak | 650 | 8 | 24 |
| Burst spikes (rare, ~1h/day) | 700 | 8 (at max before hitting `maxReplicaCount: 10` ceiling) | 8 |

Blended average replica count across a day here is roughly \((6 \times 16 + 8 \times 8)/24 \approx 6.67\) replica-hours per hour of wall clock, i.e. about **160 replica-hours/day**, versus a naive "always run 8 for the burst case" static allocation of 192 replica-hours/day. That's the autoscaling headroom paying for itself: **roughly 17% fewer GPU-hours** than statically provisioning for peak, purely from the `minReplicaCount`/`maxReplicaCount` band matching the real traffic curve instead of a fixed pool sized for the worst case.

The `minReplicaCount: 6` floor is itself a cost decision, not just a latency one: it exists because GPU pods are slow to cold-start (model load + CUDA graph capture), so scaling below 6 to chase overnight-trough savings would mean the autoscaler can't react fast enough to the next demand ramp without a TTFT SLO breach during the scale-up window. That tradeoff — floor higher than the strict minimum traffic requires, in exchange for avoiding cold-start latency spikes — is a cost-vs-reliability decision every autoscaling config makes implicitly; make it explicitly and write down why.

### 4.3 Quantization tradeoffs

gpt-oss-20b's native MXFP4 format is close to a free lunch here — it's how the model ships, not an extra step you're choosing to take on. The more general tradeoff, worth understanding for models that *don't* ship pre-quantized, is: quantizing further (e.g., additional INT4/AWQ on top of an already-BF16 checkpoint) trades a small, usually-recoverable quality loss for a real memory reduction, and that memory reduction converts directly into either (a) fitting on a cheaper/smaller GPU, or (b) fitting more KV cache on the same GPU, which raises the concurrency ceiling found in the load test and therefore lowers cost-per-request. Concretely:

\[
\text{cost per request} \propto \frac{\text{GPU \$/hr}}{\text{req/s ceiling at that quantization level}}
\]

Both terms move when you quantize further — the numerator can drop (cheaper GPU fits) and the denominator can rise (more concurrent requests fit in the freed-up memory) — which is why quantization is usually the single highest-leverage cost lever available, ahead of autoscaling tuning or GPU shopping. It's also the one with the least reversible risk if pushed too far: below a certain bit-width, quality regressions stop being subtle and start being visible to users, which is exactly the kind of thing the drift/quality monitor (Ch. 10) and the eval gate in CI need to catch before a more-quantized version ever becomes a canary.

### 4.4 Putting it together

The full monthly cost estimate for this scenario, combining 4.1-4.3:

\[
\text{monthly GPU cost} \approx 160 \text{ replica-hours/day} \times 30 \text{ days} \times \$1.99/\text{hr} \approx \$9{,}552/\text{month}
\]

against a naive fixed-8-replica baseline of:

\[
8 \times 24 \times 30 \times \$1.99 \approx \$11{,}462/\text{month}
\]

— roughly \(\$1{,}900\)/month saved purely from autoscaling headroom matching the traffic curve, on top of whatever the GPU-choice decision in 3.2 (A100 vs H100) already saved versus defaulting to the most expensive card. Neither number includes the model server's own overhead (registry storage, CI/eval compute, observability stack) — those are real but typically small (single-digit percentage) relative to the GPU line, and should be budgeted separately rather than folded into the per-request math, since they don't scale with request volume the same way.

## 5. Failure Modes That Only Show Up at the System Level

Every chapter in this guide has its own failure-mode list, and those lists are correct — but they assume the rest of the system behaves. The incidents below all happened (in one form or another, across real teams) because two components, each individually correct and each passing its own chapter's checklist, interacted in a way neither owner anticipated. This is the section to reread before an incident retro, not just before a launch.

### 5.1 The canary that passed, then got un-canaried by the autoscaler

**Symptom:** A canary rollout (3.8) completes — all analysis steps pass, weight hits 100%. Twenty minutes later, users start reporting the *old* model's behavior again, even though the Rollout object shows v2 at 100%.

**Root cause:** The Rollout controller manages traffic weight and a target replica count, but the HPA/KEDA autoscaler (Ch. 06) has its own idea of the right replica count for the *Deployment* (or ReplicaSet) it's attached to, independently of what the canary is doing. During the canary, traffic to the stable version dropped as weight shifted to v2, so the autoscaler — reacting correctly, in isolation, to falling queue depth on the stable ReplicaSet — scaled stable *down*. Then, when the canary promoted and Argo Rollouts scaled the stable ReplicaSet to zero (or attempted to), a race between the autoscaler's own next reconcile loop and the Rollout controller's scale-down briefly (or not-so-briefly, depending on cooldown settings) left stable pods running and *receiving traffic from a stale Service selector or an in-flight load balancer connection pool* that hadn't yet drained toward v2.

**Why no single chapter's checklist catches this:** Ch. 06's autoscaling checklist verifies the autoscaler correctly tracks load for a *given* Deployment. Ch. 07's canary checklist verifies traffic-weight steps and analysis gates work correctly for a *given* Rollout. Neither chapter's model of the system includes the other one's controller reconciling against the same underlying ReplicaSets on independent timers.

**Fix:** Two independent controllers must not both have opinions about the same ReplicaSet's replica count during a rollout. In practice: let Argo Rollouts fully own replica count during an active rollout (its own canary/stable ReplicaSet management already does this correctly if you don't *also* point an HPA/KEDA ScaledObject directly at the same ReplicaSets) — point the ScaledObject at the Rollout resource itself, not at the underlying Deployment/ReplicaSet, so there is exactly one control loop deciding "how many pods, of which version" at any moment. Verify this by watching `kubectl get replicasets -w` through a full canary cycle in staging before trusting it in production — the race is timing-dependent and easy to miss in a quick test.

### 5.2 The rollback that forgot the prompt template

**Symptom:** v2's canary looks fine on every automated metric — latency, error rate, throughput. Days after full promotion, a slow-building wave of user complaints about "weird" or subtly wrong answers surfaces. The on-call engineer rolls back to v1's *weights* (redeploys the v1 image). The complaints don't stop.

**Root cause:** The "version" that actually determines model behavior is the tuple mentioned in Section 1: weights + tokenizer + prompt/chat template + sampling defaults + engine config. v2 shipped with weights **and** an updated system prompt (a wording tweak meant to reduce refusals) that was deployed through a *separate* path — a config service, a feature flag, or a prompt-management tool outside the model registry (Ch. 09) entirely. Rolling back the container image reverted the weights but not the prompt template, because they were never versioned as one unit. The model now runs v1 weights against v2's prompt — a combination that was never tested, canaried, or evaluated together.

**Why no single chapter's checklist catches this:** Ch. 09 (Model Versioning) teaches you to version model artifacts rigorously — but if the prompt template lives in a separate system owned by a different team (product, or a "prompt ops" tool), it's outside that chapter's scope by construction, and nobody's checklist spans both systems. Ch. 07's canary checklist verifies the canary rollout mechanism, not what's inside the versioned unit it's rolling out.

**Fix:** The prompt template, chat template, and sampling defaults must be **immutable artifacts of the same version bump** as the weights — stored in the same registry entry (Ch. 09), baked into the same container image or pulled by version-pinned reference at startup, never mutated independently via a config flag that isn't itself versioned and rolled back atomically with the model. If product needs to iterate on prompts faster than model releases, that's a legitimate need — but it must go through the same canary/registry/rollback machinery as a weights change, not around it. A useful audit question for any platform: "if I roll back the model version right now, what *doesn't* roll back with it?" If the honest answer is "the prompt," you have this bug waiting to happen.

### 5.3 Monitoring blind spots between layers

**Symptom:** Every per-component dashboard is green — gateway 2xx rate is high, model-server P95 TTFT is under SLO, GPU utilization looks healthy — and yet real users are experiencing timeouts.

**Root cause (three variants, all real):**

1. **The gateway's timeout is shorter than the model server's.** The gateway (Ch. 01/03 layer) times out and returns a clean 504 after, say, 10 seconds; the model server metrics only record latency for requests it actually *completes*, so a request the gateway already gave up on never shows up in the model server's TTFT histogram at all — it just silently doesn't count. The model server's dashboard looks perfect because it's only measuring the subset of requests that didn't get cut off upstream.
2. **Per-pod metrics look fine while the aggregate breaches**, because the load balancer isn't distributing evenly — a sticky-session or weighted routing bug sends a disproportionate share of traffic to a subset of replicas, which individually stay under their own alert thresholds while the P95 measured *client-side* (across all replicas) breaches. Per-pod dashboards (Ch. 08's default view) can hide this completely; you need a client-side or gateway-side latency view, not just server-side histograms, to catch it.
3. **The drift monitor (Ch. 10) is watching the stable version's traffic distribution, not the canary's.** During a canary, 5-25% of traffic is going to a version whose input distribution characteristics (e.g., if the canary changes `--max-model-len` or a routing rule shifts prompt lengths) are never compared against baseline until full promotion — by which point it's not a canary anymore, it's just production, and any drift-driven quality regression has already reached 100% of users.

**Why no single chapter's checklist catches this:** each chapter's monitoring guidance is scoped to the component it teaches. Ch. 08 teaches you to monitor the model server; it doesn't mandate a client-side or gateway-side view. Ch. 10 teaches drift detection generally; nothing in that chapter forces you to apply it per-version during a canary rather than only to the aggregate production stream.

**Fix:** Maintain at least one **cross-layer** latency measurement (synthetic canary requests sent from outside the cluster, measuring true end-to-end time including the gateway) in addition to the model server's own histograms, and make sure every per-component alert and every drift check is labeled and filterable by `version`, so a canary's behavior is visible in isolation before it becomes the whole system's behavior.

### 5.4 Multi-region registry divergence

**Symptom:** A canary promotes cleanly in `us-east`, the team calls the rollout done, and three weeks later a `eu-west` on-call engineer discovers the region has been silently running the old version the entire time.

**Root cause:** The model registry (Ch. 09) was treated as a per-region resource — each region's cluster pulled from its own local registry mirror or config, and the promotion pipeline only pushed the "promote to v2" event to the region where the on-call engineer happened to run it. Nothing in the system enforced that "promoted" means the same thing in every region simultaneously.

**Fix:** Per Section 2.3, if you operate multi-region at all, the model registry must be a genuinely global source of truth (or have an explicit, monitored, alerting-backed replication/sync step) — "promoted" needs to be a single fact checked against every region's actual running version, not an event fired once and assumed to propagate.

## 6. Pre-Launch Checklist

A consolidated checklist across all eleven chapters, ordered roughly the way you'd actually verify it. Treat "no" on any item as a launch blocker unless you can name the specific, accepted risk you're taking instead.

**Serving fundamentals (Ch. 01, Ch. 02)**
- [ ] The server handles malformed requests, oversized inputs, and client disconnects without crashing or leaking GPU memory.
- [ ] The container image is pinned to an exact base-image tag and dependency versions — no `:latest`.
- [ ] The container runs as a non-root user; secrets (API keys, registry credentials) are injected at runtime, never baked into the image.
- [ ] Local `docker run --gpus all` smoke test passes before anything touches Kubernetes.

**Kubernetes (Ch. 03)**
- [ ] GPU resource requests/limits are set on every model-server pod spec (no pod can silently oversubscribe a GPU).
- [ ] `startupProbe` accounts for real model-load time (weights + CUDA graph capture), verified by timing an actual cold start, not guessed.
- [ ] A PodDisruptionBudget exists so a node drain or cluster upgrade can't take out every replica simultaneously.
- [ ] Node taints/tolerations and nodeSelectors correctly keep GPU workloads on GPU nodes and off them for everything else.

**Load testing (Ch. 04)**
- [ ] A real load test (not a guess) has produced a concurrency-vs-latency saturation curve for the actual production model, hardware, and `--max-model-len`.
- [ ] The req/s-per-replica ceiling used for capacity planning and autoscaling thresholds comes from that curve, not from a vendor blog post or a different model's numbers.
- [ ] Load tests include realistic prompt-length distributions, not just short synthetic prompts — TTFT and memory pressure both depend heavily on prefill length.

**Serving engine (Ch. 05, Ch. 11)**
- [ ] Quantization format (if any) has been validated for quality against a golden eval set, not just for throughput.
- [ ] `--max-model-len`, `--gpu-memory-utilization`, and prefix-caching settings are deliberate choices, documented with the reasoning, not defaults left untouched.
- [ ] If using Triton: backend choice (vLLM backend vs TensorRT-LLM backend) matches the actual multi-model/ensemble requirement, not just habit.

**Autoscaling (Ch. 06)**
- [ ] Autoscaling triggers on a GPU/queue-relevant signal (queue depth, GPU utilization) — not on CPU/memory alone.
- [ ] `minReplicaCount` accounts for cold-start time, not just steady-state traffic minimums.
- [ ] Cooldown/stabilization windows are tuned for GPU pod spin-up latency, not copied from a CPU-service HPA config.
- [ ] Only one control loop owns replica count for any given ReplicaSet during a rollout (see Section 5.1).

**Canary/rollout (Ch. 07)**
- [ ] Automated analysis gates (latency, error rate, and ideally a quality/eval signal) run at every canary step — no step is "promote and hope."
- [ ] Rollback is a single action that reverts weights, prompt template, and engine config together (see Section 5.2) — verified by actually triggering a rollback in staging and checking all three reverted.
- [ ] Canary traffic is labeled distinctly (`version` label) all the way through metrics, logs, and drift checks.

**Monitoring (Ch. 08)**
- [ ] Dashboards exist for TTFT, inter-token latency, queue depth, GPU utilization, and error rate, all filterable by model version.
- [ ] At least one client-side or gateway-side synthetic latency check exists in addition to server-side histograms (see Section 5.3).
- [ ] Every SLO-protecting alert has a `for:` duration tuned to avoid paging on a single noisy scrape, and a documented runbook link in the alert annotation.
- [ ] A leading-indicator warning alert (e.g., rising queue depth) exists ahead of the hard SLO-breach page, so on-call has reaction time.

**Model versioning (Ch. 09)**
- [ ] Every deployed version is a registry entry that captures weights + tokenizer + prompt template + sampling defaults + engine config as one unit.
- [ ] No config path exists that can change model-affecting behavior (prompt, sampling params) outside the versioned registry entry.
- [ ] Multi-region deployments treat "promoted" as a single global fact, checked per region, not an event assumed to propagate (see Section 5.4).

**Drift detection (Ch. 10)**
- [ ] Drift/quality monitoring runs per-version (including on canary traffic specifically), not only on the aggregate production stream.
- [ ] A documented baseline (input distribution + quality scores) exists from the last known-good version to compare against.
- [ ] Drift alerts feed back into the canary/rollout controller's promotion decision, not just into a separate dashboard nobody watches during a rollout.

**Cost (Section 4 of this chapter)**
- [ ] A capacity plan exists that's derived from the actual load-test ceiling, not a round-number guess.
- [ ] `minReplicaCount`/`maxReplicaCount` are justified against the real traffic curve (diurnal pattern), not set to "whatever felt safe."
- [ ] Someone has computed cost-per-1000-requests at the current stack choice and can defend it against at least one alternative (different GPU, different quantization).

**Organizational**
- [ ] On-call has run at least one game-day: trigger a canary rollback, kill a pod mid-request, and simulate a GPU node failure, and confirm the system (and the humans) behave as expected.
- [ ] The pre-launch checklist itself is versioned somewhere and gets revisited after the first real incident — the failure modes in Section 5 were all discovered this way, not designed for up front.

## 7. Interview Mastery

### 7.1 The 60-second answer

**"Walk me through how you'd stand up an LLM inference platform."**

> "I'd start by picking the engine based on the actual requirement — vLLM if it's one or a few open-weight models and I want max throughput per GPU dollar, Triton if I need one server fronting multiple frameworks or models. I'd containerize that engine with a pinned image and a health endpoint, then deploy it to Kubernetes with correct GPU resource requests, a startup probe that accounts for real model-load time, and a PodDisruptionBudget so node drains don't take out every replica. Before I wire any autoscaling, I load test to find the actual concurrency-vs-latency ceiling for that model on that hardware — that number drives everything downstream. Autoscaling then triggers on a GPU-relevant signal like queue depth, with a `minReplicaCount` that accounts for cold-start time, using KEDA rather than plain HPA. New model versions go through a canary — Argo Rollouts or KServe — with automated analysis gates on latency and error rate, and the version being rolled out is the full tuple: weights, prompt template, sampling config, all versioned and rolled back together, not just the checkpoint. Observability wraps the whole thing: Prometheus scraping the engine's own metrics, dashboards split by version, alerts with a leading indicator ahead of the hard SLO page. And a drift/quality monitor watches whether the model's actual behavior in production still matches what was validated, per version, feeding back into whether a canary should be trusted to promote. The parts that bite you in production are almost never inside one of those boxes — they're in the seams between them, like an autoscaler and a canary controller both trying to own the same ReplicaSet's replica count."

### 7.2 Q&A

**Q1: Why vLLM over raw HuggingFace `transformers.generate()` for production serving?**
A: Naive `generate()` processes requests with no batching sophistication — either one at a time, or static batches that stall on the slowest sequence in the batch. vLLM's continuous batching admits and evicts requests from a running batch every iteration, so a fast-finishing sequence's slot is immediately reused, and PagedAttention manages KV cache in fixed-size blocks instead of one contiguous allocation per sequence, eliminating the fragmentation that would otherwise cap concurrency far below what the GPU's memory could actually support. The practical result is roughly an order of magnitude more throughput at comparable latency (Ch. 05).

**Q2: When would you choose Triton over vLLM?**
A: When the actual requirement is multi-model or multi-framework serving behind one protocol — PyTorch, ONNX, TensorRT-LLM, and custom Python backends all fronted by one server, with ensembles or model pipelines — or when the org already has NVIDIA-centric MLOps tooling built around Triton. Triton can also run vLLM as a backend, so it's not strictly either/or; the question is whether you need Triton's multi-model management plane on top of whatever engine does the actual generation.

**Q3: How does continuous batching interact with autoscaling decisions?**
A: Continuous batching means one replica's capacity isn't a fixed req/s number — it's a saturation curve where latency stays flat until a concurrency knee, then degrades sharply (Section 3.5). Autoscaling has to trigger on a signal that reflects *queue* pressure (requests waiting because the running batch is full), not raw req/s or CPU, because req/s alone doesn't tell you where you are on that curve — the same req/s can be comfortable or already-saturated depending on prompt length and generation length.

**Q4: Explain PagedAttention and why it matters for capacity planning.**
A: It manages the KV cache in fixed-size, non-contiguous blocks (like OS virtual memory pages) instead of pre-allocating one contiguous buffer per sequence sized for the worst case. That eliminates internal fragmentation from over-allocation and external fragmentation from variable sequence lengths, so a given amount of GPU memory supports meaningfully more concurrent sequences. For capacity planning, this means the concurrency ceiling you measure in a load test is much closer to what the hardware can actually deliver, rather than being capped by memory-allocation inefficiency (Ch. 05).

**Q5: How do you correctly measure TTFT vs end-to-end latency, and why does the distinction matter for SLOs?**
A: TTFT is measured from request start to the first streamed token/chunk — for a streaming chat UI, that's the number users actually perceive as "responsiveness." End-to-end latency includes the full generation, which scales with output length and is a worse proxy for perceived responsiveness on long generations. Measuring TTFT requires reading the actual SSE/stream response and timing the first non-empty chunk (Section 3.5's load-test script), not waiting for the full response and back-computing an average — that would hide exactly the queueing behavior an SLO is meant to catch.

**Q6: Design a canary rollout strategy for a new model version — what gates would you use?**
A: Staged traffic weights (e.g., 5% → 25% → 100%) with a pause and an automated analysis gate at each step, checking at minimum: P95/P99 latency (TTFT specifically, not just end-to-end) against the SLO, error rate, and ideally an automated quality/eval signal on a golden set of prompts sampled through the canary specifically. Rollback on gate failure should be automatic, not paged-and-manual. Critically, the "version" under test must include the prompt template and sampling config, not just the weights (Section 5.2).

**Q7: What's wrong with scaling GPU pods on CPU/memory metrics?**
A: A GPU-bound inference server's CPU usage barely correlates with how saturated the GPU actually is — the bottleneck is GPU compute and memory (KV cache), not CPU cycles. Scaling on CPU either reacts too late (GPU is already saturated well before CPU shows it) or not at all. The fix is scaling on a signal that directly reflects GPU-side load: queue depth, GPU utilization from DCGM, or a custom Prometheus metric the engine exposes (Ch. 06).

**Q8: How do you handle a rollback that needs to revert more than just the model weights?**
A: Treat the deployable unit as weights + tokenizer + prompt/chat template + sampling defaults + engine config, versioned and stored together in the model registry, so that "roll back to v1" is a single atomic action that reverts all of it — never a container-image rollback plus a hope that nothing else changed independently. Verify this in staging by actually triggering a rollback and diffing every one of those components against what v1 originally shipped with (Section 5.2).

**Q9: What GPU scheduling primitives would you use in Kubernetes, and when?**
A: Whole-GPU-per-pod scheduling via the NVIDIA device plugin for production, latency-sensitive traffic — you want the full memory bandwidth and compute of the card, not a shared slice. MIG (physical partitioning with hard memory/compute isolation) or time-slicing (soft sharing of one GPU across multiple pods) are for lower-QPS environments — dev/staging, batch/offline inference, or many small models that individually don't need a full GPU — because both reduce the compute and/or memory available to any one workload, which works against a tight latency SLO (Ch. 03).

**Q10: How would you detect and respond to model drift in production?**
A: Compare live input distribution characteristics (prompt length, topic/embedding-space shift) and output quality signals (automated eval scores, user feedback signals like regeneration rate) against a documented baseline captured from the last known-good version. Crucially, run this per-version — including on canary traffic specifically during a rollout, not only on the aggregate post-promotion stream — and feed drift alerts back into the canary controller's promotion decision, not just into a dashboard (Ch. 10, Section 5.3).

**Q11: Walk through your cost model for a serving platform — what levers matter most?**
A: Start from a measured (not assumed) req/s-per-replica ceiling at your SLO, derive replica-hours needed against your actual traffic curve (not just peak or average), and price that against real GPU $/hr for your chosen card. The highest-leverage lever is usually quantization, because it moves both sides of the cost-per-request ratio at once (cheaper GPU fits, and/or more concurrency fits per GPU); the second is matching autoscaling's `min`/`maxReplicaCount` band to the real diurnal traffic curve rather than either a static peak-sized pool or a floor set too low to survive cold-start latency (Section 4).

**Q12: What happens to KV cache / prefix cache across a canary or version bump?**
A: Prefix/KV cache is per-process and per-model-version — it does not and should not transfer across a version boundary, since a cache entry computed under one set of weights or one prompt template is not valid under another. Practically this means a fresh canary replica starts "cold" on prefix caching, so early canary-traffic latency samples may look slightly worse than steady-state stable traffic purely from cache-warmth differences — worth accounting for when reading a canary's early analysis-gate metrics so you don't misattribute a warm-up artifact to a real regression.

**Q13: How do you avoid the "canary passed, then the autoscaler undid it" bug?**
A: Make sure exactly one control loop owns replica count for a given ReplicaSet during an active rollout — point the autoscaler at the Rollout resource itself rather than also pointing it independently at the underlying stable/canary ReplicaSets, so there's no race between the canary controller's traffic-weight/replica-count management and a separate autoscaler reconciling the same pods on its own timer (Section 5.1).

**Q14: Single-region vs multi-region — how do you decide, and what's the biggest operational risk in multi-region?**
A: Start single-region; move to multi-region only for a specific forcing function — global latency requirements, a contractual availability SLA a single region's outage would breach, GPU capacity constraints, or data-residency law. The biggest operational risk is registry/state divergence: "promoted" or "rolled back" needs to be a fact checked against every region's actual running version, not an event assumed to propagate — silent regional divergence is a real, recurring incident pattern (Section 5.4).

**Q15: What's your incident response plan if the TTFT SLO breaches during a traffic spike?**
A: First check whether the autoscaler is reacting — queue depth rising with replica count flat suggests either a cooldown/cap issue or a genuinely unprecedented spike beyond `maxReplicaCount`. If capacity is the issue, temporarily raise `maxReplicaCount` (with awareness of GPU-node-pool availability) rather than tuning thresholds blind mid-incident. If replica count is scaling correctly but latency still breaches, check for a non-capacity cause — a bad canary in flight, an unusually long-prompt traffic pattern, or a GPU-health issue (falling back to a degraded state without crashing, which health probes may not catch — Section 5.3). Only after capacity and health are ruled out should you suspect an actual model/code regression.

**Q16: How would you validate a quantization change before shipping it?**
A: Run the golden eval set (the same one gating any model version change, Ch. 10) against the quantized model and compare quality scores directly against the unquantized baseline, not just against a generic benchmark — some quality regressions are task-specific and won't show up on a broad benchmark. Load test separately to confirm the expected throughput/memory win actually materializes on your real hardware and prompt distribution, since quantization's benefit is workload-dependent. Ship it through the same canary pipeline as any other model version change — a quantization change is a version change, not a special case that skips the rollout process.

**Q17: What's the role of a model registry beyond just storing weights?**
A: It's the single source of truth for "what does version N mean" — the full tuple of weights, tokenizer, prompt/chat template, sampling defaults, and engine config, so that a deployment, a canary, a rollback, or a cross-region promotion all reference the same unambiguous definition of a version. Without that, teams end up with version drift where the "same" version means something subtly different depending on which system you ask (Section 5.2, Ch. 09).

**Q18: How do you load test an LLM server correctly, versus a typical web service?**
A: A typical web service load test cares about req/s and a latency percentile that's roughly load-independent up to a hard capacity wall. An LLM server's latency is a function of concurrency, prompt length, and generation length simultaneously, and TTFT specifically requires reading the stream rather than the full response. You need to ramp concurrency (not just req/s) to find the saturation knee, use a realistic prompt-length distribution (not uniform short prompts), and separately track TTFT and inter-token latency, because they degrade differently and an SLO usually cares about TTFT specifically (Ch. 04, Section 3.5).

**Q19: What monitoring blind spot is most likely to bite a team that's only looked at per-component dashboards?**
A: A mismatch between the gateway's request timeout and the model server's own latency histograms — the model server only records latency for requests it completes, so requests the gateway already gave up on vanish from its metrics entirely, making the model-server dashboard look healthier than what users actually experience. The fix is a client-side or gateway-side synthetic latency check in addition to server-side histograms (Section 5.3).

**Q20: Tell me about a failure mode that spans two systems/chapters that most engineers miss.**
A: (Use any of Section 5's incidents as source material — the autoscaler-undoes-the-canary race in 5.1 is the strongest one to lead with, because it demonstrates the core idea of this whole chapter: individually correct components, incorrect system.) Frame the answer around: what looked fine per-component, what the actual interaction was, and what changed structurally (not just "we added a dashboard") to prevent recurrence — interviewers are listening for whether you fix the root interaction or just add monitoring around the symptom.

### 7.3 Tradeoff tables

**Engine choice**

| | Raw Transformers | vLLM | Triton (+ vLLM or TensorRT-LLM backend) | Managed API |
|---|---|---|---|---|
| Throughput/GPU-$ | Low | High | High (matches underlying backend) | N/A (you pay per-token, not per-GPU) |
| Ops burden | Low (but you own scaling/batching yourself) | Medium | Medium-High (extra server layer) | Lowest |
| Multi-model/framework | Poor fit | Possible, more manual | Strong native fit | N/A |
| Time to first working demo | Fastest | Fast | Slower (backend build/config step) | Fastest |
| Control over weights/fine-tuning | Full | Full | Full | None/limited |
| Best fit | Prototype, internal tool | Single/few self-hosted models at scale | Many models/frameworks, existing NVIDIA MLOps investment | Small team, no self-hosting requirement |

**Rollout strategy**

| | Big-bang redeploy | Blue/green | Canary (staged weights) |
|---|---|---|---|
| Blast radius on regression | 100% of traffic, immediately | 100% of traffic, immediately (but instant rollback) | Bounded to canary weight until gates pass |
| Infra cost during rollout | None extra | 2x capacity briefly | Slightly more than 1x (canary + stable both running) |
| Catches regressions before full exposure | No | No | Yes, if gates are meaningful |
| Rollback speed | Slow (redeploy old image) | Fast (flip traffic back) | Fast, and often automatic |
| Best fit | Low-stakes internal tools only | Simple services, infrequent releases | Any production LLM serving path with real users |

**Single-region vs multi-region**

| | Single-region | Multi-region |
|---|---|---|
| Operational complexity | Lower | Higher — registry sync, cross-region routing, failover |
| Global latency | Worse for distant users | Better |
| Availability ceiling | Bounded by one region's SLA | Can exceed a single region's SLA, if done correctly |
| Biggest new risk introduced | — | Silent state/version divergence across regions (Section 5.4) |
| Right default | Yes, unless a specific forcing function exists | Only with a dedicated owner for cross-region consistency |

### 7.4 Red flags vs green flags

| Signal | Red flag | Green flag |
|---|---|---|
| Engine choice reasoning | "We used vLLM because everyone does" | "We compared vLLM and Triton against our multi-model requirement and picked vLLM because we only have one model family" |
| Load testing | "We estimated capacity from a blog post's numbers" | "We ran our own concurrency-ramp load test on our actual model/hardware and derived thresholds from that curve" |
| Autoscaling | "HPA on CPU, default settings" | "KEDA on queue depth, with a floor sized for cold-start time, and cooldown tuned for GPU spin-up latency" |
| Canary gates | "We watch the dashboard for 10 minutes and promote" | "Automated analysis template gates on latency, error rate, and a quality signal, with automatic rollback on failure" |
| Versioning | "The model is version-controlled; the prompt lives in a separate config service" | "Weights, prompt template, and sampling config are one versioned, atomically-rollback-able unit" |
| Monitoring | "Only server-side histograms, no client-side check" | "Client-side/synthetic latency check plus server-side histograms, both filterable by version" |
| Multi-region | "We push the promotion and assume it propagates" | "Promotion status is verified per region, with alerting on divergence" |
| Incident retros | "We added a dashboard" | "We changed the control-loop ownership / structural cause, and the dashboard is a secondary safeguard" |
| Cost reasoning | "We picked H100 because it's the fastest" | "We measured the SLO-meeting ceiling on a cheaper card first and only upgraded when the numbers forced it" |

### 7.5 Full system-design prompt

**Prompt:** *"You're the first infrastructure hire at an AI startup serving an open-weight LLM to end users. Design the inference platform for the first 12 months of growth — from a working demo to production traffic at meaningful scale, with a small team."*

**Worked answer:**

Start by asking the two questions that shape everything else: what's the actual model and SLO (assume: a 20B-class open-weight model, chat product, 2s P95 TTFT target), and what's the team (assume: 2 infra engineers, no dedicated MLOps team for at least the first two quarters).

**Months 0-1 — prove it works.** Raw Transformers or a minimal vLLM setup on a single GPU box, behind a thin FastAPI gateway, in Docker Compose (Ch. 01, Ch. 02). No Kubernetes yet — it would be pure overhead at this stage. Goal: validate the model meets product's quality bar and get a rough sense of per-request latency.

**Months 1-3 — make it fast, put it on real infra.** Move to vLLM specifically for continuous batching and PagedAttention (Ch. 05). Move to Kubernetes once there's more than one replica's worth of traffic or any uptime expectation (Ch. 03) — GPU Operator for device plugin/driver management, correct resource requests, startup probes tuned to real model-load time. Run the first real load test (Ch. 04) to get an actual concurrency-vs-latency curve — this number is the single most load-bearing artifact for everything that follows.

**Months 3-6 — stop being one deployment away from an outage.** Add KEDA-based autoscaling on queue depth (Ch. 06), sized from the load test's ceiling, with a `minReplicaCount` that respects cold-start time. Stand up Prometheus/Grafana/Alertmanager (Ch. 08) with TTFT, queue-depth, and error-rate dashboards split by version, plus a leading-indicator alert ahead of the hard SLO page. This is also when a real model registry (Ch. 09) needs to exist — even a small team benefits from "version = weights + prompt template + config, one unit" being true from day one, because retrofitting it after a rollback incident (Section 5.2) is much more painful than building it in from the start.

**Months 6-9 — ship model updates without fear.** Introduce canary rollouts (Ch. 07) — Argo Rollouts is the pragmatic choice for a small team without an existing KServe investment — with automated analysis gates on latency and error rate. Add a drift/quality monitor (Ch. 10) that runs per-version, including on canary traffic specifically, feeding into the promotion decision.

**Months 9-12 — cost discipline and resilience.** Revisit the GPU/quantization choice with real production traffic data (Section 4) — by now there's enough traffic history to make an informed cost-vs-latency tradeoff instead of a launch-day guess. Run a game-day: trigger a canary rollback, kill a pod mid-request, simulate a node failure, and specifically test the failure modes in Section 5 (does the autoscaler fight the canary controller? does a rollback actually revert the prompt template?). Only consider multi-region or Triton/multi-model serving if a specific forcing function has actually appeared by now (global user base, a second model family) — not proactively, because both add real operational surface a 2-person team can't absorb speculatively.

```
Month 0-1        Month 1-3           Month 3-6              Month 6-9                Month 9-12
┌─────────┐    ┌───────────────┐   ┌───────────────────┐  ┌──────────────────────┐  ┌───────────────────────┐
│ Docker   │ →  │ vLLM on K8s,   │ → │ + KEDA autoscaling,│→ │ + Argo Rollouts       │→ │ + cost/quantization    │
│ Compose, │    │ real load test │   │ + Prometheus/      │  │   canary + analysis   │  │   revisit, game-days,  │
│ 1 GPU box│    │ (Ch 01,03,04,  │   │   Grafana + model   │  │   gates + drift        │  │   multi-region/Triton   │
│ (Ch 01,  │    │  05)           │   │   registry          │  │   monitor per-version  │  │   only if a real trigger│
│  02)     │    │                │   │   (Ch 06,08,09)     │  │   (Ch 07, 10)          │  │   exists (Sec 2.3, 4)   │
└─────────┘    └───────────────┘   └───────────────────┘  └──────────────────────┘  └───────────────────────┘
```

The framing an interviewer is checking for: did the answer sequence things in the order risk actually justifies (prove correctness before optimizing speed, optimize speed before adding autoscaling complexity, add rollout safety before adding drift monitoring, and defer multi-region/Triton until an actual forcing function shows up) — versus building every box in the reference architecture on day one because "that's what production looks like." A team of two cannot operate all eleven chapters' worth of machinery simultaneously from a cold start, and pretending otherwise is itself a red flag in a system-design interview.

## 8. Operational Deep Dives

The core narrative (Sections 1-7) is the complete story for most teams. This section collects a few additional pieces of the system that come up once you're operating the platform for real — GPU node-level autoscaling (as distinct from pod-level autoscaling), two more system-level failure modes that surface after the ones in Section 5, and a command cheat sheet worth keeping next to the runbook.

### 8.1 GPU node autoscaling and capacity strategy

Section 3.6's KEDA config scales **pod count**. It says nothing about where those pods actually run — if the GPU node pool doesn't have a free node with a schedulable GPU, a new pod sits `Pending` no matter how correctly KEDA reacted. This is a second autoscaler, one layer down: the **cluster autoscaler** (or your cloud provider's node-pool autoscaling equivalent), which adds and removes GPU *nodes* based on unschedulable pod pressure.

The interaction that catches teams off guard: GPU nodes take meaningfully longer to provision than CPU nodes — driver installation, the GPU Operator's device-plugin registration, and (if using MIG) partitioning all add minutes on top of normal VM boot time. If KEDA scales pod count up in response to a traffic spike, but the cluster autoscaler needs 3-5 minutes to bring a new GPU node online, that gap is exactly where a TTFT SLO breaches during a burst — the pod-level autoscaler did its job correctly, and the platform still failed the SLO, because the *node*-level autoscaler was the actual bottleneck.

Two mitigations, both worth having simultaneously:

- **Keep a small buffer of pre-warmed, unschedulable-until-needed capacity** — either a node pool with a minimum node count above the steady-state pod requirement, or a low-priority "placeholder" pod pattern that reserves node capacity and gets preempted the moment a real model-server pod needs the room. This trades a small amount of idle-GPU cost for closing the node-provisioning gap.
- **Prefer scaling within existing nodes first**: if your GPU nodes support MIG or time-slicing (Ch. 03) for a fraction of your fleet, keep a portion of capacity shareable so a burst can be partially absorbed by fitting more (smaller) pods on already-running nodes while new dedicated nodes come online in parallel, rather than waiting on node provisioning for the entire burst.

On the cost side, this is also where spot/preemptible GPU capacity earns its keep — for the *steady-state floor* of replicas (the `minReplicaCount` from Section 3.6), running on-demand/reserved capacity is worth the latency-safety of guaranteed availability; for the *burst headroom* above that floor, spot capacity is often an acceptable risk, since a preempted spot node during a burst just means falling back toward the SLO boundary rather than losing the floor's worth of guaranteed serving capacity entirely. Reserved/committed-use discounts (available from most major cloud GPU providers for 1-3 year commitments) are worth layering under the steady-state floor once traffic has been stable long enough to trust the number — locking in a discount on a `minReplicaCount` you later have to shrink is its own kind of cost mistake, so this is deliberately a months-9-12 decision (Section 7.5's timeline), not a launch-day one.

### 8.2 Two more failure modes worth knowing before they happen to you

**GPU silent degradation past health checks.** A GPU can develop ECC memory errors, thermal throttling, or Xid errors that degrade performance without crashing the process — the vLLM server's `/health` endpoint (used by the readiness/liveness probes in Section 3.4) keeps returning healthy because the process is, technically, alive and responsive; it's just running 3x slower per token than a healthy replica. Kubernetes has no reason to reschedule it, and the autoscaler has no reason to add capacity, because from the outside the replica count looks sufficient — it's the *quality* of one replica's throughput that degraded, not its count. The fix is monitoring GPU health signals directly (DCGM exporter metrics — ECC error counts, thermal state, Xid error codes — installed alongside the GPU Operator) as a first-class input to alerting, separate from and in addition to the application-level health check, and routing per-replica latency metrics (not just the fleet-wide aggregate) into the dashboard so one visibly slow replica doesn't get averaged away by nine healthy ones.

**Cost runaway from an autoscaler ceiling raised during an incident and never lowered.** During the incident-response pattern described in Q15 of Section 7.2 — raising `maxReplicaCount` mid-incident to absorb an unprecedented spike — it's common, and understandable, for the temporary change to quietly become permanent, because nobody owns reverting it once the incident is resolved and the on-call engineer has moved on. Weeks later, a routine (not even unusually large) traffic bump now scales to the emergency ceiling instead of the originally-sized one, at real and recurring GPU cost, with no incident and no alert to flag it — the system is behaving exactly as configured, which is precisely why nothing catches it. The fix is procedural, not technical: every incident-time config change (autoscaler bounds, probe thresholds, alert silences) gets a tracked follow-up ticket to review and, if appropriate, revert within a fixed window, and periodic (e.g., monthly) audits diff the running autoscaler config against the last deliberately-reviewed baseline.

### 8.3 Quick reference: commands you'll actually run

A short cheat sheet worth keeping next to the runbook, pulling together the commands used across this walkthrough:

```bash
# Build and smoke-test the serving image locally (Section 3.3)
docker build -t registry.example.com/gpt-oss-20b:v1 .
docker run --gpus all -p 8000:8000 registry.example.com/gpt-oss-20b:v1
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-20b-v1","messages":[{"role":"user","content":"hello"}]}'

# Apply the base deployment (Section 3.4)
kubectl apply -f namespace.yaml -f deployment.yaml -f service.yaml -f pdb.yaml

# Watch GPU scheduling — confirm pods actually land on GPU nodes
kubectl get pods -n llm-serving -o wide
kubectl describe node <gpu-node-name> | grep -A5 "Allocated resources"

# Check KEDA's view of the scaling metric (Section 3.6)
kubectl get scaledobject gpt-oss-20b-scaler -n llm-serving -o yaml
kubectl get hpa -n llm-serving   # KEDA creates a backing HPA under the hood

# Query vLLM's own metrics directly, useful when debugging a saturation event
curl -s http://<pod-ip>:8000/metrics | grep -E "vllm:(num_requests|time_to_first_token)"

# Watch a canary rollout live (Section 3.8)
kubectl argo rollouts get rollout gpt-oss-20b -n llm-serving --watch
kubectl argo rollouts promote gpt-oss-20b -n llm-serving     # manual promote if needed
kubectl argo rollouts undo gpt-oss-20b -n llm-serving        # abort and roll back

# Confirm exactly one control loop owns replica count during a rollout (Section 5.1)
kubectl get replicasets -n llm-serving -w
```

### 8.4 A minimal Grafana panel definition

Ch. 08 covers full dashboard construction; here's the smallest useful piece — the TTFT-by-version panel referenced in Section 3.7 — as a Grafana panel JSON snippet you can drop into a dashboard's `panels` array, so the shape of the query is concrete rather than described in prose:

```json
{
  "title": "P95 Time-to-First-Token by Version",
  "type": "timeseries",
  "targets": [
    {
      "expr": "histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket{namespace=\"llm-serving\"}[5m])) by (le, version))",
      "legendFormat": "{{version}}"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "s",
      "thresholds": {
        "steps": [
          { "color": "green", "value": null },
          { "color": "yellow", "value": 1.5 },
          { "color": "red", "value": 2.0 }
        ]
      }
    }
  }
}
```

The `by (le, version)` grouping is the detail that matters most: without splitting by `version`, a canary at 5% traffic gets averaged into the stable version's 95%, and a real canary regression can hide inside an otherwise-healthy aggregate P95 for the entire canary window — exactly the blind spot described in Section 5.3's third variant.

### 8.5 Security and multi-tenancy at the gateway

The reference architecture's gateway box (Section 1) does more than route requests — in any platform serving more than one internal team or one external customer, it's also where authentication, per-tenant rate limiting, and basic input hardening live, none of which is unique to LLM serving but all of which have LLM-specific wrinkles worth naming:

- **Per-tenant rate limiting must account for token cost, not just request count.** A rate limiter that caps "100 requests/minute" per API key treats a request for one token of output the same as a request for 4,000 tokens of output, even though the second consumes vastly more GPU time. Rate limiting (or at least cost attribution and alerting) keyed on estimated or actual token usage is the practical equivalent of request-count limiting for a service where "request" is a poor proxy for cost.
- **Prompt-length limits protect the KV-cache budget, not just the gateway.** A client sending a request near your `--max-model-len` ceiling consumes proportionally more KV-cache memory per request, directly reducing the concurrency ceiling the load test in Section 3.5 measured. Enforcing a sane per-request prompt-length limit at the gateway (well below the hard `--max-model-len`) keeps one large request from degrading everyone else's latency.
- **Multi-tenant isolation for GPU workloads is coarser than for typical multi-tenant CPU services.** Unlike a CPU service where cgroups give reasonably strong per-tenant resource isolation, a shared GPU model-server pod serves all tenants' requests through the same continuous batch — there is no per-tenant GPU-memory or compute isolation *within* a replica. If strict tenant isolation is a hard requirement (e.g., contractual or regulatory), the practical answer is dedicating replicas (or MIG partitions, Ch. 03) per tenant rather than assuming batching-level isolation exists, because it doesn't.
- **Basic input hardening still matters even though "prompt injection" is a model-behavior problem, not an infra problem.** The gateway is a reasonable place to enforce request-size limits, strip or flag obviously malformed input (e.g., attempts to smuggle control tokens the tokenizer would otherwise interpret specially), and log full request/response pairs for the security and drift-monitoring teams (Ch. 10) to review — the infra platform's job is making sure that data exists and is queryable, not solving prompt injection itself.

## 9. Further Reading

Organized by the box in the reference architecture (Section 1) each source speaks to, so you can jump to what's relevant rather than reading a flat list.

### Serving engines (Ch. 05, Ch. 11)

- Kwon, Zhuohan Li, et al., ["Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180), SOSP 2023 — the original vLLM paper; the mechanism behind everything Ch. 05 teaches about memory efficiency.
- [vLLM documentation](https://docs.vllm.ai/) — the canonical reference for CLI flags (`--max-model-len`, `--gpu-memory-utilization`, `--enable-prefix-caching`, etc.) used throughout Section 3.
- [vLLM releases](https://github.com/vllm-project/vllm/releases) — check here for the current stable version before pinning a Dockerfile; this chapter cites the v0.20.x line as current at time of writing.
- [vLLM FP8 quantization docs](https://docs.vllm.ai/en/latest/quantization/fp8.html) and the vLLM blog post ["The State of FP8 KV-Cache and Attention Quantization in vLLM"](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html) — relevant to the quantization tradeoffs in Section 4.3 for models that don't ship natively quantized.
- [openai/gpt-oss-20b model card](https://huggingface.co/openai/gpt-oss-20b) and the [vLLM recipe for gpt-oss-20b](https://recipes.vllm.ai/openai/gpt-oss-20b) — the specific model used in the Section 3 worked walkthrough, including its native MXFP4 quantization and YaRN context extension details.
- [NVIDIA Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/) and the [TensorRT-LLM backend guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html) — the Ch. 11 engine, and the backend choice discussed in Section 2.1/2.4.

### Kubernetes and GPU scheduling (Ch. 03)

- [NVIDIA GPU Operator documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html) — driver install, device plugin, and the [time-slicing/MIG guide](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html) referenced in Sections 2.4 and 3.4.
- [NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) — the component that makes `nvidia.com/gpu` a schedulable Kubernetes resource.
- [vllm-project/production-stack](https://github.com/vllm-project/production-stack) — a reference Kubernetes-native deployment of vLLM, including the [Helm chart](https://docs.vllm.ai/projects/production-stack/en/latest/deployment/helm.html) and the [KEDA autoscaling guide](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html) used directly in Section 3.6.
- [LeaderWorkerSet (LWS) for multi-node vLLM](https://docs.vllm.ai/en/stable/deployment/frameworks/lws/) — relevant once a single replica needs to span multiple GPU nodes (tensor/pipeline parallelism beyond one machine), mentioned in Section 2.4.

### Autoscaling (Ch. 06)

- [KEDA documentation](https://keda.sh/docs/latest/) — the general scaler framework behind the ScaledObject in Section 3.6.
- [AWS EKS: Autoscale AI inference with HPA and KEDA](https://docs.aws.amazon.com/eks/latest/userguide/ml-inference-autoscaling-hpa-keda.html) — a concrete cloud-vendor walkthrough of the same pattern used in this chapter.

### Rollout, canary, and disaggregated serving (Ch. 07)

- [Argo Rollouts documentation](https://argoproj.github.io/rollouts/) and the [canary strategy reference](https://argo-rollouts.readthedocs.io/en/stable/features/canary/) — the controller and analysis-template pattern used in Section 3.8.
- [KServe canary rollout guide](https://kserve.github.io/website/docs/model-serving/predictive-inference/rollout-strategies/canary) — the alternative canary mechanism referenced in Section 2.4 for teams already on KServe's `InferenceService` CRD.
- [llm-d](https://llm-d.ai/) and its [prefill/decode disaggregation guide](https://llm-d.ai/docs/architecture/advanced/disaggregation) — the Kubernetes-native, KV-cache-aware routing project (backed by Red Hat, Google, IBM, and CoreWeave) referenced in Section 2.1/2.4 for extreme-scale deployments.
- [NVIDIA Dynamo](https://www.nvidia.com/en-us/ai/dynamo/) and the ["Dynamo 1.0" production blog post](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/) — NVIDIA's disaggregated-serving framework, an alternative to `llm-d` mentioned in the same section.

### Monitoring and observability (Ch. 08)

- [vLLM metrics design doc](https://docs.vllm.ai/en/stable/design/metrics/) — the source of truth for every `vllm:*` metric name used in Sections 3.7 and 8.4.
- Krisanov, ["Monitoring vLLM in Production: Metrics, PromQL, Alerts, and Runbooks"](https://akrisanov.com/vllm-metrics/) — a practitioner writeup with worked PromQL percentile queries this chapter's alert rules build on.
- [Prometheus documentation](https://prometheus.io/docs/) and [Grafana documentation](https://grafana.com/docs/) — the general observability stack underlying Ch. 08 and this chapter's dashboard/alert examples.

### Model versioning and registries (Ch. 09)

- [MLflow Model Registry documentation](https://mlflow.org/docs/latest/model-registry.html) — one concrete implementation of the registry pattern in Section 1 and Section 6's checklist.
- MLflow, ["Canary Deployment for AI Models: A 2026 Guide"](https://mlflow.org/articles/what-is-canary-deployment-ai/) — connects the versioning and canary concerns directly, relevant to the prompt-template-rollback failure mode in Section 5.2.

### Drift detection (Ch. 10)

- Return to Ch. 10's own deep dive and further-reading list for the statistical mechanism (EWMA, distribution-shift tests) behind the drift monitor box in Section 1 — this chapter deliberately does not re-derive it, only shows where it plugs into the rollout and multi-region flow (Sections 3.8, 5.3, 5.4).

### Cost and GPU pricing (Section 4)

- [IntuitionLabs, "H100 Rental Prices Compared: 15+ Cloud Providers (2026)"](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison) and [CloudZero, "H100 GPU Cost In 2026"](https://www.cloudzero.com/blog/h100-gpu-cost/) — the source range for the GPU pricing used in Section 4.1; re-check current quotes before using these numbers in a real budget review, since GPU cloud pricing moves quickly.
- [SynpixCloud, "Cloud GPU Pricing 2026"](https://www.synpixcloud.com/blog/cloud-gpu-pricing-comparison-2026) — the A100/H100 baseline figures cited in Section 3.2's comparison table.

## Appendix A: Glossary of Cross-Chapter Terms

Terms that get used across multiple chapters and multiple sections of this capstone, in one place, with a pointer back to where each is taught in depth.

| Term | Meaning | Taught in depth |
|---|---|---|
| TTFT (time-to-first-token) | Latency from request start to the first generated token/chunk reaching the client. The primary SLO metric for streaming chat UIs. | Ch. 05, Section 3.5/3.7 |
| ITL (inter-token latency) | Time between successive streamed tokens after the first. Governs how "smooth" a streaming response feels once it's started. | Ch. 05, Section 3.7 |
| TPOT (time per output token) | Related to ITL; average generation-phase time per token, sometimes measured per-request rather than per-token-gap. | Ch. 05 |
| Continuous batching | Scheduling requests into and out of a running GPU batch every iteration, instead of waiting for a static batch to fully complete before admitting new requests. | Ch. 05 |
| PagedAttention | KV-cache memory management using fixed-size, non-contiguous blocks (analogous to OS virtual memory paging), eliminating fragmentation from variable sequence lengths. | Ch. 05 |
| KV cache | Stored key/value attention tensors from previously processed tokens, reused so each new token doesn't require recomputing attention over the whole sequence from scratch. | Ch. 05 |
| Prefix caching | Reusing KV cache across requests that share an identical prompt prefix (e.g., a common system prompt), avoiding redundant prefill computation. | Ch. 05, Section 3.3 |
| Quantization (AWQ, GPTQ, FP8, MXFP4) | Reducing the numeric precision of model weights (and sometimes KV cache/activations) to shrink memory footprint and increase throughput, at some quality cost that must be validated. | Ch. 05, Section 4.3 |
| MIG (Multi-Instance GPU) | Hardware-level partitioning of one physical GPU into isolated instances with dedicated memory/compute slices. | Ch. 03, Section 2.4 |
| Time-slicing | Software-level sharing of one GPU across multiple pods without hardware partitioning — weaker isolation than MIG, cheaper to set up. | Ch. 03, Section 2.4 |
| GPU Operator | NVIDIA's Kubernetes operator that installs the driver, device plugin, DCGM exporter, and MIG/time-slicing config as one managed unit. | Ch. 03, Section 3.4 |
| HPA (Horizontal Pod Autoscaler) | Kubernetes' built-in autoscaling controller, natively driven by CPU/memory metrics (or custom metrics with extra wiring). | Ch. 06 |
| KEDA | An autoscaling framework that extends HPA to scale on arbitrary external metrics (e.g., a Prometheus query), the practical default for GPU/queue-based scaling. | Ch. 06, Section 3.6 |
| ScaledObject | KEDA's custom resource defining what metric, threshold, and bounds drive scaling for a target workload. | Ch. 06, Section 3.6 |
| Cluster autoscaler | The node-level autoscaler that adds/removes machines (as opposed to pods) based on unschedulable-pod pressure. | Section 8.1 |
| Canary deployment | Shifting a small, increasing percentage of traffic to a new version, with automated gates deciding whether to continue or roll back. | Ch. 07, Section 3.8 |
| Blue/green deployment | Running two full-capacity environments and flipping all traffic at once, with instant rollback by flipping back. | Ch. 07, Section 7.3 |
| Argo Rollouts | A Kubernetes controller implementing canary/blue-green strategies with automated analysis-based promotion/rollback. | Ch. 07, Section 3.8 |
| AnalysisTemplate | Argo Rollouts' resource defining the metrics query and success/failure thresholds evaluated at each canary step. | Section 3.8 |
| Model registry | The system of record for what a model "version" is — weights, tokenizer, prompt template, sampling defaults, engine config, versioned as one unit. | Ch. 09, Section 5.2 |
| Drift detection | Statistically monitoring whether live input distribution or output quality has shifted away from a validated baseline. | Ch. 10, Section 5.3 |
| EWMA (exponentially weighted moving average) | A common smoothing technique used in drift monitors to track a metric's trend without over-reacting to single noisy samples. | Ch. 10 |
| Golden set | A fixed, curated set of prompts with known-good expected qualities, used to regression-test a model version before and during rollout. | Ch. 10, Section 6 |
| SLO (service-level objective) | A target threshold for a metric (e.g., "P95 TTFT under 2s") that the platform is built and operated to meet. | Section 3.1 throughout |
| P50/P95/P99 | Percentile latency measures — P95 means 95% of requests were faster than this value. Production SLOs are almost always stated on a tail percentile (P95/P99), not the mean, because the mean hides exactly the slow-request behavior users notice. | Section 3.5 |
| Tensor parallelism | Splitting a single model's weight matrices across multiple GPUs so one logical replica spans several devices — needed once a model is too large to fit (or batch efficiently) on one GPU. | Ch. 05, referenced in Section 2.4 (LWS) |
| Disaggregated serving (prefill/decode) | Running the compute-bound prefill phase and the memory-bandwidth-bound decode phase on separate pools of hardware tuned for each, rather than one pool doing both. | Section 2.1/2.4, `llm-d`/Dynamo references in Section 9 |
| LWS (LeaderWorkerSet) | A Kubernetes API for managing a group of pods as one logical multi-node model replica, used for tensor/pipeline-parallel deployments spanning multiple machines. | Section 2.4 |
| PDB (PodDisruptionBudget) | A Kubernetes resource guaranteeing a minimum number of pods stay available during voluntary disruptions (node drains, upgrades). | Ch. 03, Section 3.4 |
| Xid error / ECC error | GPU-level hardware fault signals (from NVIDIA's driver/DCGM) indicating memory or execution errors that can degrade a GPU's performance without crashing the process running on it. | Section 8.2 |

## Appendix B: Chapter Cross-Reference Map

A one-page index of which section of this capstone exercises each of the guide's eleven chapters most directly — useful as a study map if you're revisiting a specific chapter and want to see it in context.

| Chapter | Where it shows up most concretely in this capstone |
|---|---|
| 01 Basic Serving | Section 1 (gateway box), Section 2.1 (the "prototype" branch of the engine decision tree), Section 7.5 (months 0-1 of the system-design answer) |
| 02 Docker | Section 3.3 (the Dockerfile), Section 6's containerization checklist items |
| 03 Kubernetes | Section 3.4 (the full Deployment/Service/PDB YAML), Section 8.1 (node-level autoscaling), Appendix A (MIG/time-slicing/GPU Operator terms) |
| 04 Load Testing | Section 3.5 (the TTFT-measuring load-test script and saturation curve), Section 6's load-testing checklist |
| 05 vLLM Serving | Section 2.1/2.4 (engine decision and comparison table), Section 3.2-3.3 (quantization and serve-command choices), Appendix A (PagedAttention, continuous batching, KV cache terms) |
| 06 Autoscaling | Section 3.6 (the KEDA ScaledObject), Section 5.1 (the canary/autoscaler race), Section 8.1 (node-level autoscaling) |
| 07 Canary Deployments | Section 3.8 (the Argo Rollouts Rollout + AnalysisTemplate), Section 5.1/5.2 (both flagship system-level incidents), Section 7.3 (rollout-strategy tradeoff table) |
| 08 Monitoring | Section 3.7 (dashboards and PromQL alerts), Section 5.3 (monitoring blind spots), Section 8.4 (Grafana panel JSON) |
| 09 Model Versioning | Section 5.2/5.4 (both versioning-related failure modes), Section 6's versioning checklist, Appendix A (model registry definition) |
| 10 Drift Detection | Section 1 (drift monitor box), Section 5.3 (per-version drift blind spot), Appendix C (wiring drift into a canary gate) |
| 11 Triton | Section 2.1/2.4 (when to choose it over vLLM, and the TensorRT-LLM vs vLLM backend choice) |

## Appendix C: Wiring a Drift/Quality Signal into the Canary Gate

Section 3.8's `AnalysisTemplate` gates on latency and error rate — both operational signals, neither of which catches a canary that's technically fast and error-free but subtly *wrong* (e.g., a quantization change that passed every latency check but quietly degraded answer quality on a class of prompts the golden set didn't cover densely enough). Ch. 10's drift/quality monitor is the component meant to catch that, and it needs to feed into the same gate, not live in a separate dashboard nobody checks mid-rollout (Section 5.3's third failure mode).

The connective piece is small: the drift monitor exposes its own Prometheus gauge, scored per version, and the AnalysisTemplate adds a third metric alongside latency and error rate.

```python
# drift_monitor_exporter.py — a minimal sketch of the gauge the drift
# monitor (Ch. 10) needs to expose per version for the canary gate to read.
from prometheus_client import Gauge

GOLDEN_SET_SCORE = Gauge(
    "model_golden_set_quality_score",
    "Rolling average quality score (0-1) against the golden eval set, per version",
    ["version"],
)

def update_score_after_eval_batch(version: str, batch_scores: list[float]) -> None:
    # In practice this would be an EWMA over recent batches (Ch. 10), not a
    # raw batch average — shown simplified here to keep the wiring visible.
    avg = sum(batch_scores) / len(batch_scores)
    GOLDEN_SET_SCORE.labels(version=version).set(avg)
```

```yaml
# analysistemplate.yaml — extended from Section 3.8 with the quality gate
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: gpt-oss-slo-check
  namespace: llm-serving
spec:
  metrics:
    - name: ttft-p95            # as in Section 3.8
      # ... unchanged ...
    - name: error-rate           # as in Section 3.8
      # ... unchanged ...
    - name: golden-set-quality
      interval: 5m
      successCondition: result >= 0.92    # baseline established from the
                                            # last known-good version (Ch. 10)
      failureLimit: 1
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc.cluster.local:9090
          query: model_golden_set_quality_score{version="canary"}
```

This closes the loop described in Section 1's diagram: the drift/quality monitor doesn't just watch production and report separately — its output becomes a real promote/abort input, with the same automatic, no-human-watching-a-dashboard guarantee the latency and error-rate gates already have.

## Appendix D: Cost Comparison Across Quantization Choices

Extending Section 4.3's quantization discussion with concrete illustrative numbers, for a case where a model does *not* ship natively pre-quantized (unlike gpt-oss-20b in the main walkthrough) and the team is choosing how far to quantize a BF16 checkpoint:

| Format | Relative VRAM footprint | Illustrative req/s ceiling per GPU (same hardware, same SLO) | Quality delta vs BF16 baseline |
|---|---|---|---|
| BF16 (no quantization) | 1.0x (baseline) | 1.0x (baseline) | None (reference point) |
| FP8 (weights + KV cache) | ~0.5x | ~1.6-1.9x | Typically small and often within noise on general benchmarks; must be checked per-task on the golden set (Section 6) |
| INT4 (AWQ/GPTQ-style) | ~0.25x | ~2.5-3.5x | Larger and more task-dependent; some reasoning/long-context tasks show visible degradation below a certain bit-width |

(Illustrative ranges reflecting commonly reported directional effects, not a benchmark run for this chapter — always validate the specific ceiling and quality delta for your model, hardware, and golden set before committing a capacity plan to a quantization choice; see Ch. 05 and the FP8 quantization docs in Section 9 for engine-specific specifics.)

The practical decision rule from Section 4.3 holds here in sharper relief: FP8 is close to a default-safe choice on modern hardware (H100-class GPUs have native FP8 tensor cores, so the throughput win is closer to "free" than the INT4 row's more dramatic — and riskier — VRAM savings). INT4-class quantization is the higher-risk, higher-reward lever, worth reaching for only after the golden-set validation in Section 6 has specifically cleared it for your model and task mix, not adopted by default purely for the cost savings.

## Appendix E: The Same Scenario, Through Triton Instead

Section 3 walks the gpt-oss-20b scenario through vLLM directly, because a single self-hosted model family is exactly the case where vLLM is the right default (Section 2.1). It's worth seeing concretely what changes if the same model needed to sit behind Triton instead — for example, because it now needs to share a server with a second, non-LLM model (a re-ranker or an embedding model) that the product also depends on, which is exactly the kind of requirement that tips the decision in Section 2.1's tree toward Triton.

Triton's model-serving unit is a **model repository** — a directory structure Triton scans at startup, not a single serve command:

```
model_repository/
└── gpt-oss-20b/
    ├── config.pbtxt
    └── 1/
        └── model.json
```

`config.pbtxt` tells Triton which backend owns this model and how many instances to run — with the vLLM backend, device placement is deliberately left to vLLM itself rather than configured here:

```protobuf
# model_repository/gpt-oss-20b/config.pbtxt
backend: "vllm"

instance_group [
  {
    count: 1
    kind: KIND_MODEL
  }
]
```

`model.json` (versioned under `1/`, Triton's convention for model version directories — note this is a *different* versioning axis than the registry versioning in Ch. 09/Section 5.2, and the two should not be conflated) carries the actual vLLM engine arguments, mirroring the flags from the Dockerfile's `CMD` in Section 3.3:

```json
{
  "model": "openai/gpt-oss-20b",
  "max_model_len": 32768,
  "gpu_memory_utilization": 0.90,
  "enable_prefix_caching": true,
  "served_model_name": "gpt-oss-20b-v1"
}
```

Everything **above** the model-server layer in the reference architecture (Section 1) is unaffected: the same Kubernetes Deployment pattern from Section 3.4 applies (Triton's container image replaces vLLM's, GPU resource requests and probes stay conceptually identical — Triton exposes its own `/v2/health/ready` endpoint for the readiness probe instead of vLLM's `/health`), the same KEDA scaling pattern from Section 3.6 applies against Triton's own Prometheus metrics endpoint, and the same Argo Rollouts canary pattern from Section 3.8 applies unchanged — a canary controller shifting traffic weight doesn't care whether the pods behind each weight are running vLLM or Triton. This is the practical payoff of the reference architecture in Section 1 being drawn as boxes rather than as a single tool: swapping the model-server box's implementation doesn't require redesigning the gateway, autoscaler, canary controller, or observability stack around it.

The tradeoff is exactly what Section 2.1 named: Triton's model-repository/config.pbtxt layer is genuine extra operational surface (a second configuration system, on top of Kubernetes YAML, that has to stay in sync with it) that buys you the ability to add that second, non-LLM model to the same server later without standing up an entirely separate serving stack for it.

## Appendix F: A Sample On-Call Runbook Entry

Section 3.7's `TTFTSLOBreach` alert annotation points to "a runbook link" — here's what a real one looks like, concretely enough to adapt rather than write from scratch:

```markdown
## Runbook: TTFTSLOBreach (gpt-oss-20b)

**Alert fires when:** P95 TTFT > 2.0s for 3+ minutes, on the stable-version
label (see the alert query in Section 3.7).

**First 2 minutes — orient, don't act yet:**
1. Open the "P95 TTFT by Version" Grafana panel (Section 8.4). Confirm this
   is fleet-wide, not one version — if it's isolated to a `canary` label,
   this is a canary-quality issue, not a capacity issue; stop the rollout
   (`kubectl argo rollouts undo gpt-oss-20b -n llm-serving`) before anything
   else.
2. Check `QueueDepthRising` (Section 3.7) — did it fire first? If yes, this
   is very likely a capacity/scaling issue, not a regression. If
   `TTFTSLOBreach` fired *without* `QueueDepthRising` firing first, suspect
   a per-replica issue (Section 8.2's GPU silent-degradation failure mode)
   rather than fleet-wide load.

**Minutes 2-10 — capacity path:**
3. `kubectl get scaledobject gpt-oss-20b-scaler -n llm-serving -o yaml` —
   confirm KEDA's current replica target vs `maxReplicaCount`. If pinned at
   the ceiling and queue depth is still rising, this is a real capacity
   shortfall — see step 5.
4. `kubectl get pods -n llm-serving -o wide` — confirm new replicas are
   actually `Running`, not stuck `Pending` (Section 8.1's node-provisioning
   gap). If `Pending`, check `kubectl describe node` for GPU node pool
   capacity; you may need to manually scale the GPU node pool while the
   cluster autoscaler catches up.
5. If genuinely capacity-constrained beyond `maxReplicaCount`, raise it
   temporarily (`kubectl edit scaledobject ...`) — **and immediately file
   the follow-up ticket described in Section 8.2 to review/revert it**,
   so this doesn't become the permanent ceiling by default.

**Minutes 2-10 — per-replica path (if queue depth was NOT rising):**
6. Compare per-pod TTFT, not just the fleet aggregate — one replica
   dragging the P95 up while others look healthy points at GPU hardware
   degradation. Check DCGM metrics (ECC/Xid errors, Section 8.2) for the
   specific node backing the slow replica; cordon and drain that node if
   confirmed, and file a hardware ticket with the cloud provider.

**If neither path resolves it within 15 minutes:** escalate to the
model-serving on-call lead and consider a manual rollback to the last
known-good version, following the atomic weights+prompt+config rollback
process (Section 5.2) — not a partial rollback of just the container image.
```

## Appendix G: Anti-Patterns Worth Naming Explicitly

A short list of design smells that show up repeatedly across real platforms, distinct from the acute incidents in Section 5 — these are chronic, not acute, and tend to accumulate quietly rather than trigger a single obvious page.

- **The autoscaler config nobody has looked at since launch.** `minReplicaCount`/`maxReplicaCount` set from the launch-day load test (Section 3.5-3.6) and never revisited as real traffic patterns, prompt-length distributions, or the model itself changed. Revisit this on the same cadence as the cost review in Section 7.5's months 9-12.
- **Alerts with no owner and no runbook.** An alert that pages but has no linked runbook (Appendix F) trains on-call to snooze it rather than act on it — by the time a real incident needs that alert to be trusted, it's been ignored for months.
- **A model registry that's a source of truth in name only.** The registry (Ch. 09) exists, but a config flag or feature-flag system outside it can still change prompt/sampling behavior in production (Section 5.2) — the registry's authority is only as real as the absence of side doors around it.
- **Dashboards that only show the aggregate, never split by version.** Built once, before canaries existed, and never updated to add the `version` label split (Section 5.3, Section 8.4) — quietly useless for exactly the moment (a canary rollout) it would matter most.
- **"We'll add monitoring for that after launch."** Said about GPU-health metrics (Section 8.2), client-side synthetic checks (Section 5.3), or drift monitoring (Section 1's drift box) — these are cheap to add before launch and expensive to retrofit after the first incident they would have caught.
- **Treating the load test as a one-time launch artifact.** The saturation curve from Section 3.5 is only valid for the model, hardware, and prompt distribution it was measured against; a model swap, a prompt-length shift from a new product feature, or a hardware change (even a "just as fast" GPU generation swap) invalidates it silently until the next incident reveals the assumption was stale.

## Closing: How to Use This Chapter

This capstone is meant to be read twice, in two different modes.

The first read is linear, before you build anything — Sections 1-2 to internalize the shape of the whole system and the decision framework, Section 3 to see a complete build end to end so the individual chapters' techniques have somewhere concrete to land, Sections 4-6 to understand what it costs and how it actually breaks once real traffic and real incidents show up, Section 7 to pressure-test your own understanding against the interview questions (a good proxy for "could I explain this to a new hire"), and Section 6's checklist immediately before any real launch.

The second read is as a reference, after you're operating something like this for real — Section 5 and Appendix G's failure modes and anti-patterns are worth rereading after your first real incident, not just before launch, because most of them are far more legible in hindsight than in a pre-launch review; Appendix F's runbook pattern is worth adapting for every SLO-protecting alert you add, not just the one shown; and Appendix B's chapter cross-reference map is the fastest way back into the numbered chapters when a specific piece (autoscaling tuning, drift statistics, Triton backend configuration) needs to go deeper than this chapter goes.

The single idea worth carrying forward past every specific YAML snippet and PromQL query in this chapter: a production LLM serving platform is not eleven separate problems that happen to share a GPU. It's one system, and the seams between the eleven pieces — where a canary controller and an autoscaler both think they own the same ReplicaSet, where a rollback reverts weights but not the prompt template that shipped alongside them, where a per-component dashboard is green while the actual user experience isn't — are where the interesting failures live. Chapters 01 through 11 teach you to build each piece correctly. This chapter is the argument for why that isn't the same thing as building the system correctly, and a worked example of closing that gap.

## Appendix H: A Concrete Game-Day Test Plan

Section 6's checklist and Section 7.5's twelve-month plan both mention "run a game-day" without specifying what to actually test. Here is a concrete plan built directly from the failure modes in Section 5 and Section 8.2 — run each of these in staging first, then, once trusted, in production during a low-traffic window with on-call actively watching.

| # | Test | What it validates | Expected result |
|---|---|---|---|
| 1 | Kill a model-server pod mid-request (`kubectl delete pod <pod> --grace-period=0`) | Readiness probe correctly removes the pod from the Service before it's fully gone; in-flight requests to *other* replicas are unaffected | Client sees at most one failed/retried request; no fleet-wide latency spike |
| 2 | Trigger a canary rollout, then manually abort mid-step (`kubectl argo rollouts undo`) | Rollback reverts traffic weight to 0% canary immediately, and — per Section 5.2 — reverts prompt template and engine config alongside the weights, not just the container image | Traffic fully back on stable within one reconcile cycle; a diff of the running config against the pre-rollout state shows zero drift in prompt/sampling config |
| 3 | Cordon and drain a GPU node hosting a live replica | PodDisruptionBudget (Section 3.4) prevents more replicas from draining simultaneously than `minAvailable` allows | Drain blocks or slows appropriately; no SLO breach during the drain |
| 4 | Simulate a node-provisioning delay (scale a test workload to fill the GPU node pool, then trigger a KEDA scale-up) | Whether the buffer/pre-warmed-capacity mitigation from Section 8.1 actually closes the node-provisioning gap | New pods reach `Running` within the SLO's error budget, not stuck `Pending` for the multi-minute node-provisioning window |
| 5 | Manually raise `maxReplicaCount`, then check one week later | Whether the follow-up-ticket process from Section 8.2 actually catches and reviews incident-time config changes, rather than letting them become permanent by default | A ticket exists, was reviewed, and the value was either intentionally kept or reverted — not simply forgotten |
| 6 | Force `QueueDepthRising` and `TTFTSLOBreach` to fire (via a synthetic load spike in staging) and confirm the on-call receives both, in the right order, with the runbook link intact | The two-tier alerting pattern (Section 3.7) and the runbook (Appendix F) actually work end to end, not just in the YAML | Warning fires first with lead time; page fires second if the situation doesn't resolve; the runbook link resolves to the current, correct document |
| 7 | Feed a batch of known-bad outputs into the drift/quality gauge (Appendix C) and confirm an in-flight canary halts | The drift/quality signal is a real gate, not a metric nobody's canary configuration actually reads | Canary's `AnalysisTemplate` fails the `golden-set-quality` check and halts/rolls back automatically |

Running this list once doesn't make it done — the honest cadence is re-running it whenever a component in the chain changes (a new autoscaler version, a new canary controller version, a change to the PDB or probe configuration), since each of these tests is validating an *interaction*, and interactions are exactly what silently break when one side of them changes without the other side being retested.

## Appendix I: Managed API Options, for Comparison

Section 2.1's decision tree starts with "hosted/managed API only," which shortcuts most of this chapter's serving stack. For context, the real options as of 2026 in that category — useful when the honest answer to "should we self-host at all" is still open:

| Provider | Model access pattern | Where it fits |
|---|---|---|
| OpenAI API | Hosted proprietary and some open-weight models via API | No self-hosting; you still need this chapter's Ch. 07/08/10 concerns (canary across model/provider versions, monitoring, drift) applied to an external dependency instead of your own cluster |
| Anthropic API | Hosted proprietary models via API | Same shape as above |
| AWS Bedrock | Hosted access to multiple model providers' models through one AWS-native API, plus (via Bedrock or SageMaker) the option to self-host open-weight models on managed endpoints | A middle ground — less operational burden than raw EKS + vLLM, more control than a single-vendor API |
| Google Vertex AI (Model Garden / endpoints) | Similar middle-ground shape to Bedrock, on GCP | Same tradeoff as Bedrock, GCP-native |
| Azure OpenAI Service | Hosted OpenAI models via Azure-native API/compliance boundary | Chosen primarily for enterprise procurement/compliance reasons rather than technical ones |
| Modal / Baseten / Replicate-style GPU-as-a-service | You bring the weights and a serving config; the platform operates the Kubernetes-equivalent layer (Sections 2-3 of this chapter) for you | The right choice when Section 2.2's Kubernetes-vs-simpler answer is "we need the scaling/reliability properties but can't staff operating them ourselves" |

The decision between these and the self-hosted stack this chapter builds is rarely purely technical — it's a build-vs-buy tradeoff between GPU-hour margin (self-hosting is cheaper per token at meaningful scale, per the cost model in Section 4) and operational headcount (every box in Section 1's reference architecture that you self-host is a box your team is now on-call for). A useful gut check: if you don't yet have someone who can execute Appendix H's game-day plan competently, that's a signal you may not be ready to self-host at production scale yet, regardless of what the cost model says.

## Appendix J: A Note on How the Numbers in This Chapter Age

Every concrete number in this chapter — the vLLM version pinned in the Dockerfile (Section 3.3), the GPU pricing used in the cost model (Section 4), the illustrative load-test curve (Section 3.5), the quantization comparison ranges (Appendix D) — is a snapshot, not a constant. This is worth stating explicitly rather than leaving implicit, because treating a snapshot as a constant is itself a small version of the same mistake Section 5 spends so much time on: trusting a number without checking whether the thing it was measured against has changed underneath it.

Concretely, before reusing this chapter's specifics in a real build:

- **Re-check the vLLM (or Triton, or Argo Rollouts, or KEDA) version** against the project's own release notes — this chapter cites the v0.20.x vLLM line and specific 2025-2026 feature landings (FP8 KV-cache work, `gpt-oss` support) as current at time of writing; serving engines in this space ship new releases frequently enough that a pinned version six months old is worth deliberately revisiting, not just inheriting.
- **Re-run the load test (Section 3.5) against your actual model and hardware.** The illustrative saturation curve in this chapter is real in shape but specific to nothing you're deploying — it exists to show *how to read* a saturation curve, not to hand you one.
- **Re-quote GPU pricing (Section 4, Appendix I)** — cloud GPU pricing has moved substantially year over year through the mid-2020s as supply has shifted, and the ranges cited here (with sources linked in Section 9) should be treated as "check whether this is still roughly right," not "use this number in a board deck."
- **Re-validate quantization quality deltas (Section 4.3, Appendix D) per model.** Quantization tooling and technique quality has been improving quickly; a quality gap that was noticeable eighteen months ago on one model family may be smaller (or larger, for a different architecture) on whatever you're actually deploying — the golden-set validation step in Section 6 exists precisely so you never have to trust a general claim about quantization quality instead of measuring your own.

None of this diminishes the chapter's core teaching content — the reference architecture's shape (Section 1), the decision framework's structure (Section 2), the order operations happen in during a real build (Section 3's walkthrough sequence), and the failure modes living in the seams between components (Section 5) all age far more slowly than any specific price or version number, because they're about how the pieces of the system relate to each other, not about which specific tool or GPU generation currently fills a given box. That distinction — what ages fast versus what doesn't — is itself worth teaching to anyone using this chapter as a reference months or years after it was written.

## Appendix K: A Postmortem, Written the Way Section 5's Incidents Actually Read

Section 5's failure modes are described analytically, by design, so they generalize. Here is one of them — 5.1, the canary-vs-autoscaler race — written the way an actual postmortem document would read, because the difference in tone matters: postmortems are specific, timestamped, and blameless about people while being precise about systems.

```markdown
## Incident: gpt-oss-20b-v2 canary — stable version resurfaced after full promotion

**Severity:** SEV-2. **Duration:** 23 minutes of intermittent stale-version
responses after a canary promotion showed "complete" in the Argo Rollouts UI.
**User impact:** ~4% of requests during the window received v1 (pre-update)
model behavior after v2 had already been communicated as fully live.

**Timeline:**
- 14:02 — Canary rollout for gpt-oss-20b-v2 begins per the staged-weight
  process in Section 3.8. All analysis gates pass at each step.
- 14:41 — Rollout reaches setWeight: 100. Argo Rollouts UI shows "Healthy."
  On-call marks the rollout complete in the team channel.
- 14:44 — KEDA's ScaledObject, which was watching queue depth on the
  underlying stable ReplicaSet directly (not the Rollout resource, per the
  misconfiguration described in Section 5.1), observes falling queue depth
  on stable as traffic had been shifting to canary throughout the rollout,
  and — reacting correctly to that signal in isolation — scales the stable
  ReplicaSet back up from 1 pod to 4, interpreting the recent dip as
  transient rather than as the deliberate result of the in-progress promotion.
- 14:47-15:07 — The Service's endpoint list includes both the (should be
  fully retired) stable pods and the new v2 pods. A subset of requests land
  on stable pods that KEDA re-created, producing v1 behavior for those
  requests.
- 15:07 — On-call notices version-label mismatch in the "requests by
  version" Grafana panel (Section 8.4) during a routine post-rollout check,
  identifies the stable ReplicaSet's unexpectedly nonzero pod count, and
  manually scales it to zero.
- 15:10 — Traffic confirmed 100% on v2. Incident closed.

**Root cause:** Two independent control loops — the Argo Rollouts canary
controller and the KEDA autoscaler — each held an opinion about the stable
ReplicaSet's replica count, on independent reconcile timers, with no
coordination between them. Neither was misconfigured relative to its own
scope; the interaction between the two scopes was the defect.

**Contributing factor:** The post-rollout check that caught this was manual
and happened to run within the incident window; there was no automated
alert for "nonzero replica count on a ReplicaSet the Rollout controller
should have scaled to zero."

**Fix (structural, not just monitoring):** Repointed the KEDA ScaledObject
at the Rollout resource rather than the underlying ReplicaSets directly
(Section 5.1's fix), verified via the game-day test in Appendix H, item 2,
run in staging through three full canary cycles with no recurrence. Added
an alert on nonzero pod count for any ReplicaSet a completed Rollout should
have zeroed, as a defense-in-depth backstop — but the primary fix is the
single-owner control-loop change, not the new alert.

**What this incident does not change:** The canary analysis gates
themselves (latency, error rate, quality) worked exactly as designed at
every step — this was never a "the canary should have caught something"
incident. It was a "two correct systems, incorrectly composed" incident,
which is precisely the category Section 5 of this chapter exists to name.
```

This is the format worth adopting for real incidents against this platform: specific and timestamped in the timeline, but the root-cause and fix sections stay at the level of "which control loop owned what," which is exactly the level Section 5's abstractions operate at — so that a postmortem doesn't just document one incident, it feeds back into the general failure-mode catalog the next engineer reads before their own launch.
