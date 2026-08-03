# NVIDIA Triton Inference Server: Production Multi-Model Serving

## Why this matters

Most inference tutorials show you how to serve *one* model with *one* framework. Production is rarely that tidy. You have a PyTorch embedding model, an ONNX classifier, a TensorRT vision model, and — increasingly — a large language model, and they all need to sit behind stable HTTP/gRPC endpoints, share expensive GPUs, batch requests to keep those GPUs busy, expose Prometheus metrics, and version cleanly.

NVIDIA Triton Inference Server is the piece that does all of that in one process. It is a **serving runtime**, not a model format: you point it at a directory of models, each with a small config file, and it loads them across whatever backends they need, batches incoming requests, runs multiple copies concurrently, and serves them on standardized endpoints. For LLMs specifically, Triton pairs with the **TensorRT-LLM backend** to deliver in-flight (continuous) batching and paged KV cache — the same class of technique that makes vLLM fast — with NVIDIA's most aggressive kernel optimizations underneath.

If you have exactly one LLM and nothing else, vLLM or TGI is often simpler. If you have a *fleet* of heterogeneous models, or you want NVIDIA's fastest LLM path with unified ops tooling, Triton is the standard answer. This chapter explains what it is, how the model repository and `config.pbtxt` work, the backend landscape, the two flavors of batching, and how it stacks up against vLLM and TGI.

---

## Core intuition

Hold one sentence in your head:

> **Triton is a serving runtime that hosts many models across many backends, with request batching and per-model concurrency built in.**

Everything else is detail hanging off that sentence:

- **Many models** — a *model repository* (a directory) holds every model. Add a folder, Triton serves it. Triton can hot-load, hot-unload, and version them.
- **Many backends** — each model declares a `backend` (or `platform`). TensorRT-LLM, vLLM, Python, ONNX Runtime, PyTorch (LibTorch), TensorRT, and more all run inside the same server process behind the same API.
- **Batching built in** — Triton groups small requests into larger batches to feed the GPU efficiently. For non-LLM models this is *dynamic batching*; for LLMs it is *in-flight batching*.
- **Concurrency built in** — *instance groups* let you run N copies of a model on one or more GPUs so requests overlap instead of queueing.

The payoff of the runtime abstraction: your infra team learns *one* server, *one* metrics format, *one* deployment story — and every model, whatever framework trained it, fits into it.

---

## Architecture and the model repository

### The big picture

A single `tritonserver` process contains:

1. **Frontends** — HTTP/REST on port **8000**, gRPC on port **8001**, and a Prometheus metrics endpoint on port **8002**.
2. **The core** — request routing, the scheduler (which does batching), model management (load/unload/versioning), and the shared-memory / pinned-memory machinery for zero-copy tensor passing.
3. **Backends** — shared libraries that actually execute a model. Each backend adapts one framework to Triton's C API. Multiple backends coexist in one server.

Requests arrive at a frontend, get routed to the named model, land in that model's scheduler queue, are (optionally) batched, dispatched to a model *instance* (a loaded copy on a specific device), and the response flows back out.

### The model repository

Triton is started with one or more repositories:

```
tritonserver --model-repository=/models
```

The layout is strict and load-bearing — Triton discovers models by walking this tree:

```
/models/
├── text_classifier/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.onnx
│   └── 2/
│       └── model.onnx
├── image_embedder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.pt
└── llama3_trtllm/
    ├── config.pbtxt
    └── 1/
        └── ...engine files...
```

Rules that trip people up:

- **The top-level directory name is the model name** clients use in requests (`text_classifier`, not the file inside).
- **Version subdirectories are integers** (`1/`, `2/`). Non-integer or `0` directories are ignored. By default Triton serves the *highest* numbered version; a `version_policy` in the config changes that (latest N, all, or specific).
- **The model file name is fixed per backend** — `model.onnx` for ONNX Runtime, `model.pt` for PyTorch/LibTorch, `model.plan` for TensorRT, `model.py` for the Python backend, etc.
- Repositories can live on local disk, S3, GCS, or Azure Blob (`--model-repository=s3://bucket/models`).

### `config.pbtxt` — the model configuration

Every model gets a `config.pbtxt` (protobuf text format) that declares its backend, tensor shapes, batching, and concurrency. For many framework backends Triton can **auto-generate** the config (`--strict-model-config=false`), but in production you write it explicitly so nothing is a surprise.

A minimal ONNX classifier config:

```pbtxt
name: "text_classifier"
backend: "onnxruntime"
max_batch_size: 32

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ 128 ]
  }
]
output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ 5 ]
  }
]
```

Key fields:

| Field | Meaning |
|-------|---------|
| `name` | Must match the directory name (optional if it does). |
| `backend` / `platform` | Which backend executes the model (`onnxruntime`, `python`, `pytorch`/`platform: "pytorch_libtorch"`, `tensorrt`/`platform: "tensorrt_plan"`, `vllm`, `tensorrtllm`). |
| `max_batch_size` | Largest batch Triton will assemble. `0` means the model does not support Triton's batching (first dim is not a batch dim). |
| `input` / `output` | Tensor `name`, `data_type` (`TYPE_FP32`, `TYPE_INT64`, `TYPE_STRING`, `TYPE_BF16`, …), and `dims`. When `max_batch_size > 0`, the batch dimension is *implicit* — you list only the per-sample shape. Use `-1` for dynamic dims. |
| `instance_group` | How many copies, on what devices (see below). |
| `dynamic_batching` | Enables server-side batching (see below). |
| `version_policy` | `{ latest: { num_versions: 1 } }`, `{ all: {} }`, or `{ specific: { versions: [1,3] } }`. |

---

## Backends: pick the right engine

A **backend** is the plug-in that runs a model. Choosing the wrong one for LLMs is the single most common Triton mistake, so internalize this table:

| Backend | `backend`/`platform` | Best for | Batching model | Notes |
|---------|----------------------|----------|----------------|-------|
| **TensorRT-LLM** | `tensorrtllm` | Production LLM inference on NVIDIA GPUs | **In-flight (continuous)** | Fastest LLM path; requires compiling the model into a TensorRT-LLM engine. Paged KV cache, tensor/pipeline parallel. |
| **vLLM** | `vllm` | LLMs you want to run with minimal conversion | **Continuous (vLLM's own)** | Wraps vLLM's `AsyncLLMEngine`; PagedAttention; no engine build step. Easiest LLM onramp inside Triton. |
| **Python** | `python` | Pre/post-processing, tokenization, glue, custom logic, BLS | Dynamic (if you enable it) | You write `model.py` with `TritonPythonModel`. The universal escape hatch; also hosts Business Logic Scripting. |
| **ONNX Runtime** | `onnxruntime` | Classifiers, embedders, small/medium models exported to ONNX | Dynamic | Portable, CPU or GPU, good default for non-LLM models. |
| **PyTorch (LibTorch)** | `pytorch` / `pytorch_libtorch` | TorchScript / traced models | Dynamic | Serve `model.pt` directly without re-exporting. |
| **TensorRT** | `tensorrt` / `tensorrt_plan` | Vision/CNN/transformer engines compiled to a `.plan` | Dynamic | Extremely fast for non-generative models; needs a TensorRT build. |

**The one rule to remember:** do **not** try to serve an LLM's token-by-token generation loop through a plain ONNX/PyTorch backend with `dynamic_batching`. Autoregressive decoding has variable-length outputs and per-request state; naive dynamic batching stalls the whole batch on the slowest sequence. LLMs need **in-flight batching**, which means the **TensorRT-LLM** or **vLLM** backend.

---

## Dynamic batching vs in-flight batching

Batching is how you keep a GPU — which loves large parallel matmuls — busy when requests trickle in one at a time. Triton has two mechanisms, and the distinction is the heart of LLM serving.

### Dynamic batching (for fixed-shape models)

For a classifier or embedder, every request does the *same* amount of work and produces a fixed-shape output. Triton's dynamic batcher waits a tiny, bounded window, collects whatever requests arrived, forms one batch, runs it once, and splits the results back out.

```pbtxt
dynamic_batching {
  preferred_batch_size: [ 8, 16 ]
  max_queue_delay_microseconds: 1000
}
```

- `preferred_batch_size` — batch sizes the scheduler prefers to form (often a power of two the engine is tuned for).
- `max_queue_delay_microseconds` — the *most* time a request will wait to be batched. This is the core latency/throughput knob: bigger delay → fuller batches → more throughput but more tail latency. `1000` µs = 1 ms.
- Optional `preserve_ordering`, and `priority_levels` for QoS.

The mental model: **one batch in, one batch out, everyone waits for the slowest member.** That is fine when all members do equal work. It is a disaster for generation, where one request might emit 5 tokens and another 500.

### In-flight (continuous) batching (for LLMs)

LLM decoding is iterative: each forward pass produces one token per active sequence, then loops. **In-flight batching** (a.k.a. continuous or iteration-level batching) exploits this. Instead of freezing a batch for its whole lifetime, the scheduler operates *per decoding iteration*:

- Finished sequences **leave** the batch immediately and return to the client.
- Newly arrived requests **join** the running batch at the next iteration, filling the freed slots.
- The batch composition changes every step — the GPU is never idle waiting for the slowest sequence.

Paired with a **paged KV cache** (the attention key/value cache stored in fixed-size blocks, like OS virtual memory pages), this eliminates the memory fragmentation and rigid padding that kill naive LLM batching. This is exactly the vLLM PagedAttention idea; the TensorRT-LLM backend implements the same class of technique with NVIDIA-optimized kernels.

In the TensorRT-LLM backend you turn it on in the `tensorrt_llm` model's config:

```pbtxt
parameters: { key: "gpt_model_type"     value: { string_value: "inflight_fused_batching" } }
parameters: { key: "batching_strategy"  value: { string_value: "inflight_fused_batching" } }
parameters: { key: "kv_cache_free_gpu_mem_fraction" value: { string_value: "0.9" } }
```

The engine itself must be built with paged KV cache (`--kv_cache_type paged` at engine-build time). Get this wrong — build a static-batch engine, or leave `batching_strategy` as `v1` — and you have thrown away the entire point of using TensorRT-LLM.

---

## Instance groups and concurrent execution

Batching fills a single GPU pass. **Instance groups** decide how many independent passes can be *in flight at once*, and where.

```pbtxt
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

- `count` — number of loaded copies (instances) of the model.
- `kind` — `KIND_GPU`, `KIND_CPU`, or `KIND_MODEL` (let the backend decide device placement — used by the vLLM and TensorRT-LLM backends).
- `gpus` — which physical GPUs to place instances on.

Two instances on one GPU means Triton can execute two requests concurrently on that GPU (overlapping compute and memory transfer via CUDA streams), improving utilization when a single request under-fills the device. Instances across multiple GPUs give you data-parallel scale-out of the *same* model.

Combining knobs:

- **Dynamic batching + multiple instances** — each instance has its own batch scheduler; requests spread across instances, each forms batches. Great for high-throughput fixed-shape models.
- **For LLMs**, you usually do **not** stack many small instances. One instance owns the GPU (or several GPUs via `tensor_parallel_size`/`world_size`) and in-flight batching handles concurrency internally. Multiple TensorRT-LLM instances only make sense across separate GPU sets.

```pbtxt
# Spread three instances across two GPUs
instance_group [
  { count: 1  kind: KIND_GPU  gpus: [ 0 ] },
  { count: 2  kind: KIND_GPU  gpus: [ 1 ] }
]
```

---

## Ensembles and Business Logic Scripting (BLS)

Real inference is a *pipeline*: tokenize → run model → detokenize; or embed → search → rerank. Triton gives you two ways to compose models server-side so the client makes **one** call.

### Ensembles (declarative DAG)

An **ensemble** is a model with `platform: "ensemble"` and no code — just a config describing how tensors flow between other models. Triton executes the graph internally, passing tensors in GPU/shared memory without extra network hops.

```pbtxt
name: "llm_pipeline"
platform: "ensemble"
max_batch_size: 8
input  [ { name: "text_input"  data_type: TYPE_STRING  dims: [ 1 ] } ]
output [ { name: "text_output" data_type: TYPE_STRING  dims: [ 1 ] } ]
ensemble_scheduling {
  step [
    {
      model_name: "preprocessing"
      model_version: -1
      input_map  { key: "QUERY"        value: "text_input" }
      output_map { key: "input_ids"    value: "ids" }
    },
    {
      model_name: "tensorrt_llm"
      model_version: -1
      input_map  { key: "input_ids"    value: "ids" }
      output_map { key: "output_ids"   value: "gen_ids" }
    },
    {
      model_name: "postprocessing"
      model_version: -1
      input_map  { key: "output_ids"   value: "gen_ids" }
      output_map { key: "OUTPUT"       value: "text_output" }
    }
  ]
}
```

This is exactly the canonical TensorRT-LLM layout: a `preprocessing` Python model (string → `input_ids`), the `tensorrt_llm` engine model, and a `postprocessing` Python model (`output_ids` → string), stitched by an `ensemble`.

**Ensembles are static graphs.** They cannot express loops or data-dependent branching.

### Business Logic Scripting (BLS)

When you need conditionals, loops, or calling model B based on model A's *output*, use **BLS**: a Python-backend model that issues inference requests to *other* Triton models from inside its `execute()`:

```python
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            prompt = pb_utils.get_input_tensor_by_name(request, "text_input")
            # Call the tokenizer model
            tok = pb_utils.InferenceRequest(
                model_name="preprocessing",
                requested_output_names=["input_ids"],
                inputs=[prompt],
            )
            ids = tok.exec().output_tensors()[0]
            # ... branch on content, loop, call the LLM, etc.
            responses.append(pb_utils.InferenceResponse(output_tensors=[...]))
        return responses
```

For LLMs, the TensorRT-LLM backend ships a `tensorrt_llm_bls` model as an alternative to the ensemble — same pipeline, but expressed in Python so you can add guardrails, retries, or multi-model routing. Rule of thumb: **ensemble for a fixed DAG, BLS when logic depends on runtime data.**

---

## HTTP/gRPC endpoints and metrics

### Endpoints (KServe v2 / "predict" protocol)

- **HTTP/REST** on `:8000`, **gRPC** on `:8001`.
- Inference: `POST /v2/models/{model}/infer` (and `/versions/{v}/infer`).
- Health/readiness: `GET /v2/health/ready`, `/v2/health/live`.
- Metadata: `GET /v2/models/{model}`, and repository/config introspection.
- LLM convenience: the generate endpoint `POST /v2/models/{model}/generate` (and `/generate_stream` for token streaming with decoupled models).

A generic infer call:

```bash
curl -s localhost:8000/v2/models/text_classifier/infer -d '{
  "inputs": [
    { "name": "input_ids", "shape": [1, 128], "datatype": "INT64",
      "data": [ 101, 2054, 2003, ... ] }
  ]
}'
```

An LLM generate call (vLLM or TensorRT-LLM ensemble):

```bash
curl -s -X POST localhost:8000/v2/models/vllm_model/generate -d '{
  "text_input": "What is Triton Inference Server?",
  "parameters": { "stream": false, "temperature": 0, "max_tokens": 128 }
}'
```

Streaming token-by-token needs a **decoupled** model — one that returns many responses per request — declared with `model_transaction_policy { decoupled: true }`, and is consumed over gRPC streaming or `/generate_stream`.

### Metrics

Triton exposes Prometheus metrics at `:8002/metrics` (`curl localhost:8002/metrics`). Core series:

| Metric | Meaning |
|--------|---------|
| `nv_inference_request_success` / `_failure` | Request counts. |
| `nv_inference_count` | Inferences performed (includes batching effects). |
| `nv_inference_queue_duration_us` | Time requests spend queued — your batching-pressure signal. |
| `nv_inference_compute_infer_duration_us` | Actual model compute time. |
| `nv_inference_compute_input_duration_us` / `_output_` | Tensor marshalling time. |
| `nv_gpu_utilization`, `nv_gpu_memory_used_bytes` | Per-GPU device metrics. |

The TensorRT-LLM backend adds custom metrics for **KV cache** block usage and **in-flight batching** (active/scheduled request counts) via Triton's custom-metrics API. Watch queue duration and KV-cache utilization together: rising queue time with KV cache near 100% means you are memory-bound and should raise `kv_cache_free_gpu_mem_fraction`, shorten max sequence length, or scale out.

For LLM-aware benchmarking, use **GenAI-Perf** (part of Perf Analyzer), which reports LLM-specific numbers: **time to first token (TTFT)**, **inter-token latency (ITL)**, **output tokens/sec**, and request throughput — the metrics that actually matter for chat workloads.

---

## Fully worked example: an ONNX classifier with batching + concurrency

This is a complete, runnable non-LLM deployment. (The LLM path via TensorRT-LLM is shown right after — it uses the ensemble above.)

### Repository layout

```
/models/
└── sentiment/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

### `config.pbtxt`

```pbtxt
name: "sentiment"
backend: "onnxruntime"
max_batch_size: 32

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ 128 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ 128 ]
  }
]
output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 2000
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

version_policy { latest { num_versions: 1 } }
```

This serves the ONNX model with up-to-32 dynamic batches (waiting at most 2 ms to fill one) and two concurrent GPU instances.

### Launch the server (Docker)

```bash
docker run --gpus all --rm -it \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /models:/models \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models --strict-model-config=true
```

You should see `sentiment` reported `READY` in the startup table, and `GET localhost:8000/v2/health/ready` returns 200.

### Call it

```bash
curl -s localhost:8000/v2/models/sentiment/infer -d '{
  "inputs": [
    { "name": "input_ids",      "shape": [1,128], "datatype": "INT64", "data": [101, 2023, 2003, 6659, 999, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      "attention_mask": [] }
  ]
}'
```

### The LLM path (TensorRT-LLM), in brief

For an LLM you do **not** hand-write a decode loop. You:

1. **Convert** the HF checkpoint to TensorRT-LLM format and **build an engine** with paged KV cache and in-flight batching enabled (`trtllm-build ... --kv_cache_type paged`).
2. Assemble the **ensemble** repository: `preprocessing/` (tokenizer, Python backend), `tensorrt_llm/` (the engine + `config.pbtxt` with `batching_strategy: inflight_fused_batching`, `engine_dir`, `kv_cache_free_gpu_mem_fraction`, `decoupled_mode`), `postprocessing/` (detokenizer), and `ensemble/`. NVIDIA's `fill_template.py` populates these configs from the engine.
3. **Launch** across GPUs:

```bash
python3 /app/scripts/launch_triton_server.py \
  --world_size=2 --model_repo=/models/trtllm_repo
```

4. Call `POST /v2/models/ensemble/generate` (or stream via `/generate_stream` with `decoupled_mode: true`).

The vLLM backend is the lower-effort alternative: skip steps 1–2, drop a `model.json` (`{"model": "...", "tensor_parallel_size": 1, "gpu_memory_utilization": 0.9}`) and a `config.pbtxt` with `backend: "vllm"` and `model_transaction_policy { decoupled: true }`, and launch the `-vllm-python-py3` image.

---

## Triton + TensorRT-LLM vs vLLM vs TGI

| Dimension | **Triton + TensorRT-LLM** | **vLLM** (standalone) | **TGI** (Text Generation Inference) |
|-----------|---------------------------|-----------------------|-------------------------------------|
| Primary goal | Multi-model serving fleet + fastest NVIDIA LLM path | Fast, simple LLM serving | HuggingFace-native LLM serving |
| Continuous batching | Yes (in-flight, fused) | Yes (native) | Yes (native) |
| KV cache | Paged | PagedAttention (originator) | Paged |
| Setup effort | **High** — engine build + ensemble wiring | **Low** — `pip install`, point at HF model | Low–medium — Docker + model id |
| Peak throughput on NVIDIA | **Highest** (tuned TensorRT kernels) | Very high | High |
| Hardware | NVIDIA only | NVIDIA-first (some others) | NVIDIA-first (some others) |
| Non-LLM models | **Yes** — same server hosts ONNX/PyTorch/TensorRT | No | No |
| Ops surface | One server, unified metrics, ensembles/BLS | Simple, LLM-scoped | Simple, LLM-scoped |
| Streaming | Decoupled + `/generate_stream` | OpenAI-compatible SSE | SSE, OpenAI-compatible |
| Best when | You run many models and/or want max NVIDIA LLM perf with unified ops | You want the least-effort fast LLM server | You are all-in on the HF stack |

Honest summary: **for a single LLM, vLLM or TGI is faster to stand up and gets you 90% of the throughput with 10% of the effort.** Triton + TensorRT-LLM wins when you need a heterogeneous model fleet under one runtime, or when you have squeezed everything else and need the last increment of GPU efficiency and are willing to pay the engine-build tax. Note also that Triton can host the **vLLM backend**, giving you vLLM's ergonomics *inside* Triton's ops framework — a common middle ground.

---

## Failure modes and pitfalls

- **Wrong backend for LLMs.** Serving generation through ONNX/PyTorch + `dynamic_batching` produces terrible throughput and head-of-line blocking. LLMs require the TensorRT-LLM or vLLM backend with in-flight batching. This is the number-one mistake.
- **Static-batch TensorRT-LLM engine.** Building the engine without paged KV cache, or leaving `batching_strategy`/`gpt_model_type` at `v1`, silently disables continuous batching. You paid the conversion cost and got none of the benefit.
- **Misconfigured dynamic batching.** `max_queue_delay_microseconds` too high → latency spikes; too low → tiny batches and idle GPU. `preferred_batch_size` mismatched to what the engine was tuned for → padding waste. Tune against GenAI-Perf, not by guessing.
- **`max_batch_size` vs shape confusion.** With `max_batch_size > 0` the batch dim is *implicit* — listing it explicitly in `dims` double-counts it and breaks shape checks. Set `max_batch_size: 0` only for models whose first dim is not a batch dimension.
- **KV-cache OOM.** `kv_cache_free_gpu_mem_fraction` too aggressive (or too many concurrent LLM instances) OOMs under load; too conservative wastes capacity. Watch KV-cache utilization metrics.
- **Version / container mismatches.** The TensorRT-LLM *engine*, the *tensorrtllm_backend*, and the Triton container are a matched set. Building an engine with one TensorRT-LLM version and loading it in a mismatched Triton image fails to load or crashes. Pin versions together.
- **Model name / directory mismatch.** `name` in `config.pbtxt` disagreeing with the directory, or a non-integer version folder, makes the model silently not load. Read the startup READY/UNAVAILABLE table.
- **Conversion complexity underestimated.** The TensorRT-LLM path (convert → build → template the ensemble → launch across `world_size`) is genuinely involved and model-specific. Budget for it; do not promise a one-day LLM deploy on TensorRT-LLM.
- **Forgetting `--shm-size` / decoupled streaming.** Python-backend ensembles need adequate shared memory; streaming needs `decoupled: true` or you get one blob at the end instead of tokens.

---

## Production checklist — what an interviewer probes

1. **"You have five models in four frameworks sharing two GPUs. Design it."** — Expect: one Triton server, one model repository, per-model `backend` and `instance_group`, dynamic batching on the fixed-shape models, an LLM on the TensorRT-LLM/vLLM backend. Tests whether you understand Triton's core value proposition.
2. **"Dynamic vs in-flight batching — when each, and why?"** — Fixed-shape/equal-work → dynamic; autoregressive LLM → in-flight/continuous, because variable output length makes static batches stall on the slowest sequence. Bonus for paged KV cache.
3. **"Walk me through the TensorRT-LLM deployment."** — Convert + `trtllm-build` with paged KV cache → ensemble (pre/trtllm/post) → `fill_template.py` → `launch_triton_server.py --world_size` → `/generate`. Honesty about the conversion complexity scores points.
4. **"How do you tune the latency/throughput tradeoff?"** — `max_queue_delay_microseconds` and `preferred_batch_size` for dynamic; instance count for concurrency; `kv_cache_free_gpu_mem_fraction` and max sequence length for LLMs — validated with GenAI-Perf (TTFT, ITL, tokens/s).
5. **"What do you monitor, and what does a bad number mean?"** — `nv_inference_queue_duration_us` (batching pressure), compute duration, `nv_gpu_utilization`, KV-cache utilization. Rising queue + full KV cache = memory-bound; scale out or trim context.
6. **"Ensemble vs BLS?"** — Ensemble for a static DAG (tokenize→infer→detokenize); BLS when the pipeline branches or loops on runtime data.
7. **"When would you NOT use Triton?"** — A single LLM where vLLM/TGI is dramatically simpler; no heterogeneous fleet; team without NVIDIA-stack depth. Knowing when the simpler tool wins signals seniority.
8. **"How do you roll out a new model version safely?"** — Version subdirectories + `version_policy`, load the new version alongside the old, shift traffic, and hot-unload — no server restart.

---

## Further reading

- Triton model configuration (config.pbtxt reference): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html
- Triton model repository layout: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md
- Dynamic batching & concurrent model execution (conceptual guide): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html
- TensorRT-LLM backend (in-flight batching, paged KV cache, ensemble): https://github.com/triton-inference-server/tensorrtllm_backend
- TensorRT-LLM backend docs: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html
- vLLM backend for Triton: https://github.com/triton-inference-server/vllm_backend
- Deploying a vLLM model in Triton (tutorial): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Quick_Deploy/vLLM/README.html
- Python backend (custom logic + BLS): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html
- Metrics reference: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md
- GenAI-Perf (LLM benchmarking): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
