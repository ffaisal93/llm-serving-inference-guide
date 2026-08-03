# NVIDIA Triton Inference Server: Production Multi-Model Serving

## Why this matters

Most inference tutorials show you how to serve *one* model with *one* framework. Production is rarely that tidy. You have a PyTorch embedding model, an ONNX classifier, a TensorRT vision model, and — increasingly — a large language model, and they all need to sit behind stable HTTP/gRPC endpoints, share expensive GPUs, batch requests to keep those GPUs busy, expose Prometheus metrics, and version cleanly.

NVIDIA Triton Inference Server is the piece that does all of that in one process. It is a **serving runtime**, not a model format: you point it at a directory of models, each with a small config file, and it loads them across whatever backends they need, batches incoming requests, runs multiple copies concurrently, and serves them on standardized endpoints. For LLMs specifically, Triton pairs with the **TensorRT-LLM backend** to deliver in-flight (continuous) batching and paged KV cache — the same class of technique that makes vLLM fast — with NVIDIA's most aggressive kernel optimizations underneath.

If you have exactly one LLM and nothing else, vLLM or TGI is often simpler. If you have a *fleet* of heterogeneous models, or you want NVIDIA's fastest LLM path with unified ops tooling, Triton is the standard answer. This chapter explains what it is, how the model repository and `config.pbtxt` work, the backend landscape, the two flavors of batching, how to build a real multi-model ensemble, how to benchmark it correctly, and how it stacks up against vLLM, SGLang, and TGI in 2026 — plus the production incidents that teach the lessons faster than any diagram.

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

Every model gets a `config.pbtxt` (protobuf text format) that declares its backend, tensor shapes, batching, and concurrency. For many framework backends Triton can **auto-generate** the config (`--strict-model-config=false`), but in production you write it explicitly so nothing is a surprise — see the "silent batching cap" war story below for what happens when you don't.

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
| **TensorRT-LLM** | `tensorrtllm` | Production LLM inference on NVIDIA GPUs | **In-flight (continuous)** | Fastest LLM path. As of TensorRT-LLM 1.x, the **PyTorch execution backend** (the "LLM API") is the default and can serve HF checkpoints directly with no separate engine-compile step; the older `trtllm-build` engine-compile workflow still exists but is the legacy path. Paged KV cache, tensor/pipeline/expert parallel. |
| **vLLM** | `vllm` | LLMs you want to run with minimal conversion | **Continuous (vLLM's own)** | Wraps vLLM's `AsyncLLMEngine`; PagedAttention; no engine build step. NVIDIA's own benchmarking puts the Triton vLLM backend within a couple of percent of standalone vLLM throughput/latency. |
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
- `max_queue_delay_microseconds` — the *most* time a request will wait to be batched. This is the core latency/throughput knob: bigger delay leads to fuller batches and more throughput but more tail latency. `1000` microseconds is 1 millisecond.
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

The engine itself must be built (or, on the newer PyTorch execution path, configured) with paged KV cache enabled. Get this wrong — build a static-batch engine, or leave `batching_strategy` as `v1` — and you have thrown away the entire point of using TensorRT-LLM.

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
name: "ensemble"
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

This is exactly the canonical TensorRT-LLM layout: a `preprocessing` Python model (string to `input_ids`), the `tensorrt_llm` engine model, and a `postprocessing` Python model (`output_ids` to string), stitched by an `ensemble`. Section (B) below walks through the full, runnable version of this pipeline with real `model.py` code.

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

### Request cancellation and priority

Two operational details that matter once real users are behind an LLM endpoint:

- **Cancellation.** If a client disconnects (closes the browser tab, hits a client-side timeout) mid-generation, a decoupled streaming request can be cancelled so the GPU stops spending cycles on a response nobody will read — gRPC's native call cancellation propagates into Triton's core and, for the TensorRT-LLM and vLLM backends, into the in-flight batching scheduler, freeing that sequence's KV-cache blocks immediately rather than waiting for it to run to completion. Skipping this is a common source of wasted GPU-seconds under bursty, abandon-prone traffic (chat UIs where users retype a prompt mid-stream).
- **Priority.** `dynamic_batching { priority_levels: N ... }` lets you declare multiple priority queues for fixed-shape models. For LLM in-flight batching, priority is typically handled one layer up — at the request-routing/gateway layer, or via backend-specific scheduling parameters — rather than through Triton's classic dynamic-batching priority mechanism, since the scheduling unit for an LLM is a decode iteration, not a whole batch.


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
| `nv_inference_first_response_histogram_ms` | Time-to-first-response histogram, exposed for coupled *and* decoupled (streaming) models in recent Triton releases — your TTFT signal straight from Prometheus. |

The TensorRT-LLM and vLLM backends add custom metrics for **KV cache** block usage and **in-flight batching** (active/scheduled request counts) via Triton's custom-metrics API. Watch queue duration and KV-cache utilization together: rising queue time with KV cache near 100% means you are memory-bound and should raise `kv_cache_free_gpu_mem_fraction`, shorten max sequence length, or scale out.

For LLM-aware benchmarking, use **GenAI-Perf** (part of Perf Analyzer), which reports LLM-specific numbers: **time to first token (TTFT)**, **inter-token latency (ITL)**, **output tokens/sec**, and request throughput — the metrics that actually matter for chat workloads. Section (B) below has a full walkthrough.

### Runtime model control — load/unload without a server restart

Triton's model management API lets you add, update, or remove models from a running server without a restart — essential for CI/CD pipelines that redeploy models frequently and for the version-rollout workflow described earlier in this chapter.

```bash
# Explicit model control mode must be enabled at startup:
# tritonserver --model-repository=/models --model-control-mode=explicit

# Load a model (or a new version of one) after adding files to the repository:
curl -X POST localhost:8000/v2/repository/models/sentiment/load

# Unload a model to free its GPU memory:
curl -X POST localhost:8000/v2/repository/models/sentiment/unload

# Ask what the server currently thinks is in the repository (useful after
# adding/removing a version directory out from under a running server):
curl -X POST localhost:8000/v2/repository/index
```

Three control modes exist: `none` (load everything at startup, no runtime changes), `poll` (periodically re-scan the repository directory and hot-reload changes — convenient, but easy to trigger an unintended reload by touching the wrong file), and `explicit` (nothing loads or unloads except via the API above — the recommended mode for production, since it makes every model change an auditable, deliberate action rather than a side effect of a filesystem write).

---

## Build it in practice

### Part 1 — an ONNX classifier with batching + concurrency

This is a complete, runnable non-LLM deployment.

#### Repository layout

```
/models/
└── sentiment/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

#### `config.pbtxt`

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

#### Launch the server (Docker)

```bash
docker run --gpus all --rm -it \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /models:/models \
  nvcr.io/nvidia/tritonserver:25.10-py3 \
  tritonserver --model-repository=/models --strict-model-config=true
```

You should see `sentiment` reported `READY` in the startup table, and `GET localhost:8000/v2/health/ready` returns 200.

#### Call it

```bash
curl -s localhost:8000/v2/models/sentiment/infer -d '{
  "inputs": [
    { "name": "input_ids",      "shape": [1,128], "datatype": "INT64", "data": [101, 2023, 2003, 6659, 999, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      "attention_mask": [] }
  ]
}'
```

### Part 2 — the full LLM ensemble: tokenizer to TensorRT-LLM to detokenizer

This is the piece most tutorials skip: the *entire* runnable pipeline, with real Python-backend code, not just the `ensemble_scheduling` skeleton. It mirrors the canonical layout used by NVIDIA's `tensorrtllm_backend` (now vendored inside the TensorRT-LLM repo under `triton_backend/`).

#### Repository layout

```
/models/
├── preprocessing/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
├── tensorrt_llm/
│   ├── config.pbtxt
│   └── 1/
│       └── (engine files or PyTorch-backend checkpoint dir)
├── postprocessing/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
└── ensemble/
    ├── config.pbtxt
    └── 1/                    # empty — ensembles have no model files
```

#### `preprocessing/config.pbtxt` — tokenizer as a Python backend model

```pbtxt
name: "preprocessing"
backend: "python"
max_batch_size: 8

input [
  { name: "QUERY"            data_type: TYPE_STRING dims: [ 1 ] },
  { name: "REQUEST_OUTPUT_LEN" data_type: TYPE_INT32 dims: [ 1 ] }
]
output [
  { name: "input_ids"      data_type: TYPE_INT32 dims: [ -1 ] },
  { name: "request_input_len" data_type: TYPE_INT32 dims: [ 1 ] }
]

parameters { key: "tokenizer_dir" value: { string_value: "/models/preprocessing/1/tokenizer" } }
parameters { key: "add_special_tokens" value: { string_value: "True" } }

instance_group [ { count: 1 kind: KIND_CPU } ]
```

`preprocessing/1/model.py`:

```python
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        tokenizer_dir = model_config["parameters"]["tokenizer_dir"]["string_value"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token

    def execute(self, requests):
        responses = []
        for request in requests:
            query = pb_utils.get_input_tensor_by_name(request, "QUERY")
            text = query.as_numpy()[0][0].decode("utf-8")

            ids = self.tokenizer.encode(text, add_special_tokens=True)
            input_ids = np.array([ids], dtype=np.int32)
            input_len = np.array([[len(ids)]], dtype=np.int32)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("input_ids", input_ids),
                        pb_utils.Tensor("request_input_len", input_len),
                    ]
                )
            )
        return responses
```

#### `tensorrt_llm/config.pbtxt` — the engine model

```pbtxt
name: "tensorrt_llm"
backend: "tensorrtllm"
max_batch_size: 64

input [
  { name: "input_ids"        data_type: TYPE_INT32 dims: [ -1 ] },
  { name: "request_input_len" data_type: TYPE_INT32 dims: [ 1 ] },
  { name: "request_output_len" data_type: TYPE_INT32 dims: [ 1 ] }
]
output [
  { name: "output_ids" data_type: TYPE_INT32 dims: [ -1, -1 ] }
]

model_transaction_policy { decoupled: true }

parameters: { key: "gpt_model_type"    value: { string_value: "inflight_fused_batching" } }
parameters: { key: "batching_strategy" value: { string_value: "inflight_fused_batching" } }
parameters: { key: "gpt_model_path"    value: { string_value: "/models/tensorrt_llm/1" } }
parameters: { key: "kv_cache_free_gpu_mem_fraction" value: { string_value: "0.9" } }
parameters: { key: "enable_chunked_context" value: { string_value: "True" } }

instance_group [ { count: 1 kind: KIND_MODEL } ]
```

`decoupled: true` is what allows this model to stream partial (per-token) responses back through the ensemble instead of buffering the whole generation.

#### `postprocessing/config.pbtxt` — detokenizer

```pbtxt
name: "postprocessing"
backend: "python"
max_batch_size: 8

input  [ { name: "output_ids" data_type: TYPE_INT32 dims: [ -1, -1 ] } ]
output [ { name: "OUTPUT"     data_type: TYPE_STRING dims: [ 1 ] } ]

parameters { key: "tokenizer_dir" value: { string_value: "/models/preprocessing/1/tokenizer" } }
instance_group [ { count: 1 kind: KIND_CPU } ]
```

`postprocessing/1/model.py`:

```python
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        tokenizer_dir = model_config["parameters"]["tokenizer_dir"]["string_value"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    def execute(self, requests):
        responses = []
        for request in requests:
            output_ids = pb_utils.get_input_tensor_by_name(request, "output_ids").as_numpy()
            text = self.tokenizer.decode(output_ids[0][0], skip_special_tokens=True)
            out = np.array([[text.encode("utf-8")]], dtype=object)
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("OUTPUT", out)])
            )
        return responses
```

#### `ensemble/config.pbtxt` — the glue

```pbtxt
name: "ensemble"
platform: "ensemble"
max_batch_size: 8

input [
  { name: "text_input"       data_type: TYPE_STRING dims: [ 1 ] },
  { name: "max_tokens"       data_type: TYPE_INT32  dims: [ 1 ] }
]
output [
  { name: "text_output" data_type: TYPE_STRING dims: [ 1 ] }
]

ensemble_scheduling {
  step [
    {
      model_name: "preprocessing"
      model_version: -1
      input_map  { key: "QUERY"                value: "text_input" }
      input_map  { key: "REQUEST_OUTPUT_LEN"    value: "max_tokens" }
      output_map { key: "input_ids"             value: "_input_ids" }
      output_map { key: "request_input_len"     value: "_request_input_len" }
    },
    {
      model_name: "tensorrt_llm"
      model_version: -1
      input_map  { key: "input_ids"            value: "_input_ids" }
      input_map  { key: "request_input_len"    value: "_request_input_len" }
      input_map  { key: "request_output_len"   value: "max_tokens" }
      output_map { key: "output_ids"           value: "_output_ids" }
    },
    {
      model_name: "postprocessing"
      model_version: -1
      input_map  { key: "output_ids"  value: "_output_ids" }
      output_map { key: "OUTPUT"      value: "text_output" }
    }
  ]
}
```

#### Launch and call the pipeline

```bash
docker run --gpus all --rm -it \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --shm-size=2G --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /models:/models \
  nvcr.io/nvidia/tritonserver:25.10-trtllm-python-py3 \
  tritonserver --model-repository=/models
```

```bash
curl -s -X POST localhost:8000/v2/models/ensemble/generate -d '{
  "text_input": "Explain in one sentence why paged KV cache matters.",
  "max_tokens": 64
}'
```

One client call, three internal model hops, zero network round trips between them — that is the entire value proposition of ensembles in one example.

### Part 2b — multi-GPU launch: tensor parallel and pipeline parallel

A single GPU cannot hold every model at every precision — a large dense model in FP16/BF16, or any model at long context with a large KV cache, needs to be split across GPUs. TensorRT-LLM (and the vLLM backend) support two orthogonal ways to split it:

- **Tensor parallelism (TP)** — shard each layer's weight matrices across GPUs, with an all-reduce after each sharded op. Reduces per-GPU memory and lets a bigger model fit, at the cost of inter-GPU communication on every layer — wants NVLink, not just PCIe, between the GPUs involved.
- **Pipeline parallelism (PP)** — assign different *layers* to different GPUs, streaming activations forward through the pipeline. Lower communication overhead per step than TP, but only helps throughput if you keep the pipeline full with enough concurrent requests, and adds bubble latency for the first request through an empty pipeline.

Real deployments often combine both (`tensor_parallel_size * pipeline_parallel_size = world_size`, the total GPU count for one model replica). This is set at build/config time and reflected in the launch command, which uses MPI to start one Triton/TensorRT-LLM process per GPU that all rendezvous into a single logical model:

```bash
# 2-way tensor parallel, 2-way pipeline parallel = 4 GPUs for one model replica
python3 /app/scripts/launch_triton_server.py \
  --world_size=4 \
  --tensorrt_llm_model_repository_path=/models \
  --tensorrt_llm_model_name=tensorrt_llm
```

Under the hood this is an `mpirun`-launched set of ranks, one per GPU, each running its shard of the model; Triton's frontend still presents a single model endpoint to clients — the parallelism is entirely invisible from the API. The config's `gpt_model_path`/checkpoint must have been built (or, on the PyTorch backend, configured) for that same `world_size`; you cannot load a 4-GPU checkpoint with `--world_size=2` or vice versa. If `instance_group { kind: KIND_MODEL }` is set (as in the example ensemble above), Triton delegates device placement entirely to the backend's own multi-GPU orchestration rather than trying to manage it itself — this is the correct setting whenever the backend is doing TP/PP internally.

### Part 3 — benchmarking it correctly with GenAI-Perf

Curl calls tell you the pipeline *works*. They tell you nothing about throughput, tail latency, or whether your batching configuration is actually helping. **GenAI-Perf** (a Perf Analyzer subcommand purpose-built for generative workloads) is the tool for that.

#### Install

```bash
pip install genai-perf
# or, with zero local setup, use the Triton SDK container:
docker run --gpus all --rm -it --net=host \
  nvcr.io/nvidia/tritonserver:25.10-py3-sdk bash
```

#### Benchmark the TensorRT-LLM ensemble over gRPC (KServe protocol)

```bash
genai-perf profile \
  -m ensemble \
  --backend tensorrtllm \
  --url localhost:8001 \
  --endpoint-type kserve \
  --streaming \
  --concurrency 16 \
  --synthetic-input-tokens-mean 200 --synthetic-input-tokens-stddev 20 \
  --output-tokens-mean 128 --output-tokens-stddev 10 \
  --measurement-interval 10000 \
  --profile-export-file trtllm_c16.json
```

#### Benchmark the vLLM backend over its OpenAI-compatible frontend

```bash
genai-perf profile \
  -m my_vllm_model \
  --service-kind openai --endpoint-type chat \
  --url localhost:8000 \
  --streaming \
  --concurrency 16 \
  --synthetic-input-tokens-mean 200 \
  --output-tokens-mean 128
```

#### Reading the output

GenAI-Perf prints (and writes to CSV/JSON) a table with, at minimum:

| Metric | What it tells you |
|--------|--------------------|
| **Time to first token (TTFT)** | Perceived "did it start responding" latency — dominated by prefill + queueing. |
| **Inter-token latency (ITL)** | Steady-state per-token decode speed — dominated by batch composition and KV-cache pressure. |
| **Output token throughput (tokens/s)** | Aggregate decode throughput across all concurrent requests — the number that answers "how many users can this GPU serve." |
| **Request throughput (req/s)** | End-to-end completions per second — sensitive to `max_tokens` distribution. |
| **p50/p90/p99 latency** | Tail behavior; the number that catches SLO violations averages hide. |

The workflow that actually matters in production: **run GenAI-Perf at a sweep of concurrencies (1, 4, 8, 16, 32, 64…) before and after any config change** — a new `kv_cache_free_gpu_mem_fraction`, a different `instance_group`, a Triton or TensorRT-LLM version bump — and diff the curves. A config that looks fine at concurrency 1 can fall off a cliff at 32 because the KV cache runs out of room; only the sweep shows you that. Treat a GenAI-Perf regression sweep as a release gate the same way you would a latency/throughput dashboard for a database migration.

---

## Triton + TensorRT-LLM vs vLLM vs TGI

| Dimension | **Triton + TensorRT-LLM** | **vLLM** (standalone) | **TGI** (Text Generation Inference) |
|-----------|---------------------------|-----------------------|-------------------------------------|
| Primary goal | Multi-model serving fleet + fastest NVIDIA LLM path | Fast, simple LLM serving | HuggingFace-native LLM serving |
| Continuous batching | Yes (in-flight, fused) | Yes (native) | Yes (native) |
| KV cache | Paged | PagedAttention (originator) | Paged |
| Setup effort | **High** (classic engine path) to **medium** (new PyTorch-backend / LLM API path) | **Low** — `pip install`, point at HF model | Low–medium — Docker + model id |
| Peak throughput on NVIDIA | **Highest** (tuned kernels, esp. with FP8/FP4 quantization) | Very high | High |
| Hardware | NVIDIA only | NVIDIA-first (some others) | NVIDIA-first (some others) |
| Non-LLM models | **Yes** — same server hosts ONNX/PyTorch/TensorRT | No | No |
| Ops surface | One server, unified metrics, ensembles/BLS | Simple, LLM-scoped | Simple, LLM-scoped |
| Streaming | Decoupled + `/generate_stream` | OpenAI-compatible SSE | SSE, OpenAI-compatible |
| Best when | You run many models and/or want max NVIDIA LLM perf with unified ops | You want the least-effort fast LLM server | You are all-in on the HF stack |

Honest summary: **for a single LLM, vLLM or TGI is faster to stand up and gets you most of the throughput with a fraction of the effort.** Triton + TensorRT-LLM wins when you need a heterogeneous model fleet under one runtime, or when you have squeezed everything else and need the last increment of GPU efficiency and are willing to pay the setup tax. Note also that Triton can host the **vLLM backend**, giving you vLLM's ergonomics *inside* Triton's ops framework — a common middle ground, and increasingly the default recommendation for teams that want Triton's multi-model story without the TensorRT-LLM engine-build tax.

---

## (A) The 2025–2026 landscape

Triton and TensorRT-LLM have both changed shape since the "build an engine, wire an ensemble" era described above was the *only* way to do this. If you are interviewing or designing a new deployment in 2026, know the current picture — not just the mechanics.

### The TensorRT-LLM backend has moved, and PyTorch became the default execution path

The Triton-specific glue code that used to live in its own `triton-inference-server/tensorrtllm_backend` repository has been **relocated into the TensorRT-LLM repository itself**, under a `triton_backend/` directory — the two projects now ship and version together rather than as loosely-coupled siblings (github.com/triton-inference-server/tensorrtllm_backend, and github.com/NVIDIA/TensorRT-LLM). More significantly, TensorRT-LLM's own execution backend changed: what used to be the *only* path — compile a model with `trtllm-build` into a `.engine`/`.plan`, then load that plan — is now the **legacy** path. The **PyTorch execution backend** (often called the "LLM API") is the default: you point it at a HuggingFace checkpoint and it builds and manages the runtime graph itself, no separate offline compile step required. By TensorRT-LLM's 1.2 release line, the PyTorch backend became effectively the *sole* supported execution backend, with the classic TensorRT engine-compile workflow deprecated (nvidia.github.io/TensorRT-LLM/release-notes.html; github.com/NVIDIA/TensorRT-LLM/releases). Practically, this means:

- New deployments should default to the PyTorch/LLM-API backend inside Triton (container tags like `*-trtllm-python-py3`) unless you have a specific reason to hand-tune a compiled engine.
- The mental model in this chapter — engine build, then ensemble, then launch — still describes how the **request path** works (in-flight batching, paged KV cache, pre/post-processing via Python models); what changed is *how the model artifact itself gets produced*, not the serving architecture around it.
- If you inherited a repo that still does `trtllm-build` by hand, it will keep working, but plan a migration; NVIDIA's own examples now lead with the PyTorch path.

### Quantization and precision: FP4/NVFP4 arrive alongside FP8

TensorRT-LLM's quantization matrix has grown substantially on Blackwell-class GPUs (nvidia.github.io/TensorRT-LLM/latest/features/quantization.html):

- **FP8** — per-tensor, block-scaling, and rowwise variants — is broadly supported from Ada/Hopper through Blackwell, including an **FP8 KV cache** to shrink the biggest LLM-serving memory line item.
- **FP4 / NVFP4** (4-bit floating point, plus an MXFP4 variant) is now supported on Blackwell (sm100/sm103) for both weights and, on the newest GPUs, KV cache — roughly halving memory footprint again versus FP8 with NVIDIA reporting minimal accuracy loss on validated model families.
- **Weight-only integer quantization** (W4A16/W4A8 AWQ and GPTQ) remains available across generations for teams on Ampere/Ada hardware without FP8/FP4 tensor cores.

The practical takeaway for an interview or a design doc: **quantization choice is now a GPU-generation decision as much as an accuracy decision.** On Blackwell, FP4/NVFP4 is usually the throughput-per-dollar winner if the model has been validated at that precision; on Hopper, FP8 is the mainstream default; older GPUs fall back to AWQ/GPTQ weight-only quantization.

In practice, quantizing a checkpoint for the PyTorch/LLM-API backend is a config-driven step rather than a bespoke script — you point the build/serve tooling at the checkpoint and declare the target precision, and it handles calibration for the formats that need it (AWQ/GPTQ still require a calibration pass over representative data; FP8 and FP4/NVFP4 on validated model families can often run with vendor-provided or activation-aware scaling with a lighter calibration step). The engineering discipline that doesn't change: always re-run your task-specific eval suite (not just perplexity) after quantizing, since generative quality degradation from aggressive quantization is uneven across tasks — code and math are typically more sensitive than open-ended chat.


### Disaggregated serving and speculative decoding are now first-class

Two techniques that used to be research topics are now supported serving patterns:

- **Disaggregated (prefill/decode-split) serving** — running prefill and decode on separate GPU pools and transferring the KV cache between them (over NVLink or a network fabric) so that the compute-bound prefill phase and the memory-bandwidth-bound decode phase don't contend for the same hardware. TensorRT-LLM added a KV Cache Connector API specifically to make this state-transfer pluggable, and Triton's own reference architectures describe 1.2–2.5x throughput gains from splitting the two phases at scale.
- **Speculative decoding** — draft-and-verify schemes (n-gram drafting, multi-layer EAGLE-3, and external draft models) are now integrated with in-flight batching and guided/structured decoding, so you can turn on speculation without giving up continuous batching.

Neither is "day one" complexity — start with a single-pool in-flight-batched deployment — but both are now the answer to "we've maxed out a single-pool deployment, what next," and are worth naming in a systems-design interview even if you have not implemented them yourself.

### Where Triton sits next to vLLM and SGLang today

vLLM and SGLang have both matured into serious standalone production servers with their own continuous batching, quantization, disaggregated-serving, and OpenAI-compatible APIs. That narrows — but does not eliminate — Triton's differentiation:

- **NVIDIA's own positioning** (see the vLLM x Triton materials NVIDIA has published) is that Triton is not competing with vLLM's engine — it *wraps* it. The Triton vLLM backend measures within roughly 2% of standalone vLLM's throughput and latency, while adding Triton's scheduling, multi-model hosting, Prometheus metrics, and an OpenAI-compatible FastAPI front door on top. In other words: you can get vLLM's engine *and* Triton's ops surface at the same time.
- **SGLang** has pulled ahead on some structured-generation and multi-turn/prefix-heavy workloads (its RadixAttention prefix cache), and is a legitimate default for teams whose workload is dominated by long shared prefixes (agents, few-shot prompting, RAG with repeated system prompts).
- **Choose Triton when:** you are serving a fleet of heterogeneous models (LLM + embedding + reranker + a classic ONNX/PyTorch model) behind one runtime; you need enterprise support and a stable KServe v2 API across model types; or you specifically need TensorRT-LLM's peak NVIDIA-hardware throughput with FP4/FP8 and are willing to operate the extra moving part.
- **Choose vLLM or SGLang directly when:** you have exactly one (or a small number of) LLMs, want the fastest path to a working OpenAI-compatible endpoint, and don't need a shared multi-framework serving runtime. Many teams now run vLLM/SGLang standalone for the LLM and only reach for Triton once a second or third *non-LLM* model shows up that needs to share the same ops story.
- **The pragmatic middle ground** more teams land on in 2026: Triton hosting the **vLLM backend** rather than TensorRT-LLM — you get Triton's multi-model repository, metrics, and ensembles, without paying the TensorRT-LLM engine/PyTorch-backend conversion tax, and you upgrade to the TensorRT-LLM backend later only if profiling shows you actually need the extra throughput.

---

## Failure modes and pitfalls

- **Wrong backend for LLMs.** Serving generation through ONNX/PyTorch + `dynamic_batching` produces terrible throughput and head-of-line blocking. LLMs require the TensorRT-LLM or vLLM backend with in-flight batching. This is the number-one mistake.
- **Static-batch TensorRT-LLM engine.** Building the engine without paged KV cache, or leaving `batching_strategy`/`gpt_model_type` at `v1`, silently disables continuous batching. You paid the conversion cost and got none of the benefit.
- **Misconfigured dynamic batching.** `max_queue_delay_microseconds` too high causes latency spikes; too low causes tiny batches and an idle GPU. `preferred_batch_size` mismatched to what the engine was tuned for wastes time on padding. Tune against GenAI-Perf, not by guessing.
- **`max_batch_size` vs shape confusion.** With `max_batch_size > 0` the batch dim is *implicit* — listing it explicitly in `dims` double-counts it and breaks shape checks. Set `max_batch_size: 0` only for models whose first dim is not a batch dimension.
- **KV-cache OOM.** `kv_cache_free_gpu_mem_fraction` too aggressive (or too many concurrent LLM instances) OOMs under load; too conservative wastes capacity. Watch KV-cache utilization metrics.
- **Version / container mismatches.** The TensorRT-LLM *engine* (or PyTorch-backend checkpoint), the *backend build*, and the Triton container are a matched set. Loading an artifact built against one TensorRT-LLM release inside a mismatched Triton image fails to load or crashes. Pin versions together, and re-validate after every upgrade — see the war story below.
- **Model name / directory mismatch.** `name` in `config.pbtxt` disagreeing with the directory, or a non-integer version folder, makes the model silently not load. Read the startup READY/UNAVAILABLE table.
- **Conversion complexity underestimated.** Even on the newer PyTorch/LLM-API path, wiring the ensemble (pre/post-processing, tokenizer directories, decoupled streaming) is genuinely involved and model-specific. Budget for it; do not promise a one-day LLM deploy.
- **Forgetting `--shm-size` / decoupled streaming.** Python-backend ensembles need adequate shared memory; streaming needs `decoupled: true` or you get one blob at the end instead of tokens.

---

## (C) Production case studies & war stories

### War story 1 — the config regeneration that silently capped throughput at 1x

**Setup:** A team ran an ONNX sentiment classifier in Triton with a hand-tuned `config.pbtxt`: `dynamic_batching { preferred_batch_size: [8, 16, 32] max_queue_delay_microseconds: 2000 }` and two GPU instances. Throughput was healthy for months.

**The incident:** A routine model refresh redeployed the model directory from a CI pipeline that regenerated `config.pbtxt` from a template — and the template had been written against `--strict-model-config=false` defaults, before anyone had added the `dynamic_batching` block. The new config still loaded fine (Triton auto-generated a valid config from the ONNX graph), the model still went `READY`, health checks still passed — but the auto-generated config had **no `dynamic_batching` block at all**, meaning every request executed one at a time. GPU utilization on the dashboard quietly dropped from ~70% to under 10%, and p99 latency crept up as the request queue backed up during traffic peaks — but nothing *failed*, so no alert fired.

**How it was caught:** A GenAI-Perf-style concurrency sweep run before the next capacity-planning review showed throughput flatlining at concurrency 4 instead of scaling to concurrency 32 the way the same model had six months earlier. Diffing `config.pbtxt` between the running container and the last known-good version showed the missing `dynamic_batching` stanza immediately.

**Lesson:** Auto-generated config (`--strict-model-config=false`) is fine for local experimentation and dangerous in a CI/CD pipeline that doesn't diff the result. Run production with `--strict-model-config=true` so a missing or malformed batching config is a hard failure at load time, not a silent throughput regression discovered a quarter later. Treat `config.pbtxt` as reviewed, versioned infrastructure code, not a generated artifact.

A cheap guardrail that would have caught war story 1 before it shipped — a CI check run against every `config.pbtxt` change:

```bash
#!/usr/bin/env bash
# ci_check_batching_config.sh — fail the pipeline if a model that should
# batch doesn't declare a batching strategy.
set -euo pipefail
for cfg in models/*/config.pbtxt; do
  name=$(dirname "$cfg")
  if grep -q 'backend: "tensorrtllm"' "$cfg"; then
    grep -q 'batching_strategy' "$cfg" || { echo "FAIL: $name missing batching_strategy"; exit 1; }
  elif grep -qE 'backend: "(onnxruntime|pytorch)"' "$cfg" && grep -q 'max_batch_size: [1-9]' "$cfg"; then
    grep -q 'dynamic_batching' "$cfg" || { echo "FAIL: $name has max_batch_size > 0 but no dynamic_batching block"; exit 1; }
  fi
done
echo "All configs OK"
```

It is a blunt instrument — it checks for the presence of a block, not that the values are well-tuned — but "present vs silently absent" is exactly the failure mode that bit this team, and a five-line grep script in CI is cheaper than a quarter of degraded throughput.


### War story 2 — an upgrade that loaded fine and ran 3x slower

**Setup:** A team running a TensorRT-LLM ensemble upgraded their Triton container to pick up a security patch, without rebuilding the model artifact, on the assumption that "the model directory didn't change, so nothing needs rebuilding."

**The incident:** The new container loaded the existing engine/checkpoint without erroring — but the backend version bundled with the new image did not match the one the artifact had been produced against closely enough to run the requested `batching_strategy` at full performance; it silently fell back to a degraded execution path. Nothing crashed. Nothing appeared in the error logs. The service was "working." Token throughput per GPU, measured only informally by an on-call engineer noticing users complaining that responses "felt slower," turned out to be down roughly 3x versus the pre-upgrade baseline.

**How it was caught:** Because there was no automated before/after GenAI-Perf comparison gating the rollout, this took days to notice and diagnose — the eventual fix was rebuilding the artifact against the new backend version and re-running the same GenAI-Perf concurrency sweep to confirm parity before calling the upgrade complete.

**Lesson:** Pin the TensorRT-LLM/backend version, the model artifact, and the Triton container tag together as one versioned unit, and treat any change to *any one* of the three as a change to all three — rebuild and re-benchmark, don't assume compatibility because it loads. Bake a GenAI-Perf regression sweep into the deployment pipeline as an automated gate (fail the rollout if p50 tokens/s at a fixed concurrency drops more than some threshold, e.g. 10%, versus the current production baseline) rather than relying on a human noticing a "feels slower" complaint.

### War story 3 — KV cache sized for the demo, not for peak concurrency

**Setup:** `kv_cache_free_gpu_mem_fraction` was set to `0.9` during initial rollout, validated against a demo workload of a handful of short prompts.

**The incident:** Under real traffic — many concurrent long-context conversations — the KV cache filled up, and new requests began queueing behind an in-flight batch that could not make room for them, producing a sawtooth pattern of throughput collapsing to zero and recovering, visible in `nv_inference_queue_duration_us` spiking in lockstep with KV-cache utilization hitting 100%.

**Lesson:** Load-test with a realistic input/output token-length distribution and concurrency — not a demo prompt — before shipping a `kv_cache_free_gpu_mem_fraction` value, and alert on KV-cache utilization directly rather than waiting for the downstream symptom (queue duration) to show up.

---

## Operating Triton at scale: Kubernetes, security, and multi-tenancy

### Kubernetes deployment shape

Triton itself doesn't manage a fleet of replicas or autoscale — that's Kubernetes' (or KServe's) job, with Triton as the container image running inside each pod. The common shape:

- **Deployment/StatefulSet** running the `tritonserver` container, GPU requested via `nvidia.com/gpu: 1` (or more, for a multi-GPU TP/PP model replica — in which case the pod typically also needs multiple GPUs scheduled onto the same node, or a multi-node MPI job for very large models).
- **Readiness/liveness probes** wired to `GET /v2/health/ready` and `/v2/health/live` — a model still loading (e.g., a large LLM checkpoint) should fail readiness, not liveness, so Kubernetes doesn't kill a pod that's merely slow to start.
- **Horizontal Pod Autoscaler** driven by a custom metric — GPU utilization alone is a poor autoscaling signal for LLM serving because a GPU running in-flight batching near KV-cache capacity can show high utilization while still queueing; `nv_inference_queue_duration_us` or a GenAI-Perf-derived tokens/s-per-replica target is a better trigger.
- **KServe's `InferenceService` CRD** wraps this pattern with a standard interface across ONNX, PyTorch, and Triton-hosted models, and is a common choice when a platform team wants one custom resource across many serving runtimes rather than hand-rolled Deployments per model.

### Security

Triton's core does not implement authentication or authorization — treat the raw HTTP/gRPC/metrics ports as internal-network-only and put a gateway in front for anything internet-facing:

- **API keys or mTLS at the gateway/ingress**, not at Triton itself — Envoy, an API gateway, or a service mesh sidecar is the right layer for this.
- **The model-repository control API (`/v2/repository/models/{name}/load`/`unload`) is an administrative capability** — anyone who can reach it can load an arbitrary model from the repository storage or unload a production model. Restrict it to an internal admin network or a separate management port/network policy; do not expose it on the same public path as inference traffic.
- **The metrics endpoint (`:8002`) can leak operational detail** (model names, request volumes) — scope its exposure to your monitoring stack's network, not the public internet.

### Multi-tenancy

Triton's model repository and instance groups give you the primitives for multi-tenant isolation, but the isolation policy is yours to build:

- **Resource isolation** — pin different tenants' models to different `instance_group { gpus: [...] }` sets (or different node pools in Kubernetes) if noisy-neighbor GPU contention between tenants is a concern; Triton does not enforce per-tenant fairness within a shared GPU on its own.
- **Namespacing** — prefixing model names by tenant (`tenantA_sentiment`, `tenantB_sentiment`) in a shared repository is simple but couples tenants to one repository's blast radius (a bad `--strict-model-config` change or a repository-wide restart affects everyone); separate model repositories per tenant, each behind its own Triton deployment, trade operational simplicity for stronger isolation.
- **Quota/rate limiting** happens above Triton — at the gateway — since Triton's own priority levels and batching config are per-model scheduling knobs, not per-tenant quota enforcement.

---

## (D) Interview mastery

### Explain when you'd choose Triton over vLLM in 60 seconds

> "Default to vLLM (or SGLang) if I have one LLM and want the fastest path to an OpenAI-compatible endpoint — it's less setup, and it gets most of the throughput of any alternative. I reach for Triton when I have more than just an LLM: an embedding model, a reranker, maybe a classic ONNX classifier, all needing to share GPUs, expose one metrics format, and version consistently — Triton is a serving *runtime* that hosts all of them, LLM included, behind one API. If I specifically need NVIDIA's peak-throughput LLM path, I'd use the TensorRT-LLM backend inside Triton, accepting the extra setup for FP8/FP4 kernels and in-flight batching tuned at the kernel level. But a very common middle ground today is Triton hosting the *vLLM* backend — you get vLLM's engine, which NVIDIA's own numbers put within a couple percent of standalone vLLM, plus Triton's multi-model hosting, ensembles, and metrics, without paying the TensorRT-LLM engine-build tax. So it's not really 'Triton vs vLLM' — it's 'do I need a multi-model runtime, and if so, which engine do I put inside it.'"

### System design prompt: serve an LLM, an embedding model, and a reranker behind one inference server

**Prompt as asked in interviews:** *"Design a system that serves three model types — a generative LLM, an embedding model, and a cross-encoder reranker — for a RAG application, behind a single inference server."*

**Sketch:**

```
                     ┌─────────────────────────────┐
                     │   Triton Inference Server    │
                     │  (single process, 1+ GPUs)   │
                     │                              │
  client ── gRPC/HTTP┼──▶ rag_pipeline (ensemble)    │
                     │        │                     │
                     │        ├─▶ embedder            (ONNX / PyTorch backend)
                     │        │     dynamic_batching, 2 instances, KIND_GPU
                     │        │
                     │        ├─▶ [external vector search — outside Triton,
                     │        │    called from a BLS step or by the client]
                     │        │
                     │        ├─▶ reranker             (ONNX / PyTorch backend,
                     │        │     cross-encoder)      dynamic_batching, small
                     │        │                         max_batch_size, 1 instance
                     │        │
                     │        └─▶ ensemble/BLS: preprocessing → tensorrt_llm/vllm
                     │              → postprocessing   (in-flight batching,
                     │                                  decoupled streaming)
                     └─────────────────────────────┘
```

Repository sketch:

```
/models/
├── embedder/            # backend: onnxruntime or pytorch, dynamic_batching
├── reranker/             # backend: onnxruntime or pytorch, dynamic_batching, small batches
├── preprocessing/        # backend: python (tokenizer for the LLM)
├── tensorrt_llm/         # or "vllm" — the generative model, in-flight batching
├── postprocessing/       # backend: python (detokenizer)
└── rag_pipeline/         # platform: ensemble or BLS, ties everything together
```

Key design points to say out loud:

1. **Different batching per model type.** The embedder and reranker are fixed-shape, equal-work models — `dynamic_batching` is correct and sufficient. The LLM is autoregressive and variable-length — it needs in-flight batching via the TensorRT-LLM or vLLM backend. Using the same batching strategy for all three is the interview red flag to avoid.
2. **Instance groups sized to workload shape.** The embedder is usually the highest-QPS, cheapest-per-call model — give it more instances or a dedicated GPU. The reranker runs on a much smaller candidate set per request (rerank top-50, not every document) — it needs less concurrency. The LLM usually owns its own GPU(s) outright because in-flight batching handles its concurrency internally.
3. **Vector search is not a Triton model.** Whether to put retrieval inside a BLS step (calling out to an external vector DB from Python) or keep it as a separate service the client/orchestrator calls between the embed step and the rerank+generate step is a real design decision — BLS keeps it inside one server call at the cost of coupling Triton to the vector DB's availability; keeping it external keeps Triton stateless but adds a network hop and moves orchestration logic to the client.
4. **One ensemble vs multiple client calls.** If retrieval must happen *between* embedding and reranking, and it's an external service, you likely cannot express the whole RAG flow as a single static `ensemble` (no external I/O mid-DAG) — you'd either do it as a BLS model that calls out over HTTP from Python, or split it into two client-visible calls: `embed_and_search` handled by the orchestrator, then `rerank_and_generate` as one ensemble.
5. **Metrics and scaling story.** All three model types show up in the same Prometheus `/metrics` endpoint with per-model queue duration and compute duration — call this out as the payoff of the unified-runtime choice versus running three separate servers.

### Red flags vs green flags

| Signal | Red flag | Green flag |
|--------|----------|------------|
| Batching choice for an LLM | "I'd use `dynamic_batching` for the LLM too, for consistency." | "LLM decoding is variable-length and iterative, so it needs in-flight/continuous batching, not `dynamic_batching`." |
| Config management | "Let auto-generated config handle it, it's simpler." | "Production runs `--strict-model-config=true`; batching config is reviewed, versioned infra." |
| Upgrades | "If the container starts and the model loads, the upgrade is safe." | "Loading isn't validating. Re-run a GenAI-Perf sweep and diff against the pre-upgrade baseline before calling it done." |
| KV cache sizing | "Set `kv_cache_free_gpu_mem_fraction` once and move on." | "Load test with realistic concurrency and context length distributions, and alert on KV-cache utilization directly." |
| Tool choice | "Triton is always better/always worse than vLLM." | "Depends on whether there's a multi-model fleet; single-LLM workloads often don't need Triton at all." |
| Ensembles vs BLS | Reaches for BLS (arbitrary Python) for every pipeline, even static DAGs. | Uses a plain `ensemble` for static DAGs; reserves BLS for branching/looping logic or calls to external services. |
| Debugging a "silent" throughput regression | Assumes the problem is the GPU or the model. | First checks whether `config.pbtxt` still contains the expected `dynamic_batching`/`instance_group`/`batching_strategy` block after the last deploy. |

### Q&A

1. **"You have five models in four frameworks sharing two GPUs. Design it."** One Triton server, one model repository, per-model `backend` and `instance_group`, dynamic batching on the fixed-shape models, an LLM on the TensorRT-LLM/vLLM backend. Tests whether you understand Triton's core value proposition.
2. **"Dynamic vs in-flight batching — when each, and why?"** Fixed-shape/equal-work models use dynamic batching; autoregressive LLMs use in-flight/continuous batching, because variable output length makes static batches stall on the slowest sequence. Bonus points for mentioning paged KV cache.
3. **"Walk me through deploying an LLM on Triton today."** Either point the TensorRT-LLM PyTorch/LLM-API backend at a HF checkpoint directly (the current default path, no offline engine compile), or fall back to the classic `trtllm-build` engine-compile path if you need it; wire pre/post-processing plus the model into an ensemble or BLS; launch with `world_size`/tensor-parallel settings matching the GPU count; call `/generate` or `/generate_stream`. Being explicit that the PyTorch-backend path is now the default, and being honest that ensemble wiring is still real engineering effort, both score points.
4. **"How do you tune the latency/throughput tradeoff?"** `max_queue_delay_microseconds` and `preferred_batch_size` for dynamic batching; instance count for concurrency; `kv_cache_free_gpu_mem_fraction` and max sequence length for LLMs — all validated with a GenAI-Perf concurrency sweep (TTFT, ITL, tokens/s), not guessed.
5. **"What do you monitor, and what does a bad number mean?"** `nv_inference_queue_duration_us` (batching pressure), compute duration, `nv_gpu_utilization`, KV-cache utilization, `nv_inference_first_response_histogram_ms` for TTFT. Rising queue time with a full KV cache means memory-bound — scale out or trim context; rising queue time with an idle GPU means an under-tuned batching window.
6. **"Ensemble vs BLS?"** Ensemble for a static DAG (tokenize→infer→detokenize); BLS when the pipeline branches, loops, or needs to call an external service based on runtime data.
7. **"When would you NOT use Triton?"** A single LLM where vLLM/SGLang is dramatically simpler and gets you most of the throughput; no heterogeneous model fleet; a team without NVIDIA-stack depth. Knowing when the simpler tool wins signals seniority.
8. **"How do you roll out a new model version safely?"** Version subdirectories plus `version_policy`, load the new version alongside the old, shift traffic, hot-unload the old one — no server restart. For an LLM engine/checkpoint change specifically, also re-run the GenAI-Perf benchmark before promoting.
9. **"What changed in the TensorRT-LLM + Triton integration recently, and why does it matter?"** The Triton-facing backend code moved into the TensorRT-LLM repo itself (`triton_backend/`), and the PyTorch execution backend (the "LLM API") replaced the classic `trtllm-build`-then-load-a-plan workflow as the default, letting you serve a HF checkpoint directly. It matters because it lowers the setup cost that used to be Triton's biggest disadvantage versus vLLM.
10. **"How would you decide between FP8 and FP4/NVFP4 quantization for an LLM deployment?"** It's largely a GPU-generation decision: FP4/NVFP4 needs Blackwell-class tensor cores and roughly halves memory again versus FP8, so it's the throughput-per-dollar default there if the model family has been validated at that precision; FP8 is the mainstream choice on Hopper/Ada; older GPUs fall back to weight-only INT4 (AWQ/GPTQ). Always validate accuracy on your own eval set before shipping a lower precision.
11. **"What is disaggregated serving and when would you reach for it?"** Splitting prefill (compute-bound, benefits from batching many prompts) and decode (memory-bandwidth-bound, benefits from many concurrent small steps) onto separate GPU pools, transferring the KV cache between them. Reach for it once a single-pool in-flight-batched deployment is GPU-saturated and profiling shows prefill and decode are contending for the same hardware — not as a first deployment.
12. **"A model is `READY` in the startup table but throughput is terrible. What do you check first?"** Whether `config.pbtxt` actually contains the batching block you expect (dynamic_batching for fixed-shape, correct `batching_strategy` for TensorRT-LLM) — auto-generated or regenerated configs silently omitting it is the single most common cause of a model that "loads fine" but runs at a fraction of expected throughput.
13. **"Why would decoupled mode matter even if you don't need streaming to the end user?"** Any model that internally produces a variable number of responses per request — including intermediate steps inside a BLS pipeline — needs `model_transaction_policy { decoupled: true }`, or Triton will only deliver the final buffered response, breaking any pipeline stage that expects to see partial output.
14. **"How do you avoid a repeat of a 'looked fine after the upgrade, was actually 3x slower' incident?"** Pin the model artifact, backend build, and container tag together as one versioned unit; gate every upgrade behind an automated GenAI-Perf sweep compared against the current production baseline, not a manual smoke test that only checks the model loads.
15. **"How do you hot-swap a model version with zero downtime?"** Set `--model-control-mode=explicit`, drop the new version directory into the repository, `POST /v2/repository/models/{name}/load` to bring it up alongside the still-serving old version, shift client traffic (or update `version_policy` to prefer the new version), then `unload` the old one — no server restart, no dropped requests during the transition.
16. **"What happens if a client abandons a streaming LLM request halfway through?"** Without cancellation handling, the in-flight batching scheduler keeps generating tokens nobody will read, wasting GPU-seconds and holding KV-cache blocks. gRPC call cancellation should propagate down into the backend so the sequence is evicted and its KV-cache blocks freed as soon as the client disconnects — worth calling out explicitly, since it's an easy thing to leave unhandled until a cost review surfaces it.
17. **"Tensor parallel vs pipeline parallel — when would you pick one over the other?"** Tensor parallelism shards every layer's weights across GPUs with an all-reduce per layer — needs NVLink-class bandwidth, but keeps latency for a single request low. Pipeline parallelism assigns whole layers to different GPUs with lower per-step communication, but only pays off with enough concurrent requests to keep the pipeline full, and adds bubble latency to the first request through an empty pipeline. Many large-model deployments combine both.

---

## Further reading

- Triton model configuration (config.pbtxt reference): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html
- Triton model repository layout: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md
- Dynamic batching & concurrent model execution (conceptual guide): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html
- TensorRT-LLM backend (now vendored inside the TensorRT-LLM repo, `triton_backend/`): https://github.com/triton-inference-server/tensorrtllm_backend
- TensorRT-LLM backend docs: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html
- TensorRT-LLM main repository (PyTorch/LLM-API execution backend, release notes): https://github.com/NVIDIA/TensorRT-LLM
- TensorRT-LLM release notes (PyTorch backend, disaggregated serving, speculative decoding): https://nvidia.github.io/TensorRT-LLM/release-notes.html
- TensorRT-LLM quantization reference (FP8, FP4/NVFP4, AWQ/GPTQ, KV-cache quantization): https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html
- vLLM backend for Triton: https://github.com/triton-inference-server/vllm_backend
- Deploying a vLLM model in Triton (tutorial): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Quick_Deploy/vLLM/README.html
- NVIDIA vLLM x Triton positioning (integration, benchmarks, disaggregated serving): https://developer.download.nvidia.com/triton/vLLM-x-Triton-meetup-External.pdf
- Python backend (custom logic + BLS): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html
- Metrics reference: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md
- GenAI-Perf (LLM benchmarking): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html
- Triton Inference Server release notes index (container tags, backend versions bundled per release): https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html
