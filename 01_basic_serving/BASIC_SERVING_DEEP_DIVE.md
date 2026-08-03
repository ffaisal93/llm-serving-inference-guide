# Basic LLM Serving — A Deep Dive

> Standing up a language model behind an HTTP API from first principles, and understanding *why* the naive version is slow.

## Why this matters

Every production LLM system — ChatGPT, a support bot, a code assistant — is at bottom a loop that turns text into tokens, runs a forward pass, and turns tokens back into text, wrapped in a network server. If you understand that loop end to end, and understand the two costs that dominate it (compute and GPU memory), the rest of this book is just engineering to make the loop cheaper and more concurrent.

This chapter builds the loop the *obvious* way: one model, one process, one request at a time. That version works, and it is a perfect teaching tool precisely because it is slow. By the end you will be able to say, with numbers, exactly where the time and the memory go — and that motivates batching, KV-cache management, and dedicated engines (vLLM, TGI, Triton) in the chapters that follow.

We keep the intuition first and the mechanism precise. Where there is a tradeoff, we name it honestly.

---

## Core intuition: an LLM is an autoregressive next-token loop

A decoder-only transformer computes one thing: given a sequence of tokens, a probability distribution over the *next* token. Generation is just calling that repeatedly.

```
prompt: "The capital of France is"
   -> tokenizer -> [464, 3139, 286, 4881, 318]
   -> model -> logits over ~50k vocab -> pick "Paris" (token 6342)
   -> append -> [464, 3139, 286, 4881, 318, 6342]
   -> model -> pick "." -> append -> ...
   -> stop on EOS or max_new_tokens
   -> tokenizer.decode(...) -> " Paris."
```

Two things follow immediately, and they structure everything:

1. **Generation is sequential.** Token *N+1* depends on token *N*. You cannot decode a 200-token answer in one shot; you do (at least) 200 forward passes. This is why latency scales with output length.
2. **The model re-reads its own context every step** — unless you cache. The naive loop re-processes the whole sequence on every token, which is quadratic waste. The fix is the **KV cache** (below), and the KV cache is what eats your GPU memory.

Hold those two facts. The whole performance story is a consequence of them.

---

## Loading a model: weights, dtype, device, tokenizer

Before serving anything you load four things. Each has a failure mode.

### Weights and where they live

A model is a set of tensors (the parameters) plus a config describing the architecture. With Hugging Face Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype="bfloat16",   # precision — see below
    device_map="cuda",        # placement
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
```

The weights download once to a local cache and are memory-mapped from `safetensors` shards on subsequent loads. Cold start = download + load + allocate; warm start = load + allocate. This distinction matters for autoscaling (Chapter 6): a cold pod may take minutes.

### dtype / precision — the single biggest memory lever

Parameters are stored as floating-point numbers, and the *bytes per parameter* is a choice you make at load time.

| dtype | bytes/param | Typical use |
|-------|-------------|-------------|
| FP32  | 4 | Rarely for inference; training reference |
| FP16 / BF16 | 2 | Standard inference precision on GPU |
| INT8  | 1 | Quantized inference (small quality hit) |
| INT4 / NF4 | ~0.5 | Aggressive quantization, edge/consumer GPUs |

BF16 (bfloat16) is usually preferred over FP16 on modern GPUs: same 2 bytes, but a wider exponent range, so it is less prone to overflow/`NaN` during the forward pass. FP32 doubles your memory for almost no inference quality gain — do not serve in FP32 by accident (it is the default if you forget `torch_dtype`).

### Device placement

Weights must sit in GPU memory (VRAM) for fast inference. `device_map="cuda"` puts everything on one GPU; `device_map="auto"` will *shard* across multiple GPUs or spill to CPU/disk if the model does not fit — convenient, but CPU offload is catastrophically slow for serving. For a serving path you want the whole model resident on the GPU and you want to *know* it fits (memory math below).

### Tokenizer — small, and a classic source of silent bugs

The tokenizer maps text <-> integer IDs. It must be the *exact* one the model was trained with; a mismatch produces garbage output with no error. Two things to get right in a server:

- **Chat template.** Instruct/chat models expect a specific formatting of roles (`<|user|>`, `<|assistant|>`, etc.). Use `tokenizer.apply_chat_template(messages, add_generation_prompt=True)` rather than hand-concatenating strings — getting the special tokens wrong quietly degrades quality.
- **Padding side and pad token.** For batched generation, decoder-only models must **left-pad** (`tokenizer.padding_side = "left"`), and many models ship without a `pad_token` — set `tokenizer.pad_token = tokenizer.eos_token`. Right-padding a decoder batch corrupts the generation. (We revisit padding when we build real batching.)

---

## Mechanism in depth: prefill vs decode, and the KV cache

This is the heart of the chapter. A single generation request has **two phases with completely different performance characteristics.**

### Prefill (the prompt pass)

You feed the entire prompt (say 500 tokens) through the model **in one forward pass**. Because all prompt tokens are known up front, they are processed *in parallel* — the GPU does one big matrix-multiply-heavy pass over all 500 positions at once. This is **compute-bound**: it saturates the GPU's arithmetic units. Prefill is what you pay for **time-to-first-token (TTFT)**, and its cost grows with prompt length.

During prefill the model computes, for every layer and every attention head, a **key (K)** and **value (V)** vector for each prompt token, and stores them — that is the KV cache.

### Decode (the generation loop)

Now you generate one token at a time. Each decode step feeds **only the single newest token** through the model. Its attention needs the K/V of *all previous tokens* — but those are already in the cache, so you do **not** recompute them. Each decode step is therefore tiny in arithmetic (one token's worth of matmuls) but must **read the entire KV cache and all model weights from GPU memory**. Decode is **memory-bandwidth-bound**, not compute-bound: the GPU spends its time moving data, and its expensive tensor cores sit mostly idle.

This is *the* central asymmetry of LLM serving:

| | Prefill | Decode |
|---|---|---|
| Tokens processed per pass | whole prompt (parallel) | 1 |
| Bottleneck | compute (FLOPs) | memory bandwidth |
| Grows with | prompt length | output length |
| Determines | TTFT | TPOT / inter-token latency |
| GPU utilization | high | low (single request) |

The decode phase being memory-bound and low-utilization is *exactly* why one-request-at-a-time wastes the GPU, and *exactly* why batching helps: multiple requests can share the same weight read. Hold that thought for the tradeoff section.

### Why the KV cache exists

Without a cache, generating token *N* would re-run attention over all *N* prior tokens from scratch — an \(O(N^2)\) blowup over a full sequence. The KV cache trades memory for compute: store each token's K and V once, reuse them for every future step. It turns per-step attention cost from "re-read and recompute everything" into "read the cache." The price is GPU memory that grows linearly with every token in every active request — which becomes the binding constraint on how many requests you can serve at once.

---

## Worked example 1: a minimal FastAPI + Transformers server

Here is a complete, correct, single-file server. It is deliberately naive — synchronous generation, one request at a time — so it exposes every pitfall we then discuss. This is the "before" picture for the whole book.

```python
# server.py
#   pip install fastapi "uvicorn[standard]" transformers torch accelerate
#   uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup, not per request.
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    STATE["tok"], STATE["model"] = tok, model
    yield
    STATE.clear()


app = FastAPI(lifespan=lifespan)


class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@torch.inference_mode()
def _generate(req: GenRequest) -> dict:
    tok, model = STATE["tok"], STATE["model"]
    messages = [{"role": "user", "content": req.prompt}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    prompt_len = inputs.shape[1]

    t0 = time.perf_counter()
    out = model.generate(
        inputs,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        temperature=req.temperature,
        top_p=req.top_p,
        pad_token_id=tok.pad_token_id,
    )
    dt = time.perf_counter() - t0

    new_tokens = out[0, prompt_len:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    n_out = new_tokens.shape[0]
    return {
        "text": text,
        "prompt_tokens": int(prompt_len),
        "output_tokens": int(n_out),
        "latency_s": round(dt, 3),
        "tokens_per_s": round(n_out / dt, 1),
    }


@app.post("/generate")
async def generate(req: GenRequest):
    # Blocking, CPU/GPU-bound work goes to a thread so it does not
    # freeze the async event loop (see pitfalls).
    return await run_in_threadpool(_generate, req)


@app.get("/healthz")
async def healthz():
    return {"ok": "model" in STATE}
```

Test it:

```bash
curl -s localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "Explain KV cache in one sentence.", "max_new_tokens": 64}'
```

What this server gets right, and what it deliberately does not:

- **Right:** model loaded once at startup (not per request); `@torch.inference_mode()` disables gradient bookkeeping; blocking work offloaded off the event loop; chat template + pad token set correctly; returns real token counts and throughput.
- **Deliberately wrong / naive:** it serves **one request at a time** per worker (the model is a shared object and `generate` holds the GPU), it does **not** stream tokens (no TTFT benefit for the client), and it does **no batching**. Two simultaneous callers queue behind each other. That is the motivation for everything after this chapter.

Streaming, when you add it, uses `TextIteratorStreamer` + a background thread and a FastAPI `StreamingResponse`, so the client gets the first token as soon as prefill finishes rather than waiting for the whole answer — a large *perceived* latency win with no throughput change.

---

## Generation parameters (what the knobs actually do)

`generate` is controlled by a `GenerationConfig`. The ones that matter for serving:

| Param | Effect | Note |
|-------|--------|------|
| `max_new_tokens` | hard cap on output length | The #1 latency and cost lever — decode time is ~linear in it. Always set it. |
| `do_sample` | greedy (`False`) vs sampling (`True`) | With `do_sample=False`, `temperature`/`top_p` are ignored and you get deterministic output. |
| `temperature` | flattens (`>1`) or sharpens (`<1`) the distribution | `0` is not literally valid for sampling; use greedy for determinism. |
| `top_p` (nucleus) | sample only from the smallest set of tokens summing to prob `p` | Common: `0.9`–`0.95`. |
| `top_k` | sample only from the `k` highest-prob tokens | Alternative/complement to `top_p`. |
| `repetition_penalty` | discourage repeating tokens | Helps loops; tune carefully. |
| `stop` / `eos_token_id` | stop conditions | Wrong EOS = runaway generation to `max_new_tokens`. |

A subtle correctness trap: if a caller passes `do_sample=False` **and** a non-default `temperature`, recent Transformers will warn that the sampling flags are ignored. Decide your server's contract explicitly rather than passing user knobs through blindly.

---



### Adding streaming (TTFT the user can feel)

The naive server returns the whole answer at once. Streaming emits tokens as they are decoded, so the client sees output right after prefill:

```python
from threading import Thread
from transformers import TextIteratorStreamer
from fastapi.responses import StreamingResponse

@app.post("/generate/stream")
async def generate_stream(req: GenRequest):
    tok, model = STATE["tok"], STATE["model"]
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": req.prompt}],
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(inputs=inputs, streamer=streamer,
                  max_new_tokens=req.max_new_tokens, do_sample=req.do_sample,
                  temperature=req.temperature, top_p=req.top_p,
                  pad_token_id=tok.pad_token_id)
    Thread(target=model.generate, kwargs=kwargs).start()  # runs off the event loop

    def emit():
        for piece in streamer:      # yields decoded text as tokens arrive
            yield piece
    return StreamingResponse(emit(), media_type="text/plain")
```

`generate` runs in a background thread and pushes tokens into the streamer; the handler yields them to the client. Same total work, dramatically better *perceived* latency — but note it still occupies the GPU serially. Streaming improves TTFT, not throughput.

---

## Worked example 2: GPU memory budget (weights + KV cache + activations)

You cannot reason about serving without the memory math. The GPU must simultaneously hold **model weights**, the **KV cache** for every in-flight request, and transient **activations**. Run out and you get a CUDA OOM — the most common production failure.

### Weights

\[
\text{weight bytes} = (\text{number of parameters}) \times (\text{bytes per parameter})
\]

For a 7-billion-parameter model in BF16:

\[
7 \times 10^{9} \ \text{params} \times 2 \ \text{bytes} = 14 \times 10^{9} \ \text{bytes} \approx 14 \ \text{GB}
\]

The rule of thumb "~2 GB per billion params in FP16/BF16" (and ~1 GB/B in INT8, ~0.5 GB/B in INT4) falls straight out of this.

### KV cache (per token, then per request)

The KV cache stores a key and a value vector for every token, every layer, every KV head:

\[
\text{bytes per token} = 2 \times L \times H_{kv} \times D_{h} \times b
\]

where the leading \(2\) covers **K and V**, \(L\) is the number of transformer layers, \(H_{kv}\) the number of key/value heads, \(D_{h}\) the head dimension, and \(b\) the bytes per element. For a LLaMA-style 7B model with \(L = 32\), \(H_{kv} = 32\), \(D_{h} = 128\), BF16 (\(b = 2\)):

\[
2 \times 32 \times 32 \times 128 \times 2 = 524{,}288 \ \text{bytes} \approx 0.5 \ \text{MB per token}
\]

For a request with a full context of 4,096 tokens:

\[
4096 \times 524{,}288 \ \text{bytes} = 2{,}147{,}483{,}648 \ \text{bytes} = 2 \ \text{GB}
\]

So a single 4K-context request costs **~2 GB of KV cache on top of the 14 GB of weights**. Note that models using grouped-query attention (GQA) have far fewer KV heads \(H_{kv}\) than query heads, which is *specifically* a trick to shrink this number — one reason modern models are cheaper to serve.

### Activations and overhead

Beyond weights and KV cache, each forward pass allocates transient **activation** tensors, and the CUDA context / allocator reserves a fixed slab (often 1–2 GB). Activation memory scales with batch size and sequence length but is freed between steps, so it is usually a smaller, bounded term than the KV cache — which only *grows*. When you size a GPU, budget weights + peak KV + a safety margin for activations and fragmentation; do not plan to use the last gigabyte.

### Putting it together on a 24 GB GPU

\[
\text{free for KV + activations} \approx 24 \ \text{GB} - 14 \ \text{GB (weights)} - \sim 1\text{–}2 \ \text{GB (activations, CUDA ctx)} \approx 8 \ \text{GB}
\]

At ~2 GB per 4K-token request, that GPU holds on the order of **4 concurrent full-length requests** before OOM — and fewer if prompts are longer. This single calculation is why KV-cache efficiency (PagedAttention, Chapter 5) is the highest-leverage optimization in LLM serving: memory, not compute, usually caps your concurrency.

---

## Metrics and the latency–throughput tradeoff

You cannot improve what you do not measure. Four numbers define an LLM endpoint.

| Metric | Definition | Formula |
|--------|------------|---------|
| **TTFT** (time to first token) | Prompt submitted -> first output token received. Dominated by queueing + prefill. | measured directly |
| **TPOT / ITL** (time per output token / inter-token latency) | Average gap between successive output tokens in the "steady stream." | \(\text{TPOT} = \dfrac{\text{E2E} - \text{TTFT}}{N_{out} - 1}\) |
| **E2E latency** | Request in -> full response out. | \(\text{E2E} = \text{TTFT} + \text{generation time}\) |
| **Throughput** | Tokens or requests completed per second, across all concurrent users. | \(\text{TPS} = \dfrac{N_{out}}{T_{last} - T_{first}}\), \(\ \text{RPS} = \dfrac{\text{completed requests}}{\text{time}}\) |

Definitions and formulas follow the Anyscale benchmarking guide (see Further Reading). Report **percentiles** (p50/p95/p99), never just the mean — tail latency is what users feel and what SLOs are written against.

### The tradeoff

Here is the crux, and it is why "just add batching" is not free:

- **A single request in isolation** gets the lowest possible latency: the whole GPU is devoted to it. But decode is memory-bound, so the GPU's compute units are ~idle — **terrible throughput per dollar.**
- **Batching many requests together** amortizes each weight/KV read across all of them: one memory pass serves \(B\) requests. Throughput (tokens/s, req/s) rises sharply — you use the idle compute. But any individual request may wait to be batched and shares GPU cycles, so **its TTFT and TPOT rise.**

So batch size is a dial between **latency** (small batch, low utilization, expensive per token) and **throughput** (large batch, high utilization, cheap per token, worse tail latency). There is no single right setting — it depends on your SLO. Real serving engines make this dial dynamic (continuous batching), which we introduce next and detail in Chapter 5.

---

### Worked example 3: reading a latency timeline

Numbers make the asymmetry concrete. Suppose a request has a 500-token prompt and generates 200 tokens, and we measure:

- prefill takes 120 ms (this *is* the TTFT, ignoring queueing),
- each decode step takes 15 ms.

Then:

\[
\text{TTFT} = 120 \ \text{ms}, \qquad
\text{generation} = 199 \times 15 \ \text{ms} \approx 2985 \ \text{ms}
\]

\[
\text{E2E} = 120 + 2985 \approx 3.1 \ \text{s}, \qquad
\text{TPOT} = \frac{3105 - 120}{200 - 1} \approx 15 \ \text{ms}, \qquad
\text{user TPS} = \frac{200}{2.985} \approx 67 \ \text{tok/s}
\]

Two lessons. First, **decode dominates E2E** here (~3 s vs ~120 ms) — output length is your biggest latency lever, which is why capping `max_new_tokens` matters so much. Second, if you *stream*, the user sees a token at 120 ms instead of waiting 3.1 s: identical work, far better perceived latency. Streaming trades nothing in throughput; it only reshapes when the user first sees output.

## Why one-request-at-a-time is wasteful — the road to batching

The naive server above processes requests serially. During each request's **decode** phase the GPU is memory-bandwidth-bound and its tensor cores are mostly idle — you are paying for an A100/H100 and using a fraction of its FLOPs. Meanwhile a second caller just *waits*.

Batching fixes this by running multiple sequences through the model together, so a single read of the weights (and a single scheduling step) produces a token for *every* request in the batch. Because decode was memory-bound, adding more requests is nearly free on compute up to a point — you convert idle compute into throughput.

There are three flavors, in increasing sophistication (full treatment in Chapter 5):

| Batching strategy | How it works | Weakness |
|-------------------|--------------|----------|
| **Static batching** | Collect \(N\) requests, pad to the same length, run them together to completion. | Head-of-line blocking: the whole batch waits for the slowest/longest sequence; padding wastes compute; new requests wait for the batch to finish. |
| **Dynamic batching** | Server briefly buffers incoming requests (a few ms) to form a batch, then runs it. Common in Triton. | Still runs the batch to completion; a short request is stuck behind a long one. |
| **Continuous batching** (a.k.a. in-flight / iteration-level) | The scheduler works at the granularity of a *single decode step*: finished sequences leave the batch and new ones join every iteration. | More complex; needs paged KV memory to do well. This is what vLLM and TGI do, and it is the big throughput unlock. |

The mental model: static/dynamic batching batches **requests**; continuous batching batches **token-generation steps**. The latter keeps the GPU full even when requests have wildly different lengths — which is the normal case. This is the single biggest reason a dedicated engine outperforms the naive server, often by an order of magnitude in throughput at the same latency.

---

## Failure modes and pitfalls

The naive server fails in predictable ways. Know them cold.

- **CUDA out of memory (OOM).** The #1 killer. Causes: model too big for the GPU in the chosen dtype; too many concurrent requests inflating the KV cache; a single very long prompt/output. Symptoms: `CUDA out of memory. Tried to allocate ...`. Fixes: smaller dtype/quantization, cap `max_new_tokens` and context length, limit concurrency, use an engine with paged KV. Do the memory math *before* deploying.
- **Blocking the async event loop.** FastAPI is async, but `model.generate()` is a long, synchronous, GPU-bound call. If you `await` it directly in the handler (or call it inline), it **freezes the entire event loop** — health checks time out, every other connection stalls. Fix: `run_in_threadpool` (as above) or a dedicated worker/queue. This bug looks like "the server randomly hangs under load."
- **No batching / serial serving.** Two users -> the second waits for the first. Throughput is capped at one request's worth of decode, and the expensive GPU sits underutilized. This is not a bug to fix in this server — it is the reason to graduate to vLLM/TGI.
- **Tokenizer mismatch / wrong chat template.** Using a tokenizer from a different model, skipping `apply_chat_template`, or missing special tokens produces fluent-looking garbage with **no error**. Always pair the exact tokenizer with the model and use the official chat template.
- **Padding-side and pad-token mistakes.** Right-padding a decoder-only batch, or a missing `pad_token`, silently corrupts batched generation. Left-pad; set `pad_token = eos_token` if absent.
- **FP32 by accident.** Forgetting `torch_dtype` loads in FP32 and doubles weight memory — an instant OOM on models that would fit fine in BF16.
- **Unbounded generation.** No `max_new_tokens` and a wrong/missing EOS -> the model runs until it hits some default cap, burning GPU time and blocking others. Always bound output.
- **Cold-start latency ignored.** First request after a scale-up waits for model download + load (seconds to minutes). Add readiness probes (`/healthz` gating on model-loaded) so traffic is not routed to a not-yet-ready pod (Chapters 3, 6).

---

## Tools comparison (brief — pointers to later chapters)

| | What it is | Batching | Best for | Covered in |
|---|-----------|----------|----------|-----------|
| **Raw Transformers + FastAPI** | Hand-rolled server (this chapter) | None (DIY) | Learning, prototypes, custom logic | This chapter |
| **vLLM** | High-throughput OSS inference engine | Continuous + PagedAttention | Throughput-critical OSS serving; OpenAI-compatible API | Chapter 5 |
| **TGI** (Text Generation Inference) | Hugging Face's production server (Rust + Python) | Continuous, tensor-parallel sharding | HF ecosystem, Inference Endpoints | referenced Ch. 5 |
| **Triton Inference Server** | NVIDIA multi-framework server | Dynamic batching; pairs with TensorRT-LLM / vLLM backends | Multi-model, mixed workloads, tight NVIDIA stack | Chapter 12 |

Rule of thumb: **build the raw server once to understand the loop, then never ship it.** For anything real, reach for an engine that does continuous batching and paged KV memory. TorchServe is another general-purpose model server in this space, but for LLMs specifically the KV-cache-aware engines (vLLM, TGI, TensorRT-LLM) win decisively.

---

## Production checklist — what an interviewer probes

1. **"Walk me through a request."** Expected: request -> tokenize (+ chat template) -> prefill (compute-bound, sets TTFT) -> decode loop reusing KV cache (memory-bound, sets TPOT) -> detokenize -> response. Naming the prefill/decode split is the tell that you actually understand serving.
2. **"How much GPU memory does model X need?"** Expected: weights = params × bytes/param (~2 GB/B in FP16); plus KV cache = \(2 \times L \times H_{kv} \times D_h \times b\) per token per request; plus activations/overhead. Know the ~0.5 MB/token, ~2 GB-per-4K-request order of magnitude.
3. **"Your p99 latency is bad but the GPU is at 30% util — why?"** Expected: decode is memory-bandwidth-bound and you are serving serially / with tiny batches; add continuous batching to convert idle compute into throughput.
4. **"How do you trade latency for throughput?"** Expected: batch size is the dial; larger batches amortize weight reads (higher throughput) at the cost of per-request TTFT/TPOT; pick per SLO; use continuous batching to get most of the throughput without static batching's head-of-line blocking.
5. **"What's your OOM story?"** Expected: memory math up front; cap `max_new_tokens` and context; limit concurrency; quantize; paged KV; monitor KV-cache utilization, not just GPU memory.
6. **"Why not just `await model.generate()` in the handler?"** Expected: it blocks the event loop; offload to a threadpool/worker or use an engine with its own scheduler.
7. **"Which metrics do you alert on?"** Expected: TTFT, TPOT/ITL, E2E — all at p50/p95/p99 — plus throughput (tok/s, req/s), queue depth, KV-cache utilization, and error/OOM rate. Percentiles, not means.
8. **"When do you reach for vLLM/TGI/Triton over your own server?"** Expected: as soon as you need concurrency — continuous batching + paged KV are hard to build well and are the whole point; roll your own only to learn or for genuinely custom logic.

---

## Further reading

- Hugging Face — Text generation tutorial (`generate`): <https://huggingface.co/docs/transformers/en/llm_tutorial>
- Hugging Face — Generation config / `GenerationConfig` reference: <https://huggingface.co/docs/transformers/en/main_classes/text_generation>
- FastAPI documentation: <https://fastapi.tiangolo.com/>
- Text Generation Inference (TGI) docs: <https://huggingface.co/docs/text-generation-inference/en/index>
- TGI source and features: <https://github.com/huggingface/text-generation-inference>
- vLLM — PagedAttention & continuous batching overview: <https://www.runpod.io/articles/guides/vllm-pagedattention-continuous-batching>
- NVIDIA — Mastering LLM Techniques: Inference Optimization (KV cache, batching): <https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/>
- Anyscale — Understand LLM latency and throughput metrics (TTFT/TPOT/ITL/TPS/RPS): <https://docs.anyscale.com/llm/serving/benchmarking/metrics>
- KV cache memory calculation (worked example): <https://mbrenndoerfer.com/writing/kv-cache-memory-calculation-llm-inference-gpu>
- Pierre Lienhart — LLM Inference Series: KV caching, a deeper look: <https://medium.com/@plienhar/llm-inference-series-4-kv-caching-a-deeper-look-4ba9a77746c8>
- NVIDIA Triton Inference Server: <https://github.com/triton-inference-server/server>

---

> **Where this goes next:** Chapter 2 containerizes this server; Chapter 4 load-tests it to *see* the serial bottleneck; Chapter 5 replaces it with vLLM and continuous batching to fix everything this chapter exposed.
