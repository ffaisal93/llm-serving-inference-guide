# Basic LLM Serving — A Deep Dive

> Standing up a language model behind an HTTP API from first principles, and understanding *why* the naive version is slow — then hardening it into something you could actually put in front of traffic, and defending every decision in an interview.

## Why this matters

Every production LLM system — ChatGPT, a support bot, a code assistant — is at bottom a loop that turns text into tokens, runs a forward pass, and turns tokens back into text, wrapped in a network server. If you understand that loop end to end, and understand the two costs that dominate it (compute and GPU memory), the rest of this book is just engineering to make the loop cheaper and more concurrent.

This chapter builds the loop the *obvious* way: one model, one process, one request at a time. That version works, and it is a perfect teaching tool precisely because it is slow. By the end you will be able to say, with numbers, exactly where the time and the memory go — and that motivates batching, KV-cache management, and dedicated engines (vLLM, TGI, Triton) in the chapters that follow.

We then go further than "it works on my laptop": we add concurrency limits, structured error handling, and health/readiness endpoints, load-test the result, and walk through two real incidents that this kind of naive-but-hardened server either causes or prevents. By the end of this chapter you should be able to build a small serving stack yourself *and* survive a senior interviewer asking "walk me through what happens when this GPU gets 50 requests per second."

We keep the intuition first and the mechanism precise. Where there is a tradeoff, we name it honestly.

**How to use this chapter.** Read "Core intuition" through "Failure modes" in order the first time — it is one continuous argument from the autoregressive loop to why naive serving breaks. "Production case studies," "The 2025–2026 landscape," and "Interview mastery" are meant to be revisited independently: before a design review, before an interview, or after an incident, to re-anchor on the mechanism that explains it.

> **Saying it out loud.** So every LLM product you've ever used is, underneath, the same small loop: text goes in, gets chopped into tokens, the model runs a forward pass, and a token comes back out — over and over until it decides to stop. Everything that makes serving hard is just that loop being expensive in two specific ways: it burns GPU compute, and it eats GPU memory. If you can say where the time goes and where the bytes go, you can explain batching, KV caching, and why anybody bothers with vLLM. The reason I'd build the naive version first is that it's slow in a *diagnosable* way — you can point at the exact millisecond and the exact gigabyte, and that's what turns "vLLM is faster" into an argument instead of a slogan.

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

> **Saying it out loud.** The one thing to internalize is that a language model only ever predicts the *next* token — generation is just calling that in a loop and feeding the output back in. Two consequences fall straight out. First, it's inherently sequential: a 200-token answer means at least 200 forward passes, so your latency scales with how long the answer is, not how clever the model is. Second, without a cache the model re-reads its entire context on every single step, which is quadratic waste — and the fix for that, the KV cache, is exactly what ends up eating your GPU memory. So the two costs that dominate serving, latency and memory, both fall out of that one sentence about next-token prediction.

---

## Loading a model: weights, dtype, device, tokenizer

Before serving anything you load four things. Each has a failure mode.

> **Saying it out loud.** Loading a model looks like one line of code, but there are really four decisions in it and each one has a way of biting you. You're picking which weights, at what numeric precision, on what device, with which tokenizer. The precision choice is the biggest single memory lever you'll ever pull — BF16 instead of FP32 literally halves your weight footprint. And the tokenizer is the sneaky one, because a mismatched tokenizer or a skipped chat template doesn't throw an error, it just quietly produces fluent garbage. The failure mode I'd name is forgetting `torch_dtype`: you silently load in FP32, double your weights, and OOM on a model that would have fit fine.

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

`safetensors` is not an incidental detail — it is the format almost every serious model ships in today, and it is worth knowing *why* (we return to this with dates and sources in "The 2025–2026 landscape" below): the older PyTorch default, `pickle` (`.bin`/`.pt` checkpoints), can execute arbitrary code on load because `pickle.load` deserializes by calling constructors named in the file — a well-known supply-chain attack vector for downloaded model weights. `safetensors` stores only raw tensor bytes plus a JSON header of shapes/dtypes, so loading it can never execute code, and because it is a flat memory-mappable layout, loading is also faster (`mmap` + lazy paging instead of unpickling). If you see `pytorch_model.bin` instead of `model.safetensors` in a repo today, treat it as a legacy artifact.

### dtype / precision — the single biggest memory lever

Parameters are stored as floating-point numbers, and the *bytes per parameter* is a choice you make at load time.

| dtype | bytes/param | Typical use |
|-------|-------------|-------------|
| FP32  | 4 | Rarely for inference; training reference |
| FP16 / BF16 | 2 | Standard inference precision on GPU |
| INT8  | 1 | Quantized inference (small quality hit) |
| INT4 / NF4 | ~0.5 | Aggressive quantization, edge/consumer GPUs |

BF16 (bfloat16) is usually preferred over FP16 on modern GPUs: same 2 bytes, but a wider exponent range, so it is less prone to overflow/`NaN` during the forward pass. FP32 doubles your memory for almost no inference quality gain — do not serve in FP32 by accident (it is the default if you forget `torch_dtype`).

> **Saying it out loud.** Precision is just how many bytes you spend storing each parameter, and it maps directly to gigabytes on the card. FP32 is four bytes, FP16 and BF16 are two, INT8 is one, INT4 is about a half — so a 7-billion-parameter model is 28 GB, 14 GB, 7 GB, or 3.5 GB depending purely on that one choice. BF16 is the modern default over FP16 because it costs the same two bytes but has a wider exponent range, so you're far less likely to hit overflow or NaNs mid forward pass. The tradeoff to name: dropping to INT8 or INT4 buys you memory and concurrency, but it's a quality hit you have to *measure* on your own eval set — it's never free, and "quantization is lossless" is the wrong answer.

### Device placement

Weights must sit in GPU memory (VRAM) for fast inference. `device_map="cuda"` puts everything on one GPU; `device_map="auto"` will *shard* across multiple GPUs or spill to CPU/disk if the model does not fit — convenient, but CPU offload is catastrophically slow for serving. For a serving path you want the whole model resident on the GPU and you want to *know* it fits (memory math below).

### Tokenizer — small, and a classic source of silent bugs

The tokenizer maps text <-> integer IDs. It must be the *exact* one the model was trained with; a mismatch produces garbage output with no error. Two things to get right in a server:

- **Chat template.** Instruct/chat models expect a specific formatting of roles (`<|user|>`, `<|assistant|>`, etc.). Use `tokenizer.apply_chat_template(messages, add_generation_prompt=True)` rather than hand-concatenating strings — getting the special tokens wrong quietly degrades quality.
- **Padding side and pad token.** For batched generation, decoder-only models must **left-pad** (`tokenizer.padding_side = "left"`), and many models ship without a `pad_token` — set `tokenizer.pad_token = tokenizer.eos_token`. Right-padding a decoder batch corrupts the generation. (We revisit padding when we build real batching.)

> **Saying it out loud.** The tokenizer is the boring part that causes the scariest bugs, because when it's wrong nothing crashes — the model just gets nonsense and confidently answers it. Two things I'd check every time. One, use the model's own chat template rather than gluing role strings together by hand, because instruct models are trained on very specific special tokens and getting them subtly wrong degrades quality invisibly. Two, for batched generation on a decoder-only model you must left-pad, and lots of models ship without a pad token so you set it to the EOS token. The named failure mode is right-padding a decoder batch: no error, no warning, just quietly corrupted output for every request in the batch.

---

## Mechanism in depth: prefill vs decode, and the KV cache

This is the heart of the chapter. A single generation request has **two phases with completely different performance characteristics.**

> **Saying it out loud.** If there's one thing worth saying unprompted in an interview, it's that a request has two phases with completely different bottlenecks, not one. Prefill runs your whole prompt through the model in a single parallel pass — that's compute-bound, and it's what sets time-to-first-token. Then decode generates one token at a time, and each of those tiny steps still has to drag the entire model weights and the whole growing KV cache across GPU memory, so decode is memory-bandwidth-bound. That asymmetry is the reason your GPU can sit at low utilization while your latency is terrible, and it's why the fix is batching rather than a bigger card.

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

> **Saying it out loud.** Here's the thing that surprises people: decode does almost no math. Each step only pushes one new token through the model, because every earlier token's keys and values are already cached — so the arithmetic is trivial, but you still have to read all fourteen-plus gigabytes of weights plus the whole KV cache out of GPU memory to do it. That's what "memory-bandwidth-bound" means: the tensor cores are idle, waiting on the memory bus. Practically, that's why a single request wastes an H100 — you're paying for roughly 989 teraflops of dense BF16 compute and using a sliver of it — and it's exactly why batching works, because one memory read can serve every request in the batch at once.

### Why the KV cache exists

Without a cache, generating token *N* would re-run attention over all *N* prior tokens from scratch — an \(O(N^2)\) blowup over a full sequence. The KV cache trades memory for compute: store each token's K and V once, reuse them for every future step. It turns per-step attention cost from "re-read and recompute everything" into "read the cache." The price is GPU memory that grows linearly with every token in every active request — which becomes the binding constraint on how many requests you can serve at once.

> **Saying it out loud.** The KV cache is a straight memory-for-compute trade. Without it, generating token N means re-running attention over all N previous tokens from scratch, which is quadratic over a full sequence and completely wasteful, since those keys and values never change once computed. So you compute each token's key and value once and keep them around. The catch is that the cache grows linearly with every token in every active request and it only ever grows during a request — so on a typical 7B model it's about half a megabyte per token, meaning one 4K-context request is roughly 2 GB. That's why concurrency in LLM serving is almost always capped by memory, not by FLOPs.

### The 60-second version (say this in an interview)

If you only remember one paragraph, make it this one — it is the answer to "explain prefill and decode" under time pressure:

> "A request has two phases. Prefill processes the whole prompt in one parallel forward pass — it's compute-bound, and its cost sets time-to-first-token. Decode then generates one token at a time, each step only running the new token through the model because past tokens' key/value vectors are cached — that's the KV cache. Decode is memory-bandwidth-bound, not compute-bound, because each tiny step still has to read the full model weights and the whole growing KV cache off the GPU. That's why decode is inefficient for a single request — the GPU's compute sits idle waiting on memory — and it's exactly why batching multiple requests' decode steps together helps: one memory read now produces tokens for many requests at once. And it's why KV cache memory, which grows every step, is usually the real limit on concurrency, not raw compute."

That is roughly 60 seconds spoken aloud and it hits every point an interviewer is listening for: the two phases, their bottlenecks, what each determines (TTFT vs TPOT), why batching works, and why memory — not FLOPs — usually caps you.

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

> **Saying it out loud.** The minimal server is maybe forty lines and the important thing is what it gets right versus what it deliberately doesn't. It loads the model once at startup instead of per request, it wraps generation in inference mode so PyTorch isn't tracking gradients, and — the one people miss — it pushes the blocking generate call off to a thread pool. That last one matters because `model.generate()` is a long synchronous call, and if you run it directly inside an async handler it freezes the entire event loop, including your own health check endpoint. What it deliberately doesn't do is batch or stream, so two callers just queue behind each other — and that serial bottleneck is the whole motivation for every chapter after this one.

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

A related, increasingly common knob that does **not** appear in the table above because it is not a `GenerationConfig` field: **constrained / structured decoding**, where the server forces every generated token to come from a grammar (a JSON Schema, a regex, a context-free grammar) rather than the raw vocabulary. We cover why this matters and how it works mechanically in "The 2025–2026 landscape" below, because it has become a default expectation for tool-calling and JSON-emitting endpoints, not a niche feature.

> **Saying it out loud.** Most of the sampling knobs are quality dials, but one of them is a cost dial and that's the one I'd lead with: `max_new_tokens` is your single biggest latency and money lever, because decode time is basically linear in output length. Temperature and top-p just reshape the probability distribution you sample from, and if you set `do_sample=False` they're ignored entirely — you get deterministic greedy output. The stop condition is the quiet danger: a wrong or missing EOS token means the model runs all the way to the cap on every request, burning GPU time and blocking other callers. So the rule is always bound output length, and bound it server-side, because a client's `max_new_tokens` should be a request, not a command.

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

> **Saying it out loud.** Streaming means you push each token to the client as it's decoded instead of waiting for the whole answer, and the honest framing is that it's a *perceived* latency win, not a real one. Same total work, same throughput, same moment the last token lands — but the user sees something at, say, 120 milliseconds instead of staring at a spinner for three seconds. Mechanically you run generate on a background thread that pushes tokens into a streamer, and the handler yields them out as a streaming response. The tradeoff to name: streaming improves TTFT and nothing else — it does not free up the GPU, so under load a streaming server serializes exactly as badly as a non-streaming one.

### Constrained decoding by hand (what "structured outputs" actually costs you without an engine)

Section A below explains that hosted APIs and dedicated engines now offer schema-guaranteed JSON as a first-class feature, implemented as grammar-constrained decoding: at every decode step, mask out logits for any token that would violate the grammar. It is worth seeing the mechanism on the raw server, because it makes concrete exactly how much an engine is doing for you for free. Here is the simplest possible version — forcing the model to only ever emit digits and a decimal point (a toy "grammar," but the mechanism generalizes to a full JSON Schema automaton):

```python
import torch
from transformers import LogitsProcessor, LogitsProcessorList

class DigitsOnlyLogitsProcessor(LogitsProcessor):
    # Mask every token whose decoded text contains a character outside
    # "0123456789." -- a minimal stand-in for a compiled JSON-Schema/regex
    # automaton. Real constrained decoding (Outlines, XGrammar) precomputes
    # which token IDs are valid at each automaton state so this mask is a
    # cheap lookup, not a per-step string scan like this toy version.

    def __init__(self, tokenizer, allowed_chars=set("0123456789. ")):
        self.tokenizer = tokenizer
        self.allowed_chars = allowed_chars
        self._valid_ids = None  # lazily computed once, then reused every step

    def _compute_valid_ids(self, vocab_size: int) -> torch.Tensor:
        valid = torch.zeros(vocab_size, dtype=torch.bool)
        for tok_id in range(vocab_size):
            text = self.tokenizer.decode([tok_id])
            if all(c in self.allowed_chars for c in text) and text != "":
                valid[tok_id] = True
        return valid

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._valid_ids is None:
            self._valid_ids = self._compute_valid_ids(scores.shape[-1]).to(scores.device)
        scores = scores.masked_fill(~self._valid_ids, float("-inf"))
        return scores


# Wire it into generate() as an extra constraint on top of everything else:
processors = LogitsProcessorList([DigitsOnlyLogitsProcessor(tok)])
out = model.generate(inputs, max_new_tokens=16, logits_processor=processors,
                      pad_token_id=tok.pad_token_id)
```

This toy processor decodes every candidate token's *text* on every step to check it, which is far too slow for a real vocabulary of 50k+ tokens at production latency — that per-step cost is exactly the engineering problem XGrammar and `outlines` solve, by precompiling the grammar into an automaton once and reducing each decode step's mask computation to following one transition and reading a precomputed bitmask of valid token IDs for the current state, rather than re-deriving validity from scratch. The takeaway to carry into an interview: structured output is not prompt engineering, it is a `LogitsProcessor` (or the engine's equivalent) applied every single decode step, and its cost and correctness both hinge on how the grammar-to-token-mask compilation is done.

> **Saying it out loud.** When an API promises you guaranteed valid JSON, that's not a better prompt — it's constrained decoding. At every single decode step, the schema gets compiled into a state machine over the vocabulary, and every token that would break the grammar has its logit set to negative infinity before you sample. So schema-valid output is true by construction, not by retrying until the JSON parses. The cost is real, though: you're computing a valid-token mask on every step, and if you do that naively by decoding all fifty thousand vocabulary entries and string-matching, you've just made yourself the bottleneck. That's exactly the problem XGrammar and Outlines solve — precompile the grammar once, then each step is a state transition and a bitmask lookup.

---

## Build it in practice — extended: from "works on my laptop" to "survives real traffic"

The server above is correct but has three gaps between it and something you would actually deploy: it has **no concurrency control** (a hundred simultaneous callers all get admitted and race for the GPU, and the process has no idea how loaded it is), **no structured error handling** (a CUDA OOM or a bad request produces an ugly 500 with a stack trace instead of a machine-readable error the caller can act on), and **no readiness signal separate from liveness** (a load balancer cannot tell "the process is up" from "the process is ready to take more work"). None of these require an inference engine to fix — they are ordinary backend engineering, and skipping them is exactly what produces the incidents in the war-stories section below.

> **Saying it out loud.** There are three things standing between a correct server and a deployable one, and none of them need an inference engine — they're just ordinary backend engineering. You need admission control, so a hundred simultaneous callers don't all get let in to race for a GPU that fits four. You need structured errors, so a caller can tell "retry me later" from "your request is malformed." And you need liveness and readiness as two separate signals, so a load balancer can stop sending traffic to a busy pod without Kubernetes killing it. Skip these and you get exactly the two incidents at the end of this chapter: a healthy pod restarted by its own health check, and an unbounded OOM that takes down unrelated requests with it.

### 1. Bound concurrency to what the GPU can actually hold

Do not let every accepted request race for the GPU unbounded. Gate concurrent generations with a semaphore sized from the **memory math you will do in Worked Example 2** below — not a guess — and shed load past a bounded queue depth instead of accepting requests you cannot serve in time.

```python
# server.py (extended)
#   pip install fastapi "uvicorn[standard]" transformers torch accelerate httpx
#   uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
import asyncio
import time
from contextlib import asynccontextmanager
from enum import Enum

import torch
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# Sized from the KV-cache math in Worked Example 2, NOT a guess: on a 24 GB
# GPU with this model we computed room for ~4 full-length concurrent
# requests before OOM. Leave headroom for shorter, cheaper requests too.
MAX_CONCURRENT_GENERATIONS = 3
MAX_QUEUE_DEPTH = 16          # requests allowed to wait for a GPU slot
REQUEST_TIMEOUT_S = 30.0      # end-to-end deadline, queue + generation
SERVER_MAX_NEW_TOKENS = 512   # hard server-side cap — never trust the client's value alone

STATE: dict = {}


class ErrorCode(str, Enum):
    OVERLOADED = "overloaded"
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    BAD_REQUEST = "bad_request"
    INTERNAL = "internal_error"


@asynccontextmanager
async def lifespan(app: FastAPI):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    STATE["tok"], STATE["model"] = tok, model
    STATE["gpu_gate"] = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
    STATE["in_flight"] = 0
    STATE["queued"] = 0
    STATE["ready"] = True     # flips readyz on; healthz is independent (below)
    yield
    STATE["ready"] = False
    STATE.clear()


app = FastAPI(lifespan=lifespan)


class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
```

> **Saying it out loud.** The rule is that concurrency should be a number you derived, not a number you guessed. You do the memory math — weights plus KV cache per request plus a margin — and that tells you how many generations actually fit; then a semaphore holds you at or below it, and a bounded queue holds a few more waiting. Anything past that gets a fast 503 instead of being admitted into a fight it can't win. The reason to reject rather than queue forever is that an unbounded queue doesn't remove overload, it just hides it — it turns a capacity problem into runaway process memory, and the client's timeout fires anyway. On a 24 GB card running a 7B model in BF16, that number lands around three or four, not thirty.

### 2. Structured error handling — turn crashes into contracts

A caller that gets a bare 500 with a Python traceback cannot tell "retry me" from "your request is malformed" from "the server is out of capacity forever." Classify failures and return a small, stable error shape:

```python
def _generate(req: GenRequest) -> dict:
    tok, model = STATE["tok"], STATE["model"]
    n = min(req.max_new_tokens, SERVER_MAX_NEW_TOKENS)  # server has the last word

    messages = [{"role": "user", "content": req.prompt}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    prompt_len = inputs.shape[1]

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            inputs, max_new_tokens=n, do_sample=req.do_sample,
            temperature=req.temperature, top_p=req.top_p,
            pad_token_id=tok.pad_token_id,
        )
    dt = time.perf_counter() - t0

    new_tokens = out[0, prompt_len:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    n_out = new_tokens.shape[0]
    return {
        "text": text, "prompt_tokens": int(prompt_len), "output_tokens": int(n_out),
        "latency_s": round(dt, 3), "tokens_per_s": round(n_out / dt, 1) if dt > 0 else 0.0,
    }


def _generate_safe(req: GenRequest) -> dict:
    """Runs in the threadpool. Classify every failure into a stable error code
    so callers can distinguish 'retry later' from 'fix your request'."""
    try:
        return _generate(req)
    except torch.cuda.OutOfMemoryError:
        # Free whatever we can; do NOT try to keep serving this request.
        torch.cuda.empty_cache()
        return {"_error": ErrorCode.OUT_OF_MEMORY.value,
                "detail": "GPU ran out of memory; retry with a shorter prompt or lower max_new_tokens"}
    except ValueError as e:
        return {"_error": ErrorCode.BAD_REQUEST.value, "detail": str(e)}
    except Exception as e:  # last-resort classification, still structured
        return {"_error": ErrorCode.INTERNAL.value, "detail": str(e)}
```

> **Saying it out loud.** A bare 500 with a Python traceback tells the caller nothing useful — they can't tell whether to retry, to fix their input, or to give up. So you classify every failure into a small, stable set of codes and map them onto meaningful HTTP statuses: out-of-memory is different from a bad request, which is different from "we're at capacity, come back with backoff." Concretely, you catch `torch.cuda.OutOfMemoryError` explicitly, free what you can, and return something the client can act on rather than letting it escape as a generic 500. The payoff is that your retry logic, your load balancer, and your alerting all get an honest signal — and the failure mode you're avoiding is clients hammering retries against an error that will never succeed.

### 3. The gated, backpressured endpoint

Wire the semaphore, the bounded queue, and the deadline together. The key design decision: **a full queue fails fast with 503 rather than growing without bound.** An unbounded queue does not remove the overload — it just hides it, converts it into ballooning process memory, and lets client timeouts fire anyway once the queue is long enough. Bounded queues plus fast rejection is a deliberate load-shedding choice, and it is what lets the rest of your system (a load balancer, a retry-with-backoff client, an autoscaler) actually react to the overload signal.

```python
@app.post("/generate")
async def generate(req: GenRequest):
    if STATE["queued"] >= MAX_QUEUE_DEPTH:
        return JSONResponse(status_code=503, content={
            "error": ErrorCode.OVERLOADED.value,
            "detail": "server at capacity, retry with backoff",
        })

    STATE["queued"] += 1
    acquired = False
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT_S):  # Python 3.11+
            await STATE["gpu_gate"].acquire()
            acquired = True
            STATE["queued"] -= 1
            STATE["in_flight"] += 1
            result = await run_in_threadpool(_generate_safe, req)
    except TimeoutError:
        return JSONResponse(status_code=504, content={
            "error": ErrorCode.TIMEOUT.value,
            "detail": f"exceeded {REQUEST_TIMEOUT_S}s waiting for a GPU slot or generating",
        })
    finally:
        if acquired:
            STATE["in_flight"] -= 1
            STATE["gpu_gate"].release()
        else:
            STATE["queued"] -= 1

    if "_error" in result:
        status = 507 if result["_error"] == ErrorCode.OUT_OF_MEMORY.value else \
                  400 if result["_error"] == ErrorCode.BAD_REQUEST.value else 500
        return JSONResponse(status_code=status, content=result)
    return result
```

Walk through the accounting carefully, because it is easy to get this subtly wrong (and a subtly wrong version is worse than none — it lies about server load): `queued` is incremented the moment a request is admitted past the queue-depth check, and decremented **either** when it successfully acquires a GPU slot (`acquired = True` path) **or** in the `finally` block if it times out or errors out before acquiring. `in_flight` is only ever touched once a slot is actually held, and is always released in `finally`, so a timeout, an exception inside `_generate_safe`, or a clean return all leave the semaphore and counters consistent. This is the difference between a load number you can trust for autoscaling and alerting, and one that slowly drifts until a restart papers over it.

> **Saying it out loud.** Backpressure is just being honest about capacity: when the queue is full you say no immediately instead of accepting work you can't finish. The bookkeeping is where people get it subtly wrong — you increment "queued" the moment you admit a request past the depth check, decrement it either when it grabs a GPU slot or in the finally block if it times out first, and you only ever touch "in flight" once a slot is actually held. Get that wrong and the counters drift upward until a restart papers over it, and now your autoscaler and your dashboards are lying to you. The tradeoff worth naming: a fast 503 looks worse on an error-rate graph than a slow success, but it's the only thing that gives a retrying client, a load balancer, and an autoscaler something real to react to.

### 4. Health vs readiness — two different questions

Kubernetes (and any sane load balancer) asks two different questions and expects two different answers. **Liveness** ("is the process alive enough to not be killed and restarted?") should be cheap and should *not* depend on the model or the GPU — otherwise a temporarily saturated but perfectly healthy process gets killed mid-batch, which is its own incident (see war stories). **Readiness** ("should traffic be routed to this pod right now?") should reflect real capacity: model loaded, and — ideally — not already saturated.

```python
@app.get("/healthz")
async def healthz():
    # Liveness: process can respond at all. Deliberately does NOT touch the
    # model, the GPU, or the semaphore — must stay cheap even while the GPU
    # is fully busy, or a liveness probe timeout kills a perfectly healthy pod.
    return {"status": "alive"}


@app.get("/readyz")
async def readyz():
    # Readiness: is there a loaded model AND spare capacity worth routing to?
    if not STATE.get("ready", False):
        return JSONResponse(status_code=503, content={"ready": False, "reason": "model not loaded"})
    saturated = STATE["queued"] >= MAX_QUEUE_DEPTH
    body = {
        "ready": not saturated,
        "in_flight": STATE["in_flight"],
        "queued": STATE["queued"],
        "capacity": MAX_CONCURRENT_GENERATIONS,
    }
    return JSONResponse(status_code=200 if not saturated else 503, content=body)
```

Point the Kubernetes `livenessProbe` at `/healthz` with a generous period, and the `readinessProbe` at `/readyz` so the Service stops sending new traffic to a saturated pod (letting the autoscaler add capacity) without killing it.

> **Saying it out loud.** These are two genuinely different questions and conflating them causes outages. Liveness asks "is this process broken enough that I should kill and restart it?" — so it must be dirt cheap and must not touch the model or the GPU, because a saturated-but-perfectly-healthy pod that fails its liveness probe gets killed mid-request. Readiness asks "should I route new traffic here right now?" — so that one *should* reflect real load: model loaded, queue not full. The crucial asymmetry is what happens when each fails: a readiness failure just stops new traffic and lets in-flight work finish, while a liveness failure destroys everything currently running on that pod. If you take one thing: never let your liveness endpoint depend on GPU state.

### 5. Load-test it — see the bottleneck, don't just believe in it

A claim like "this server serializes under concurrency" should be a measurement, not folklore. This is a minimal concurrent load test using `httpx`; it fires `N_REQUESTS` through `CONCURRENCY` simultaneous clients and reports the standard percentiles from the metrics section below.

```python
# loadtest.py
#   pip install httpx
import asyncio
import time

import httpx

URL = "http://localhost:8000/generate"
N_REQUESTS = 30
CONCURRENCY = 10


async def one_request(client: httpx.AsyncClient, i: int) -> dict:
    t0 = time.perf_counter()
    r = await client.post(
        URL, json={"prompt": f"Count to five. (req {i})", "max_new_tokens": 64}, timeout=60.0
    )
    return {"status": r.status_code, "latency_s": time.perf_counter() - t0}


async def main():
    sem = asyncio.Semaphore(CONCURRENCY)

    async def bound(client, i):
        async with sem:
            return await one_request(client, i)

    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*(bound(client, i) for i in range(N_REQUESTS)))
        wall = time.perf_counter() - t0

    lat = sorted(r["latency_s"] for r in results)
    ok = sum(r["status"] == 200 for r in results)
    shed = sum(r["status"] == 503 for r in results)

    def pct(p):
        return lat[min(len(lat) - 1, int(p * len(lat)))]

    print(f"requests={N_REQUESTS} concurrency={CONCURRENCY} wall_s={wall:.2f} "
          f"observed_rps={N_REQUESTS / wall:.2f}")
    print(f"ok={ok} shed_503={shed}")
    print(f"p50={pct(0.50):.2f}s  p95={pct(0.95):.2f}s  p99={pct(0.99):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
```

Run the server, then in another shell `python loadtest.py`. With `MAX_CONCURRENT_GENERATIONS = 3` and `CONCURRENCY = 10`, expect to *see* the serialization directly: p50 will be close to a single request's latency, but p95/p99 will be roughly \(3\times\) to \(4\times\) higher, because the 4th-through-10th concurrent caller queue behind the semaphore for one or more full generation cycles before their own request even starts. If you push `N_REQUESTS`/`CONCURRENCY` high enough to exceed `MAX_QUEUE_DEPTH`, you will start seeing `shed_503 > 0` — the bounded queue doing exactly its job instead of the process quietly ballooning. That gap between p50 and p99 **is** the naive server's serial bottleneck, measured instead of asserted — and it is the number continuous batching (Chapter 5) exists to close.

> **Saying it out loud.** "This server serializes under load" should be a measurement, not folklore — and it takes about thirty lines to prove. You fire N requests through a fixed number of concurrent clients and report p50, p95, and p99, never the average, because the average is exactly the statistic that hides a starved tail. What you'll see with a concurrency gate of three and ten simultaneous callers is a p50 near a single request's latency and a p99 roughly three to four times higher, because callers four through ten are waiting through entire generation cycles before theirs even starts. That gap between p50 and p99 *is* the serial bottleneck, and closing it is precisely what continuous batching exists to do.

### 6. Confirm the error contract by hand before trusting the load test

Before trusting the load test's numbers, confirm each error path returns the structured shape you designed rather than a bare stack trace — this is a five-minute check that catches most of Section B's wiring mistakes immediately:

```bash
# Healthy request -- expect 200 with text/prompt_tokens/output_tokens/latency_s
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "Say hi.", "max_new_tokens": 8}'

# Readiness under normal load -- expect 200 with in_flight/queued/capacity
curl -s localhost:8000/readyz

# Liveness -- expect 200 instantly, even while /generate is busy elsewhere
curl -s localhost:8000/healthz

# Force capacity exhaustion: fire more concurrent requests than
# MAX_CONCURRENT_GENERATIONS + MAX_QUEUE_DEPTH allow, and expect some 503s
# with {"error": "overloaded", ...} rather than hung connections
for i in $(seq 1 40); do
  curl -s -o /dev/null -w '%{http_code} ' localhost:8000/generate \
    -H 'content-type: application/json' \
    -d '{"prompt": "Write a long story.", "max_new_tokens": 512}' &
done; wait; echo
```

If the overload run above returns only `200`s and hung connections instead of a mix of `200` and `503`, the bounded-queue accounting in Section B's `/generate` handler is not actually shedding load — go back and re-check the `queued`/`in_flight` bookkeeping before you trust anything the load test reports.

---

## Worked example 2: GPU memory budget (weights + KV cache + activations)

You cannot reason about serving without the memory math. The GPU must simultaneously hold **model weights**, the **KV cache** for every in-flight request, and transient **activations**. Run out and you get a CUDA OOM — the most common production failure, and the exact one Section B's `MAX_CONCURRENT_GENERATIONS` and Section C's second war story both revolve around.

> **Saying it out loud.** The memory budget is three terms and you should be able to do it on a whiteboard. Weights are parameters times bytes per parameter — roughly two gigabytes per billion parameters in BF16, so a 7B model is about 14 GB. Then KV cache, which is per token, per request, and only grows: about half a megabyte per token on a 7B model, so a 4K-context request costs around 2 GB all by itself. Then a gigabyte or two for activations and CUDA overhead. On a 24 GB card that leaves you about eight gigabytes of headroom, which is four concurrent full-length requests — and that number, not intuition, is where your concurrency limit comes from. The takeaway an interviewer wants: memory caps concurrency, not compute.

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

So a single 4K-context request costs **~2 GB of KV cache on top of the 14 GB of weights**. Note that models using grouped-query attention (GQA) have far fewer KV heads \(H_{kv}\) than query heads, which is *specifically* a trick to shrink this number — one reason modern models are cheaper to serve. As a concrete comparison: if the same architecture used full multi-head attention with \(H_{kv}\) equal to 32 query heads unchanged, doubling \(H_{kv}\) doubles KV-cache bytes per token linearly — GQA with, say, 8 KV heads instead of 32 shrinks the same request's KV cache fourfold, from 2 GB to 0.5 GB, which is the difference between fitting 4 concurrent long requests and fitting 16.

> **Saying it out loud.** The KV cache formula is two — for keys and values — times layers, times KV heads, times head dimension, times bytes per element, and that gives you bytes per token. For a classic 7B model with 32 layers, 32 KV heads, head dim 128, in BF16, that works out to about half a megabyte per token, so a 4,096-token request is roughly 2 GB. The interesting term is the KV head count, because grouped-query attention deliberately shrinks it — dropping from 32 KV heads to 8 cuts that request's cache fourfold, from 2 GB down to 0.5 GB. That's the difference between fitting four concurrent long requests and fitting sixteen on the same card, which is why GQA is a serving-cost decision, not a model-card footnote.

### Activations and overhead

Beyond weights and KV cache, each forward pass allocates transient **activation** tensors, and the CUDA context / allocator reserves a fixed slab (often 1–2 GB). Activation memory scales with batch size and sequence length but is freed between steps, so it is usually a smaller, bounded term than the KV cache — which only *grows*. When you size a GPU, budget weights + peak KV + a safety margin for activations and fragmentation; do not plan to use the last gigabyte.

### Putting it together on a 24 GB GPU

\[
\text{free for KV + activations} \approx 24 \ \text{GB} - 14 \ \text{GB (weights)} - \sim 1\text{–}2 \ \text{GB (activations, CUDA ctx)} \approx 8 \ \text{GB}
\]

At ~2 GB per 4K-token request, that GPU holds on the order of **4 concurrent full-length requests** before OOM — and fewer if prompts are longer. This is precisely where `MAX_CONCURRENT_GENERATIONS = 3` in Section B's server came from — a deliberately conservative choice below the theoretical ceiling of ~4, to leave headroom for activation spikes and fragmentation rather than run at the edge. This single calculation is why KV-cache efficiency (PagedAttention, Chapter 5) is the highest-leverage optimization in LLM serving: memory, not compute, usually caps your concurrency.

> **Saying it out loud.** Put the numbers together on a real card: 24 GB total, minus 14 for BF16 weights, minus a gig or two for activations and the CUDA context, leaves you about eight gigabytes for KV cache. At roughly 2 GB per 4K-token request, that's about four concurrent full-length requests — and you'd actually set your limit at three, because running at the theoretical edge means fragmentation and activation spikes will OOM you eventually. That's the whole argument for PagedAttention in one calculation: if memory is what caps concurrency, then using memory more efficiently is the highest-leverage optimization in the entire stack, worth more than any scheduler tuning you'll do.

---

### Quantization changes the concurrency budget, not just the weight size

The same 24 GB-GPU calculation above assumed BF16 weights. Quantization is the other big lever, and it is worth running the numbers once so the tradeoff is concrete rather than a slogan ("quantize to fit more"). Take the same 7B model, same 24 GB GPU, same ~2 GB-per-4K-token-request KV cache (quantizing weights does not by itself shrink KV-cache dtype unless you separately quantize the cache):

\[
\text{INT8 weights} = 7 \times 10^{9} \times 1 \ \text{byte} = 7 \ \text{GB}, \qquad
\text{free for KV} \approx 24 - 7 - 1.5 \approx 15.5 \ \text{GB}
\]

At ~2 GB per request, that is room for roughly **7–8 concurrent full-length requests**, up from ~4 in BF16 — nearly double the concurrency on identical hardware, at the cost of some generation quality (INT8 post-training quantization is usually a small, workload-dependent quality hit; validate it against your own eval set rather than assuming it is free). Push to INT4 and weights drop to ~3.5 GB, freeing room for on the order of 15+ concurrent requests on the same GPU. This is the concrete version of the "quantization vs. smaller model" interview question below: quantizing a 7B model to INT8 and running a smaller, unquantized 3B model in BF16 can land at similar total memory, but they are not equivalent — quantization keeps the larger model's capability at a precision cost, while a smaller model changes the capability itself. Which one wins depends on whether your bottleneck is quality or throughput per request, and the only way to know is to measure both against your task.

> **Saying it out loud.** People pitch quantization as "make the model smaller," but the interesting effect is second-order: shrinking the weights frees room for KV cache, and KV cache is what caps concurrency. Same 24 GB card, same 7B model — BF16 weights take 14 GB and leave room for about four concurrent requests; INT8 takes 7 GB and leaves room for seven or eight; INT4 takes about 3.5 GB and gets you past fifteen. So you roughly quadrupled your concurrency on identical hardware. The tradeoff you have to name honestly is quality: INT8 post-training quantization is usually a small hit, but it's workload-dependent, and the only way to know is your own eval set. And note that quantizing weights doesn't shrink the KV cache unless you separately quantize the cache too.

---

## Metrics and the latency–throughput tradeoff

You cannot improve what you do not measure. Four numbers define an LLM endpoint.

| Metric | Definition | Formula |
|--------|------------|---------|
| **TTFT** (time to first token) | Prompt submitted -> first output token received. Dominated by queueing + prefill. | measured directly |
| **TPOT / ITL** (time per output token / inter-token latency) | Average gap between successive output tokens in the "steady stream." | \(\text{TPOT} = \dfrac{\text{E2E} - \text{TTFT}}{N_{out} - 1}\) |
| **E2E latency** | Request in -> full response out. | \(\text{E2E} = \text{TTFT} + \text{generation time}\) |
| **Throughput** | Tokens or requests completed per second, across all concurrent users. | \(\text{TPS} = \dfrac{N_{out}}{T_{last} - T_{first}}\), \(\ \text{RPS} = \dfrac{\text{completed requests}}{\text{time}}\) |

Definitions and formulas follow the Anyscale benchmarking guide (see Further Reading). Report **percentiles** (p50/p95/p99), never just the mean — tail latency is what users feel and what SLOs are written against. This is exactly what the `loadtest.py` script in Section B reports, and exactly why it reports percentiles rather than an average: an average latency can look fine while p99 callers are being starved behind the semaphore.

> **Saying it out loud.** Four numbers define an LLM endpoint and you should rattle them off. Time to first token — how long until the user sees anything, dominated by queueing plus prefill. Time per output token, the gap between successive tokens once it's streaming. End-to-end latency, which is just TTFT plus generation. And throughput, tokens or requests per second across everyone. The rule that separates a senior answer from a junior one is: always report p50, p95, and p99, never the mean — because an average latency can look perfectly healthy while your p99 callers are starving in a queue, and your SLO is written against the tail, not the middle.

### The tradeoff

Here is the crux, and it is why "just add batching" is not free:

- **A single request in isolation** gets the lowest possible latency: the whole GPU is devoted to it. But decode is memory-bound, so the GPU's compute units are ~idle — **terrible throughput per dollar.**
- **Batching many requests together** amortizes each weight/KV read across all of them: one memory pass serves \(B\) requests. Throughput (tokens/s, req/s) rises sharply — you use the idle compute. But any individual request may wait to be batched and shares GPU cycles, so **its TTFT and TPOT rise.**

So batch size is a dial between **latency** (small batch, low utilization, expensive per token) and **throughput** (large batch, high utilization, cheap per token, worse tail latency). There is no single right setting — it depends on your SLO. Real serving engines make this dial dynamic (continuous batching), which we introduce next and detail in Chapter 5.

> **Saying it out loud.** Batch size is a dial between latency and throughput, and there's no universally right setting — it depends entirely on your SLO. Turn it down and a single request gets the whole GPU to itself: lowest possible latency, but since decode is memory-bound the compute units sit idle, so it's terrible throughput per dollar. Turn it up and one read of the weights produces a token for every request in the batch, so throughput climbs sharply — but any individual request now waits to be batched and shares cycles, so its TTFT and per-token latency get worse. The important nuance: naive batching genuinely *hurts* single-request latency, which is why real engines make the dial dynamic with continuous batching instead of picking one static number.

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

> **Saying it out loud.** Put real numbers on it: 500-token prompt, 200 tokens out, prefill takes 120 milliseconds and each decode step takes 15. So your TTFT is 120 milliseconds, but generation is 199 more steps at 15 milliseconds each — about three seconds. End to end you're at roughly 3.1 seconds, and decode is over ninety-five percent of it. Two things fall out. Output length is by far your biggest latency lever, which is why capping `max_new_tokens` matters more than almost any tuning. And if you stream, the user sees a token at 120 milliseconds instead of three seconds — identical work, completely different experience.

## Why one-request-at-a-time is wasteful — the road to batching

The naive server above processes requests serially. During each request's **decode** phase the GPU is memory-bandwidth-bound and its tensor cores are mostly idle — you are paying for an A100/H100 and using a fraction of its FLOPs. Meanwhile a second caller just *waits*.

Batching fixes this by running multiple sequences through the model together, so a single read of the weights (and a single scheduling step) produces a token for *every* request in the batch. Because decode was memory-bound, adding more requests is nearly free on compute up to a point — you convert idle compute into throughput.

There are three flavors, in increasing sophistication (full treatment in Chapter 5):

| Batching strategy | How it works | Weakness |
|-------------------|--------------|----------|
| **Static batching** | Collect \(N\) requests, pad to the same length, run them together to completion. | Head-of-line blocking: the whole batch waits for the slowest/longest sequence; padding wastes compute; new requests wait for the batch to finish. |
| **Dynamic batching** | Server briefly buffers incoming requests (a few ms) to form a batch, then runs it. Common in Triton. | Still runs the batch to completion; a short request is stuck behind a long one. |
| **Continuous batching** (a.k.a. in-flight / iteration-level) | The scheduler works at the granularity of a *single decode step*: finished sequences leave the batch and new ones join every iteration. | More complex; needs paged KV memory to do well. This is what vLLM and TGI do, and it is the big throughput unlock. |

The mental model: static/dynamic batching batches **requests**; continuous batching batches **token-generation steps**. The latter keeps the GPU full even when requests have wildly different lengths — which is the normal case. This is the single biggest reason a dedicated engine outperforms the naive server, often by an order of magnitude in throughput at the same latency. Section B's concurrency gate is a poor person's version of this: it prevents catastrophe, but it still runs each admitted request's decode loop independently rather than sharing steps across requests — the semaphore buys you safety, not the throughput win of continuous batching.

> **Saying it out loud.** Serving one request at a time means that during decode your expensive GPU is mostly waiting on memory while a second caller just sits there — you're renting tensor cores and using a fraction of them. Batching fixes it because one read of the weights produces a token for every sequence in the batch, and since decode was memory-bound, adding sequences is nearly free on compute up to a point. There are three flavors: static batching pads a fixed group and runs it to completion, dynamic batching buffers arrivals for a few milliseconds then does the same, and continuous batching schedules at the level of a single decode step, so a finished sequence leaves and a new one joins every iteration. The failure mode the first two share is head-of-line blocking — one 2,000-token answer holds the whole batch hostage — and continuous batching is what eliminates it, often for an order of magnitude more throughput at the same latency.

---

## Failure modes and pitfalls

The naive server fails in predictable ways. Know them cold.

- **CUDA out of memory (OOM).** The #1 killer. Causes: model too big for the GPU in the chosen dtype; too many concurrent requests inflating the KV cache; a single very long prompt/output. Symptoms: `CUDA out of memory. Tried to allocate ...`. Fixes: smaller dtype/quantization, cap `max_new_tokens` and context length, limit concurrency, use an engine with paged KV. Do the memory math *before* deploying.
- **Blocking the async event loop.** FastAPI is async, but `model.generate()` is a long, synchronous, GPU-bound call. If you `await` it directly in the handler (or call it inline), it **freezes the entire event loop** — health checks time out, every other connection stalls. Fix: `run_in_threadpool` (as above) or a dedicated worker/queue. This bug looks like "the server randomly hangs under load" — see the first war story below for exactly how this plays out in production.
- **No batching / serial serving.** Two users -> the second waits for the first. Throughput is capped at one request's worth of decode, and the expensive GPU sits underutilized. This is not a bug to fix in this server — it is the reason to graduate to vLLM/TGI.
- **Tokenizer mismatch / wrong chat template.** Using a tokenizer from a different model, skipping `apply_chat_template`, or missing special tokens produces fluent-looking garbage with **no error**. Always pair the exact tokenizer with the model and use the official chat template.
- **Padding-side and pad-token mistakes.** Right-padding a decoder-only batch, or a missing `pad_token`, silently corrupts batched generation. Left-pad; set `pad_token = eos_token` if absent.
- **FP32 by accident.** Forgetting `torch_dtype` loads in FP32 and doubles weight memory — an instant OOM on models that would fit fine in BF16.
- **Unbounded generation.** No `max_new_tokens` and a wrong/missing EOS -> the model runs until it hits some default cap, burning GPU time and blocking others. Always bound output — and bound it server-side, not just as a client-supplied default (Section B's `SERVER_MAX_NEW_TOKENS`).
- **Unbounded concurrency / unbounded queues.** Accepting every connection and either racing them all for the GPU or queueing them forever converts a capacity problem into a memory problem or a client-timeout problem, one level removed. Gate concurrency from real memory math and shed load explicitly (Section B).
- **Cold-start latency ignored.** First request after a scale-up waits for model download + load (seconds to minutes). Add readiness probes (`/healthz` gating on model-loaded) so traffic is not routed to a not-yet-ready pod (Chapters 3, 6) — and keep liveness probes independent of load, or a saturated-but-healthy pod gets killed (Section B, war story one).

> **Saying it out loud.** If you asked me what actually breaks these servers, I'd name three in order. CUDA out-of-memory is number one, and it's almost always unbounded concurrency or unbounded output length rather than the model itself being too big. Second is blocking the async event loop by calling `model.generate()` directly in an async handler — that freezes every other coroutine in the process including your health checks, and it presents as "the server randomly hangs under load." Third is the silent class: a mismatched tokenizer, a skipped chat template, or right-padding a decoder batch, all of which produce fluent-looking garbage with no error at all. The pattern is that the loud failures are the easy ones — it's the silent ones that ship to production.

---

## Production case studies & war stories

Theory is cheap; here are two incidents shaped exactly like ones teams hit in production, and the lesson each one teaches. Both are avoidable with the hardening in Section B, which is why that section exists.

### War story 1: the health check that killed a healthy pod

**Setup.** A team shipped a version of the naive server close to Worked Example 1, but with one shortcut: the `/generate` handler called `model.generate()` directly inside the `async def` function instead of routing it through `run_in_threadpool`. It passed every test — a single curl request worked fine, latency looked right, and it sailed through staging, which only ever sent one request at a time.

**What happened in production.** Once real traffic arrived — five or six concurrent users, nothing exotic — requests started stacking up, and every few minutes the pod would restart. On-call saw `502`s from the load balancer and Kubernetes events showing `Liveness probe failed` followed by a container restart, which looked like a memory leak or a crash. It was neither. `model.generate()` is a long-running, synchronous, CPU-orchestrated call; running it directly in an `async def` handler blocks Python's single event loop for the entire generation — often several seconds. While it was blocked, the event loop could not service *any* other coroutine, including the `/healthz` liveness endpoint on the same process. Kubelet's liveness probe timed out, decided the process was unresponsive, and killed it mid-request — dropping every in-flight generation, including ones that had nothing to do with the blocked request.

**Diagnosis.** A `py-spy dump` against the running process during a slow period showed every worker thread's Python-level stack sitting inside `torch.nn.functional` deep under `model.generate`, with the asyncio event loop task for `/healthz` never getting scheduled. That is the smoking gun for a blocked event loop: the liveness endpoint's *code* is fine, it simply never runs.

**Lesson.** Never run long, synchronous, CPU/GPU-bound work directly inside an `async def` handler — always route it through `run_in_threadpool` (as in Worked Example 1) or a separate process/worker pool. Just as important: **keep liveness probes independent of the workload.** `/healthz` should answer instantly regardless of GPU saturation (Section B's version does, deliberately); it is `/readyz` that should reflect load, and unlike a liveness failure, a readiness failure only stops *new* traffic — it does not kill in-flight work. Conflating the two, or letting the workload block the endpoint that answers either, turns ordinary load into a self-inflicted outage.

> **Saying it out loud.** This one's my favorite because the bug and the symptom look nothing alike. A team called `model.generate()` directly inside an async handler — worked perfectly in staging, which only ever sent one request at a time. In production with five or six concurrent users, pods started restarting every few minutes with liveness probe failures, which everyone read as a memory leak. It wasn't: generate is a long synchronous call, so it blocked Python's single event loop for seconds at a time, and the health endpoint living on that same loop simply never got scheduled. Kubelet saw no response, concluded the process was dead, and killed it mid-generation — taking down every unrelated in-flight request too. Two lessons: never run blocking work inside an async handler, and never let your liveness probe depend on the workload.

### War story 2: the silent OOM from an unbounded batch

**Setup.** A different team's server accepted arbitrary concurrent requests with no admission control, and let clients pass their own `max_new_tokens` and prompt length with no server-side cap — reasoning that "the model will just run a bit slower" under load. There was no memory math behind that assumption; nobody had run the calculation in Worked Example 2 for their actual GPU and model.

**What happened in production.** A marketing push drove a burst of traffic, much of it from a batch-import script sending long documents with `max_new_tokens: 4096` set on every call, all fired concurrently. Each concurrent request's KV cache grew independently and kept growing every decode step; there was nothing gating how many of these could run at once. Partway through the spike, the CUDA allocator ran out of memory mid-batch. Because there was no isolation between requests, the failure did not stay contained to the offending calls: some in-flight requests raised `CUDA out of memory` immediately, others returned corrupted or truncated output as the allocator scrambled to free fragmented memory, and a few hung entirely until the process was restarted — losing every request that happened to be in flight at that moment, including well-behaved ones from unrelated callers.

**Diagnosis.** Post-incident, the team ran the memory math from Worked Example 2 against their actual model and GPU for the first time and discovered the theoretical safe concurrency for full-length, full-`max_new_tokens` requests was around 4 — they had been letting 30-plus requests race for the GPU simultaneously with no cap, and it had simply not yet coincided with a burst large and long enough to blow the budget.

**Lesson.** Two independent fixes, both from Section B, and both necessary: (1) **cap `max_new_tokens` and effective context length server-side**, regardless of what the client requests — `SERVER_MAX_NEW_TOKENS` in Worked Example 1's extended server exists precisely so a client's number is a request, not a command; and (2) **gate concurrency using the KV-cache math from Worked Example 2**, not intuition — the semaphore in Section B turns "the model will just run a bit slower" into "the fourth-plus concurrent caller waits in a bounded queue or gets a clean 503," which is a survivable, observable failure mode instead of an unbounded, correlated one. As a standing practice: monitor **KV-cache/GPU-memory utilization** as a first-class metric next to request count and latency — request count alone hid the real signal here until the GPU had already run out.

> **Saying it out loud.** A team let clients pass their own `max_new_tokens` with no server-side cap and no admission control, on the theory that "it'll just run a bit slower." Then a batch-import script fired long documents with `max_new_tokens: 4096` all at once. Each request's KV cache grew independently every decode step until the allocator ran dry mid-batch — and because nothing isolated requests from each other, it wasn't contained: some raised OOM, some returned truncated output, some hung until restart, and well-behaved callers went down with them. Post-mortem, they ran the memory math for the first time and found their safe concurrency was about four; they'd been running thirty-plus. Two fixes, both necessary: cap output length server-side, and derive your concurrency limit from the KV-cache math rather than intuition.

---

## Tools comparison (brief — pointers to later chapters)

| | What it is | Batching | Best for | Covered in |
|---|-----------|----------|----------|-----------|
| **Raw Transformers + FastAPI** | Hand-rolled server (this chapter) | None (DIY, or a semaphore gate as in Section B) | Learning, prototypes, custom logic | This chapter |
| **vLLM** | High-throughput OSS inference engine | Continuous + PagedAttention | Throughput-critical OSS serving; OpenAI-compatible API | Chapter 5 |
| **TGI** (Text Generation Inference) | Hugging Face's production server (Rust + Python) | Continuous, tensor-parallel sharding | HF ecosystem, Inference Endpoints | referenced Ch. 5 |
| **Triton Inference Server** | NVIDIA multi-framework server | Dynamic batching; pairs with TensorRT-LLM / vLLM backends | Multi-model, mixed workloads, tight NVIDIA stack | Chapter 12 |

Rule of thumb: **build the raw server once to understand the loop, then never ship it as-is.** For anything real, reach for an engine that does continuous batching and paged KV memory. TorchServe is another general-purpose model server in this space, but for LLMs specifically the KV-cache-aware engines (vLLM, TGI, TensorRT-LLM) win decisively. The hardened version in Section B narrows — but does not close — that gap: it makes the naive server *safe* to run under real traffic (bounded concurrency, structured errors, correct health signals), which is table stakes for any service; it does not give you continuous batching's throughput, which requires the engine-level scheduling covered in Chapter 5.

> **Saying it out loud.** My honest rule is: build the raw server once so you understand the loop, then never ship it as-is. For anything with real concurrency you want an engine that does continuous batching and paged KV memory — vLLM if you want throughput and an OpenAI-compatible API, TGI if you're deep in the Hugging Face ecosystem, Triton if you're serving many models across frameworks in an NVIDIA shop. Hardening the raw server the way we just did makes it *safe* — bounded concurrency, structured errors, honest health signals — but safe isn't the same as fast. It still runs one request's decode loop per slot, so it never gets the throughput win, and that gap is the actual reason you migrate, not "vLLM is industry standard."

---

## The 2025–2026 landscape

Everything above is timeless mechanism — prefill/decode, the KV cache, memory math — but the tooling and API surface around it moves fast. Here is what "basic serving" looks like against the current landscape, with sources, so you can reason about where the raw-server pattern in this chapter still fits and where the industry has moved on.

### The default artifact format is `safetensors`, and the reason is security, not just speed

By 2025–2026, `safetensors` (<https://github.com/huggingface/safetensors>) is the default distribution format on the Hugging Face Hub and in production runtimes (TGI's own docs describe why: <https://huggingface.co/docs/text-generation-inference/main/en/conceptual/safetensors>). The mechanism, worth restating precisely because it comes up in interviews: PyTorch's legacy checkpoint format is a `pickle` blob, and `pickle.load` is not a passive data format — it can invoke arbitrary Python callables encoded in the file, which is a documented remote-code-execution vector for a checkpoint downloaded from an untrusted source. `safetensors` stores only tensor bytes and a JSON metadata header, so there is no code path from "load this file" to "execute arbitrary code," and because the layout is flat and contiguous, it also loads faster via `mmap`. If your serving pipeline still ingests raw `.bin`/`.pt` files from third parties without conversion, that is a supply-chain gap worth flagging, not a stylistic preference.

> **Saying it out loud.** Everyone assumes safetensors won on speed, and the speed is real, but the actual reason is security. PyTorch's old checkpoint format is a pickle blob, and unpickling isn't passive parsing — it can call arbitrary Python constructors named inside the file, which is a documented remote-code-execution path for any weights you downloaded from someone else. Safetensors stores only raw tensor bytes plus a JSON header of shapes and dtypes, so there's simply no code path from "load this file" to "execute something." The speed comes free as a side effect, because a flat contiguous layout can be memory-mapped instead of deserialized. Practical version: if your pipeline still ingests third-party `.bin` or `.pt` files without conversion, that's a supply-chain gap, not a style preference.

### Native structured outputs are now a first-class API feature, not a prompting trick

As of 2024–2026, "ask nicely for JSON" has been replaced by API-level guarantees. OpenAI's Structured Outputs (announced 2024, current docs at <https://developers.openai.com/api/docs/guides/structured-outputs>, original post at <https://openai.com/index/introducing-structured-outputs-in-the-api/>) let you pass a JSON Schema and get a response *guaranteed* to satisfy it — no missing keys, no hallucinated enum values, no manual retry-on-invalid-JSON loop. Anthropic and Google expose comparable schema-constrained/tool-argument guarantees on their current APIs. Mechanically, this is not a smarter prompt — it is **constrained (grammar-guided) decoding**: the schema is compiled into a finite-state or pushdown automaton over the tokenizer's vocabulary, and at every decode step the set of grammatically-valid next tokens is computed and every other token's logit is masked to \(-\infty\) before sampling. The open-source engines implement this directly — vLLM's structured-outputs backends (<https://docs.vllm.ai/en/latest/features/structured_outputs/>) support both the `outlines` library and XGrammar, a purpose-built constrained-decoding engine (paper: <https://arxiv.org/pdf/2411.15100>) chosen for compiling grammars fast enough to mask logits every single decode step without becoming the bottleneck. The practical implication for this chapter's naive server: adding structured outputs to `model.generate()` directly means wiring a `LogitsProcessor` that does this masking yourself (Transformers exposes the hook, but you own the FSM); a dedicated engine gives you `response_format={"type": "json_schema", ...}` for free. This is a concrete, common reason teams move off the raw server well before they need continuous batching.

> **Saying it out loud.** "Please respond in JSON" is dead as an engineering technique — every major API now takes a JSON Schema and guarantees the response satisfies it. Under the hood it's constrained decoding: the schema compiles into an automaton over the tokenizer's vocabulary, and at each decode step every token that would violate the grammar gets its logit masked to negative infinity before sampling. So you get no missing keys, no invented enum values, and no retry-until-it-parses loop. The catch is that this cost lands on *every* decode step, which is why engines use purpose-built compilers like XGrammar rather than re-validating from scratch — and it's a very common reason teams leave the raw server long before they ever need continuous batching.

### Prompt / context caching is now standard, and it changes the economics of "basic serving"

All three major hosted APIs now cache repeated prompt prefixes so you do not pay full price to re-process a system prompt or a large shared context on every call:

- **Anthropic prompt caching** (<https://platform.claude.com/docs/en/build-with-claude/prompt-caching>): you mark a `cache_control` breakpoint; a cache write costs a multiplier over the base input rate (roughly 1.25× for a 5-minute TTL, 2× for a 1-hour TTL), while a cache *read* costs roughly 0.1× the base rate — a ~90% discount on tokens the model has already "seen" in cache. Minimum cacheable prefix length is model-dependent (from roughly 1,024 tokens up to 4,096 on smaller models); below that, caching is silently skipped, so check the response's `cache_read_input_tokens` / `cache_creation_input_tokens` fields rather than assuming.
- **OpenAI automatic prompt caching** (<https://developers.openai.com/api/docs/guides/prompt-caching>, announced at <https://openai.com/index/api-prompt-caching/>): caching is automatic above roughly a 1,024-token prefix, keyed by an exact-prefix hash of (typically) the first portion of the prompt, and routed by a `prompt_cache_key` so repeated calls land on a machine likely already holding the cached prefix; cached content is retained on the order of minutes to about an hour depending on traffic and model.
- **Gemini context caching** (<https://ai.google.dev/gemini-api/docs/caching>): supports both *implicit* caching (automatic, enabled by default on current Gemini models) and *explicit* caching (you create and manage a cache object yourself, useful for a large shared document or system context reused across many calls), again with a per-model minimum token threshold before caching activates.

Why this matters for a chapter about a raw, self-hosted server: this is precisely the *prefix-reuse* problem that a naive `model.generate()` loop does nothing about — every call re-runs prefill from scratch, even if 90% of the prompt (a shared system prompt, a long set of tool definitions, a large retrieved context) is byte-identical to the previous call. Self-hosted engines solve the same problem under a different name — **prefix caching / automatic prefix reuse** (vLLM's implementation is often discussed alongside RadixAttention-style prefix trees) — by keeping the KV cache for a previously-seen prefix resident and reusing it instead of recomputing prefill. If your workload has long, repeated system prompts or tool schemas, prefix caching is frequently a bigger win than raw batching throughput, and it is a capability the raw server in this chapter simply does not have.

> **Saying it out loud.** If your prompts share a long prefix — a big system prompt, a pile of tool definitions, a retrieved document — you're re-running prefill on identical bytes every single call, and everybody now has a fix for that. On hosted APIs it's prompt caching: Anthropic charges roughly 1.25x base rate to write a cache entry and about 0.1x to read it, so a cache hit is around a ninety percent discount on those tokens; OpenAI does it automatically above about a 1,024-token prefix. Self-hosted engines call the same idea prefix caching — they keep the previously computed KV entries resident instead of recomputing them. The thing to say: this is a completely separate axis from batching, and for prefix-heavy workloads it's often the bigger win — and it's a capability the raw `generate()` loop simply does not have.

### The shift from raw `generate()` to production runtimes is now the norm, not the exception

The direction of travel across 2025–2026 write-ups on production LLM serving (see, e.g., the state-of-the-field surveys and deployment guides in Further Reading) is consistent: teams prototype against `transformers.generate()` and move to a dedicated runtime — vLLM, TGI, TensorRT-LLM, or SGLang — as soon as concurrency, latency SLOs, or cost per token matter, because continuous batching plus paged KV memory is very hard to reproduce well by hand, and because these engines have absorbed structured-output support, prefix caching, speculative decoding, and multi-GPU sharding as built-in features rather than bespoke code. That does not make this chapter's raw server obsolete knowledge — it is the mental model every one of those engines optimizes *against* — but it does mean the honest scope of "basic serving" in 2026 is: understand the loop and the memory math cold, harden it enough to survive real traffic if you must run it (Section B), and know precisely which of its gaps (batching, prefix reuse, structured decoding) are the reasons you reach for vLLM/TGI/Triton rather than vague "it's faster" hand-waving.

> **Saying it out loud.** The industry pattern is consistent: prototype on `transformers.generate()`, then move to a real runtime — vLLM, TGI, TensorRT-LLM, SGLang — the moment concurrency, latency SLOs, or cost per token start to matter. The reason isn't that engines are magically faster; it's that continuous batching plus paged KV memory is genuinely hard to reproduce by hand, and those engines have already absorbed structured outputs, prefix caching, speculative decoding, and multi-GPU sharding as built-in features instead of bespoke code you maintain. That doesn't make the raw server useless knowledge — it's the mental model those engines are optimizing against. What it does mean is you should be able to say exactly which gap pushed you over, not hand-wave that "it's faster."

### Where "basic serving" still fits

None of the above obsoletes this chapter — it contextualizes it. A raw, hardened `generate()` server is still the right answer when: you need custom, non-standard generation logic an engine does not expose; you are serving a small number of concurrent users where continuous batching's throughput gains do not matter; you are prototyping or teaching; or you are running a model on hardware/software combinations the major engines do not yet support well. The 2025–2026 baseline expectation, though, is that you can articulate *exactly* why you are not using vLLM/TGI in a given case — "we don't need it yet, here's the math" — rather than not knowing the alternative exists.

> **Saying it out loud.** The raw hardened server is still the right call in a few real cases: you need custom generation logic no engine exposes, you're serving a handful of concurrent users where batching gains don't materialize, you're prototyping or teaching, or you're on hardware the big engines don't support well. What's changed by 2026 isn't that this pattern is wrong — it's that the burden of proof moved. The expectation now is that you can articulate *why* you're not on vLLM with actual numbers, something like "our peak is four concurrent users, continuous batching buys us nothing at that scale, here's the math." Not knowing the alternative exists is the answer that loses you the interview; deliberately deferring it with a reason is the answer that wins.

---

## Interview mastery

This section replaces and substantially expands the old "production checklist." It is organized as: a bank of Q&A with model answers, a system-design prompt worked end to end, and a red-flags-vs-green-flags table you can use to grade yourself or a candidate.

### Q&A bank

1. **"Walk me through a request."** Request -> tokenize (+ chat template) -> prefill (compute-bound, sets TTFT) -> decode loop reusing the KV cache (memory-bound, sets TPOT) -> detokenize -> response. Naming the prefill/decode split unprompted is the single strongest signal you understand serving; see the 60-second answer above for the full version.
2. **"How much GPU memory does model X need?"** Weights = params × bytes/param (~2 GB/B in FP16/BF16); plus KV cache = \(2 \times L \times H_{kv} \times D_h \times b\) per token per request; plus activations/overhead. Know the order of magnitude — ~0.5 MB/token, ~2 GB per 4K-token request for a 7B model — and be able to say why GQA shrinks the KV term specifically.
3. **"Your p99 latency is bad but the GPU is at 30% util — why?"** Decode is memory-bandwidth-bound and you are serving serially or with tiny batches; the compute units sit idle waiting on memory traffic. Add continuous batching to convert that idle compute into throughput; a semaphore-gated raw server (Section B) prevents overload but does not fix this — it still runs one request's decode loop at a time per slot.
4. **"How do you trade latency for throughput?"** Batch size is the dial: larger batches amortize weight reads across more requests (higher throughput) at the cost of per-request TTFT/TPOT, because a given request now waits on or shares cycles with others. Pick the setting from your SLO; continuous batching gets most of the throughput without static batching's head-of-line blocking.
5. **"What's your OOM story?"** Memory math up front (Worked Example 2), server-side caps on `max_new_tokens`/context (never trust client values alone), concurrency gated to a number derived from that math (not intuition), quantization as a lever, paged KV in a real engine. Monitor KV-cache/GPU-memory utilization as a first-class metric — see war story 2 for what happens when nobody does.
6. **"Why not just `await model.generate()` in the handler?"** It blocks the single-threaded asyncio event loop for the whole generation, freezing every other coroutine on that process — including your own health checks. Offload to a threadpool/worker, or use an engine with its own scheduler. See war story 1 for the exact failure this produces in production.
7. **"Which metrics do you alert on?"** TTFT, TPOT/ITL, E2E — all at p50/p95/p99 — plus throughput (tok/s, req/s), queue depth, KV-cache/GPU-memory utilization, and error/OOM rate. Percentiles, not means; a healthy-looking average can hide starved tail requests.
8. **"When do you reach for vLLM/TGI/Triton over your own server?"** As soon as you need real concurrency: continuous batching plus paged KV are hard to build well and are the whole point of those engines. Roll your own to learn, for genuinely custom logic, or for small enough traffic that batching's gains don't matter — and be able to say which of those applies to you.
9. **"Static vs dynamic vs continuous batching — what's the actual difference?"** Static and dynamic batching both batch at the *request* level and run the batch to completion, so a short request can be stuck behind a long one (head-of-line blocking). Continuous batching operates at the *decode-step* level: a finished sequence leaves the batch and a new one joins on the very next iteration, so the GPU stays full even when request lengths vary wildly — which is the normal case in production.
10. **"Explain GQA's effect on serving, not just training."** Grouped-query attention reduces the number of KV heads \(H_{kv}\) relative to query heads. Since KV-cache bytes per token scale linearly in \(H_{kv}\), this directly shrinks the KV cache per token — e.g., dropping from 32 to 8 KV heads cuts KV-cache memory 4×, which multiplies directly into how many concurrent requests fit on a given GPU. It is a serving-cost decision baked into the architecture, not just an efficiency footnote from the model card.
11. **"Does speculative decoding belong in 'basic serving'?"** It's an optimization layered on top of the decode loop: a small draft model proposes several tokens, the full model verifies them in one parallel forward pass, and correct guesses are accepted for free, cutting the number of full-model decode steps needed. It doesn't change the memory story (you still need the KV cache for both models) and it adds real scheduling complexity, so in practice it shows up in dedicated engines, not the raw server in this chapter — know what it is and why it doesn't fit here yet.
12. **"How does prompt caching change things at the API-provider level, and does the raw server in this chapter get that benefit?"** No — a naive `model.generate()` loop re-runs prefill from scratch on every call, even for a byte-identical shared system prompt or tool schema. Hosted APIs (Anthropic, OpenAI, Gemini) and self-hosted engines both solve this via prefix reuse — either provider-side prompt caching or engine-side automatic prefix caching — by keeping a previously-computed KV-cache prefix resident and skipping its recomputation. It's a distinct optimization axis from batching, and workloads with long, repeated prefixes often benefit from it more.
13. **"What is 'structured output' / JSON mode under the hood?"** Grammar-constrained (constrained) decoding: the JSON Schema or grammar is compiled into an automaton over the vocabulary, and at every decode step the logits for grammatically-invalid tokens are masked to \(-\infty\) before sampling, so every generated token is schema-valid by construction — not by asking nicely and retrying on failure. This is a decode-time cost (computing/re-checking the valid token set every step), which is why engines use purpose-built compilers (e.g., XGrammar) rather than naive per-step re-validation.
14. **"Why bound the request queue instead of letting it grow to absorb bursts?"** An unbounded queue doesn't remove overload, it hides it — it turns a capacity problem into unbounded process memory growth and lets client-side timeouts fire anyway once wait time exceeds them, except now invisibly. A bounded queue with a fast, explicit rejection (503) gives callers, load balancers, and autoscalers an honest signal to react to.
15. **"What's the difference between your liveness and readiness checks, and why does it matter for a GPU pod specifically?"** Liveness asks "is the process alive enough to not be killed and restarted" and must stay cheap and independent of GPU load, or a temporarily saturated-but-healthy pod gets killed mid-request (war story 1). Readiness asks "should traffic be routed here right now" and should reflect real capacity — model loaded, not already at its concurrency cap — so a load balancer stops sending new work without killing in-flight work.

16. **"Quantization or a smaller model — how do you decide?"** They are not interchangeable levers even when they land at similar memory footprints. Quantizing a larger model (e.g., 7B to INT8) keeps its underlying capability at a precision cost; swapping to a smaller model in full precision changes the capability itself. Run the memory math for both (Worked Example 2's quantization variant), then validate quality against your own eval set — never assume either choice is quality-neutral by default.
17. **"What would you monitor in week one of running this in production that you might not think of on day one?"** Beyond the obvious latency/throughput dashboards: KV-cache/GPU-memory utilization over time (not just point-in-time), queue-depth and 503-rate as a leading indicator of undersized concurrency limits, p99-to-p50 latency ratio as a serialization signal, and cold-start duration on new pods — each of these is a metric that stayed invisible right up until the two war stories above turned into incidents.
18. **"If you could only add one thing to the naive server before it sees production traffic, what would it be?"** A defensible answer names the concurrency gate sized from real memory math (Worked Example 2 + Section B) over any other single change — it is the one guard that converts an unbounded-failure incident (war story 2) into a bounded, observable one, and most other hardening (structured errors, readiness checks) matters most in service of that same goal: knowing and respecting the server's actual capacity.

### System design prompt: "One GPU, 50 requests per second — walk me through your design"

This is a common live prompt because it forces you to reconcile everything above under a concrete constraint. A worked sketch:

**Step 1 — clarify the workload.** Ask: what's the model size, typical prompt/output length, and latency SLO? Assume, for concreteness: a 7B model in BF16 on a single 80 GB H100, average prompt 300 tokens, average output 150 tokens, target p95 E2E under ~2s.

**Step 2 — do the memory math first, not last.** Weights: \(7\text{B} \times 2\text{ bytes} \approx 14\text{ GB}\). KV cache per token (from Worked Example 2's formula, LLaMA-style with GQA, roughly ~0.15–0.5 MB/token depending on \(H_{kv}\)) times a typical active sequence length of ~450 tokens (300 prompt + 150 output) gives on the order of 100–200 MB of KV cache per request. On an 80 GB GPU with ~60 GB free after weights and overhead, that is *theoretical* room for hundreds of concurrent sequences purely on memory — the real constraint at 50 req/s is compute and scheduling, not raw KV capacity, which is itself a useful thing to say out loud (it shows you check both constraints instead of assuming one).

**Step 3 — pick the serving layer.** At 50 req/s sustained, this is squarely continuous-batching territory: reach for vLLM (or TGI/TensorRT-LLM) rather than a hand-rolled server — the naive per-request server from this chapter cannot get anywhere close to 50 req/s on one GPU because it never overlaps requests' decode steps. State this directly: "I would not build this on raw `transformers.generate()`; I'd deploy vLLM with continuous batching and PagedAttention on this GPU."

**Step 4 — size the batch/concurrency for the SLO.** With continuous batching, throughput scales with how many sequences' decode steps you can interleave before per-step latency (which grows mildly with batch size) blows the p95 SLO. Talk through the dial explicitly: push batch size up until TPOT growth threatens the 2s target, then back off — this is the latency/throughput tradeoff from the metrics section, now as a concrete scheduling decision rather than an abstract tradeoff.

**Step 5 — add the operational layer.** Structured error handling and admission control (Section B) still apply even in front of vLLM — you still want a bounded request queue in your API gateway, health/readiness endpoints that don't depend on GPU load, and metrics on TTFT/TPOT/queue depth/GPU utilization. If a burst pushes sustained demand past what continuous batching on one GPU can sustain at your SLO, the answer is horizontal scaling (more GPU replicas behind a load balancer) or admission control (shed load past capacity), not trying to fit more into one GPU than the SLO tolerates.

**Step 6 — name the next lever if traffic grows.** Prefix/prompt caching if requests share system prompts or tools; quantization (INT8/FP8/INT4) to shrink weights and free more room for KV cache and larger batches; speculative decoding to cut decode steps for latency-sensitive traffic. Naming these as *considered and deferred*, with a reason, is stronger than listing them as things you'd "probably also do."

The throughline an interviewer is grading: did you reach for memory math before guessing, did you correctly identify that a single naive server cannot hit this target, and did you reason about the latency/throughput dial explicitly rather than asserting "vLLM is fast."

> **Saying it out loud.** The move on a question like this is to clarify first, then do the memory math before you pick any technology. So: what's the model size, typical prompt and output length, and what's the latency SLO? Say it's a 7B in BF16 on an 80 GB H100, 300-token prompts, 150-token outputs, p95 under two seconds. Weights are 14 GB, leaving around 60 free; at roughly 100 to 200 megabytes of KV cache per request that's theoretical room for hundreds of sequences — so memory isn't the binding constraint here, scheduling and compute are, and saying that out loud shows you checked both. Then you pick continuous batching, because 50 requests per second is flatly impossible on a serial server. What they're grading is whether you reached for arithmetic before you reached for a tool name.

### Red flags vs green flags

| Signal | Red flag | Green flag |
|---|---|---|
| Explaining a request | Talks about "the model" as one black-box step | Names prefill and decode separately, with different bottlenecks |
| GPU memory | Guesses a number, or only mentions weights | Computes weights + KV cache + overhead, cites the KV-cache-per-token formula |
| Concurrency | "Just add more workers" | Ties concurrency limits to the memory math, mentions bounded queues and load shedding |
| Async server design | Doesn't know `model.generate()` blocks the event loop | Volunteers `run_in_threadpool` / worker-process reasoning unprompted |
| Health checks | Treats liveness and readiness as the same thing | Explains why liveness must stay load-independent and readiness should reflect capacity |
| Batching | "Batching makes it faster" with no mechanism | Explains static vs dynamic vs continuous batching and head-of-line blocking |
| When to use an engine | Says "vLLM is industry standard" with no reasoning | States the specific gap (continuous batching, paged KV, prefix caching) the raw server has |
| Metrics | Reports only average latency | Reports TTFT/TPOT/E2E at p50/p95/p99 plus throughput and queue depth |
| Structured output | Thinks it's a prompting trick | Explains constrained decoding / logit masking against a compiled grammar |
| Incidents | Has no story, or blames "the GPU" vaguely | Can narrate a specific failure (blocked event loop, unbounded batch OOM) and the fix |

---

### Glossary — quick reference for review

| Term | One-line definition |
|---|---|
| **Prefill** | The single, parallel forward pass over the whole prompt; compute-bound; sets TTFT. |
| **Decode** | The one-token-at-a-time generation loop after prefill; memory-bandwidth-bound; sets TPOT. |
| **KV cache** | Stored key/value vectors per token/layer/head, reused every decode step instead of recomputed. |
| **TTFT** | Time to first token — request submitted to first output token received. |
| **TPOT / ITL** | Time per output token / inter-token latency — average gap between subsequent tokens. |
| **GQA (grouped-query attention)** | Fewer KV heads than query heads; shrinks KV-cache bytes/token linearly. |
| **Static batching** | Pad and run a fixed batch of requests to completion together; head-of-line blocking. |
| **Dynamic batching** | Briefly buffer arriving requests into a batch, then run it to completion; still head-of-line-blocked. |
| **Continuous batching** | Schedule at the decode-step granularity; sequences join/leave the running batch every iteration. |
| **PagedAttention** | KV cache stored in fixed-size non-contiguous pages, like OS virtual memory, to cut fragmentation. |
| **Prefix caching** | Reusing a previously computed KV-cache prefix across requests sharing a prompt prefix. |
| **Prompt / context caching** | The hosted-API equivalent of prefix caching: discounted, cached reuse of a repeated prompt prefix. |
| **Structured outputs** | API/engine-guaranteed schema-conformant generation via constrained (grammar-guided) decoding. |
| **Constrained decoding** | Masking invalid-per-grammar tokens' logits to \(-\infty\) at every decode step. |
| **safetensors** | Tensor-only serialization format; no arbitrary code execution on load, unlike `pickle`. |
| **Quantization** | Storing weights (and sometimes activations/KV cache) in fewer bits per value to save memory. |
| **Speculative decoding** | A small draft model proposes tokens; the full model verifies several in one pass to cut decode steps. |
| **Liveness probe** | "Is the process alive enough not to be restarted" — should be cheap and load-independent. |
| **Readiness probe** | "Should traffic be routed here right now" — should reflect real load/capacity. |
| **Backpressure / load shedding** | Rejecting requests past a bounded capacity (e.g., 503) instead of queueing without bound. |

---

## Further reading

**Core mechanism, generation, and this chapter's server**
- Hugging Face — Text generation tutorial (`generate`): <https://huggingface.co/docs/transformers/en/llm_tutorial>
- Hugging Face — Generation config / `GenerationConfig` reference: <https://huggingface.co/docs/transformers/en/main_classes/text_generation>
- FastAPI documentation: <https://fastapi.tiangolo.com/>
- FastAPI — concurrency and async/await: <https://fastapi.tiangolo.com/async/>
- httpx (used for the load test): <https://www.python-httpx.org/>
- Kubernetes — configuring liveness, readiness, and startup probes: <https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/>

**KV cache, memory, and batching**
- NVIDIA — Mastering LLM Techniques: Inference Optimization (KV cache, batching): <https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/>
- KV cache memory calculation (worked example): <https://mbrenndoerfer.com/writing/kv-cache-memory-calculation-llm-inference-gpu>
- Pierre Lienhart — LLM Inference Series: KV caching, a deeper look: <https://medium.com/@plienhar/llm-inference-series-4-kv-caching-a-deeper-look-4ba9a77746c8>
- vLLM — PagedAttention & continuous batching overview: <https://www.runpod.io/articles/guides/vllm-pagedattention-continuous-batching>
- Anyscale — Understand LLM latency and throughput metrics (TTFT/TPOT/ITL/TPS/RPS): <https://docs.anyscale.com/llm/serving/benchmarking/metrics>

**Model formats and safety**
- `safetensors` — project and rationale: <https://github.com/huggingface/safetensors>
- Text Generation Inference docs — why safetensors: <https://huggingface.co/docs/text-generation-inference/main/en/conceptual/safetensors>

**Structured / constrained decoding**
- OpenAI — Introducing Structured Outputs in the API: <https://openai.com/index/introducing-structured-outputs-in-the-api/>
- OpenAI — Structured model outputs guide: <https://developers.openai.com/api/docs/guides/structured-outputs>
- vLLM — Structured Outputs feature docs: <https://docs.vllm.ai/en/latest/features/structured_outputs/>
- XGrammar — Flexible and Efficient Structured Generation Engine for LLMs (paper): <https://arxiv.org/pdf/2411.15100>
- Red Hat Developer — Structured outputs in vLLM: guiding AI responses: <https://developers.redhat.com/articles/2025/06/03/structured-outputs-vllm-guiding-ai-responses>

**Prompt / context caching (2025–2026 provider docs)**
- Anthropic — Prompt caching: <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>
- OpenAI — Prompt caching guide: <https://developers.openai.com/api/docs/guides/prompt-caching>
- OpenAI — Prompt Caching in the API (announcement): <https://openai.com/index/api-prompt-caching/>
- Google — Gemini API context caching: <https://ai.google.dev/gemini-api/docs/caching>

**Production serving engines**
- Text Generation Inference (TGI) docs: <https://huggingface.co/docs/text-generation-inference/en/index>
- TGI source and features: <https://github.com/huggingface/text-generation-inference>
- NVIDIA Triton Inference Server: <https://github.com/triton-inference-server/server>
- vLLM documentation: <https://docs.vllm.ai/>

---

> **Where this goes next:** Chapter 2 containerizes this server; Chapter 4 load-tests it to *see* the serial bottleneck at larger scale than Section B's local script; Chapter 5 replaces it with vLLM and continuous batching to fix everything this chapter exposed — and revisits prefix caching and structured decoding as engine-native features rather than DIY additions.
