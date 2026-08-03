# vLLM Serving — PagedAttention, Continuous Batching, and Production Tuning

## Why this matters

If you serve open-weight LLMs at any real scale, vLLM is very likely the engine underneath — or the baseline everything else is measured against. It became the default high-throughput inference server because it attacked the single biggest bottleneck in LLM serving: **memory**, specifically the KV cache. Its headline result, from the original paper, is a **2–4× throughput improvement at the same latency** versus the prior state of the art (FasterTransformer and Orca).

The insight is almost embarrassingly simple in hindsight: LLM serving was wasting **60–80% of KV-cache memory** to fragmentation and over-reservation. vLLM borrowed *virtual memory and paging* from operating systems, applied it to the KV cache, and turned that wasted memory back into batch capacity. More batch capacity means more requests share each expensive weight-load from GPU memory, which is exactly what raises throughput.

This chapter goes deep on the mechanisms (PagedAttention, continuous batching, prefix caching), the knobs that actually matter in production (`gpu-memory-utilization`, `max-num-seqs`, `max-num-batched-tokens`, chunked prefill), how to scale across GPUs, and how to trade quality for speed with quantization and speculative decoding. Then a fully worked tuning walkthrough, a comparison to TGI and TensorRT-LLM, and the failure modes that page you at 3am.

---

## Core intuition: LLM inference is memory-bound, and the KV cache is the problem

Two facts drive everything:

1. **Autoregressive decoding is memory-bandwidth-bound, not compute-bound.** Generating one token touches the *entire* model weights but does very little arithmetic per token. The GPU spends most of its time reading weights from HBM, not doing math. The fix is **batching**: process many sequences at once so a single weight read serves many tokens. Bigger batch → higher throughput, until you run out of memory.

2. **What runs you out of memory is the KV cache.** For every token in every sequence, attention must remember the key and value vectors of all previous tokens. This "KV cache" grows linearly with sequence length *and* with the number of concurrent sequences. On a typical setup the weights are fixed, and whatever HBM is left is a fixed budget you spend on KV cache. **The more efficiently you pack the KV cache, the bigger your batch, the higher your throughput.**

So the game is: *fit as many sequences' KV caches into the leftover HBM as possible, and keep the GPU busy on all of them at once.* PagedAttention wins the first half; continuous batching wins the second.

### How big is the KV cache?

Per token, the KV cache size is:

\[
\text{bytes/token} = 2 \times n_\text{layers} \times n_\text{kv\_heads} \times d_\text{head} \times \text{dtype\_bytes}
\]

The leading \(2\) is for K and V. Note \(n_\text{kv\_heads}\), not the number of query heads — models with **grouped-query attention (GQA)** share KV heads across query heads, which shrinks the cache dramatically.

**Worked numbers (fp16, 2 bytes):**

- **OPT-13B** (40 layers, hidden 5120, full MHA): \(2 \times 40 \times 5120 \times 2 = 819{,}200\) bytes ≈ **800 KB per token**. A single 2048-token sequence needs ~1.6 GB of KV cache — this is the number from the vLLM paper.
- **Llama-3-8B** (32 layers, 8 KV heads, \(d_\text{head}=128\), GQA): \(2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\) bytes = **128 KB per token**. GQA makes it ~6× cheaper than a same-size MHA model.

At 800 KB/token, KV cache is enormous and *dynamic* — you don't know a request's final length in advance. That combination is exactly what classic allocators handle badly.

---

## PagedAttention in depth

### The problem it solves: fragmentation and over-reservation

Pre-vLLM systems (Orca, FasterTransformer) stored each sequence's KV cache in **one contiguous chunk of memory, sized to the maximum possible length**. If your model supports 2048 tokens, every sequence reserved space for 2048 tokens the moment it started — even if it only ever generated 30. Three kinds of waste result, and the paper measures them directly:

- **Internal fragmentation** (13.3%–57.3%): the slot reserved for max length but never filled.
- **Reservation waste** (part of 25.2%–96.3%): space reserved for future tokens of a still-running sequence — technically "will be used" but idle *now*, so it can't serve other requests.
- **External fragmentation**: gaps between contiguous chunks of different sizes that no new sequence fits into.

Net effect: measured **effective KV utilization of only 20.4%–38.2%**. Four out of five bytes wasted.

### The idea: page the KV cache like virtual memory

Operating systems solved this decades ago. A process sees a contiguous virtual address space, but physically it's scattered across fixed-size **pages** mapped by a **page table**. No process reserves all of physical RAM up front; pages are handed out on demand.

PagedAttention does the same for the KV cache:

- The KV cache of a sequence is split into fixed-size **KV blocks**, each holding the K and V vectors for a fixed number of tokens — the **block size, default 16 tokens**.
- Blocks live in a global pool of physical GPU memory and **need not be contiguous**.
- Each sequence has a **block table** mapping its logical block index → physical block number, exactly like a page table.
- The attention kernel is modified to gather K/V from these scattered blocks using the block table, so attention runs correctly over non-contiguous memory.

### Diagram-in-words

```
Logical view (what the sequence "sees"):
  Seq A tokens: [ t0 t1 ... t15 | t16 t17 ... t31 | t32 t33 ... ]
                 logical block 0   logical block 1   logical block 2

Block table for Seq A:  [ 0 -> phys #7 ] [ 1 -> phys #3 ] [ 2 -> phys #11 ]

Physical KV block pool (16 tokens each, scattered in HBM):
  #0  #1  #2  #3(A1)  #4  #5  #6  #7(A0)  #8  #9  #10  #11(A2)  #12 ...
  free free free  used  free ... free  used   ...      used    free
```

A block is allocated **only when the sequence's current block fills up**. A sequence generating 30 tokens uses 2 blocks (32 slots), wasting at most 15 token-slots in its last block — **at most one block of internal fragmentation per sequence, and zero reservation waste**. External fragmentation vanishes because all blocks are the same size and interchangeable. Effective utilization approaches ~96%.

### Sharing and copy-on-write

Because blocks are indirected through a block table, **two sequences can point their block tables at the same physical block**. This is where paging pays a second dividend:

- **Shared prompts.** In parallel sampling or beam search, \(n\) outputs share the same prompt. Instead of \(n\) copies of the prompt's KV cache, all \(n\) block tables point at one shared set of prompt blocks. The paper reports up to **55% memory savings** on parallel sampling / beam search.
- **Copy-on-write (CoW).** When one sharer needs to diverge (e.g., append a different token into a shared block), vLLM copies just that one block, updates that sequence's block table, and leaves the others untouched — the same trick `fork()` uses. Reference counts on each block track sharing.

This block-level sharing is the foundation that **automatic prefix caching** (below) builds on.

### Worked memory example

Serve **Llama-3-8B in fp16 on one A100-80GB**.

1. **Weights:** \(8\text{B} \times 2\ \text{bytes} = 16\ \text{GB}\).
2. **Budget:** `--gpu-memory-utilization 0.9` → vLLM may use \(0.9 \times 80 = 72\ \text{GB}\).
3. **Non-KV overhead:** CUDA context, activations, CUDA graphs — call it ~2 GB.
4. **KV cache pool:** \(72 - 16 - 2 = 54\ \text{GB}\).
5. **Per token:** 128 KB (from above). **Per block (16 tokens):** \(16 \times 128\ \text{KB} = 2\ \text{MB}\).
6. **Total blocks:** \(54\ \text{GB} / 2\ \text{MB} \approx 27{,}600\) blocks = **~442,000 tokens** of KV capacity.

That single number, ~442k tokens, is your batch budget. It can be **54 sequences at the full 8192-context** (\(442000/8192\)), or **~880 concurrent chatbot turns** averaging 500 tokens each, or anything in between. vLLM logs this at startup as `# GPU blocks: 27600` and reports "Maximum concurrency for 8192 tokens per request." Watch that log line — it tells you exactly how much headroom you bought.

---

## Continuous (in-flight) batching in depth

PagedAttention gives you the memory to run a big batch. Continuous batching keeps that batch *full*.

### Static batching (the naive baseline)

Collect \(N\) requests, run them together, wait for **all** to finish, return, repeat. The problem: generations have wildly different lengths. If request A emits 20 tokens and request B emits 500, A's slot sits **idle for 480 steps** while B finishes, because the batch can't return or refill until the whole batch is done. GPU utilization craters, and latency for A is dictated by the slowest sibling.

### Dynamic batching (a partial fix)

Servers like Triton's dynamic batcher wait a few milliseconds to *form* a larger batch before launching, then still run it to completion as a unit. This improves batch size but does **not** solve the ragged-completion problem — it's still batch-at-a-time.

### Continuous batching (a.k.a. in-flight / iteration-level scheduling)

vLLM schedules at the granularity of **a single decode step, not a whole request** (the idea comes from Orca's iteration-level scheduling). At every forward pass:

1. Any sequence that emitted its EOS/stop this step is **evicted immediately**, and its KV blocks are freed.
2. Waiting requests are **admitted mid-flight** to fill the vacated slots.
3. The next forward pass runs over the new mix of prefills and decodes.

No sequence waits on a slower sibling. The batch is continuously topped up, so the GPU stays saturated. Anyscale's widely cited benchmark measured up to **23× throughput** from continuous batching plus paging versus naive batching, while *also* reducing p50 latency — a rare win on both axes, because higher utilization means less queueing.

**Why it raises throughput:** decoding is memory-bound, so throughput scales with how many sequences you can run per weight-load. Static batching leaves the effective batch shrinking toward 1 as siblings finish; continuous batching holds it near its memory-limited maximum every single step.

---

## Prefill vs decode, and chunked prefill

A request has two phases with opposite performance profiles:

- **Prefill** — process the whole prompt in one big parallel pass. Compute-bound, high FLOPs, fills the pipeline. A 4000-token prompt is one heavy step.
- **Decode** — generate tokens one at a time. Memory-bound, tiny per-step compute.

Mixing them is awkward. A giant prefill can **monopolize a forward pass and stall every decoding sequence**, spiking inter-token latency (ITL) for everyone already streaming. This is the classic prefill/decode interference.

**Chunked prefill** (`--enable-chunked-prefill`) splits a large prefill into token-sized chunks and **co-schedules a prefill chunk alongside ongoing decodes** in the same batch, bounded by `max-num-batched-tokens`. Benefits:

- Smooths out ITL — decodes no longer freeze behind a monster prompt.
- Improves GPU utilization — decode steps are compute-light, so padding the batch with prefill tokens uses otherwise-idle FLOPs.
- In modern vLLM (V1 engine) chunked prefill is **on by default**, and prefill/decode are unified in one scheduler.

Tuning: **raise `max-num-batched-tokens` for throughput** (bigger chunks, more prefill work per step); **lower it to protect decode latency** (smaller chunks yield to decodes more often).

---

## Prefix caching (automatic KV reuse)

Many requests share a prefix: the same long system prompt, a shared few-shot preamble, a document everyone asks questions about, a multi-turn conversation where each turn re-sends the history.

**Automatic prefix caching** (`--enable-prefix-caching`) hashes KV blocks by their content (and the tokens preceding them). When a new request's prefix hashes to blocks already in the cache, vLLM **skips recomputing that prefill entirely** and points the new sequence's block table at the cached blocks. It's the block-sharing / CoW machinery from PagedAttention, applied across requests and over time.

- **When it wins big:** long shared system prompts, RAG with a fixed instruction preamble, multi-turn chat (each turn reuses the whole prior conversation's KV), agent loops that resend context.
- **Cost:** cached blocks occupy KV memory that could otherwise hold active batch. Under memory pressure, cached prefix blocks are evicted LRU. It's a hit-rate bet — near-free when hits are common, mild overhead when they're not.
- In current vLLM (V1) prefix caching is **enabled by default**.

The saving is real work avoided: a 2000-token shared system prompt cached across 1000 requests skips ~2,000,000 tokens of prefill compute.

---

## Memory and the engine args that matter

These are the flags you actually turn in production. Names are the current `vllm serve` CLI form (dashes); the Python `LLM(...)` form uses underscores.

| Flag | Default | What it does | How to tune |
|---|---|---|---|
| `--gpu-memory-utilization` | `0.9` | Fraction of each GPU's HBM vLLM may use (weights + KV + activations). Sets the KV pool size. | Raise toward `0.92–0.95` to grow the batch if you have headroom; **lower if you OOM** or co-locate other processes. Leave slack for activation spikes. |
| `--max-num-seqs` | `256` (V1; was model-dependent) | Max sequences in a batch (concurrency cap). | Raise for throughput if KV memory allows; lower to cap per-request latency and memory. Often the real batch limit is KV memory, not this. |
| `--max-num-batched-tokens` | auto (e.g. `8192`/`2048`) | Max tokens processed per iteration (prefill chunks + decode tokens). | **Raise for throughput**, lower to protect ITL. Must be ≥ `max-model-len` unless chunked prefill is on. |
| `--max-model-len` | from model config | Max context (prompt + output) per request. | **Lower it to fit more sequences / avoid OOM** when the model's native context exceeds your needs. Directly bounds worst-case KV per sequence. |
| `--block-size` | `16` | Tokens per KV block. | Rarely changed. Larger blocks = less overhead but more internal fragmentation. |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | on (V1) | Reuse KV of shared prefixes across requests. | Keep on for chat/RAG/agents; disable only if prefixes never repeat and you want the memory back. |
| `--enable-chunked-prefill` | on (V1) | Split prefills into chunks, co-schedule with decodes. | Keep on; tune via `max-num-batched-tokens`. |
| `--tensor-parallel-size` (`-tp`) | `1` | Shard each layer across N GPUs (intra-node). | Set to fit a model too big for one GPU / to cut latency. Use ≤ GPUs per node with fast NVLink. |
| `--pipeline-parallel-size` (`-pp`) | `1` | Split layers into stages across GPUs/nodes. | Use to span **multiple nodes** or when TP alone can't fit the model. |
| `--quantization` | none | Weight/activation quant scheme (`awq`, `gptq`, `fp8`, `bitsandbytes`, ...). | Use to shrink weights → more KV room / smaller GPU. Costs some quality. |
| `--kv-cache-dtype` | `auto` | Store KV cache in `fp8` etc. | `fp8` ~halves KV memory → bigger batch/context; small accuracy cost. |
| `--swap-space` | `4` (GiB/GPU) | CPU RAM for swapping out preempted sequences' KV. | Raise if you see frequent preemption + recompute; swap can be cheaper than recompute for long sequences. |
| `--max-num-seqs` + `--max-num-batched-tokens` together | — | The two levers that shape the batch. | Co-tune: token budget caps work/step; seq budget caps concurrency. |
| `--dtype` | `auto` | Compute dtype (`bfloat16`, `float16`). | `bfloat16` on Ampere+; matters for numerical stability. |
| `--speculative-config` | none | Speculative decoding config (JSON). | See below — latency win when acceptance is high. |

Startup log lines to watch: `# GPU blocks:` (your KV capacity), `Maximum concurrency for N tokens`, and any `Sequence group ... is preempted` warnings (you're memory-starved).

---

## Parallelism for big models

When a model (plus its KV cache) doesn't fit on one GPU, or single-GPU latency is too high, split it.

### Tensor parallelism (TP) — `--tensor-parallel-size`

**Mechanism:** shard every layer's weight matrices across N GPUs; each GPU computes its slice, and an **all-reduce** combines partial results each layer. The KV cache is also sharded (by heads), so TP grows your KV budget too.

**When:** the model is too big for one GPU, or you want lower latency on a single request (more GPUs working the same forward pass). Best **within one node over NVLink**, because the per-layer all-reduce is bandwidth-hungry.

**Tradeoff:** communication overhead grows with N; going cross-node over slower interconnect (Ethernet/PCIe) tanks efficiency. Keep `-tp` ≤ GPUs-per-node. TP size must divide the number of attention heads.

### Pipeline parallelism (PP) — `--pipeline-parallel-size`

**Mechanism:** assign contiguous *stages* of layers to different GPUs; activations flow stage → stage. Communication is a small point-to-point hand-off between stages, tolerant of slower links.

**When:** to span **multiple nodes**, or to fit truly huge models where even TP-across-a-node isn't enough. Common pattern: `-tp 8` within each node × `-pp 2` across two nodes = 16 GPUs.

**Tradeoff:** introduces pipeline bubbles (stages idle waiting for the previous stage); throughput-friendly with enough in-flight requests, but adds latency per request. Combine TP (intra-node) + PP (inter-node) for the best of both.

Rule of thumb: **TP first, up to one node; PP to cross nodes.** vLLM also supports data parallelism / multi-replica behind a router for pure scale-out.

---

## Quantization — trade quality for memory and speed

Quantization shrinks weights (and optionally activations/KV) to fewer bits. Smaller weights free HBM for KV cache and can speed up the memory-bound decode. All of it costs *some* accuracy; how much depends on scheme and model.

| Scheme | Bits | What's quantized | When to use | Tradeoff |
|---|---|---|---|---|
| **AWQ** | 4-bit | Weights only (activation-aware, protects salient weights) | Serving throughput on Ampere/Ada; strong quality at 4-bit | Needs a pre-quantized AWQ checkpoint; INT4 weights dequantized for compute |
| **GPTQ** | 3/4/8-bit | Weights only (2nd-order error minimization) | Broad hardware/checkpoint availability | Quality can degrade at 3-bit; per-model calibration sensitivity |
| **FP8** (E4M3) | 8-bit | Weights + activations (and KV via `--kv-cache-dtype fp8`) | **Hopper/H100, Ada** with hardware FP8; near-lossless, high throughput | Requires FP8-capable GPUs for full speedup |
| **INT8 (SmoothQuant/W8A8)** | 8-bit | Weights + activations | Good quality/speed balance where FP8 HW absent | More setup; less dramatic memory savings than 4-bit |
| **bitsandbytes** | 4/8-bit | Weights, on-the-fly | Quick experiments, no pre-quant step | Slower kernels; not the throughput champion |

Guidance:

- **Memory-constrained, throughput-focused, Ampere/Ada:** AWQ 4-bit is the workhorse — cuts weight memory ~4×, freeing large KV headroom.
- **H100 / Hopper:** prefer **FP8** — near-lossless and uses native tensor-core FP8 for real speedups; add `--kv-cache-dtype fp8` to roughly double KV capacity.
- **Quality-sensitive tasks (code, math, long reasoning):** measure. 4-bit weight-only can visibly hurt; validate on *your* eval set, not just perplexity.
- Quantization reduces *weight* memory, not KV — for long-context blowup, `--kv-cache-dtype fp8` and `--max-model-len` are the relevant levers.

---

## Speculative decoding — lower latency, not more throughput

**Mechanism:** a cheap **draft** proposes several tokens ahead; the big **target** model verifies them all in one forward pass. Accepted tokens are kept; the first rejection resets to the target's own token. Because verification is parallel, a good draft yields **multiple tokens per target forward pass** — output is provably identical in distribution to the target alone (it's exact, not approximate).

Draft sources vLLM supports include a **small draft model**, **n-gram / prompt-lookup** (propose from repeated text — great for code and RAG where output echoes input), and **EAGLE / Medusa**-style self-speculative heads.

**When to use:** latency-sensitive, **low-to-moderate batch** serving where the GPU has spare compute (decode is memory-bound, so verification is nearly free). Interactive chat, single-user, or bursty low-QPS endpoints.

**Tradeoff:** the win depends entirely on **acceptance rate**. Low acceptance means you paid for drafting and got little back — it can *reduce* throughput. And under **high batch load the GPU is already compute-saturated**, so speculation's "free" parallel verification isn't free anymore — its benefit shrinks or reverses. Rule: speculate when you're latency-bound and under-batched; skip it when you're throughput-bound and saturated.

---

## Fully worked example: serve Llama-3-8B on one A100-80GB, then tune

### 1. Baseline launch (OpenAI-compatible server)

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

At startup, read the log:

```
INFO ... # GPU blocks: 27600, # CPU blocks: 2048
INFO ... Maximum concurrency for 8192 tokens per request: 53.9x
```

That confirms the ~442k-token / ~54-concurrent budget we computed by hand.

### 2. Call it with the OpenAI client

The server speaks the OpenAI API, so existing SDKs work unchanged — just point `base_url` at vLLM:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Explain PagedAttention in two sentences."},
    ],
    max_tokens=128,
    temperature=0.2,
    stream=True,  # tokens stream as they decode
)
for chunk in resp:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
```

`curl` sanity check:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Meta-Llama-3-8B-Instruct","prompt":"Hello","max_tokens":16}'
```

### 3. Tuning walkthrough (target: max throughput for a RAG service)

Symptoms to check first with a load test (`vllm bench serve` or a locust/k6 run): GPU util, tokens/s, p99 ITL, and any `preempted` warnings.

- **Shared 1500-token system+retrieval preamble across requests** → keep `--enable-prefix-caching` (default on). Hit rate is high; prefill work drops sharply.
- **GPU shows headroom, no OOM** → push memory:
  ```bash
  vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --gpu-memory-utilization 0.94 \
    --max-num-seqs 384 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --max-model-len 8192
  ```
  Bigger token budget = fatter prefill chunks and higher throughput; more seqs = more concurrency, backed by the enlarged KV pool.
- **p99 inter-token latency too high** (big prompts stalling decodes) → *lower* `--max-num-batched-tokens` (e.g. to 4096). Smaller chunks yield to decodes more often, smoothing ITL at a small throughput cost.
- **`Sequence group is preempted by ...` warnings** → you over-committed KV. Either drop `--max-num-seqs`, drop `--max-model-len`, or raise `--swap-space` so preempted sequences swap to CPU instead of recomputing.
- **Need more KV for longer contexts** → add `--kv-cache-dtype fp8` (roughly doubles KV capacity) and/or `--quantization fp8` on an H100 to also shrink weights.
- **Model too big for one GPU** (e.g. Llama-3-70B) → `--tensor-parallel-size 4` (or 8) within the node; across nodes add `--pipeline-parallel-size 2`.

Iterate: change one knob, re-run the load test, compare tokens/s and p99. Stop when you're memory-limited (preemptions appear) or latency SLO-limited.

---

## Comparison: vLLM vs TGI vs TensorRT-LLM / Triton

| Dimension | **vLLM** | **TGI (HF Text Generation Inference)** | **TensorRT-LLM + Triton** |
|---|---|---|---|
| Core strength | PagedAttention + continuous batching; best throughput/$ out of the box | Solid production server, tight HF ecosystem fit | Peak NVIDIA-GPU performance via compiled engines |
| Batching | Continuous, iteration-level | Continuous (in-flight) | In-flight batching (Triton backend) |
| KV memory mgmt | PagedAttention (near-zero waste) | Paged KV (adopted vLLM-style ideas) | Paged KV |
| Prefix caching | Automatic, on by default | Supported | Supported |
| Quantization | AWQ, GPTQ, FP8, INT8, bnb | AWQ, GPTQ, EETQ, FP8, bnb | INT4/8, FP8 (compiled, very fast) |
| Speculative decode | Draft model, n-gram, EAGLE/Medusa | Medusa / n-gram | EAGLE, Medusa, draft |
| Setup cost | Low — `pip install`, one command | Low — Docker image | **High** — per-model engine build/compile step |
| Hardware | NVIDIA + AMD ROCm + others | NVIDIA + AMD | NVIDIA only |
| API | OpenAI-compatible server | OpenAI-compatible + native | Triton (OpenAI frontend available) |
| Best when | Default choice; open models, fast iteration, high throughput | HF-centric stacks wanting a batteries-included server | Squeezing max perf on fixed NVIDIA hardware, willing to pay build complexity |

Reality check: the three have converged — TGI and TensorRT-LLM adopted paged KV and in-flight batching. **TensorRT-LLM often wins raw latency/throughput on NVIDIA** thanks to ahead-of-time kernel compilation, at the cost of a per-model engine-build step and NVIDIA lock-in. **vLLM wins on flexibility, ease, and hardware breadth**, and is the usual default. Always benchmark on *your* model, hardware, and traffic shape before deciding.

---

## Failure modes and pitfalls

- **OOM at startup from `gpu-memory-utilization` too high.** vLLM pre-allocates the KV pool; if you set 0.98 with no slack, activation spikes or CUDA-graph capture push you over and it crashes on load. Back off to ~0.90 and grow gradually. Co-located processes share the same HBM — vLLM only sees the fraction you give it.
- **Preemption and recompute thrash.** When admitted sequences collectively exceed KV capacity, vLLM **preempts** some — either swapping their KV to CPU (`--swap-space`) or **discarding and recomputing** it later. Frequent `preempted` warnings mean you over-committed `max-num-seqs`/`max-model-len`; throughput drops as work is redone. Fix by lowering concurrency, shortening `max-model-len`, or adding swap.
- **Long-context KV blowup.** KV grows linearly with context. A handful of 128k-token requests can consume the entire pool and starve everyone else. Bound it with `--max-model-len`, `--kv-cache-dtype fp8`, and admission limits; don't advertise a context you can't afford to serve concurrently.
- **Quantization quality loss.** 4-bit weight-only (AWQ/GPTQ) can degrade code/math/reasoning noticeably even when perplexity looks fine. Always validate on a task-specific eval, and prefer FP8 on Hopper where it's near-lossless.
- **Speculative decoding backfiring.** Low draft acceptance, or high batch load, turns speculation into pure overhead. Measure acceptance rate; disable under saturation.
- **`max-num-batched-tokens` too low with chunked prefill off.** If a prompt exceeds the token budget and chunked prefill isn't enabled, requests fail. Keep chunked prefill on, or set the budget ≥ `max-model-len`.
- **Assuming `max-num-seqs` is the batch limit.** Usually **KV memory** binds first. Raising `max-num-seqs` without KV headroom just causes preemption. Watch the `# GPU blocks` log, not just the seq cap.
- **Prefix cache eviction under pressure.** Cached prefixes compete with active KV; under load they're evicted and hit rate falls — throughput quietly regresses. Size memory for both if prefix reuse is core to your workload.

---

## Production checklist — what an interviewer probes

1. **"Why is LLM decode memory-bound, and why does that make batching the key throughput lever?"** — Weights are re-read every token; batching amortizes the read. Expect you to connect this to KV cache being the batch-size limiter.
2. **"Explain PagedAttention and what waste it eliminates."** — Blocks + block table, non-contiguous KV, block size 16, kills internal/external fragmentation and reservation waste (from ~20–38% utilization to ~96%).
3. **"Continuous vs static batching — why does continuous raise throughput *and* cut latency?"** — Iteration-level scheduling evicts finished seqs and admits new ones each step; no slot idles behind a slow sibling.
4. **"Walk me through sizing KV cache and picking `gpu-memory-utilization` / `max-num-seqs` / `max-model-len` for a given GPU."** — Do the bytes/token math, subtract weights, derive block count and concurrency; explain preemption when over-committed.
5. **"When TP vs PP? What are the comms costs?"** — TP intra-node over NVLink (per-layer all-reduce), PP across nodes (stage hand-off, pipeline bubbles).
6. **"Quantization choices and their quality cost — AWQ vs GPTQ vs FP8, and when each?"** — 4-bit weight-only for memory on Ampere; FP8 near-lossless on Hopper; validate on task evals.
7. **"When does speculative decoding help vs hurt?"** — Helps latency at low batch with high acceptance; hurts under saturation or low acceptance.
8. **"How do you debug an OOM or a latency spike in production?"** — Check `# GPU blocks` and preemption logs; lower `gpu-memory-utilization`/`max-model-len`; tune `max-num-batched-tokens` for ITL; add `swap-space`; confirm prefix-cache hit rate.

Bonus signals: knowing chunked prefill and prefix caching are **on by default in the V1 engine**, that `max-num-seqs` rarely binds before KV memory, and that you always benchmark on real traffic rather than trusting a spec sheet.

---

## How the attention kernel actually reads paged blocks

It's worth being precise about *why* PagedAttention needs a custom kernel, because interviewers probe it. Standard fused attention (FlashAttention) assumes K and V for a sequence live in one contiguous tensor it can stride through. Paged KV breaks that assumption: a sequence's K/V are scattered across physical blocks in arbitrary order.

The PagedAttention kernel therefore takes the **block table** as an input. For a query at the current position it:

1. Reads the sequence's block table (logical block → physical block number).
2. For each logical block, computes attention scores \( q \cdot k \) against the K vectors *in that physical block*, iterating block by block.
3. Accumulates the softmax-weighted sum of V vectors from the same blocks.

Because the block is the unit of gather, the kernel does a small indirection per block (once per 16 tokens), not per token — cheap relative to the matmul. The block table lives in GPU memory alongside the cache. Modern vLLM builds this on FlashAttention/FlashInfer backends that natively accept paged KV, so you keep FlashAttention's IO-awareness *and* paging. This is the crux: paging costs almost nothing at kernel time, yet returns most of the wasted memory as usable batch.

---

## The scheduler: waiting, running, swapped

vLLM's scheduler maintains three queues and reconciles them every step — this is the machinery behind continuous batching, preemption, and swapping.

- **Waiting** — admitted requests not yet started (need KV blocks allocated for their prefill).
- **Running** — sequences actively decoding (or being prefilled) this step.
- **Swapped** — sequences preempted out of GPU KV, their blocks parked in CPU swap space.

Each iteration the scheduler:

1. Frees blocks of any sequence that finished last step.
2. Tries to admit **waiting** requests into **running**, subject to the KV block budget and `max-num-seqs` / `max-num-batched-tokens`.
3. If **running** collectively needs more blocks than exist (e.g., all sequences grew a token and a new block boundary was crossed), it **preempts** the lowest-priority sequences — either **swap** (copy their KV blocks to CPU, restore later) or **recompute** (drop KV, re-run prefill when readmitted). Recompute is the default for short sequences; swap wins for long ones where recompute is expensive.

The default policy is FCFS-ish with the newest/lowest-priority preempted first. The practical takeaway: **preemption is the pressure-relief valve, and seeing it constantly in logs means your admission settings exceed your true KV capacity.** It is correct behavior, not a bug — but it costs throughput, so tune it away.

---

## Benchmarking vLLM properly

Never tune by feel. vLLM ships a benchmark harness that mirrors real serving:

```bash
# Start the server, then in another shell:
vllm bench serve \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000 \
  --request-rate 20
```

Metrics that matter, and what they mean:

- **Throughput (tokens/s, requests/s)** — the number to maximize for batch/offline workloads.
- **TTFT (time to first token)** — dominated by prefill and queueing; what a chat user feels as "lag before it starts."
- **ITL / TPOT (inter-token latency / time per output token)** — decode smoothness; hurt by big un-chunked prefills.
- **p50 vs p99** — always look at the tail. High p99 with fine p50 usually means preemption or prefill interference.

Sweep one knob at a time (`gpu-memory-utilization`, `max-num-batched-tokens`, `max-num-seqs`) and plot throughput vs p99 latency. The right operating point is the knee of that curve for your SLO — not the max-throughput point, which usually violates latency targets.

---

## V1 engine and disaggregated serving (where vLLM is heading)

Two architectural notes worth knowing:

- **The V1 engine** (default in current vLLM) rewrote the core for lower CPU overhead and a unified scheduler where prefill and decode are co-scheduled by default. Chunked prefill and prefix caching are **on by default** there. If you read older tutorials that tell you to manually enable these, that advice is stale.
- **Disaggregated prefill/decode** separates the compute-bound prefill and memory-bound decode onto **different GPU pools**, streaming the KV cache between them. Because the two phases have opposite resource profiles, dedicating hardware to each — and scaling them independently — can beat co-locating them, especially at high scale with long prompts. This is an emerging pattern (KV transfer over NVLink/RDMA) that large deployments increasingly adopt.

## Second worked example: Llama-3-70B across GPUs

An 8B model fits one card; a 70B does not. In fp16, weights alone are \( 70\text{B} \times 2 = 140\ \text{GB} \) — larger than a single 80 GB GPU. Options:

**Option A — tensor-parallel across 4 GPUs (one node, NVLink):**

```bash
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92
```

Weights shard to \( 140/4 = 35\ \text{GB} \) per GPU, leaving each card ~\( 0.92 \times 80 - 35 \approx 38\ \text{GB} \) (minus overhead) for its KV shard. KV is also split by heads across the 4 GPUs, so aggregate KV capacity is roughly 4× a single card's leftover — that is what makes big batches on 70B feasible.

**Option B — quantize to AWQ 4-bit, fit on fewer GPUs:**

```bash
vllm serve casperhansen/llama-3-70b-instruct-awq \
  --quantization awq \
  --tensor-parallel-size 2 \
  --max-model-len 8192
```

4-bit weights are ~\( 70\text{B} \times 0.5 = 35\ \text{GB} \), fitting two 80 GB cards with room for KV. Fewer GPUs, lower cost, at some quality cost — validate on your eval set.

**Option C — two nodes, TP×PP:** `--tensor-parallel-size 8 --pipeline-parallel-size 2` spreads a large model across 16 GPUs, TP within each node over NVLink and PP across the two nodes over the slower inter-node link. Launch via vLLM's Ray-based multi-node path.

Decision order: **fit on one node with TP first; quantize to shrink weights and cut GPU count; go multi-node with PP only when a single node genuinely cannot hold model + working KV.**

---

## Reading the startup logs (your first diagnostic)

Every launch prints the numbers that tell you whether your config is sane. Learn to read them before touching load tests:

```
INFO ... Available KV cache memory: 53.7 GiB
INFO ... GPU KV cache size: 442,368 tokens
INFO ... Maximum concurrency for 8192 tokens per request: 54.0x
INFO ... # GPU blocks: 27648, # CPU blocks: 2048
```

- **Available KV cache memory** — what's left after weights + overhead. If this is tiny or negative-adjacent, lower `max-model-len`, quantize, or add GPUs.
- **GPU KV cache size (tokens)** — your total batch budget in tokens; divide by average request length to estimate real concurrency.
- **Maximum concurrency** — full-context sequences you can run at once. If it's < your expected concurrency, you *will* preempt under load.
- **# CPU blocks** — swap capacity, sized by `--swap-space`.

If you never look at anything else, look at these four lines. They convert the abstract flags into the one number that governs throughput: how many tokens of KV you can hold.

---

## Quick-reference config recipes

| Goal | Starting flags |
|---|---|
| Max throughput, batch/offline | `--gpu-memory-utilization 0.95 --max-num-batched-tokens 16384 --max-num-seqs 512` |
| Low-latency interactive chat | `--max-num-batched-tokens 4096` (protect ITL) `--enable-chunked-prefill`, consider speculative decoding |
| Long-context serving | `--kv-cache-dtype fp8 --max-model-len <needed> --swap-space 16`, cap concurrency |
| Memory-tight single GPU | `--quantization awq` (or `fp8` on Hopper) `--gpu-memory-utilization 0.90` |
| RAG with shared preamble | keep `--enable-prefix-caching` (default), moderate `max-num-seqs` |
| Model bigger than one GPU | `--tensor-parallel-size N` (one node) `[+ --pipeline-parallel-size M across nodes]` |

## Further reading

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023 — the vLLM paper: https://arxiv.org/abs/2309.06180
- ACM DL (SOSP '23 proceedings): https://dl.acm.org/doi/10.1145/3600006.3613165
- vLLM PagedAttention design doc: https://docs.vllm.ai/en/latest/design/paged_attention/
- vLLM Engine Arguments reference: https://docs.vllm.ai/en/stable/configuration/engine_args/
- vLLM Optimization and Tuning (chunked prefill, batching knobs): https://docs.vllm.ai/en/latest/configuration/optimization/
- vLLM Automatic Prefix Caching: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html
- vLLM Distributed Inference (TP/PP): https://docs.vllm.ai/en/latest/serving/distributed_serving.html
- vLLM Quantization overview: https://docs.vllm.ai/en/latest/features/quantization/
- vLLM Speculative Decoding: https://docs.vllm.ai/en/latest/features/spec_decode.html
- vLLM OpenAI-compatible server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- Anyscale, "How continuous batching enables 23x throughput in LLM inference": https://www.anyscale.com/blog/continuous-batching-llm-inference
- Orca (iteration-level scheduling), OSDI 2022: https://www.usenix.org/conference/osdi22/presentation/yu
