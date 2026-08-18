# vLLM Serving — PagedAttention, Continuous Batching, and Production Tuning

## Why this matters

If you serve open-weight LLMs at any real scale, vLLM is very likely the engine underneath — or the baseline everything else is measured against. It became the default high-throughput inference server because it attacked the single biggest bottleneck in LLM serving: **memory**, specifically the KV cache. Its headline result, from the original paper, is a **2–4× throughput improvement at the same latency** versus the prior state of the art (FasterTransformer and Orca).

The insight is almost embarrassingly simple in hindsight: LLM serving was wasting **60–80% of KV-cache memory** to fragmentation and over-reservation. vLLM borrowed *virtual memory and paging* from operating systems, applied it to the KV cache, and turned that wasted memory back into batch capacity. More batch capacity means more requests share each expensive weight-load from GPU memory, which is exactly what raises throughput.

This chapter goes deep on the mechanisms (PagedAttention, continuous batching, prefix caching), the knobs that actually matter in production (`gpu-memory-utilization`, `max-num-seqs`, `max-num-batched-tokens`, chunked prefill), how to scale across GPUs and nodes, and how to trade quality for speed with quantization and speculative decoding. It then covers the 2025–2026 state of the art (V1 engine, disaggregated prefill/decode, EAGLE-family speculation, FP8/INT4-Marlin quantization), a fully worked multi-node tuning walkthrough, real production incidents and their fixes, and an interview-mastery section with 20 Q&A, a system-design prompt, and a red-flag/green-flag table.

**Chapter map**, since this is long: mechanisms and core tuning knobs come first (PagedAttention -> continuous batching -> chunked prefill -> prefix caching -> engine args -> parallelism -> quantization -> speculative decoding); section **(A)** is the 2025-2026 state-of-the-art landscape; section **(B)** is the extended multi-node build-and-tune walkthrough with observability and cost math; section **(C)** is three real production war stories plus an on-call runbook; section **(D)** is interview mastery (60-second explanation, a system-design prompt, 20 Q&A, red/green flags); a glossary and further reading close it out.

> **Saying it out loud.** vLLM became the default open-source serving engine because it attacked the actual bottleneck, which is memory — specifically the KV cache — rather than compute. The insight is embarrassingly simple in hindsight: pre-vLLM systems were wasting 60 to 80% of KV-cache memory to fragmentation and over-reservation, because each sequence grabbed a contiguous block sized for the maximum possible length up front. vLLM borrowed paging from operating systems, applied it to the KV cache, and turned that wasted memory back into batch capacity. And more batch capacity means more requests share each expensive weight read from HBM, which is exactly what raises throughput. The headline number from the paper is 2 to 4x throughput at the same latency versus the prior state of the art.

---

## Core intuition: LLM inference is memory-bound, and the KV cache is the problem

Two facts drive everything:

1. **Autoregressive decoding is memory-bandwidth-bound, not compute-bound.** Generating one token touches the *entire* model weights but does very little arithmetic per token. The GPU spends most of its time reading weights from HBM, not doing math. The fix is **batching**: process many sequences at once so a single weight read serves many tokens. Bigger batch → higher throughput, until you run out of memory.

2. **What runs you out of memory is the KV cache.** For every token in every sequence, attention must remember the key and value vectors of all previous tokens. This "KV cache" grows linearly with sequence length *and* with the number of concurrent sequences. On a typical setup the weights are fixed, and whatever HBM is left is a fixed budget you spend on KV cache. **The more efficiently you pack the KV cache, the bigger your batch, the higher your throughput.**

So the game is: *fit as many sequences' KV caches into the leftover HBM as possible, and keep the GPU busy on all of them at once.* PagedAttention wins the first half; continuous batching wins the second.

> **Saying it out loud.** Two facts drive everything. First, autoregressive decoding is memory-bandwidth-bound, not compute-bound: generating one token reads the entire model weights but does very little arithmetic, so the GPU spends its time on the memory bus while the tensor cores idle. The fix is batching — process many sequences at once so one weight read serves many tokens. Second, the thing that stops you batching more is the KV cache, which grows linearly with sequence length *and* with concurrent sequences. So the game is: fit as many sequences' KV caches into the leftover HBM as possible, and keep the GPU busy on all of them every step. PagedAttention wins the first half, continuous batching wins the second. That's the whole chapter in two sentences.

### How big is the KV cache?

Per token, the KV cache size is:

\[
\text{bytes/token} = 2 \times n_\text{layers} \times n_\text{kv\_heads} \times d_\text{head} \times \text{dtype\_bytes}
\]

The leading \(2\) is for K and V. Note \(n_\text{kv\_heads}\), not the number of query heads — models with **grouped-query attention (GQA)** share KV heads across query heads, which shrinks the cache dramatically.

**Worked numbers (fp16, 2 bytes):**

- **OPT-13B** (40 layers, hidden 5120, full MHA): \(2 \times 40 \times 5120 \times 2 = 819{,}200\) bytes ≈ **800 KB per token**. A single 2048-token sequence needs ~1.6 GB of KV cache — this is the number from the vLLM paper.
- **Llama-3-8B** (32 layers, 8 KV heads, \(d_\text{head}=128\), GQA): \(2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\) bytes = **128 KB per token**. GQA makes it ~6× cheaper than a same-size MHA model.

**Why this matters more every year:** newer model families push KV-per-token even lower via more aggressive GQA/MQA ratios and, in some cases (DeepSeek-V2/V3's Multi-head Latent Attention) a compressed latent KV representation instead of full per-head K/V — because the whole industry has converged on the same conclusion this chapter opened with: KV cache, not weights, is what limits your batch, so every model architecture decision since ~2023 has been under pressure to shrink it. A rough sense of scale, fp16, per token:

| Model family | Attention scheme | Approx. KV bytes/token |
|---|---|---|
| OPT-13B-class (full MHA) | Full multi-head attention | ~800 KB |
| Llama-3-8B (GQA, 8 KV heads) | Grouped-query attention | ~128 KB |
| Llama-3-70B (GQA, 8 KV heads, 80 layers) | Grouped-query attention | ~320 KB |
| Mixtral-8x7B (GQA, MoE FFN — attention KV unaffected by MoE) | Grouped-query attention | ~128 KB |
| DeepSeek-V2/V3-class (Multi-head Latent Attention) | Compressed latent KV cache | Substantially below GQA at comparable model size — the specific figure is architecture-version-dependent; check the model card |

Always compute the real figure for *your* model from its config (layers × KV heads × head dim × dtype bytes) rather than trusting a rule of thumb — the whole point of the formula above is that it's cheap to derive exactly, and it's the single number every other capacity decision in this chapter depends on.

At 800 KB/token, KV cache is enormous and *dynamic* — you don't know a request's final length in advance. That combination is exactly what classic allocators handle badly.

> **Saying it out loud.** The formula is two — for keys and values — times layers, times *KV* heads, times head dimension, times bytes per element, per token. The word doing the work there is KV heads rather than query heads, because grouped-query attention shares KV heads across query heads and shrinks the cache dramatically. Concretely, in fp16: an OPT-13B-class model with full multi-head attention is about 800 kilobytes per token, so a single 2,048-token sequence needs 1.6 gigabytes. Llama-3-8B with GQA and 8 KV heads is 128 kilobytes per token — roughly six times cheaper. That's why every architecture since about 2023 has been under pressure to shrink this number. The habit worth building: derive the real figure from your model's config rather than trusting a rule of thumb, because every capacity decision downstream depends on it.

---

## PagedAttention in depth

### The problem it solves: fragmentation and over-reservation

Pre-vLLM systems (Orca, FasterTransformer) stored each sequence's KV cache in **one contiguous chunk of memory, sized to the maximum possible length**. If your model supports 2048 tokens, every sequence reserved space for 2048 tokens the moment it started — even if it only ever generated 30. Three kinds of waste result, and the paper measures them directly:

- **Internal fragmentation** (13.3%–57.3%): the slot reserved for max length but never filled.
- **Reservation waste** (part of 25.2%–96.3%): space reserved for future tokens of a still-running sequence — technically "will be used" but idle *now*, so it can't serve other requests.
- **External fragmentation**: gaps between contiguous chunks of different sizes that no new sequence fits into.

Net effect: measured **effective KV utilization of only 20.4%–38.2%**. Four out of five bytes wasted.

> **Saying it out loud.** Before vLLM, each sequence's KV cache lived in one contiguous chunk sized to the maximum possible length. So if your model supports 2,048 tokens, every request reserved room for 2,048 the moment it started — even if it generated thirty. That produces three kinds of waste: internal fragmentation from the reserved-but-never-filled slot, reservation waste from space held for a running sequence's future tokens, and external fragmentation from gaps between differently-sized chunks that nothing fits into. The paper measured the damage directly: effective KV utilization of only 20 to 38 percent. Four out of five bytes wasted — on the exact resource that caps your batch size, which caps your throughput.

### The idea: page the KV cache like virtual memory

Operating systems solved this decades ago. A process sees a contiguous virtual address space, but physically it's scattered across fixed-size **pages** mapped by a **page table**. No process reserves all of physical RAM up front; pages are handed out on demand.

PagedAttention does the same for the KV cache:

- The KV cache of a sequence is split into fixed-size **KV blocks**, each holding the K and V vectors for a fixed number of tokens — the **block size, default 16 tokens**.
- Blocks live in a global pool of physical GPU memory and **need not be contiguous**.
- Each sequence has a **block table** mapping its logical block index → physical block number, exactly like a page table.
- The attention kernel is modified to gather K/V from these scattered blocks using the block table, so attention runs correctly over non-contiguous memory.

> **Saying it out loud.** The fix is borrowed straight from operating systems. A process sees a contiguous virtual address space, but physically its memory is scattered across fixed-size pages tracked by a page table, and no process reserves all of RAM up front. PagedAttention does exactly that for the KV cache: a sequence's cache is split into fixed-size blocks — sixteen tokens by default — those blocks live in a global pool and don't need to be contiguous, and each sequence has a block table mapping logical block index to physical block number. The attention kernel is modified to gather K and V through that block table. The payoff: a block is allocated only when the current one fills, so you waste at most one partial block per sequence and zero reservation, taking effective utilization from about 20% up toward 96%.

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

> **Saying it out loud.** Because blocks are indirected through a table, two sequences can just point at the same physical block — and that's where paging pays a second dividend. In parallel sampling or beam search, n outputs share the same prompt, so instead of n copies of the prompt's KV cache you have n block tables pointing at one set of blocks; the paper reports up to 55% memory savings there. When one sharer needs to diverge, vLLM copies just that one block and updates only that sequence's table — literally the same copy-on-write trick `fork()` uses, with reference counts per block. This block-level sharing isn't a side feature; it's the machinery that automatic prefix caching is built on top of.

### Worked memory example

Serve **Llama-3-8B in fp16 on one A100-80GB**.

1. **Weights:** \(8\text{B} \times 2\ \text{bytes} = 16\ \text{GB}\).
2. **Budget:** `--gpu-memory-utilization 0.9` → vLLM may use \(0.9 \times 80 = 72\ \text{GB}\).
3. **Non-KV overhead:** CUDA context, activations, CUDA graphs — call it ~2 GB.
4. **KV cache pool:** \(72 - 16 - 2 = 54\ \text{GB}\).
5. **Per token:** 128 KB (from above). **Per block (16 tokens):** \(16 \times 128\ \text{KB} = 2\ \text{MB}\).
6. **Total blocks:** \(54\ \text{GB} / 2\ \text{MB} \approx 27{,}600\) blocks = **~442,000 tokens** of KV capacity.

That single number, ~442k tokens, is your batch budget. It can be **54 sequences at the full 8192-context** (\(442000/8192\)), or **~880 concurrent chatbot turns** averaging 500 tokens each, or anything in between. vLLM logs this at startup as `# GPU blocks: 27600` and reports "Maximum concurrency for 8192 tokens per request." Watch that log line — it tells you exactly how much headroom you bought.

> **Saying it out loud.** Walk the arithmetic for Llama-3-8B in fp16 on one 80-gigabyte A100. Weights are 16 gigabytes. At `--gpu-memory-utilization 0.9` vLLM may use 72. Subtract about two for CUDA context, activations, and graph capture, and you're left with 54 gigabytes of KV pool. At 128 kilobytes per token, a sixteen-token block is 2 megabytes, so 54 gigabytes is about 27,600 blocks, or roughly 442,000 tokens of KV capacity. That one number is your entire batch budget — it can be 54 sequences at full 8K context, or about 880 concurrent chat turns averaging 500 tokens, or anything between. vLLM prints it at startup as `# GPU blocks`, and watching that log line is the fastest way to know whether your config bought you the headroom you thought it did.

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

> **Saying it out loud.** The key move is that vLLM schedules at the granularity of a single decode step rather than a whole request. Every forward pass, any sequence that just emitted its stop token is evicted immediately and its KV blocks freed, waiting requests are admitted mid-flight to fill the vacated slots, and the next pass runs over the new mix. So no sequence ever waits on a slower sibling — which is exactly the head-of-line blocking that kills static batching, where a request emitting twenty tokens sits idle for 480 steps while its batch-mate finishes. The measured win from Anyscale's widely cited benchmark is up to 23x throughput versus naive batching while *also* cutting p50 latency, which is rare — you get both because higher utilization means less queueing.

---

## Prefill vs decode, and chunked prefill

A request has two phases with opposite performance profiles:

- **Prefill** — process the whole prompt in one big parallel pass. Compute-bound, high FLOPs, fills the pipeline. A 4000-token prompt is one heavy step.
- **Decode** — generate tokens one at a time. Memory-bound, tiny per-step compute.

Mixing them is awkward. A giant prefill can **monopolize a forward pass and stall every decoding sequence**, spiking inter-token latency (ITL) for everyone already streaming. This is the classic prefill/decode interference — and, as section (A) below covers, it is the exact motivation for physically separating prefill and decode onto different GPUs in 2025–2026 deployments.

**Chunked prefill** (`--enable-chunked-prefill`) splits a large prefill into token-sized chunks and **co-schedules a prefill chunk alongside ongoing decodes** in the same batch, bounded by `max-num-batched-tokens`. Benefits:

- Smooths out ITL — decodes no longer freeze behind a monster prompt.
- Improves GPU utilization — decode steps are compute-light, so padding the batch with prefill tokens uses otherwise-idle FLOPs.
- In modern vLLM (V1 engine) chunked prefill is **on by default**, and prefill/decode are unified in one scheduler.

Tuning: **raise `max-num-batched-tokens` for throughput** (bigger chunks, more prefill work per step); **lower it to protect decode latency** (smaller chunks yield to decodes more often).

> **Saying it out loud.** Prefill and decode have opposite profiles: prefill processes the whole prompt in one heavy compute-bound pass, decode generates one token at a time and is memory-bound with almost no arithmetic. Mixing them naively is awkward, because one giant prefill can monopolize a forward pass and freeze every sequence that's currently streaming — that's the classic prefill/decode interference, and it shows up as a spike in inter-token latency for users who did nothing wrong. Chunked prefill splits a big prompt into pieces and co-schedules a chunk alongside ongoing decodes in the same batch. It smooths inter-token latency and it also uses otherwise-idle FLOPs, since decode steps are compute-light. The tuning rule: raise `max-num-batched-tokens` for throughput, lower it to protect decode latency.

---

## Prefix caching (automatic KV reuse)

Many requests share a prefix: the same long system prompt, a shared few-shot preamble, a document everyone asks questions about, a multi-turn conversation where each turn re-sends the history.

**Automatic prefix caching** (`--enable-prefix-caching`) hashes KV blocks by their content (and the tokens preceding them). When a new request's prefix hashes to blocks already in the cache, vLLM **skips recomputing that prefill entirely** and points the new sequence's block table at the cached blocks. It's the block-sharing / CoW machinery from PagedAttention, applied across requests and over time.

- **When it wins big:** long shared system prompts, RAG with a fixed instruction preamble, multi-turn chat (each turn reuses the whole prior conversation's KV), agent loops that resend context.
- **Cost:** cached blocks occupy KV memory that could otherwise hold active batch. Under memory pressure, cached prefix blocks are evicted LRU. It's a hit-rate bet — near-free when hits are common, mild overhead when they're not.
- In current vLLM (V1) prefix caching is **enabled by default**, and the V1 rewrite specifically made it "near-zero performance degradation, even when the cache hit rate is 0%" — so leaving it on is close to a free option (see section A).

The saving is real work avoided: a 2000-token shared system prompt cached across 1000 requests skips ~2,000,000 tokens of prefill compute.

> **Saying it out loud.** Lots of requests share a prefix — the same long system prompt, a fixed few-shot preamble, a document everyone's asking about, or a multi-turn chat that resends the whole history each turn. Automatic prefix caching hashes KV blocks by content and, when a new request's prefix hashes to blocks already resident, just points the new sequence's block table at them and skips recomputing that prefill entirely. It's literally the copy-on-write block sharing from PagedAttention, applied across requests and across time. The scale of the saving is real work avoided: a 2,000-token shared system prompt cached across a thousand requests skips two million tokens of prefill compute. The cost is that cached blocks occupy memory the active batch could use, so it's a hit-rate bet — though in V1 it's engineered to be near-free even at a 0% hit rate.

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
| `--quantization` | none | Weight/activation quant scheme (`awq`, `awq_marlin`, `gptq`, `gptq_marlin`, `fp8`, `bitsandbytes`, ...). | Use to shrink weights → more KV room / smaller GPU. Costs some quality. |
| `--kv-cache-dtype` | `auto` | Store KV cache in `fp8` etc. | `fp8` ~halves KV memory → bigger batch/context; small accuracy cost. |
| `--swap-space` | `4` (GiB/GPU) | CPU RAM for swapping out preempted sequences' KV. | Raise if you see frequent preemption + recompute; swap can be cheaper than recompute for long sequences. |
| `--max-num-seqs` + `--max-num-batched-tokens` together | — | The two levers that shape the batch. | Co-tune: token budget caps work/step; seq budget caps concurrency. |
| `--dtype` | `auto` | Compute dtype (`bfloat16`, `float16`). | `bfloat16` on Ampere+; matters for numerical stability. |
| `--speculative-config` | none | Speculative decoding config (JSON): draft model, n-gram, or EAGLE/Medusa method. | See below — latency win when acceptance is high. |

Startup log lines to watch: `# GPU blocks:` (your KV capacity), `Maximum concurrency for N tokens`, and any `Sequence group ... is preempted` warnings (you're memory-starved). vLLM also exposes these as **live Prometheus metrics** (`vllm:gpu_cache_usage_perc`, `vllm:gpu_prefix_cache_hit_rate`, `vllm:num_requests_running`, `vllm:num_requests_waiting`) on `/metrics` — section (B) below shows how to read them while tuning.

> **Saying it out loud.** There are really five flags that matter and the rest is detail. `--gpu-memory-utilization` sets what fraction of HBM vLLM may claim, which determines your KV pool — push it toward 0.94 if you have headroom, back off if you OOM. `--max-num-seqs` caps concurrency, but here's the thing people get wrong: KV memory usually binds *before* that number does, so raising it without memory headroom just causes preemption. `--max-num-batched-tokens` is your throughput-versus-inter-token-latency dial. `--max-model-len` directly bounds worst-case KV per sequence, so lowering it is often the cheapest way to fit more requests. And `--kv-cache-dtype fp8` roughly doubles KV capacity for a small accuracy cost. Watch the startup log for `# GPU blocks` and watch for `preempted` warnings — those two tell you whether your settings are honest.

---

## Parallelism for big models

When a model (plus its KV cache) doesn't fit on one GPU, or single-GPU latency is too high, split it.

> **Saying it out loud.** When a model plus its KV cache doesn't fit on one GPU, you split it, and there are two ways with very different communication profiles. Tensor parallelism shards every layer's weight matrices across N GPUs and does an all-reduce each layer to combine partial results — bandwidth-hungry, so it wants NVLink and belongs inside one node. Pipeline parallelism assigns contiguous stages of layers to different GPUs, so communication is just a small point-to-point handoff between stages, which tolerates slower links and lets you span nodes. The rule of thumb is: tensor parallel first, up to one node's worth of GPUs, then pipeline parallel to cross nodes. A very common shape is TP=8 within each node times PP=2 across two nodes for sixteen GPUs total.

### Tensor parallelism (TP) — `--tensor-parallel-size`

**Mechanism:** shard every layer's weight matrices across N GPUs; each GPU computes its slice, and an **all-reduce** combines partial results each layer. The KV cache is also sharded (by heads), so TP grows your KV budget too.

**When:** the model is too big for one GPU, or you want lower latency on a single request (more GPUs working the same forward pass). Best **within one node over NVLink**, because the per-layer all-reduce is bandwidth-hungry.

**Tradeoff:** communication overhead grows with N; going cross-node over slower interconnect (Ethernet/PCIe) tanks efficiency. Keep `-tp` ≤ GPUs-per-node. TP size must divide the number of attention heads.

> **Saying it out loud.** Tensor parallelism shards each layer's weight matrices across GPUs — every GPU computes its slice and an all-reduce combines the partial results, once per layer. Two things worth knowing beyond the mechanism. It shards the KV cache too, by heads, so raising TP grows your KV budget as well as fitting bigger weights. And it lowers single-request latency, because more GPUs are working the same forward pass. The tradeoff is that all-reduce traffic grows with the degree, so going cross-node over Ethernet or PCIe rather than NVLink tanks efficiency — keep TP at or below your GPUs-per-node. One hard constraint people forget: the TP size has to divide the number of attention heads.

### Pipeline parallelism (PP) — `--pipeline-parallel-size`

**Mechanism:** assign contiguous *stages* of layers to different GPUs; activations flow stage → stage. Communication is a small point-to-point hand-off between stages, tolerant of slower links.

**When:** to span **multiple nodes**, or to fit truly huge models where even TP-across-a-node isn't enough. Common pattern: `-tp 8` within each node × `-pp 2` across two nodes = 16 GPUs.

**Tradeoff:** introduces pipeline bubbles (stages idle waiting for the previous stage); throughput-friendly with enough in-flight requests, but adds latency per request. Combine TP (intra-node) + PP (inter-node) for the best of both.

Rule of thumb: **TP first, up to one node; PP to cross nodes.** vLLM also supports data parallelism / multi-replica behind a router for pure scale-out.

> **Saying it out loud.** Pipeline parallelism cuts the model by layers instead of by weight matrices — GPU zero holds the first block of layers, GPU one the next, and activations flow from stage to stage. The communication is a small point-to-point handoff rather than an all-reduce every layer, which is why it tolerates slower inter-node links and is the right tool for spanning nodes. The cost is pipeline bubbles: a stage sits idle waiting on the one before it, so per-request latency goes up even though throughput holds up fine once you have enough requests in flight to keep every stage fed. That's the honest framing — pipeline parallelism is throughput-friendly and latency-unfriendly, which is why you use it to cross nodes and tensor parallelism inside them.

---

## Quantization — trade quality for memory and speed

Quantization shrinks weights (and optionally activations/KV) to fewer bits. Smaller weights free HBM for KV cache and can speed up the memory-bound decode. All of it costs *some* accuracy; how much depends on scheme and model.

| Scheme | Bits | What's quantized | When to use | Tradeoff |
|---|---|---|---|---|
| **AWQ (+ Marlin kernel)** | 4-bit | Weights only (activation-aware, protects salient weights) | Serving throughput on Ampere/Ada; strong quality at 4-bit | Needs a pre-quantized AWQ checkpoint; without a fast kernel INT4×FP16 GEMM only wins at batch size 1 (see below) |
| **GPTQ (+ Marlin kernel)** | 3/4/8-bit | Weights only (2nd-order error minimization) | Broad hardware/checkpoint availability | Quality can degrade at 3-bit; per-model calibration sensitivity |
| **FP8** (E4M3) | 8-bit | Weights + activations (and KV via `--kv-cache-dtype fp8`) | **Hopper/H100, Ada** with hardware FP8; near-lossless, high throughput | Requires FP8-capable GPUs for full speedup |
| **INT8 (SmoothQuant/W8A8)** | 8-bit | Weights + activations | Good quality/speed balance where FP8 HW absent | More setup; less dramatic memory savings than 4-bit |
| **bitsandbytes** | 4/8-bit | Weights, on-the-fly | Quick experiments, no pre-quant step | Slower kernels; not the throughput champion |
| **NVFP4 (via NVIDIA Model Optimizer)** | 4-bit float | Weights + activations, Blackwell tensor-core native | Newest Blackwell (B200/GB200)-class hardware | Newest path — check current vLLM docs for model/kernel coverage before depending on it in production |

> **Saying it out loud.** Quantization shrinks weights to fewer bits, which frees HBM for KV cache and can speed up memory-bound decode — and it always costs some accuracy, the only question is how much. The practical guidance splits by hardware. On Ampere or Ada, AWQ 4-bit with the Marlin kernel is the workhorse: roughly 4x less weight memory, which is a lot of KV headroom. On Hopper, prefer FP8 — it's near-lossless and uses native tensor-core FP8 for a real speedup, and you can add `--kv-cache-dtype fp8` to roughly double KV capacity on top. The thing to say out loud that separates a good answer: quantization shrinks *weights*, not the KV cache, so for long-context blowup the relevant levers are KV dtype and `--max-model-len`, not weight quantization.

### Why a 4-bit checkpoint alone doesn't make inference fast: the Marlin kernel

Quantizing weights to INT4 shrinks *memory*, but it does not automatically make *compute* faster. Running INT4 weights against FP16 activations means an unusual mixed-precision GEMM, and naive kernels for it only beat FP16 at **batch size 1** — the exact regime where you're not serving many users. At realistic serving batch sizes (8–32+ concurrent sequences), a poorly-implemented INT4 kernel leaves tensor cores underused and the 4-bit weight advantage evaporates.

**Marlin** (Frantar et al., "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models") is the kernel that fixes this: it achieves close to the **ideal 4× speedup for batch sizes up to ~32 tokens**, by using tricks like asynchronous global-to-shared memory copies and careful weight layout so tensor cores stay busy across the whole realistic batch range, not just batch 1. vLLM ships Marlin-backed kernels as `awq_marlin` and `gptq_marlin` — when you pass `--quantization awq` or `gptq` today, vLLM auto-selects the Marlin kernel path where available, which is why a 2024-era checkpoint quantized with plain AutoAWQ still serves fast in current vLLM.

Guidance:

- **Memory-constrained, throughput-focused, Ampere/Ada:** AWQ 4-bit (Marlin kernel) is the workhorse — cuts weight memory ~4×, freeing large KV headroom, and holds close to ideal speedup up to moderate batch sizes.
- **H100 / Hopper:** prefer **FP8** — near-lossless and uses native tensor-core FP8 for real speedups; add `--kv-cache-dtype fp8` to roughly double KV capacity.
- **Blackwell (B200/GB200):** NVFP4 via NVIDIA Model Optimizer is the emerging frontier for 4-bit that's native to the hardware rather than dequantized on the fly — newer and less battle-tested than AWQ/FP8, validate carefully.
- **Quality-sensitive tasks (code, math, long reasoning):** measure. 4-bit weight-only can visibly hurt; validate on *your* eval set, not just perplexity (see the war story in section C).
- Quantization reduces *weight* memory, not KV — for long-context blowup, `--kv-cache-dtype fp8` and `--max-model-len` are the relevant levers.

> **Saying it out loud.** This is a genuinely non-obvious point: quantizing weights to INT4 shrinks memory, but it does not automatically make compute faster. Running INT4 weights against FP16 activations is an unusual mixed-precision matmul, and naive kernels for it only beat FP16 at *batch size one* — which is exactly the regime where you're not serving anybody. At realistic serving batches of eight to thirty-two concurrent sequences, a bad INT4 kernel leaves the tensor cores underused and the whole 4-bit advantage evaporates. Marlin is the kernel that fixes it, holding close to the ideal 4x speedup out to about batch 32 via asynchronous global-to-shared copies and careful weight layout. vLLM auto-selects the Marlin path when you ask for AWQ or GPTQ, which is why a 2024-era checkpoint still serves fast today.

---

## Speculative decoding — lower latency, not more throughput

**Mechanism:** a cheap **draft** proposes several tokens ahead; the big **target** model verifies them all in one forward pass. Accepted tokens are kept; the first rejection resets to the target's own token. Because verification is parallel, a good draft yields **multiple tokens per target forward pass** — output is provably identical in distribution to the target alone (it's exact, not approximate).

vLLM supports several draft sources, each with a different cost/benefit:

- **A small draft model** — a separate, much smaller checkpoint from the same family runs ahead of the target. Simple, but needs a compatible small model and its own (tiny) forward-pass cost.
- **n-gram / prompt-lookup decoding** — instead of a neural draft, propose tokens by matching repeated n-grams already seen in the prompt or generation so far. Free of extra model weights; shines on code, tool-call replays, and RAG where the output echoes the input verbatim.
- **EAGLE / EAGLE-3 / EAGLE-3.1** — a lightweight draft **head** attached to the target model's own hidden states, trained to predict the next few tokens using the target's internal representations rather than a fully independent model. Because it reuses target hidden states, EAGLE gets much higher acceptance length than an independent draft model of similar size. EAGLE-3.1 (vLLM blog, 2026-05-26) is a joint release between the EAGLE authors, the vLLM team, and the TorchSpec project that fixes an "attention drift" problem in deep speculation via FC-normalization and post-norm hidden-state feedback, reporting **up to 2× longer accepted length** than EAGLE-3 on long-context workloads and **~2.03× higher per-user throughput at single concurrency**, tapering to **~1.7× at concurrency 4** and **~1.66× at concurrency 16** on the Kimi K2.6 model — a clean illustration that speculative gains shrink as the batch (and GPU compute saturation) grows.
- **Medusa** — multiple parallel decoding heads trained on top of the target model, each predicting a token at a fixed future offset; verified in the same forward pass as EAGLE-style methods. Simpler training setup than EAGLE, generally slightly lower acceptance.

**When to use:** latency-sensitive, **low-to-moderate batch** serving where the GPU has spare compute (decode is memory-bound, so verification is nearly free). Interactive chat, single-user, or bursty low-QPS endpoints, and — per the EAGLE-3.1 numbers above — still worthwhile at moderate concurrency, just with diminishing returns.

**Tradeoff:** the win depends entirely on **acceptance rate**. Low acceptance means you paid for drafting and got little back — it can *reduce* throughput. And under **high batch load the GPU is already compute-saturated**, so speculation's "free" parallel verification isn't free anymore — its benefit shrinks or reverses. Rule: speculate when you're latency-bound and under-batched; skip it when you're throughput-bound and saturated. vLLM's `speculators` project (v0.3.0, vLLM blog 2025-12-13) now also supports **training** your own draft/EAGLE heads against a target model, rather than relying only on community-published drafts — useful if your traffic distribution doesn't match the drafts published for a given base model.

> **Saying it out loud.** The mechanism is: a cheap draft proposes several tokens ahead, the big target model verifies all of them in one forward pass, accepted tokens are kept and the first rejection resets. Because verification is parallel, a good draft yields multiple tokens per target forward pass — and critically the output is provably identical in distribution to the target alone, so it's exact, not an approximation. But the title is the important part. Speculation helps when you're latency-bound and under-batched, because decode is memory-bound and the spare compute makes verification nearly free. Under high batch load the GPU is already compute-saturated, so that free lunch disappears — the EAGLE-3.1 numbers show it: roughly 2x at concurrency one, tapering to about 1.66x at concurrency sixteen. Low acceptance rate can make it a net loss.

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

> **Saying it out loud.** The baseline launch is genuinely one command — `vllm serve` with a model ID, a host and port, and a couple of memory flags — and what you get is an OpenAI-compatible HTTP server, so any existing SDK works by just repointing `base_url`. That API compatibility is a bigger deal than it sounds, because it means migrating off a hosted provider is a config change rather than a rewrite. What you should do immediately after launching, before touching any tuning, is read the startup logs: available KV cache memory, GPU KV cache size in tokens, and maximum concurrency. Those three lines convert your abstract flags into the one number that governs throughput — how many tokens of KV you can actually hold.

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

> **Saying it out loud.** Tuning is symptom-driven, and the discipline is one knob at a time with a load test between each. If you have a big shared system-and-retrieval preamble, keep prefix caching on — hit rate is high and prefill work drops sharply. If the GPU has headroom and you're not OOM-ing, push memory utilization toward 0.94 and raise both the sequence and token budgets. If p99 inter-token latency is bad, *lower* `max-num-batched-tokens`, because smaller prefill chunks yield to decodes more often at a small throughput cost. If you see `preempted` warnings, you've over-committed KV — drop `max-num-seqs` or `max-model-len`, or add swap space. And you stop when you hit either preemption or your latency SLO, whichever comes first.

---

## (A) The 2025–2026 landscape

vLLM has moved fast since the original SOSP '23 paper. If you last read the codebase in 2023–2024, four things have materially changed the picture — an interviewer who works in this space will expect you to know all four, with rough dates.

> **Saying it out loud.** If you last looked at vLLM in 2023, four things have materially changed the picture. The V1 engine is a from-scratch rewrite with a unified scheduler and an isolated engine process, and it made chunked prefill and prefix caching default-on. Disaggregated prefill/decode moved from experimental flag to a real pattern, splitting the two phases onto physically separate GPU pools. Speculative decoding grew from "small draft model" into a family with EAGLE-3 as the acceptance-length leader. And quantization consolidated on Marlin-backed 4-bit for Ampere and FP8 for Hopper. The practical version: if a tutorial tells you to manually pass `--enable-chunked-prefill`, that advice is stale, and repeating it in an interview signals your mental model is frozen at the paper.

### A.1 The V1 engine rewrite

**vLLM V1** ("V1: A Major Upgrade to vLLM's Core Architecture," vLLM blog, **2025-01-27**) is a from-scratch rewrite of the engine core, not an incremental patch, and it's the default engine in current releases. The headline changes:

- **Unified scheduler.** V0 had separate code paths and heuristics for prefill-only steps, decode-only steps, and chunked-prefill mixing. V1 represents every request uniformly as `{request_id: num_tokens}` and schedules prefill tokens and decode tokens through one code path — this is what makes chunked prefill, prefix caching, and speculative decoding compose cleanly instead of each needing bespoke scheduler logic.
- **Isolated `EngineCore` process.** The scheduler/model-execution loop runs in its own process, decoupled from the API server, communicating over a fast IPC path. Tokenization, de-tokenization, and request bookkeeping — all CPU work — overlap with GPU execution instead of serializing with it. This matters more every year as GPUs get faster and CPU-side overhead becomes the bottleneck it never used to be.
- **On by default:** chunked prefill and prefix caching are default-on in V1, and prefix caching in particular was specifically engineered to add "near-zero performance degradation, even when the cache hit rate is 0%" — so there's little reason to ever turn it off defensively.
- **Measured gains:** vLLM's own numbers show up to **1.7× higher throughput vs V0** on Llama-3.1-8B/70B text serving, with **even larger gains on vision-language models** (e.g. Qwen2-VL) due to multimodal-specific scheduling improvements.

Practical implication: if you're reading a pre-2025 vLLM tutorial telling you to manually pass `--enable-chunked-prefill` or explain why prefix caching costs overhead at 0% hit rate, that advice is stale — verify against the current `vllm serve --help` and the V1 guide before repeating it in an interview.

> **Saying it out loud.** V1 is a rewrite of the engine core, not a patch, and it's the default now. Two structural changes matter. The scheduler is unified: V0 had separate code paths for prefill-only steps, decode-only steps, and chunked mixing, whereas V1 represents every request uniformly as a request-ID-to-token-count map and schedules both through one path — which is what lets chunked prefill, prefix caching, and speculation compose cleanly instead of each needing bespoke logic. And the engine core runs in its own process, so tokenization, detokenization, and bookkeeping — all CPU work — overlap with GPU execution instead of serializing with it. That second one matters more every year, because as GPUs get faster the CPU side becomes a bottleneck it never used to be. Measured gain: up to 1.7x throughput over V0.

### A.2 Disaggregated prefill/decode serving

Co-locating prefill (compute-bound) and decode (memory-bound) on the same GPUs is simple but creates exactly the interference chunked prefill only partially smooths over: a burst of long prompts still steals cycles from in-flight decodes, spiking ITL.

**Disaggregated serving** takes this further: prefill and decode run on **physically separate GPU pools**, with the KV cache computed during prefill **transferred** to the decode pool (over NVLink/RDMA) rather than recomputed. A **connector/proxy** layer routes each request through prefill first, then decode, streaming KV in between. This has moved from "experimental" (an early disaggregated-prefill feature flag shipped as far back as v0.7.x) to a first-class pattern with dedicated KV-transfer connectors.

A concrete, dated example: AMD's **MORI-IO KV connector** (vLLM blog, **2026-04-07**, "Next-Level Inference: Why Your Single-Node vLLM Setup Needs Prefill-Decode Disaggregation") splits one 8-GPU node into a 4-GPU prefill pool and a 4-GPU decode pool, with an RDMA-based connector transferring KV cache in either **read mode** (decode pulls KV after prefill finishes) or **write mode** (prefill streams KV concurrently as it computes). Reported result: **~2.5× higher goodput** than standard collocated serving on identical hardware, at the cost of somewhat higher TTFT (the request now hops between two GPU pools) in exchange for much more stable ITL. PyTorch's own engineering blog ("Disaggregated Inference at Scale with PyTorch & vLLM") and Ray Serve's LLM docs (`prefill-decode.html`) describe the same pattern at larger, multi-node scale.

**When it's worth the complexity:** high-scale deployments with long, variable prompts and a strict ITL/streaming-smoothness SLO, where you can afford separate autoscaling groups for prefill vs decode. For most single-node, moderate-QPS deployments, chunked prefill co-location is simpler and sufficient — disaggregation is an optimization you reach for once co-located tuning has plateaued.

> **Saying it out loud.** Chunked prefill smooths prefill/decode interference but doesn't eliminate it — a burst of long prompts still steals cycles from in-flight decodes. Disaggregation takes it further: prefill and decode run on physically separate GPU pools, and the KV cache computed during prefill is *transferred* over NVLink or RDMA to the decode pool rather than recomputed. AMD's MORI-IO connector work splits one 8-GPU node into a 4-GPU prefill pool and a 4-GPU decode pool and reports about 2.5x higher goodput on identical hardware. The tradeoff is explicit and worth naming: TTFT goes *up*, because the request now hops between two pools, in exchange for much more stable inter-token latency. For a single-node moderate-QPS deployment, co-located chunked prefill is simpler and sufficient.

### A.3 Speculative decoding, current state

As detailed in the Speculative Decoding section above, the field has moved from "small draft model" as the default mental model to a family of methods, with **EAGLE-3 / EAGLE-3.1** as the current state of the art for acceptance length (see the 2026-05-26 vLLM blog for the joint EAGLE/vLLM/TorchSpec numbers), **Medusa** as a simpler parallel-head alternative, and **n-gram/prompt-lookup** as a zero-extra-weights option for code/RAG-style repetition. AMD's Quark toolchain now also supports training and serving EAGLE-3 drafters on Instinct GPUs (vLLM blog, 2026-07-13), and Red Hat's developer blog (2026-04-16) documents concrete speedups applying speculative decoding to gpt-oss-style open models — evidence this is now routine production tooling, not a research curiosity.

> **Saying it out loud.** The mental model has moved on from "speculative decoding means a small draft model." There are three families now. A separate small draft model is the classic, simple but needs a compatible checkpoint. N-gram or prompt-lookup drafting uses no extra weights at all — it just proposes tokens by matching repeated n-grams already in the prompt, which shines on code, tool-call replays, and RAG where output echoes input verbatim. And EAGLE-family methods attach a lightweight draft *head* to the target model's own hidden states, which is why they get much higher acceptance length than an independent draft of similar size. EAGLE-3.1 is the current state of the art. Tooling has caught up too — vLLM's `speculators` project can now train your own drafts against your traffic distribution.

### A.4 Quantization, current state

`awq_marlin` and `gptq_marlin` (the Marlin-kernel-backed paths, see the Quantization section above) are the default fast path for 4-bit weight-only quantization on Ampere/Ada/Hopper today — vLLM auto-selects them when you request `awq`/`gptq` if the kernel is available for your hardware. FP8 W8A8 (via LLM Compressor / `llmcompressor`) is the standard Hopper-class recipe. NVIDIA Model Optimizer integration brings **NVFP4** onto vLLM's roadmap for Blackwell-class (B200/GB200) hardware — check the current `docs.vllm.ai/en/latest/features/quantization/` page for exact model/kernel coverage before committing to it, since this is the newest and fastest-moving corner of the quantization stack.

> **Saying it out loud.** The current picture is simple enough to state in one breath. On Ampere, Ada, and Hopper, the Marlin-backed paths — `awq_marlin` and `gptq_marlin` — are the default fast path for 4-bit weight-only, and vLLM auto-selects them when you ask for AWQ or GPTQ if the kernel exists for your hardware. On Hopper-class, FP8 W8A8 via LLM Compressor is the standard recipe and is close to lossless in practice. And NVFP4 through NVIDIA's Model Optimizer is the emerging Blackwell-native 4-bit path. The caveat to attach every time: this is the fastest-moving corner of the stack, so check the current quantization docs for exact model and kernel coverage before you commit a production deployment to any of it.

### A.5 vLLM vs SGLang vs TensorRT-LLM, today

The three serious open-source-adjacent engines have converged on the same core ideas (paged KV, continuous/in-flight batching) and now differentiate on **specialization and operational cost**:

| Dimension | **vLLM** | **SGLang** | **TensorRT-LLM** |
|---|---|---|---|
| Core differentiator | Broadest model/hardware support, fastest to deploy, strong default performance | **RadixAttention** — a radix-tree KV cache index built specifically to maximize prefix-sharing across requests | Ahead-of-time compiled kernels/engines for peak NVIDIA-GPU performance |
| Best workload fit | General-purpose serving, fast iteration across many model families | Heavy prefix-sharing workloads: chatbots, RAG, agent loops, few-shot prompting | Fixed, long-lived production model where engine-build cost amortizes |
| Setup / cold start | Low — single `pip install` / one command; ~1 minute cold start | Low — comparable to vLLM | **High** — per-model, per-shape engine compilation; can take tens of minutes |
| Hardware | NVIDIA + AMD ROCm + others | Primarily NVIDIA, growing ROCm support | NVIDIA only |
| Quantization | AWQ/GPTQ (Marlin), FP8, INT8, bnb, emerging NVFP4 | AWQ, GPTQ, FP8 | INT4/INT8, FP8 — deeply kernel-optimized |
| Speculative decoding | Draft model, n-gram, EAGLE/EAGLE-3.1, Medusa | EAGLE, other draft methods | Draft model, EAGLE, Medusa |

A representative **third-party** benchmark (Spheron Blog, "vLLM vs TensorRT-LLM vs SGLang: Which Is Fastest? H100 Benchmarks," dated **2026-03-23**; single H100 SXM5 80GB, Llama-3.3-70B-Instruct at FP8, 50 concurrent requests) reported: **TensorRT-LLM ≈ 2,100 tok/s**, **SGLang ≈ 1,920 tok/s**, **vLLM ≈ 1,850 tok/s** on raw output throughput, with TTFT p50 at 10 requests of **105 ms / 112 ms / 120 ms** respectively — but a **~28-minute TensorRT-LLM engine-compilation cold start** versus ~1 minute for vLLM and SGLang. Treat any single third-party number like this as a snapshot, not gospel — engines change fast, and you should always benchmark on your own model, hardware, and traffic shape before deciding. The qualitative conclusion that has held up across most 2025–2026 write-ups: **TensorRT-LLM wins raw throughput/latency on fixed NVIDIA hardware at the cost of build complexity and vendor lock-in; SGLang wins when prefix-sharing dominates your traffic (its RadixAttention is purpose-built for exactly that); vLLM remains the pragmatic default** for broad model support, hardware flexibility, and fast time-to-serve.

> **Saying it out loud.** All three have converged on paged KV and continuous batching, so they now differentiate on specialization and operational cost. TensorRT-LLM wins raw throughput and latency on fixed NVIDIA hardware — a representative H100 benchmark put it around 2,100 tokens per second against SGLang's 1,920 and vLLM's 1,850 — but it pays for that with a roughly 28-minute per-model engine compilation versus about a minute for the other two, plus NVIDIA-only lock-in. SGLang's differentiator is RadixAttention, a radix-tree KV index purpose-built to maximize prefix sharing, so it wins when your traffic is chatbots, RAG, or agent loops. And vLLM stays the pragmatic default for breadth of model and hardware support and time-to-serve. Treat any single benchmark as a snapshot — benchmark your own model and traffic before deciding.

### A.6 Kubernetes-native orchestration: llm-d

Running one `vllm serve` process is straightforward; running a *fleet* of disaggregated prefill and decode workers, each independently autoscaled, behind smart routing, is a distributed-systems problem in its own right. **llm-d** (announced 2025-05-20, authored by engineers from Red Hat, Google, and IBM, now a CNCF Sandbox project) exists to standardize that problem: it's a Kubernetes-native framework that, in its own words, provides "a well-lit path for anyone to serve at scale," built directly on top of vLLM rather than replacing it.

Concretely, llm-d integrates two pieces with vLLM:

- **The Gateway API Inference Extension (IGW)** — Kubernetes-native routing that understands LLM-specific signals (KV-cache locality, current queue depth, prefix-cache affinity) instead of routing on generic HTTP load metrics. A request can be routed to the replica most likely to already have its prefix cached, turning prefix-cache hit rate (section B.4) into a cluster-wide property instead of a single-replica one.
- **Disaggregated serving as a first-class deployment pattern** — llm-d's "well-lit paths" (its v0.2 release, per the llm-d blog) package the prefill/decode split from section (A.2) as a supported, documented Kubernetes deployment topology — separate prefill and decode `Deployment`s, independently autoscaled, with the KV-transfer connector wiring already solved — rather than something each team hand-rolls.

**When it's worth adopting:** once you're already running vLLM behind Kubernetes at a scale where you'd otherwise be hand-building the routing and disaggregation plumbing from section (A.2) and (B) yourself. For a single-node or small-fleet deployment, plain `vllm serve` behind a conventional load balancer (with sticky routing by prefix hash if you want DIY prefix-affinity) remains simpler and is not something to give up prematurely.

> **Saying it out loud.** Running one `vllm serve` is easy; running a fleet of independently-autoscaled disaggregated prefill and decode workers behind smart routing is a distributed-systems problem. llm-d exists to standardize that — a Kubernetes-native framework built on top of vLLM rather than replacing it, now a CNCF sandbox project. Two pieces matter. It uses the Gateway API Inference Extension, which routes on LLM-specific signals like prefix-cache affinity and queue depth instead of generic HTTP metrics — that turns prefix-cache hit rate from a per-replica property into a cluster-wide one. And it packages the prefill/decode split as a supported deployment topology with the KV-transfer wiring already solved. Worth adopting once you'd otherwise be hand-building that plumbing; premature otherwise.

### A.7 Timeline: how fast this space is moving

Concrete, dated milestones from 2025–2026 worth having ready in an interview — the point isn't memorizing dates, it's demonstrating you track a fast-moving space with real sources rather than a static mental model frozen at the 2023 paper:

| Date | Milestone |
|---|---|
| 2025-01-27 | vLLM V1 alpha released — unified scheduler, isolated `EngineCore`, chunked prefill + prefix caching on by default |
| 2025-05-20 | llm-d announced (Red Hat/Google/IBM) — Kubernetes-native distributed inference built on vLLM |
| 2025-07-01 | Red Hat developer blog on EAGLE-3 speculative decoding speedups in vLLM |
| 2025-12-13 | vLLM `speculators` v0.3.0 — training support for custom draft/EAGLE heads |
| 2025-12-17 | vLLM large-scale MoE serving results (DeepSeek-class, wide expert-parallelism) |
| 2026-03-23 | Third-party H100 benchmark comparing vLLM, SGLang, and TensorRT-LLM throughput/TTFT/cold-start |
| 2026-04-07 | AMD MORI-IO KV connector blog — disaggregated prefill/decode, ~2.5× goodput on one node |
| 2026-04-16 | Red Hat developer blog on speculative decoding performance for gpt-oss-style models |
| 2026-05-26 | EAGLE 3.1 released jointly by the EAGLE authors, vLLM, and TorchSpec teams |
| 2026-07-13 | EAGLE-3 speculative decoding training/serving support on AMD Instinct via AMD Quark |

Treat this table as a snapshot as of this chapter's writing, not a permanent record — check `vllm.ai/blog` and `docs.vllm.ai` directly for what's shipped since.



---

## (B) Build it in practice — extended: multi-node 70B+, benchmarking, and reading the tuning signal

The single-GPU 8B walkthrough above teaches the knobs. Real "flagship" deployments — a 70B+ model, multiple nodes, a load test, and a principled read of *why* you're moving each dial — is where interviews (and production) actually live. This section builds that end to end.

> **Saying it out loud.** The single-GPU walkthrough teaches you the knobs; the multi-node one is where interviews actually live, because it forces you to reason about topology, measurement, and cost together. The arc is: size the problem from the weight math, launch across nodes with the right TP-times-PP split, load-test it the way clients will actually hit it, read the Prometheus metrics to diagnose rather than guess, turn those metrics into alerts, and finally convert throughput into dollars per million tokens. Each of those steps is a place people skip straight to "we tuned it and it seemed fine" — and the difference between guessing and diagnosing is entirely in step four, reading KV cache usage and prefix hit rate side by side.

### B.1 Sizing the problem: Llama-3.1-70B, fp16, across 2 nodes

Weights: \(70\text{B} \times 2\ \text{bytes} = 140\ \text{GB}\) — bigger than any single 80 GB GPU, and even TP-8 on one 8×80GB node leaves only \(80 - 140/8 = 62.5\ \text{GB}\) per GPU before KV, activations, and overhead — workable, but tight if you also want a large `--max-model-len` and high concurrency. We'll instead spread the model across **two 8-GPU nodes (16 GPUs total)**: `-tp 8` inside each node (over NVLink) and `-pp 2` across the two nodes (over the slower inter-node fabric), per the TP-first-then-PP rule from the Parallelism section above.

> **Saying it out loud.** Start from the weight math. 70 billion parameters in fp16 is 140 gigabytes, which is bigger than any single 80-gig card. TP=8 on one 8-GPU node would put 17.5 gigabytes of weights on each card, leaving around 62 for KV and overhead — workable, but tight if you also want long context and high concurrency. So you spread across two 8-GPU nodes: TP=8 inside each node over NVLink, and PP=2 across the two nodes over the slower fabric. That's the tensor-parallel-first-then-pipeline rule applied concretely. And note what drove the decision — not "it doesn't fit," because it technically does, but the KV headroom left over after it fits, which is the number that determines your actual concurrency.

### B.2 Multi-node launch with Ray

vLLM's multi-node path is built on **Ray**: one node starts the Ray head, the other joins as a worker, and `vllm serve` is launched once, on the head node, with the combined `-tp` × `-pp` world size.

**On node 0 (head):**

```bash
# Start the Ray cluster head. Pick a stable port; other nodes connect here.
ray start --head --port=6379 --num-gpus=8

# Confirm the cluster sees both nodes before launching vLLM:
ray status
```

**On node 1 (worker):**

```bash
# Point at node 0's IP (the Ray head), join with its 8 GPUs.
ray start --address='<NODE0_IP>:6379' --num-gpus=8
```

**Back on node 0, launch the server** once Ray reports 16 GPUs total:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --enable-prefix-caching
```

vLLM's launcher detects the existing Ray cluster and places the 16 model-parallel workers across both nodes automatically (8 TP ranks per PP stage, one PP stage per node). Watch the startup log for **per-GPU** `# GPU blocks` — with TP-8, each GPU holds \(1/8\) of the weights and KV heads, so KV capacity is aggregated across all 8 GPUs of a TP group, not duplicated.

**Topology, in words:**

```
Node 0 (Ray head, PP stage 0, TP ranks 0-7)      Node 1 (Ray worker, PP stage 1, TP ranks 0-7)
+------------------------------------------+     +------------------------------------------+
| GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7   |     | GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7   |
|  \____/\____/\____/\____/\____/\____/\____/   |     |  \____/\____/\____/\____/\____/\____/\____/   |
|   NVLink all-reduce per layer (TP=8)      |     |   NVLink all-reduce per layer (TP=8)      |
|   holds layers 1..40 (first half)         |     |   holds layers 41..80 (second half)       |
+--------------------+----------------------+     +----------------------+-------------------+
                      |  inter-node link (PP hand-off: activations only, point-to-point) |
                      +----------------------------------------------------------------->+
```

Each node's 8 GPUs form one **TP group** cooperating over fast NVLink on the *same* half of the model's layers; the *only* traffic crossing the slower inter-node link is the **PP hand-off** — the activation tensor passed from the last layer of stage 0 to the first layer of stage 1 — which is exactly why PP, not TP, is the parallelism strategy that tolerates crossing nodes. Confusing the two (e.g. setting `-tp 16` across both nodes instead of `-tp 8 -pp 2`) forces the per-layer all-reduce itself over the slow inter-node link, which is a common and expensive multi-node misconfiguration.

**Quantized alternative, fewer GPUs, one node:** if 16 GPUs aren't available, an AWQ 4-bit checkpoint drops weights to ~35 GB, fitting comfortably on **2 GPUs** with room for KV:

```bash
vllm serve casperhansen/llama-3-70b-instruct-awq \
  --quantization awq_marlin \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

Decision order, restated for a 70B+ model: **fit on one node with TP first → quantize to shrink weights and cut GPU count if hardware is scarce → go multi-node with PP only when a single node genuinely cannot hold model + working KV.**

> **Saying it out loud.** vLLM's multi-node path runs on Ray: one node starts the Ray head, the other joins as a worker, and then you launch `vllm serve` exactly once, on the head node, with the combined tensor-parallel times pipeline-parallel world size. The thing that surprises people is that it's a single launch, not one per node — Ray places the workers for you. Two practical gotchas: the inter-node network has to actually be fast enough for the pipeline hand-off, and every node needs the same model weights accessible, which usually means a shared filesystem or a pre-warmed local cache. And your failure domain just got bigger — losing either node takes the whole replica down, which is why the readiness probe and router behavior matter more here than in the single-node case.

### Health checks and readiness probes

A vLLM pod behind Kubernetes needs its readiness probe to reflect the real 60–90 second cold start from section C.3, not just "process is up":

```yaml
readinessProbe:
  httpGet:
    path: /health          # returns 200 only once the engine has finished loading and is serving
    port: 8000
  initialDelaySeconds: 20
  periodSeconds: 5
  failureThreshold: 30      # allow up to ~2.5 minutes for weight load + CUDA graph capture
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 120  # only starts checking liveness well after cold start should be done
  periodSeconds: 15
  failureThreshold: 3
```

The distinction matters operationally: a **readiness** failure just removes the pod from the routing pool (correct during cold start); a **liveness** failure restarts the container (wrong during cold start — it would restart a pod that's still legitimately loading, resetting its progress and potentially causing the exact oscillation from section C.3). Give liveness a much longer `initialDelaySeconds` than readiness for precisely this reason.

> **Saying it out loud.** A vLLM pod's readiness probe has to reflect a real 60-to-90-second cold start, not just "the process is up." That cold start is dominated by loading weights into GPU memory and then CUDA graph capture, and neither is something you can rush at probe time. So the probe hits an endpoint that only returns healthy once the engine is actually serving, and you give it a startup budget sized to the measured worst case — measured, not guessed. Getting this wrong has two distinct failure modes worth naming separately: too tight and Kubernetes crash-loops a perfectly healthy pod mid-load, too loose and the load balancer routes real traffic at a replica that isn't ready and every one of those requests times out.


### B.3 Benchmark run against the multi-node deployment

Never tune by feel — load-test the deployment exactly as clients will hit it:

```bash
vllm bench serve \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --base-url http://<NODE0_IP>:8000 \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 2000 \
  --request-rate 30 \
  --max-concurrency 256
```

This reports, per run: **throughput (tokens/s, req/s)**, **TTFT** (p50/p90/p99), **ITL/TPOT** (p50/p90/p99), and total duration. Sweep `--request-rate` across a few values (e.g. 10, 20, 30, 50) to trace out a throughput-vs-latency curve — the shape you're looking for is a "knee" where p99 latency starts climbing steeply; that's your practical capacity ceiling, not whatever number a spec sheet claims.

> **Saying it out loud.** Never tune by feel — load-test the deployment exactly the way clients will hit it. `vllm bench serve` against a realistic dataset like ShareGPT gives you throughput in tokens and requests per second, plus TTFT and inter-token latency at p50, p90, and p99. The important part isn't running it once, it's sweeping `--request-rate` across a few values — say 10, 20, 30, 50 — to trace the throughput-versus-latency curve. What you're looking for is the knee, the point where p99 starts climbing steeply while throughput has stopped rising. That knee is your practical capacity ceiling, and it's almost always well below whatever number a spec sheet or a vendor benchmark claims.

### B.4 Reading GPU memory utilization vs KV cache hit rate while tuning

This is the step people skip, and it's the difference between guessing and diagnosing. vLLM exposes Prometheus metrics on `/metrics`; the two to watch side by side are:

- **`vllm:gpu_cache_usage_perc`** — how full the KV cache pool is right now (0–1). Sustained values near 1.0 mean you are memory-bound and close to triggering preemption.
- **`vllm:gpu_prefix_cache_hit_rate`** (and the block-level hit/query counters it's derived from) — the fraction of prefill tokens served from the prefix cache instead of recomputed.

Pull them directly while load-testing:

```bash
curl -s http://<NODE0_IP>:8000/metrics | grep -E 'gpu_cache_usage_perc|gpu_prefix_cache|num_requests_running|num_requests_waiting'
```

**How to read the combination:**

| KV cache usage | Prefix hit rate | Diagnosis | Action |
|---|---|---|---|
| Low (< 0.5) | Low | Under-loaded — you have slack in both memory and reuse | Raise `--request-rate` in your test, or accept more traffic; consider raising `max-num-seqs` |
| High (> 0.9), no preemption | High | Healthy: cache is full of *useful* (reused) blocks doing real work | This is close to the target operating point — leave it, or push `gpu-memory-utilization` up slightly for more headroom |
| High (> 0.9), preemption warnings appearing | Low | You're evicting active batch to make room, and it isn't even prefix cache doing the crowding | Lower `max-num-seqs` / `max-model-len`, or add `--swap-space`; this is memory over-commitment, the failure mode in section C.1 |
| Low–moderate | Dropping over time under load | Cache pressure is evicting prefix blocks before they're reused | Traffic doesn't have the reuse you assumed, or memory is too tight to keep both active batch and cached prefixes — reconsider whether prefix caching is earning its keep for this workload |

The general principle: **KV cache usage tells you how full the tank is; prefix hit rate tells you how much of that fullness is "free" reused work versus active, paying-for-itself batch.** A full tank with a high hit rate is efficient. A full tank with a low hit rate and rising preemption is the KV over-commitment failure mode — tune `max-num-seqs`/`max-model-len`/`swap-space`, not `gpu-memory-utilization` upward.

> **Saying it out loud.** This is the step people skip and it's the difference between guessing and diagnosing. Two metrics, read side by side. `gpu_cache_usage_perc` tells you how full the KV tank is; sustained near 1.0 means you're memory-bound and about to preempt. `gpu_prefix_cache_hit_rate` tells you what fraction of prefill tokens came from cache instead of being recomputed. The combination is what diagnoses. High usage with a high hit rate is healthy — the tank is full of reused work doing real good. High usage with a *low* hit rate and preemption warnings is the over-commitment failure mode, and the fix is lowering `max-num-seqs` or `max-model-len`, not raising `gpu-memory-utilization`. Usage tells you how full; hit rate tells you how much of that fullness is free.

### B.5 Observability: turning the metrics into alerts

Reading `/metrics` by hand during a tuning session is step one; the same signals belong in a **Grafana dashboard fed by Prometheus scraping vLLM**, with alerts that page *before* an SLO breach rather than after. A minimal, high-signal alert set built directly on the metrics from B.4:

```yaml
# Prometheus alerting rules (excerpt) for a vLLM deployment
groups:
  - name: vllm-capacity
    rules:
      - alert: VLLMKVCacheNearFull
        expr: vllm:gpu_cache_usage_perc > 0.95
        for: 2m
        labels: {severity: warning}
        annotations:
          summary: "KV cache >95% full for 2m — preemption risk rising"

      - alert: VLLMPreemptionRateHigh
        expr: rate(vllm:num_preemptions_total[5m]) > 0
        for: 1m
        labels: {severity: critical}
        annotations:
          summary: "Active preemption/recompute thrash — see section C.1 playbook"

      - alert: VLLMPrefixCacheHitRateCollapsed
        expr: vllm:gpu_prefix_cache_hit_rate < 0.2
        for: 10m
        labels: {severity: warning}
        annotations:
          summary: "Prefix cache hit rate collapsed — cache pressure or traffic shape changed"

      - alert: VLLMQueueBacklogGrowing
        expr: vllm:num_requests_waiting > 20
        for: 3m
        labels: {severity: warning}
        annotations:
          summary: "Requests queueing — under-provisioned for current load"
```

The key discipline: **alert on `VLLMPreemptionRateHigh` directly**, not only on downstream p99-latency SLO burn. Section (C.1)'s incident was visible in preemption logs minutes before it became a paging-worthy latency alert — closing that gap is the entire value of this dashboard. Pair it with a panel plotting `gpu_cache_usage_perc` and `gpu_prefix_cache_hit_rate` on the same time axis as request rate, so a traffic-shape change (e.g. the C.1 burst of long documents) is visually obvious as "cache usage climbed while hit rate stayed flat" rather than requiring someone to piece it together from logs after the fact.

> **Saying it out loud.** Reading `/metrics` by hand during a tuning session is fine; leaving it there is not. The same signals belong on a Grafana dashboard with alerts that page *before* an SLO breach rather than after. A minimal high-signal set: KV cache usage sustained near full, preemption rate above zero for any meaningful window, queue depth trending up, and prefix cache hit rate dropping. The point that generalizes past vLLM: the metric that would have caught each of this chapter's incidents earliest is almost never the one on the primary latency-and-error-rate dashboard. Alerting only on SLO burn means you always find out after the users do — alerting on the engine's own internal pressure signals is what turns postmortems into things you catch in minutes.

### B.6 Cost-per-token worked calculation

Tie the tuning knobs back to a dollar figure, since that's what a "lowest cost per token" system-design answer (section D.2) ultimately needs to produce. Using the multi-node Llama-3.1-70B deployment from B.1–B.2 as the base case, and approximate on-demand H100 pricing of **~$2/GPU-hour** (use your actual contracted rate — this varies widely by cloud and commitment):

| Configuration | GPUs | $/hour (fleet) | Throughput (tok/s, from a `vllm bench serve` sweep) | Approx cost per 1M output tokens |
|---|---|---|---|---|
| fp16, TP8×PP2 (B.2 Option A/C) | 16 | $32/hr | ~2,000 tok/s aggregate | $32 / (2000×3600/1e6) ≈ **$4.44** |
| AWQ 4-bit, TP2 (B.2 Option B) | 2 | $4/hr | ~1,400 tok/s (single replica, smaller batch ceiling) | $4 / (1400×3600/1e6) ≈ **$0.79** |
| FP8, TP4, one node | 4 | $8/hr | ~1,800 tok/s | $8 / (1800×3600/1e6) ≈ **$1.23** |

The arithmetic is simply \(\text{cost per 1M tokens} = \dfrac{\text{fleet \$/hr}}{\text{tokens/s} \times 3600 / 10^6}\). The point isn't that any one row is "correct" — real throughput numbers must come from your own `vllm bench serve` sweep against real traffic (section B.3), and real GPU pricing depends on your contract — it's that **this is the calculation an interviewer wants to see you reach for**, and that quantization plus right-sizing GPU count can move cost per token by 3–5× for the same model, which is usually a bigger lever than any single scheduler flag.

> **Saying it out loud.** Tie the knobs to dollars, because that's what a "lowest cost per token" answer needs to produce. The arithmetic is just fleet dollars per hour divided by tokens per second times 3,600, over a million. Using roughly $2 per H100-hour as of this writing — use your own contracted rate, it moves a lot — a 16-GPU fp16 fleet at 2,000 tokens per second is about $4.44 per million output tokens, a 4-GPU FP8 setup at 1,800 tokens per second is about $1.23, and a 2-GPU AWQ 4-bit replica at 1,400 is about $0.79. The point isn't that any row is right — real throughput has to come from your own sweep. It's that quantization plus right-sizing GPU count moves cost per token by three to five times, which dwarfs any single scheduler flag.

### B.7 Smoke-test checklist before going live

A short, concrete pre-launch checklist that exercises the failure modes this chapter covers, run against the actual deployment before it takes real traffic:

1. **Startup log sanity** — confirm `# GPU blocks`, `Available KV cache memory`, and `Maximum concurrency` (section "Reading the startup logs") match hand-calculated expectations from section B.1; a mismatch usually means a flag was set differently than intended.
2. **Max-context request** — send one request at exactly `--max-model-len` tokens and confirm it succeeds without error (catches the "`max-num-batched-tokens` too low with chunked prefill off" failure mode).
3. **Concurrent burst at target peak QPS** — run `vllm bench serve` at the traffic rate you actually expect (section B.3), and confirm zero `preempted` log lines appear at that load; if they do, capacity is under-provisioned for the stated SLO, not just for the benchmark.
4. **Prefix-cache hit-rate check** — replay a realistic sample of production-shaped traffic (shared system prompt / multi-turn conversation) and confirm `vllm:gpu_prefix_cache_hit_rate` is non-trivial if your workload assumes reuse; a flat 0% means either prefix caching is misconfigured or the traffic doesn't actually share prefixes the way you assumed.
5. **Cold-start timing** — measure actual time from pod scheduling to `/health` returning 200, and set the readiness probe / autoscaler lead time (section C.3, B.2) from the measured number, not a guess.
6. **Quantization quality gate** (if applicable) — run the task-specific pass-rate eval from section C.2 against the quantized checkpoint before it serves any real traffic, not just a perplexity check.
7. **Failover / restart** — kill one replica (or one node, in the multi-node TP×PP case) under load and confirm the router removes it from rotation via the readiness probe without cascading latency elsewhere.

None of these are exotic — each maps directly to one of the failure modes or war stories earlier in this chapter. Running them once, deliberately, before launch is far cheaper than discovering them live.

> **Saying it out loud.** Seven things I'd run against the actual deployment before it sees traffic, and each one maps to a failure mode from earlier in the chapter. Check the startup logs match your hand calculation, because a mismatch means a flag didn't take. Send one request at exactly `max-model-len` to confirm it succeeds. Run a burst at your real expected peak and confirm zero preemption lines — if they appear, you're under-provisioned for the SLO, not just for the benchmark. Replay production-shaped traffic and confirm prefix hit rate is non-trivial if your design assumed reuse. Measure actual cold start and set the readiness probe from the measurement. Run the task-specific quality eval if you quantized. And kill a replica under load to confirm the router drops it cleanly.


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

Reality check: the three have converged — TGI and TensorRT-LLM adopted paged KV and in-flight batching. **TensorRT-LLM often wins raw latency/throughput on NVIDIA** thanks to ahead-of-time kernel compilation, at the cost of a per-model engine-build step and NVIDIA lock-in. **vLLM wins on flexibility, ease, and hardware breadth**, and is the usual default. See section (A.5) above for SGLang's place in this picture and concrete 2026 benchmark numbers. Always benchmark on *your* model, hardware, and traffic shape before deciding.

> **Saying it out loud.** The short comparison: vLLM gives you the best throughput per dollar out of the box with the broadest model support and a one-command setup. TGI is a solid production server that fits tightly into the Hugging Face ecosystem and has adopted most of the same ideas. TensorRT-LLM behind Triton gets you peak NVIDIA-GPU performance through ahead-of-time compiled engines, and it genuinely is faster — but it costs a per-model, per-shape compilation step that can run tens of minutes, plus NVIDIA-only lock-in. So the decision rule is about how fixed your deployment is: if the model and shapes are stable and long-lived, the compile cost amortizes and TensorRT wins. If you're iterating across model families, vLLM's time-to-serve is worth more than the last 15% of throughput.

---

## Failure modes and pitfalls

- **OOM at startup from `gpu-memory-utilization` too high.** vLLM pre-allocates the KV pool; if you set 0.98 with no slack, activation spikes or CUDA-graph capture push you over and it crashes on load. Back off to ~0.90 and grow gradually. Co-located processes share the same HBM — vLLM only sees the fraction you give it.
- **Preemption and recompute thrash.** When admitted sequences collectively exceed KV capacity, vLLM **preempts** some — either swapping their KV to CPU (`--swap-space`) or **discarding and recomputing** it later. Frequent `preempted` warnings mean you over-committed `max-num-seqs`/`max-model-len`; throughput drops as work is redone. Fix by lowering concurrency, shortening `max-model-len`, or adding swap. See the full incident writeup in section (C.1).
- **Long-context KV blowup.** KV grows linearly with context. A handful of 128k-token requests can consume the entire pool and starve everyone else. Bound it with `--max-model-len`, `--kv-cache-dtype fp8`, and admission limits; don't advertise a context you can't afford to serve concurrently.
- **Quantization quality loss.** 4-bit weight-only (AWQ/GPTQ) can degrade code/math/reasoning noticeably even when perplexity looks fine. Always validate on a task-specific eval, and prefer FP8 on Hopper where it's near-lossless. See section (C.2) for a real incident.
- **Speculative decoding backfiring.** Low draft acceptance, or high batch load, turns speculation into pure overhead. Measure acceptance rate; disable under saturation.
- **`max-num-batched-tokens` too low with chunked prefill off.** If a prompt exceeds the token budget and chunked prefill isn't enabled, requests fail. Keep chunked prefill on, or set the budget ≥ `max-model-len`.
- **Assuming `max-num-seqs` is the batch limit.** Usually **KV memory** binds first. Raising `max-num-seqs` without KV headroom just causes preemption. Watch the `# GPU blocks` log, not just the seq cap.
- **Prefix cache eviction under pressure.** Cached prefixes compete with active KV; under load they're evicted and hit rate falls — throughput quietly regresses. Size memory for both if prefix reuse is core to your workload.

> **Saying it out loud.** The recurring vLLM failures, in rough order of frequency. OOM at startup from setting `gpu-memory-utilization` too high, because vLLM pre-allocates the KV pool and activation spikes push you over — back off to 0.90 and grow. Preemption and recompute thrash, when admitted sequences collectively exceed KV capacity; frequent `preempted` warnings mean you over-committed. Long-context blowup, where a handful of 128K requests eat the whole pool and starve everyone. Quantization quality loss that perplexity doesn't catch. Speculative decoding backfiring under high batch load. And the conceptual one: assuming `max-num-seqs` is your batch limit when KV memory almost always binds first — raising it without memory headroom just buys you preemption.

---

## (C) Production case studies & war stories

Real incidents read differently from tuning tables — they teach you what the failure *feels like* from the on-call seat, before you've had time to read a metrics dashboard calmly. Two representative ones.

### C.1 Preemption/recompute thrashing under bursty long-context traffic

**Setup:** a RAG assistant serving Llama-3-8B on a single A100-80GB, `--max-model-len 8192`, `--max-num-seqs 256`, `gpu-memory-utilization 0.90`. Sized and load-tested against typical traffic: short questions, ~1200-token average retrieved context, comfortably inside the ~442k-token / ~54-concurrent KV budget from the worked example above.

**Incident:** a product launch drove a burst of users pasting **entire long documents** (6,000–8,000 tokens) for summarization, at the same time normal short-query traffic continued. Symptoms, in order of appearance:

1. p99 latency alarms fired first — some requests took 10–20× longer than normal, with no corresponding drop in throughput dashboards (which looked almost fine).
2. Server logs filled with `Sequence group ... is preempted by PreemptionMode.RECOMPUTE` warnings, dozens per second during the burst.
3. GPU utilization looked *high* the whole time — easy to misread as "the GPU is just busy," when it was actually busy **redoing work it had already done**.

**Root cause:** each long-document request consumed far more KV blocks than the average request the system was sized for. Once enough long requests were admitted concurrently, the running batch's collective KV footprint exceeded the pool. vLLM's scheduler did exactly what it's supposed to do — preempted lower-priority sequences to make room, discarding their KV and **recomputing** it from scratch when they were readmitted (the default preemption mode when `--swap-space` is small). Every recompute is a full prefill redone; for an 8,000-token document that's a very expensive redo, and it kept happening because the burst didn't clear before the preempted sequences got starved again. This is thrash: the system spent its cycles re-deriving state it had already computed, instead of making forward progress — classic memory-over-commitment behavior, just at the KV-cache layer instead of the OS page-cache layer it's modeled on.

**Fix, in order applied:**

1. **Immediate mitigation:** capped `--max-model-len` down to what the product actually needed to guarantee (4096), rejecting (with a clear error) documents beyond that instead of admitting them and starving everyone. This is a blunt instrument but stops the bleeding in minutes.
2. **Real fix:** raised `--swap-space` from the default 4 GiB to 32 GiB per GPU, so that under a burst, preempted sequences' KV is **swapped to CPU RAM and restored** rather than discarded and recomputed — much cheaper for long sequences specifically, at the cost of some CPU↔GPU transfer time. Recompute is fine for short sequences (cheap to redo); swap is fine for long ones (expensive to redo, and DMA transfer is comparatively cheap).
3. **Structural fix:** split the deployment into two pools behind a router — a small-context, high-concurrency pool for typical short queries, and a separate large-context, lower-concurrency pool with a bigger `--swap-space` and smaller `--max-num-seqs` for document-heavy requests — so one traffic pattern can no longer starve the other. This is a lightweight, single-node precursor to the full disaggregated prefill/decode architecture in section (A.2); the same principle (isolate workloads with different resource profiles) applies at both scales.

**Lesson:** preemption logs are not noise — they are the single earliest, cheapest signal that your admission control (`max-num-seqs`, `max-model-len`) is miscalibrated against your *actual* traffic distribution, not the average case you load-tested against. Size for your **tail** request shape, not your median, and alert on preemption rate directly rather than waiting for it to show up as a latency SLO breach.

> **Saying it out loud.** A RAG assistant on one A100, sized and load-tested against typical traffic — short questions, twelve-hundred-token contexts, comfortably inside a 442,000-token KV budget. Then a launch drove users pasting entire six-to-eight-thousand-token documents for summarization, while normal short traffic continued. The KV pool filled, the scheduler started preempting, and preempted sequences got their KV discarded and recomputed later — so the system was redoing prefill work it had already done, which made it slower, which made it preempt more. p99 latency spiked while throughput looked deceptively normal. The lesson: your capacity number is a function of the *traffic shape*, not just the request rate, and a single long-context traffic class can invalidate a load test that was honest for the workload you had yesterday.

### C.2 A quantization choice that quietly hurt output quality

**Setup:** a coding-assistant service moved a 34B code model from fp16 to **AWQ 4-bit weight-only quantization** to fit two GPUs instead of four, halving infrastructure cost. Pre-launch validation ran the standard perplexity check on a held-out slice of a general text corpus — the numbers looked fine, within ~1–2% of the fp16 baseline — and the team shipped.

**Incident:** not a page-you-at-3am outage — worse, in some ways: a slow-burning **quality regression** that only showed up as a rising rate of user "thumbs down" feedback and support tickets about "the assistant writing subtly broken code" over the following two weeks. Nothing crashed; nothing alerted; the metrics that would have caught it (perplexity, latency, error rate) were all green.

**Root cause:** perplexity on general text is a weak proxy for quality on **narrow, structured tasks** like code generation. 4-bit weight-only quantization applies uniform-ish precision loss across all weights; for a general-purpose completion task the aggregate effect is small and perplexity captures it fine. But code correctness depends on getting a small number of **high-precision decisions** exactly right — matching brackets, correct off-by-one indices, exact API argument order — and those are disproportionately sensitive to the quantization noise that a perplexity average smooths right over. The regression was real but statistically invisible to the metric the team trusted.

**Fix:**

1. Rolled back to fp16 immediately once a task-specific eval (a held-out set of "does the generated code pass its unit tests" checks, not perplexity) was run retroactively and showed a **measurable drop in pass rate** versus the fp16 baseline — the smoking gun the perplexity check had missed entirely.
2. Re-quantized to **FP8** instead of AWQ 4-bit (the team's GPUs were H100s) — FP8 is close to lossless in practice for most models, and the pass-rate eval confirmed no measurable regression versus fp16, while still shrinking weights enough to recover most of the cost savings the team wanted from the original migration.
3. Added the task-specific pass-rate eval to the **pre-deployment gate** for any future quantization or model change — perplexity remained a sanity check, but was no longer the sole quality gate.

**Lesson:** quantization quality loss is *task-dependent*, and a generic proxy metric like perplexity can be flat-out blind to regressions that matter enormously to users on structured tasks (code, math, precise extraction). Before shipping any quantization change, validate on an eval set that resembles what your users actually do — and prefer FP8 over 4-bit weight-only when the hardware supports it and the task is precision-sensitive, exactly as the Quantization section above recommends.

> **Saying it out loud.** This is the scariest kind of incident because nothing alerts. A coding-assistant team moved a 34B code model from fp16 to AWQ 4-bit to halve their GPU count, validated with a perplexity check on general text that came back within one or two percent, and shipped. Over the following two weeks, thumbs-down feedback climbed and tickets came in about "subtly broken code." Nothing crashed; latency, error rate, and perplexity were all green. The root cause is that perplexity averages over everything, while code correctness depends on a small number of high-precision decisions — matching brackets, off-by-one indices, exact argument order — that are disproportionately sensitive to quantization noise. The fix was FP8 instead of 4-bit, plus a task-specific pass-rate eval as a permanent pre-deployment gate.

### C.3 Autoscaling cold-start thrashing

**Setup:** a customer-support chatbot on Llama-3-8B running behind a Kubernetes Horizontal Pod Autoscaler (HPA), scaling vLLM replicas on GPU utilization, targeting 70% average GPU utilization per replica, minimum 2 replicas, maximum 10.

**Incident:** during a marketing push, traffic ramped from baseline to 4× over about ten minutes. The HPA reacted correctly, in principle — GPU utilization crossed the scale-up threshold and new pods were scheduled. But each new vLLM pod took **60–90 seconds to become ready**: pulling the container image (if not already cached on the node), loading a full copy of model weights from remote storage into GPU memory, then running CUDA graph capture before it could serve its first request. During that window, the *existing* replicas kept absorbing the full ramping load, GPU utilization on them climbed well past the scale-up threshold, and the HPA — seeing utilization still high — kept requesting **more** new replicas on top of the ones still warming up. When the first batch of new pods finally came online, aggregate capacity briefly overshot demand, utilization dropped, and the HPA started scaling back down — right as the next traffic wave arrived. The result was an oscillating replica count and a period of elevated p99 latency and a handful of request timeouts on the saturated original replicas, even though the cluster had (eventually) more than enough aggregate GPU capacity for the actual load.

**Root cause:** the HPA's reaction model implicitly assumes new capacity comes online fast relative to the metric's response time. vLLM's model-loading and CUDA-graph-capture startup cost violates that assumption badly compared to, say, a stateless web server pod that's ready in a second or two — the feedback loop was scaling on a signal (current GPU utilization) that couldn't reflect capacity already "in flight" but not yet serving.

**Fix:**

1. **Immediate mitigation:** manually pinned replica count above the oscillation range for the duration of the marketing push, trading elasticity for stability until the traffic pattern was well understood.
2. **Real fix — scale on a leading indicator, not a lagging one.** Switched the HPA's scaling signal from raw GPU utilization to **`vllm:num_requests_waiting`** (queue depth) with a much lower, earlier-triggering threshold, so scale-up starts *before* existing replicas are saturated rather than after — giving the 60–90 second warm-up time to actually land before it's needed.
3. **Cut the warm-up time itself.** Pre-baked model weights into the node image / a local NVMe cache instead of pulling from remote object storage on every pod start, and pre-warmed a small pool of "standby" replicas during known high-traffic windows (marketing pushes, product launches) rather than relying purely on reactive autoscaling for predictable bursts.
4. **Added a scale-down cooldown** long enough to ride out a single traffic wave, preventing the "scale up, immediately scale back down" oscillation once new capacity did land.

**Lesson:** vLLM replicas are not fungible with stateless microservice pods for autoscaling purposes — a 60–90 second cold start (dominated by weight loading and CUDA graph capture) means the autoscaler must react to a **leading** signal (queue depth, request rate trend) with real lead time, not a **lagging** one (current utilization), or it will systematically over- and under-shoot during any traffic ramp. This is the same "measure the right signal, not just any green-looking metric" theme as section C.2, applied to the autoscaling layer instead of the quality-eval layer.

> **Saying it out loud.** A chatbot behind an HPA scaling on GPU utilization at a 70% target. Traffic ramped 4x over ten minutes, the HPA correctly scheduled new pods — but each vLLM pod took 60 to 90 seconds to become ready, dominated by loading weights and CUDA graph capture. During that window the existing replicas absorbed the whole ramp, utilization stayed high, and the HPA kept asking for *more* replicas on top of ones still warming. When the first batch finally landed, capacity overshot, utilization dropped, and it started scaling back down right as the next wave arrived. The fix was scaling on `num_requests_waiting` — a leading signal — instead of current utilization, a lagging one, plus pre-baked weights and a scale-down cooldown. vLLM replicas are simply not fungible with stateless pods for autoscaling.

### C.4 On-call quick-reference: what these three incidents teach as a single checklist

All three war stories reduce to the same underlying discipline — *know which signal actually leads the problem, and alert on that, not on its downstream symptom.* As a runbook a new on-call engineer can use directly:

| Symptom you're paged for | Check this signal first | If it confirms, this is probably... | Playbook |
|---|---|---|---|
| p99 latency spike, throughput looks fine | `vllm:num_preemptions_total` rate, "preempted" log lines | KV over-commitment / recompute thrash (C.1) | Cap `max-model-len` for the offending traffic class immediately; raise `--swap-space`; consider splitting into a separate pool for that traffic shape |
| Slow-building quality complaints, all standard metrics green | Task-specific pass-rate eval (not perplexity) on recent quantization/model changes | Quantization (or any model swap) hurt a narrow, precision-sensitive capability (C.2) | Roll back the change; re-evaluate with a task-specific gate before re-shipping; prefer FP8 over 4-bit weight-only on precision-sensitive tasks |
| Oscillating replica count, intermittent timeouts during traffic ramps | `vllm:num_requests_waiting` trend vs GPU-utilization trend during the ramp | Autoscaler reacting to a lagging signal against a slow (60–90s) cold start (C.3) | Scale on queue depth / request-rate trend instead of raw utilization; pre-warm for known bursts; add a scale-down cooldown |
| Prefix-cache hit rate silently dropped | `vllm:gpu_prefix_cache_hit_rate` alongside `gpu_cache_usage_perc` | Cache pressure evicting reused prefixes, or a genuine traffic-shape change | Confirm which with the B.4 table; add memory or reduce concurrency if it's pressure, investigate traffic if it's a shape change |

The common thread across all four rows: **the metric that would have caught the problem earliest is almost never the one on the primary latency/error-rate dashboard.** Building the section-B.5 alerting rules directly from `vllm:*` Prometheus metrics — rather than only alerting on downstream SLO burn — is what turns these from postmortems into things you catch in minutes.

> **Saying it out loud.** All three incidents reduce to one discipline: know which signal actually leads the problem, and alert on that rather than its downstream symptom. Paged for a p99 spike while throughput looks fine? Check the preemption counter first — that's KV over-commitment. Slow-building quality complaints with every standard metric green? That's a model or quantization change that hurt a narrow capability perplexity can't see. Oscillating replica count during ramps? That's an autoscaler on a lagging signal against a slow cold start. The common thread, and the thing worth saying explicitly: the metric that would have caught each of these earliest is almost never the one on your primary latency and error-rate dashboard.



---

## How the attention kernel actually reads paged blocks

It's worth being precise about *why* PagedAttention needs a custom kernel, because interviewers probe it. Standard fused attention (FlashAttention) assumes K and V for a sequence live in one contiguous tensor it can stride through. Paged KV breaks that assumption: a sequence's K/V are scattered across physical blocks in arbitrary order.

The PagedAttention kernel therefore takes the **block table** as an input. For a query at the current position it:

1. Reads the sequence's block table (logical block → physical block number).
2. For each logical block, computes attention scores \( q \cdot k \) against the K vectors *in that physical block*, iterating block by block.
3. Accumulates the softmax-weighted sum of V vectors from the same blocks.

Because the block is the unit of gather, the kernel does a small indirection per block (once per 16 tokens), not per token — cheap relative to the matmul. The block table lives in GPU memory alongside the cache. Modern vLLM builds this on FlashAttention/FlashInfer backends that natively accept paged KV, so you keep FlashAttention's IO-awareness *and* paging. This is the crux: paging costs almost nothing at kernel time, yet returns most of the wasted memory as usable batch.

> **Saying it out loud.** Interviewers probe this, so be precise. Standard fused attention like FlashAttention assumes a sequence's K and V live in one contiguous tensor it can stride through — and paged KV breaks that assumption outright, since the blocks are scattered in arbitrary order. So the PagedAttention kernel takes the block table as an *input*: for each query it reads the logical-to-physical mapping, computes attention scores against the K vectors in each physical block, and accumulates the softmax-weighted V sum block by block. The key efficiency point is that the block is the unit of gather, so the indirection happens once per sixteen tokens rather than once per token — negligible against the matmul. That's the crux: paging costs almost nothing at kernel time, and returns most of the wasted memory as usable batch.

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

The default policy is FCFS-ish with the newest/lowest-priority preempted first. The practical takeaway: **preemption is the pressure-relief valve, and seeing it constantly in logs means your admission settings exceed your true KV capacity.** It is correct behavior, not a bug — but it costs throughput, so tune it away. Section (C.1) walks through exactly this failure end to end.

> **Saying it out loud.** The scheduler keeps three queues and reconciles them every single step. Waiting is admitted-but-not-started. Running is actively decoding or prefilling. Swapped is preempted sequences whose KV blocks are parked in CPU memory. Each iteration it frees blocks from anything that finished, tries to promote waiting requests into running subject to the block budget and the sequence and token caps, and if running collectively needs more blocks than exist, it preempts — either swapping KV to CPU or discarding and recomputing it later. Recompute is default for short sequences, swap wins for long ones where recompute is expensive. The takeaway to say out loud: preemption is the pressure-relief valve and it's correct behavior, not a bug — but seeing it constantly means your admission settings exceed your true KV capacity, and it costs throughput.

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

Sweep one knob at a time (`gpu-memory-utilization`, `max-num-batched-tokens`, `max-num-seqs`) and plot throughput vs p99 latency. The right operating point is the knee of that curve for your SLO — not the max-throughput point, which usually violates latency targets. Section (B.3)–(B.4) above extends this to a multi-node 70B deployment and shows how to read the Prometheus KV-cache-usage and prefix-hit-rate metrics alongside it.

> **Saying it out loud.** Never tune by feel — vLLM ships a benchmark harness that mirrors real serving, and the discipline is one knob at a time. The four metrics that matter: throughput in tokens and requests per second, which is what you maximize for batch work; TTFT, which is what a chat user feels as lag before anything happens; inter-token latency, which is decode smoothness and is what gets hurt by big un-chunked prefills; and always p50 versus p99, because a fine p50 with a bad p99 almost always means preemption or prefill interference specifically. Then plot throughput against p99 latency and pick the knee of that curve for your SLO — deliberately *not* the max-throughput point, which will violate your latency target.

---

## V1 engine and disaggregated serving — quick recap

Two architectural notes worth knowing at a glance (see section (A) for full depth and dates):

- **The V1 engine** (default in current vLLM) rewrote the core for lower CPU overhead and a unified scheduler where prefill and decode are co-scheduled by default. Chunked prefill and prefix caching are **on by default** there. If you read older tutorials that tell you to manually enable these, that advice is stale.
- **Disaggregated prefill/decode** separates the compute-bound prefill and memory-bound decode onto **different GPU pools**, streaming the KV cache between them. Because the two phases have opposite resource profiles, dedicating hardware to each — and scaling them independently — can beat co-locating them, especially at high scale with long prompts. This is an increasingly mainstream pattern (KV transfer over NVLink/RDMA) for large deployments, with concrete production numbers now published (section A.2).

## Second worked example: Llama-3-70B across GPUs (single-node summary)

An 8B model fits one card; a 70B does not. In fp16, weights alone are \( 70\text{B} \times 2 = 140\ \text{GB} \) — larger than a single 80 GB GPU. Options, in the order you should consider them (see section B for the full multi-node walkthrough):

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
  --quantization awq_marlin \
  --tensor-parallel-size 2 \
  --max-model-len 8192
```

4-bit weights are ~\( 70\text{B} \times 0.5 = 35\ \text{GB} \), fitting two 80 GB cards with room for KV. Fewer GPUs, lower cost, at some quality cost — validate on your eval set (section C.2 is a cautionary tale about skipping this step).

**Option C — two nodes, TP×PP:** `--tensor-parallel-size 8 --pipeline-parallel-size 2` spreads a large model across 16 GPUs, TP within each node over NVLink and PP across the two nodes over the slower inter-node link. Section (B.2) above walks through the full Ray-based multi-node launch for exactly this configuration.

Decision order: **fit on one node with TP first; quantize to shrink weights and cut GPU count; go multi-node with PP only when a single node genuinely cannot hold model + working KV.**

> **Saying it out loud.** An 8B fits one card; a 70B in fp16 is 140 gigabytes of weights, so it does not. The options in the order you'd consider them: tensor-parallel across four or eight GPUs on one node over NVLink, which is the first thing to try because intra-node all-reduce is cheap. Quantize to 4-bit or FP8, which can bring a 70B down to two or four cards and is often the bigger cost lever. Or go multi-node with pipeline parallelism across nodes on top of tensor parallelism within them. The reasoning to make explicit: you're not just asking "does it fit," you're asking "how much KV headroom is left after it fits," because that leftover is what determines concurrency and therefore cost per token.

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

> **Saying it out loud.** Four log lines at startup tell you whether your config is sane, and reading them takes ten seconds. Available KV cache memory is what's left after weights and overhead — if it's tiny, lower `max-model-len`, quantize, or add GPUs. GPU KV cache size in tokens is your total batch budget; divide by average request length for realistic concurrency. Maximum concurrency is how many full-context sequences fit at once, and if that's below your expected load you *will* preempt under traffic. And CPU blocks is your swap capacity from `--swap-space`. If you look at nothing else, look at those four, because they convert abstract flags into the single number that governs throughput: how many tokens of KV you can hold.

---

## Appendix: additional operational flags

Beyond the core memory/parallelism/quantization flags covered above, these come up often enough in real deployments to be worth knowing by name:

| Flag | What it does | When you reach for it |
|---|---|---|
| `--served-model-name` | Name the OpenAI-API-facing model string differently from the HF repo id. | Present a stable public model name while swapping checkpoints behind it. |
| `--api-key` | Require a bearer token on the OpenAI-compatible endpoints. | Any deployment reachable outside a trusted network. |
| `--load-format` | Control how weights are loaded (`auto`, `safetensors`, `pt`, `bitsandbytes`, ...). | Speeding up cold start (safetensors mmap is fast) or loading from a non-default checkpoint format. |
| `--enforce-eager` | Disable CUDA graph capture, run in eager PyTorch mode. | Debugging a crash/NaN that only reproduces without graph capture; costs decode throughput. |
| `--cpu-offload-gb` | Offload some weight layers to CPU RAM, streamed to GPU on demand. | Squeezing a model that almost — but doesn't quite — fit in GPU memory, at a latency cost. |
| `--num-scheduler-steps` | Batch multiple scheduler steps together to cut CPU-side scheduling overhead. | High QPS deployments where CPU scheduling overhead (not GPU compute) is the bottleneck. |
| `--disable-log-stats` | Turn off periodic throughput/latency log lines. | Noisy logs in a low-traffic environment; keep on in production for the diagnostics this chapter relies on. |
| `--tokenizer` | Point at a different tokenizer than the model's default. | Serving a fine-tune with a custom tokenizer, or a quantized checkpoint that omits tokenizer files. |
| `--trust-remote-code` | Allow executing custom modeling code shipped with a HF repo. | Required for some model architectures; understand the supply-chain implication before enabling in production. |
| `--disable-sliding-window` | Force full attention even for models that support sliding-window attention. | Debugging correctness differences against a reference implementation. |

These rarely need tuning day-to-day, but showing you know they exist — and specifically *why* each one is reached for — is exactly the kind of breadth an interviewer probing "have you actually operated this in production" is listening for.

## Quick-reference config recipes

| Goal | Starting flags |
|---|---|
| Max throughput, batch/offline | `--gpu-memory-utilization 0.95 --max-num-batched-tokens 16384 --max-num-seqs 512` |
| Low-latency interactive chat | `--max-num-batched-tokens 4096` (protect ITL) `--enable-chunked-prefill`, consider speculative decoding |
| Long-context serving | `--kv-cache-dtype fp8 --max-model-len <needed> --swap-space 16`, cap concurrency |
| Memory-tight single GPU | `--quantization awq_marlin` (or `fp8` on Hopper) `--gpu-memory-utilization 0.90` |
| RAG with shared preamble | keep `--enable-prefix-caching` (default), moderate `max-num-seqs` |
| Model bigger than one GPU | `--tensor-parallel-size N` (one node) `[+ --pipeline-parallel-size M across nodes]` |
| Burst-prone long-context traffic | isolate a large-context pool with higher `--swap-space`, lower `--max-num-seqs` (see section C.1) |

---

## (D) Interview mastery

### D.1 "Explain PagedAttention in 60 seconds"

The question that separates people who've read the abstract from people who've internalized it. A tight answer, timed:

> "LLM serving is memory-bound: decoding one token rereads the whole model's weights, so the only way to get throughput is to batch many sequences together and amortize that read. What limits batch size is the KV cache — the attention memory of every token in every active sequence — and pre-vLLM systems stored each sequence's KV cache in one contiguous block sized to the maximum possible length, wasting 60–80% of it on fragmentation and unused reservation. PagedAttention borrows virtual memory from operating systems: it splits the KV cache into small fixed-size blocks — 16 tokens each — that live anywhere in a global pool, indexed per-sequence by a block table, exactly like a page table. Blocks are only allocated as a sequence actually grows, so waste drops to at most one partial block per sequence — effective utilization goes from ~20–38% to ~96%. That reclaimed memory becomes batch capacity, which is why vLLM gets 2–4× the throughput of the prior state of the art at the same latency. And because it's memory *indirection*, not physical layout, you get sharing for free — two sequences can point at the same physical block, which is what makes prefix caching and beam-search memory savings possible without extra machinery."

If you only remember one structural trick to hit every beat: **problem (fragmentation/waste) → borrowed idea (OS paging) → mechanism (blocks + block table) → payoff (utilization number) → bonus (sharing enables prefix caching)**.

### D.2 System-design prompt: "Serve a 70B model at the lowest cost per token while hitting a TTFT SLO"

A realistic senior-level prompt. Worked sketch, in the order an interviewer wants to hear it:

**1. Clarify the SLO and traffic shape first.** What's the TTFT target (e.g. p99 < 500 ms)? What's expected QPS, average/tail prompt length, and output length? Is traffic bursty? Is a high prefix-reuse rate expected (chat/RAG) or is every prompt unique? Cost-per-token optimization and TTFT protection pull in different directions, so the answer depends entirely on these numbers — say so explicitly; don't guess.

**2. Pick hardware and precision for cost per token.** For a 70B model, the cost-per-token lever with the biggest single effect is usually quantization, because it directly cuts both weight memory (more KV headroom, bigger batches, lower $/token) and, on the right hardware, compute time. On H100/Hopper, FP8 is close to lossless and gets full tensor-core speedup — default choice unless there's a hard reason to need fp16. On older Ampere/Ada hardware without native FP8, AWQ 4-bit with the Marlin kernel is the next-best cost lever, at some quality risk to validate (section C.2). Quantifying: fewer/cheaper GPUs directly divides your $/GPU-hour by however many requests you can now batch onto each.

**3. Pick parallelism to fit the model and hit latency.** Decide TP size to fit weights + working KV on a node (TP-first rule); use PP only if you must cross nodes. More TP ranks also lowers single-request latency (more GPUs cooperating on one forward pass), which helps TTFT directly — a real tension with the "fewer GPUs is cheaper" instinct from step 2, and worth naming as a tradeoff explicitly.

**4. Batch aggressively for cost, but protect the TTFT SLO with chunked prefill.** Cost per token falls as batch size rises (weight-load amortized over more sequences), so push `--gpu-memory-utilization` and `--max-num-seqs` up until the KV budget or the TTFT SLO — whichever binds first — says stop. Keep chunked prefill on with a `--max-num-batched-tokens` sized to protect TTFT: too large and a big prompt from another request can stall a fresh request's own first token; too small and prefill throughput (hence cost) suffers. This is the direct knob connecting the SLO to the cost target.

**5. Turn on prefix caching if traffic has any repeated structure.** For chat/RAG/agent workloads, this alone can remove a large fraction of prefill compute — approximately free in the current V1 engine even at a 0% hit rate, so there's no real reason to leave it off.

**6. Decide on disaggregation only after co-located tuning plateaus.** If, after all the above, TTFT is still being blown by prefill bursts stealing cycles from decode — and traffic/scale justify the operational cost — split prefill and decode onto separate GPU pools with independent autoscaling (section A.2). This is a scale-dependent decision: don't reach for it by default, it roughly doubles the operational surface area (two fleets, a KV-transfer connector, a routing proxy).

**7. Consider speculative decoding as a final latency lever, with eyes open on batch size.** If the SLO is tight and typical concurrency per GPU is low-to-moderate (i.e., you're not compute-saturated), an EAGLE-family draft head can materially cut TTFT-adjacent per-token latency; at high, cost-optimized batch sizes the benefit shrinks or reverses, so measure before committing it to the cost-optimized fleet specifically.

**8. Close the loop with load testing and live metrics.** State that the final numbers come from `vllm bench serve` sweeps against the real traffic shape, read alongside `gpu_cache_usage_perc` and `gpu_prefix_cache_hit_rate` (section B.4), not from a spec sheet — and that the operating point chosen is the cost-minimizing point on the throughput/latency curve that still clears the TTFT SLO, not the raw-throughput maximum.

**Putting numbers on it** (illustrative, using the B.6 cost table's configurations against a hypothetical TTFT SLO of p99 < 500 ms):

| Candidate config | Cost per 1M tokens (B.6) | Meets TTFT SLO at target QPS? | Decision |
|---|---|---|---|
| AWQ 4-bit, TP2, one node | ~$0.79 | No — smaller batch ceiling and 2 GPUs undersized for peak concurrency, TTFT p99 blows past 500 ms under load | Reject: cheapest per token but violates the SLO, which is a hard constraint, not a soft preference |
| FP8, TP4, one node | ~$1.23 | Yes, with headroom, and validated against a task-specific eval (section C.2) | **Likely pick** — clears the SLO and is meaningfully cheaper than the fp16 fleet |
| fp16, TP8×PP2, two nodes | ~$4.44 | Yes, with the most headroom | Reject as the default: clears the SLO but at ~3.6× the cost of the FP8 option for no additional SLO benefit — only justified if FP8 fails the quality eval |

This is the shape of answer that closes the loop: **the cheapest option that still clears the hard latency constraint, chosen from measured numbers, not the cheapest option in isolation and not the safest-but-most-expensive option by default.**

A strong answer names the tension between cost and latency explicitly at each step, rather than treating them as independently optimizable — that's usually the actual signal the interviewer is listening for.

> **Saying it out loud.** The structure that wins this one: treat the SLO as a hard constraint and cost as the thing you minimize inside it — never the other way round. So first, do the weight and KV math to enumerate feasible configurations: fp16 across two nodes, FP8 on one node, 4-bit on two GPUs. Second, compute cost per million tokens for each from a real `vllm bench serve` sweep, not a guess. Third, check each against the TTFT SLO under real load, and reject anything that violates it *regardless* of how cheap it is — the 4-bit config is often cheapest per token and still the wrong answer because its batch ceiling blows the tail latency. Fourth, gate the quantized option on a task-specific quality eval, not perplexity. That sequencing is the signal being graded.

### D.3 Red flags vs green flags

| Signal | Red flag (weak answer) | Green flag (strong answer) |
|---|---|---|
| Explaining PagedAttention | "It's like caching" / can't name blocks or block table | Names fixed block size (16), block table indirection, and the specific fragmentation numbers it eliminates |
| Batching | Conflates dynamic (Triton-style) batching with continuous/in-flight batching | Explains iteration-level scheduling: eviction + admission every step, not per-batch |
| Sizing KV cache | Guesses a `gpu-memory-utilization` number with no math | Derives bytes/token from layers × KV heads × head dim × dtype, subtracts weights, computes block count |
| Quantization | "4-bit is basically free, just do it" | Distinguishes AWQ/GPTQ+Marlin vs FP8 vs NVFP4 by hardware, and insists on task-specific eval, not just perplexity |
| Speculative decoding | Assumes it always helps | Ties benefit to acceptance rate and batch saturation; knows it can hurt at high concurrency |
| Parallelism | Picks TP or PP without justifying node topology | States TP-first-within-node, PP-to-cross-nodes, and why (comms cost profile) |
| Debugging a latency spike | Jumps straight to "add more GPUs" | Reads preemption logs / `gpu_cache_usage_perc` first, diagnoses over-commitment vs genuine capacity limit |
| Comparing engines | "vLLM is just the best one" with no caveat | Names SGLang's RadixAttention fit for prefix-heavy workloads and TensorRT-LLM's compile-time/throughput tradeoff, and says "benchmark your own traffic" |
| Talking about 2025–2026 state | Describes only the 2023 paper mechanics | Knows V1 is the current default engine, disaggregation is a live production pattern, and can name EAGLE-3.1 or a dated source |
| Production incident story | Generic "we scaled up" | Has a specific metric (preemption rate, pass-rate eval) that caught the problem and a specific config change that fixed it |

### D.4 Question bank (18 Q&A)

1. **Why is LLM decode memory-bound, and why does that make batching the key throughput lever?**
   Weights are re-read from HBM every token with comparatively little arithmetic per token; batching amortizes that read across many sequences' tokens. Throughput scales with batch size until KV memory runs out — which is why KV cache efficiency, not raw compute, is the binding constraint.

2. **Explain PagedAttention and what waste it eliminates.**
   Fixed-size KV blocks (default 16 tokens) in a global pool, addressed per-sequence via a block table (like an OS page table); eliminates internal fragmentation, reservation waste, and external fragmentation, raising effective KV utilization from ~20–38% to ~96%.

3. **Continuous vs static batching — why does continuous raise throughput *and* cut latency?**
   Iteration-level scheduling evicts finished sequences and admits new ones every step; no slot idles behind a slow sibling, so GPU utilization stays high and queueing (hence tail latency) drops too — a rare win on both axes at once.

4. **Walk me through sizing KV cache and picking `gpu-memory-utilization` / `max-num-seqs` / `max-model-len` for a given GPU.**
   Compute bytes/token from layers × KV heads × head dim × dtype bytes; subtract model weights and overhead from the memory budget to get the KV pool; divide by block size × bytes/token for block count; that determines realistic concurrency at a given context length. Preemption in logs means the chosen seq/context caps exceed this budget.

5. **When TP vs PP? What are the comms costs?**
   TP shards every layer, needs a per-layer all-reduce — bandwidth-hungry, so keep it intra-node over NVLink. PP splits layers into stages with point-to-point hand-offs — tolerant of slower links, so use it to cross nodes. Rule: TP first up to one node, PP to scale further.

6. **Quantization choices and their quality cost — AWQ vs GPTQ vs FP8 vs NVFP4, and when each?**
   AWQ/GPTQ (Marlin-kernel-backed) 4-bit weight-only for memory savings on Ampere/Ada; FP8 near-lossless with native tensor-core support on Hopper; NVFP4 is the emerging Blackwell-native path, less battle-tested. Always validate on a task-specific eval, not just perplexity — quantization damage is task-dependent.

7. **When does speculative decoding help vs hurt?**
   Helps at low-to-moderate batch with high draft-acceptance rate, because verification is nearly free on a memory-bound decode step. Hurts under saturation (GPU already compute-bound, verification stops being free) or low acceptance (you paid for a draft pass and got little back).

8. **How do you debug an OOM or a latency spike in production?**
   Check `# GPU blocks` / `Available KV cache memory` at startup and preemption warnings live; lower `gpu-memory-utilization`/`max-model-len` or add `swap-space` if over-committed; tune `max-num-batched-tokens` down for ITL if a big-prompt/decode interference pattern is visible; confirm prefix-cache hit rate hasn't collapsed under memory pressure.

9. **What's the difference between chunked prefill and disaggregated prefill/decode, and when do you reach for the latter?**
   Chunked prefill splits one big prefill into token-sized pieces co-scheduled with decodes on the *same* GPUs — cheap, on by default, solves most prefill/decode interference. Disaggregation moves prefill and decode to physically *separate* GPU pools with KV transferred between them over NVLink/RDMA — a bigger operational step reached for at high scale when co-location has been tuned out and ITL stability is still not good enough (e.g. the MORI-IO connector's ~2.5× goodput gain, at a TTFT cost).

10. **What does automatic prefix caching actually reuse, and what does it cost?**
    It reuses whole KV blocks whose content-hash (and preceding-token context) matches a cached block, skipping recomputation of that prefill span entirely — this is PagedAttention's block-sharing/CoW machinery applied across requests over time. Cost is competing with active batch for KV memory; under pressure cached blocks are evicted LRU and the hit rate — and the savings — fall. In the V1 engine it's engineered to cost almost nothing even at a 0% hit rate, so it's normally left on.

11. **Why does a 4-bit quantized checkpoint need a special kernel like Marlin to actually be fast?**
    INT4×FP16 mixed-precision GEMM is easy to get "free" at batch size 1 (memory-bound regime) but naive kernels leave tensor cores idle at realistic serving batch sizes, erasing the 4-bit advantage. Marlin is specifically engineered to hold close to the ideal 4× speedup up to ~32-token batches, which is why current vLLM auto-selects Marlin-backed kernels (`awq_marlin`/`gptq_marlin`) for AWQ/GPTQ checkpoints.

12. **Explain EAGLE-style speculative decoding and how it differs from an independent draft model.**
    EAGLE attaches a lightweight draft *head* directly to the target model's own hidden states, rather than running a fully separate small model — reusing the target's internal representations gives materially higher acceptance length than an independent draft of similar size. EAGLE-3.1 further fixes "attention drift" in deep speculation, roughly doubling accepted length over EAGLE-3 in long-context settings.

13. **What causes preemption, and what are the two ways vLLM handles it?**
    Preemption happens when the running batch's collective KV need exceeds the pool (e.g., growth crosses a new block boundary with no free blocks left). vLLM either **swaps** the lowest-priority sequences' KV to CPU RAM (cheap to restore, good for long sequences) or **discards and recomputes** it later (cheap for short sequences, expensive for long ones). Constant preemption in logs signals admission settings exceed true KV capacity for your traffic's actual (not average) shape.

14. **How would you decide between vLLM, SGLang, and TensorRT-LLM for a new deployment?**
    Default to vLLM for broad model/hardware support and fast iteration. Choose SGLang if the workload is dominated by prefix reuse (chat, RAG, few-shot, agents) — RadixAttention is purpose-built for that. Choose TensorRT-LLM if the model and shapes are fixed and long-lived enough to amortize the compile-time cost, and you need the last few percent of throughput on fixed NVIDIA hardware. Always benchmark your own model/hardware/traffic rather than trusting a spec sheet or a single third-party blog's numbers.

15. **Why does raising `max-num-seqs` sometimes do nothing?**
    Because KV memory, not the seq cap, is usually the real binding constraint — raising the cap without KV headroom just produces more preemption instead of more useful concurrency. Check `# GPU blocks` / `gpu_cache_usage_perc` before touching `max-num-seqs`.

16. **What's the relationship between `gpu-memory-utilization`, `max-model-len`, and the number of GPU blocks logged at startup?**
    `gpu-memory-utilization` sets the total HBM budget; subtracting weights and overhead gives the KV pool; dividing by bytes-per-block gives `# GPU blocks`; `max-model-len` bounds the worst-case blocks a single sequence can consume, which (with block count) determines the "maximum concurrency" figure vLLM logs.

17. **Walk through what happens end-to-end when a request arrives at a V1-engine vLLM server with prefix caching, chunked prefill, and speculative decoding all enabled.**
    Tokenized request enters the unified scheduler as `{request_id: num_tokens}`. Prefix cache is checked block-by-block; any matching prefix is pointed at existing physical blocks instead of recomputed. Remaining prefill tokens are chunked and co-scheduled with ongoing decodes up to `max-num-batched-tokens`. Each decode step, if speculative decoding is configured, the draft (EAGLE head / n-gram / small model) proposes several tokens, the target verifies them in one forward pass, accepted tokens are kept and the KV cache grows by however many were accepted. The scheduler frees blocks for any sequence completing this step and admits waiting requests into the vacated capacity.

18. **How do you decide whether disaggregated serving is worth adopting for a given deployment?**
    Only after co-located tuning (chunked prefill, prefix caching, batch sizing) has plateaued and ITL/TTFT SLOs are still not consistently met under real traffic bursts, and only if scale justifies doubling the operational surface (two independently-scaled fleets plus a KV-transfer connector and routing proxy). It's an optimization reached for at scale, not a default starting architecture — most single-node, moderate-QPS deployments should stop at chunked prefill co-location.

19. **What is llm-d, and why isn't it just "vLLM on Kubernetes with a Helm chart"?**
    llm-d (CNCF Sandbox, backed by Red Hat/Google/IBM engineers) packages the *distributed systems* problems that show up once vLLM is running as a fleet — KV-cache-aware routing via the Gateway API Inference Extension, and disaggregated prefill/decode as a supported deployment topology with the connector wiring solved — as "well-lit paths," rather than each team re-solving cache-affinity routing and disaggregation plumbing from scratch. It's an orchestration layer built on top of vLLM, not a replacement for it.

20. **You're paged for a p99 latency spike but throughput and error-rate dashboards look normal — what do you check first, and why?**
    Preemption signals (`vllm:num_preemptions_total` rate and "preempted" log lines) before anything else — a KV over-commitment/recompute-thrash episode (section C.1) is exactly the failure mode that spikes tail latency while aggregate throughput and error rate still look fine, because the GPU is genuinely busy the whole time, just redoing work instead of making new progress. This is the single highest-value "know which signal leads the symptom" habit from this chapter (section C.4).

---

---

## Glossary — quick lookup

| Term | One-line definition |
|---|---|
| **PagedAttention** | Attention mechanism that gathers K/V from non-contiguous, fixed-size GPU memory blocks via a per-sequence block table, eliminating KV-cache fragmentation. |
| **KV block** | Fixed-size (default 16-token) chunk of a sequence's key/value cache; the unit of allocation, sharing, and eviction in vLLM. |
| **Block table** | Per-sequence map from logical block index to physical block number — the "page table" of PagedAttention. |
| **Continuous / in-flight batching** | Scheduling at the granularity of one decode step: evict finished sequences and admit waiting ones every iteration, instead of running a batch to completion as a unit. |
| **Chunked prefill** | Splitting a large prefill into token-sized pieces co-scheduled with ongoing decodes in the same iteration, bounded by `max-num-batched-tokens`. |
| **Prefix caching** | Reusing already-computed KV blocks across requests whose prompts share a content-hashed prefix, skipping recomputation. |
| **Copy-on-write (CoW)** | When a shared KV block needs to diverge for one sequence, only that block is copied; other sharers are untouched. |
| **TTFT** | Time to first token — dominated by prefill compute and queueing; the "lag before it starts" a user feels. |
| **ITL / TPOT** | Inter-token latency / time per output token — decode-phase smoothness; hurt by large un-chunked prefills stalling decodes. |
| **Tensor parallelism (TP)** | Sharding each layer's weight matrices across GPUs, combined via a per-layer all-reduce; best over fast intra-node links (NVLink). |
| **Pipeline parallelism (PP)** | Splitting the model into sequential stages across GPUs/nodes, with point-to-point activation hand-offs; tolerant of slower inter-node links. |
| **AWQ / GPTQ** | Post-training weight-only quantization schemes (4-bit typical); need a Marlin-class kernel to realize speedup at realistic batch sizes, not just batch 1. |
| **Marlin** | GPU kernel achieving near-ideal 4× INT4×FP16 mixed-precision GEMM speedup up to ~32-token batches, underlying `awq_marlin`/`gptq_marlin`. |
| **FP8 (E4M3)** | 8-bit floating-point weight/activation format with native Hopper/Ada tensor-core support; near-lossless in practice for most models. |
| **NVFP4** | Emerging 4-bit floating-point format native to Blackwell-class tensor cores, exposed via NVIDIA Model Optimizer integration. |
| **Speculative decoding** | Cheap draft model/head proposes several tokens; target model verifies them in one parallel forward pass; output distribution is exact, not approximate. |
| **EAGLE / EAGLE-3 / EAGLE-3.1** | Draft-head speculative decoding methods that reuse the target model's own hidden states, yielding higher acceptance length than an independent draft model. |
| **Medusa** | Multiple parallel decoding heads trained on the target model, each predicting a fixed future offset token, verified alongside EAGLE-style methods. |
| **n-gram / prompt-lookup decoding** | Speculative drafting by matching repeated n-grams already in the prompt/generation, with no extra model weights. |
| **Preemption** | vLLM evicting lower-priority sequences from the running batch when KV demand exceeds supply — via swap (to CPU) or recompute (discard and redo). |
| **V1 engine** | vLLM's 2025 core rewrite: unified scheduler, isolated `EngineCore` process, chunked prefill and prefix caching on by default. |
| **Disaggregated prefill/decode** | Running prefill and decode on physically separate GPU pools, streaming KV cache between them via a connector (e.g. NVLink/RDMA). |
| **RadixAttention** | SGLang's radix-tree KV-cache index purpose-built for maximizing prefix-sharing across requests. |
| **llm-d** | CNCF Sandbox, Kubernetes-native distributed inference framework built on vLLM, adding KV-cache-aware routing and disaggregation as supported deployment topologies. |
| **Gateway API Inference Extension (IGW)** | Kubernetes routing extension used by llm-d that routes on LLM-specific signals (KV-cache locality, queue depth) instead of generic HTTP load metrics. |

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
- vLLM Blog, "vLLM V1: A Major Upgrade to vLLM's Core Architecture" (2025-01-27): https://vllm.ai/blog/2025-01-27-v1-alpha-release
- vLLM V1 usage guide: https://docs.vllm.ai/en/stable/usage/v1_guide/
- Red Hat Developer, "vLLM V1 Alpha: A major upgrade to vLLM's core architecture" (2025-01-28): https://developers.redhat.com/articles/2025/01/28/vllm-v1-a-major-upgrade-vllms-core-architecture
- vLLM Blog, "Next-Level Inference: Why Your Single-Node vLLM Setup Needs Prefill-Decode Disaggregation" (MORI-IO KV connector, 2026-04-07): https://vllm.ai/blog/2026-04-07-moriio-kv-connector
- vLLM disaggregated prefilling docs (experimental, v0.7.1): https://docs.vllm.ai/en/v0.7.1/features/disagg_prefill.html
- PyTorch Blog, "Disaggregated Inference at Scale with PyTorch & vLLM": https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/
- Ray Serve LLM, prefill/decode disaggregation guide: https://docs.ray.io/en/latest/serve/llm/user-guides/prefill-decode.html
- vLLM EAGLE Draft Models docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/
- vLLM Blog, "EAGLE 3.1: Advancing Speculative Decoding Through Collaboration Between the EAGLE Team, vLLM, and TorchSpec" (2026-05-26): https://vllm.ai/blog/2026-05-26-eagle-3-1
- vLLM Blog, "EAGLE-3 Speculative Decoding on AMD Instinct GPUs: Training and Serving with vLLM and AMD Quark" (2026-07-13): https://vllm.ai/blog/2026-07-13-eagle-3-amd-instinct
- vLLM Blog, "Diving into speculative decoding training support for vLLM with Speculators v0.3.0" (2025-12-13): https://vllm.ai/blog/2025-12-13-speculators-v030
- Red Hat Developer, "Fly Eagle(3) fly: Faster inference with vLLM & speculative decoding" (2025-07-01): https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding
- Red Hat Developer, "Performance improvements with speculative decoding in vLLM for gpt-oss" (2026-04-16): https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss
- Red Hat Developer, "How Marlin pushes the boundaries of mixed-precision LLM inference" (2024-04-17): https://developers.redhat.com/articles/2024/04/17/how-marlin-pushes-boundaries-mixed-precision-llm-inference
- Frantar et al., "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models" (paper): https://arxiv.org/pdf/2408.11743
- vLLM AWQ-Marlin quantization API reference: https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/awq_marlin/
- vLLM FP8 W8A8 docs: https://docs.vllm.ai/en/v0.8.5/features/quantization/fp8.html
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (RadixAttention paper): https://arxiv.org/abs/2312.07104
- Spheron Blog, "vLLM vs TensorRT-LLM vs SGLang: Which Is Fastest? (H100 Benchmarks, 2026)" (2026-03-23): https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/
- AMD ROCm Blog, "vLLM V1 Meets AMD Instinct GPUs: A New Era for LLM Inference Performance": https://rocm.blogs.amd.com/software-tools-optimization/vllmv1-rocm-llm/README.html
- NVIDIA TensorRT-LLM project: https://github.com/NVIDIA/TensorRT-LLM
- llm-d project, "Announcing the llm-d community!" (2025-05-20): https://llm-d.ai/blog/llm-d-announce
- llm-d GitHub repository: https://github.com/llm-d/llm-d
- llm-d, "llm-d 0.2: Our first well-lit paths": https://llm-d.ai/blog/llm-d-v0.2-our-first-well-lit-paths
- Red Hat Developer, "llm-d: Kubernetes-native distributed inferencing" (2025-05-20): https://developers.redhat.com/articles/2025/05/20/llm-d-kubernetes-native-distributed-inferencing
- Red Hat Developer, "Introduction to distributed inference with llm-d" (2025-11-21): https://developers.redhat.com/articles/2025/11/21/introduction-distributed-inference-llm-d
- vLLM Blog, "vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP" (2025-12-17): https://vllm.ai/blog/2025-12-17-large-scale-serving
- vLLM Distributed inference and serving (Ray, multi-node): https://docs.vllm.ai/en/latest/serving/distributed_serving.html
- Kubernetes Gateway API Inference Extension project: https://github.com/kubernetes-sigs/gateway-api-inference-extension
- vLLM GitHub repository (issues, release notes, source of truth for current flag defaults): https://github.com/vllm-project/vllm
- vLLM Blog index (check here for what's shipped since this chapter was written): https://vllm.ai/blog
- vLLM Prometheus/Grafana production metrics guide: https://docs.vllm.ai/en/latest/usage/metrics.html
- Kubernetes Horizontal Pod Autoscaler, custom/external metrics docs (for scaling on queue depth, section C.3): https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- vLLM `vllm bench` CLI reference (benchmarking harness used throughout section B): https://docs.vllm.ai/en/latest/cli/bench/

---

*A closing note on staying current: everything in section (A) and the timeline in (A.7) is a snapshot as of this writing. vLLM, SGLang, and TensorRT-LLM all ship fast — engine defaults, kernel selection, and quantization coverage change between minor versions. Before an interview or a production decision, re-check `vllm serve --help` for your installed version and skim the last few entries at `vllm.ai/blog` rather than relying on any single source, including this chapter, as permanently current.*
