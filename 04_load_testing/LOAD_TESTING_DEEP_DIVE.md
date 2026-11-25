# Load Testing LLM Inference — Measuring Throughput and Latency Correctly Under Load

## Why This Matters

You cannot capacity-plan, price, or SLA a serving system you have not load-tested. And LLM inference is unusually easy to benchmark *wrong*: a single request against an idle server tells you almost nothing about behavior at 200 concurrent users, because the whole point of a modern engine (vLLM, TGI, TensorRT-LLM) is **continuous batching** — throughput and latency both change as concurrency changes.

The stakes are concrete:

- **Capacity planning.** "How many GPUs do I need for 500 chat sessions?" is answerable only from a concurrency sweep.
- **SLA definition.** "p95 time-to-first-token under 500 ms" is meaningless without saying *at what load*.
- **Cost.** Output tokens/second/GPU is the number your finance model divides into. A 2× throughput win halves your bill.
- **Regression gating.** A benchmark you can rerun in CI catches the day someone flips `--enable-chunked-prefill` off.

A wrong benchmark is worse than none: it gives false confidence. Most of this chapter is about the ways benchmarks lie, and how to stop them.

> **Saying it out loud.** You can't capacity-plan, price, or write an SLA for a system you haven't load-tested — and LLM inference is unusually easy to benchmark wrong. One request against an idle server tells you essentially nothing about behavior at 200 concurrent users, because the entire point of a modern engine is continuous batching, which means throughput *and* latency both change as concurrency changes. Concretely: "p95 time-to-first-token under 500 milliseconds" is a meaningless claim unless you say at what load. And output tokens per second per GPU is the number your finance model divides into, so a 2x throughput win literally halves the bill. The framing I'd use: a wrong benchmark is worse than no benchmark, because it buys you false confidence right up until launch day.

---

## Core Intuition

### Why average latency lies

Latency distributions in a batched system are **heavy-tailed and multi-modal**. A request that lands in an empty batch returns fast; an identical request that lands when the batch is full waits for a scheduler slot, then shares GPU compute with 63 neighbors. Same input, wildly different latency.

Average that distribution and you get a number that describes *no actual request*. Consider ten requests with latencies (ms):

```
90, 95, 100, 100, 105, 110, 110, 120, 130, 2000
```

The mean is **296 ms**. Nine of ten users saw ≤130 ms; one saw 2 s. The mean reports a latency nobody experienced and hides the 2 s tail that will dominate your support tickets. The **p90 is 130 ms** and the **p100 (max) is 2000 ms** — those describe reality. Tail latency is where user pain, timeouts, and retry storms live, so you report **percentiles**, never just the mean.

Rule: **the mean is for throughput accounting; percentiles are for latency SLAs.**

> **Saying it out loud.** Latency in a batched system is heavy-tailed and multi-modal — the same request is fast if it lands in an empty batch and slow if it lands behind sixty-three neighbors. So the mean describes no actual request. Take ten samples: nine between 90 and 130 milliseconds, one at two seconds. The mean is 296 milliseconds, which nobody experienced, and it completely buries the two-second tail that's going to generate your support tickets. The p90 is 130 and the max is 2000 — those describe reality. The rule I'd give: the mean is for throughput accounting, percentiles are for latency SLAs. Tail latency is where timeouts, retries, and retry storms live, and none of them show up in an average.

### Why concurrency defines the operating point

There is no single "latency" or "throughput" for a serving system — there is a **curve** parameterized by load. As you push more concurrent requests:

1. **Low load:** GPU underutilized. Latency is flat and near-minimal. Throughput rises roughly linearly with concurrency.
2. **The knee:** GPU compute (or KV-cache memory) saturates. Throughput flattens — you've hit the roofline. Latency starts climbing because requests now queue.
3. **Overload:** Throughput is flat (or falls, from scheduling/paging overhead), but latency climbs without bound as the queue grows.

```
 throughput (tok/s)                    latency p95 (ms)
        |            ____________              |                 /
        |          /                           |               /
        |        /   <- knee                   |         _____/
        |      /                               |    ____/
        |    /                                 |___/
        |  /                                   |
        |/____________________ concurrency     |________________ concurrency
```

The engineering goal is to **find the knee** and operate just below it: that's where you get near-peak throughput while latency is still bounded. A benchmark that reports one concurrency level has told you one point on a two-dimensional curve. Always **sweep**.

> **Saying it out loud.** There is no single latency or throughput number for a serving system — there's a curve parameterized by load, and it has three regions. At low load the GPU is underutilized, latency is flat, and throughput rises roughly linearly with concurrency. Then you hit the knee, where compute or KV-cache memory saturates: throughput flattens out and latency starts climbing because requests are now queueing. Past that is overload, where throughput is flat or falling and latency climbs without bound. The engineering goal is to find the knee and operate just below it — near-peak throughput while latency is still bounded. Which means a benchmark that reports one concurrency level has told you exactly one point on a two-dimensional curve. Always sweep.

---

## Metrics, Defined Precisely

Let a single streaming request produce output tokens at wall-clock times \\( t_1 < t_2 < \dots < t_N \\), with the request sent at \\( t_0 \\). Over a whole test, let \\( R \\) requests complete in wall-clock window \\( T \\) seconds, producing \\( O \\) total output tokens.

> **Saying it out loud.** There are really six numbers and you should be able to define each precisely. Time to first token is when the cursor starts moving — prefill plus any queueing wait. Time per output token is the average gap between tokens after the first, which is your perceived typing speed. End-to-end is the sum of those, and it scales with output length, so it's only comparable across runs whose output distributions match. Request throughput is completed requests per second; output-token throughput is generated tokens per second across everyone, and that's the money metric. And every latency number gets reported at p50, p95, and p99, never as a mean. The subtlety worth naming: a "latency regression" is very often just longer outputs, not a slower system.

### Time To First Token (TTFT)

\\[ \text{TTFT} = t_1 - t_0 \\]

The latency until the first token appears. Dominated by **prefill** (processing the prompt) plus any **queueing wait** for a scheduler slot. This is what a user perceives as "responsiveness" — the cursor starting to move. TTFT is the metric most sensitive to load, because a queued request pays its wait entirely before \\( t_1 \\).

*Micro-example:* prompt sent at \\( t_0 = 0 \\), first token at \\( t_1 = 0.18\text{ s} \\) → TTFT = 180 ms.

> **Saying it out loud.** TTFT is simply the time from sending the request to the first token arriving, and it's what a user experiences as responsiveness — the cursor starting to move. Two things go into it: prefill, meaning the model processing the whole prompt in one compute-bound pass, plus however long the request sat in a queue waiting for a scheduler slot. That second term is why TTFT is the metric most sensitive to load — a queued request pays its entire wait before the first token ever appears, so TTFT p95 is usually the first thing to blow up as you approach saturation. Concretely, if you send at time zero and the first token lands at 180 milliseconds, that's your TTFT, and under load that same request might see 2 seconds without anything about the model changing.

### Time Per Output Token (TPOT) / Inter-Token Latency (ITL)

**TPOT** is the average gap between output tokens after the first, for one request:

\\[ \text{TPOT} = \frac{t_N - t_1}{N - 1} \\]

**ITL** is the per-gap version — the distribution of individual \\( t_{i+1} - t_i \\) values. TPOT is the mean of a request's ITLs. (Tools differ: vLLM reports both TPOT and ITL; some tools call TPOT "inter-token latency." Know which your tool means.) TPOT is governed by the **decode** phase; \\( 1/\text{TPOT} \\) is the per-user tokens/second — the perceived "typing speed."

*Micro-example:* a request emits 201 tokens, \\( t_1 = 0.18 \\), \\( t_{201} = 4.18 \\). TPOT \\( = (4.18 - 0.18)/200 = 20\text{ ms} \\) → each user sees ~50 tokens/s.

> **Saying it out loud.** TPOT is the average gap between output tokens after the first — total generation time divided by the number of gaps — and one over TPOT is the per-user tokens per second, the perceived typing speed. ITL is the same thing but as a distribution rather than an average, so TPOT is really the mean of a request's ITLs. Where TTFT is governed by prefill, TPOT is governed by decode, which is the memory-bandwidth-bound phase. Concretely: a request emitting 201 tokens with a 20-millisecond TPOT gives the user about 50 tokens per second. One practical warning — tools disagree on naming, and some call TPOT "inter-token latency," so check which your tool means before comparing numbers across tools.

### End-to-End (E2E) Request Latency

\\[ \text{E2E} = t_N - t_0 = \text{TTFT} + (N-1)\cdot\text{TPOT} \\]

Total time from send to last token. **This is the number that scales with output length**, so it is only comparable across runs if output-length distributions match. A "latency regression" is often just longer outputs.

> **Saying it out loud.** End-to-end is just TTFT plus the decode time for the rest of the tokens — send to last token. The critical property is that it scales directly with output length, which makes it the most commonly misread metric in the whole chapter. If your E2E p95 jumps 40% between two runs, the first thing to check isn't the server, it's whether the output-length distribution changed, because a "latency regression" is very often just longer answers. That's why you either hold the output distribution fixed across runs, or you report normalized latency — E2E divided by token count — so length divides out and the runs become comparable.

### Normalized latency (per-token E2E)

\\[ \text{normalized latency} = \frac{\text{E2E}}{N} \\]

Divides out output length, making runs with different output distributions comparable. vLLM's benchmark historically reported this.

### Request throughput

\\[ \lambda_{\text{out}} = \frac{R}{T} \quad [\text{req/s}] \\]

Completed requests per second across the whole run. The right top-line number when your unit of work is "a request" (e.g., classification).

### Output-token throughput

\\[ X = \frac{O}{T} \quad [\text{tok/s}] \\]

Generated tokens per second across all concurrent requests. **This is the money metric for generative workloads** and the one that goes up with better batching. Report **output** tokens (excludes the prompt); also report **total token throughput** (prompt + output) if prefill cost matters to you. Divide by GPU count for tok/s/GPU.

*Micro-example:* 64 concurrent requests each streaming at 50 tok/s → aggregate \\( X \approx 3200 \\) tok/s, even though each *user* still sees only 50 tok/s. Per-user speed and aggregate throughput are different axes.

> **Saying it out loud.** Output-token throughput is total generated tokens divided by wall-clock time, aggregated across every concurrent request — and for generative workloads it's the money metric, because it's what improves when batching improves and it's what your cost per token divides into. The distinction that trips people up: per-user speed and aggregate throughput are different axes entirely. Sixty-four concurrent requests each streaming at a modest 50 tokens per second is 3,200 tokens per second aggregate, even though no individual user sees anything faster than 50. Report output tokens separately from prompt tokens, since prefill and decode cost differently, and divide by GPU count so you get tokens per second per GPU — that's the number that actually compares across hardware.

### Percentiles

For a metric with sorted samples, the p-th percentile \\( P_p \\) is the smallest value \\( \geq p\\% \\) of samples:

\\[ P_p = \text{value at rank } \lceil \tfrac{p}{100} \cdot n \rceil \text{ in sorted order} \\]

Report **p50 (median), p95, p99** for TTFT, TPOT, and E2E. p99 matters more than it looks: if a page makes 10 backend calls, the chance all 10 beat p99 is \\( 0.99^{10} \approx 0.90 \\) — so ~10% of *page loads* hit a p99 tail. Tail latency compounds.

A practical note on computing these from a live stream of samples rather than a saved-then-sorted list: naive percentile computation needs the full sorted sample set in memory, which is fine for a single sweep level (thousands of samples) but awkward for a long-running soak test. If you need percentiles over an unbounded stream, use a streaming quantile sketch (e.g. t-digest or HDRHistogram) instead of re-sorting on every report — both are available as Python/Go/JS libraries and are what k6 and Gatling use internally to report percentiles without buffering every sample.

> **Saying it out loud.** Report p50, p95, and p99 for TTFT, TPOT, and end-to-end separately — and p99 matters more than people intuit. Here's why: if a single page makes ten backend calls, the probability all ten beat p99 is 0.99 to the tenth, about 90% — so roughly one in ten *page loads* hits a p99 tail even though only one in a hundred requests does. Tail latency compounds with fan-out. One practical note for long soak tests: computing percentiles naively needs the whole sorted sample set in memory, which is fine for a sweep point but awkward over hours, so use a streaming quantile sketch like t-digest or HDRHistogram — that's what k6 and Gatling do internally.

---

## Open-Loop vs Closed-Loop Load Generation

This is the single most important methodology decision, and the one most benchmarks get wrong.

> **Saying it out loud.** This is the single most important methodology decision, and it's the one most benchmarks get wrong. Closed-loop means a fixed pool of virtual users, each of which sends a request, waits for the full response, then sends the next — so concurrency is capped by construction. Open-loop means requests fire on an arrival schedule, usually Poisson, completely independent of whether earlier requests have finished, so in-flight concurrency is an emergent property free to grow when the server slows down. Real user traffic is open-loop: people arrive whether or not you're keeping up. That difference isn't academic — it's the entire reason a green pre-launch test can be followed by a red launch day.

### Closed-loop

A fixed pool of \\( C \\) "virtual users." Each sends a request, **waits for the full response**, then immediately sends the next. Concurrency is capped at \\( C \\) by construction. This models a fixed number of clients in a tight loop (e.g., a batch job, or exactly \\( C \\) synchronous callers).

### Open-loop

Requests are launched on an **arrival schedule** (e.g., Poisson at rate \\( \lambda \\)) **independent of whether prior requests have finished**. In-flight concurrency is an *emergent property*, free to grow if the server slows down. This models real traffic: users arrive whether or not your server is keeping up.

### Coordinated omission — why closed-loop hides overload

Here is the trap. In a closed-loop test, when the server slows down, each virtual user's loop stalls waiting for its response — so **it stops sending new requests**. The offered load automatically backs off exactly when the system is struggling. The load generator "coordinates" with the server's slowness and **omits** the requests a real (open) world would have kept sending.

Consequences:

- Your measured request rate silently drops below target, and you may not notice.
- The latency samples you *do* collect exclude the requests that would have queued behind a slow one, so **tail latency is dramatically underreported**. The one 2-second stall in a real system would have delayed 40 requests behind it; closed-loop just... didn't send them.
- You conclude the system is healthy at a load it actually cannot sustain.

The fix in an open-loop tool is to **schedule requests on absolute wall-clock times** and measure each request's latency from its *intended* send time, not from when a freed-up worker got around to it. k6's `constant-arrival-rate`/`ramping-arrival-rate` executors and Gatling's open model do this; `constant-vus`/`ramping-vus` are closed. Corrected closed-loop tools (wrk2, Gatling) reconstruct the omitted samples by back-dating latency to the scheduled time.

> **Saying it out loud.** Here's the trap, and it's genuinely subtle. In a closed-loop test, when the server slows down, each virtual user stalls waiting for its response — which means it stops sending. So the offered load automatically backs off at exactly the moment the system is struggling. The generator has "coordinated" with the server's slowness and omitted the requests a real world would have kept sending. Two consequences: your measured rate silently drops below target, and your latency samples exclude everything that would have queued behind the slow request — so tail latency is dramatically underreported. One two-second stall in production delays forty requests behind it; closed-loop just never sent them. The fix is to schedule on absolute wall-clock times and measure latency from *intended* send time.

### When each is right

- **Open-loop / fixed request-rate:** validating an SLA against realistic traffic ("can we hold p95 TTFT < 500 ms at 30 req/s?"). This is the honest default for user-facing services.
- **Closed-loop / fixed concurrency:** finding **maximum sustainable throughput** and the saturation curve, where a bounded, known concurrency is exactly the independent variable you want to sweep. LLMPerf and GenAI-Perf's concurrency mode work this way, and that is fine *because you are deliberately measuring the throughput ceiling*, not pretending to reproduce arrival traffic.

Use both: a concurrency sweep to find the knee, then an open-loop test at your chosen arrival rate to confirm the SLA holds with realistic tail behavior.

> **Saying it out loud.** Both models are legitimate, they just answer different questions. Open-loop at a fixed arrival rate is what you use to validate an SLA against realistic traffic — "can we hold p95 TTFT under 500 milliseconds at 30 requests per second" — and that's the honest default for anything user-facing. Closed-loop at fixed concurrency is what you use to find maximum sustainable throughput, because there a bounded known concurrency is exactly the independent variable you want to sweep. That's why LLMPerf and GenAI-Perf's concurrency mode are fine — they're deliberately measuring a ceiling, not pretending to reproduce arrival traffic. The practice: sweep closed-loop to find the knee, then run open-loop at your chosen rate to confirm the tail behavior holds.

---

## Little's Law — Reasoning About Concurrency

Little's Law relates the three quantities you care about, for any stable system in steady state:

\\[ L = \lambda \\, W \\]

- \\( L \\) = average number of requests **in the system** (concurrency / in-flight).
- \\( \lambda \\) = average **arrival = completion rate** (req/s), in steady state.
- \\( W \\) = average **time in system** (E2E latency, seconds).

It is an identity — no assumptions about distributions. Uses:

**Sanity-check a benchmark.** If your open-loop test offers \\( \lambda = 20 \\) req/s and you measure mean E2E \\( W = 4 \\) s, then average in-flight concurrency is \\( L = 20 \times 4 = 80 \\). If your engine's `--max-num-seqs` is 64, you are **oversubscribed**: requests are queuing, latency will keep rising, and the system is not in steady state. The law just told you the offered load exceeds capacity *before* you stare at a climbing latency graph.

**Convert closed-loop to a rate.** A closed-loop test with \\( C = 100 \\) virtual users measuring \\( W = 2.5 \\) s achieves \\( \lambda = L/W = 100/2.5 = 40 \\) req/s. That's how you translate a concurrency-sweep point into "requests per second this config sustains."

**Token form.** Apply it to tokens: aggregate output throughput \\( X \\) (tok/s) with \\( n \\) requests in flight each of length \\( N \\) tokens taking \\( W \\) seconds gives \\( X = nN/W \\) — decompose regressions into "fewer concurrent" vs "slower per request."

Caveat: Little's Law holds **in steady state**. During warmup, ramp, or overload it does not, which is one more reason to discard warmup and to hold each sweep point long enough to stabilize.

> **Saying it out loud.** Little's Law says the average number of requests in the system equals arrival rate times average time in system — L equals lambda W — and it's an identity, no distributional assumptions at all. Two uses make it worth memorizing. Sanity-checking: if you offer 20 requests per second and measure a mean end-to-end of 4 seconds, average in-flight concurrency is 80. If your engine's `--max-num-seqs` is 64, you're oversubscribed and the system isn't in steady state — the law told you that before you stared at a climbing graph. And converting: a closed-loop test with 100 virtual users at 2.5 seconds mean latency is sustaining 40 requests per second. The caveat: it only holds in steady state, so discard warmup and hold each sweep point long enough to stabilize.

---

## A Fully Worked Example — Async Open-Loop Client

Below is a self-contained async Python client that drives an OpenAI-compatible streaming endpoint (vLLM, TGI, etc.), generates a **Poisson arrival schedule** (true open-loop — arrivals do not wait for completions), records **TTFT/TPOT/E2E per request from the intended send time** (coordinated-omission-safe), discards **warmup**, and prints **percentiles and throughput**.

```python
#!/usr/bin/env python3
"""Open-loop load test for an OpenAI-compatible LLM endpoint.

Launches requests on a Poisson schedule at a target rate, independent of
whether prior requests finished, and measures latency from the INTENDED
send time to avoid coordinated omission.

Usage:
  python load_test.py --url http://localhost:8000/v1/chat/completions \
      --model my-model --rate 20 --duration 60 --warmup 10
"""
import argparse, asyncio, json, random, statistics, time
import aiohttp

PROMPTS = [
    "Explain how continuous batching improves LLM throughput.",
    "Write a haiku about GPU memory fragmentation.",
    "Summarize the tradeoffs of speculative decoding in three sentences.",
    "What is time-to-first-token and why does it depend on load?",
]

async def one_request(session, url, model, prompt, max_tokens, sched_t, t0, results):
    """Fire one request. Latency is measured from sched_t (intended send),
    NOT from now — this is the coordinated-omission fix."""
    # Sleep until this request's scheduled arrival time (open-loop).
    delay = sched_t - (time.perf_counter() - t0)
    if delay > 0:
        await asyncio.sleep(delay)
    intended = t0 + sched_t                       # absolute intended send time
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    token_times, first_t, err = [], None, None
    send = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    now = time.perf_counter()
                    if first_t is None:
                        first_t = now
                    token_times.append(now)
    except Exception as e:                          # noqa: BLE001
        err = repr(e)
    end = time.perf_counter()
    n = len(token_times)
    results.append({
        "intended": intended,
        "ttft": (first_t - send) if first_t else None,
        # E2E measured from INTENDED time, not send — captures scheduling debt.
        "e2e": end - send,                          # from actual send time
        "e2e_true": end - (t0 + sched_t),           # from intended arrival (CO-safe)
        "n_tokens": n,
        "tpot": ((token_times[-1] - first_t) / (n - 1)) if n > 1 else None,
        "error": err,
    })

def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * len(xs) + 0.5)) - 1))
    return xs[k]

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rate", type=float, default=10.0, help="req/s (Poisson mean)")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds")
    ap.add_argument("--warmup", type=float, default=10.0, help="seconds to discard")
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    # Pre-build a Poisson arrival schedule: gaps ~ Exponential(rate).
    schedule, t = [], 0.0
    while t < args.duration:
        t += random.expovariate(args.rate)
        schedule.append(t)

    results = []
    conn = aiohttp.TCPConnector(limit=0)            # no client-side cap!
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(one_request(
                session, args.url, args.model,
                random.choice(PROMPTS), args.max_tokens, s, t0, results))
            for s in schedule
        ]
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

    # Drop warmup window and errored requests.
    ok = [r for r in results if r["error"] is None
          and r["intended"] - t0 >= args.warmup]
    measured_window = wall - args.warmup
    ttfts = [r["ttft"] * 1000 for r in ok if r["ttft"]]
    tpots = [r["tpot"] * 1000 for r in ok if r["tpot"]]
    e2es = [r["e2e_true"] * 1000 for r in ok]
    out_tokens = sum(r["n_tokens"] for r in ok)

    print(f"\n=== target rate {args.rate} req/s | wall {wall:.1f}s | "
          f"measured window {measured_window:.1f}s ===")
    print(f"requests ok: {len(ok)}  errors: {sum(1 for r in results if r['error'])}")
    print(f"achieved req/s:      {len(ok)/measured_window:8.2f}")
    print(f"output tok/s:        {out_tokens/measured_window:8.1f}")
    for name, xs in (("TTFT ms", ttfts), ("TPOT ms", tpots), ("E2E  ms", e2es)):
        if xs:
            print(f"{name:9s} p50 {pct(xs,50):8.1f}  p95 {pct(xs,95):8.1f}  "
                  f"p99 {pct(xs,99):8.1f}  mean {statistics.mean(xs):8.1f}")

if __name__ == "__main__":
    asyncio.run(main())
```

Two design points that matter:

- **`TCPConnector(limit=0)`** removes the client's own connection cap. If you leave aiohttp's default (100) or run one CPU core hot parsing SSE, *the client* becomes the bottleneck and you benchmark your laptop, not the server. (See pitfalls.)
- **`e2e_true` measures from the intended arrival time** (`t0 + sched_t`), so a request delayed because the event loop was busy still counts its full latency — coordinated-omission-safe. (Use the `e2e_true` field, `end - (t0 + sched_t)`, for reporting; the plain `e2e` from actual send time will under-report latency when the event loop falls behind.)

> **Saying it out loud.** A correct load-testing client has five properties, and it's worth being able to list them. It generates a Poisson arrival schedule so arrivals don't wait on completions — that's what makes it genuinely open-loop. It records TTFT, TPOT, and end-to-end per request measured from the *intended* send time, not from when a free worker got around to it, which is what makes it coordinated-omission-safe. It discards a warmup window, because cold TTFT can be five to fifty times steady-state and a handful of those samples will wreck your p99. It reports percentiles rather than means. And it parses SSE streaming properly, because you cannot measure time-to-first-token from a non-streaming response at all.

### Sample results — a concurrency/rate sweep

Run the client at increasing rates against one A100 serving an 8B model, fixed input ≈512 / output ≈200 tokens:

| Target rate (req/s) | Achieved req/s | Output tok/s | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | E2E p95 (ms) | Avg in-flight \\(L=\lambda W\\) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5   | 5.0  | 1000 | 42   | 70   | 15 | 3200  | 16 |
| 10  | 10.0 | 2000 | 55   | 95   | 17 | 3600  | 36 |
| 20  | 20.0 | 4000 | 88   | 210  | 21 | 4400  | 88 |
| 30  | 29.8 | 5900 | 180  | 620  | 28 | 6800  | 200 |
| 40  | 33.1 | 6600 | 540  | 2400 | 41 | 14200 | 470 |
| 50  | 32.9 | 6600 | 1900 | 9000 | 63 | 41000 | 1350 |

> **Saying it out loud.** A sweep table is worth reading out loud once, because the pattern is the whole lesson. On one A100 serving an 8B model: at 5 through 20 requests per second, achieved rate tracks target and output throughput scales roughly linearly from 1,000 to 4,000 tokens per second, with TTFT p95 staying under a quarter second. At 30 you're at the knee — 5,900 tokens per second, still hitting target, but TTFT p95 has jumped to 620 milliseconds. Then at 40 and 50 the achieved rate flatlines at about 33 per second no matter what you offer, while TTFT p95 goes from 2.4 seconds to 9. Offering 50 doesn't get you 50; it just grows the queue. That flatline is your true saturation throughput.

### Reading the saturation curve

- **5 → 20 req/s:** achieved rate tracks target, output tok/s scales ~linearly (1000→4000), TTFT p95 stays modest. Underloaded region — GPU has headroom.
- **~30 req/s is the knee.** Output tok/s (5900) is close to the ceiling; achieved rate still ≈ target but TTFT p95 has jumped to 620 ms. This is the **operating point** you'd target for a latency-sensitive service, maybe backing off to ~25 for headroom.
- **40 → 50 req/s: overload.** Achieved rate **flatlines at ~33 req/s** even as you offer more — that ~33 req/s (≈6600 tok/s) is the true saturation throughput. Meanwhile latency explodes (TTFT p95 2.4 s → 9 s; E2E p95 41 s) and Little's-Law in-flight \\( L \\) blows past any sane `max-num-seqs`. Offering 50 doesn't get you 50; it just grows the queue.

The signature of the knee: **throughput stops rising while latency starts rising superlinearly.** Peak throughput and acceptable latency are *different* points — publish both, and state which you're operating at.

> **Saying it out loud.** The signature of the knee is one sentence: throughput stops rising while latency starts rising superlinearly. Below it, achieved rate tracks the target and tokens per second scale roughly with load. At it, throughput is near ceiling but the tail has begun to move. Past it, achieved rate flatlines no matter how much more you offer, and the extra load just grows the queue — Little's Law will show in-flight concurrency blowing well past any sane `max-num-seqs`. The thing to actually say in a review: peak throughput and acceptable latency are *different operating points*, so publish both numbers and state clearly which one you're running at. Most teams quote the peak and operate at it, which is exactly how you end up with no headroom for a traffic spike.

---

## A Second Worked Example — Locust (open-model)

Locust is convenient for HTTP services and dashboards. Locust is closed-loop by default (each user loops), but the `constant_throughput`/`constant_pacing` shape plus a high user count approximates open arrivals. Here is a `locustfile.py` that measures streaming TTFT and records it as a custom metric:

```python
# locustfile.py  —  run:  locust -f locustfile.py --host http://localhost:8000
import json, time
from locust import HttpUser, task, constant_throughput

class LLMUser(HttpUser):
    # Each user targets 1 req/s; scale arrivals via -u (number of users).
    # constant_throughput paces to a rate rather than back-to-back looping,
    # which is closer to open-loop than the default.
    wait_time = constant_throughput(1.0)

    @task
    def chat(self):
        payload = {
            "model": "my-model",
            "messages": [{"role": "user", "content": "Explain KV cache paging."}],
            "max_tokens": 200, "stream": True, "temperature": 0.0,
        }
        start = time.perf_counter()
        first_t = None
        n_tokens = 0
        with self.client.post("/v1/chat/completions", json=payload,
                              stream=True, catch_response=True,
                              name="chat-stream") as resp:
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0]["delta"].get("content")
                if delta:
                    if first_t is None:
                        first_t = time.perf_counter()
                    n_tokens += 1
            # Report TTFT as a named event so it shows in Locust stats/percentiles.
            if first_t is not None:
                ttft_ms = (first_t - start) * 1000
                self.environment.events.request.fire(
                    request_type="METRIC", name="TTFT_ms",
                    response_time=ttft_ms, response_length=n_tokens,
                    exception=None, context={})
```

Locust's own percentile table then gives you p50/p95/p99 for both the full request and the synthetic `TTFT_ms` event. Drive concurrency with `-u <users>` and `-r <ramp>`; use the web UI's charts to watch the knee live. Caveat: Locust workers are Python and can become the bottleneck — **run distributed workers** (`--worker`) and confirm client CPU isn't saturated before trusting numbers at high load.

> **Saying it out loud.** Locust is worth knowing because it's convenient — a real dashboard, easy custom flows, arbitrary Python task logic — but there's a catch you have to state upfront: it's closed-loop by default, since every user loops. You can approximate open arrivals with `constant_throughput` pacing plus a high user count, but that's an approximation, not an arrival-rate executor. And measuring TTFT requires custom code, because you have to parse the SSE stream yourself and record the first chunk's timestamp as a custom metric — the built-in timings only know about the whole response. So: Locust for bespoke multi-step traffic where you want a dashboard, k6 when the honesty of the load model is the point.

---

## Tools Comparison

| Tool | Load model | Metrics reported | Endpoints | Best for | Watch out |
|---|---|---|---|---|---|
| **vLLM `benchmark_serving.py` / `vllm bench serve`** | Open-loop via `--request-rate` (`inf` = burst all `--num-prompts` at once); Poisson/`--burstiness` | Request throughput (req/s), Output token throughput (tok/s), Total token throughput, Mean/Median/P99 **TTFT, TPOT, ITL**, E2E | OpenAI-compatible + native vLLM/TGI backends | Purpose-built LLM serving benchmarks; realistic datasets (`--dataset-name sharegpt/random/sonnet`) | Ships with vLLM version; `--request-rate inf` is a burst, not steady rate |
| **LLMPerf** (Ray) | Closed-loop, `--num-concurrent-requests` | TTFT, inter-token latency, E2E, output throughput per-request and aggregate | Many providers (OpenAI, Anthropic, Together, Bedrock, Vertex, SageMaker, vLLM) | Cross-provider apples-to-apples; the LLMPerf leaderboard | Concurrency mode = throughput ceiling, not arrival-rate SLA; token counts are provider-tokenized approximations |
| **NVIDIA GenAI-Perf** (Triton Perf Analyzer) | Both: `--concurrency` (closed) or `--request-rate` (open) | TTFT, inter-token latency, output token throughput, request throughput, seq lengths, all with avg/p90/p99 | OpenAI-compatible, Triton (TRT-LLM, vLLM), gRPC/HTTP | Deep NVIDIA/Triton stacks; synthetic + custom datasets; rich exports | Heavier setup; Triton-centric defaults |
| **Locust** | Closed-loop by default; `constant_throughput` ≈ open | Whatever you instrument; built-in RPS + percentile table + web UI | Any HTTP (write Python tasks) | Custom flows, quick dashboards, mixed traffic | Python workers can be the bottleneck; streaming/TTFT needs custom code |
| **k6** | **Open-loop** (`constant/ramping-arrival-rate`) or closed (`*-vus`) | RPS, latency percentiles, custom Trends | Any HTTP/gRPC (JS scripts) | Honest open-loop SLA tests, CI gating | Go runtime doesn't tokenize; TTFT needs manual SSE parsing/custom metrics |
| **NVIDIA AIPerf** (successor to GenAI-Perf, 2026) | Both, plus native **session/conversation** replay (`--conversation-num`, `--session-concurrency`) | Everything GenAI-Perf reports, plus per-session turn/context growth stats | OpenAI-compatible, Triton (TRT-LLM, vLLM) | Multi-turn/agentic benchmarking on an NVIDIA-centric stack | Newer tool; flags and docs are still migrating off the GenAI-Perf names, check the migration guide |

Rule of thumb: **`benchmark_serving.py`/GenAI-Perf** when you want LLM-native metrics and datasets out of the box; **k6** when you want a rigorous open-loop SLA test; **LLMPerf** for cross-provider comparisons; **Locust** for bespoke multi-step traffic with a dashboard; **AIPerf** when the workload itself is multi-turn or agentic rather than independent requests.

None of these five rows model output length as anything other than a knob you set — a fixed number, a random range. That was a reasonable simplification when most production traffic really was short-and-bounded. It stopped being reasonable once reasoning models and agent frameworks became common, which is exactly the gap the next section covers.

The table above is the 2023–2024 baseline. Tooling has moved fast since — see [The 2025–2026 Landscape](#the-20252026-landscape) below for how multi-turn/agentic replay and reasoning-model output-length variability changed what "a good load test" even means.

> **Saying it out loud.** My rule of thumb across the tools: reach for `vllm bench serve` when you want LLM-native metrics and realistic datasets out of the box with no glue code. Reach for k6 when you need a rigorous open-loop SLA test, especially as a CI gate, because its arrival-rate executors don't suffer coordinated omission. Use LLMPerf for cross-provider comparisons — same client against OpenAI, Anthropic, and your own endpoint — but treat its numbers as a throughput ceiling, not an SLA, because it's closed-loop by design. Locust for bespoke multi-step flows with a dashboard. And AIPerf when the workload is genuinely multi-turn or agentic. The limitation shared by all of them: none models output length as anything richer than a knob you set, which is precisely where reasoning models break them.

---

## The 2025–2026 Landscape

Everything above is timeless methodology. The tooling that implements it has moved fast, and — more importantly — the *workloads* people benchmark have changed shape: fewer isolated single-turn chat completions, more multi-turn agent sessions and long, unpredictable reasoning traces. A benchmark built for 2023's "one prompt in, one answer out" chatbot systematically misleads you about 2026's agent that calls tools across 40 turns while thinking for 3,000 tokens before answering. This section covers what changed and why it matters for the numbers you report.

> **Saying it out loud.** The methodology in this chapter is timeless; the workloads people benchmark are not. The big shift is that the unit of load moved from *the request* to *the session*. A benchmark built for 2023's one-prompt-in-one-answer-out chatbot systematically misleads you about a 2026 agent that calls tools across forty turns and thinks for three thousand tokens before answering. Two forces drove it: reasoning models, whose output length varies with problem difficulty in ways you can't know in advance, and agentic workloads, where the vast majority of input tokens are reused across turns via prefix caching. The practical consequence is concrete — a single-turn benchmark against an agentic deployment overstates TTFT and understates achievable concurrency, because it never creates the prefix-reuse conditions real traffic does.

### vLLM: from a script to a CLI, plus native multi-turn replay

`benchmark_serving.py` has grown into a first-class subcommand, `vllm bench serve` (the standalone script still exists and both are maintained). Its dataset support has broadened well past `sharegpt`/`random`/`sonnet`: as of the current `vllm/benchmarks/serve.py`, `--dataset-name` accepts `random`, `random-mm` (multi-modal), `random-rerank`, `prefix_repetition` (purpose-built to stress prefix-cache reuse), `sharegpt`, `custom`, `sonnet`, `hf` (any HuggingFace dataset), `spec_bench`, and `timed_trace` (replay requests at recorded timestamps — the building block for realistic session replay). Traffic shaping is also richer than a flat Poisson process: `--burstiness` scales a gamma-distributed arrival process (1.0 = pure Poisson, <1 = burstier, >1 = smoother), and `--ramp-up-strategy {linear,exponential}` with `--ramp-up-start-rps`/`--ramp-up-end-rps` lets a single run sweep from idle to saturation instead of requiring one process per rate — directly automating the sweep this chapter has been doing by hand. (Source: <https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py>; docs: <https://docs.vllm.ai/en/latest/cli/bench/serve/>.)

More significant: vLLM has merged a **dedicated multi-turn benchmark**, `benchmarks/multi_turn/benchmark_serving_multi_turn.py`, tracked under RFC #20265 and shipped via a PR from Pliops (`pliops-daniels`), originally built to benchmark **KV-cache offloading under realistic multi-turn conversation replay** — i.e., does the engine actually reuse a session's prior-turn KV cache instead of recomputing it, and what does that do to TTFT as conversations get longer. (PR: <https://github.com/vllm-project/vllm/pull/20267>; write-up: Pliops, "Setting the Standard: Multi-Turn Benchmarking in vLLM," <https://pliops.com/setting-the-standard-multi-turn-benchmarking-in-vllm/>.) The practical shift: a session is now a first-class unit of load, not an afterthought bolted onto a single-turn tool.

> **Saying it out loud.** vLLM's benchmark grew up. It's a first-class subcommand now, `vllm bench serve`, and the dataset support is much broader — not just ShareGPT and random, but `prefix_repetition` built specifically to stress prefix-cache reuse, `hf` for any Hugging Face dataset, and `timed_trace` to replay requests at their recorded timestamps. Traffic shaping got richer too: `--burstiness` scales a gamma arrival process, where 1.0 is pure Poisson and below 1 is burstier, and `--ramp-up-strategy` lets one run sweep from idle to saturation instead of needing a process per rate — which automates exactly the sweep this chapter does by hand. The bigger deal is the dedicated multi-turn benchmark, built originally to measure whether the engine actually reuses a session's prior-turn KV cache instead of recomputing it.

### NVIDIA GenAI-Perf → AIPerf: sessions and turns as native concepts

NVIDIA's GenAI-Perf (built on the Triton Perf Analyzer) added explicit **multi-turn session modeling**: `--num-sessions` and `--session-concurrency` control how many independent conversations run and how many run in parallel; `--session-turns-mean`/`--session-turns-stddev` control how long each conversation is; `--session-turn-delay-mean`/`--session-turn-delay-stddev` (plus `--session-delay-ratio` to rescale an imported trace) model human "think time" between turns instead of firing the next turn the instant the last one finishes. (Docs: <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/docs/multi_turn.html>.)

NVIDIA has since folded this into a successor tool, **AIPerf** (documented at v0.8.0 as of 2026), which the docs describe as superseding GenAI-Perf, with a migration guide and a "GenAI-Perf vs AIPerf CLI feature comparison matrix" for teams porting scripts. AIPerf's multi-turn flags are renamed but conceptually identical — `--conversation-num`, `--conversation-turn-mean`/`--conversation-turn-stddev`, `--conversation-turn-delay-mean`/`--conversation-turn-delay-stddev` — with the docs explicit that `--request-count` is for single-turn benchmarking and `--conversation-num` is for multi-turn. (Docs: <https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/multi-turn-conversations> and <https://docs.nvidia.com/aiperf/getting-started/ai-perf-comprehensive-llm-benchmarking>.) If you last touched this tool as "GenAI-Perf," expect the name (and some flags) to have moved.

> **Saying it out loud.** NVIDIA's tool made sessions a first-class concept rather than something you bolt on. You control how many independent conversations run and how many run in parallel, how many turns each conversation has as a mean and standard deviation, and — the one people forget — the *delay between turns*, modeling human think time instead of firing the next turn the instant the last one finishes. That delay matters because it changes whether a session's KV cache is still resident when the next turn arrives, which is the whole question you're trying to answer. Practical note: GenAI-Perf has been superseded by AIPerf, and the flags were renamed — `--num-sessions` became `--conversation-num` and so on — so if you last touched this as GenAI-Perf, expect to consult the migration guide.

### LLMPerf: still the cross-provider workhorse, still concurrency-based

LLMPerf (Ray) hasn't fundamentally changed its load model — it remains a **closed-loop, `--num-concurrent-requests`** tool, which is exactly why it's still the default for cross-provider leaderboard-style comparisons (OpenAI vs Anthropic vs a self-hosted vLLM endpoint, apples-to-apples). Its output-length handling is still fixed-`max_tokens`-per-request rather than a distribution sampled from data, which is the main reason it under-represents reasoning-model workloads well: a reasoning model's *actual* output length depends on problem difficulty in a way a flat `max_tokens` target doesn't capture (see below). Treat LLMPerf numbers as "throughput ceiling at concurrency C for this fixed output length," not as an SLA-representative measurement. (<https://github.com/ray-project/llmperf>)

> **Saying it out loud.** LLMPerf hasn't fundamentally changed and that's part of why it's still useful: it's closed-loop on a concurrency knob, which makes it a genuinely apples-to-apples way to compare OpenAI against Anthropic against your own vLLM endpoint with one client. But you have to interpret it correctly. It sets a fixed `max_tokens` per request rather than sampling from a distribution, which under-represents reasoning-model workloads badly, because a reasoning model's actual output length depends on problem difficulty in a way a flat cap simply cannot capture. So read LLMPerf numbers as "throughput ceiling at concurrency C for this fixed output length" — never quote them as an SLA-representative latency measurement.

### Reasoning models broke the "fixed output length" assumption

Chain-of-thought reasoning models (the o-series-style and DeepSeek-R1-style models that think before answering) don't just produce longer outputs — they produce **output lengths that vary with problem difficulty in a way you cannot know in advance**. Li et al.'s 2025 empirical study of reasoning-LLM serving found KV-cache utilization swinging from **3% to 70%** within the same batch (versus a steady sub-3% for standard LLMs), and identified a **straggler-request problem**: a batch containing one hard problem that reasons for thousands of tokens has its entire completion time set by that one request, dragging down every easy request batched alongside it. Their evaluation deliberately moves off idealized fixed-batch benchmarking to a **Gamma-distributed open-arrival workload**, precisely because batch-shaped synthetic tests hide this behavior. (Li et al., "Reasoning Language Model Inference Serving: An Empirical Study," arXiv:2510.18672, 2025: <https://arxiv.org/abs/2510.18672>.)

NVIDIA's 2026 infrastructure-economics analysis of long chain-of-thought models makes the cost implication concrete: because a reasoning model can burn hundreds to thousands of intermediate tokens before its final answer, **cost per token becomes the dominant economic variable**, and disaggregated prefill/decode serving (NVIDIA Dynamo) plus speculative decoding (reported tripling throughput to ~30,000 tok/s on some models) are presented as the mitigations. (NVIDIA Perspectives, "Infrastructure Economics: Reasoning Models & Chain-of-Thought," updated Apr 13, 2026: <https://perspectives.nvidia.com/infrastructure-economics-reasoning-models-chain-of-thought>.)

**What this means for your load test:** stop assuming a single `max_tokens`/mean-output-length number describes your workload. Sample output length from a distribution correlated with a proxy for difficulty (or, at minimum, a heavy-tailed/bimodal distribution — short direct answers mixed with long reasoning traces), track **KV-cache occupancy as a time series** rather than a single steady-state number, and watch batch-level p99 for straggler contamination, not just per-request percentiles. Section B below gives runnable code for this.

> **Saying it out loud.** Chain-of-thought reasoning models don't just produce longer outputs, they produce output lengths that vary with problem difficulty in ways you can't predict — and that breaks a load-testing assumption everyone was quietly relying on. A 2025 empirical study found KV-cache utilization swinging from 3% to 70% *within the same batch*, versus a steady sub-3% for standard models. That produces a straggler-request problem: a batch containing one hard problem that reasons for thousands of tokens has its whole completion time set by that one request, dragging down every easy request beside it. So stop using a single `max_tokens` number. Sample output length from a heavy-tailed or bimodal distribution, track KV-cache occupancy as a time series rather than a steady-state value, and look at batch-level p99 for straggler contamination.

### Agentic and multi-turn workloads: replay sessions, not requests

Two 2026 data points quantify just how different agentic traffic is from single-turn chat. First, a characterization of ReAct-style agents across ADE-Bench, DABStep, GAIA, SWE-bench Pro, and Terminal-Bench 2.0 found execution is **long-tailed in two separate dimensions** — some tasks run to hundreds of turns (up to 786 observed), others accumulate huge context (up to 171K tokens) in relatively few turns — yet despite those huge contexts, time is **decode-dominated (91–98.6% of LLM time)**, because **84.6–99.5% of input tokens are reused across turns** via prefix caching rather than freshly attended-to. Tool calls (file ops, Bash, web search, etc.) consume 2–29% of wall-clock time depending on domain, and failed tool calls trigger retry loops that inflate both turn count and context length. (Yuan, Nayak, Kundu, Talati, "Agentic AI Workload Characteristics," arXiv:2605.26297, May 2026: <https://arxiv.org/abs/2605.26297>.)

Second, vLLM's own May 2026 write-up on integrating Mooncake Store as a distributed KV cache quantifies what happens when you *don't* engineer for this: analyzing 610 real Codex/SWE-bench Pro traces, contexts reach roughly **80K tokens by turn 30** with a **131:1 input-to-output token ratio** (i.e., overwhelmingly prefix-dominated), and when load-balanced round-robin across replicas without a shared cache, the achievable cache-hit rate **collapses to 1.7%** — every re-routed turn recomputes its ~80K-token prefix from scratch. With a distributed cache (Mooncake Store), hit rate recovers to **92.2%**, delivering measured **3.8× higher throughput, 46× lower p50 TTFT, and 8.6× lower end-to-end latency** on a 12-GPU test bed, scaling near-linearly to 60 GPUs while holding >95% hit rate. (vLLM Blog, "Serving Agentic Workloads at Scale with vLLM x Mooncake," May 6, 2026: <https://vllm.ai/blog/2026-05-06-mooncake-store>.)

**What this means for your load test:** a single-turn `benchmark_serving` run against an agentic-serving deployment will systematically **overstate TTFT and understate achievable concurrency**, because it never creates the prefix-reuse conditions real session traffic does — every synthetic request is a cold, independent prefix. If your product is agentic, your load test must (1) replay session-shaped traffic (fixed or sampled turn counts, realistic think-time between turns, growing per-session context) using vLLM's multi-turn benchmark, GenAI-Perf/AIPerf's `--num-sessions`/`--conversation-num` modes, or a `timed_trace` replay of real logs; (2) report **prefix/KV-cache hit rate** as a first-class metric alongside TTFT/TPOT/throughput; and (3) explicitly test cross-instance routing and cache-eviction behavior under concurrent multi-session load, not just single-replica throughput.

> **Saying it out loud.** Agentic traffic looks nothing like chat, and the numbers are striking. Characterizations of ReAct-style agents found tasks running up to 786 turns and contexts up to 171,000 tokens — yet 91 to 98.6% of LLM time is decode, because 84 to 99.5% of input tokens are *reused* across turns via prefix caching rather than freshly attended to. And vLLM's own analysis of 610 real coding-agent traces found contexts hitting roughly 80,000 tokens by turn 30 with a 131-to-1 input-to-output ratio. Here's the punchline: round-robin load balancing across replicas without a shared cache collapses the hit rate to 1.7%, because every re-routed turn recomputes its 80K prefix from scratch. With a distributed cache that recovers to 92%, worth a measured 3.8x throughput and 46x lower p50 TTFT.

### What actually changed, in one paragraph

If you take away one thing from this section: the unit of load has shifted from **the request** to **the session**. Tooling caught up (multi-turn flags in vLLM's benchmark, GenAI-Perf/AIPerf's `--session-*`/`--conversation-*` families), and the workloads driving that shift — reasoning models with difficulty-correlated output length, agents with long-tailed turn counts and heavy prefix reuse — mean that a benchmark reporting a single fixed-length, single-turn number is answering a question production traffic no longer asks. Everything else below is about picking the right tool for the shape of session your product actually generates.

### Choosing a tool in 2026 — decision matrix

| If you need... | Reach for | Because |
|---|---|---|
| LLM-native metrics + realistic single-turn datasets, fast | `vllm bench serve` | Built-in TTFT/TPOT/ITL, `sharegpt`/`hf`/`prefix_repetition` datasets, ramp-up sweeps |
| Multi-turn / KV-cache-offload benchmarking on vLLM specifically | `benchmarks/multi_turn/benchmark_serving_multi_turn.py` | Purpose-built for session replay against vLLM's own KV-cache paths |
| Cross-provider apples-to-apples (OpenAI vs Anthropic vs self-hosted) | LLMPerf | Same client, many backends; treat as throughput-ceiling only |
| NVIDIA/Triton-centric stack, deep session control | AIPerf (formerly GenAI-Perf) | `--conversation-num`/`--session-*` flags, rich exports, active development |
| Rigorous open-loop SLA gate in CI | k6 | True arrival-rate executors, scriptable assertions, no coordinated omission |
| Reasoning-model workload with variable output length | Any of the above, fed a *sampled* (not fixed) output-length distribution | Fixed `max_tokens` hides the straggler-request problem entirely |
| Agentic / tool-calling workload | Session/trace replay (vLLM multi-turn, AIPerf conversations, or your own `timed_trace`) + KV-cache hit-rate metric | Independent-request benchmarks never exercise prefix reuse, so they mismeasure both latency and achievable concurrency |

One caveat applies to every row of that table: this space is moving fast enough that flag names change under you (GenAI-Perf's own docs point at a CLI migration guide for exactly this reason). Whatever tool you pick, pin its version alongside the model, dataset, and length distribution in your CI gate (Production Checklist, item 8) — a benchmark that silently picks up a new default when the tool auto-updates is just as unreproducible as one that was never pinned at all.

> **Saying it out loud.** The short version of choosing a tool: `vllm bench serve` for fast LLM-native single-turn work, vLLM's multi-turn benchmark if you're specifically testing KV-cache offload, LLMPerf for cross-provider comparison as a ceiling measurement only, AIPerf for deep session control on an NVIDIA stack, and k6 when you need a rigorous open-loop gate in CI. For a reasoning-model workload, whichever tool you pick, feed it a *sampled* output-length distribution rather than a fixed cap. For an agentic workload, use session replay and report cache hit rate as a first-class metric. One caveat applies to every row: this space moves fast enough that flag names change under you, so pin the tool version alongside the model and dataset — an auto-updating benchmark is just as unreproducible as an unpinned one.

---

## Failure Modes and Pitfalls

**Closed-loop coordinated omission.** Covered above — the big one. A closed-loop tool stops sending when the server stalls, so it under-reports tail latency and over-reports sustainable load. Fix: use open-loop arrival-rate executors, or a corrected tool that back-dates latency to the intended send time. If you *must* go closed-loop, only use it to measure the throughput ceiling, and never quote its latencies as an SLA.

**No warmup.** The first requests hit cold caches: CUDA graph capture, torch.compile / TRT-LLM engine warmup, cuBLAS autotune, KV-cache allocation, JIT. Cold TTFT can be 5–50× steady-state. Including warmup poisons your percentiles (a handful of huge samples wreck p99). Fix: send a warmup burst and **discard the first N seconds/requests** (the example's `--warmup`). Also warm up the *client* (DNS, TLS, connection pool).

**Unrealistic input/output lengths.** A benchmark with 128-in/128-out tokens tells you nothing about a RAG workload with 4000-in/500-out. Prefill cost scales with input length; decode cost and E2E scale with output length; KV-cache pressure scales with both × concurrency. **Fixed lengths also hide batching dynamics** because every request finishes together. Fix: replay a **realistic length distribution** (ShareGPT, your own production logs, or `--dataset-name random` with a mean/std matching production). Report the distribution you used.

**Measuring only averages.** The mean hides the tail and can be dominated by a few slow requests. Always report p50/p95/p99 (and max) for TTFT, TPOT, and E2E separately. And never average latency *across different output lengths* without normalizing — longer outputs inflate E2E and masquerade as a regression.

**Client-side bottleneck.** The most insidious: your load generator, not the server, is the limit. Symptoms: achieved rate plateaus well below the server's known capacity, client CPU pegged at 100%, or latency that scales with *client* concurrency. Causes: single-threaded Python parsing SSE, aiohttp/`requests` connection-pool caps, GIL contention, running the client on the same box as the server, or a network link between client and server that's the actual bottleneck. Fixes: pin and monitor client CPU, remove connection caps (`TCPConnector(limit=0)`), use async or distributed workers (k6/Locust workers, multiple client hosts), and **sanity-check with Little's Law** — if measured \\( L = \lambda W \\) can't reach the server's `max-num-seqs`, your client is starving it.

**Reusing identical prompts / prefix caching artifacts.** If every request sends the same prompt, prefix caching makes prefill nearly free and TTFT looks unrealistically good. Vary prompts (or explicitly test both with- and without-cache) so you measure the case you'll actually run.

**Not holding steady state / too-short runs.** Little's Law and stable percentiles need steady state. A 5-second run at 30 req/s is ~150 samples — too few for a trustworthy p99 (you want ≥1000+ post-warmup samples) and too short to reach queue equilibrium. Run each sweep point long enough that metrics stop drifting.

> **Saying it out loud.** Seven ways benchmarks lie, and I'd name them in this order. Coordinated omission from a closed-loop tool — the big one. No warmup, so cold-cache samples with five-to-fifty-times TTFT poison your p99. Unrealistic input and output lengths, since 128-in-128-out tells you nothing about a 4000-in-500-out RAG workload. Measuring only averages. A client-side bottleneck, where your own load generator is the ceiling — the tell is a plateau while server GPU utilization sits at 40%. Reusing identical prompts, so prefix caching makes prefill nearly free and TTFT looks unrealistically good. And runs too short to reach steady state — you want at least a thousand post-warmup samples before you trust a p99. The unifying habit: cross-check achieved rate, server utilization, and Little's-Law concurrency against each other.

---

## Build It in Practice — Extended

The async client earlier in this chapter is a correct open-loop tool for *one* rate. Three things turn it into a practice you can actually run before a launch: (1) a length sampler that replays a realistic input/output distribution instead of a handful of fixed prompts, (2) a sweep driver that finds the saturation knee automatically instead of you eyeballing a table, and (3) a plotting step that turns the sweep into a chart you can put in a review deck.

> **Saying it out loud.** Three things turn a correct one-rate client into something you'd actually run before a launch. A length sampler that replays a real input/output distribution instead of a couple of fixed prompts, because fixed lengths mean every request finishes together and you never see straggler effects or realistic KV-cache pressure. A sweep driver that finds the saturation knee automatically instead of you eyeballing a table, so it can run before every release rather than once a quarter. And a plotting step, because the artifact that actually changes a decision in a design review is a chart with the knee marked on it, not a CSV. That's the difference between load testing as a one-off exercise and load testing as a gate.

### B1. A realistic input/output length sampler

Fixed-length benchmarks hide exactly the dynamics that matter (see Pitfalls, above): requests that finish together, no straggler effects, no realistic KV-cache pressure. The right fix is to sample lengths from real data rather than invent a mean and stddev. The cheapest way to get "real data" without a bespoke dataset is to reuse the ShareGPT-style conversation data that `vllm bench serve --dataset-name sharegpt` already knows how to load, extract the **empirical (input_len, output_len) pairs after tokenization**, cache them once, and bootstrap-sample from that empirical pool for every request — this preserves the real joint distribution (including its correlation and heavy tail) instead of fitting a parametric approximation that smooths it away.

```python
#!/usr/bin/env python3
"""length_sampler.py — build and sample from an empirical length distribution.

Run once to build the pool (needs a tokenizer and a ShareGPT-format JSON,
e.g. the file vLLM's own benchmarks download):

    python length_sampler.py build \
        --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
        --tokenizer meta-llama/Llama-3.1-8B-Instruct \
        --out lengths.json

Then import build_pool()/sample() from a load-test script to draw
(input_len, output_len) pairs that match a real production-like shape.
"""
import argparse, json, random

def build_pool(dataset_path: str, tokenizer_name: str, limit: int = 20000):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    with open(dataset_path) as f:
        convos = json.load(f)
    pool = []
    for c in convos:
        turns = c.get("conversations", [])
        # Pair each human turn with the assistant turn that follows it.
        for i in range(len(turns) - 1):
            if turns[i]["from"] == "human" and turns[i + 1]["from"] == "gpt":
                in_len = len(tok(turns[i]["value"]).input_ids)
                out_len = len(tok(turns[i + 1]["value"]).input_ids)
                if 1 <= in_len <= 8192 and 1 <= out_len <= 4096:
                    pool.append((in_len, out_len))
        if len(pool) >= limit:
            break
    return pool

def sample(pool, reasoning_mix: float = 0.0, rng: random.Random | None = None):
    """Draw one (input_len, output_len) pair.

    reasoning_mix in [0, 1]: fraction of requests that get a synthetic
    'hard reasoning' output-length multiplier instead of the empirical
    output length, reflecting the finding that reasoning models produce
    output length correlated with problem difficulty rather than a fixed
    ceiling (Li et al., arXiv:2510.18672). 0.0 reproduces the raw empirical
    distribution; try 0.15-0.3 to approximate a mixed reasoning workload.
    """
    rng = rng or random
    in_len, out_len = rng.choice(pool)
    if reasoning_mix > 0 and rng.random() < reasoning_mix:
        # Hard problem: reasoning chain multiplies output length heavily,
        # with its own long tail (occasionally 10-20x).
        out_len = int(out_len * rng.lognormvariate(1.6, 0.7))
        out_len = min(out_len, 8192)
    return in_len, out_len

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dataset", required=True)
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--limit", type=int, default=20000)
    args = ap.parse_args()
    if args.cmd == "build":
        pool = build_pool(args.dataset, args.tokenizer, args.limit)
        with open(args.out, "w") as f:
            json.dump(pool, f)
        lens_in = sorted(p[0] for p in pool)
        lens_out = sorted(p[1] for p in pool)
        n = len(pool)
        print(f"built pool of {n} pairs")
        print(f"input_len  p50={lens_in[n//2]}  p95={lens_in[int(n*0.95)]}")
        print(f"output_len p50={lens_out[n//2]}  p95={lens_out[int(n*0.95)]}")
```

To wire this into the earlier `load_test.py`, replace the fixed `PROMPTS` list and `--max-tokens` flag with a call to `sample(pool, reasoning_mix=args.reasoning_mix)` per request: pad/truncate a filler prompt to `in_len` tokens (or, better, reuse the ShareGPT prompt text itself alongside its measured length) and set that request's `max_tokens` to the sampled `out_len` instead of a global constant. The rest of `one_request` — scheduling from `sched_t`, measuring from the intended send time — is unchanged; only the per-request length now varies realistically instead of being global.

Why bootstrap from an empirical pool instead of fitting a parametric distribution (lognormal, gamma) to the same data? A fitted distribution smooths over exactly the structure that matters here — the correlation between a conversation's input and output length, and the heavy tail of occasional very long turns. Sampling `(in_len, out_len)` pairs together, verbatim, from real conversations preserves both; fitting marginal distributions separately and sampling them independently would silently discard the correlation and could understate how often a long input and a long output co-occur, which is exactly the combination that stresses KV-cache capacity hardest.

> **Saying it out loud.** The right way to get realistic lengths isn't to invent a mean and standard deviation — it's to take real conversation data, tokenize it, extract the empirical input-length and output-length *pairs*, and bootstrap-sample from that pool for every request. The reason to keep them as pairs matters: it preserves the real joint distribution, including the correlation between prompt length and answer length and the heavy tail, both of which a fitted parametric distribution smooths right out. And the heavy tail is precisely the thing that produces straggler requests and realistic KV-cache pressure. Practical detail: tokenize once, cache the pool to disk, and record which tokenizer you used — length distributions aren't portable across tokenizers, so an unlabeled pool is an unreproducible benchmark.

### B2. A concurrency sweep that finds the knee automatically

Manually reading a table (as in the sample-results section above) works for a chapter; it doesn't scale to "rerun this before every release." The sweep below is deliberately **closed-loop** — fixed worker pools per level — because finding the *ceiling* is precisely the closed-loop use case this chapter argues for: a bounded, known concurrency is the independent variable, and you read off achieved throughput and p95 latency at each level. It doubles concurrency until throughput growth falls below a threshold, then binary-searches between the last two doublings to pinpoint the knee more tightly.

```python
#!/usr/bin/env python3
"""sweep.py — closed-loop concurrency sweep with automatic knee detection.

Usage:
  python sweep.py --url http://localhost:8000/v1/chat/completions \
      --model my-model --pool lengths.json --out sweep_results.csv
"""
import argparse, asyncio, csv, json, statistics, time
import aiohttp
from length_sampler import sample

async def worker_loop(session, url, model, pool, stop_event, out_tokens_box, latencies):
    """Closed-loop worker: send, await full response, immediately send next."""
    while not stop_event.is_set():
        in_len, out_len = sample(pool)
        payload = {
            "model": model,
            "messages": [{"role": "user",
                          "content": "Explain this topic in detail. " * max(1, in_len // 8)}],
            "max_tokens": out_len, "stream": True, "temperature": 0.0,
        }
        t_send = time.perf_counter()
        n = 0
        try:
            async with session.post(url, json=payload) as resp:
                async for raw in resp.content:
                    line = raw.decode("utf-8").strip()
                    if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                        n += 1
        except Exception:
            continue
        if stop_event.is_set():
            break                     # don't count a request that straddles the boundary
        latencies.append((time.perf_counter() - t_send) * 1000)
        out_tokens_box[0] += n

async def run_level(url, model, pool, concurrency, measure_s, warmup_s):
    stop = asyncio.Event()
    out_tokens_box, latencies = [0], []
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn,
                                      timeout=aiohttp.ClientTimeout(total=None)) as session:
        tasks = [asyncio.create_task(worker_loop(session, url, model, pool, stop,
                                                  out_tokens_box, latencies))
                 for _ in range(concurrency)]
        await asyncio.sleep(warmup_s)
        latencies.clear(); out_tokens_box[0] = 0     # discard warmup
        t0 = time.perf_counter()
        await asyncio.sleep(measure_s)
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        wall = time.perf_counter() - t0
    n = len(latencies)
    p95 = sorted(latencies)[int(0.95 * (n - 1))] if n else float("nan")
    return {
        "concurrency": concurrency,
        "req_per_s": n / wall if wall else 0.0,
        "out_tok_per_s": out_tokens_box[0] / wall if wall else 0.0,
        "e2e_p95_ms": p95,
        "n_requests": n,
    }

def efficiency(level):
    """Throughput per unit of concurrency — falls as you approach the knee."""
    return level["out_tok_per_s"] / level["concurrency"]

async def sweep(url, model, pool, measure_s, warmup_s, max_concurrency=256):
    levels = []
    c = 1
    while c <= max_concurrency:
        lvl = await run_level(url, model, pool, c, measure_s, warmup_s)
        levels.append(lvl)
        print(f"concurrency={c:4d}  req/s={lvl['req_per_s']:7.2f}  "
              f"tok/s={lvl['out_tok_per_s']:8.1f}  p95={lvl['e2e_p95_ms']:8.1f}ms")
        if len(levels) >= 2:
            growth = lvl["out_tok_per_s"] / max(levels[-2]["out_tok_per_s"], 1e-9) - 1
            latency_blowup = lvl["e2e_p95_ms"] / max(levels[-2]["e2e_p95_ms"], 1e-9)
            # Stop doubling once a doubling of concurrency buys <10% more
            # throughput, or p95 latency has more than tripled since the
            # last level -- either is the overload signature from the
            # Core Intuition section above.
            if growth < 0.10 or latency_blowup > 3.0:
                break
        c *= 2
    if len(levels) < 2:
        return levels, levels[-1]["concurrency"] if levels else 1
    # Binary-search refine between the last two doubling points for a
    # tighter knee estimate than "somewhere between C and 2C".
    lo, hi = levels[-2]["concurrency"], levels[-1]["concurrency"]
    lo_lvl = levels[-2]
    while hi - lo > max(1, lo // 8):          # stop refining once step is <~12% of lo
        mid = (lo + hi) // 2
        mid_lvl = await run_level(url, model, pool, mid, measure_s, warmup_s)
        levels.append(mid_lvl)
        print(f"  refine concurrency={mid:4d}  tok/s={mid_lvl['out_tok_per_s']:8.1f}  "
              f"p95={mid_lvl['e2e_p95_ms']:8.1f}ms")
        growth = mid_lvl["out_tok_per_s"] / max(lo_lvl["out_tok_per_s"], 1e-9) - 1
        if growth < 0.10:
            hi = mid
        else:
            lo, lo_lvl = mid, mid_lvl
    knee = lo
    return sorted(levels, key=lambda l: l["concurrency"]), knee

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True, help="lengths.json from length_sampler.py build")
    ap.add_argument("--out", default="sweep_results.csv")
    ap.add_argument("--measure-seconds", type=float, default=30.0)
    ap.add_argument("--warmup-seconds", type=float, default=10.0)
    ap.add_argument("--max-concurrency", type=int, default=256)
    args = ap.parse_args()
    with open(args.pool) as f:
        pool = json.load(f)
    levels, knee = await sweep(args.url, args.model, pool, args.measure_seconds,
                                args.warmup_seconds, args.max_concurrency)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(levels[0].keys()))
        w.writeheader()
        for lvl in levels:
            w.writerow(lvl)
    print(f"\ndetected knee at concurrency ~= {knee}  (wrote {args.out})")

if __name__ == "__main__":
    asyncio.run(main())
```

The knee rule implements exactly the "throughput stops rising while latency starts rising superlinearly" signature from the Core Intuition section: growth-per-doubling `< 10%` catches the throughput flattening, `latency_blowup > 3.0` catches the queueing explosion, and either one alone is enough to trigger — some engines saturate on compute (throughput flattens first) while others saturate on scheduler/KV-cache pressure (latency explodes while throughput is still creeping up). The binary-search phase afterward turns "somewhere between 32 and 64" into a specific number worth writing on a slide.

> **Saying it out loud.** The sweep is deliberately closed-loop, and that's not a contradiction of everything earlier — finding the *ceiling* is exactly the closed-loop use case, because a bounded known concurrency is the independent variable you want to vary. The algorithm is simple: double concurrency until throughput growth falls below a threshold, then binary-search between the last two doublings to pin down the knee more tightly. Doubling gets you into the right neighborhood in a handful of runs instead of a linear scan, and the binary search buys precision where it matters. The output is a CSV with achieved rate, output tokens per second, and latency percentiles at each level — and the rule stays: use it for the ceiling, then confirm the SLA open-loop at your chosen rate.

### B3. Plotting the sweep

```python
#!/usr/bin/env python3
"""plot_results.py — render sweep_results.csv as a throughput/latency chart.

Usage: python plot_results.py sweep_results.csv --knee 40 --out sweep.png
"""
import argparse, csv

def load(path):
    with open(path) as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    rows = [r for r in rows if r["n_requests"] and int(r["n_requests"]) > 0]
    rows.sort(key=lambda r: int(r["concurrency"]))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--knee", type=float, default=None)
    ap.add_argument("--out", default="sweep.png")
    args = ap.parse_args()
    rows = load(args.csv_path)
    xs = [int(r["concurrency"]) for r in rows]
    tok_s = [float(r["out_tok_per_s"]) for r in rows]
    p95 = [float(r["e2e_p95_ms"]) for r in rows]

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # No plotting deps available -- fall back to an ASCII sparkline so
        # the pipeline still produces *something* reviewable over SSH.
        def spark(vals):
            bars = " .:-=+*#%@"
            lo, hi = min(vals), max(vals)
            span = (hi - lo) or 1.0
            return "".join(bars[min(9, int((v - lo) / span * 9))] for v in vals)
        print("matplotlib not installed; ASCII fallback:")
        print("concurrency:", xs)
        print("tok/s      :", spark(tok_s), f"  (min={min(tok_s):.0f}, max={max(tok_s):.0f})")
        print("p95 ms     :", spark(p95), f"  (min={min(p95):.0f}, max={max(p95):.0f})")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(xs, tok_s, marker="o")
    ax1.set_xlabel("concurrency"); ax1.set_ylabel("output tok/s")
    ax1.set_title("Throughput vs concurrency")
    ax2.plot(xs, p95, marker="o", color="firebrick")
    ax2.set_xlabel("concurrency"); ax2.set_ylabel("E2E p95 (ms)")
    ax2.set_title("Tail latency vs concurrency")
    if args.knee is not None:
        for ax in (ax1, ax2):
            ax.axvline(args.knee, linestyle="--", color="gray")
            ax.text(args.knee, ax.get_ylim()[1] * 0.95, f" knee~{args.knee:g}",
                    va="top", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
```

The `matplotlib`-missing fallback matters more than it looks: this pipeline is meant to run unattended (a pre-release CI job, a bastion box with no X11 and a minimal Python image), and a plotting script that hard-crashes when a dependency is missing means nobody ever sees the sweep at all. Chained together (`length_sampler.py build` → `sweep.py` → `plot_results.py`), these three scripts are the difference between "we eyeballed some numbers" and "here is the knee, here is the chart, here is the exact concurrency we're launching at" in a launch-readiness review.

---

## Production Case Studies & War Stories

### War story 1 — the load test that lied

**Setup.** A team preparing to launch a chat feature ran a pre-launch load test with 200 fixed virtual users hammering a vLLM endpoint in a tight closed loop: send, wait for the full response, send again. The report looked clean — median E2E 1.2 s, p99 2.1 s — and the launch was signed off as "validated at 500 req/s equivalent load."

**What actually happened.** On launch day, real open-loop traffic arrived on its own schedule, indifferent to server health. An autoscaler scale-up lagged behind a traffic ramp by about ninety seconds, during which the existing pods queued hard. In production, p99 spiked past 40 seconds and client timeouts triggered retries, which added *more* load on top of an already-saturated system — a classic retry storm. Nothing about this was visible in the pre-launch numbers.

**Root cause.** The pre-launch test was closed-loop. When the (simulated, in staging) system slowed down, each of the 200 virtual users simply stalled waiting for its in-flight response and stopped sending new requests — the offered load quietly dropped exactly when the system was struggling, and no sample ever recorded the 90-second queueing debt building up behind the stall. This is coordinated omission exactly as described earlier in this chapter, and it is not a hypothetical: it is the single most common reason a "green" load test is followed by a "red" launch.

**Diagnosis and fix.** Post-incident, the team reran the same nominal load using k6's `ramping-arrival-rate` executor (open-loop) and reproduced the p99 spike in staging by chaos-injecting an artificial 90-second slow window on one backend pod. The open-loop test showed the queueing debt immediately — p99 blew out precisely as it had in production, while a control run of the *old* closed-loop VU-based script, replayed against the same chaos-injected staging environment, again failed to show it. That side-by-side rerun is what convinced the team the tooling, not the server, had been the blind spot.

**Lesson.** Any load test used as a launch gate for latency SLAs must be open-loop with a realistic arrival distribution, and it must include at least one injected-failure scenario (a slow pod, a delayed autoscale event), not just steady-state load. A closed-loop VU test is still useful — as a throughput-ceiling probe — but it must never be the artifact that says "go."

> **Saying it out loud.** A team ran a pre-launch test with 200 fixed virtual users hammering vLLM in a tight closed loop. Median 1.2 seconds, p99 2.1 — signed off as validated. On launch day, real open-loop traffic arrived on its own schedule, an autoscaler lagged a ramp by about ninety seconds, p99 spiked past 40 seconds, client timeouts fired retries, and the retries added load to an already-saturated system. Root cause: when the system slowed in staging, all 200 virtual users simply stalled and stopped sending — the offered load quietly dropped exactly when it mattered, and no sample ever recorded the queueing debt. They reproduced it by rerunning open-loop with an injected slow pod. The lesson: any load test used as a launch gate for latency must be open-loop and must include an injected-failure scenario, not just steady state.

### War story 2 — the GPU that wasn't the bottleneck

**Setup.** An engineer benchmarking a new vLLM engine configuration (a scheduler flag change intended to improve throughput) measured a plateau at roughly 1,800 output tok/s at 32 concurrent requests on an A100 running an 8B model — well under the several-thousand tok/s the same card had hit in a vendor reference benchmark for a comparable model. The natural conclusion: the new configuration regressed something, and the engineer began reverting flags one at a time, burning the better part of a day.

**What was actually happening.** `nvidia-smi` showed GPU utilization at roughly 40% throughout the test — not a saturated GPU at all. The load generator was a single Python process using `aiohttp` with its **default connection-pool limit of 100** and synchronous JSON parsing of each SSE chunk on the event loop; at 32 concurrent streaming connections doing that parsing, one CPU core was pegged at 100%. The client, not the server, had run out of capacity.

**How it was confirmed.** Cross-checking with Little's Law settled it without more guessing: the achieved request rate and measured mean E2E latency implied an average in-flight concurrency \\( L = \lambda W \\) far below the server's configured `--max-num-seqs`. If the server had genuinely been the bottleneck at 32 concurrent requests, \\( L \\) should have tracked close to 32; instead it sat well under that, meaning the server had spare scheduling slots the client simply wasn't filling.

**Fix.** Removing the `aiohttp` connector limit (`TCPConnector(limit=0)`) and splitting the load across four client processes (one per CPU core) with `k6`'s distributed workers eliminated the client bottleneck. Rerun under the *same* engine configuration reached roughly 6,100 output tok/s — in line with the vendor reference, and confirming the scheduler-flag change the engineer had been about to revert was fine all along.

**Lesson.** Before attributing a throughput plateau to the server, check client CPU and connection-pool limits, and cross-validate the achieved in-flight concurrency against Little's Law and the server's own `max-num-seqs`/`max-num-batched-tokens` settings. The load generator is part of the system under test, and it is very easy for it to quietly become the ceiling.

> **Saying it out loud.** An engineer saw throughput plateau at 1,800 output tokens per second at 32 concurrent requests on an A100 — well under a vendor reference — and started reverting scheduler flags one at a time, losing most of a day. The actual tell was sitting right there: `nvidia-smi` showed the GPU at about 40% utilization. The load generator was a single Python process with aiohttp's default 100-connection pool limit, parsing every SSE chunk synchronously on the event loop, and one CPU core was pegged. Little's Law confirmed it — the implied in-flight concurrency was far below the server's `max-num-seqs`, meaning the server had free scheduling slots the client wasn't filling. Removing the connector limit and splitting across four processes got 6,100 tokens per second on the *same* config. The load generator is part of the system under test.

### Telling these apart quickly, without waiting for a postmortem

Both stories above are diagnosable in minutes if you check the right signal early, rather than staring at a latency graph and guessing:

| Symptom you observe | Likely cause | Quick check |
|---|---|---|
| Achieved rate tracks target rate, but production still times out under real traffic | Coordinated omission — the test was closed-loop | Rerun the same nominal load open-loop (arrival-rate executor); if the tail latency only appears there, that's the confirmation |
| Throughput plateaus well below expectation, server GPU utilization is low (well under ~80%) | Client-side bottleneck | Check client CPU, connector/pool limits, and Little's-Law-implied concurrency against `max-num-seqs` |
| Throughput plateaus, server GPU utilization is high (~90%+), latency climbing superlinearly | Genuine server-side saturation — you've found the real knee | This is the good outcome: sweep further to confirm the plateau holds, then pick an operating point below it |
| Achieved rate silently falls below target rate as you push higher | Either genuine overload (expected past the knee) or a client that can't keep up with its own schedule | Compare against a known-good baseline concurrency/rate; if the shortfall appears *before* the expected knee, suspect the client first |

The unifying habit in both incidents: never trust a single graph in isolation. Cross-check achieved-vs-target rate, server-side utilization, and the Little's-Law-implied concurrency against each other before deciding which side of the client/server boundary the bottleneck is on.

> **Saying it out loud.** You can separate these in minutes with one question: what is the server's GPU utilization at the plateau? If throughput plateaus and utilization is *low*, well under 80%, it's a client bottleneck — check client CPU, connection pool limits, and Little's-Law-implied concurrency against `max-num-seqs`. If throughput plateaus and utilization is *high*, around 90-plus, with latency climbing superlinearly, that's genuine server saturation and you've found the real knee, which is the good outcome. And if achieved rate tracks target beautifully in test but production still times out, that's coordinated omission — rerun the identical nominal load open-loop and see whether the tail appears. The habit underneath all three: never trust a single graph in isolation.

---

## Production Checklist — What an Interviewer Probes

1. **"What do you measure, and why not just average latency?"** — Name TTFT, TPOT/ITL, E2E, request- and output-token throughput; report **p50/p95/p99**, not means, because latency is heavy-tailed and the tail is what users feel. Mean is only for throughput accounting.
2. **"Open-loop or closed-loop, and what's coordinated omission?"** — Open-loop (arrival-rate) for SLA validation; closed-loop (fixed concurrency) only to find the throughput ceiling. Coordinated omission = closed-loop stops sending when the server stalls, so it hides the tail. Bonus: measure latency from the *intended* send time.
3. **"How do you find the right operating point?"** — Sweep concurrency/rate, plot throughput and p95 latency vs load, find the **knee** where throughput flattens and latency turns up; operate just below it with headroom.
4. **"Apply Little's Law here."** — \\( L = \lambda W \\). Given a rate and E2E latency, compute in-flight concurrency and compare to `max-num-seqs` to detect oversubscription; convert closed-loop concurrency to sustainable req/s.
5. **"How do you make the workload realistic?"** — Match production input/output length distributions (ShareGPT / replayed logs), vary prompts to avoid prefix-cache flattery, and report the distribution used.
6. **"How do you know the client isn't the bottleneck?"** — Monitor client CPU, remove connection caps, use async/distributed generators, separate client and server hosts, and cross-check achieved \\( L \\) against server capacity.
7. **"Warmup and steady state?"** — Discard cold-start requests (CUDA graphs/compile/cache allocation); run long enough (≥~1000 post-warmup samples, metrics stable) for trustworthy p99.
8. **"How do you gate regressions in CI?"** — A reproducible benchmark (pinned tool/model/dataset/lengths) run at a fixed rate, asserting on p95 TTFT and output tok/s thresholds, so a config regression fails the build.

---

## Interview Mastery

The checklist above covers the essentials. This section drills deeper: a tight verbal answer to the question interviewers open with, a system-design prompt with a worked sketch, a red-flags/green-flags table for spotting bad methodology fast, and an expanded Q&A bank.

### "Explain open-loop vs closed-loop load testing in 60 seconds"

> A load test's *load model* is either closed-loop or open-loop. Closed-loop means a fixed pool of virtual users, each looping: send, wait for the response, send again — concurrency is capped at the pool size by construction. Open-loop means requests arrive on a schedule — say, Poisson at a target rate — independent of whether earlier requests have finished; concurrency is whatever emerges from that arrival rate hitting server capacity. The difference matters because of coordinated omission: when a closed-loop system slows down, each virtual user stalls waiting for its response and simply stops sending — the offered load backs off exactly when the server is struggling, so the tail latency you'd see in a real, indifferent arrival process never gets sampled. That's why closed-loop is fine for finding a throughput *ceiling* — a fixed, known concurrency is exactly the variable you want for that — but open-loop, with latency measured from the *intended* send time, is the only honest way to validate a latency SLA against realistic traffic.

### System design prompt: "Design a load-testing plan before a product launch"

A plausible interview prompt: *"We're launching a customer-facing chat feature backed by a self-hosted 70B model on 8 GPUs in three weeks. Design the load-testing plan."* A strong answer moves through phases, each with a concrete deliverable:

```
Phase 0 — Sanity check (day 1)
  single request, cold and warm -> confirm TTFT/TPOT look reasonable,
  no obvious misconfiguration (wrong dtype, tiny max-num-seqs, etc.)

Phase 1 — Concurrency sweep (days 2-4)
  closed-loop sweep (Section B2 above) with a *realistic* length sampler
  (Section B1), doubling until the knee triggers, then bisecting
  -> deliverable: throughput-vs-concurrency and p95-vs-concurrency chart,
     a single number ("knee ~= 45 concurrent requests, ~9,000 tok/s")

Phase 2 — Open-loop SLA validation (days 5-8)
  k6 (or the async client above) at the *target* production arrival rate
  and traffic mix, run long enough for >=1000 post-warmup samples per
  metric, with prompts/lengths sampled from real or representative data
  -> deliverable: pass/fail against the stated SLA (e.g. "p95 TTFT < 500ms
     at 25 req/s"), with margin to the Phase 1 knee stated explicitly

Phase 3 — Failure-mode / chaos injection (days 9-11)
  re-run Phase 2 while injecting: a slow/unhealthy pod, an autoscaler
  delay, a burst 3x above target rate for 60s, a cold-start (freshly
  scaled pod with no warmup)
  -> deliverable: a short runbook of "what breaks and how it degrades"
     (graceful backpressure vs cascading timeout/retry storm)

Phase 4 — Soak test (days 12-13)
  hold the Phase 2 rate for several hours -> catch slow memory/KV-cache
  fragmentation leaks that a 10-minute test can't show

Phase 5 — CI regression gate (ongoing, from day 14)
  pin tool/model/dataset/lengths; run the Phase 1 sweep (or a cheap
  single-concurrency proxy of it) on every config change; assert on
  p95 TTFT and output tok/s thresholds so a scheduler-flag regression
  fails the build before it reaches production
```

The parts an interviewer is listening for: naming the knee-finding phase as *closed-loop by design*, explicitly separating it from the *open-loop* SLA-validation phase, including a chaos/failure phase (most candidates forget this and only test steady state), and ending with something that survives past launch day — a CI gate, not a one-time report.

> **Saying it out loud.** For a pre-launch plan I'd sequence it in four stages. First, characterize the workload from real or expected data — input and output length distributions, whether traffic is single-turn or session-shaped, and what the actual SLO is, stated as a percentile at a stated load. Second, run a closed-loop concurrency sweep to find the saturation knee and the true throughput ceiling per replica, which gives you your capacity math. Third, run an open-loop test at your target arrival rate to validate the SLA with honest tail behavior — and include at least one injected failure, a slow pod or a delayed scale-up, because that's where the retry storms come from. Fourth, pin the whole thing — tool version, model, dataset, length distribution — into CI as a regression gate. The signal being graded is whether you use each load model for the question it can actually answer.

### Red flags vs green flags

| Signal | Red flag | Green flag |
|---|---|---|
| Latency reporting | Only a mean latency number | p50/p95/p99 (and max) per metric, reported separately for TTFT/TPOT/E2E |
| Load model for an SLA gate | Fixed-VU closed-loop test used to certify a latency SLA | Open-loop, arrival-rate test used for the SLA; closed-loop reserved for finding the throughput ceiling |
| Operating point | A single concurrency number presented as "the" benchmark | A concurrency/rate sweep with the knee explicitly identified and an operating point chosen with headroom below it |
| Workload shape | Same short prompt repeated, fixed `max_tokens` | Lengths sampled from a real/representative distribution; reasoning/agentic mix modeled if relevant |
| Warmup | Cold-start requests included in the reported percentiles | Warmup window explicitly discarded; client itself warmed up (DNS/TLS/pool) |
| Client health | Client CPU/connection limits never checked | Client CPU monitored, connector caps removed, achieved concurrency cross-checked against Little's Law |
| Reproducibility | "We ran it once and it looked fine" | Tool/model/dataset/lengths pinned and rerun in CI with pass/fail thresholds |
| Reasoning-model workloads | Fixed `max_tokens` per request | Output length sampled from a distribution correlated with difficulty; KV-cache occupancy tracked over time, not just at steady state |
| Agentic/multi-turn workloads | Independent single-turn requests only | Session/multi-turn replay with realistic think-time between turns; prefix/KV-cache hit rate reported as a first-class metric |
| Failure handling | Only steady-state load tested | At least one chaos/failure-injection scenario (slow pod, autoscale delay, burst) included before launch |

### Expanded Q&A

1. **What is TTFT dominated by, and why is it the most load-sensitive metric?** Prefill cost plus queueing wait for a scheduler slot; queueing wait is paid entirely before the first token, so TTFT reacts to load faster and harder than TPOT does.
2. **Why report output-token throughput separately from request throughput?** They answer different questions — req/s is the right top-line number when the unit of work is "a request" (e.g. classification); output tok/s is the money metric for generative workloads and is what improves when batching gets better, even if req/s stays flat because outputs got longer.
3. **State Little's Law and use it to sanity-check a benchmark.** \\( L = \lambda W \\). If a test offers 20 req/s and measures 4 s mean E2E, in-flight concurrency is 80; if `max-num-seqs` is 64, the system is oversubscribed and not in steady state, which the latency graph alone won't tell you until it's already climbing.
4. **What, concretely, is coordinated omission, and name one tool feature that fixes it.** A closed-loop load generator stops sending when the server slows down, because each virtual user is blocked waiting on its own response, so the tail latency a real, indifferent arrival stream would have produced never gets sampled. Fix: an open-loop arrival-rate executor (k6's `constant-arrival-rate`) that measures latency from the *scheduled* send time rather than the actual send time.
5. **Walk through how you'd automatically find the concurrency knee without eyeballing a chart.** Double concurrency each step; stop when a doubling buys less than ~10% more output tok/s *or* p95 latency more than triples versus the previous level (either condition alone is the overload signature); then binary-search between the last two doubling points to tighten the estimate.
6. **Your pre-launch load test showed a healthy p99, but production had multi-second stalls. What's your hypothesis, and how do you confirm it?** Hypothesis: the pre-launch test was closed-loop and coordinated omission hid a queueing-debt scenario (e.g. an autoscaler lag). Confirm by rerunning the same nominal load open-loop while chaos-injecting the suspected failure (a delayed/slow pod) in staging, and showing the closed-loop tool fails to reproduce the same spike side-by-side.
7. **You suspect your load generator, not the server, is the bottleneck. What do you check, in order?** Client CPU utilization; connection-pool/connector limits (e.g. aiohttp's default 100); whether the client is single-threaded/GIL-bound doing synchronous parsing on the event loop; then cross-check achieved in-flight concurrency via Little's Law against the server's configured capacity (`max-num-seqs`) — if the implied \\( L \\) is well under server capacity, the client is starving it.
8. **How do you make a synthetic benchmark's workload realistic?** Sample input/output lengths from real data (production logs, or a public conversational dataset like ShareGPT) rather than a fixed pair; vary prompt content so prefix caching doesn't flatter TTFT unless you're deliberately testing the cached case; and match the arrival process (Poisson vs bursty vs session-shaped) to how traffic actually behaves.
9. **How does a reasoning model change what and how you benchmark?** Output length becomes correlated with problem difficulty rather than a fixed ceiling, KV-cache utilization swings far more than in standard LLM serving (documented ranges of roughly 3-70% within a batch), and a single hard request in a batch can become a straggler that drags down every other request's completion time. Practical fix: sample output length from a distribution (ideally bimodal/heavy-tailed) instead of a flat `max_tokens`, and track KV-cache occupancy as a time series rather than a single number.
10. **How would you load-test an agentic, multi-turn system differently from a single-turn chatbot?** Replay session-shaped traffic — realistic turn counts, think-time between turns, growing per-session context — rather than independent single-turn requests, using a tool with native multi-turn support (vLLM's multi-turn benchmark, GenAI-Perf/AIPerf's session/conversation modes, or a timestamped trace replay); and report prefix/KV-cache hit rate as a first-class metric, because agentic sessions are typically decode-dominated with very high input-token reuse across turns, and an independent-request benchmark never creates that reuse condition.
11. **Compare vLLM's `benchmark_serving`/`vllm bench serve`, NVIDIA GenAI-Perf/AIPerf, and LLMPerf — when do you reach for each?** `vllm bench serve` for fast, LLM-native metrics and datasets against a vLLM-family or OpenAI-compatible endpoint, including built-in ramp-up sweeps; AIPerf (formerly GenAI-Perf) when you're deep in the NVIDIA/Triton stack and want first-class session/conversation modeling; LLMPerf when you need the same client hitting multiple *different* providers for an apples-to-apples comparison, remembering its concurrency mode measures a ceiling, not an SLA, and its fixed-`max_tokens` model under-represents reasoning workloads.
12. **Why does a "regression" in mean E2E latency sometimes not be a regression at all?** E2E scales with output length (\\( \text{E2E} = \text{TTFT} + (N-1)\cdot\text{TPOT} \\)); if the new run's requests happened to generate longer outputs (different random seed, different sampled dataset order, a longer reasoning trace), E2E goes up with no change in per-token cost. Always compare either matched length distributions or normalized (per-token) latency.
13. **What's the minimum sample size you'd want before trusting a reported p99?** On the order of 1,000+ post-warmup samples per metric per level; fewer than that and a single unlucky/lucky request can move the reported p99 by a large margin, and the run may not have reached queueing steady-state for Little's Law to hold.
14. **How do you turn a one-off load test into something that gates CI?** Pin the tool version, model, dataset, and length distribution so the run is reproducible; run it (or a cheaper proxy of the full sweep) automatically on every serving-config change; assert on concrete thresholds (e.g. p95 TTFT and output tok/s) so a regression fails the build rather than waiting to be noticed in production.
15. **What is prefix/KV-cache hit rate, and why does it belong in an agentic load-testing report?** The fraction of a request's prompt tokens served from a cached prior computation instead of recomputed; agentic sessions can be 80-99%+ prefix-reuse, so a benchmark that never creates that reuse (independent single-turn requests instead of session replay) will report a TTFT far worse than production actually sees, and a concurrency ceiling far lower than what's actually achievable once caching is engineered for.
16. **Why might load-balancing strategy alone cause a large throughput/latency regression in a multi-replica agentic deployment?** Naive round-robin routing sends a session's later turns to a different replica than its earlier turns, forcing that replica to recompute the whole accumulated prefix from scratch; the fix is either session-affinity routing or a distributed KV cache shared across replicas, and the difference between the two shows up as an order-of-magnitude swing in achievable cache-hit rate, not a subtle one.
17. **A batch of requests to a reasoning model has wildly inconsistent completion times even though every request entered the batch together. What's going on?** A straggler effect: one or two requests in the batch happen to need a much longer reasoning chain (harder problem), and because they share the batch, their completion time sets a floor for how long the whole batch's compute is tied up, dragging down the easier requests' effective throughput even though nothing is wrong with the server.
18. **If you can only run one load test before a launch, which one do you run?** The open-loop SLA validation at the target production arrival rate and realistic traffic mix (Phase 2 of the system-design sketch above) — it's the one that most directly answers "will this hold under real traffic," and it implicitly exercises most of what a concurrency sweep would show, even though a dedicated sweep gives you a cleaner knee estimate and more margin to reason about.

---

## Further Reading

**Core methodology**

- vLLM — Benchmark CLI (`vllm bench serve`): <https://docs.vllm.ai/en/latest/cli/bench/serve/>
- vLLM — Benchmarking overview & datasets: <https://docs.vllm.ai/en/latest/benchmarking/cli/>
- vLLM — `benchmarks/serve.py` source (dataset names, `--burstiness`, `--ramp-up-strategy`): <https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py>
- LLMPerf (Ray) — benchmarking library and `token_benchmark_ray.py`: <https://github.com/ray-project/llmperf>
- LLMPerf Leaderboard: <https://github.com/ray-project/llmperf-leaderboard>
- NVIDIA GenAI-Perf (Triton Perf Analyzer): <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html>
- NVIDIA AIPerf — Metrics Reference (TTFT/ITL definitions): <https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference>
- k6 — Open vs closed workload models: <https://grafana.com/docs/k6/latest/using-k6/scenarios/concepts/open-vs-closed/>
- Locust — documentation: <https://docs.locust.io/en/stable/>
- Gil Tene — "How NOT to Measure Latency" (coordinated omission, source): <https://www.infoq.com/presentations/latency-response-time/>
- ScyllaDB — On Coordinated Omission: <https://www.scylladb.com/2021/04/22/on-coordinated-omission/>
- Marc Brooker — Open, Closed, Omission and Collapse: <https://brooker.co.za/blog/2023/05/10/open-closed.html>
- Little's Law (overview): <https://en.wikipedia.org/wiki/Little%27s_law>
- Red Hat AI Inference Server — validating with key metrics (TTFT/TPOT/throughput): <https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.1/html/getting_started/validating-benefits-with-key-metrics_getting-started>

**Multi-turn, agentic, and reasoning-model benchmarking (2025-2026)**

- vLLM — multi-turn conversation benchmark PR (RFC #20265, KV-cache-offload replay): <https://github.com/vllm-project/vllm/pull/20267>
- Pliops — "Setting the Standard: Multi-Turn Benchmarking in vLLM": <https://pliops.com/setting-the-standard-multi-turn-benchmarking-in-vllm/>
- NVIDIA GenAI-Perf — Multi-Turn Chat benchmarking docs: <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/docs/multi_turn.html>
- NVIDIA AIPerf — Multi-Turn Conversations docs: <https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/multi-turn-conversations>
- NVIDIA AIPerf — Comprehensive LLM Benchmarking (successor to GenAI-Perf): <https://docs.nvidia.com/aiperf/getting-started/ai-perf-comprehensive-llm-benchmarking>
- Li et al. — "Reasoning Language Model Inference Serving: An Empirical Study" (arXiv:2510.18672, 2025): <https://arxiv.org/abs/2510.18672>
- NVIDIA Perspectives — "Infrastructure Economics: Reasoning Models & Chain-of-Thought" (2026): <https://perspectives.nvidia.com/infrastructure-economics-reasoning-models-chain-of-thought>
- Yuan, Nayak, Kundu, Talati — "Agentic AI Workload Characteristics" (arXiv:2605.26297, May 2026): <https://arxiv.org/abs/2605.26297>
- vLLM Blog — "Serving Agentic Workloads at Scale with vLLM x Mooncake" (May 6, 2026): <https://vllm.ai/blog/2026-05-06-mooncake-store>
