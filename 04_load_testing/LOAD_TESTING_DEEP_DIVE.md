# Load Testing LLM Inference — Measuring Throughput and Latency Correctly Under Load

## Why This Matters

You cannot capacity-plan, price, or SLA a serving system you have not load-tested. And LLM inference is unusually easy to benchmark *wrong*: a single request against an idle server tells you almost nothing about behavior at 200 concurrent users, because the whole point of a modern engine (vLLM, TGI, TensorRT-LLM) is **continuous batching** — throughput and latency both change as concurrency changes.

The stakes are concrete:

- **Capacity planning.** "How many GPUs do I need for 500 chat sessions?" is answerable only from a concurrency sweep.
- **SLA definition.** "p95 time-to-first-token under 500 ms" is meaningless without saying *at what load*.
- **Cost.** Output tokens/second/GPU is the number your finance model divides into. A 2× throughput win halves your bill.
- **Regression gating.** A benchmark you can rerun in CI catches the day someone flips `--enable-chunked-prefill` off.

A wrong benchmark is worse than none: it gives false confidence. Most of this chapter is about the ways benchmarks lie, and how to stop them.

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

---

## Metrics, Defined Precisely

Let a single streaming request produce output tokens at wall-clock times \( t_1 < t_2 < \dots < t_N \), with the request sent at \( t_0 \). Over a whole test, let \( R \) requests complete in wall-clock window \( T \) seconds, producing \( O \) total output tokens.

### Time To First Token (TTFT)

\[ \text{TTFT} = t_1 - t_0 \]

The latency until the first token appears. Dominated by **prefill** (processing the prompt) plus any **queueing wait** for a scheduler slot. This is what a user perceives as "responsiveness" — the cursor starting to move. TTFT is the metric most sensitive to load, because a queued request pays its wait entirely before \( t_1 \).

*Micro-example:* prompt sent at \( t_0 = 0 \), first token at \( t_1 = 0.18\text{ s} \) → TTFT = 180 ms.

### Time Per Output Token (TPOT) / Inter-Token Latency (ITL)

**TPOT** is the average gap between output tokens after the first, for one request:

\[ \text{TPOT} = \frac{t_N - t_1}{N - 1} \]

**ITL** is the per-gap version — the distribution of individual \( t_{i+1} - t_i \) values. TPOT is the mean of a request's ITLs. (Tools differ: vLLM reports both TPOT and ITL; some tools call TPOT "inter-token latency." Know which your tool means.) TPOT is governed by the **decode** phase; \( 1/\text{TPOT} \) is the per-user tokens/second — the perceived "typing speed."

*Micro-example:* a request emits 201 tokens, \( t_1 = 0.18 \), \( t_{201} = 4.18 \). TPOT \( = (4.18 - 0.18)/200 = 20\text{ ms} \) → each user sees ~50 tokens/s.

### End-to-End (E2E) Request Latency

\[ \text{E2E} = t_N - t_0 = \text{TTFT} + (N-1)\cdot\text{TPOT} \]

Total time from send to last token. **This is the number that scales with output length**, so it is only comparable across runs if output-length distributions match. A "latency regression" is often just longer outputs.

### Normalized latency (per-token E2E)

\[ \text{normalized latency} = \frac{\text{E2E}}{N} \]

Divides out output length, making runs with different output distributions comparable. vLLM's benchmark historically reported this.

### Request throughput

\[ \lambda_{\text{out}} = \frac{R}{T} \quad [\text{req/s}] \]

Completed requests per second across the whole run. The right top-line number when your unit of work is "a request" (e.g., classification).

### Output-token throughput

\[ X = \frac{O}{T} \quad [\text{tok/s}] \]

Generated tokens per second across all concurrent requests. **This is the money metric for generative workloads** and the one that goes up with better batching. Report **output** tokens (excludes the prompt); also report **total token throughput** (prompt + output) if prefill cost matters to you. Divide by GPU count for tok/s/GPU.

*Micro-example:* 64 concurrent requests each streaming at 50 tok/s → aggregate \( X \approx 3200 \) tok/s, even though each *user* still sees only 50 tok/s. Per-user speed and aggregate throughput are different axes.

### Percentiles

For a metric with sorted samples, the p-th percentile \( P_p \) is the smallest value \( \geq p\% \) of samples:

\[ P_p = \text{value at rank } \lceil \tfrac{p}{100} \cdot n \rceil \text{ in sorted order} \]

Report **p50 (median), p95, p99** for TTFT, TPOT, and E2E. p99 matters more than it looks: if a page makes 10 backend calls, the chance all 10 beat p99 is \( 0.99^{10} \approx 0.90 \) — so ~10% of *page loads* hit a p99 tail. Tail latency compounds.

---

## Open-Loop vs Closed-Loop Load Generation

This is the single most important methodology decision, and the one most benchmarks get wrong.

### Closed-loop

A fixed pool of \( C \) "virtual users." Each sends a request, **waits for the full response**, then immediately sends the next. Concurrency is capped at \( C \) by construction. This models a fixed number of clients in a tight loop (e.g., a batch job, or exactly \( C \) synchronous callers).

### Open-loop

Requests are launched on an **arrival schedule** (e.g., Poisson at rate \( \lambda \)) **independent of whether prior requests have finished**. In-flight concurrency is an *emergent property*, free to grow if the server slows down. This models real traffic: users arrive whether or not your server is keeping up.

### Coordinated omission — why closed-loop hides overload

Here is the trap. In a closed-loop test, when the server slows down, each virtual user's loop stalls waiting for its response — so **it stops sending new requests**. The offered load automatically backs off exactly when the system is struggling. The load generator "coordinates" with the server's slowness and **omits** the requests a real (open) world would have kept sending.

Consequences:

- Your measured request rate silently drops below target, and you may not notice.
- The latency samples you *do* collect exclude the requests that would have queued behind a slow one, so **tail latency is dramatically underreported**. The one 2-second stall in a real system would have delayed 40 requests behind it; closed-loop just... didn't send them.
- You conclude the system is healthy at a load it actually cannot sustain.

The fix in an open-loop tool is to **schedule requests on absolute wall-clock times** and measure each request's latency from its *intended* send time, not from when a freed-up worker got around to it. k6's `constant-arrival-rate`/`ramping-arrival-rate` executors and Gatling's open model do this; `constant-vus`/`ramping-vus` are closed. Corrected closed-loop tools (wrk2, Gatling) reconstruct the omitted samples by back-dating latency to the scheduled time.

### When each is right

- **Open-loop / fixed request-rate:** validating an SLA against realistic traffic ("can we hold p95 TTFT < 500 ms at 30 req/s?"). This is the honest default for user-facing services.
- **Closed-loop / fixed concurrency:** finding **maximum sustainable throughput** and the saturation curve, where a bounded, known concurrency is exactly the independent variable you want to sweep. LLMPerf and GenAI-Perf's concurrency mode work this way, and that is fine *because you are deliberately measuring the throughput ceiling*, not pretending to reproduce arrival traffic.

Use both: a concurrency sweep to find the knee, then an open-loop test at your chosen arrival rate to confirm the SLA holds with realistic tail behavior.

---

## Little's Law — Reasoning About Concurrency

Little's Law relates the three quantities you care about, for any stable system in steady state:

\[ L = \lambda \, W \]

- \( L \) = average number of requests **in the system** (concurrency / in-flight).
- \( \lambda \) = average **arrival = completion rate** (req/s), in steady state.
- \( W \) = average **time in system** (E2E latency, seconds).

It is an identity — no assumptions about distributions. Uses:

**Sanity-check a benchmark.** If your open-loop test offers \( \lambda = 20 \) req/s and you measure mean E2E \( W = 4 \) s, then average in-flight concurrency is \( L = 20 \times 4 = 80 \). If your engine's `--max-num-seqs` is 64, you are **oversubscribed**: requests are queuing, latency will keep rising, and the system is not in steady state. The law just told you the offered load exceeds capacity *before* you stare at a climbing latency graph.

**Convert closed-loop to a rate.** A closed-loop test with \( C = 100 \) virtual users measuring \( W = 2.5 \) s achieves \( \lambda = L/W = 100/2.5 = 40 \) req/s. That's how you translate a concurrency-sweep point into "requests per second this config sustains."

**Token form.** Apply it to tokens: aggregate output throughput \( X \) (tok/s) with \( n \) requests in flight each of length \( N \) tokens taking \( W \) seconds gives \( X = nN/W \) — decompose regressions into "fewer concurrent" vs "slower per request."

Caveat: Little's Law holds **in steady state**. During warmup, ramp, or overload it does not, which is one more reason to discard warmup and to hold each sweep point long enough to stabilize.

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

### Sample results — a concurrency/rate sweep

Run the client at increasing rates against one A100 serving an 8B model, fixed input ≈512 / output ≈200 tokens:

| Target rate (req/s) | Achieved req/s | Output tok/s | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | E2E p95 (ms) | Avg in-flight \(L=\lambda W\) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5   | 5.0  | 1000 | 42   | 70   | 15 | 3200  | 16 |
| 10  | 10.0 | 2000 | 55   | 95   | 17 | 3600  | 36 |
| 20  | 20.0 | 4000 | 88   | 210  | 21 | 4400  | 88 |
| 30  | 29.8 | 5900 | 180  | 620  | 28 | 6800  | 200 |
| 40  | 33.1 | 6600 | 540  | 2400 | 41 | 14200 | 470 |
| 50  | 32.9 | 6600 | 1900 | 9000 | 63 | 41000 | 1350 |

### Reading the saturation curve

- **5 → 20 req/s:** achieved rate tracks target, output tok/s scales ~linearly (1000→4000), TTFT p95 stays modest. Underloaded region — GPU has headroom.
- **~30 req/s is the knee.** Output tok/s (5900) is close to the ceiling; achieved rate still ≈ target but TTFT p95 has jumped to 620 ms. This is the **operating point** you'd target for a latency-sensitive service, maybe backing off to ~25 for headroom.
- **40 → 50 req/s: overload.** Achieved rate **flatlines at ~33 req/s** even as you offer more — that ~33 req/s (≈6600 tok/s) is the true saturation throughput. Meanwhile latency explodes (TTFT p95 2.4 s → 9 s; E2E p95 41 s) and Little's-Law in-flight \( L \) blows past any sane `max-num-seqs`. Offering 50 doesn't get you 50; it just grows the queue.

The signature of the knee: **throughput stops rising while latency starts rising superlinearly.** Peak throughput and acceptable latency are *different* points — publish both, and state which you're operating at.

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

---

## Tools Comparison

| Tool | Load model | Metrics reported | Endpoints | Best for | Watch out |
|---|---|---|---|---|---|
| **vLLM `benchmark_serving.py` / `vllm bench serve`** | Open-loop via `--request-rate` (`inf` = burst all `--num-prompts` at once); Poisson/`--burstiness` | Request throughput (req/s), Output token throughput (tok/s), Total token throughput, Mean/Median/P99 **TTFT, TPOT, ITL**, E2E | OpenAI-compatible + native vLLM/TGI backends | Purpose-built LLM serving benchmarks; realistic datasets (`--dataset-name sharegpt/random/sonnet`) | Ships with vLLM version; `--request-rate inf` is a burst, not steady rate |
| **LLMPerf** (Ray) | Closed-loop, `--num-concurrent-requests` | TTFT, inter-token latency, E2E, output throughput per-request and aggregate | Many providers (OpenAI, Anthropic, Together, Bedrock, Vertex, SageMaker, vLLM) | Cross-provider apples-to-apples; the LLMPerf leaderboard | Concurrency mode = throughput ceiling, not arrival-rate SLA; token counts are provider-tokenized approximations |
| **NVIDIA GenAI-Perf** (Triton Perf Analyzer) | Both: `--concurrency` (closed) or `--request-rate` (open) | TTFT, inter-token latency, output token throughput, request throughput, seq lengths, all with avg/p90/p99 | OpenAI-compatible, Triton (TRT-LLM, vLLM), gRPC/HTTP | Deep NVIDIA/Triton stacks; synthetic + custom datasets; rich exports | Heavier setup; Triton-centric defaults |
| **Locust** | Closed-loop by default; `constant_throughput` ≈ open | Whatever you instrument; built-in RPS + percentile table + web UI | Any HTTP (write Python tasks) | Custom flows, quick dashboards, mixed traffic | Python workers can be the bottleneck; streaming/TTFT needs custom code |
| **k6** | **Open-loop** (`constant/ramping-arrival-rate`) or closed (`*-vus`) | RPS, latency percentiles, custom Trends | Any HTTP/gRPC (JS scripts) | Honest open-loop SLA tests, CI gating | Go runtime doesn't tokenize; TTFT needs manual SSE parsing/custom metrics |

Rule of thumb: **`benchmark_serving.py`/GenAI-Perf** when you want LLM-native metrics and datasets out of the box; **k6** when you want a rigorous open-loop SLA test; **LLMPerf** for cross-provider comparisons; **Locust** for bespoke multi-step traffic with a dashboard.

---

## Failure Modes and Pitfalls

**Closed-loop coordinated omission.** Covered above — the big one. A closed-loop tool stops sending when the server stalls, so it under-reports tail latency and over-reports sustainable load. Fix: use open-loop arrival-rate executors, or a corrected tool that back-dates latency to the intended send time. If you *must* go closed-loop, only use it to measure the throughput ceiling, and never quote its latencies as an SLA.

**No warmup.** The first requests hit cold caches: CUDA graph capture, torch.compile / TRT-LLM engine warmup, cuBLAS autotune, KV-cache allocation, JIT. Cold TTFT can be 5–50× steady-state. Including warmup poisons your percentiles (a handful of huge samples wreck p99). Fix: send a warmup burst and **discard the first N seconds/requests** (the example's `--warmup`). Also warm up the *client* (DNS, TLS, connection pool).

**Unrealistic input/output lengths.** A benchmark with 128-in/128-out tokens tells you nothing about a RAG workload with 4000-in/500-out. Prefill cost scales with input length; decode cost and E2E scale with output length; KV-cache pressure scales with both × concurrency. **Fixed lengths also hide batching dynamics** because every request finishes together. Fix: replay a **realistic length distribution** (ShareGPT, your own production logs, or `--dataset-name random` with a mean/std matching production). Report the distribution you used.

**Measuring only averages.** The mean hides the tail and can be dominated by a few slow requests. Always report p50/p95/p99 (and max) for TTFT, TPOT, and E2E separately. And never average latency *across different output lengths* without normalizing — longer outputs inflate E2E and masquerade as a regression.

**Client-side bottleneck.** The most insidious: your load generator, not the server, is the limit. Symptoms: achieved rate plateaus well below the server's known capacity, client CPU pegged at 100%, or latency that scales with *client* concurrency. Causes: single-threaded Python parsing SSE, aiohttp/`requests` connection-pool caps, GIL contention, running the client on the same box as the server, or a network link between client and server that's the actual bottleneck. Fixes: pin and monitor client CPU, remove connection caps (`TCPConnector(limit=0)`), use async or distributed workers (k6/Locust workers, multiple client hosts), and **sanity-check with Little's Law** — if measured \( L = \lambda W \) can't reach the server's `max-num-seqs`, your client is starving it.

**Reusing identical prompts / prefix caching artifacts.** If every request sends the same prompt, prefix caching makes prefill nearly free and TTFT looks unrealistically good. Vary prompts (or explicitly test both with- and without-cache) so you measure the case you'll actually run.

**Not holding steady state / too-short runs.** Little's Law and stable percentiles need steady state. A 5-second run at 30 req/s is ~150 samples — too few for a trustworthy p99 (you want ≥1000+ post-warmup samples) and too short to reach queue equilibrium. Run each sweep point long enough that metrics stop drifting.

---

## Production Checklist — What an Interviewer Probes

1. **"What do you measure, and why not just average latency?"** — Name TTFT, TPOT/ITL, E2E, request- and output-token throughput; report **p50/p95/p99**, not means, because latency is heavy-tailed and the tail is what users feel. Mean is only for throughput accounting.
2. **"Open-loop or closed-loop, and what's coordinated omission?"** — Open-loop (arrival-rate) for SLA validation; closed-loop (fixed concurrency) only to find the throughput ceiling. Coordinated omission = closed-loop stops sending when the server stalls, so it hides the tail. Bonus: measure latency from the *intended* send time.
3. **"How do you find the right operating point?"** — Sweep concurrency/rate, plot throughput and p95 latency vs load, find the **knee** where throughput flattens and latency turns up; operate just below it with headroom.
4. **"Apply Little's Law here."** — \( L = \lambda W \). Given a rate and E2E latency, compute in-flight concurrency and compare to `max-num-seqs` to detect oversubscription; convert closed-loop concurrency to sustainable req/s.
5. **"How do you make the workload realistic?"** — Match production input/output length distributions (ShareGPT / replayed logs), vary prompts to avoid prefix-cache flattery, and report the distribution used.
6. **"How do you know the client isn't the bottleneck?"** — Monitor client CPU, remove connection caps, use async/distributed generators, separate client and server hosts, and cross-check achieved \( L \) against server capacity.
7. **"Warmup and steady state?"** — Discard cold-start requests (CUDA graphs/compile/cache allocation); run long enough (≥~1000 post-warmup samples, metrics stable) for trustworthy p99.
8. **"How do you gate regressions in CI?"** — A reproducible benchmark (pinned tool/model/dataset/lengths) run at a fixed rate, asserting on p95 TTFT and output tok/s thresholds, so a config regression fails the build.

---

## Further Reading

- vLLM — Benchmark CLI (`vllm bench serve`): <https://docs.vllm.ai/en/latest/cli/bench/serve/>
- vLLM — Benchmarking overview & datasets: <https://docs.vllm.ai/en/latest/benchmarking/cli/>
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
