# Monitoring LLM Serving — Observability for GPU Inference in Production

> **Why this matters.** A classic web service is healthy when latency and error
> rate look good. An LLM server can pass both of those checks and still be
> quietly on fire: the KV cache is 98% full, requests are piling up in a queue
> you never graphed, and your p50 looks fine only because the p99 users already
> gave up and disconnected. Serving LLMs introduces metrics that ordinary
> dashboards don't have — token-level latency, cache pressure, batch dynamics —
> and if you don't measure them you cannot run the system. This chapter is about
> the metrics that actually matter for GPU inference, where they come from, and
> how to wire Prometheus, Grafana, DCGM, and tracing together into an
> observability stack you can put on-call against.

---

## Core intuition: LLM serving has latency your web stack never had

A REST endpoint has essentially one latency: request in, response out. An LLM
endpoint has **three** latencies, and users feel all of them differently.

1. **Time To First Token (TTFT)** — how long until the first token appears.
   This is the *prefill* cost: the model processes the whole prompt before it
   can emit anything. Long prompts, cold caches, and queue waiting all inflate
   TTFT. For a chat UI this is the "is it thinking?" delay and it dominates
   perceived responsiveness.

2. **Time Per Output Token (TPOT)**, a.k.a. inter-token latency (ITL) — the
   gap between subsequent tokens during *decode*. This sets the "typing speed"
   of the stream. A user reads at maybe 5–10 tokens/sec; if TPOT is 100 ms
   (10 tok/s) the stream feels fluid, at 300 ms it feels painful.

3. **End-to-end latency** — total wall-clock for the whole response. This is
   roughly \( \text{TTFT} + (N_{\text{output}} - 1) \times \text{TPOT} \), so a
   long answer amplifies a small per-token regression into a large total.

The second thing that's different: **the bottleneck is a fixed pool of GPU
memory, not CPU or connection count.** Modern engines (vLLM, TGI) batch many
sequences together and store each sequence's attention state in a **KV cache**
carved out of GPU HBM. When the cache fills, the scheduler stops admitting new
sequences — they wait in a queue — or it **preempts** running ones and
recomputes them later. So the health signals that predict a latency cliff are
*queue depth* and *KV-cache utilization*, not GPU-percent-busy. A GPU can read
100% utilized while the real problem is that it's thrashing the cache.

Keep two mental models side by side:

- **RED** (Rate, Errors, Duration) — the *request-centric* view. Good for the
  API surface users touch.
- **USE** (Utilization, Saturation, Errors) — the *resource-centric* view.
  Good for the GPU and the KV cache. **Saturation** — the queue and the cache —
  is where LLM serving lives or dies, and it's the box most teams forget.

---

## Metrics catalog — what to measure and why

| Metric | What it means | Why it matters | Healthy range / notes |
|---|---|---|---|
| **TTFT p50/p95/p99** | Time until first token | Perceived responsiveness; captures prefill + queue wait | Interactive chat: p95 < 1–2 s. Rising p95 with flat p50 = queue building |
| **TPOT / inter-token latency** | Steady-state gap between output tokens | Stream "typing speed"; regressions multiply over long outputs | 10–50 ms/token typical; > 100 ms feels slow |
| **E2E latency p50/p95/p99** | Full request wall-clock | The SLO users actually sign; skewed by output length | Always report percentiles, never the mean |
| **Throughput — requests/s** | Completed requests per second | Capacity planning, autoscaling signal | Compare against offered load; gap = queue growth |
| **Throughput — tokens/s** | Generated tokens per second (decode) | The real work rate of the GPU; the currency of cost | Track output tok/s separately from prompt tok/s |
| **Queue depth / waiting seqs** | Requests admitted-but-waiting | Leading indicator of a latency cliff | Should hover near 0; sustained > 0 = under-provisioned |
| **Running sequences** | Sequences decoding right now | Effective batch size; drives GPU efficiency | Low + full queue = memory-bound, not compute-bound |
| **GPU utilization** | % time GPU had work scheduled | Coarse "is the GPU busy" signal | High util ≠ efficient; can be high while thrashing |
| **GPU memory used / free** | HBM in use (framebuffer) | OOM risk; headroom for larger batches/KV | Leave headroom; OOM crashes the whole replica |
| **KV-cache utilization** | Fraction of paged KV blocks in use | *The* saturation signal for LLM serving | > 90% sustained → preemption, TTFT spikes |
| **Batch size** | Sequences processed per step | Throughput vs latency tradeoff knob | Larger = more throughput, higher TPOT |
| **Preemptions** | Sequences evicted & recomputed | Direct evidence of cache pressure | Any sustained rate is a red flag |
| **Error rate** | Failed / total requests | Availability SLI; 5xx, OOM, timeouts, truncations | Alert on rate, not raw count |
| **Cost per 1k tokens** | \$ per 1000 tokens served | Turns efficiency into money; the exec-facing number | Derived: GPU \$/hr ÷ (tokens/s × 3.6) |

The rule of thumb: **latency metrics are the SLIs; queue, KV-cache, and
preemptions are the leading indicators; GPU/memory are the resource ceiling;
cost is the business translation.**

---

## Mechanism 1 — Scraping engine metrics

Both major open-source engines expose Prometheus metrics natively. You don't
instrument the model; you scrape the server.

### vLLM

vLLM publishes a `/metrics` endpoint on its OpenAI-compatible API server (same
port as the API, default `8000`). Every metric is prefixed `vllm:`. The ones
that matter, by type
([vLLM production metrics](https://docs.vllm.ai/en/v0.6.1/serving/metrics.html),
[metrics design](https://docs.vllm.ai/en/latest/design/metrics/)):

**Histograms** (latency — these give you percentiles):

- `vllm:time_to_first_token_seconds` — TTFT
- `vllm:time_per_output_token_seconds` — TPOT / inter-token latency
- `vllm:e2e_request_latency_seconds` — full request latency
- `vllm:request_prompt_tokens` — prompt length distribution
- `vllm:request_generation_tokens` — output length distribution

**Gauges** (instantaneous system state):

- `vllm:num_requests_running` — sequences currently decoding
- `vllm:num_requests_waiting` — sequences queued (the saturation signal)
- `vllm:num_requests_swapped` — swapped to CPU under pressure
- `vllm:gpu_cache_usage_perc` — KV-cache utilization (**a fraction 0–1**, so
  multiply by 100 to get a percent — the name is misleading)
- `vllm:gpu_prefix_cache_hit_rate` — prefix-cache reuse rate

**Counters** (cumulative — take `rate()` of these):

- `vllm:prompt_tokens_total` — prompt tokens processed
- `vllm:generation_tokens_total` — output tokens generated (throughput source)
- `vllm:request_success_total` — successful requests (has a `finished_reason`
  label so you can separate `stop` vs `length` vs `abort`)
- `vllm:num_preemptions_total` — cache-pressure evictions

Histograms are exposed as three series each: `_bucket` (cumulative, labelled by
`le`), `_sum`, and `_count`. You compute percentiles from `_bucket` and averages
from `_sum / _count`.

### TGI (Text Generation Inference)

Hugging Face TGI exposes `/metrics` with a `tgi_` prefix
([TGI metrics reference](https://huggingface.co/docs/text-generation-inference/main/en/reference/metrics)):

- `tgi_request_duration` — end-to-end latency (histogram)
- `tgi_request_inference_duration` — inference time excluding queue (histogram)
- `tgi_request_queue_duration` — time spent waiting in queue (histogram)
- `tgi_request_mean_time_per_token_duration` — inter-token latency (histogram)
- `tgi_batch_current_size` — current batch size (gauge)
- `tgi_batch_current_max_tokens` — token budget of current batch (gauge)
- `tgi_queue_size` — requests waiting (gauge)
- `tgi_request_count` / `tgi_request_success` — request counters

Note the naming gap: TGI does not ship a single metric literally named "TTFT."
You approximate it as `tgi_request_queue_duration` + the prefill portion, or you
capture first-token timing at the client / gateway. This is a common source of
dashboard confusion — always confirm which engine you're scraping and map its
names onto your canonical SLIs.

---

## Mechanism 2 — GPU metrics with DCGM

Engine metrics tell you about *requests*. They don't tell you the GPU is at
90 °C, throttling its clocks, or that another process is stealing HBM. For that
you run NVIDIA's **DCGM exporter**, which reads the Data Center GPU Manager and
exposes Prometheus metrics on **port 9400**
([NVIDIA/dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter),
[DCGM exporter docs](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html)).

Key fields (all prefixed `DCGM_FI_`):

| Metric | Meaning |
|---|---|
| `DCGM_FI_DEV_GPU_UTIL` | GPU utilization (% of time a kernel was resident) |
| `DCGM_FI_DEV_FB_USED` | Framebuffer (HBM) memory used, MiB |
| `DCGM_FI_DEV_FB_FREE` | Framebuffer memory free, MiB |
| `DCGM_FI_DEV_POWER_USAGE` | Board power draw, watts |
| `DCGM_FI_DEV_GPU_TEMP` | GPU die temperature, °C |
| `DCGM_FI_DEV_SM_CLOCK` | SM clock, MHz (watch for throttling) |
| `DCGM_FI_PROF_GR_ENGINE_ACTIVE` | Graphics/compute engine active ratio |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor-core pipe active ratio (real compute intensity) |
| `DCGM_FI_PROF_DRAM_ACTIVE` | Memory-bandwidth active ratio |

Two subtleties worth internalizing:

- **`DCGM_FI_DEV_GPU_UTIL` is a liar for LLM decode.** It reports "a kernel was
  scheduled," which is nearly always true during autoregressive decode even when
  the GPU is memory-bandwidth-bound and compute-idle. Use
  `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` and `DCGM_FI_PROF_DRAM_ACTIVE` to see whether
  you're compute-bound or bandwidth-bound.
- Every DCGM series carries a `gpu` (index) and usually `UUID`/`modelName`
  label, so on a multi-GPU node you aggregate or break down per device.

---

## Mechanism 3 — Prometheus + Grafana wiring

Prometheus **pulls** metrics on an interval from targets you list; Grafana
queries Prometheus with PromQL to draw panels. For LLM serving you point
Prometheus at three kinds of targets: the inference engines, the DCGM
exporters, and (optionally) your gateway/load balancer.

A worked, correct scrape config (`prometheus.yml`):

```yaml
global:
  scrape_interval: 15s          # pull every 15s
  evaluation_interval: 15s      # evaluate alert rules every 15s

rule_files:
  - "alerts/llm_serving.yml"    # alert rules loaded below

scrape_configs:
  # vLLM / TGI inference servers (engine metrics)
  - job_name: "vllm"
    metrics_path: /metrics
    static_configs:
      - targets:
          - "vllm-0.inference.svc:8000"
          - "vllm-1.inference.svc:8000"
        labels:
          engine: vllm
          model: "llama-3-8b-instruct"

  # DCGM exporter (one per GPU node), port 9400
  - job_name: "dcgm"
    static_configs:
      - targets:
          - "gpu-node-0:9400"
          - "gpu-node-1:9400"

  # In Kubernetes you'd usually replace static_configs with
  # kubernetes_sd_configs + relabeling, or annotate pods with
  # prometheus.io/scrape and let the k8s SD discover them.
```

> Shell tip: to sanity-check a target before wiring it up, just
> `curl -s http://vllm-0:8000/metrics | grep vllm:` — the `$` you see in a
> prompt is literal.

### PromQL: the queries that earn their keep

These are copy-pasteable against the metric names above. Histogram percentiles
use `histogram_quantile` over the `_bucket` series, summed by the `le` label.

**TTFT p95 over the last 5 minutes:**

```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m]))
)
```

**Inter-token latency (TPOT) p99:**

```promql
histogram_quantile(
  0.99,
  sum by (le) (rate(vllm:time_per_output_token_seconds_bucket[5m]))
)
```

**Output-token throughput (tokens/s), the real work rate:**

```promql
sum(rate(vllm:generation_tokens_total[1m]))
```

**Request throughput (req/s), broken down by outcome:**

```promql
sum by (finished_reason) (rate(vllm:request_success_total[1m]))
```

**KV-cache utilization as a percent** (remember it's a 0–1 fraction):

```promql
avg(vllm:gpu_cache_usage_perc) * 100
```

**Queue depth (waiting sequences) — your saturation early warning:**

```promql
sum(vllm:num_requests_waiting)
```

**GPU utilization vs. real tensor activity, per device:**

```promql
avg by (gpu) (DCGM_FI_DEV_GPU_UTIL)
avg by (gpu) (DCGM_FI_PROF_PIPE_TENSOR_ACTIVE) * 100
```

**GPU memory used percent:**

```promql
100 * DCGM_FI_DEV_FB_USED
  / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)
```

**Cost per 1k tokens** — combine a static price with live throughput. With a
recording rule holding the GPU hourly price, cost per 1k output tokens is:

\[
\text{cost}_{1k} = \frac{\text{price}_{\$/\text{hr}}}{\text{tokens/s} \times 3.6}
\]

```promql
# price_per_gpu_hour is a constant series you set (e.g. via a recording rule)
(sum(price_per_gpu_hour))
  / (sum(rate(vllm:generation_tokens_total[5m])) * 3.6)
```

(The `3.6` converts tokens/second into thousands-of-tokens/hour:
\( \text{tok/s} \times 3600\,\text{s/hr} \div 1000 = \text{tok/s} \times 3.6 \).)

### Grafana

Build one dashboard per concern and template it with a `$model` /
`$engine` / `$gpu` variable so a single dashboard serves every replica:

- **Latency row** — TTFT p50/p95/p99, TPOT p95, E2E p95/p99 (time-series).
- **Throughput row** — req/s and tokens/s, with offered-vs-served overlaid.
- **Saturation row** — waiting sequences, running sequences, KV-cache %,
  preemption rate. This row is what tells you *why* latency moved.
- **Resource row** — GPU util, tensor-active, HBM used %, power, temp, SM clock
  from DCGM.
- **Cost row** — \$/1k tokens and \$/hr per replica.

Always plot percentiles as separate series; never a single "avg latency" line.

---

## Mechanism 4 — SLIs, SLOs, and alerting

An **SLI** is a measured signal; an **SLO** is the target you promise; an
**alert** fires when you're at risk of missing it. For interactive LLM serving a
reasonable starting SLO set:

| SLI | Example SLO |
|---|---|
| TTFT p95 | < 1.5 s over rolling 5 min |
| TPOT p95 | < 80 ms/token |
| E2E availability (non-error rate) | ≥ 99.9% over 30 days |
| Error rate | < 0.1% of requests |

Alert on **symptoms users feel** (SLO burn) and on **leading indicators**
(saturation), not on raw resource gauges. A full example rule file
(`alerts/llm_serving.yml`):

```yaml
groups:
  - name: llm_serving
    rules:
      # ---- Symptom: TTFT SLO breach ----
      - alert: TTFTHighP95
        expr: |
          histogram_quantile(
            0.95,
            sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m]))
          ) > 1.5
        for: 10m
        labels:
          severity: page
        annotations:
          summary: "TTFT p95 above 1.5s SLO"
          description: "p95 first-token latency is {{ $value | humanizeDuration }} on {{ $labels.model }}."

      # ---- Leading indicator: queue building ----
      - alert: RequestQueueBuilding
        expr: sum(vllm:num_requests_waiting) > 20
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Requests queueing at the engine"
          description: "{{ $value }} sequences waiting — scale out or shed load before TTFT breaches."

      # ---- Leading indicator: KV cache saturation ----
      - alert: KVCacheSaturated
        expr: avg(vllm:gpu_cache_usage_perc) * 100 > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "KV cache > 90%"
          description: "Cache pressure imminent; expect preemptions and TTFT spikes."

      # ---- Resource: GPU memory near OOM ----
      - alert: GPUMemoryHigh
        expr: |
          100 * DCGM_FI_DEV_FB_USED
            / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE) > 95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU {{ $labels.gpu }} HBM > 95%"

      # ---- Symptom: error budget burn ----
      - alert: HighErrorRate
        expr: |
          sum(rate(vllm:request_success_total{finished_reason="abort"}[5m]))
            / sum(rate(vllm:request_success_total[5m])) > 0.01
        for: 5m
        labels:
          severity: page
        annotations:
          summary: "Request abort rate > 1%"
```

The `for:` clause suppresses flapping — the condition must hold continuously
before it pages. Pair symptom pages (wake someone up) with leading-indicator
warnings (fix it before it pages). Advanced teams add **multi-window
multi-burn-rate** error-budget alerts so a fast burn pages immediately and a
slow burn opens a ticket.

---

## RED and USE, applied to inference

**RED** — instrument the request surface:

- **Rate** — `rate(vllm:request_success_total[1m])` (req/s).
- **Errors** — the abort / failure ratio shown above.
- **Duration** — TTFT, TPOT, and E2E histograms. LLM serving splits "Duration"
  into three because users feel three.

**USE** — instrument the constrained resource (the GPU and its cache):

- **Utilization** — `DCGM_FI_DEV_GPU_UTIL`, and more honestly
  `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` / `DCGM_FI_PROF_DRAM_ACTIVE`; HBM used %.
- **Saturation** — `vllm:num_requests_waiting` (queue) and
  `vllm:gpu_cache_usage_perc` (KV cache). **This is the LLM-specific box.**
  Classic USE saturation is CPU run-queue; here it's *sequences waiting for
  cache blocks.*
- **Errors** — OOM kills, CUDA errors, preemption-driven recompute
  (`vllm:num_preemptions_total`).

Run both: RED catches the user-facing symptom, USE tells you which resource
caused it. The link between them is almost always the saturation row — queue and
cache — which is exactly the pair that generic dashboards omit.

---

## Distributed tracing with OpenTelemetry

Metrics tell you *that* p99 TTFT is bad; a **trace** tells you *where* the time
went for one slow request as it crossed gateway → queue → prefill → decode.
vLLM ships OpenTelemetry support: start it with
`--otlp-traces-endpoint <collector:4317>` and it emits spans with attributes for
queue time, TTFT, and per-request token counts, exported over OTLP to a
collector (Jaeger, Tempo, etc.)
([vLLM OpenTelemetry example](https://docs.vllm.ai/en/v0.9.0/examples/online_serving/opentelemetry.html)).

Propagate a `traceparent` header from your gateway through to the engine so the
model's spans nest under the user request. The payoff: for a single tail-latency
request you can see whether the 4 seconds was 3.8 s of queue wait (scale out),
prefill on an 8k-token prompt (input-length problem), or slow decode (batch /
memory-bandwidth problem). Metrics aggregate; traces let you debug one victim.
In practice you **sample** traces (e.g. 1–5%, plus always-sample on error) to
keep cost and cardinality sane.

---

## Failure modes and pitfalls

- **Alerting on averages.** A mean latency of 400 ms can hide a p99 of 12 s.
  Averages hide the tail, and the tail is who churns. Always alert on
  percentiles from `_bucket` series, and compute them with `histogram_quantile`
  over a `rate()`, never over a raw counter.

- **No queue or KV-cache metrics.** The single most common LLM-monitoring gap.
  Without `num_requests_waiting` and `gpu_cache_usage_perc` you get *no warning*
  before the latency cliff — GPU util reads high and everything "looks fine"
  right up until TTFT triples. These are your leading indicators; graph and
  alert on them.

- **Trusting `DCGM_FI_DEV_GPU_UTIL`.** It says "a kernel ran," not "the GPU did
  useful work." During decode it sits near 100% while the device is
  memory-bandwidth-bound and compute-starved. Cross-check with
  `PIPE_TENSOR_ACTIVE` / `DRAM_ACTIVE` before concluding you're compute-bound.

- **Cardinality explosions.** Labelling metrics with unbounded values —
  `request_id`, raw prompt text, user IDs, full model paths — multiplies time
  series until Prometheus OOMs. Keep labels low-cardinality (`model`, `engine`,
  `gpu`, `finished_reason`). Push per-request detail into **traces/logs**, not
  metric labels.

- **Misreading `gpu_cache_usage_perc`.** It's a 0–1 fraction despite the
  `_perc` suffix. Forgetting the `* 100` silently makes a "90% full" alert fire
  at 9000% or never fire at all.

- **Percentiles over the wrong window.** `histogram_quantile` on a `[5m]` rate
  is a 5-minute view; too short and it's noisy, too long and it lags an
  incident. Match the window to the SLO evaluation period.

- **Averaging pre-computed percentiles across replicas.** You cannot average
  p95s. Sum the `_bucket` series across replicas *first*, then take the
  quantile. Aggregating already-quantiled numbers gives a wrong answer.

- **No cost visibility.** If nobody graphs \$/1k tokens, efficiency regressions
  (a bad batch-size change, an underutilized replica) go unnoticed until the
  cloud bill arrives. Cost is the metric leadership reads; derive it from
  tokens/s and GPU price.

- **Blind spot between gateway and engine.** If you only scrape the engine you
  miss load-balancer queuing and network time. Measure TTFT at the edge too, and
  reconcile the two.

---

## Production checklist — what an interviewer probes

1. **"Which latency metrics for an LLM, and why not just one?"** — Name TTFT,
   TPOT/ITL, and E2E; explain prefill vs decode and that E2E ≈ TTFT + (N−1)·TPOT.
2. **"How do you know before users do?"** — Point to queue depth
   (`num_requests_waiting`) and KV-cache utilization as leading indicators, not
   GPU util.
3. **"Show me the PromQL for TTFT p95."** — `histogram_quantile(0.95, sum by (le)
   (rate(vllm:time_to_first_token_seconds_bucket[5m])))`, and know why you sum
   `_bucket` by `le` first.
4. **"GPU util is 100% — are you compute-bound?"** — Not necessarily; decode is
   often memory-bandwidth-bound. Cross-check `PIPE_TENSOR_ACTIVE` /
   `DRAM_ACTIVE`.
5. **"What do you alert on?"** — Symptoms (SLO burn on TTFT/error rate) *plus*
   leading indicators (queue, KV cache), with `for:` to debounce; ideally
   multi-burn-rate error budgets.
6. **"How do you avoid a Prometheus cardinality blowup?"** — Low-cardinality
   labels only; per-request detail goes to traces/logs.
7. **"How do you debug one slow request?"** — OpenTelemetry tracing with
   propagated `traceparent`, sampled, to split queue vs prefill vs decode time.
8. **"What's your cost metric?"** — \$/1k tokens derived from GPU \$/hr and
   `rate(generation_tokens_total)`, tracked per model/replica.

---

## Further reading

- vLLM — Production Metrics: <https://docs.vllm.ai/en/v0.6.1/serving/metrics.html>
- vLLM — Metrics design (V1): <https://docs.vllm.ai/en/latest/design/metrics/>
- vLLM — OpenTelemetry example: <https://docs.vllm.ai/en/v0.9.0/examples/online_serving/opentelemetry.html>
- TGI — Metrics reference: <https://huggingface.co/docs/text-generation-inference/main/en/reference/metrics>
- NVIDIA DCGM exporter (GitHub): <https://github.com/NVIDIA/dcgm-exporter>
- NVIDIA DCGM exporter docs: <https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html>
- Prometheus — Querying / `histogram_quantile`: <https://prometheus.io/docs/prometheus/latest/querying/functions/#histogram_quantile>
- Prometheus — Alerting rules: <https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/>
- Grafana — The RED Method (Tom Wilkie): <https://grafana.com/blog/the-red-method-how-to-instrument-your-services/>
- Brendan Gregg — The USE Method: <https://www.brendangregg.com/usemethod.html>
- Google SRE Workbook — Alerting on SLOs: <https://sre.google/workbook/alerting-on-slos/>
- OpenTelemetry — Documentation: <https://opentelemetry.io/docs/>
