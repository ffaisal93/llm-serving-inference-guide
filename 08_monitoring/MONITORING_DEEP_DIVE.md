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

> **Saying it out loud.** So the short version is that an LLM endpoint doesn't
> have one latency, it has three, and users feel each one differently. There's
> time to first token — how long you stare at a blank screen while the model
> reads your prompt — then time per output token, which is basically the
> typing speed of the stream, and then total wall clock, which is just the
> first one plus the length of the answer times the second one. The other
> thing that surprises people coming from web services is that the resource
> you run out of isn't CPU or connections, it's a fixed pool of GPU memory
> called the KV cache, which holds the attention state for every in-flight
> request. So the signals that predict trouble are queue depth and how full
> that cache is — GPU utilization can read 100% while the card is
> memory-bandwidth-bound and doing almost no useful math.

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
| **Prefix-cache hit rate** | Fraction of prompt tokens served from reused KV blocks | Reused context is free prefill; regressions inflate TTFT with no traffic change | vLLM V1 replaced the raw hit-rate gauge with `cache_query_hit` / `cache_query_total` counters — derive the ratio yourself |
| **Spec-decode acceptance rate** | Fraction of speculatively drafted tokens the target model accepts | Tells you whether speculative decoding is actually buying throughput | < 50% acceptance usually means the draft model or config is miscalibrated for current traffic |
| **Prefill vs. decode time** | Split of per-request time between prompt processing and generation | Separates "the prompt got longer" from "steady-state decode got slower" | Exposed directly as separate histograms in newer engine versions |
| **Error rate** | Failed / total requests | Availability SLI; 5xx, OOM, timeouts, truncations | Alert on rate, not raw count |
| **Cost per 1k tokens** | $ per 1000 tokens served | Turns efficiency into money; the exec-facing number | Derived: GPU $/hr ÷ (tokens/s × 3.6) |

The rule of thumb: **latency metrics are the SLIs; queue, KV-cache, and
preemptions are the leading indicators; GPU/memory are the resource ceiling;
cost is the business translation.** The newer rows above — prefix-cache hit
rate, spec-decode acceptance, and the prefill/decode split — are recent
additions to the major engines' metric surfaces; see "The 2025–2026 landscape"
below for exactly what changed, when, and why it matters for your dashboards.

> **Saying it out loud.** If someone asks what I'd measure, I'd give four
> buckets rather than reciting a list. Latency percentiles are the SLIs — the
> thing you actually promise users. Queue depth, KV-cache utilization, and
> preemptions are the leading indicators, the stuff that moves before the SLI
> does. GPU and HBM are the ceiling you're pushing against, and dollars per
> thousand tokens is the translation for whoever pays the bill. The one people
> forget is preemptions — any sustained rate of sequences being evicted and
> recomputed means you're already past the cliff, you just haven't seen it in
> latency yet.

---

## The 2025–2026 landscape

Two things changed in the last eighteen months or so: (1) OpenTelemetry started
standardizing GenAI *metric names* — not just traces — which matters for
serving infrastructure specifically, not only application code; and (2) the
engines and GPU exporters you already scrape grew new fields for prefix
caching, speculative decoding, and profiling. This section is a dated snapshot
of where things stand as of mid-2026 so you know which names are stable enough
to build alerts on and which are still moving.

> **Saying it out loud.** The honest framing here is that the metric names you
> build dashboards on are still moving, so the useful skill is knowing which
> ones are safe to alert on. Two things changed recently: OpenTelemetry
> started standardizing GenAI metric names, not just trace attributes, and the
> engines themselves grew new fields for prefix caching and speculative
> decoding. My rule is that I alert on engine-native names like the vLLM
> time-to-first-token histogram, because those are what actually get populated
> today, and I treat the OpenTelemetry gen_ai names as the cross-vendor join
> key I'll migrate to once they settle. As of mid-2026 nothing in that
> dedicated GenAI conventions repo is marked Stable and no major engine emits
> those names natively, so putting a page on them would be building on sand.

### OpenTelemetry GenAI semantic conventions reach the serving layer

OpenTelemetry has had GenAI *span* conventions (`gen_ai.request.model`,
`gen_ai.usage.input_tokens`, …) for a while, but the **metrics** side is newer
and, importantly, defines two distinct families
([GenAI metrics reference](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md)):

- **`gen_ai.server.*`** — serving-layer metrics, meant to be emitted by the
  inference server itself: `gen_ai.server.time_to_first_token`,
  `gen_ai.server.time_per_output_token`, and `gen_ai.server.request.duration`
  (all histograms, unit seconds). This is the vendor-neutral overlay for
  exactly the TTFT/TPOT/E2E triad this chapter has been building dashboards
  around.
- **`gen_ai.client.*`** — application-layer metrics, meant to be emitted by
  whatever code *calls* a model API: `gen_ai.client.token.usage`,
  `gen_ai.client.operation.duration`, and (for streaming)
  `gen_ai.client.operation.time_to_first_chunk` /
  `gen_ai.client.operation.time_per_output_chunk`.

The practical read: `server.*` is what you'd expect an inference engine or
gateway to export next to `vllm:time_to_first_token_seconds`; `client.*` is
what a RAG service or agent framework exports about its own calls out to that
engine. They answer different questions — "is the model server healthy" vs.
"is my application's use of the model server healthy" — and conflating them is
a common dashboard-design mistake once teams start adopting both.

The conventions are still moving fast and are **not yet Stable**. Version
history worth knowing
([state of OTel GenAI semconv, July 2026](https://john-hodge.com/blog/opentelemetry-genai-semantic-conventions/)):

- **v1.37.0** (Aug 2025) — `gen_ai.system` renamed to `gen_ai.provider.name`.
- **v1.40.0** (Feb 2026) — agent- and RAG-telemetry additions.
- **v1.41.0** (Apr 2026) — client/internal agent span splitting.
- **v1.42.0** (Jun 2026) — the GenAI conventions were fully deprecated out of
  the core `open-telemetry/semantic-conventions` repo and migrated to a
  dedicated project,
  [`open-telemetry/semantic-conventions-genai`](https://github.com/open-telemetry/semantic-conventions-genai)
  ([migration notice](https://opentelemetry.io/docs/specs/semconv/gen-ai/)).

As of July 2026 no GenAI-specific metric or attribute in that dedicated repo is
marked Stable, and the repo has no versioned releases of its own yet
([state of OTel GenAI semconv, July 2026](https://john-hodge.com/blog/opentelemetry-genai-semantic-conventions/)).
In practice this means: no major serving engine has switched its native
Prometheus metrics over to `gen_ai.server.*` names, so you still scrape
`vllm:`/`tgi_` metrics day to day, and you treat the OTel GenAI names as the
emerging cross-vendor join key to watch, not yet something to alert on
directly.

> **Saying it out loud.** The distinction that matters is server versus
> client. The gen_ai.server metrics are emitted by the inference server itself
> — is the model server healthy, what's its time to first token, its time per
> output token. The gen_ai.client metrics are emitted by whatever application
> calls a model API — is my RAG service's use of that server healthy, how many
> tokens did it burn. Teams blend them onto one dashboard and then can't tell
> whether the model server is slow or their own code is, which is a genuinely
> painful thirty minutes during an incident. And one line of history is worth
> knowing: the GenAI conventions were moved out of the core
> semantic-conventions repo into their own project in mid-2026 and still have
> no stable release, so treat them as direction, not dependency.

### vLLM's V1 metrics — what's new since the V0 engine

vLLM's rewritten V1 engine kept the core metric names this chapter already
covers, but the current design adds several fields that didn't exist in the
older API-server-only metrics endpoint
([vLLM metrics design, current](https://docs.vllm.ai/en/latest/design/metrics/),
[vLLM engine metrics reference](https://docs.vllm.ai/en/v0.9.2/api/vllm/engine/metrics.html)):

- **`vllm:cpu_cache_usage_perc`** — the CPU-side counterpart of
  `gpu_cache_usage_perc`, for deployments that swap KV blocks to host memory
  under pressure instead of only GPU HBM.
- **`vllm:cache_config_info`** — an Info metric (labels only, no useful value)
  that pins the exact cache configuration (block size, GPU/CPU block counts)
  a given process is running with, useful for correlating a dashboard change
  with a config change.
- **Prefix-cache counters replace the hit-rate gauge.** The old
  `vllm:gpu_prefix_cache_hit_rate` / `vllm:cpu_prefix_cache_hit_rate` gauges are
  deprecated in favor of `cache_query_hit` / `cache_query_total` counters —
  compute the ratio yourself with `rate(cache_query_hit[5m]) /
  rate(cache_query_total[5m])`, which behaves correctly across restarts and
  Prometheus aggregation (a rate of two counters composes; averaging a
  pre-computed gauge across replicas does not).
- **Speculative decoding is now implemented, not just planned.**
  `vllm:spec_decode_draft_acceptance_rate`, `vllm:spec_decode_efficiency`, and
  the counters `vllm:spec_decode_num_accepted_tokens_total` /
  `_num_draft_tokens_total` / `_num_emitted_tokens_total` let you watch whether
  a speculative-decoding config is actually paying for itself in practice.
- **Prefill and decode time are split**, as separate histograms
  (`vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds`),
  which is what makes the "Prefill vs. decode time" catalog row above possible
  without guessing from TTFT and TPOT alone.
- **`vllm:lora_requests_info`** — a gauge for multi-LoRA deployments, so you
  can see adapter-level request mix on a shared base model.
- Three older metrics are deprecated/removed and worth knowing so you don't
  chase ghosts in old dashboards: `vllm:num_requests_swapped`,
  `vllm:time_in_queue_requests` (duplicated `request_queue_time_seconds`), and
  an unimplemented `vllm:tokens_total`.

### TGI

TGI's metric surface (`tgi_request_duration`, `tgi_queue_size`, and friends,
covered in Mechanism 1 below) hasn't grown a comparable set of new fields in
this window; the ecosystem instead grew around it — there's now a community
Grafana dashboard purpose-built for TGI on Kubernetes
([TGI dashboard, Grafana Labs](https://grafana.com/grafana/dashboards/20246-text-generation-inference/))
that you can import rather than hand-build panels for `tgi_` metric names.

> **Saying it out loud.** The V1 change I'd actually bring up in an interview
> is the prefix-cache one, because it shows you understand Prometheus, not
> just vLLM. The old gauge handed you a hit rate directly; V1 replaced it with
> two counters — cache hits and cache queries — and you compute the ratio
> yourself. That's strictly better, because a rate of two counters composes
> correctly when you sum across replicas and it survives process restarts,
> whereas averaging a pre-computed hit-rate gauge across five replicas is
> arithmetically meaningless. The other additions worth naming are
> speculative-decoding acceptance rate, where anything under about 50% usually
> means the draft model is miscalibrated for your traffic, and the split of
> prefill time from decode time, which lets you tell "the prompts got longer"
> apart from "decode got slower" without guessing.

### DCGM exporter's current metric set — and a real gotcha

The default DCGM exporter field set, as documented by NVIDIA
([DCGM exporter docs](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html)),
groups into: clocks (`DCGM_FI_DEV_SM_CLOCK`, `_MEM_CLOCK`), thermals
(`_GPU_TEMP`, `_MEMORY_TEMP`), power (`_POWER_USAGE`,
`_TOTAL_ENERGY_CONSUMPTION`), utilization (`_GPU_UTIL`, `_MEM_COPY_UTIL`,
`_ENC_UTIL`, `_DEC_UTIL`), framebuffer memory (`_FB_USED`, `_FB_FREE`,
`_FB_RESERVED`), reliability (`_XID_ERRORS`,
`_UNCORRECTABLE_REMAPPED_ROWS`, `_CORRECTABLE_REMAPPED_ROWS`,
`_ROW_REMAP_FAILURE`), NVLink bandwidth, and the **DCP profiling group**
(`DCGM_FI_PROF_GR_ENGINE_ACTIVE`, `_PIPE_TENSOR_ACTIVE`, `_DRAM_ACTIVE`,
`_PCIE_TX_BYTES`, `_PCIE_RX_BYTES`) that this chapter leans on to see past a
misleadingly-high `GPU_UTIL` during decode.

The gotcha: **the DCP profiling metrics are not guaranteed on by default the
way they were in dcgm-exporter 3.x.** Two concrete, dated reports:

- Running GPU Operator v25.3.0 with dcgm-exporter v4.1.1-2, operators have hit
  `DCGM_FI_PROF_GR_ENGINE_ACTIVE: metric not enabled`
  ([NVIDIA/gpu-operator#1397](https://github.com/NVIDIA/gpu-operator/issues/1397)) —
  the profiling module has to be explicitly available/enabled on the driver
  and exporter side; it doesn't just show up because you upgraded.
- Metrics available by default in 3.x, like `DCGM_FI_PROF_PCIE_TX_BYTES`, have
  been reported missing after upgrading to dcgm-exporter 4.x
  ([NVIDIA/dcgm-exporter#513](https://github.com/NVIDIA/dcgm-exporter/issues/513)).

The operational takeaway: **after any dcgm-exporter or GPU Operator version
bump, re-verify with `curl -s http://<node>:9400/metrics | grep PROF`** before
trusting a dashboard panel that depends on `PIPE_TENSOR_ACTIVE` or
`DRAM_ACTIVE` — a silently-missing profiling metric reads as "no data," not as
an error, and a panel that quietly goes blank is easy to miss until the exact
moment you need it during an incident.

> **Saying it out loud.** DCGM is NVIDIA's GPU telemetry daemon; the exporter
> turns it into Prometheus metrics on port 9400, and it's how you see
> temperature, power, HBM usage, and ECC errors. The gotcha worth knowing is
> that the profiling group — the DCGM_FI_PROF fields like tensor-pipe-active —
> is not guaranteed to be on; it depends on the DCGM profiling module being
> enabled, and there are dated public reports of those fields disappearing
> after a routine dcgm-exporter or GPU Operator upgrade. That failure mode is
> nasty because a missing metric renders as "no data," not as an error, so the
> panel just goes quietly blank and you discover it during exactly the
> incident where you needed it. So after any exporter bump, curl the metrics
> endpoint and grep for PROF before you trust the panel.

### Building unified dashboards across the serving stack

The pattern that's emerged for tying engine metrics and GPU metrics into one
view is: one Prometheus with multiple scrape jobs (engine + DCGM + gateway/
router), one Grafana dashboard templated with `$model`/`$engine`/`$gpu`
variables, and — once the OTel GenAI conventions stabilize — those names as a
long-term cross-vendor join key.

vLLM's own reference deployment,
[`vllm-project/production-stack`](https://github.com/vllm-project/production-stack)
(launched January 2025; latest release `vllm-stack-0.1.11`, May 2026), ships
exactly this: a router in front of multiple vLLM engines, plus a Grafana
dashboard whose panels mix vLLM-specific series (available instance count,
E2E/TTFT latency distributions, active and pending requests) with GPU-facing
series (KV-cache utilization, KV-cache/prefix-cache hit rate) — all fed by one
Prometheus scraping both the router and the engines. A companion dashboard
covers LMCache (vLLM's disaggregated KV-cache backend) separately. If you don't
want to hand-roll the JSON in the next section, this repo — or the community
[`kubeai-project/kubeai` vLLM Grafana dashboard](https://github.com/kubeai-project/kubeai/blob/main/examples/observability/vllm-grafana-dashboard.json) —
is a reasonable starting point to fork.

> **Saying it out loud.** The pattern that's settled out is boring, and that's
> the point: one Prometheus with several scrape jobs — the engines, the DCGM
> exporters, the router — feeding one Grafana dashboard templated on model,
> engine, and GPU, so a single dashboard serves every replica. You don't have
> to hand-roll it either; vLLM's own production-stack repo ships a router plus
> a reference dashboard that already mixes engine series like TTFT
> distributions with GPU-facing series like KV-cache utilization. If I were
> asked to design this I'd say fork that and spend the saved time on alert
> tiering instead. The thing to avoid is a dashboard per team — that's how you
> end up with five different definitions of p95 in one org.

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
- `vllm:request_queue_time_seconds` — time spent waiting to be scheduled
- `vllm:request_prefill_time_seconds` — prefill portion of inference time
- `vllm:request_decode_time_seconds` — decode portion of inference time
- `vllm:request_prompt_tokens` — prompt length distribution
- `vllm:request_generation_tokens` — output length distribution
- `vllm:iteration_tokens_total` — tokens processed per scheduler step

**Gauges** (instantaneous system state):

- `vllm:num_requests_running` — sequences currently decoding
- `vllm:num_requests_waiting` — sequences queued (the saturation signal)
- `vllm:gpu_cache_usage_perc` — KV-cache utilization (**a fraction 0–1**, so
  multiply by 100 to get a percent — the name is misleading)
- `vllm:cpu_cache_usage_perc` — CPU-side KV-cache utilization, for deployments
  that swap blocks to host memory
- `vllm:lora_requests_info` — active adapter mix, for multi-LoRA serving
- `vllm:spec_decode_draft_acceptance_rate` / `vllm:spec_decode_efficiency` —
  speculative-decoding health, if enabled

**Counters** (cumulative — take `rate()` of these):

- `vllm:prompt_tokens_total` — prompt tokens processed
- `vllm:generation_tokens_total` — output tokens generated (throughput source)
- `vllm:request_success_total` — successful requests (has a `finished_reason`
  label so you can separate `stop` vs `length` vs `abort`)
- `vllm:num_preemptions_total` — cache-pressure evictions
- `vllm:cache_query_hit` / `vllm:cache_query_total` — prefix-cache hits vs.
  lookups; the current, correct way to compute prefix-cache hit rate (the
  older `gpu_prefix_cache_hit_rate` gauge is deprecated — see "The 2025–2026
  landscape" above for why a rate-of-counters beats an averaged gauge here)
- `vllm:spec_decode_num_accepted_tokens_total` / `_num_draft_tokens_total` /
  `_num_emitted_tokens_total` — speculative-decoding token accounting

Histograms are exposed as three series each: `_bucket` (cumulative, labelled by
`le`), `_sum`, and `_count`. You compute percentiles from `_bucket` and averages
from `_sum / _count`.

A handful of older metric names are deprecated or were never implemented —
`vllm:num_requests_swapped`, `vllm:time_in_queue_requests`, and
`vllm:tokens_total` — so if you inherit a dashboard built against an older
vLLM version, expect a few blank panels until you re-map them onto the current
names above.

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
names onto your canonical SLIs. If you don't want to hand-build a TGI
dashboard from scratch, the community-maintained
[TGI Grafana dashboard (ID 20246)](https://grafana.com/grafana/dashboards/20246-text-generation-inference/)
is a ready import for a Kubernetes deployment.

> **Saying it out loud.** The key point is that you don't instrument the
> model, you scrape the server — both vLLM and TGI already expose a Prometheus
> endpoint, so this is configuration, not code. What bites people is that the
> two engines don't agree on names. vLLM gives you a real time-to-first-token
> histogram; TGI ships no metric literally called TTFT, so you either
> approximate it from queue duration plus the prefill portion, or you capture
> first-token timing at the gateway. So step one on any new stack is a mapping
> table from engine-native names onto your canonical SLIs — otherwise p95
> quietly means something different depending on which engine served the
> request.

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
| `DCGM_FI_DEV_FB_RESERVED` | Framebuffer memory reserved by the driver, MiB |
| `DCGM_FI_DEV_POWER_USAGE` | Board power draw, watts |
| `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` | Cumulative energy draw, useful for a $/token energy view |
| `DCGM_FI_DEV_GPU_TEMP` | GPU die temperature, °C |
| `DCGM_FI_DEV_SM_CLOCK` | SM clock, MHz (watch for throttling) |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | Memory-copy engine utilization |
| `DCGM_FI_DEV_ENC_UTIL` / `_DEC_UTIL` | Video encode/decode engine utilization (irrelevant for text LLMs, relevant for multimodal) |
| `DCGM_FI_DEV_XID_ERRORS` | Hardware/driver XID error codes — a leading indicator of a dying GPU |
| `DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS` | HBM rows remapped after uncorrectable ECC errors — rising count means the GPU is degrading |
| `DCGM_FI_PROF_GR_ENGINE_ACTIVE` | Graphics/compute engine active ratio |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor-core pipe active ratio (real compute intensity) |
| `DCGM_FI_PROF_DRAM_ACTIVE` | Memory-bandwidth active ratio |
| `DCGM_FI_PROF_PCIE_TX_BYTES` / `_RX_BYTES` | Host-device PCIe transfer rate |

Three subtleties worth internalizing:

- **`DCGM_FI_DEV_GPU_UTIL` is a liar for LLM decode.** It reports "a kernel was
  scheduled," which is nearly always true during autoregressive decode even when
  the GPU is memory-bandwidth-bound and compute-idle. Use
  `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` and `DCGM_FI_PROF_DRAM_ACTIVE` to see whether
  you're compute-bound or bandwidth-bound.
- Every DCGM series carries a `gpu` (index) and usually `UUID`/`modelName`
  label, so on a multi-GPU node you aggregate or break down per device.
- **The `DCGM_FI_PROF_*` (DCP) group is not guaranteed to be present.** As
  covered in "The 2025–2026 landscape," profiling metrics depend on the DCGM
  profiling module being available and enabled, and reports of these fields
  silently missing after a dcgm-exporter upgrade are common enough to be worth
  a post-upgrade `curl | grep PROF` check rather than an assumption.

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

> **Saying it out loud.** Prometheus pulls, it doesn't receive: you hand it a
> list of targets and a scrape interval and it fetches slash-metrics on a
> schedule. For LLM serving that's three kinds of target — the inference
> engines on their API port, the DCGM exporter on 9400 on every GPU node, and
> your gateway or router. In Kubernetes you'd swap the static target list for
> service discovery so new replicas get scraped automatically instead of
> someone editing YAML at 2am. And keep the scrape interval in your head as a
> real limit: at 15 seconds, anything that spikes and resolves inside 15
> seconds is invisible to you, which is why a KV-cache threshold needs
> headroom rather than sitting at 100%.

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

> **Saying it out loud.** The one piece of PromQL I'd want to be able to write
> on a whiteboard is the percentile: histogram_quantile at 0.95 over a
> sum-by-le of the rate of the bucket series. And I'd explain every piece —
> rate because the buckets are counters, sum by le because that's how you
> merge histograms from every replica into one distribution, and
> histogram_quantile applied once at the very end. The mistake that shows up
> constantly is computing p95 per host and then averaging those p95s. That's
> arithmetically invalid — percentiles don't average — and it usually
> understates the tail, which is the exact number you're being paged about.

### Grafana

Build one dashboard per concern and template it with a `$model` /
`$engine` / `$gpu` variable so a single dashboard serves every replica:

- **Latency row** — TTFT p50/p95/p99, TPOT p95, E2E p95/p99 (time-series).
- **Throughput row** — req/s and tokens/s, with offered-vs-served overlaid.
- **Saturation row** — waiting sequences, running sequences, KV-cache %,
  preemption rate. This row is what tells you *why* latency moved.
- **Resource row** — GPU util, tensor-active, HBM used %, power, temp, SM clock
  from DCGM.
- **Cost row** — $/1k tokens and $/hr per replica.

Always plot percentiles as separate series; never a single "avg latency" line.
The next section, "Build it in practice — extended," turns this row list into
an actual importable dashboard sketch and a full alert runbook.

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

> **Saying it out loud.** The framing I'd use is: an SLI is what you measure,
> an SLO is what you promised, and an alert should fire when the promise is at
> risk — not when a number merely looks big. So I split alerts into two tiers.
> Symptom alerts, like TTFT p95 over the SLO or error rate climbing, page a
> human, because a user is feeling that right now. Leading indicators — queue
> depth building, KV cache above 90% — open a ticket, because they say a cliff
> is coming and buy you time to scale out. And everything gets a for-clause so
> the condition has to hold for several minutes; without that you're paging on
> one noisy scrape, which is how you train your on-call to ignore you.

---

## Build it in practice — extended

Mechanism 3 gave you the row layout; this section turns it into something you
can actually import and page against: a full panel list with the PromQL wired
in, a dashboard JSON sketch, and a runbook for the one alert that most
directly encodes the "LLM serving is different" lesson from this chapter — KV
cache saturation.

### The golden-signals dashboard — full panel list

| # | Row | Panel | Type | Query |
|---|---|---|---|---|
| 1 | Latency | TTFT p50/p95/p99 | Time series | `histogram_quantile(0.50/0.95/0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))` |
| 2 | Latency | TPOT p95 | Time series | `histogram_quantile(0.95, sum by (le) (rate(vllm:time_per_output_token_seconds_bucket[5m])))` |
| 3 | Latency | E2E p50/p95/p99 | Time series | `histogram_quantile(0.50/0.95/0.99, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket[5m])))` |
| 4 | Throughput | Requests/s by outcome | Stacked time series | `sum by (finished_reason) (rate(vllm:request_success_total[1m]))` |
| 5 | Throughput | Output tokens/s | Time series | `sum(rate(vllm:generation_tokens_total[1m]))` |
| 6 | Saturation | Waiting sequences | Time series + threshold line at 20 | `sum(vllm:num_requests_waiting)` |
| 7 | Saturation | Running sequences | Time series | `sum(vllm:num_requests_running)` |
| 8 | Saturation | KV-cache utilization % | Time series/gauge + threshold at 90 | `avg(vllm:gpu_cache_usage_perc) * 100` |
| 9 | Saturation | Preemptions/s | Time series | `sum(rate(vllm:num_preemptions_total[5m]))` |
| 10 | Saturation | Prefix-cache hit rate | Time series | `sum(rate(vllm:cache_query_hit[5m])) / sum(rate(vllm:cache_query_total[5m]))` |
| 11 | Resource | GPU util vs. tensor-active, per GPU | Time series, two series overlaid | `avg by (gpu) (DCGM_FI_DEV_GPU_UTIL)` and `avg by (gpu) (DCGM_FI_PROF_PIPE_TENSOR_ACTIVE) * 100` |
| 12 | Resource | HBM used % | Time series + threshold at 95 | `100 * DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)` |
| 13 | Resource | GPU temp / power | Time series | `DCGM_FI_DEV_GPU_TEMP`, `DCGM_FI_DEV_POWER_USAGE` |
| 14 | Cost | $ per 1k tokens | Stat panel | `sum(price_per_gpu_hour) / (sum(rate(vllm:generation_tokens_total[5m])) * 3.6)` |

Rows 1–5 are RED; rows 6–10 are the LLM-specific USE-saturation row that most
generic dashboards omit; rows 11–13 are USE-utilization/errors from DCGM; row
14 is the business translation. That ordering — latency, then throughput, then
saturation, then resource, then cost — mirrors how an on-call engineer should
actually read the dashboard during an incident: symptom first, cause last.

### Dashboard JSON sketch

A trimmed but structurally real Grafana dashboard JSON — enough to see the
templating variables and how a panel's `targets` wire to the PromQL above. In
practice you'd have 14 panels (per the table); this sketch shows the pattern
for one panel per row so you can extend it mechanically:

```json
{
  "title": "LLM Serving — Golden Signals",
  "schemaVersion": 39,
  "tags": ["llm", "vllm", "dcgm"],
  "templating": {
    "list": [
      { "name": "model", "type": "query", "datasource": "Prometheus",
        "query": "label_values(vllm:request_success_total, model)" },
      { "name": "engine", "type": "query", "datasource": "Prometheus",
        "query": "label_values(vllm:request_success_total, engine)" },
      { "name": "gpu", "type": "query", "datasource": "Prometheus",
        "query": "label_values(DCGM_FI_DEV_GPU_UTIL, gpu)" }
    ]
  },
  "panels": [
    {
      "id": 1, "title": "TTFT p50/p95/p99",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
      "targets": [
        { "expr": "histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{model=\"$model\"}[5m])))",
          "legendFormat": "p95" },
        { "expr": "histogram_quantile(0.50, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{model=\"$model\"}[5m])))",
          "legendFormat": "p50" }
      ]
    },
    {
      "id": 8, "title": "KV-cache utilization %",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 },
      "fieldConfig": { "defaults": { "thresholds": {
        "steps": [ { "value": null, "color": "green" },
                   { "value": 90, "color": "red" } ] } } },
      "targets": [
        { "expr": "avg(vllm:gpu_cache_usage_perc{model=\"$model\"}) * 100",
          "legendFormat": "KV cache %" }
      ]
    },
    {
      "id": 11, "title": "GPU util vs. tensor-active",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 8 },
      "targets": [
        { "expr": "avg by (gpu) (DCGM_FI_DEV_GPU_UTIL{gpu=~\"$gpu\"})",
          "legendFormat": "util (gpu {{gpu}})" },
        { "expr": "avg by (gpu) (DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{gpu=~\"$gpu\"}) * 100",
          "legendFormat": "tensor-active (gpu {{gpu}})" }
      ]
    },
    {
      "id": 14, "title": "$/1k tokens",
      "type": "stat",
      "gridPos": { "h": 4, "w": 6, "x": 12, "y": 8 },
      "targets": [
        { "expr": "sum(price_per_gpu_hour) / (sum(rate(vllm:generation_tokens_total{model=\"$model\"}[5m])) * 3.6)" }
      ]
    }
  ]
}
```

Remember this is Grafana JSON, not MathJax input — the `$model`/`$engine`/
`$gpu` here are Grafana template-variable interpolations, evaluated by Grafana
before the query ever reaches Prometheus.

### Runbook alert: `KVCacheSaturated`

This is the alert most worth having a written runbook for, because it is the
leading indicator specific to LLM serving that generic on-call runbooks don't
cover.

**Alert (from Mechanism 4):**

```promql
avg(vllm:gpu_cache_usage_perc) * 100 > 90
for: 5m
```

**Why 90%, not 100% or 75%?** vLLM's scheduler starts preempting running
sequences (or refusing to admit new ones) once it cannot allocate KV blocks for
the next scheduling step — it doesn't wait for the cache to be *literally*
full, because a burst of a few large requests can consume the remaining
headroom before the next scrape interval even lands. 90% leaves roughly 10% of
blocks — typically enough for one or two more average-sized sequences — as
shock absorber against normal traffic variance between your 15-second scrape
interval and the 5-minute alerting window. Set the threshold lower (75–80%) if
your traffic has high variance in prompt/output length; set it higher only if
you've measured that your specific workload never bursts past a gap that
small.

**First three diagnostic steps when this pages:**

1. **Check `num_requests_waiting` and `rate(num_preemptions_total[5m])` in the
   same window.** If both are also rising, this is genuine, ongoing cache
   pressure — go to remediation. If KV usage is high but queue and preemptions
   are flat, it may be a temporary hold from a burst that already passed;
   confirm before paging further.
2. **Check whether `request_prompt_tokens` / `request_generation_tokens`
   distributions shifted recently.** A new customer, a changed prompt
   template, or a longer default `max_tokens` inflates KV usage per request
   with no change in request *count* — this is a traffic-mix problem, not a
   capacity regression, and the fix is different (right-size `max_model_len`
   or `max_num_seqs`, not just "add replicas").
3. **Check the prefix-cache hit ratio
   (`cache_query_hit`/`cache_query_total`).** A drop here — often caused by a
   routing change that stopped sending same-prefix traffic to the same
   replica — means requests that used to reuse cached KV blocks are now
   recomputing them from scratch, inflating effective cache usage without any
   real increase in offered load.

**Remediation, roughly in order of speed:** shed or defer low-priority batch
traffic first (it's the cheapest lever); then scale out replicas if the router
supports fast rebalancing; then, if it's a traffic-mix issue, adjust
`max_num_seqs` / `max_model_len` or fix prefix-cache-aware routing rather than
just adding hardware against a problem hardware won't solve.

> **Saying it out loud.** If a KV-cache alert wakes me up I'm not immediately
> scaling out — I want to know which of three things it is. First I check
> whether queue depth and the preemption rate are rising too; if the cache is
> full but nothing's queueing or getting evicted, it was a burst that already
> passed. Second I check whether the prompt and output length distributions
> moved, because one new customer with much longer prompts inflates cache
> usage with zero change in request count — that's a traffic-mix problem and
> adding replicas is the wrong fix. Third I check the prefix-cache hit ratio,
> since a routing change that stops sending same-prefix traffic to the same
> replica makes you recompute KV blocks you used to get for free. And the
> reason the threshold is 90 rather than 100 is that the scheduler starts
> preempting the moment it can't allocate blocks for the next step, so you
> need a couple of average sequences' worth of headroom as a shock absorber.

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

> **Saying it out loud.** RED and USE are just two lenses and you want both on
> the wall. RED — rate, errors, duration — is the request's point of view,
> which is what users actually experience. USE — utilization, saturation,
> errors — is the resource's point of view, which tells you why. The
> LLM-specific twist lives in the saturation box: in a classic web service
> that's the CPU run queue, but here it's sequences waiting for KV-cache
> blocks. That's exactly the box generic dashboards leave empty, and it's the
> one that connects "p95 got worse" to "because the cache filled up."

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

**Where OTel GenAI semantic conventions fit in.** As covered in "The
2025–2026 landscape," the emerging `gen_ai.server.*` metric names
(`time_to_first_token`, `time_per_output_token`, `request.duration`) are
designed to be the vendor-neutral equivalent of exactly these vLLM span
attributes — the idea being that a trace exported by vLLM, a trace exported by
a different engine, and a trace exported by a hosted API could all carry the
same attribute names, so one Grafana/Tempo/Jaeger view works regardless of
which engine served the request. Since the conventions are still at
Development stability with no engine having adopted the `gen_ai.server.*`
names as its native metric names, treat this as the direction things are
heading rather than something to depend on for today's alerting — keep
alerting on the engine-native names (`vllm:...`, `tgi_...`) and treat
`gen_ai.*` attributes on your trace spans as a bonus cross-vendor label for
now.

> **Saying it out loud.** Metrics tell you that p99 is bad; a trace tells you
> where the time went for one specific slow request. So for a four-second
> request, a trace splits it three ways: 3.8 seconds of queue wait means scale
> out, a huge prefill means somebody sent an eight-thousand-token prompt, and
> slow decode means a batching or memory-bandwidth problem. Those are three
> completely different fixes and aggregate metrics can't tell them apart. The
> practical requirements are propagating the traceparent header from your
> gateway into the engine so the engine's spans nest under the user's request,
> and sampling — one to five percent plus always-sample-on-error — because
> full-rate tracing at scale becomes its own cost and cardinality problem.

---

## Production case studies & war stories

These are composite scenarios — the pattern shows up across enough real
on-call postmortems that they're worth walking through in detail, without
attaching them to a specific company. Both are directly traceable to the
concepts above.

### War story 1 — alert fatigue swallowed the real page

**Setup.** A team stood up an LLM-serving cluster by cloning their existing
web-service alerting pack: disk I/O latency, node network error rate, CPU
steal time, per-pod restart count, and about a dozen more — roughly 40 alert
rules total, most inherited wholesale and never re-tuned for GPU nodes. Two of
them — a disk-latency alert tuned for spinning-disk-era thresholds and a
node-network-error alert overly sensitive to normal NIC counter resets — fired
several times a week with no real incident behind them.

**Timeline.**

| Time | Event |
|---|---|
| T+0:00 | A new customer's traffic mix shifts to much longer prompts; `gpu_cache_usage_perc` crosses 90% and stays there |
| T+0:02 | `KVCacheSaturated` and `RequestQueueBuilding` both fire (correctly) |
| T+0:03 | The same on-call channel also receives the familiar disk-latency and network-error false alarms, as it does most nights |
| T+0:04 | On-call, trained by weeks of noise, acks the whole notification group without reading each one individually and goes back to sleep |
| T+0:45 | User complaints escalate through support; someone finally opens the dashboard and sees TTFT p95 at 9 seconds |
| T+0:52 | Mitigated by scaling out and shedding batch traffic |

**Lesson.** The KV-cache and queue alerts did their job — they fired within
minutes of the real onset. The failure was organizational: 40 minutes of
degraded service happened *after* a correct page, purely because the signal
was buried in noise. The fix wasn't a better KV-cache alert; it was deleting
or fixing the two chronically-noisy generic-infra alerts, splitting severities
so only true symptom-of-SLO-burn alerts page (leading-indicator alerts like
`RequestQueueBuilding` can reasonably be a ticket, not a page, if a
higher-severity symptom alert also exists), and reviewing "alerts that fired
in the last 30 days with no action taken" on a regular cadence. Alert volume
is itself a metric worth graphing.

> **Saying it out loud.** The lesson here isn't a missing metric — the
> KV-cache and queue alerts fired correctly, within two minutes of onset. The
> failure was that they landed in a channel that also got two chronically
> noisy inherited alerts most nights, so the on-call acked the whole group
> without reading it and went back to sleep. Forty minutes of degraded service
> happened after a correct page. So the fix wasn't a better alert, it was
> deleting the noisy ones, splitting severities so only symptom alerts page,
> and reviewing "alerts that fired in the last thirty days with no action
> taken" on a regular cadence. Alert volume is itself a metric worth graphing.

### War story 2 — the metric that was already there, just not on a dashboard

**Setup.** A team ran vLLM in production with a Grafana dashboard built early
in the project, before the KV-cache row existed. It had latency percentiles
and `DCGM_FI_DEV_GPU_UTIL`, which — per this chapter's warning about that
metric — read a steady ~60% and looked comfortably "healthy." Nobody had gone
back to add `vllm:gpu_cache_usage_perc` or `vllm:num_requests_waiting` once
those became available; the engine was already emitting them, they just
weren't graphed or alerted on.

**Timeline.**

| Time | Event |
|---|---|
| T-3:00 | A product change increases average prompt length roughly 3x for a subset of traffic |
| T-3:00 | `gpu_cache_usage_perc` (ungraphed) climbs past 90% and starts triggering silent preemptions |
| T-2:50 → T-0:10 | TTFT p95 creeps from ~900 ms to ~6 s over roughly three hours, visible only if someone happened to look at the latency panel |
| T-0:10 | Support tickets about "slow responses" reach a volume that triggers a manual investigation |
| T-0:05 | An engineer, following this chapter's advice, checks `gpu_cache_usage_perc` for the first time in the incident and finds it pinned at 97% |
| T+0:00 | Root cause identified: cache pressure from the longer prompts, not a GPU or infra problem; mitigated by reducing `max_num_seqs` and scaling out |

**Lesson.** This wasn't a missing-instrumentation problem — vLLM had exported
the KV-cache gauge the entire time. It was a dashboarding-and-alerting
discipline problem: the leading indicator existed but nobody looked at it
until after three hours of silent degradation, because `GPU_UTIL` "looked
fine" and nobody had wired an alert to the metric that actually predicts an
LLM-serving latency cliff. The postmortem action item was almost embarrassingly
simple — add the KV-cache and queue-depth rows from Mechanism 3/"Build it in
practice" above and wire the `KVCacheSaturated` alert — which is exactly why
this chapter treats those two metrics as non-optional rather than nice-to-have.

> **Saying it out loud.** This one is the opposite failure and it's more
> common than people admit: the metric existed the whole time, the engine was
> exporting it, nobody had put it on a dashboard. GPU utilization read a
> comfortable sixty percent so everything looked fine, while KV cache sat at
> 97% and TTFT crept from about 900 milliseconds to six seconds over three
> hours. Nobody noticed until support tickets piled up. The remediation was
> embarrassing in its simplicity — add the KV-cache and queue-depth panels,
> wire the saturation alert — which is exactly why I treat those two as
> non-optional rather than nice-to-have.

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
  alert on them. (See "War story 2" above for exactly how expensive this gap
  gets in practice.)

- **Trusting `DCGM_FI_DEV_GPU_UTIL`.** It says "a kernel ran," not "the GPU did
  useful work." During decode it sits near 100% while the device is
  memory-bandwidth-bound and compute-starved. Cross-check with
  `PIPE_TENSOR_ACTIVE` / `DRAM_ACTIVE` before concluding you're compute-bound.

- **Assuming DCGM profiling metrics are always present.** As covered in "The
  2025–2026 landscape," `DCGM_FI_PROF_*` fields depend on the DCGM profiling
  module and have been reported missing after routine dcgm-exporter/GPU
  Operator upgrades. A panel built on `PIPE_TENSOR_ACTIVE` that goes quietly
  blank after an unrelated infra upgrade is a trap during exactly the incident
  where you need it — verify with a `curl | grep PROF` after any such upgrade.

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

- **No cost visibility.** If nobody graphs $/1k tokens, efficiency regressions
  (a bad batch-size change, an underutilized replica) go unnoticed until the
  cloud bill arrives. Cost is the metric leadership reads; derive it from
  tokens/s and GPU price.

- **Blind spot between gateway and engine.** If you only scrape the engine you
  miss load-balancer queuing and network time. Measure TTFT at the edge too, and
  reconcile the two.

- **Alert-pack inheritance without re-tuning.** Cloning a generic web-service
  alert pack onto GPU-serving infrastructure, unchanged, produces exactly the
  alert-fatigue failure in "War story 1" above — every alert rule you carry
  over should be re-justified against this workload, not assumed correct
  because it worked for a REST API.

> **Saying it out loud.** If I had to name the three pitfalls that bite
> hardest: alerting on averages, having no saturation metrics at all, and
> trusting GPU utilization. A mean of 400 milliseconds can hide a p99 of
> twelve seconds, and the tail is who churns. No queue or KV-cache metric
> means zero warning before the latency cliff — everything looks fine right up
> until TTFT triples. And GPU util says "a kernel was scheduled," not "the GPU
> did useful work"; during decode it sits near 100% while the card is
> memory-bandwidth-starved. There's a purely arithmetic one too: you cannot
> average p95s across replicas, you have to merge the bucket series first and
> take the quantile once.

---

## Production checklist & interview mastery — what an interviewer probes

### Explain the golden signals for LLM serving in 60 seconds

> "A REST service has one latency; an LLM server has three: time-to-first-token,
> which is your prefill and queue cost and drives 'is it thinking'; inter-token
> latency, which is your decode cost and drives streaming feel; and end-to-end,
> which is roughly TTFT plus output-length times inter-token latency. On top of
> that, the resource that actually constrains an LLM server isn't CPU or
> connections, it's a fixed pool of GPU memory holding the KV cache. So the two
> signals that predict a latency cliff *before users feel it* are queue depth —
> requests waiting to be admitted — and KV-cache utilization — how full that
> memory pool is. GPU utilization alone is misleading, because during decode
> it reads high even when the GPU is memory-bandwidth-bound, not compute-bound.
> So: RED for the request surface — rate, errors, and the three durations — and
> USE for the resource, where saturation is queue-plus-cache, not CPU run-queue.
> Alert on SLO burn for the symptoms, and on queue/cache for the leading
> indicators, so you get paged before the cliff, not after."

### Q&A

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
8. **"What's your cost metric?"** — $/1k tokens derived from GPU $/hr and
   `rate(generation_tokens_total)`, tracked per model/replica.
9. **"vLLM's V1 engine changed some metric names — what, and why should I
   care?"** — The prefix-cache hit-rate gauge was replaced by
   `cache_query_hit`/`cache_query_total` counters (rate-of-counters composes
   correctly across replicas and restarts; an averaged gauge doesn't);
   speculative-decoding metrics and a prefill/decode time split were added.
   Caring about this signals you keep dashboards current as engines evolve
   instead of alerting on names that quietly stopped being populated.
10. **"What's the difference between `gen_ai.server.*` and `gen_ai.client.*`
    in the OpenTelemetry GenAI conventions?"** — `server.*` is emitted by the
    inference server itself (serving-layer TTFT/TPOT/duration); `client.*` is
    emitted by the application code calling a model API (token usage,
    operation duration). Conflating the two produces a dashboard that can't
    tell you whether the *model server* or *your service's use of it* is slow.
11. **"Are those conventions something you'd build alerts on today?"** — No;
    as of mid-2026 they're still Development-stability with no versioned
    release and no major engine has adopted them as native metric names. Track
    them as the emerging cross-vendor join key, keep alerting on engine-native
    names.
12. **"How would you build one dashboard across vLLM and GPU metrics?"** — One
    Prometheus scraping both the engine `/metrics` and DCGM exporter on 9400,
    one Grafana dashboard templated on `$model`/`$engine`/`$gpu`, rows ordered
    latency → throughput → saturation → resource → cost. Bonus: cite that
    `vllm-project/production-stack` ships exactly this as a reference.
13. **"Tell me about a monitoring incident and what you changed."** — Use the
    alert-fatigue or ungraphed-KV-cache story above: a correct alert or a
    correct metric existed, but noise or dashboard neglect delayed the
    response; the fix was alert hygiene / dashboard discipline, not new
    instrumentation.
14. **"Why might a DCGM profiling metric like `PIPE_TENSOR_ACTIVE` be missing
    on a node you just upgraded?"** — The DCP profiling group depends on the
    DCGM profiling module being enabled; it isn't guaranteed on by default the
    way it was in older exporter versions, and there are dated public reports
    of exactly this after dcgm-exporter/GPU Operator upgrades. Verify with a
    `/metrics | grep PROF` check post-upgrade rather than assuming.
15. **"What would you NOT alert on, and why?"** — Raw resource gauges without
    context (e.g., bare GPU util), anything without a `for:` debounce, and
    anything inherited from a generic web-service pack that hasn't been
    re-justified for GPU-serving traffic patterns.
16. **"How do percentiles fail you if you're not careful?"** — Averaging
    already-computed p95s across replicas is mathematically wrong; you must
    sum `_bucket` series first, then take the quantile. Also: window size
    trades off noise against incident-detection lag.

### System design prompt: "Design the monitoring stack for a new inference platform"

A common senior-level prompt: *"We're launching a new inference platform
serving several open-weight models across a few hundred GPUs, multi-tenant,
mixing interactive chat and best-effort batch traffic. Design the monitoring
stack."* A strong answer walks through layers, not just tool names:

```
                     ┌───────────────────────────┐
   users/apps  ───▶  │   Gateway / router        │──▶ traces (OTLP, sampled)
                     │  (traceparent propagate)  │
                     └─────────────┬─────────────┘
                                   │ scraped
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                      ▼
     ┌────────────────┐  ┌──────────────────┐   ┌──────────────────┐
     │ vLLM / TGI      │  │  DCGM exporter    │   │ Gateway metrics  │
     │ engines (:8000) │  │  (:9400 / node)   │   │ (req/s, 4xx/5xx) │
     └────────┬────────┘  └─────────┬─────────┘   └─────────┬────────┘
              └──────────────┬──────┴──────────────────────┘
                              ▼
                     ┌──────────────────┐
                     │   Prometheus      │── alert rules (SLO burn + leading indicators)
                     │ (multi scrape job)│
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐        ┌────────────────────┐
                     │     Grafana       │        │  Alertmanager      │
                     │ $model/$engine/   │        │ severity routing:  │
                     │ $gpu templated    │        │ page vs ticket     │
                     └──────────────────┘        └────────────────────┘
```

Points a strong candidate hits, roughly in order:

- **Separate SLIs for interactive vs. batch traffic** — they have different
  SLOs (batch tolerates queueing; chat doesn't), so route them to different
  alert thresholds or even different metric label values (`priority=interactive`
  vs `priority=batch`) rather than one blended TTFT p95.
- **Multi-tenant labeling without cardinality blowup** — tenant/customer as a
  label is tempting and dangerous; either keep it low-cardinality (a handful of
  tier buckets) or push per-tenant detail to logs/traces instead of Prometheus
  labels.
- **Scrape topology** — one Prometheus (or a federated pair for scale) hitting
  engine `/metrics`, DCGM `:9400` per GPU node, and the gateway; Kubernetes
  service discovery over static targets at this scale.
- **Alert tiering** — SLO-burn symptom alerts page; queue/cache/HBM
  leading-indicator alerts open a ticket unless a symptom alert is also firing,
  which avoids the alert-fatigue failure mode above.
- **Tracing sampling strategy** — low fixed-rate sampling plus always-sample-on
  -error, propagated `traceparent` end to end, because at a few hundred GPUs
  100%-sampled traces are a cost and cardinality problem of their own.
- **Cost as a first-class row**, not an afterthought — $/1k tokens per model,
  visible to more than just the on-call engineer.
- **A plan for engine-version drift** — new engine versions rename/deprecate
  metrics (see the vLLM V1 changes above); dashboards need an owner who updates
  them when that happens, not a "set it and forget it" assumption.

> **Saying it out loud.** For a prompt like this I'd answer in layers rather
> than naming tools. The gateway emits traces and its own request metrics;
> engines and DCGM exporters get scraped by one Prometheus, or a federated
> pair at a few hundred GPUs; Grafana templated on model, engine and GPU;
> Alertmanager doing severity routing. Then I'd hit the two things that make
> it a senior answer. Interactive and batch traffic need separate SLIs,
> because batch tolerates queueing and chat doesn't, so a blended TTFT p95
> tells you nothing. And multi-tenant labeling is a cardinality trap — tenant
> identity goes into logs and traces, not into a Prometheus label, unless it's
> bucketed down to a handful of tiers. I'd also name an owner for metric-name
> drift, because engines rename metrics between versions and an unowned
> dashboard quietly rots.

### Red flags vs. green flags

| Signal | Red flag | Green flag |
|---|---|---|
| Latency dashboard | Single "avg latency" line | Separate TTFT/TPOT/E2E percentile series |
| Primary GPU signal | Alerts on `GPU_UTIL` alone | Cross-checks `PIPE_TENSOR_ACTIVE`/`DRAM_ACTIVE` before concluding compute-bound |
| Saturation | No queue or KV-cache metric graphed at all | Queue depth and KV-cache % both graphed and alerted |
| Alert design | Every alert pages, no `for:` debounce | Symptom alerts page, leading indicators ticket, `for:` on everything |
| Alert provenance | Inherited wholesale from a generic web-service pack | Every rule re-justified for GPU-serving traffic patterns |
| Percentile math | Averages pre-computed p95s across replicas | Sums `_bucket` series first, then takes the quantile |
| Cost visibility | No $/1k tokens metric anywhere | Cost row on the same dashboard as latency |
| Engine-version hygiene | Dashboard built once, never revisited across engine upgrades | Someone owns re-mapping deprecated/renamed metrics (e.g. vLLM V1 changes) after upgrades |
| Tracing | No propagated `traceparent`; traces (if any) don't nest under the user request | `traceparent` propagated gateway → engine; sampled + always-sample-on-error |
| GenAI semantic conventions | Alerting depends on `gen_ai.server.*` names today | Treated as an emerging cross-vendor join key, not yet load-bearing |

---

## Further reading

- vLLM — Production Metrics: <https://docs.vllm.ai/en/v0.6.1/serving/metrics.html>
- vLLM — Metrics design (current): <https://docs.vllm.ai/en/latest/design/metrics/>
- vLLM — Engine metrics API reference: <https://docs.vllm.ai/en/v0.9.2/api/vllm/engine/metrics.html>
- vLLM — OpenTelemetry example: <https://docs.vllm.ai/en/v0.9.0/examples/online_serving/opentelemetry.html>
- vLLM production stack (reference deployment + Grafana dashboards): <https://github.com/vllm-project/production-stack>
- kubeai — example vLLM Grafana dashboard JSON: <https://github.com/kubeai-project/kubeai/blob/main/examples/observability/vllm-grafana-dashboard.json>
- TGI — Metrics reference: <https://huggingface.co/docs/text-generation-inference/main/en/reference/metrics>
- TGI — Community Grafana dashboard (ID 20246): <https://grafana.com/grafana/dashboards/20246-text-generation-inference/>
- NVIDIA DCGM exporter (GitHub): <https://github.com/NVIDIA/dcgm-exporter>
- NVIDIA DCGM exporter docs: <https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html>
- NVIDIA GPU Operator — DCGM profiling metric not enabled (real-world gotcha): <https://github.com/NVIDIA/gpu-operator/issues/1397>
- NVIDIA dcgm-exporter — 4.x vs. 3.x default metric changes (real-world gotcha): <https://github.com/NVIDIA/dcgm-exporter/issues/513>
- OpenTelemetry GenAI semantic conventions (dedicated repo): <https://github.com/open-telemetry/semantic-conventions-genai>
- OpenTelemetry GenAI metrics reference (`gen_ai.server.*` / `gen_ai.client.*`): <https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md>
- OpenTelemetry blog — "Inside the LLM Call: GenAI Observability with OpenTelemetry" (2026): <https://opentelemetry.io/blog/2026/genai-observability/>
- OpenTelemetry — GenAI semantic conventions migration notice: <https://opentelemetry.io/docs/specs/semconv/gen-ai/>
- The state of the OpenTelemetry GenAI semantic conventions, July 2026 (version timeline): <https://john-hodge.com/blog/opentelemetry-genai-semantic-conventions/>
- Prometheus — Querying / `histogram_quantile`: <https://prometheus.io/docs/prometheus/latest/querying/functions/#histogram_quantile>
- Prometheus — Alerting rules: <https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/>
- Grafana — The RED Method (Tom Wilkie): <https://grafana.com/blog/the-red-method-how-to-instrument-your-services/>
- Brendan Gregg — The USE Method: <https://www.brendangregg.com/usemethod.html>
- Google SRE Workbook — Alerting on SLOs: <https://sre.google/workbook/alerting-on-slos/>
- OpenTelemetry — Documentation: <https://opentelemetry.io/docs/>
