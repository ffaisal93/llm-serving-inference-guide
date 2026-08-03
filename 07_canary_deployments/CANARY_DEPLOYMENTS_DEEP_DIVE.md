# Canary Deployments for Model Serving

> Safely rolling out new models and versions — when the thing you are canarying can pass every infra gate and still be *worse*.

## Why this matters

Shipping application code and shipping a new model version look identical from thirty thousand feet: build an artifact, put it behind a service, shift traffic, watch dashboards. They are not the same problem.

A web service is mostly *right or wrong*. It returns a 200 or a 500, it is fast or slow, and your infrastructure metrics — error rate, p95 latency, saturation — catch essentially every regression that matters. A canary that watches those four numbers is a very good canary.

A model is *right, wrong, or plausibly-wrong-in-a-way-that-looks-right*. A new checkpoint can return HTTP 200 on every request, at lower latency than the old one, while quietly hallucinating more, refusing more legitimate prompts, drifting in tone, or regressing on your hardest 5% of inputs. Every infra gate is green. Your users are unhappy. This is the single most important thing to internalize in this chapter:

**Model regressions live in the response *body*, not in the response *envelope*. Infra canaries only watch the envelope.**

So model serving needs everything a normal progressive-delivery pipeline has — traffic splitting, automated analysis, progressive promotion, automatic rollback — *plus* a quality gate that reads the body. The rest of this chapter builds that pipeline from the bottom up and then shows you the failure modes that bite teams who forget the italicized sentence above.

---

## Core intuition: models fail in ways infra canaries don't catch

Picture two versions of a summarization model behind a gateway. You send 10% of live traffic to `v2`. Over an hour you observe:

| Signal | v1 (stable) | v2 (canary) | Infra verdict |
|---|---|---|---|
| HTTP 5xx rate | 0.02% | 0.02% | ✅ pass |
| p95 latency | 480 ms | 410 ms | ✅ pass (faster!) |
| Pod restarts / OOMs | 0 | 0 | ✅ pass |
| Throughput | 220 rps | 235 rps | ✅ pass |
| **Groundedness / factuality** | 0.91 | **0.78** | ❌ **regression** |
| **Refusal rate on valid prompts** | 1.2% | **6.5%** | ❌ **regression** |

A classic canary — Flagger or Argo Rollouts watching request-success-rate and request-duration — **promotes this rollout**. It is faster and just as reliable by every number it knows how to read. The two bottom rows are invisible to it because they require *scoring the generated text*, which is not a Prometheus counter you get for free.

The mental model:

- **Infra metrics** answer "did the request complete correctly?" — cheap, real-time, always available.
- **Quality metrics** answer "was the answer good?" — expensive, often delayed, and specific to *your* task.

Canary analysis for models = the union of both. If your pipeline only has the first, you have built a very sophisticated way to ship regressions with confidence.

---

## The four strategies and when each fits model serving

Before mechanisms, get the taxonomy straight. All four move traffic from an old version to a new one; they differ in *how much blast radius* a bad version gets and *how fast you can undo it*.

- **Rolling update** — replace old pods with new ones a few at a time. There is no "old vs new" concept at the traffic layer; once a pod is new, it serves real users. This is the Kubernetes Deployment default.
- **Blue-green** — stand up the full new version (green) alongside the full old version (blue), test green out-of-band, then flip 100% of traffic at once. Instant cutover, instant rollback (flip back), but you pay for two full fleets during the overlap.
- **Canary** — run the new version at small scale, send it a *slice* of real traffic (1% → 5% → 25% → …), analyze, and promote in steps. Small blast radius, gradual confidence.
- **Shadow (mirror)** — send the new version a *copy* of real traffic but **discard its responses**. Users never see canary output. Pure evaluation, zero user risk.

### Strategy comparison

| Strategy | User blast radius if bad | Rollback speed | Extra cost | Catches quality regressions? | Best fit for models |
|---|---|---|---|---|---|
| **Rolling** | Grows as pods replace; hard to bound | Slow (roll back = another rollout) | ~none | No — new pods serve users immediately | Low-risk config bumps, sidecar updates |
| **Blue-green** | 100% at the instant of flip | Instant (flip traffic back) | High (2× fleet during overlap) | Only if you gate the flip on an eval suite | Big/atomic version jumps where partial-mix is unacceptable |
| **Canary** | Bounded to the canary weight (e.g. 5%) | Fast (set weight → 0) | Moderate (small extra fleet) | Only if analysis includes a quality gate | **The default for model rollouts** |
| **Shadow** | **Zero** (responses discarded) | N/A (no user traffic to roll back) | High (full 2nd inference path, doubled GPU) | Yes, offline — no user exposure | Pre-canary validation of risky checkpoints |

Rules of thumb for model serving:

- **Never** roll out a new model with a bare rolling update. You lose the old/new traffic distinction exactly when you most need it, and GPU pods are slow to spin up/down so a "quick" rollback isn't quick.
- **Canary is the workhorse.** Small weight, automated analysis on infra *and* quality, progressive promotion.
- **Shadow first for scary changes** (new architecture, new quantization, new base model). It gives you production-distribution eval data with zero user exposure — then canary the survivors.
- **Blue-green when the mix itself is the problem** — e.g. a prompt-format or tokenizer change where having v1 and v2 answer the same *conversation* would be incoherent, so you want an atomic switch gated on a full eval run.

---

## Mechanism 1: Traffic splitting

Canary and shadow both need to route a *fraction* of requests somewhere. There are four common layers to do it, from crudest to most precise.

### 1a. Kubernetes Service — replica-ratio splitting (crude, avoid)

The oldest trick: two Deployments (`model-v1`, `model-v2`) sharing one Service via a common label selector. The Service load-balances across all matching pods, so the traffic split ≈ the replica ratio. Want 10% canary? Run 9 stable pods and 1 canary pod.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: model
spec:
  selector:
    app: model        # matches BOTH v1 and v2 pods
  ports:
  - port: 80
    targetPort: 8000
```

Why this is bad for models:

- Weight is quantized by replica count. A GPU pod might be an entire A100; you cannot cheaply run "0.5 of one" to get 5%.
- Weight and capacity are coupled — you can't send 1% of traffic to a canary that has 3 replicas for latency headroom.
- No session affinity, no header-based routing, no clean rollback primitive.

Use it only for a quick-and-dirty test. For anything real, split at the mesh/gateway layer.

### 1b. Istio VirtualService — weighted routing

Istio decouples weight from replica count. A `DestinationRule` defines subsets by label; a `VirtualService` assigns weights that sum to 100.

```yaml
apiVersion: networking.istio.io/v1
kind: DestinationRule
metadata:
  name: model
spec:
  host: model
  subsets:
  - name: v1
    labels: { version: v1 }
  - name: v2
    labels: { version: v2 }
---
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata:
  name: model
spec:
  hosts:
  - model
  http:
  - route:
    - destination:
        host: model
        subset: v1
      weight: 90
    - destination:
        host: model
        subset: v2
      weight: 10
```

To advance the canary you edit the two `weight` values (90/10 → 75/25 → 0/100). To roll back, set `v2` to `0`. This is the primitive Argo Rollouts and Flagger drive for you (below) — they rewrite these weights automatically.

### 1c. Gateway API — the vendor-neutral successor

Gateway API (`HTTPRoute`) does the same weighting in a mesh-agnostic way. Note the docs' precise wording: **weight is a proportional split, not a percentage** — the sum of weights in a rule is the denominator.

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: model-split
spec:
  parentRefs:
  - name: model-gateway
  rules:
  - backendRefs:
    - name: model-v1
      port: 8000
      weight: 90
    - name: model-v2
      port: 8000
      weight: 10
```

`90 + 10 = 100` so v2 gets 10%. If you'd written `9` and `1`, v2 still gets 10% — the ratio is what matters. Gateway API is where new tooling is converging; prefer it for greenfield.

### 1d. Header / session-based routing (for stateful serving)

Weighted splitting assumes requests are independent. LLM chat sessions are not — a multi-turn conversation must hit the *same* model version, or turn 3 answers in v2's voice after turns 1–2 were v1's. Route by a stable key instead:

```yaml
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"     # opt-in cohort, internal users, etc.
    route:
    - destination: { host: model, subset: v2 }
  - route:                   # everyone else
    - destination: { host: model, subset: v1 }
```

More on session pinning in Failure Modes.

---

## Mechanism 2: Automated canary analysis

Manually staring at Grafana while you bump weights doesn't scale and doesn't fire at 3 a.m. Automated analysis makes the promote/rollback decision from metrics. Two dominant tools: **Argo Rollouts** and **Flagger**.

The shape of both:

1. You declare **steps** (weights + pauses) and **metrics with thresholds**.
2. The controller shifts traffic to the first step.
3. At each step it queries metrics (usually Prometheus) over an interval, a number of times.
4. If a metric violates its condition too many times → **abort and roll back**.
5. If all steps pass → **promote** (canary becomes stable).

### Argo Rollouts: AnalysisTemplate

An `AnalysisTemplate` is a reusable metric-check bundle. This one watches **success rate** and **p95 latency** from Istio's Prometheus metrics:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: infra-metrics
spec:
  args:
  - name: service-name
  metrics:
  - name: success-rate
    interval: 1m
    count: 5                       # take 5 measurements
    successCondition: result[0] >= 0.99
    failureLimit: 2                # allow 2 bad reads before aborting
    provider:
      prometheus:
        address: http://prometheus.istio-system:9090
        query: |
          sum(irate(istio_requests_total{
            destination_service=~"{{args.service-name}}",
            response_code!~"5.."}[1m]))
          /
          sum(irate(istio_requests_total{
            destination_service=~"{{args.service-name}}"}[1m]))
  - name: p95-latency
    interval: 1m
    count: 5
    successCondition: result[0] <= 500   # milliseconds
    failureLimit: 2
    provider:
      prometheus:
        address: http://prometheus.istio-system:9090
        query: |
          histogram_quantile(0.95,
            sum(irate(istio_request_duration_milliseconds_bucket{
              destination_service=~"{{args.service-name}}"}[1m]))
            by (le))
```

Field semantics that trip people up:

- `interval` — how often to run the query.
- `count` — how many times total; the analysis runs `count × interval` before it can succeed.
- `successCondition` / `failureCondition` — a boolean expression over `result` (the query's returned vector). Provide one or the other.
- `failureLimit` — how many failed measurements are tolerated before the whole AnalysisRun fails and triggers rollback. `failureLimit: 0` means one bad read aborts.

### Wiring it into a Rollout

The `Rollout` object replaces your `Deployment`. Its canary `steps` interleave `setWeight`, `pause`, and `analysis`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: model
spec:
  replicas: 6
  selector:
    matchLabels: { app: model }
  template:
    metadata:
      labels: { app: model }
    spec:
      containers:
      - name: server
        image: registry.example.com/model:v2
        ports:
        - containerPort: 8000
        resources:
          limits: { nvidia.com/gpu: 1 }
  strategy:
    canary:
      canaryService: model-canary      # Service pointing only at canary pods
      stableService: model-stable      # Service pointing only at stable pods
      trafficRouting:
        istio:
          virtualService:
            name: model
            routes: [primary]
      steps:
      - setWeight: 5
      - pause: { duration: 10m }
      - analysis:
          templates:
          - templateName: infra-metrics
          args:
          - name: service-name
            value: model-canary.default.svc.cluster.local
      - setWeight: 25
      - pause: { duration: 10m }
      - analysis:
          templates:
          - templateName: infra-metrics
          args:
          - name: service-name
            value: model-canary.default.svc.cluster.local
      - setWeight: 50
      - pause: { duration: 30m }
      - setWeight: 100
```

What happens on `kubectl apply` with a new image:

1. Argo creates canary pods, points `model-canary` at them, and sets the VirtualService to 5% canary / 95% stable.
2. Waits 10m, then runs `infra-metrics` five times over five minutes.
3. Any check fails past `failureLimit` → **weight snapped back to 0, rollout marked Degraded, canary pods torn down.** Automatic rollback.
4. All pass → advance to 25%, repeat, then 50%, then 100%. At 100% the canary ReplicaSet becomes stable.

This is a **correct, faster, greener** rollout of the bad summarizer from the intuition section — because `infra-metrics` never looks at output quality. Now we fix that.

### Adding a model-quality gate

The quality gate is *just another metric in the AnalysisTemplate* — the trick is where the number comes from. Three patterns, cheapest to strongest:

**Pattern A — proxy signals already in Prometheus.** Some quality signals *are* cheap counters if your server emits them: refusal rate, empty-completion rate, average output token count (a proxy for truncation/degeneration), guardrail-filter trigger rate, mean logprob. Gate on those directly:

```yaml
  - name: refusal-rate
    interval: 2m
    count: 5
    failureCondition: result[0] > 0.03      # >3% refusals on valid prompts = bad
    failureLimit: 1
    provider:
      prometheus:
        address: http://prometheus.istio-system:9090
        query: |
          sum(irate(model_refusals_total{version="canary"}[2m]))
          /
          sum(irate(model_requests_total{version="canary"}[2m]))
```

**Pattern B — an online judge/eval job that writes a gauge.** Run an evaluator (LLM-as-judge, a reward model, or a reference-based scorer on prompts that have known-good answers) against a sample of *canary* responses, and have it push a score to Prometheus (Pushgateway) or an HTTP metrics endpoint. Then:

```yaml
  - name: quality-score
    interval: 5m
    count: 4
    successCondition: result[0] >= 0.85     # judge score, 0..1
    failureLimit: 1
    provider:
      prometheus:
        address: http://prometheus.istio-system:9090
        query: avg_over_time(canary_quality_score[5m])
```

**Pattern C — a web/job provider that runs an eval suite synchronously.** Argo Rollouts also supports `web` and `job` metric providers. A `job` provider spins up a Kubernetes Job that runs your offline eval harness against the canary endpoint and exits non-zero on regression; a `web` provider hits an eval service that returns JSON you assert on. Use these when the eval is heavy (a full benchmark set) and you want it as a hard gate before promoting past, say, 25%.

```yaml
  - name: eval-suite
    provider:
      job:
        spec:
          template:
            spec:
              containers:
              - name: eval
                image: registry.example.com/eval-harness:latest
                args: ["--endpoint", "http://model-canary:8000", "--suite", "regression-v3"]
              restartPolicy: Never
          backoffLimit: 0
```

Reference this template alongside `infra-metrics` in a later canary step so quality is a *blocking* condition, not an afterthought. This closes the loop: the fast/green/worse summarizer now fails `quality-score` at 5% and rolls back automatically.

> Tie-in to evaluation: these gates are only as good as the eval behind them. Everything from your offline eval chapter — golden datasets, LLM-as-judge calibration, reference-based metrics, statistical significance on small samples — is exactly what feeds Pattern B and C. A canary quality gate is *your offline eval, run online, on a traffic sample, wired to a rollback switch.*

### Flagger: the same idea, declared on one object

Flagger folds steps + metrics + webhooks into a single `Canary` resource and drives the mesh for you. Equivalent rollout:

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: model
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model
  service:
    port: 8000
  analysis:
    interval: 1m          # analyze every minute
    threshold: 5          # roll back after 5 failed checks
    maxWeight: 50         # cap canary at 50% before promote-to-100
    stepWeight: 10        # +10% each successful interval
    metrics:
    - name: request-success-rate
      thresholdRange: { min: 99 }
      interval: 1m
    - name: request-duration
      thresholdRange: { max: 500 }   # ms
      interval: 1m
    - name: quality-score            # custom Prometheus MetricTemplate
      thresholdRange: { min: 0.85 }
      interval: 5m
    webhooks:
    - name: eval-suite
      type: pre-rollout              # must pass before any traffic shifts
      url: http://eval-harness.default/run
      timeout: 5m
      metadata:
        endpoint: http://model-canary.default:8000
        suite: regression-v3
    - name: load-test
      type: rollout
      url: http://flagger-loadtester.default/
      metadata:
        cmd: "hey -z 1m -q 20 http://model-canary.default:8000/generate"
```

Flagger's control loop: every `interval` it nudges weight up by `stepWeight`, checks all `metrics`, and runs `rollout`-phase `webhooks`. Built-in `request-success-rate` and `request-duration` come from your mesh's Prometheus; **custom quality metrics are `MetricTemplate` objects** (arbitrary PromQL) referenced by name. A `pre-rollout` webhook is a hard gate that runs *before* the first traffic shift — the right place for an expensive full eval suite. Cross `threshold` failed checks → automatic rollback to primary.

Argo vs Flagger, briefly: Argo Rollouts is imperative-steps + first-class `AnalysisTemplate`/`Experiment` (great when you want fine-grained control and blue-green *and* canary in one tool); Flagger is declarative and convention-driven with batteries-included webhooks (great when you want less YAML and a strong load-test/conformance story). Both roll back automatically on metric breach. Pick one; don't run both on the same workload.

---

## Mechanism 3: Progressive promotion and automatic rollback

The promotion ladder is the heart of canarying. A sane model ladder:

```
  weight   dwell     gate
  ------   -----     ----
    1%     10 min    infra only (smoke: is it even up?)
    5%     15 min    infra + cheap quality proxies (refusal, empty, logprob)
   25%     30 min    infra + online judge sample (quality-score)
   50%     60 min    infra + full eval-suite job (blocking)
  100%     —         promote; keep old fleet for N minutes before scale-down
```

Design principles:

- **Dwell long enough to see the signal.** Quality metrics are noisy on small samples. At 1% traffic you may not accumulate enough judged responses in 10 minutes for a stable estimate — either lengthen the dwell, widen the sample, or don't gate quality until a higher weight. Gating quality at 1% on 12 samples is how you get flaky rollbacks.
- **Rollback must be *cheaper* than roll-forward.** With Argo/Flagger, rollback = set canary weight to 0 and keep serving stable. It's instantaneous at the traffic layer *because you never tore down stable.* This is why you keep the old fleet warm until promotion fully completes.
- **Automatic beats manual.** The controller aborts the moment `failureLimit`/`threshold` is crossed. Humans add an optional `pause: {}` (indefinite) step for a manual approval gate before 100% on high-stakes rollouts — but the *failure* path should never require a human.
- **Analysis can also run for the whole rollout**, not just per-step. Argo's `spec.strategy.canary.analysis` (background analysis) runs continuously and can abort at any weight the instant a metric breaches — useful for a "circuit breaker" on error rate that shouldn't wait for the next step boundary.

---

## Shadow / mirror deployments

Shadowing (a.k.a. mirroring, dark launch) sends the new version a **copy** of real requests and **throws away its responses**. Users are served *entirely* by stable; the canary sees production-distribution traffic with zero user risk. This is the safest possible way to evaluate a scary model change.

### Istio mirroring

```yaml
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata:
  name: model
spec:
  hosts:
  - model
  http:
  - route:
    - destination:
        host: model
        subset: v1        # 100% of user-visible traffic → stable
      weight: 100
    mirror:
      host: model
      subset: v2          # a copy also goes to canary
    mirrorPercentage:
      value: 100.0        # mirror 100% of it (or dial down for GPU cost)
```

Semantics that matter:

- Mirrored requests are **fire-and-forget** — Envoy does not wait for and *discards* the canary's response. Canary latency and errors cannot hurt users.
- Istio appends `-shadow` to the `Host`/`Authority` header of mirrored requests, so the canary (and any downstream) can tell it's shadow traffic.
- `mirrorPercentage.value` controls what fraction is copied. GPU inference is expensive; mirror 100% only if you can afford a second full inference path, else sample (e.g. `10.0`).

Argo Rollouts expresses the same idea with a `setMirrorRoute` step (mirror by percentage/match), letting you shadow before you canary within one Rollout.

### What shadow buys you, and what it can't

Shadow gives you real prompts against the new model with no user exposure — perfect for:

- Comparing v1 vs v2 outputs on *identical* live inputs (paired diffing → the strongest quality comparison you can get).
- Load/soak testing on real traffic shape (bursty, long-context, adversarial) that synthetic tests miss.
- Warming caches and JIT/compilation before real traffic arrives (see cache warmup below).

What it cannot do:

- **Anything with side effects.** If the model call writes to a DB, calls a tool, sends an email, or bills a token budget, the *shadow* will do it too unless downstream services are shadow-aware (check that `-shadow` header and no-op). A mirrored request that triggers a real tool call is a production incident. Make write paths shadow-aware or don't mirror them.
- **Measure user-facing outcomes.** Shadow responses are discarded, so you get model-quality signals but not click-through, thumbs-up, or downstream conversion. Those need a real canary.

**The mature pattern: shadow → canary → promote.** Shadow the risky checkpoint to catch gross regressions with zero risk; the survivors graduate to a small canary with real users and outcome metrics; the ones that pass promote.

---

## Failure modes and pitfalls

**1. Quality regression sails through infra gates.** The headline failure, restated because it is the whole point: latency/error/saturation are all green while factuality, refusal rate, format-adherence, or tone regress. *Mitigation:* a quality gate (Patterns A–C) is non-negotiable in any model canary. If you only remember one thing, remember this.

**2. Sample size / noise → flaky rollbacks (and flaky promotions).** Quality metrics on a 1% slice over 10 minutes are statistically thin. Too tight a threshold → you roll back good models on noise; too loose → you promote bad ones. *Mitigation:* size the dwell/weight so each analysis has enough judged samples for a stable estimate; use `count`/`failureLimit` to require *repeated* breaches, not one bad read; gate quality at higher weights where volume is sufficient.

**3. Session / stateful pinning broken by weighted splitting.** Weighted routing assigns each *request* independently. A multi-turn chat then flips between v1 and v2 mid-conversation — incoherent, and it also corrupts your per-version quality attribution. *Mitigation:* route by a stable key (session id / user id via consistent-hash or header match) so a conversation stays on one version for its lifetime; only split *new* sessions by weight.

**4. KV-cache / prefix-cache warmup and cold-start cliffs.** A freshly started model pod has a cold KV cache, cold prefix cache, cold CUDA graphs / compiled kernels, and possibly a cold model load from disk/network. Its first minutes show *inflated* latency and lower throughput that have nothing to do with the model's steady-state quality. A canary that measures latency in that window rolls back a perfectly good model. *Mitigation:* add a warmup/`pause` before the first analysis; use a readiness probe that only passes post-warmup; pre-load and pre-compile (shadow traffic is great for this); exclude the warmup window from analysis.

**5. Cost of running two model copies.** Canary and shadow both mean a second inference path on scarce, expensive accelerators. Shadow at 100% mirror = 2× GPU for the whole shadow period; blue-green = 2× fleet during overlap. *Mitigation:* right-size the canary (small replica count is fine at 5% weight — but watch pitfall #4, too few replicas + cold cache skews latency); sample shadow traffic (`mirrorPercentage`) instead of mirroring everything; keep overlap windows tight; scale the old fleet down promptly *after* promotion is confirmed (but not before — you need it for instant rollback).

**6. Metric attribution bleed.** If canary and stable share a Service/Prometheus label, your "canary success rate" query silently averages both and hides the regression. *Mitigation:* distinct `version` labels and separate `canaryService`/`stableService`; always scope analysis PromQL to the canary subset.

**7. Rollback that isn't actually fast.** Teams assume rollback is instant, then discover the old fleet was already scaled to zero, so "rollback" means cold-starting GPUs for minutes under a live incident. *Mitigation:* keep stable fully warm until promotion completes; make weight→0 the rollback, never a redeploy.

**8. Shadow side effects.** Covered above — mirrored traffic hitting real write paths. *Mitigation:* shadow-aware downstreams keyed on the `-shadow` header; never mirror non-idempotent paths blindly.

---

## Production checklist — what an interviewer probes

1. **"Your canary is green on latency and error rate — how do you know the new model is actually good?"** Answer must name a *quality gate*: cheap proxies in Prometheus (refusal/empty/logprob), an online judge/eval-run writing a metric, or a blocking eval-suite job/webhook. If your answer stops at latency+errors, you've failed the question.
2. **Traffic-split mechanism and why.** Can you go beyond replica-ratio Service splitting to mesh/Gateway weights? Do you know weight is proportional, not percentage, in Gateway API? Do you handle session pinning for multi-turn?
3. **Automatic rollback path.** What exact condition fires it (`failureLimit`/`threshold`), how fast is it, and why is it fast (old fleet stays warm; rollback = weight→0, not redeploy)?
4. **Shadow vs canary tradeoff.** When do you shadow first? What can shadow *not* tell you (user outcomes), and what's the side-effect hazard?
5. **Statistical soundness of the gate.** How do you avoid flaky rollbacks from thin quality samples? Dwell time, sample size, repeated-breach thresholds.
6. **Cold-start / cache warmup handling.** How do you keep KV/prefix-cache and kernel-compile cold starts from skewing the first analysis window?
7. **Cost.** How much extra GPU does your canary/shadow burn, and how do you bound it (mirror sampling, tight overlap, prompt post-promotion scale-down)?
8. **Blue-green vs canary decision.** When is an atomic flip (tokenizer/prompt-format change) actually the *right* call over a mixed canary?

---

## Further reading

- Argo Rollouts — Analysis, AnalysisTemplate & metrics: <https://argo-rollouts.readthedocs.io/en/stable/features/analysis/>
- Argo Rollouts — Canary strategy & steps: <https://argo-rollouts.readthedocs.io/en/stable/features/canary/>
- Argo Rollouts — Istio traffic management: <https://argo-rollouts.readthedocs.io/en/stable/features/traffic-management/istio/>
- Flagger — How it works (Canary spec, promotion/rollback): <https://docs.flagger.app/usage/how-it-works>
- Flagger — Metrics analysis (MetricTemplate, thresholds): <https://docs.flagger.app/usage/metrics>
- Flagger — Webhooks (pre-rollout gates, load testing): <https://docs.flagger.app/usage/webhooks>
- Flagger — Deployment strategies (canary, blue-green, mirroring): <https://docs.flagger.app/usage/deployment-strategies>
- Istio — Traffic shifting (weighted VirtualService): <https://istio.io/latest/docs/tasks/traffic-management/traffic-shifting/>
- Istio — Traffic mirroring (shadow): <https://istio.io/latest/docs/tasks/traffic-management/mirroring/>
- Gateway API — Traffic splitting (HTTPRoute weights): <https://gateway-api.sigs.k8s.io/guides/traffic-splitting/>
- Kubernetes — Deployment rolling updates: <https://kubernetes.io/docs/concepts/workloads/controllers/deployment/>
