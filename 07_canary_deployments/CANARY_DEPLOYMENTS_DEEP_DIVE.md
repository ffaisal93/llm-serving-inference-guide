# Canary Deployments for Model Serving

> Safely rolling out new models and versions — when the thing you are canarying can pass every infra gate and still be *worse*.

## Why this matters

Shipping application code and shipping a new model version look identical from thirty thousand feet: build an artifact, put it behind a service, shift traffic, watch dashboards. They are not the same problem.

A web service is mostly *right or wrong*. It returns a 200 or a 500, it is fast or slow, and your infrastructure metrics — error rate, p95 latency, saturation — catch essentially every regression that matters. A canary that watches those four numbers is a very good canary.

A model is *right, wrong, or plausibly-wrong-in-a-way-that-looks-right*. A new checkpoint can return HTTP 200 on every request, at lower latency than the old one, while quietly hallucinating more, refusing more legitimate prompts, drifting in tone, or regressing on your hardest 5% of inputs. Every infra gate is green. Your users are unhappy. This is the single most important thing to internalize in this chapter:

**Model regressions live in the response *body*, not in the response *envelope*. Infra canaries only watch the envelope.**

So model serving needs everything a normal progressive-delivery pipeline has — traffic splitting, automated analysis, progressive promotion, automatic rollback — *plus* a quality gate that reads the body. The rest of this chapter builds that pipeline from the bottom up and then shows you the failure modes that bite teams who forget the italicized sentence above.

> **Saying it out loud.** Shipping code and shipping a model look identical from a distance — build an artifact, put it behind a service, shift traffic, watch dashboards — but they're not the same problem. A web service is basically right or wrong: it returns a 200 or a 500, it's fast or slow, and error rate plus latency plus saturation catch essentially every regression that matters. A model is right, wrong, or *plausibly wrong in a way that looks right*. A new checkpoint can return 200 on everything, faster than the old one, while hallucinating more, refusing more valid prompts, or regressing on your hardest five percent of inputs. The one sentence to remember: model regressions live in the response *body*, and infra canaries only watch the *envelope*.

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

> **Saying it out loud.** Picture ten percent of traffic on a new summarizer. Error rate identical, p95 latency actually 70 milliseconds *better*, no restarts, throughput up. A standard Flagger or Argo canary promotes that rollout, confidently. Meanwhile groundedness dropped from 0.91 to 0.78 and refusal rate on valid prompts went from 1.2% to 6.5%. Those two rows are invisible to the canary because reading them requires *scoring the generated text*, which is not a Prometheus counter you get for free. The distinction to hold: infra metrics answer "did the request complete correctly," and they're cheap and real-time. Quality metrics answer "was the answer good," and they're expensive, often delayed, and specific to your task. Canary analysis for models is the union — with only the first half, you've built a very sophisticated way to ship regressions confidently.

### A second example, different modality: code completion

The summarization example makes the point with text-quality metrics, but the pattern generalizes to any generative task with its own notion of "correct." Picture a code-completion model behind an IDE plugin, canaried the same way:

| Signal | v1 (stable) | v2 (canary) | Infra verdict |
|---|---|---|---|
| HTTP 5xx rate | 0.01% | 0.01% | ✅ pass |
| p95 latency | 220 ms | 190 ms | ✅ pass (faster!) |
| Tokens/sec throughput | 340 | 365 | ✅ pass |
| **Suggestion compile-rate** | 94% | **88%** | ❌ **regression** |
| **Suggestion test-pass-rate (on a held-out repo suite)** | 71% | **58%** | ❌ **regression** |

Same shape, different domain-specific quality signal: for a summarizer it's groundedness/refusal; for a code-completion model it's does-it-compile and does-it-pass-the-tests. The infra gate is blind to both for the same structural reason — neither compile-rate nor test-pass-rate is a property of the HTTP response envelope, they're properties of *running the generated code*, which means the quality gate here necessarily looks like a `job` metric provider (Pattern C from Mechanism 2): spin up a sandboxed compile/test harness against a sample of the canary's suggestions and gate on its pass rate. The specific metric changes with the task; the requirement that *some* task-specific quality signal exists in the gate does not.

> **Saying it out loud.** The pattern generalizes to any generative task with its own notion of correct. Take a code-completion model in an IDE: same story — error rate flat, p95 latency down 30 milliseconds, throughput up — while suggestion compile-rate dropped from 94% to 88% and test-pass-rate on a held-out repo suite fell from 71% to 58%. Same structural blindness for the same reason: neither compile-rate nor test-pass-rate is a property of the HTTP envelope, they're properties of *running the generated code*. Which tells you what the gate has to look like in practice — a sandboxed compile-and-test harness run against a sample of the canary's suggestions. The specific metric changes with the task; the requirement that some task-specific quality signal exists in the gate does not.

---

## The four strategies and when each fits model serving

Before mechanisms, get the taxonomy straight. All four move traffic from an old version to a new one; they differ in *how much blast radius* a bad version gets and *how fast you can undo it*.

- **Rolling update** — replace old pods with new ones a few at a time. There is no "old vs new" concept at the traffic layer; once a pod is new, it serves real users. This is the Kubernetes Deployment default.
- **Blue-green** — stand up the full new version (green) alongside the full old version (blue), test green out-of-band, then flip 100% of traffic at once. Instant cutover, instant rollback (flip back), but you pay for two full fleets during the overlap.
- **Canary** — run the new version at small scale, send it a *slice* of real traffic (1% → 5% → 25% → …), analyze, and promote in steps. Small blast radius, gradual confidence.
- **Shadow (mirror)** — send the new version a *copy* of real traffic but **discard its responses**. Users never see canary output. Pure evaluation, zero user risk.

> **Saying it out loud.** Four ways to move traffic from old to new, differing in blast radius and undo speed. Rolling update swaps pods a few at a time with no old-versus-new concept at the traffic layer — that's the Kubernetes default and it's the wrong choice here. Blue-green stands up the full new fleet alongside the old and flips 100% at once: instant cutover and instant rollback, but you pay for two full fleets. Canary runs the new version small, sends it a slice of real traffic, analyzes, and promotes in steps. Shadow sends the new version a *copy* of traffic and throws the responses away — zero user risk, pure evaluation. For models, canary is the workhorse and shadow goes first for anything scary.

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

> **Saying it out loud.** My rules of thumb. Never roll out a new model with a bare rolling update — you lose the old-versus-new traffic distinction exactly when you need it most, and GPU pods are slow enough to cycle that a "quick" rollback isn't quick. Canary is the default: small weight, automated analysis on infra *and* quality, progressive promotion. Shadow first for genuinely scary changes — new architecture, new quantization, new base model — because it gives you production-distribution eval data with zero user exposure, and then you canary the survivors. And blue-green specifically when the *mix* is the problem: a tokenizer or prompt-format change where having v1 and v2 answer alternating turns of the same conversation would be incoherent, so you want an atomic switch gated on a full eval run.

---

## The 2025–2026 landscape

The mechanisms in this chapter (Istio/Gateway API weights, Argo Rollouts, Flagger) are stable, years-old primitives. What has changed recently is (1) how much of this you can now do **without a service mesh**, (2) how the "quality gate" piece — Pattern B/C, below — is being formalized instead of hand-rolled, and (3) a brand-new, LLM-serving-specific routing layer that understands models the way a mesh understands services.

**Argo Rollouts keeps shipping steadily as a CNCF project.** As of mid-2026 the stable line is v1.9.x: v1.9.0 (March 20, 2026) fixed canary-weight/DestinationRule-update calculations and a blue-green analysis timing bug where success was reported prematurely while the ReplicaSet was still undersaturated; v1.8.4 (Feb 13, 2026) and v1.8.3 (Jun 5, 2025) were patch releases addressing analysis edge cases and an OAuth2 CVE. None of this changes the `AnalysisTemplate`/`Rollout` shapes used in this chapter — the API has been stable for years, which is exactly why it's safe to build a model-serving control plane on top of it. Release notes: <https://github.com/argoproj/argo-rollouts/releases>.

**Flagger has moved decisively toward Gateway API as a first-class, mesh-optional target.** Flagger v1.42.0 (Oct 16, 2025) bumped Gateway API support to v1.4.0 and added CORS policy configuration on `HTTPRoute`, plus a `trafficDistribution` field (Kubernetes 1.33+) and an `unmanagedMetadata` option so GitOps controllers and Flagger can co-own the same Service without fighting over labels. Flagger v1.43.0 (Apr 21, 2026) went further on observability and session handling: a Kubernetes External Metrics provider (wired to things like the Datadog Cluster Agent) so canary analysis can pull SLO metrics from outside Prometheus, and a configurable primary-cookie name for session affinity in both the Istio and Gateway API routers. Changelog: <https://github.com/fluxcd/flagger/blob/main/CHANGELOG.md>.

**The bigger shift: mesh-free canaries via Gateway API are now a documented, supported path, not a workaround.** Flagger's own tutorial for this reads almost exactly like the Istio walkthrough later in this chapter, except the traffic-shifting object is a plain `HTTPRoute` attached to a `Gateway`, and Flagger edits its backend weights directly — no sidecars, no `VirtualService`, no mesh control plane to operate: <https://docs.flagger.app/tutorials/gatewayapi-progressive-delivery>. This matters for model serving specifically because GPU inference pods are already resource-heavy; not running an Envoy sidecar per pod is a real cost and complexity saving when your bottleneck is GPU memory, not network hops. The catch, per Flagger's own docs, is that you inherit whatever your specific Gateway implementation supports — session affinity needs `ResponseHeaderModifier` support, mirroring needs `RequestMirror` support, and not every Gateway controller implements every optional Gateway API feature yet.

**The Gateway API project has been formalizing exactly this "mesh-agnostic canary" use case since GEP-1324** (the GAMMA initiative — Gateway API for Mesh Management and Administration), which is explicitly framed around requests like "I want to deploy a canary version of my application that splits traffic based on HTTP properties," and deliberately stays agnostic to sidecar-vs-sidecar-free mesh data planes. Practically: whether your traffic layer is Istio, Linkerd, a Gateway-API-native controller, or no mesh at all, the same `HTTPRoute` weight-splitting vocabulary now works, which is why Argo Rollouts, Flagger, and the mesh vendors have all converged on it as the interchange format. GEP: <https://gateway-api.sigs.k8s.io/geps/gep-1324/>.

> **Saying it out loud.** The mechanisms here are years-old stable primitives; three things changed recently. First, you can now do canaries *without a service mesh* — Gateway API `HTTPRoute` weight-splitting is a documented, supported path in both Argo Rollouts and Flagger, which matters for GPU serving specifically because not running an Envoy sidecar per pod is a real saving when your bottleneck is GPU memory. Second, the quality-gate piece is getting formalized into specs and papers instead of hand-rolled scripts. Third, there's now a purpose-built LLM routing layer that understands models rather than just replica pools. Net for 2026: pick Argo or Flagger on their merits, route through Gateway API unless you already depend on mesh features, and treat the quality-gate webhook as a first-class metric provider rather than a manual step before or after.

### LLM-specific routing: the Gateway API Inference Extension

Everything above splits traffic by *replica pool*, unaware that the thing behind the pool is an LLM. As of mid-2025 there is a purpose-built layer for exactly that gap: the **Gateway API Inference Extension** (Kubernetes SIG, announced June 5, 2025, actively developed through early 2026), which adds inference-aware routing on top of Gateway API instead of treating an inference server like any other HTTP backend. The project's own framing is that generic load balancers don't understand LLM serving's actual constraints — long-running, resource-heavy requests, in-memory KV-cache state that makes some backends cheaper to route to than others for a given prefix, and the fact that "the model" isn't one thing but potentially many LoRA adapters sharing a base model. Announcement: <https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/>; deep dive: <https://www.cncf.io/blog/2025/04/21/deep-dive-into-the-gateway-api-inference-extension/>; repo: <https://github.com/kubernetes-sigs/gateway-api-inference-extension>.

Two new CRDs carry this: an **`InferencePool`** groups model-server replicas (the way a `Service` groups pods, but with KV-cache-aware, queue-depth-aware load balancing done by an "Endpoint Picker" instead of round robin), and an **`InferenceModel`**/`InferenceObjective` sits in front of it to do *model-identity* routing — which named model or adapter a request actually wants, and at what priority. Istio's 2025 support for this extension demonstrates the part that matters most for this chapter: weighted canary rollout expressed **in terms of model versions, not just replica pools** —

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceModel
metadata:
  name: customer-support-router
spec:
  modelName: customer-support
  criticality: Critical
  poolRef:
    name: llama-pool          # the InferencePool backing both versions
  targetModels:
  - name: llama-3-8b-customer-v1
    weight: 80
  - name: llama-3-8b-customer-v2
    weight: 20
```

(Schema per Istio's Gateway API Inference Extension integration post — the CRD is young and its exact fields are still evolving, so treat this as illustrative of the pattern rather than a copy-paste-stable spec. Source: <https://istio.io/latest/blog/2025/inference-extension-support/>.) The `weight` semantics are the same proportional-split idea as `HTTPRoute` backend weights, but the split happens **inside model-aware routing** — the Endpoint Picker can simultaneously balance by KV-cache/queue state *and* respect the 80/20 version split, something a plain `HTTPRoute` weight can't do because it has no concept of cache state at all. The project's own docs describe the target use cases plainly: "A/B traffic splitting, and safe blue-green base model and model server upgrades" for exactly the canary/progressive-rollout scenarios this chapter covers.

Where this fits relative to everything else in this chapter: it is a **routing-layer** improvement, not a substitute for the analysis/rollback machinery. You would still put an `InferencePool`/`InferenceModel` pair behind an Argo Rollouts or Flagger-style automated analysis loop — the CRDs give you a better *dial* (model-identity- and cache-aware weighting) to turn, not a different *decision process*. It is early — expect the API surface to keep changing — but it is the clearest sign yet that "canary a model" is becoming a first-class Kubernetes networking concept, not something you bolt onto generic HTTP traffic splitting and hope for the best.

**On the quality-gate side, the "webhook that scores the canary" pattern is getting a name and a spec, not just ad-hoc scripts.** A March 2026 paper, *Automated Self-Testing as a Quality Gate for LLM Applications* (Maiorano, arXiv:2603.15676), formalizes this as a release gate that scores every candidate build against a living "question bank" across five dimensions — task success rate (≥80%), multi-turn context preservation (≥90%), p95 latency (<15s), guardrail/safety pass rate (≥95%), and citation/evidence coverage (≥80%) — and emits one of three deterministic decisions: PROMOTE, HOLD, or ROLLBACK, with a second "70%-of-target" threshold that forces an automatic ROLLBACK for systemic failures rather than a human judgment call. Two findings from that paper are directly relevant to the AnalysisTemplate patterns in this chapter: evidence/citation coverage was the strongest discriminator of severe regressions across their 38 evaluation runs (stronger than latency or routing signals), and automated structural checks plus content-focused LLM judgment caught *different* failure modes — neither subsumes the other, which is the same "infra gate ≠ quality gate" argument this chapter opened with, now with data behind it. The framework as described is a pre-production gate rather than a live-traffic canary, but its five-dimension scoring and PROMOTE/HOLD/ROLLBACK vocabulary maps directly onto an Argo `web` metric provider or a Flagger `pre-rollout` webhook — see the worked example later in this chapter.

**Industry write-ups through 2026 echo the same message this chapter leads with.** MLflow's 2026 canary-deployment guide for AI models states it plainly: "A model that looks healthy on error rate and latency dashboards can still be producing logically inferior outputs," and pushes teams toward hallucination-rate and LLM-as-judge metrics running *concurrently* with the canary, not just in a pre-launch eval run (<https://mlflow.org/articles/what-is-canary-deployment-ai/>). None of this changes the mechanisms in this chapter — it validates that infra-only canaries for models are now a widely recognized anti-pattern, not a niche concern.

Net for 2026: pick Argo Rollouts or Flagger on their existing merits (imperative fine-grained steps vs. declarative convention — see the comparison later in this chapter), route traffic through Gateway API `HTTPRoute` unless you have a mesh-specific reason not to (mTLS, L7 policy you already depend on), watch the Gateway API Inference Extension if your routing layer needs to be model- and cache-aware rather than just replica-aware, and treat the quality-gate webhook/job as a first-class metric provider in your AnalysisTemplate/Canary, not a separate manual step that happens "before" or "after" the automated pipeline.

> **Saying it out loud.** Everything else in this chapter splits traffic by replica pool, completely unaware that the thing behind the pool is an LLM. The Inference Extension adds two CRDs that fix that. An `InferencePool` groups model-server replicas but load-balances with an Endpoint Picker that's aware of KV-cache state and queue depth instead of doing round robin. And an `InferenceModel` sits in front doing model-*identity* routing — which named model or LoRA adapter a request wants, at what priority — which is where you express an 80/20 canary in terms of model versions rather than pools. The reason that matters: a plain `HTTPRoute` weight can't simultaneously respect a version split *and* route for cache locality, because it has no concept of cache state. But it's a better dial, not a different decision process — you still wrap it in Argo or Flagger analysis.

---

## Mechanism 1: Traffic splitting

Canary and shadow both need to route a *fraction* of requests somewhere. There are four common layers to do it, from crudest to most precise.

> **Saying it out loud.** Canary and shadow both need to route a fraction of requests somewhere, and there are four layers to do it at, from crudest to most precise. Replica-ratio splitting through a shared Service — nine stable pods and one canary pod gives you roughly ten percent. Mesh-level weighted routing via an Istio `VirtualService`. Vendor-neutral weighting via Gateway API `HTTPRoute`. And header or session-based routing when requests aren't independent. That last distinction is the LLM-specific one: weighted splitting assumes each request stands alone, and a multi-turn chat conversation absolutely does not — if turn three lands on v2 after turns one and two hit v1, the assistant contradicts itself in a different voice.

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

> **Saying it out loud.** The oldest trick is two Deployments sharing one Service through a common label selector, so the traffic split is roughly the replica ratio — nine stable pods, one canary, ten percent. It needs no mesh and no extra controller, which is its only virtue. The problems are real, though. Your minimum canary weight is bounded by replica count, so a one-percent canary needs ninety-nine stable pods, which is absurd on GPUs. You can't decouple weight from capacity, so a small canary is structurally slower per request. And there's no session stickiness at all. For GPU model serving specifically, that coupling is what makes it wrong — you end up choosing between a meaningful traffic weight and a sane number of expensive replicas.

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

> **Saying it out loud.** Istio's contribution is decoupling weight from replica count: a `DestinationRule` defines named subsets by pod label, and a `VirtualService` assigns weights across them that sum to 100. So you can send one percent of traffic to a canary that has two replicas — the weight and the capacity are independent knobs, which is exactly what the Service-based approach couldn't do. That decoupling is what makes real canary ladders possible on GPUs, where you can't afford ninety-nine stable replicas just to express one percent. The cost is that you're now operating a mesh control plane and a sidecar per pod, which on GPU nodes is memory and complexity you may not want — hence the move toward Gateway API for teams that don't otherwise need mTLS or L7 policy.

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

`90 + 10 = 100` so v2 gets 10%. If you'd written `9` and `1`, v2 still gets 10% — the ratio is what matters. Gateway API is where new tooling is converging; prefer it for greenfield (see The 2025–2026 landscape, above), and note that the Gateway API Inference Extension's `InferencePool`/`InferenceModel` pair is built as a layer *on top of* this same `HTTPRoute` vocabulary, not a replacement for it.

> **Saying it out loud.** Gateway API does the same weighting in a mesh-agnostic way through `HTTPRoute` backend refs, and there's one detail worth getting right because interviewers ask it: **weight is a proportional split, not a percentage.** The sum of the weights in a rule is the denominator. So weights of 3 and 1 mean 75/25, not 3% and 1% with 96% going nowhere. The reason this vocabulary won is that Argo Rollouts, Flagger, and the mesh vendors all converged on it as an interchange format, so the same weight-splitting expression works whether you're on Istio, Linkerd, a Gateway-native controller, or no mesh at all. The catch: optional features like mirroring and session affinity depend on what your specific Gateway controller actually implements.

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

> **Saying it out loud.** Weighted splitting quietly assumes requests are independent, and LLM chat sessions are not. If a multi-turn conversation gets split across versions, turn three answers in v2's voice with v2's context handling after turns one and two came from v1 — incoherent to the user, and it also corrupts your quality measurement, because now neither version is being evaluated on a clean conversation. So you route by a stable key instead: a session ID header, a consistent hash of the user ID, or an explicit opt-in cohort header for internal users. The rule to state plainly: pin at the *conversation* level, not the request level, whenever the model's output depends on prior turns — which is essentially always for chat.

### 1e. Model-serving-native canary: KServe

Everything above is generic Kubernetes traffic-splitting, bolted onto a model server as if it were any other HTTP backend. If you deploy models via **KServe**, canarying is a first-class part of the `InferenceService` spec instead of a separate mesh object. Set `canaryTrafficPercent` on the component you're updating and KServe (in its serverless, Knative-backed mode) manages two revisions for you directly:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: model
spec:
  predictor:
    canaryTrafficPercent: 10       # 10% of traffic to the new revision
    model:
      modelFormat:
        name: huggingface
      storageUri: gs://my-bucket/model/v2
```

KServe tracks this with two rollout pointers: `status.components.predictor.latestRolledoutRevision` (the revision serving the *other* 1-`canaryTrafficPercent` slice — effectively "stable") and the newly created revision receiving the canary slice; on rollback it points `previousRolledoutRevision` back to 100%. Apply a new `storageUri` with `canaryTrafficPercent: 0` first to stage the revision, then ramp the percentage up the same way you'd ramp an Istio weight. The official walkthrough is a good reference: <https://kserve.github.io/website/docs/model-serving/predictive-inference/rollout-strategies/canary-example>.

Two things worth knowing before you reach for it:

- **Serverless-mode only.** KServe's docs are explicit that this canary strategy is only supported when the `InferenceService` runs in serverless (Knative) deployment mode, not in the raw-Kubernetes-Deployment mode. If you're on the raw mode for GPU-scheduling reasons, you're back to Istio/Gateway API weights or Argo/Flagger driving a `Deployment` directly.
- **It's a traffic-splitting primitive, not an analysis engine.** `canaryTrafficPercent` is the same idea as an `HTTPRoute` weight, expressed at the ML-serving-platform layer instead of the networking layer — it does not, by itself, run an `AnalysisTemplate` or check a quality gate. A March 2026 write-up on combining GitOps with KServe canaries pairs `canaryTrafficPercent` with Argo Rollouts/Prometheus for the actual promote/rollback decision, and calls out a very on-topic warning for this chapter: "canary windows and traffic percentages should account for warm-up latency" for large models, i.e. the cold-start pitfall from Failure Modes applies here just as much as it does to an Istio-routed canary (<https://devopsie.com/2026-03-19/gitops-driven-canary-rollouts-for-ml-models-with-argo-cd-and-kserve.html>).

The takeaway: whether the split lives in a `VirtualService`, an `HTTPRoute`, an `InferenceModel`, or `canaryTrafficPercent`, it's still just the *traffic* half of the problem — the analysis/quality-gate half from Mechanism 2 is what actually decides whether to keep ramping.

> **Saying it out loud.** Everything above is generic Kubernetes traffic splitting bolted onto a model server as if it were any HTTP backend. If you deploy through KServe, canarying is a first-class field on the `InferenceService` — you set `canaryTrafficPercent` on the component you're updating and KServe manages the two revisions for you, using Knative underneath. The appeal is that the canary concept lives at the same level as the model, so a rollout is one field change rather than coordinated edits across a Deployment, a DestinationRule, and a VirtualService. The tradeoff is the usual one with abstractions: you get less control over the exact ladder and analysis, and you're debugging through a layer — so it fits teams standardizing on KServe, not teams that need fine-grained step control.

---

## Mechanism 2: Automated canary analysis

Manually staring at Grafana while you bump weights doesn't scale and doesn't fire at 3 a.m. Automated analysis makes the promote/rollback decision from metrics. Two dominant tools: **Argo Rollouts** and **Flagger**.

The shape of both:

1. You declare **steps** (weights + pauses) and **metrics with thresholds**.
2. The controller shifts traffic to the first step.
3. At each step it queries metrics (usually Prometheus) over an interval, a number of times.
4. If a metric violates its condition too many times → **abort and roll back**.
5. If all steps pass → **promote** (canary becomes stable).

> **Saying it out loud.** Staring at Grafana while you bump weights doesn't scale and definitely doesn't fire at three in the morning, so you automate the promote-or-rollback decision. Both dominant tools have the same shape: you declare steps — weights and pauses — plus metrics with thresholds; the controller shifts traffic to the first step, queries the metrics over an interval a set number of times, and if a metric violates its condition too often it aborts and rolls back, otherwise it advances. Argo Rollouts is the imperative, fine-grained one where you spell out each step. Flagger is the declarative one where you state a convention and it generates the ladder. The important design property in both: the *failure* path is fully automatic, and only the success path is allowed to wait on a human.

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

> **Saying it out loud.** An `AnalysisTemplate` is a reusable bundle of metric checks, and each metric has four numbers that matter. The `interval` is how often you query. The `count` is how many times total. The `failureCondition` is the expression that means bad. And `failureLimit` is how many bad readings you tolerate before aborting — which exists specifically so one noisy measurement doesn't roll back a healthy deploy. A typical infra template watches success rate and p95 latency from your mesh's Prometheus metrics. The thing to notice about the design: it's reusable and referenced by name, so the same quality gate can be shared across every model rollout in the org rather than copy-pasted, which is how a quality bar actually gets enforced rather than suggested.

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

> **Saying it out loud.** The `Rollout` object replaces your Deployment, and its canary strategy is an explicit list of steps you interleave: `setWeight` to shift traffic, `pause` to dwell, and `analysis` to run a template. So you literally write out one percent, wait ten minutes, run infra checks, five percent, wait fifteen, run infra plus cheap proxies, and so on. The virtue of that verbosity is that the ladder is legible in one place and reviewable in a pull request — someone can see exactly how much user exposure each gate is protecting. The other capability worth knowing: `spec.strategy.canary.analysis` runs *background* analysis continuously across the whole rollout, which is how you get a circuit breaker on error rate that doesn't have to wait for the next step boundary.

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

> **Saying it out loud.** Here's the key reframe: the quality gate is *just another metric in the AnalysisTemplate*. The whole trick is where the number comes from, and there are three patterns from cheapest to strongest. Pattern A is proxy signals already in Prometheus if your server emits them — refusal rate, empty-completion rate, mean output length as a truncation proxy, guardrail trigger rate, mean logprob. Those cost nothing and catch gross failures. Pattern B is a `web` provider calling an eval service that scores a window of recent canary responses. Pattern C is a `job` provider running a blocking eval suite. Start with A because it's free and instrument it today; graduate to B and C as the stakes rise. What you must not do is stop at latency and error rate.

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

> **Saying it out loud.** Flagger folds the steps, metrics, and webhooks into a single `Canary` resource and drives the mesh for you — you declare a step weight, a max weight, an interval, and a failure threshold, and Flagger generates the ladder rather than you enumerating it. The real philosophical difference from Argo is declarative convention versus imperative control: Flagger is less YAML for the common case and less flexible when you want a non-uniform ladder, like dwelling much longer at 25% than at 5%. The other thing to know is that Flagger treats every configured metric and every rollout-phase webhook as a hard AND — a single failure halts and rolls back. Which is what you want for a quality gate, and worth confirming rather than assuming.

---

## Build it in practice — extended

The AnalysisTemplate/Flagger snippets above showed infra metrics and quality metrics as separate examples. In production you gate on **both at once** — a single analysis step where a canary must clear a Prometheus-based infra threshold *and* a webhook-based quality score before the next weight bump fires. Below is a complete, corrected worked example: the eval-service contract, the Argo Rollouts side, and the Flagger equivalent.

> **Saying it out loud.** In real pipelines you gate on infra *and* quality at once — a single analysis step where the canary must clear a Prometheus threshold and a webhook-based quality score before the next weight bump fires. That means three pieces: an eval service you own that scores recent canary responses, the Argo or Flagger wiring that calls it, and — the piece people get wrong — a clean distinction between "failed the gate" and "not enough data yet." Those two are completely different outcomes, and conflating them is precisely how you get flaky rollbacks that erode trust in the pipeline until someone turns the gate off. Which is worse than never having built it.

### The eval-service contract both tools call into

Whatever provider (`web`, `job`, or a webhook) you use, the actual scoring logic lives in a small service you own. It doesn't need to be elaborate — it needs to (a) pull a window of recently-logged canary responses, (b) score them (LLM-as-judge, reference match, or whatever your offline-eval chapter calibrated), and (c) return enough information for the gate to check *both* the score and the sample size. A minimal sketch:

```python
# quality-eval-service — called by Argo's `web` provider or Flagger's `rollout` webhook
from fastapi import FastAPI, Response
import time

app = FastAPI()

@app.post("/score")
def score(req: dict):
    window_s = req.get("window_minutes", 5) * 60
    since = time.time() - window_s
    samples = fetch_canary_responses(          # your logging/store lookup
        endpoint=req["endpoint"], since_ts=since
    )
    if len(samples) < req.get("min_samples", 100):
        # Not enough data yet — this is NOT the same as "failed the gate".
        # Argo: return low n_samples, successCondition's n_samples check catches it.
        # Flagger: return 202 (not yet ready) rather than a hard 4xx/5xx.
        return Response(status_code=202,
                         content='{"quality_score": null, "n_samples": %d}' % len(samples))
    score = judge_score(samples)               # LLM-as-judge / reference-based scorer
    passed = score >= 0.85
    body = {"quality_score": score, "n_samples": len(samples)}
    # Argo's `web` provider just wants the JSON body (see jsonPath below).
    # Flagger's `rollout` webhook wants the pass/fail encoded as the HTTP status.
    return Response(status_code=200 if passed else 500, content=str(body).replace("'", '"'))
```

The `202` branch matters: "not enough samples yet" and "the model failed" are different outcomes, and conflating them is how you get the flaky-rollback pitfall from Mechanism 3. Argo's JSON-based `successCondition` can distinguish the two explicitly (below); a pure-status-code webhook (Flagger) has to be more careful about what it returns while still warming up.

> **Saying it out loud.** The scoring service is small and you own it: pull a window of recently-logged canary responses, score them however your offline eval work calibrated, and return enough for the gate to check both the score *and* the sample size. That second return value is the whole point. If you've only got twelve judged samples so far, that is not a failure — it's not-yet-ready, and it should return a 202 rather than a 500. Argo can distinguish them explicitly because its `successCondition` reads named JSON fields, so you can require both a score threshold and a minimum sample count. A pure status-code webhook like Flagger's has to be more careful, because a 500 while warming up looks identical to a genuine regression.

### Argo Rollouts: one AnalysisTemplate, two independent metrics, both must pass

Argo Rollouts semantics: a `Rollout`'s `analysis` step references one or more `AnalysisTemplate`s, and — per Argo's own docs — "if multiple templates are referenced, then the controller will merge the templates together," so an `AnalysisRun` only reaches `Successful` when **every** metric in **every** referenced template passes its condition. There is no "OR" — a `web` provider failing rolls the run back exactly like a `prometheus` provider failing. That is precisely the "gated on both" behavior we want.

Argo's `web` metric provider calls the eval service directly; when `jsonPath` resolves to a JSON *object* rather than a scalar, `successCondition` can reference its fields by name — this is documented behavior, not a hack:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: infra-and-quality-gate
spec:
  args:
  - name: service-name
  metrics:
  # --- infra metric #1: success rate, from Prometheus ---
  - name: success-rate
    interval: 1m
    count: 5
    successCondition: result[0] >= 0.99
    failureLimit: 2
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
  # --- infra metric #2: p95 latency, from Prometheus ---
  - name: p95-latency
    interval: 1m
    count: 5
    successCondition: result[0] <= 500
    failureLimit: 2
    provider:
      prometheus:
        address: http://prometheus.istio-system:9090
        query: |
          histogram_quantile(0.95,
            sum(irate(istio_request_duration_milliseconds_bucket{
              destination_service=~"{{args.service-name}}"}[1m]))
            by (le))
  # --- quality metric: webhook-scored model output, must ALSO pass ---
  - name: model-quality
    interval: 5m
    count: 3
    failureLimit: 0                 # zero tolerance: any bad read aborts
    successCondition: "result.quality_score >= 0.85 && result.n_samples >= 100"
    provider:
      web:
        url: "http://quality-eval-service.default.svc.cluster.local/score"
        method: POST
        timeoutSeconds: 30
        jsonBody:
          endpoint: "http://model-canary.default.svc.cluster.local:8000"
          window_minutes: 5
          min_samples: 100
        jsonPath: "{$}"                # whole response body as `result`
```

Notes on the parts that are easy to get wrong:

- `jsonPath: "{$}"` returns the *whole* JSON body as `result`, which is what lets `successCondition` reference `result.quality_score` and `result.n_samples` together — this is how you enforce a minimum sample size (protecting against the "flaky rollback on 12 samples" pitfall) in the *same* condition as the score itself, without needing a second metric.
- `failureLimit: 0` on the quality metric is a deliberate asymmetry versus `failureLimit: 2` on the infra metrics: infra metrics tolerate a couple of noisy scrapes, but a bad quality read after clearing the sample-size bar is treated as real signal, not noise, and aborts immediately.
- `interval: 5m` / `count: 3` on the quality metric versus `interval: 1m` / `count: 5` on infra: quality scoring is expensive (it may itself call an LLM judge) and needs more wall-clock time to accumulate 100+ samples; don't run it on the same cadence as a cheap Prometheus scrape.
- All three metrics live in **one** `AnalysisTemplate`, referenced once from the `Rollout`'s `analysis` step — Argo evaluates them concurrently and the step only succeeds when all three do.

Wire it into the same canary ladder as before, just swapping the template name:

```yaml
      steps:
      - setWeight: 5
      - pause: { duration: 10m }
      - setWeight: 25
      - pause: { duration: 5m }
      - analysis:
          templates:
          - templateName: infra-and-quality-gate
          args:
          - name: service-name
            value: model-canary.default.svc.cluster.local
      - setWeight: 50
      - pause: { duration: 30m }
      - setWeight: 100
```

Quality gating is deliberately deferred to the 25% step rather than 5% — at 5% traffic the eval service can't reliably gather 100 samples in a 5-minute window, so gating it there would violate the sample-size principle from Mechanism 3, below. At 25% of, say, 300 rps, 100 samples arrive in well under a minute.

> **Saying it out loud.** Argo's semantics here are worth stating precisely because they're what makes the composite gate work. When a rollout step references multiple `AnalysisTemplate`s, the controller *merges* them, and the `AnalysisRun` only reaches Successful when every metric in every referenced template passes. There is no OR — a `web` provider failing rolls back exactly like a `prometheus` provider failing. That's precisely the gated-on-both behavior you want, and it means you can keep infra checks and quality checks in separate reusable templates without inventing any combination logic. The other documented detail that makes it practical: when a `web` provider's `jsonPath` resolves to an object rather than a scalar, `successCondition` can reference its fields by name — so score and sample count can be checked in one condition.

### Flagger: the same "both must pass" gate via a metric + a pre-rollout webhook

Flagger expresses the same idea with a built-in `request-success-rate`/`request-duration` metric pair plus a **synchronous webhook** that must return HTTP 200 before Flagger advances the weight. Flagger treats *every* configured metric and *every* `rollout`-phase webhook as a hard AND — any single failure halts and rolls back:

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
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange: { min: 99 }
      interval: 1m
    - name: request-duration
      thresholdRange: { max: 500 }
      interval: 1m
    webhooks:
    - name: model-quality-gate
      type: rollout                  # runs at every weight step, not just once
      url: http://quality-eval-service.default/score
      timeout: 30s
      metadata:
        endpoint: "http://model-canary.default:8000"
        window_minutes: "5"
        min_samples: "100"
      # Flagger treats any non-2xx response from this webhook as a failed check,
      # counted against `threshold` exactly like a failed metric. Our eval-service
      # sketch returns 202 (not yet enough samples) rather than 5xx while warming
      # up, so Flagger's retry-on-next-interval semantics don't burn `threshold`
      # budget on "still collecting data" the way a hard failure would.
```

The eval service here must return a 2xx only when it judges the canary healthy (e.g. `quality_score >= 0.85` internally, with the same `n_samples >= 100` guard) and a non-2xx otherwise — Flagger doesn't parse a JSON body from `rollout` webhooks the way Argo's `web` provider does, so the scoring logic (and the sample-size guard) has to live inside the webhook handler rather than in the CRD. This is the main practical difference between the two tools' quality-gate ergonomics: Argo lets the CRD assert on the JSON response; Flagger wants a boolean expressed as an HTTP status code.

> **Saying it out loud.** Flagger expresses the same composite gate differently: its built-in success-rate and duration metrics, plus a synchronous webhook that has to return HTTP 200 before Flagger advances the weight. And Flagger treats every metric and every rollout-phase webhook as a hard AND, so any single failure halts and rolls back — same semantics as Argo, different surface. The practical difference is that Flagger's gate signal is an HTTP *status code* rather than a JSON body, which is exactly why the not-enough-samples case needs care: you can't encode "score is 0.87 but n is only 12" in a status code without deciding what that means. Returning 202 for warming-up is the convention, but it's a convention you have to implement deliberately.

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

> **Saying it out loud.** A sane model ladder looks like: 1% for ten minutes with infra-only smoke checks, 5% for fifteen with cheap quality proxies, 25% for thirty with an online judge sample, 50% for an hour with a blocking full eval suite, then promote — and keep the old fleet warm for a bake period. Three design principles hold it together. Dwell long enough to actually see the signal, because gating a quality metric at 1% traffic on twelve samples is how you get flaky rollbacks. Rollback must be *cheaper* than roll-forward, which it is only because you never tore down stable — rollback is setting the canary weight to zero, instantaneous at the traffic layer. And automatic beats manual: humans can add an approval pause before 100%, but the failure path should never wait for a person.

### Statistical rigor: the peeking problem

There's a subtler version of the "sample size" pitfall worth naming explicitly, because it's the kind of thing that separates a good answer from a great one in an interview. `count`/`interval`/`failureLimit` checks a metric *repeatedly* over the dwell window — which means you are, whether you call it that or not, running a sequential statistical test. Classic fixed-sample significance testing (the kind behind a simple "is 0.78 significantly worse than 0.91?" calculation) assumes you look at the data *once*. An `AnalysisRun` that queries a quality score every 5 minutes for an hour and aborts the instant one reading crosses a threshold is looking at the data repeatedly — this is the "peeking problem," and it inflates your false-rollback rate beyond what a naive confidence interval suggests, because with enough repeated looks, pure noise will eventually cross almost any fixed threshold by chance.

Practical mitigations, roughly in order of how much rigor they buy you:

- **Require repeated breaches, not one bad read** (`failureLimit > 0`, or Flagger's `threshold`) — this is already standard practice in this chapter's examples, and it's a crude but effective defense against a single noisy measurement triggering rollback.
- **Widen the confidence margin as a function of how many looks you'll take.** A quality gate checked 12 times over an hour needs a stricter per-check threshold than one checked once, if you want the same overall false-rollback rate — a Bonferroni-style correction is a blunt but defensible way to reason about this out loud in an interview.
- **Prefer group-sequential or always-valid testing methods if you're building this properly.** These are designed exactly for "check repeatedly, stop as soon as you're confident" scenarios (this is standard territory in A/B-testing platforms and clinical-trial-style sequential analysis) and control the false-positive rate under repeated looks, unlike a naive fixed threshold checked on a loop.
- **Don't conflate "the gate uses a threshold" with "the gate is statistically rigorous."** A candidate who says "we check the score five times and require two breaches" has a reasonable *engineering* answer; a candidate who can also say "and we're aware that's a repeated-testing problem, so we've widened the threshold / used a sequential test to compensate" is showing they understand *why* the pitfall in Mechanism 3 (flaky rollbacks) happens at a statistical level, not just that it happens.

None of this changes the YAML in this chapter — `failureLimit`/`threshold` already exist specifically to blunt this problem — but understanding *why* they exist, rather than treating them as an arbitrary knob, is exactly the depth a senior interviewer is probing for when they ask "how do you know your threshold is right?"

> **Saying it out loud.** Here's the subtlety that separates a good answer from a great one. When you configure a metric with an interval, a count, and a failure limit, you are running a sequential statistical test whether you call it that or not. Classic significance testing assumes you look at the data *once*. An analysis run that queries a quality score every five minutes for an hour and aborts the instant a reading crosses a threshold is looking twelve times — that's the peeking problem, and it inflates your false-rollback rate, because with enough looks pure noise will eventually cross almost any fixed threshold. The mitigations: require repeated breaches rather than one bad read, widen the per-check threshold as a function of how many looks you'll take, or use a properly sequential test. `failureLimit` exists specifically to blunt this — knowing *why* is the depth being probed.

---

## Shadow / mirror deployments

Shadowing (a.k.a. mirroring, dark launch) sends the new version a **copy** of real requests and **throws away its responses**. Users are served *entirely* by stable; the canary sees production-distribution traffic with zero user risk. This is the safest possible way to evaluate a scary model change.

> **Saying it out loud.** Shadowing sends the new version a *copy* of real requests and throws its responses away. Users are served entirely by stable; the canary sees production-distribution traffic — real prompt lengths, real adversarial inputs, real burstiness — with literally zero user risk. That makes it the safest possible way to evaluate a scary model change, and the strongest quality comparison available, because you can diff v1 and v2 outputs on *identical* live inputs rather than on statistically similar samples. The cost is that you're running a full second inference path, so you're doubling GPU spend for the duration. The mature pattern is shadow, then canary, then promote: shadow catches gross regressions at zero user risk, and only the survivors graduate to real users.

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

> **Saying it out loud.** Istio expresses mirroring as a `mirror` destination on the route plus a `mirrorPercentage`, and the important semantics are that the mirrored request is fire-and-forget — the response is discarded and the mirror's latency doesn't affect the user's request at all. Two details that matter operationally. The mirror percentage lets you shadow ten or twenty percent rather than the full hundred, which for paired-diff quality comparison is usually plenty and costs a fraction of the GPU. And Istio adds a suffix to the mirrored request's Host header, which is your hook for making downstream services shadow-aware — because a mirrored request that reaches a real tool call or a real database write is a production incident, not a test.

### What shadow buys you, and what it can't

Shadow gives you real prompts against the new model with no user exposure — perfect for:

- Comparing v1 vs v2 outputs on *identical* live inputs (paired diffing → the strongest quality comparison you can get).
- Load/soak testing on real traffic shape (bursty, long-context, adversarial) that synthetic tests miss.
- Warming caches and JIT/compilation before real traffic arrives (see cache warmup below).

What it cannot do:

- **Anything with side effects.** If the model call writes to a DB, calls a tool, sends an email, or bills a token budget, the *shadow* will do it too unless downstream services are shadow-aware (check that `-shadow` header and no-op). A mirrored request that triggers a real tool call is a production incident. Make write paths shadow-aware or don't mirror them.
- **Measure user-facing outcomes.** Shadow responses are discarded, so you get model-quality signals but not click-through, thumbs-up, or downstream conversion. Those need a real canary.

**The mature pattern: shadow → canary → promote.** Shadow the risky checkpoint to catch gross regressions with zero risk; the survivors graduate to a small canary with real users and outcome metrics; the ones that pass promote.

> **Saying it out loud.** Shadow is great for three things: paired diffing of v1 versus v2 on identical live inputs, which is the strongest quality comparison you can get; soak testing on real traffic shape that synthetic tests never reproduce; and warming caches before real traffic arrives. Two hard limits. It cannot do anything with side effects — if the model call writes to a database, invokes a tool, or bills a token budget, the shadow does it *too* unless downstream services check the shadow header and no-op. A mirrored request triggering a real tool call is an incident, not a test. And it cannot measure user-facing outcomes at all: responses are discarded, so you get quality signals but never click-through, thumbs-up, or conversion. Those need a real canary with real users.

### Sizing the canary: a back-of-envelope cost model

Every extra GPU-hour a canary or shadow deployment burns is a real line item, so it's worth being able to reason about the cost out loud rather than just asserting "small canary, cheap." A simple model: if your stable fleet is \\(N\\) replicas of a GPU class costing \\(c\\) per hour, a canary at replica count \\(k\\) held for \\(h\\) hours costs \\(k \cdot c \cdot h\\) — independent of its traffic *weight*, because the weight controls how much traffic it gets, not how many GPUs it occupies. This is precisely why pitfall 4 (cold-start skew from too few canary replicas) and cost containment pull in opposite directions: a 1-replica canary is cheap but structurally slower per-request than a 12-replica stable fleet, which can look like a latency regression that isn't one; a canary sized to match stable's per-replica load costs proportionally more.

A worked example: stable is 12 replicas of an 8×H100 node-class instance at, say, \\(c\\) = $28/hour effective cost, running continuously. A canary at 2 replicas (roughly matching per-replica load if canary traffic is capped around 15–20% during the early steps) for a 3-hour ladder (1%→5%→25%→50%→100% with the dwell times from Mechanism 3) costs \\(2 \times 28 \times 3 = \\$168\\) — a small, bounded, one-time cost per rollout. Shadowing the same checkpoint first, mirroring 100% of traffic for a 1-hour soak at the same 2-replica canary size, adds another \\(2 \times 28 \times 1 = \\$56\\). Compare that to the cost of *not* canarying: the "regressed on the 5%" case study above ran for four days at full production traffic before anyone noticed, on a fleet sized for 100% of load — orders of magnitude more expensive in both direct GPU cost and reputational cost than the canary that would have caught it. The point isn't the specific dollar figures — it's that the canary/shadow tax is a small, bounded, up-front cost, and the failure-to-canary tax is unbounded and paid in production.

Two knobs matter most for keeping the bounded side small: **dwell time** (don't let steps sit longer than needed to accumulate a statistically sound sample — see the peeking-problem discussion in Mechanism 3) and **mirror percentage** for shadow (mirroring 10–20% of traffic instead of 100% is usually enough for paired-diff quality comparison, at a fraction of the GPU cost of a full mirror).

> **Saying it out loud.** The cost model is simple and the key insight is counterintuitive: a canary's cost depends on its *replica count*, not its traffic weight — weight controls how much traffic it gets, not how many GPUs it occupies. So a one-percent canary on two replicas costs exactly the same as a fifty-percent canary on two replicas. Concretely, two replicas of an 8xH100 node-class instance at roughly $28 an hour effective, held for a three-hour ladder, is about $168 per rollout, plus another $56 for a one-hour shadow soak — those are of-a-date figures and GPU pricing moves. Compare that to the alternative: the incident later in this chapter ran a regression at full production traffic for four days. The canary tax is small, bounded, and paid up front; the failure-to-canary tax is unbounded and paid in production.

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

**9. The peeking problem — repeated statistical looks inflating false rollbacks.** Checking a noisy quality metric many times over a dwell window is a sequential test, not a single fixed-sample test, and naive fixed thresholds checked on a loop roll back good models more often than the raw confidence interval suggests. *Mitigation:* require repeated breaches (`failureLimit`/`threshold` > 0), widen per-check thresholds as a function of how many looks you'll take, or use a group-sequential/always-valid testing approach if you're building this at scale. See Mechanism 3 for the full discussion.

**10. Sampling proportional to traffic, not proportional to risk.** An online judge or eval-suite sample drawn in proportion to live traffic mix inherits the traffic distribution's blind spots — a rare-but-important query category (a 5% long-tail intent, a long-context conversation, an edge-case language) gets a proportionally thin slice of the analysis sample, so a regression confined to that category hides behind a healthy aggregate score. *Mitigation:* stratify the quality sample with a guaranteed minimum count per category that matters, not a fair share of the aggregate — see the "regressed on the 5% no one canaries" case study, next.

**11. Engine-level state that doesn't carry across versions.** Modern serving engines (vLLM, SGLang, TensorRT-LLM) keep substantial engine-level state — paged KV-cache blocks, prefix/radix caches, continuous-batching scheduler queues — that is specific to a running engine instance and is **not portable across a version bump**, even a "minor" one. A new engine version with a different attention kernel, a different default block size, or a changed scheduler policy starts every canary pod with none of that warm state, which compounds Failure Mode 4: it's not just the *model weights* that are cold, the *serving engine* itself hasn't built up the request-shape-specific scheduling behavior (batch-size heuristics, cache eviction patterns) that steady-state stable has accumulated. *Mitigation:* when the canary is an engine/runtime upgrade rather than a pure checkpoint swap, budget extra warmup time proportional to how different the new engine's caching/batching behavior is, and consider shadowing specifically to pre-populate prefix caches with your production prompt-prefix distribution before any user traffic hits the canary.

> **Saying it out loud.** The recurring ways model canaries go wrong. Quality regressions sailing through green infra gates — the headline one. Gating quality too early on too few samples, producing flaky rollbacks that teach people to ignore the gate. Cold-start and topology skew, where a one-replica canary against a twelve-replica stable fleet looks slower for structural reasons that have nothing to do with the model. Session splitting, where a multi-turn conversation gets served by two versions. Sampling that inherits your traffic distribution's blind spots. Decommissioning the old fleet immediately on promotion, which throws away your cheap rollback path. And the eval service becoming a single point of failure whose outage reads as a model regression. The pattern: most of these make the *canary* wrong, not the model.

---

## Production case studies & war stories

### Case: the "regressed on the 5% no one canaries" incident

The following is a composite, but it is an extremely common shape — variations of it are the concrete reason the MLflow and arXiv sources cited in The 2025–2026 landscape exist, and most engineers who have run model canaries for more than a year have lived some version of it.

**Setup.** A customer-support chat assistant serves three broad query shapes: general product questions (~70% of traffic), billing/account questions (~25%), and a long tail of multi-step troubleshooting conversations (~5%) that involve several turns of the user providing diagnostic details before the model proposes a fix. The team ships `v7`, a new base-model checkpoint, mainly to cut p50 latency and cost. It goes through the canary ladder from Mechanism 3:

| Step | Weight | Gate | Result |
|---|---|---|---|
| 1 | 1% | infra only (smoke) | pass — up, responding |
| 2 | 5% | infra + cheap proxies (refusal rate, empty-completion rate) | pass — both flat vs v6 |
| 3 | 25% | infra + online judge sample (general Q&A prompts only) | pass — judge score 0.89 vs 0.87 baseline, *v7 looks better* |
| 4 | 50% | infra + full eval-suite job | pass — the eval suite's fixed prompt set is dominated by general-Q&A-style prompts, mirroring the 70% traffic mix |
| 5 | 100% | promote | promoted |

Every gate in this chapter's own AnalysisTemplate examples was green. p95 latency actually *improved* by 80ms. The online-judge sample at step 3 sampled *proportionally to traffic*, so it drew mostly general-Q&A turns — exactly the class `v7` was good at — and almost no multi-step troubleshooting conversations, because 25% of traffic times a further sampling rate leaves very few multi-turn troubleshooting sessions in any given analysis window.

**What actually happened.** `v7` had a subtle regression specific to multi-step troubleshooting: it was slightly more prone to *losing track of which diagnostic step the user was on* after 4+ turns, occasionally re-asking a question the user had already answered two turns earlier. This is exactly the "context preservation" dimension the arXiv quality-gate paper calls out as a distinct axis from task success — a model can nail single-turn task success while regressing on multi-turn coherence, and a judge sample skewed toward single-turn prompts will never see it.

It surfaced four days after full promotion, via a spike in support-escalation tickets tagged "assistant repeated itself" — a lagging, human-reported signal, not a canary metric. By then `v6` had been fully decommissioned per the "scale down promptly after promotion" guidance, so the fix required re-deploying `v6` from the image registry (fast) and re-running the canary ladder for a patched `v7.1` (slow) rather than an instant weight-based rollback.

**Root causes, and the fix for each:**

1. **Stratified sampling was missing from the judge/eval step.** The fix: the eval-suite job and the online-judge sampler were changed to sample a *fixed minimum count per query category* (general, billing, multi-step-troubleshooting), not proportional-to-traffic — mirroring the "four-tier stratification" idea from the arXiv paper (functional / orchestration / edge-case / adversarial), so a 5%-of-traffic category still gets meaningfully evaluated instead of being drowned out.
2. **The eval-suite's fixed prompt set didn't include multi-turn conversations at all** — it had grown organically from single-turn Q&A pairs. The fix: every retro after this incident, a new failing conversation gets added to the permanent eval suite as a regression test, the same "living question bank fed by post-mortems" pattern the quality-gate paper recommends.
3. **Decommissioning `v6` immediately on promotion removed the cheap rollback path.** The fix: keep the previous stable version's fleet at a small warm standby (not zero) for a defined bake period (e.g. 72 hours) after full promotion, specifically to cover "regression surfaces after promotion, before decommission" — cheap insurance against exactly this failure.
4. **No metric existed for "context preservation" at all** — only task-level judge scores. The fix: added a specific multi-turn-coherence check (a judge prompt that scores whether the assistant references only correctly-carried context across a conversation) as its own AnalysisTemplate metric, independent from the general quality score, per the "different metrics catch different failure modes" finding above.

**The lesson, restated for interviews:** a quality gate that samples *proportionally to traffic* inherits your traffic distribution's blind spots. Rare-but-important query classes need guaranteed representation in the eval sample, not a fair share of it — and "the eval suite passed" is only meaningful to the extent the eval suite actually contains the failure mode you're worried about. This is a sharper, more specific version of the chapter's opening claim: it's not merely "infra gates miss quality regressions," it's "even a quality gate misses regressions the *eval data* doesn't represent."

> **Saying it out loud.** A support assistant serving 70% general questions, 25% billing, and a 5% tail of multi-step troubleshooting. A new checkpoint went up the full ladder — 1%, 5%, 25% with a judge sample, 50% with the full eval suite — and passed every gate; p95 latency even improved 80 milliseconds. But the judge sampled *proportionally to traffic*, so it drew almost entirely general Q&A, the exact class the new model was good at. The regression was in multi-turn context preservation: after four-plus turns it started re-asking questions the user had already answered. It surfaced four days later through support tickets, and because the old fleet had been decommissioned on promotion, there was no instant rollback. The sharpened lesson: it's not just that infra gates miss quality regressions — even a quality gate misses regressions the *eval data* doesn't represent. Rare classes need guaranteed representation, not a fair share.

### Shorter incident notes worth knowing

- **Cold-start latency masquerading as a real regression.** A team's canary at 5% weight failed the p95-latency gate every time, on every version, canary or not — because the canary ReplicaSet only had 1 replica against 12 stable replicas, so its per-pod queueing depth was structurally worse regardless of model quality, and its KV-cache was permanently cold (never warm enough between analysis windows to amortize). Not a model problem — a topology problem. Fix: size canary replica count so per-replica load is comparable to stable, and pre-warm before the first analysis window (see Failure Mode 4, above).
- **The webhook quality gate became a single point of failure.** A `quality-eval-service` outage (unrelated to the model) caused every canary's `web`/webhook metric to fail, which correctly aborted rollouts — but on-call initially assumed the *model* was regressing, and wasted an hour bisecting a bad checkpoint. Fix: alert on and dashboard the eval service's own health distinctly from "canary failed," so a dependency outage reads as a dependency outage, not a false model regression.
- **A quantized canary that "passed" on aggregate but not on long context.** A team shipped an INT4-quantized version of an existing checkpoint purely for cost savings, canaried it on latency/error/refusal-rate — all fine — and promoted. Weeks later, an analysis of downstream conversion showed a small but real drop confined to conversations over ~6k tokens of context, where quantization error compounded enough to nudge factual answers wrong more often. The team's eval sample, like most, skewed toward shorter interactions. Fix: when the change under canary is specifically a compression/quantization technique, add a context-length-stratified quality check (short/medium/long) rather than relying on an aggregate score, since compression artifacts are exactly the kind of regression that scales with sequence length.

> **Saying it out loud.** Three shorter ones worth internalizing. A canary that failed the p95 latency gate on *every* version because it had one replica against twelve stable ones — structurally worse queueing and a permanently cold KV cache, so it was a topology problem masquerading as a model problem; the fix is sizing canary replicas so per-replica load is comparable, and pre-warming before the first analysis window. An eval-service outage that correctly aborted every rollout, while on-call spent an hour bisecting a checkpoint that was fine — so dashboard the eval service's health separately from "canary failed." And an INT4-quantized canary that passed on aggregate but had a real regression confined to conversations over about six thousand tokens, because quantization error compounds with sequence length and the eval sample skewed short.

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

## Interview mastery

### "Explain why model canaries need quality gates, not just latency, in 60 seconds"

A tight answer, timed:

> "A normal canary watches error rate, latency, and saturation — the response *envelope*. That's necessary but not sufficient for models, because a new checkpoint can return HTTP 200, at lower latency, on every request, while the response *body* quietly gets worse — more hallucination, more wrongful refusals, worse multi-turn coherence, tone drift. None of that shows up as a 5xx or a slow request. So a model canary needs a second class of gate that actually scores the generated output — cheap proxy signals like refusal rate or empty-completion rate as a first line, and an LLM-judge or reference-based eval running on a traffic sample as the real gate — wired into the same automated promote/rollback loop as the infra metrics, with enough dwell time and sample size that the quality signal isn't noise. Skip that, and you've built a very fast, very reliable way to ship regressions with high confidence."

That's the whole chapter in one breath: envelope vs. body, cheap proxies vs. real judge, wired into automation, sized for statistical validity.

### Q&A bank (18 questions, roughly ordered easy → hard)

1. **What's the difference between a rolling update, blue-green, canary, and shadow deployment?** Blast radius and cost tradeoffs — rolling has unbounded/growing exposure and slow rollback; blue-green is atomic with 2x cost during overlap; canary bounds exposure to a weight and rolls back by re-zeroing that weight; shadow has zero user exposure but doubles inference cost and can't measure real user outcomes.
2. **Why is a bare Kubernetes Service + replica-ratio a bad way to canary a model?** Weight is quantized by replica count (can't cheaply do 1% with expensive GPU pods), weight is coupled to capacity, and there's no clean single-object rollback primitive — use Istio/Gateway API weighted routing instead.
3. **In Gateway API, is `weight: 10` a percentage?** No — it's a proportion. `weight: 90` / `weight: 10` gives 10%, but so would `9`/`1`. The sum of weights in the rule is the denominator.
4. **Why must multi-turn chat sessions be pinned to one model version?** Weighted routing splits per-*request*, not per-*conversation*; a session that flips versions mid-conversation is incoherent to the user and corrupts per-version quality attribution. Route by a stable session/user key instead of pure weight for anything stateful.
5. **What's the core failure mode this whole topic exists to prevent?** A new model version that passes every infra signal (error rate, latency, saturation) while regressing on output quality — hallucination, refusal, tone, multi-turn coherence — because infra metrics read the response envelope, not the body.
6. **Name three ways to get a quality signal into an automated canary analysis, cheapest to most expensive.** (A) proxy counters already emittable to Prometheus — refusal rate, empty-completion rate, mean output length/logprob; (B) an online judge/eval job scoring a sample of canary responses and pushing a gauge; (C) a synchronous webhook/job that runs a full eval suite and gates on a JSON/HTTP-status result before promoting.
7. **How do Argo Rollouts and Flagger differ architecturally?** Argo replaces your Deployment with a `Rollout` CRD holding two ReplicaSets and drives imperative `steps` (setWeight/pause/analysis) with first-class `AnalysisTemplate`s; Flagger leaves your Deployment untouched and creates a shadow "primary" deployment alongside it, driven by a declarative `Canary` CRD with metrics + webhooks. Argo defaults to more manual control; Flagger defaults to more automation out of the box.
8. **What exactly triggers an automatic rollback in Argo Rollouts?** A metric's `count`/`interval` measurements are taken; each measurement is checked against `successCondition`/`failureCondition`; if the number of failing measurements exceeds `failureLimit`, the whole `AnalysisRun` fails, Argo sets canary weight back to 0, and marks the Rollout `Degraded`. In Flagger, it's the built-in `threshold` counter across all metrics/webhooks — cross it and Flagger scales the canary to zero and routes 100% back to primary.
9. **Why must rollback be "reset a weight," not "redeploy"?** Because the fast part of rollback is *not creating anything new* — the stable fleet is already running and warm, so flipping traffic back to it is near-instant. If you'd already scaled stable down, "rollback" means cold-starting GPU pods during a live incident, which is exactly what you were trying to avoid.
10. **How do you avoid flaky rollbacks from noisy quality metrics?** Size the dwell time and traffic weight so each analysis window accumulates enough judged samples for a stable estimate; require repeated breaches (`failureLimit`/`threshold` > 0) rather than one bad read; consider a combined `successCondition` that asserts both the score *and* a minimum sample count (see the `result.n_samples >= 100` example earlier) so a thin-sample bad read can't fail the gate on its own.
11. **What can shadow/mirror traffic not tell you, and why mirror at all if the response is thrown away?** It can't measure real user-facing outcomes (clicks, thumbs-up, conversions) since users never see canary responses — but it gives you paired real-traffic comparisons and load/soak testing with zero risk, and it's the safest way to pre-screen a scary architecture/checkpoint change before it ever gets a canary weight.
12. **What's the side-effect hazard with shadow traffic, specifically?** If serving the request has side effects — a tool call, a DB write, sending an email, decrementing a billing quota — the mirrored copy will trigger them too unless the downstream is shadow-aware (e.g. checks Istio's appended `-shadow` Host/Authority header and no-ops). A mirrored request that actually calls a payment API is a production incident, not a test.
13. **When would you choose blue-green over canary for a model rollout?** When a *mixed* population of v1/v2 responses is actually incoherent or unsafe — e.g. a tokenizer or prompt-format change where any user seeing a blend of formats mid-session is worse than a clean atomic cutover gated on a full offline eval pass, even though blue-green costs a full second fleet during the overlap.
14. **What's the "cold start" pitfall and how do you defend against it?** A freshly started canary pod has a cold KV/prefix cache and possibly uncompiled kernels, so its first minutes show inflated latency/lower throughput unrelated to model quality — a naive analysis window right after `setWeight` rolls back a fine model. Defend with a warmup pause before the first analysis, readiness probes that gate on post-warmup state, and pre-warming via shadow traffic.
15. **How would you decide where in the canary ladder to first apply a quality gate (1%? 5%? 25%?)** Based on how many judged/scored samples that weight and dwell time will actually produce — gate quality once the analysis window can gather enough samples for a stable estimate (often not until 25%+ of traffic for a rare traffic pattern), and use cheaper proxy signals at the earliest, thinnest-traffic steps.
16. **Why does sampling proportional to traffic mix create a blind spot for quality gates?** Rare-but-important query categories (a 5% long-tail intent, or long-context conversations under a quantization change) get a proportionally thin slice of the analysis sample, so a regression confined to that category can hide behind a healthy aggregate score. Fix with stratified/guaranteed-minimum sampling per category rather than pure proportional sampling — see the war stories above.
17. **What's the difference between Argo's `web` metric provider and Flagger's `rollout`-type webhook, practically?** Argo's `web` provider lets `successCondition` assert directly on fields in the JSON response (`result.quality_score`, `result.n_samples`), so sample-size and score logic can both live in the CRD; Flagger's webhook contract is a bare HTTP status code (2xx = pass), so the scoring and sample-size logic must live inside the webhook handler itself.
18. **If your quality-eval service itself goes down, what happens to your canary — and is that the right behavior?** Its metric/webhook check fails, which (correctly, conservatively) aborts the rollout — a missing quality signal should not default to "assume it's fine." The operational gotcha is distinguishing "eval service is down" from "model regressed" quickly, which means alerting on the eval service's own health separately from canary-failure alerts.

### System-design prompt: "Design the rollout process for a new model version at a company with strict SLAs"

A sketch of a strong answer, roughly in the order an interviewer wants to hear it. Start with the shape of the pipeline, then narrate each box:

```
                         ┌───────────────────────────┐
                         │   Offline eval + shadow    │
                         │  (stratified prompt suite, │
                         │   mirrored live traffic,   │
                         │   zero user exposure)      │
                         └─────────────┬─────────────┘
                                       │ survivors only
                                       ▼
        ┌────────────────────────────────────────────────────────┐
        │              Weighted traffic split (HTTPRoute /       │
        │              VirtualService / InferenceModel)          │
        │                                                        │
        │   1% ──10m──▶ 5% ──15m──▶ 25% ──30m──▶ 50% ──60m──▶ 100%│
        │    │            │           │            │              │
        │  smoke        infra +     infra +      infra +          │
        │  (infra       cheap       judge        blocking         │
        │   only)       proxies     sample       eval-suite       │
        └───────┬──────────┬───────────┬────────────┬─────────────┘
                │          │           │            │
                ▼          ▼           ▼            ▼
         ┌────────────────────────────────────────────┐
         │   AnalysisTemplate / Canary CRD:            │
         │   Prometheus (success rate, p95/p99 lat.)   │
         │   AND quality-eval webhook (score + n)      │
         │   -> any breach = automatic rollback         │
         │      (weight -> 0, stable never scaled down)│
         └────────────────────────────────────────────┘
                                       │ all steps pass
                                       ▼
                         ┌───────────────────────────┐
                         │  Promote + warm-standby    │
                         │  bake period for old        │
                         │  version (hours-days)       │
                         │  before full decommission   │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │  Any prod incident feeds    │
                         │  back into the eval suite   │
                         │  as a new regression test   │
                         └───────────────────────────┘
```

**1. Clarify the SLA and blast-radius constraints first.** What's the latency SLA (p95/p99, in ms)? What's the error-budget? Is this a multi-turn conversational product (session pinning required) or stateless request/response? Is a mixed-version user experience acceptable at all, or does this specific change (tokenizer, prompt format) require atomicity?

**2. Pre-production: shadow before anything touches a real user.** Mirror a sample of live traffic to the new version, discard its responses, diff v1 vs v2 outputs on identical inputs, and run it against the full offline eval suite (stratified across query categories, including the rare-but-important ones — see the war stories above). This is the cheapest place to catch a gross regression, at zero user risk.

**3. Progressive canary via a weighted traffic split** (Gateway API `HTTPRoute` or a mesh `VirtualService`, or a model-aware `InferenceModel` split if you've adopted the Gateway API Inference Extension; Argo Rollouts or Flagger driving it), with:
   - A ladder of weights and dwell times sized so each step accumulates enough samples for the *thinnest* traffic category you care about, not just the aggregate.
   - **Two classes of gate at every step past the smoke-test step:** infra (success rate, p95/p99 latency, saturation) from Prometheus, **and** quality (cheap proxies early, judge/eval-webhook score once volume supports it) — both required to pass, neither sufficient alone.
   - Session/user-key pinning so a conversation never straddles versions.
   - A pre-warm step (readiness gate or shadow-fed warmup) before the first analysis window, to avoid cold-cache false rollbacks.

**4. Automatic rollback as the default failure path**, with rollback = re-zero the weight against a fleet that was never scaled down — not a redeploy. A human-approval `pause` before the final 100% step is reasonable for high-stakes changes; the *failure* path must never wait on a human.

**5. Post-promotion bake, not instant decommission.** Keep the previous stable version warm at reduced replica count for a defined window (hours to a few days depending on how rare your riskiest query category is) so a regression that surfaces late still has a fast rollback path, before fully scaling the old fleet to zero.

**6. Feed every incident back into the eval suite.** Any regression caught in production — by a user report, an escalation, anything — becomes a permanent regression test in the offline suite and, if it's a new *category* of failure, a new stratified sample bucket in the online quality gate. The eval suite is a living artifact, not a fixed asset.

**What a strong candidate says that a weak one doesn't:** naming the sample-size/stratification problem unprompted, insisting quality gates are "and" not "or" with infra gates, and explaining *why* rollback is fast (stable never scales down until confirmed) rather than just asserting it is.

> **Saying it out loud.** I'd narrate it as a funnel. Offline eval against a *stratified* prompt suite first, then a shadow soak on mirrored live traffic — zero user exposure, paired diffing against stable, and only survivors move on. Then a weighted canary ladder through Gateway API or a mesh: 1%, 5%, 25%, 50%, with dwell times long enough to accumulate a real sample at each. Every step gated on infra *and* a task-specific quality signal, both must pass, with automatic rollback that's just setting the weight to zero. Session-pinned routing so multi-turn conversations don't split across versions. And after promotion, keep the old fleet on warm standby for a defined bake period — 72 hours — because the regression that surfaces on day three needs an instant rollback, not a redeploy.

### Red flags vs. green flags

| Signal in a candidate's (or a team's) design | Red flag | Green flag |
|---|---|---|
| Canary gate composition | Only latency/error-rate/saturation | Infra metrics **and** a model-quality signal, both blocking |
| Quality sampling | Proportional to live traffic mix | Stratified with a guaranteed minimum per query category |
| Rollback mechanism | "Redeploy the old version" | "Re-zero a weight against a fleet that's still warm" |
| Session handling | Pure weighted routing for chat | Stable key (session/user) pinning per conversation |
| Cold start | Analysis starts the instant weight shifts | Warmup/pre-load window before the first analysis |
| Shadow traffic and side effects | Mirrors write paths unconditionally | Downstream is shadow-aware (checks `-shadow` header) or write paths are excluded from mirroring |
| Sample-size discipline | Gates on a raw score alone | Gates on score **and** a minimum sample count |
| Decommission timing | Old fleet scaled to zero immediately on promotion | Bake period at reduced replica count before full decommission |
| Eval suite evolution | Fixed prompt set, never updated | Living suite fed by every production incident/post-mortem |
| Eval-service failure handling | Eval outage silently treated as "pass" or conflated with a model regression | Eval-service health monitored/alerted separately from canary-failure alerts |
| Tooling choice justification | "We use Argo Rollouts" (no why) | Explains the Argo-vs-Flagger tradeoff (imperative/manual vs declarative/automated) relative to their team's needs |

---

## Quick reference: cheap quality proxies to instrument today

Before you can gate on an expensive judge score (Pattern B/C), you need Pattern A's cheap proxies actually emitting metrics your server can scrape. This is the fastest, lowest-effort improvement most teams can make to an infra-only canary — none of these require a judge, a reward model, or an eval job, just counters in your inference server.

| Proxy metric | What it approximates | How to emit it | A reasonable starting threshold |
|---|---|---|---|
| Refusal rate | Model wrongly declining valid requests | Classify output against a refusal-phrase list or a small classifier at generation time; increment a counter | fail if `> 3%` above stable's baseline |
| Empty / truncated completion rate | Degeneration, hitting max-tokens on non-trivial prompts, decoding failures | Counter on `len(output) == 0` or `finish_reason == "length"` on prompts that shouldn't need it | fail if `> 1–2%` |
| Mean output token count | Truncation or verbosity drift vs. baseline | Histogram of completion token counts | fail if it moves `> 20%` either direction vs. stable |
| Mean logprob / perplexity of the generated sequence | Gross fluency or confidence collapse | Sum/average token logprobs the server already computes during generation | fail on a large negative shift vs. stable's rolling baseline |
| Guardrail / safety-filter trigger rate | Safety regressions, not just refusals | Counter on your existing content-filter or moderation layer firing | fail if `> 2×` stable's baseline rate |
| Repetition / degenerate-loop rate | n-gram repetition, a classic quality failure mode independent of refusal | Simple n-gram-repeat detector over the output, incremented as a counter | fail if `> 1%` |
| Tool-call malformation rate (agentic serving) | Broken function-calling / tool-use output that infra metrics never see (a 200 with unparseable JSON is still a 200) | Counter on JSON-schema/tool-call parse failures | fail if `> 0.5%` |

None of these require you to change or slow down the response path — they're counters incremented next to logging, scraped by the same Prometheus that already backs your infra `AnalysisTemplate`. Wire two or three of these into the earliest canary steps (1–5% weight) as Pattern A, and reserve the expensive judge/eval-suite gate (Pattern B/C) for the steps with enough traffic volume to support it — this is exactly the ladder from Mechanism 3.

> **Saying it out loud.** If you take one action from this chapter, it's this: instrument the cheap quality proxies before you build anything sophisticated. Refusal rate on valid prompts. Empty or near-empty completion rate. Mean and p95 output token count, as a proxy for truncation and degeneration. Guardrail-filter trigger rate. Mean token logprob, as a crude confidence signal. Every one of those is a Prometheus counter your server can emit today with no judge model, no eval service, and no extra GPU. They won't catch a subtle factuality regression — but they *would* have caught the 6.5% refusal-rate jump from the opening example, and a gate that catches gross regressions today is worth more than a perfect gate you're still designing next quarter.

---

## Glossary: fields and terms used in this chapter

A field-by-field lookup for the YAML above — useful when you're skimming back through this chapter mid-incident and need the exact knob, not the prose around it.

| Term | Tool / layer | Meaning |
|---|---|---|
| `weight` | Istio `VirtualService`, Gateway API `HTTPRoute` | Proportional traffic share; the sum of weights in a rule is the denominator, not 100 by convention |
| `mirror` / `mirrorPercentage` | Istio `VirtualService` | Sends a copy of traffic to a subset and discards the response; percentage of requests copied |
| `setWeight` | Argo Rollouts `Rollout` step | Imperative step that sets the canary traffic weight to a specific value |
| `pause` | Argo Rollouts `Rollout` step | Holds at the current weight for a fixed duration, or indefinitely (manual gate) if empty |
| `analysis` (step) | Argo Rollouts `Rollout` | Runs one or more `AnalysisTemplate`s at the current weight before advancing |
| `AnalysisTemplate` / `AnalysisRun` | Argo Rollouts | Reusable metric-check bundle (template) and its live execution instance (run) |
| `interval` | Argo Rollouts metric | How often a metric is queried during an `AnalysisRun` |
| `count` | Argo Rollouts metric | Total number of measurements to take; analysis runs for `count × interval` |
| `successCondition` / `failureCondition` | Argo Rollouts metric | Boolean expression evaluated against `result` from the provider |
| `failureLimit` | Argo Rollouts metric | Number of failed measurements tolerated before the whole `AnalysisRun` fails |
| `provider: prometheus` | Argo Rollouts metric | Metric source is a PromQL query against a given Prometheus address |
| `provider: web` | Argo Rollouts metric | Metric source is an HTTP call; `jsonPath` extracts the value(s) bound to `result` |
| `provider: job` | Argo Rollouts metric | Metric source is a Kubernetes Job's exit code / output, for heavyweight eval suites |
| `jsonPath` | Argo Rollouts `web` provider | JSONPath expression selecting a scalar or object from the HTTP response body |
| `canaryService` / `stableService` | Argo Rollouts `Rollout` | Services scoped to only the canary or only the stable ReplicaSet, for unambiguous metric attribution |
| `stepWeight` | Flagger `Canary` | Fixed traffic-percentage increment applied each successful `interval` |
| `maxWeight` | Flagger `Canary` | Ceiling on canary traffic before Flagger promotes to 100% |
| `threshold` | Flagger `Canary` | Number of failed checks (metrics or webhooks combined) tolerated before automatic rollback |
| `thresholdRange` | Flagger metric | `{min, max}` acceptable range for a built-in or custom metric |
| `MetricTemplate` | Flagger | Custom PromQL metric definition referenced by name from a `Canary`'s `metrics` list |
| `webhooks[].type: pre-rollout` | Flagger | Hard gate that must pass before any traffic shifts at all |
| `webhooks[].type: rollout` | Flagger | Gate re-checked at every weight step during the rollout |
| `canaryTrafficPercent` | KServe `InferenceService` | Native traffic-split field on a predictor component; serverless-mode only |
| `latestRolledoutRevision` / `previousRolledoutRevision` | KServe `InferenceService` status | Pointers KServe uses to track which revision is "stable" and which to roll back to |
| `InferencePool` | Gateway API Inference Extension | Groups model-server replicas with cache/queue-aware load balancing, analogous to a `Service` |
| `InferenceModel` / `InferenceObjective` | Gateway API Inference Extension | Model-identity routing in front of an `InferencePool`; carries `targetModels`/`weight` for canarying by model version |
| `-shadow` header suffix | Istio mirroring | Appended to `Host`/`Authority` on mirrored requests so shadow-aware downstreams can detect and no-op |
| `Degraded` (Rollout status) | Argo Rollouts | State a `Rollout` enters when an `AnalysisRun` fails, signaling automatic rollback has occurred |
| `backoffLimit` | Kubernetes `Job` (used by Argo's `job` metric provider) | Number of retries for a failed eval-suite Job before it's treated as a hard failure |

---

## Further reading

- Argo Rollouts — Analysis, AnalysisTemplate & metrics: <https://argo-rollouts.readthedocs.io/en/stable/features/analysis/>
- Argo Rollouts — Web metric provider spec (jsonPath, successCondition on object fields): <https://argo-rollouts.readthedocs.io/en/stable/analysis/web/>
- Argo Rollouts — Canary strategy & steps: <https://argo-rollouts.readthedocs.io/en/stable/features/canary/>
- Argo Rollouts — Istio traffic management: <https://argo-rollouts.readthedocs.io/en/stable/features/traffic-management/istio/>
- Argo Rollouts — GitHub releases (v1.9.0, Mar 20 2026; v1.8.4, Feb 13 2026; v1.8.3, Jun 5 2025): <https://github.com/argoproj/argo-rollouts/releases>
- Flagger — How it works (Canary spec, promotion/rollback): <https://docs.flagger.app/usage/how-it-works>
- Flagger — Metrics analysis (MetricTemplate, thresholds): <https://docs.flagger.app/usage/metrics>
- Flagger — Webhooks (pre-rollout gates, load testing): <https://docs.flagger.app/usage/webhooks>
- Flagger — Deployment strategies (canary, blue-green, mirroring): <https://docs.flagger.app/usage/deployment-strategies>
- Flagger — Gateway API canary tutorial (mesh-free progressive delivery): <https://docs.flagger.app/tutorials/gatewayapi-progressive-delivery>
- Flagger — CHANGELOG (v1.42.0, Oct 16 2025: Gateway API v1.4.0, CORS, trafficDistribution; v1.43.0, Apr 21 2026: External Metrics provider, session-cookie naming): <https://github.com/fluxcd/flagger/blob/main/CHANGELOG.md>
- Istio — Traffic shifting (weighted VirtualService): <https://istio.io/latest/docs/tasks/traffic-management/traffic-shifting/>
- Istio — Traffic mirroring (shadow): <https://istio.io/latest/docs/tasks/traffic-management/mirroring/>
- Istio — Bringing AI-aware traffic management to Istio (Gateway API Inference Extension support, InferenceModel weighted canary): <https://istio.io/latest/blog/2025/inference-extension-support/>
- Kubernetes blog — Introducing the Gateway API Inference Extension (Jun 5, 2025): <https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/>
- CNCF — Deep dive into the Gateway API Inference Extension: <https://www.cncf.io/blog/2025/04/21/deep-dive-into-the-gateway-api-inference-extension/>
- Gateway API Inference Extension — source repo (InferencePool, InferenceModel/InferenceObjective): <https://github.com/kubernetes-sigs/gateway-api-inference-extension>
- Gateway API — Traffic splitting (HTTPRoute weights): <https://gateway-api.sigs.k8s.io/guides/traffic-splitting/>
- Gateway API — GEP-1324, service mesh support / GAMMA initiative: <https://gateway-api.sigs.k8s.io/geps/gep-1324/>
- Kubernetes — Deployment rolling updates: <https://kubernetes.io/docs/concepts/workloads/controllers/deployment/>
- Maiorano, "Automated Self-Testing as a Quality Gate for LLM Applications" (arXiv:2603.15676, Mar 2026): <https://arxiv.org/html/2603.15676v1>
- MLflow — "Canary Deployment for AI Models: A 2026 Guide": <https://mlflow.org/articles/what-is-canary-deployment-ai/>
- Buoyant — "Flagger vs Argo Rollouts vs Service Meshes: A Guide to Progressive Delivery in Kubernetes": <https://www.buoyant.io/blog/flagger-vs-argo-rollouts-for-progressive-delivery-on-linkerd>
