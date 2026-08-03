# Drift Detection for Deployed LLMs — A Practical Guide

> Noticing when the inputs your model sees, or the outputs it produces, have quietly stopped looking like what you tested against.

## Why this matters

You shipped a model. Evals were green, latency was fine, the demo delighted the VP. Then three weeks later support tickets spike, a red-team screenshot lands in Slack, and someone asks the question you cannot answer: *did the model change, or did the world change?*

An LLM endpoint is a **static function** deployed into a **non-stationary environment**. The weights are frozen at a checkpoint. But the prompts arriving at 3pm on a Tuesday in month four are drawn from a different distribution than the ones you curated for your eval set. Users discover new use cases. A marketing launch sends a new persona your way. An upstream service starts truncating context. A competitor publishes a jailbreak. None of these touch your weights, and none of them show up in a unit test — but all of them change what your system does in production.

Drift detection is the monitoring discipline that makes this observable. It answers three operational questions:

1. **Are inputs shifting?** (Are we being asked things we were not built for?)
2. **Are outputs shifting?** (Is quality, length, refusal rate, or latency decaying?)
3. **Is the input→output relationship shifting?** (Concept drift — the *right answer* changed even though the question looks the same.)

Get this right and you catch regressions before your users file them. Get it wrong and you either miss real degradation or drown in false alarms until the on-call engineer mutes the channel. Both failure modes are common, and both are avoidable with a small amount of statistics applied with judgment.

This chapter is deliberately intuition-first: every method gets a plain-English picture before a formula, then a worked micro-example with numbers you can reproduce, then the honest caveat about where it breaks.

---

## Core intuition: the model is static, the world is not

Hold one picture in your head for the whole chapter.

At training/eval time you sampled a **reference distribution** \( P_{\text{ref}} \) — the prompts, embeddings, and outputs you validated against. In production you observe a stream that, windowed, gives you a **live distribution** \( P_{\text{live}} \). Drift is simply:

\[
P_{\text{live}} \ne P_{\text{ref}}
\]

Every technique in this chapter is a way to *measure a distance* between these two distributions from finite samples, and then decide whether that distance is large enough to act on. That is the whole game:

- Pick **what** you measure (a feature: prompt length, embedding, refusal flag, latency).
- Pick a **distance / test** (PSI, KS, MMD, embedding distance, a classifier).
- Pick a **window** and a **threshold**.
- Decide **what happens** when the threshold trips.

The subtlety — and where most production systems fail — is not the math. It is choosing a reference window that means something, choosing a live window that is neither too jittery nor too laggy, and resisting the urge to page a human every time noise crosses a line. A drift monitor that fires ten times a week is worse than no monitor, because it trains everyone to ignore it.

One more framing that matters for LLMs specifically: **you usually have no labels.** In classic ML monitoring you eventually learn the ground truth (the loan defaulted, the click happened) and can measure real performance. For a chat endpoint, "was this answer good?" often never arrives, or arrives weeks later as a thumbs-down on 0.3% of turns. So drift detection on the *inputs and the observable outputs* is frequently the **only** early-warning signal you get. That raises its stakes — and it means you must be honest that an input-drift alarm is *smoke*, not a *diagnosis*.

---

## A drift taxonomy

Four kinds of drift, in the order you typically detect them:

| Type | What shifts | LLM example | Typically detected via |
|---|---|---|---|
| **Input / prompt drift** | The distribution of raw inputs \( P(x) \) | Prompts get longer; a new language appears; topic mix changes | PSI / KS on scalar features (length, token count, language ID); topic classifiers |
| **Embedding / semantic drift** | The distribution of inputs (or outputs) in vector space \( P(\phi(x)) \) | Users start asking about a product feature that did not exist at launch | MMD, domain classifier, centroid / cosine distance on embeddings |
| **Output / quality drift** | The distribution of outputs \( P(y) \) or a quality proxy | Answers get shorter, refusals rise, latency creeps up, tone changes | PSI/KS on output length, refusal rate, latency; LLM-judge scores |
| **Concept drift** | The conditional \( P(y \mid x) \) — the correct mapping | "Best model" now points to a newer model; a policy changed so the *right* answer flipped | Requires labels or re-eval; input/output drift can be flat while this moves |

Two things worth internalizing:

- **Input drift and output drift can move independently.** Inputs can look identical while outputs decay (e.g., a silent upstream change to your system prompt, or a provider swapping the model behind an alias). Outputs can look identical while inputs shift (the model gracefully handles new topics — good!). Monitor both, and monitor them separately so you can tell which one moved.
- **Concept drift is the dangerous one and the hardest to see.** \( P(x) \) can be perfectly stable while \( P(y \mid x) \) rots underneath you. Detecting it genuinely requires ground truth or periodic re-evaluation — no unsupervised distance on inputs will find it. Say this out loud in an interview; it separates people who have run monitoring from people who have read about it.

A useful mental decomposition: the joint \( P(x,y) = P(x)\,P(y\mid x) \). Input/embedding drift is a change in \( P(x) \); concept drift is a change in \( P(y\mid x) \). Output drift is a change in the marginal \( P(y) \), which can be caused by *either* — which is exactly why an output-drift alarm alone cannot tell you whether users changed or the model rotted. You have to look at inputs and outputs together.

---

## Detection methods in depth

For each method: the intuition, the precise formula, and a small worked example with real numbers.

### 1. Population Stability Index (PSI)

**Intuition.** Bin a feature. Compare the *share* of traffic in each bin now vs. at reference time. If mass sloshed from one bin to another, PSI grows. It is a symmetric, binned relative-entropy-flavored score, and it is the workhorse of tabular drift monitoring because it produces a single interpretable number with battle-tested thresholds.

**Formula.** With \( B \) bins, reference proportion \( r_b \) and live proportion \( l_b \) in bin \( b \):

\[
\text{PSI} = \sum_{b=1}^{B} \left( l_b - r_b \right)\, \ln\!\frac{l_b}{r_b}
\]

Each term is \( \ge 0 \) (a bin that moves in either direction adds positively), so PSI is a non-negative divergence. It is the symmetrized KL contribution per bin: \( (l_b - r_b)\ln(l_b/r_b) = \text{KL term}_{l\|r} + \text{KL term}_{r\|l} \) collapsed into one expression. Because it sums per-bin contributions, PSI also *localizes* drift — you can read off which bin is driving the score, which KS cannot do. Empty bins blow up the log, so clamp proportions to a small \( \epsilon \) (e.g. \( 10^{-6} \)) or add a pseudo-count.

**Standard thresholds** (from credit-risk practice, widely reused):

- \( \text{PSI} < 0.1 \): no meaningful shift.
- \( 0.1 \le \text{PSI} < 0.2 \): moderate shift — investigate.
- \( \text{PSI} \ge 0.2 \): significant shift — act. (Some shops use 0.25 as the "major" line.)

**Worked micro-example.** Four equal reference bins, so \( r_b = 0.25 \) each. Live proportions drift toward the top bin: \( l = (0.10,\, 0.20,\, 0.30,\, 0.40) \).

| Bin | \( r_b \) | \( l_b \) | \( l_b - r_b \) | \( \ln(l_b/r_b) \) | term |
|---|---|---|---|---|---|
| 1 | 0.25 | 0.10 | \(-0.15\) | \(-0.916\) | 0.1375 |
| 2 | 0.25 | 0.20 | \(-0.05\) | \(-0.223\) | 0.0112 |
| 3 | 0.25 | 0.30 | \(+0.05\) | \(+0.182\) | 0.0091 |
| 4 | 0.25 | 0.40 | \(+0.15\) | \(+0.470\) | 0.0705 |

\[
\text{PSI} = 0.1375 + 0.0112 + 0.0091 + 0.0705 = 0.2282
\]

Above 0.2 — a significant shift. Notice the top bin dominates the score: PSI is most sensitive where a large *relative* change lands, which is exactly why quantile bins (equal mass at reference) behave better than equal-width bins for skewed features like token counts. With equal-width bins on a heavy-tailed feature, the tail bins are nearly empty at reference, so a handful of new samples there produce a huge \( \ln(l_b/r_b) \) and a jumpy, unreliable score.

**Data type:** scalar / categorical features. **Not** for raw high-dimensional embeddings — you would have to bin per-dimension and lose all cross-dimensional structure.

### 2. Kolmogorov–Smirnov (KS) two-sample test

**Intuition.** Forget bins. Compare the two *empirical cumulative distribution functions* directly, and take the single point of maximum vertical gap between them. Big gap ⇒ the distributions differ. KS is non-parametric (assumes nothing about shape) and needs no binning choice, which makes it a clean default for continuous features.

**Formula.** For empirical CDFs \( F_{\text{ref}} \) and \( F_{\text{live}} \):

\[
D = \sup_{x} \bigl| F_{\text{live}}(x) - F_{\text{ref}}(x) \bigr|
\]

\( D \in [0,1] \). The p-value comes from the Kolmogorov distribution; for sample sizes \( n, m \) you reject "same distribution" at level \( \alpha \) when

\[
D > c(\alpha)\,\sqrt{\frac{n+m}{n\,m}}, \qquad c(0.05) \approx 1.36 .
\]

**Worked micro-example.** Reference sample \( \{1,2,3,4\} \), live sample \( \{2,3,4,5\} \) (each shifted up by 1). Step through the pooled sorted values and read both CDFs:

| \( x \) | \( F_{\text{ref}} \) | \( F_{\text{live}} \) | gap |
|---|---|---|---|
| 1 | 0.25 | 0.00 | 0.25 |
| 2 | 0.50 | 0.25 | 0.25 |
| 3 | 0.75 | 0.50 | 0.25 |
| 4 | 1.00 | 0.75 | 0.25 |
| 5 | 1.00 | 1.00 | 0.00 |

\( D = 0.25 \). With \( n=m=4 \) that is nowhere near significant (the critical value is enormous for four points) — a reminder that KS on tiny windows is uninformative, and that the **statistic** and its **significance** are different things. On the realistic 5000-vs-1500 example below, \( D = 0.26 \) with a p-value around \( 10^{-67} \): same statistic magnitude, wildly different verdict, because sample size collapses the noise band.

**Caveats.** KS is most sensitive near the center of the distribution and comparatively blind in the tails. With very large windows it becomes *hypersensitive* — trivial, operationally irrelevant differences produce \( p < 0.001 \). That is why you pair the p-value with an **effect-size** threshold on \( D \) itself (say, alert only if \( D > 0.1 \) *and* \( p < 0.01 \)). For categorical features KS does not apply — use a chi-square test of the count table instead.

### 3. Maximum Mean Discrepancy (MMD)

**Intuition.** The right tool when your feature is a **vector** (an embedding), not a scalar. Map every sample through a kernel into a high-dimensional space, take the mean of each set there, and measure the distance between those means. If the distributions are identical, the mean embeddings coincide and MMD is zero. Unlike PSI/KS it is inherently multivariate — no binning, no per-dimension decomposition — which is why it shows up in embedding-drift toolkits.

**Formula.** With kernel \( k \) (commonly RBF, \( k(a,b)=\exp(-\gamma\lVert a-b\rVert^2) \)), the (biased) empirical squared MMD between reference \( X=\{x_i\}_{i=1}^m \) and live \( Y=\{y_j\}_{j=1}^n \):

\[
\widehat{\text{MMD}}^2 = \frac{1}{m^2}\sum_{i,i'} k(x_i,x_{i'}) + \frac{1}{n^2}\sum_{j,j'} k(y_j,y_{j'}) - \frac{2}{mn}\sum_{i,j} k(x_i,y_j)
\]

Read it as **(within-reference similarity) + (within-live similarity) − 2·(cross similarity)**. When the two clouds overlap, the cross term matches the within terms and everything cancels toward zero. Significance comes from a **permutation test**: shuffle the pooled labels many times, recompute MMD each time to build the null distribution, and see where your observed value falls in that null.

**Worked micro-example.** 1-D, RBF with \( \gamma = 0.5 \). Reference \( X=\{0,1,2\} \), live \( Y=\{3,4,5\} \). Computing the three kernel-matrix means:

- within-reference mean \( = 0.633 \)
- within-live mean \( = 0.633 \) (same spacing, so same self-similarity)
- cross mean \( = 0.101 \) (clouds are far apart, so kernel values are small)

\[
\widehat{\text{MMD}}^2 = 0.633 + 0.633 - 2(0.101) = 1.0635, \qquad \widehat{\text{MMD}} = 1.031
\]

The large value reflects two clearly separated clouds. **Pitfall:** MMD's scale is meaningless in the abstract — a value of 1.03 is only "large" relative to the permutation null for *your* data and *your* \( \gamma \). The kernel bandwidth \( \gamma \) matters a lot; a common heuristic sets it from the **median pairwise distance** of the pooled sample. Always calibrate the threshold empirically; never hard-code an MMD number. Cost is \( O((m+n)^2) \) per window, so subsample for large windows.

### 4. Embedding distance & clustering

**Intuition.** The cheapest embedding-drift signals. Summarize each set of embeddings by its **centroid** (mean vector) and measure how far the centroids moved, either by Euclidean distance or by cosine of the angle. Fast, streaming-friendly, and interpretable — but coarse.

**Formulas.** Centroids \( \bar\phi_{\text{ref}} = \frac1m\sum_i \phi(x_i) \) and \( \bar\phi_{\text{live}} \):

\[
d_{\text{euclid}} = \lVert \bar\phi_{\text{ref}} - \bar\phi_{\text{live}} \rVert_2,
\qquad
d_{\cos} = 1 - \frac{\bar\phi_{\text{ref}} \cdot \bar\phi_{\text{live}}}{\lVert \bar\phi_{\text{ref}}\rVert\,\lVert \bar\phi_{\text{live}}\rVert}
\]

**Worked micro-example.** Two 2-D centroids, normalized: \( \bar\phi_{\text{ref}} = (0.8,0.6) \) and \( \bar\phi_{\text{live}} = (0.6,0.8) \) (both already unit-norm).

\[
\cos = 0.8(0.6) + 0.6(0.8) = 0.96 \Rightarrow d_{\cos} = 0.04, \qquad d_{\text{euclid}} = \lVert(0.2,-0.2)\rVert = 0.2828
\]

**The failure mode you must know:** centroid distance is **blind to variance and multimodal shifts**. If half your traffic moves far left and half moves far right, the centroid can sit *exactly where it started* while the distribution has torn in two. This is why serious embedding-drift setups prefer a **domain classifier** (train a binary model to tell reference from live; if it achieves ROC-AUC meaningfully above 0.5, the sets are distinguishable ⇒ drift, and the classifier's important features tell you *why*) or MMD, both of which see distributional shape, not just the mean. Centroid distance is a good *first* alarm, never the *only* one.

### 5. Wasserstein distance & chi-square (honorable mentions)

Two more you should be able to name:

- **Wasserstein (earth-mover's) distance** on a scalar feature: the minimum "work" to reshape one distribution into the other, \( W_1 = \int |F_{\text{ref}}(x) - F_{\text{live}}(x)|\,dx \). Unlike KS (a single sup gap) it integrates the *whole* difference and is reported in the feature's real units (tokens, milliseconds), which makes thresholds interpretable. Evidently uses it per-dimension in one of its embedding-drift methods.
- **Chi-square test** for categorical features (topic labels, language, refusal/no-refusal): compares observed vs. expected counts, \( \chi^2 = \sum_b (O_b - E_b)^2 / E_b \). This is the categorical analogue of KS — reach for it whenever the feature is a label rather than a number.

---

## A fully worked example: PSI + KS on a live window

A drop-in monitor over one scalar feature — here **prompt length in tokens**. The reference is captured at deploy time; the live window is the last \( N \) requests. The numbers in the comments are the actual output of this code (seed fixed), so you can run it and reproduce them exactly.

```python
import numpy as np
from scipy import stats

# ----- Two windows of a real feature: prompt length (tokens) -----
# Reference: captured at deploy/eval time.
# Live: users are now pasting more context -> longer prompts.
rng = np.random.default_rng(42)
ref  = rng.gamma(shape=4.0, scale=30.0, size=5000)   # mean ~120 tokens
live = rng.gamma(shape=4.0, scale=42.0, size=1500)   # mean ~168 tokens


def psi(ref, live, bins=10, eps=1e-6):
    """PSI with quantile bins fixed by the REFERENCE distribution.

    Quantile (equal-mass) bins are the right default for skewed
    features like token counts: equal-width bins would leave the
    long tail nearly empty and make the score unstable.
    """
    # Bin edges = reference deciles; outer edges pushed to +/- inf
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    r_counts, _ = np.histogram(ref,  bins=edges)
    l_counts, _ = np.histogram(live, bins=edges)

    # Clip to avoid log(0) / divide-by-zero on empty live bins
    r_prop = np.clip(r_counts / r_counts.sum(), eps, None)
    l_prop = np.clip(l_counts / l_counts.sum(), eps, None)

    return float(np.sum((l_prop - r_prop) * np.log(l_prop / r_prop)))


# ----- Compute both signals -----
psi_val = psi(ref, live)
ks = stats.ks_2samp(ref, live)          # returns (statistic D, p-value)

print(f"PSI            = {psi_val:.4f}")        # PSI            = 0.4345
print(f"KS statistic D = {ks.statistic:.4f}")   # KS statistic D = 0.2567
print(f"KS p-value     = {ks.pvalue:.2e}")      # KS p-value     = 2.51e-67


# ----- Turn signals into an alert -----
PSI_ALERT   = 0.20    # significant-shift threshold
KS_D_ALERT  = 0.10    # minimum effect size we care about
KS_P_ALERT  = 0.01    # significance level

psi_fires = psi_val >= PSI_ALERT
ks_fires  = (ks.statistic >= KS_D_ALERT) and (ks.pvalue < KS_P_ALERT)

if psi_fires and ks_fires:
    print("ALERT: prompt-length drift (PSI + KS agree). "
          "Inputs are longer than at deploy time; "
          "check truncation, context limits, and eval coverage.")
elif psi_fires or ks_fires:
    print("WATCH: one signal tripped; monitor next windows before paging.")
else:
    print("OK: no meaningful prompt-length drift.")
```

**Output:**

```
PSI            = 0.4345
KS statistic D = 0.2567
KS p-value     = 2.51e-67
ALERT: prompt-length drift (PSI + KS agree). ...
```

Both signals agree — PSI 0.43 is well past the 0.2 line, and KS gives \( D=0.26 \) with an astronomically small p-value. The **AND** of an effect-size gate and a significance gate is the pattern that keeps this from crying wolf: on a huge window KS alone would flag a 2-token difference; requiring \( D \ge 0.10 \) suppresses that. PSI alone, on a tiny window, would be jumpy; requiring both cross-checks it. Two cheap, independent tests on the same feature is a good default posture.

Note the two design choices that make this production-safe: (1) bin **edges are frozen from the reference** — if you re-derive quantile edges from the live window each time, both histograms are uniform by construction and PSI collapses to zero, hiding the drift; (2) the alert message is *actionable* — it names the likely causes and next checks, not just "drift detected."

---

## Extending to embeddings: MMD + a domain classifier

Scalars are the easy case. The moment you want *semantic* drift — "are users asking about different things?" — the feature is an embedding vector and you need a multivariate method. Here is a compact, correct monitor that runs both MMD (with a median-heuristic bandwidth and a permutation p-value) and a domain classifier, on a synthetic 16-dim embedding stream where four dimensions have shifted.

```python
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
d = 16
ref  = rng.normal(0.0, 1.0, size=(2000, d))
live = rng.normal(0.0, 1.0, size=(600, d))
live[:, :4] += 0.6                      # 4 of 16 dims drift (a new topic cluster)


def median_gamma(Z):
    """RBF bandwidth from the median pairwise distance (standard heuristic)."""
    med = np.median(pdist(Z))
    return 1.0 / (2.0 * med ** 2)


def rbf_mmd2(X, Y, gamma):
    """Biased empirical squared MMD with an RBF kernel."""
    XX = np.exp(-gamma * ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    YY = np.exp(-gamma * ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    XY = np.exp(-gamma * ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    return XX.mean() + YY.mean() - 2.0 * XY.mean()


def mmd_permutation_test(X, Y, n_perm=200, sub=300, seed=0):
    """MMD^2 plus a permutation p-value. Subsample first — MMD is O(n^2)."""
    r = np.random.default_rng(seed)
    X = X[r.choice(len(X), min(sub, len(X)), replace=False)]
    Y = Y[r.choice(len(Y), min(sub, len(Y)), replace=False)]
    Z = np.vstack([X, Y])
    gamma = median_gamma(Z)
    n = len(X)
    obs = rbf_mmd2(X, Y, gamma)
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = r.permutation(len(Z))          # shuffle labels under H0
        null[i] = rbf_mmd2(Z[idx[:n]], Z[idx[n:]], gamma)
    pval = (1 + (null >= obs).sum()) / (1 + n_perm)
    return obs, pval


def domain_classifier_auc(ref, live):
    """Train ref-vs-live; AUC ~0.5 => indistinguishable, ~1.0 => strong drift."""
    X = np.vstack([ref, live])
    y = np.r_[np.zeros(len(ref)), np.ones(len(live))]
    clf = LogisticRegression(max_iter=1000)
    return cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()


mmd2, mmd_p = mmd_permutation_test(ref, live)
auc = domain_classifier_auc(ref, live)

# Representative run:
#   MMD^2 = 0.037,  permutation p = 0.005
#   domain classifier AUC = 0.80
print(f"MMD^2 = {mmd2:.3f}   perm p = {mmd_p:.3f}")
print(f"domain classifier AUC = {auc:.2f}")

AUC_ALERT = 0.65        # AUC this far above 0.5 => sets are clearly separable
if mmd_p < 0.01 and auc >= AUC_ALERT:
    print("ALERT: embedding drift (MMD + classifier agree). "
          "Cluster the live embeddings to find the new topic(s).")
```

Two takeaways. First, the **domain-classifier AUC is the most interpretable embedding-drift number you can report** — 0.80 means a simple model tells reference from live 80% of the time, which is unambiguous drift, and the classifier's coefficients point at *which* dimensions moved. Second, **MMD's raw value (0.037) is meaningless without the permutation p-value** — the same shift under a different bandwidth produces a different MMD magnitude, so always report significance, never the bare statistic. When these two disagree, trust the classifier for "is there drift?" and use MMD as a cheaper continuous tripwire between retrains.

---

## Methods comparison

| Method | Data type | Output | Sensitivity | Pros | Cons |
|---|---|---|---|---|---|
| **PSI** | Scalar / categorical | Unbounded score, standard thresholds (0.1 / 0.2) | Sensitive where relative mass changes; binning-dependent | One interpretable number; no p-value plumbing; localizes to bins; industry-standard cutoffs | Needs binning choice; unstable with sparse bins; not for raw vectors |
| **KS test** | Continuous scalar | \( D\in[0,1] \) + p-value | Strong at distribution center, weak in tails | Non-parametric, no binning, principled p-value | Hypersensitive on huge windows; univariate only; tail-blind; not for categoricals |
| **MMD** | Vectors / embeddings | \( \ge 0 \), null via permutation | Detects general distributional shape shifts | Truly multivariate; kernel-flexible; theoretically grounded | \( O(n^2) \) cost; threshold not intuitive; bandwidth \( \gamma \) tuning matters |
| **Domain classifier** | Vectors / embeddings | ROC-AUC \( \in [0.5,1] \) | Sees any separable shift, incl. multimodal | Interpretable (AUC + feature importance), robust across embedding types, a strong default | Needs training per window; can overfit small windows |
| **Centroid / cosine dist** | Vectors / embeddings | Distance \( \ge 0 \) | Only mean shift | Cheap, streaming, interpretable | Blind to variance & multimodal splits; threshold hand-tuned |
| **Wasserstein (per-dim)** | Scalar / per-embedding-dim | \( \ge 0 \), in feature units | Sensitive to any shift incl. tails | Metric with real units; tail-aware | Univariate per dim; needs aggregation across dims |
| **Chi-square** | Categorical | \( \chi^2 \) + p-value | Any count-table change | Right tool for labels; simple | Needs adequate expected counts per cell |

Rule of thumb: **scalars ⇒ PSI or KS; categoricals ⇒ chi-square or PSI; embeddings ⇒ domain classifier (default) or MMD; want a cheap tripwire ⇒ centroid distance.**

---

## LLM-specific drift signals

Generic distribution tests get you far, but LLM serving has signals you should monitor by name. In every case the *feature* is LLM-specific; the *test* is the same PSI/KS/MMD/chi-square machinery.

- **Topic / intent shift.** Run a lightweight topic or intent classifier (or cluster embeddings) and watch the category mix with PSI/chi-square over the topic histogram. A launch, a season, or an outage upstream shows up here first.
- **Rising refusals.** Track the fraction of outputs that are refusals or safety deflections ("I can't help with that"). A climbing refusal rate on stable-looking inputs often means a system-prompt or provider-side model change, not user behavior. Alert on the *rate*, and segment by topic — a global rise and a single-topic rise mean very different things.
- **Jailbreak / adversarial probing.** A rising share of prompts matching known jailbreak patterns, or a spike in prompt-injection markers, is an attack signal masquerading as input drift. Monitor it as its own series so it doesn't get averaged away in overall input drift.
- **Quality decay.** With no labels, use proxies: an LLM-as-judge score on a sampled slice, self-consistency across samples, retrieval-hit rate for RAG, or user thumbs-down rate. Treat judge scores as a drifting *feature* themselves and run PSI/KS on them window over window.
- **Output length shift.** Sudden shortening often signals truncation, a max-tokens change, or the model bailing early; sudden lengthening can signal rambling or a prompt-template regression. Cheap to compute, high signal.
- **Latency / cost shift.** p50/p95 latency and tokens-per-request are drift signals too. A provider swapping the model behind an alias frequently shows up as a latency and length change *before* anyone notices quality — sometimes it is your earliest signal of a silent model change.
- **Language / encoding shift.** A new language appearing, or a jump in non-ASCII / emoji ratio, changes tokenization and can silently degrade a model tuned for English.
- **Format / schema conformance.** For structured-output endpoints, monitor the rate of JSON-parse failures or schema-validation errors. A creeping failure rate is quality drift you *can* measure without labels.

---

## Windowing & thresholds

The math is easy; the windowing is where judgment lives.

- **Reference window.** Prefer a **fixed, curated** reference (your eval-time distribution or a known-good production period), not a rolling one — a rolling reference lets slow drift redefine "normal" and hides exactly the gradual decay you care about. Refresh it deliberately, versioned, when you re-eval or retrain, and keep the old one around so you can diff.
- **Live window.** Big enough to be stable, small enough to be timely. Two common shapes: a **tumbling** window (disjoint hourly/daily batches — clean, but bursty) and a **sliding** window (smoother, but overlapping samples correlate, so consecutive readings are not independent). Size it to your traffic: a KS test wants hundreds-to-thousands of points to be meaningful; below ~50 the statistic is mostly noise.
- **Thresholds.** Start from priors (PSI 0.2, KS \( D>0.1 \) with \( p<0.01 \)), then **calibrate against your own history**: replay past known-good weeks, look at the natural spread of each metric, and set thresholds a few standard deviations above that baseline. For embedding methods with no intrinsic scale (MMD, centroid distance), thresholds *must* be empirical (permutation null or historical quantiles) — a literature number is meaningless for your data.
- **Require agreement / persistence.** Two independent signals agreeing, or one signal tripping for \( k \) consecutive windows, cuts false alarms dramatically versus a single-window single-metric trigger. This is the cheapest reliability lever you have.
- **Segment before you aggregate.** A global metric averages away a severe shift confined to one client, region, or topic. Compute drift per meaningful segment; alert on the worst segment, not the mean.

---

## Response playbook: what to do on drift

An alert is a question, not a verdict. Work the ladder:

1. **Confirm it's real.** Is it persistent across windows or a single-window blip? Do independent signals agree? Rule out a data-pipeline bug (a logging change, a tokenizer upgrade, a new client version that reformats prompts looks *exactly* like drift) before anything else. This is the single most common "drift" root cause.
2. **Localize it.** Which feature, which segment, which topic, which customer, which region? Slice by metadata. "Overall PSI up" is useless; "prompt-length drift confined to the mobile client after the 4.2 release" is actionable.
3. **Classify it.** Input drift, output drift, or (suspected) concept drift? Benign (new-but-handled use case) or harmful (quality decay)? Input drift with stable outputs may need nothing but a note and an eval-coverage ticket.
4. **Re-evaluate.** Run your eval suite against a fresh sample of *current* traffic, not last quarter's fixtures. This is the only way to convert "the distribution moved" into "quality actually dropped." If you have any labels or can label a slice, do it now — even 100 hand-labeled current examples beats zero.
5. **Mitigate, cheapest first:**
   - **Prompt / template fix** if a system-prompt or template regression is the cause (fastest, most common).
   - **Roll back** the model, provider alias, or config change if the drift lines up with a deploy.
   - **Guardrail / route** — add a filter or route the drifted segment to a fallback or a stronger model.
   - **Expand evals** to cover the new distribution so it stops being a blind spot next time.
   - **Retrain / fine-tune / update RAG index** — the heaviest, slowest lever; reserve it for genuine, persistent concept drift, not a one-week anomaly.
6. **Close the loop.** Update the reference window (versioned) once you have accepted the new normal, and write down what the alert meant so the next on-call doesn't re-derive it at 2am. A drift runbook with past incidents is worth more than any single dashboard.

---

## Failure modes & pitfalls

- **Seasonality masquerading as drift.** Traffic looks different at 3am, on weekends, on the 1st of the month. A reference captured Tuesday-midday will "drift" every Saturday. Compare like-for-like windows (same weekday/hour), or model the seasonality out; otherwise you train the team to ignore alerts.
- **Wrong / stale reference window.** Rolling references silently absorb slow decay. Too-short references are noisy. A reference from a broken period bakes the breakage into "normal" and you will never see the fault.
- **Big-window hypersensitivity.** On millions of requests, KS and chi-square reject *everything* — every trivial difference is "significant." Pair p-values with effect-size gates, always.
- **High-dimensional embedding drift is subtle.** Centroid distance can read zero while the distribution splits in two; per-dimension tests miss cross-dimensional structure; and in high dimensions everything is far from everything (distance concentration), so raw Euclidean thresholds are treacherous. Prefer domain-classifier or MMD, and reduce dimensionality thoughtfully — drift can hide in the components you discarded, so don't PCA blindly.
- **No ground-truth labels.** You can prove the inputs moved; you cannot prove quality dropped without evaluation. Unsupervised drift is a *smoke alarm*, not a *diagnosis* — never auto-remediate off it alone.
- **Alerting on everything.** One metric per feature per window with a tight threshold = a muted channel within a week. Aggregate, require persistence/agreement, and route by severity.
- **Multiple comparisons.** Running KS on 200 features guarantees ~10 "significant" hits at \( \alpha=0.05 \) by pure chance. Correct for it (Bonferroni / Benjamini–Hochberg FDR) or you will chase ghosts daily.
- **Confusing statistic with significance.** A tiny p-value on a \( D=0.02 \) shift is real but irrelevant; a large \( D \) on 8 samples is irrelevant noise. Report and gate on both the effect size and the p-value.
- **Reference/live binning mismatch.** Re-deriving quantile bin edges from each window makes every histogram uniform and PSI identically zero. Freeze edges from the reference — a subtle bug that silently disables the whole monitor.

---

## Production checklist / what an interviewer probes

1. **"What exactly do you monitor, and why those features?"** — Expect a named list: prompt length/token count, topic mix, refusal rate, output length, latency, format-conformance, and at least one embedding-based signal. Bonus for explaining input vs. output vs. concept coverage.
2. **"PSI vs. KS vs. MMD — when each?"** — Scalars → PSI/KS; categoricals → chi-square; embeddings → MMD or domain classifier. Know PSI's 0.1/0.2 thresholds and that KS is a supremum-of-CDF-gap statistic.
3. **"How do you pick the reference window?"** — Fixed, curated, versioned; not rolling; refreshed deliberately. Red flag if they roll it automatically.
4. **"How do you avoid false alarms?"** — Effect-size + significance gates, persistence across windows, multi-signal agreement, seasonality handling, per-segment analysis, multiple-comparison correction.
5. **"You have no labels — how do you know quality actually dropped?"** — Must acknowledge unsupervised drift ≠ quality drop; re-eval on current traffic, LLM-judge on a slice, thumbs-down rate, canary/labeled sample.
6. **"Drift fires — walk me through the response."** — Confirm (rule out pipeline bug) → localize → classify → re-eval → mitigate cheapest-first (prompt fix / rollback before retrain) → update reference.
7. **"How would you detect embedding drift, and what breaks the naive approach?"** — Domain classifier / MMD; centroid distance is blind to variance and multimodal splits; distance concentration in high dimensions.
8. **"Concept drift with stable inputs — how do you catch it?"** — Honest answer: unsupervised input monitoring won't; you need labels, periodic re-eval, or downstream outcome tracking.

---

## Where this sits in the serving stack

A drift monitor is a small, boring pipeline that runs *beside* the hot path, never in it:

1. **Log at the edge.** For every request/response, emit cheap features — token counts, latency, refusal flag, topic label, format-valid flag — plus a sampled subset of embeddings. Sample embeddings (they are expensive to store); log scalars in full.
2. **Snapshot a reference.** At each deploy/eval, freeze a reference profile: bin edges per scalar feature, a held-out embedding sample, and the baseline rates. Version it alongside the model.
3. **Window & score offline.** On a schedule (hourly/daily), pull the latest window, run PSI/KS/chi-square on scalars and MMD/classifier on embeddings, per segment.
4. **Gate & alert.** Apply effect-size + significance gates, require persistence or multi-signal agreement, then route by severity to a dashboard (low) or a page (high).
5. **Feed the runbook.** Every alert links to the playbook above and appends to an incident log so patterns become institutional knowledge.

The key architectural property: the monitor is **asynchronous and best-effort**. It must never add latency to inference, and a monitor outage must never take down serving. Compute drift from logs, not inline.

---

## Further reading

- Fiddler AI — *Measuring Data Drift with the Population Stability Index (PSI)*: https://www.fiddler.ai/blog/measuring-data-drift-population-stability-index
- GeeksforGeeks — *Population Stability Index (PSI)*: https://www.geeksforgeeks.org/data-science/population-stability-index-psi/
- Gretton et al. — *A Kernel Two-Sample Test* (JMLR 2012, the MMD paper): https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
- TorchDrift — *Intuition for the Maximum Mean Discrepancy two-sample test*: https://torchdrift.org/notebooks/note_on_mmd.html
- Evidently AI — *5 methods to detect drift in ML embeddings*: https://www.evidentlyai.com/blog/embedding-drift-detection
- Evidently AI — *Data drift algorithm* (how thresholds/tests are chosen): https://docs-old.evidentlyai.com/reference/data-drift-algorithm
- Evidently AI — *Monitoring embeddings drift* (open-source course module): https://learn.evidentlyai.com/ml-observability-course/module-3-ml-monitoring-for-unstructured-data/monitoring-embeddings-drift
- NannyML — *Multivariate Drift Detection (PCA reconstruction error)*: https://nannyml.readthedocs.io/en/stable/how_it_works/multivariate_drift.html
- NannyML — *Monitoring data drift: univariate & multivariate methods*: https://www.nannyml.com/blog/monitoring-data-drift
- SciPy — `ks_2samp` reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
- Real Statistics — *Two-sample Kolmogorov–Smirnov test*: https://real-statistics.com/non-parametric-tests/goodness-of-fit-tests/two-sample-kolmogorov-smirnov-test/
- WhyLabs — *whylogs* (data logging & drift profiling, open source): https://github.com/whylabs/whylogs
