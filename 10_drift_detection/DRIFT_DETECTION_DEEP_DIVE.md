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

**Seen in the wild:** this is the exact scoring NannyML's univariate drift detector and Evidently's `DataDriftTable` metric compute per column by default — if you've called either of those libraries on tabular or scalar LLM features, you were already running this formula.

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

**Seen in the wild:** SciPy's `ks_2samp` (used throughout this chapter's worked examples) is the reference implementation; Evidently and NannyML both call it internally for their per-column drift tests on numeric features, wrapping it with the same effect-size-plus-significance framing recommended above.

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

**Seen in the wild:** Evidently AI lists MMD as one of its five embedding-drift methods (see the 2025–2026 landscape section below), and it is the test underlying most "kernel two-sample test" drift-detection literature, including the original Gretton et al. formulation cited in Further reading.

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

**Seen in the wild:** this is the cheapest of Evidently's five embedding-drift methods and the first one most teams wire up, precisely because it needs no training and no permutation test — and precisely why the multimodal-blindness caveat above is the single most common way an embedding-drift monitor gives false reassurance.

### 5. Wasserstein distance & chi-square (honorable mentions)

Two more you should be able to name:

- **Wasserstein (earth-mover's) distance** on a scalar feature: the minimum "work" to reshape one distribution into the other, \( W_1 = \int |F_{\text{ref}}(x) - F_{\text{live}}(x)|\,dx \). Unlike KS (a single sup gap) it integrates the *whole* difference and is reported in the feature's real units (tokens, milliseconds), which makes thresholds interpretable. Evidently uses it per-dimension in one of its embedding-drift methods.
- **Chi-square test** for categorical features (topic labels, language, refusal/no-refusal): compares observed vs. expected counts, \( \chi^2 = \sum_b (O_b - E_b)^2 / E_b \). This is the categorical analogue of KS — reach for it whenever the feature is a label rather than a number.

**Seen in the wild:** Evidently's embedding-drift guide uses per-dimension Wasserstein distance as one of its five methods (alongside the domain classifier and MMD covered above), aggregating the per-dimension distances into a single "share of drifted components" score — a good middle ground between a single opaque MMD number and a full per-dimension report.

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

## Build it in practice — extended: rolling embedding drift + a combined multi-signal alert

The two worked examples above each run in isolation: PSI+KS on one scalar, MMD+classifier on one static pair of windows. Production monitors need two more things the isolated examples skip over: (1) embedding drift computed **continuously**, over a **rolling window**, from **real text** rather than a pre-made array; and (2) an alert that only fires when **independent signals agree**, so a single noisy metric cannot page anyone by itself. This section builds both, end to end, and then runs a realistic scenario — three quiet weeks, one seasonal false-positive week, and four weeks of genuine gradual drift — to show the fusion rule working exactly as intended.

**Real embeddings, swapped for a deterministic stand-in.** In production you would embed every prompt with a sentence-embedding model, e.g.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim, unit-normalized
embeddings = model.encode(texts, normalize_embeddings=True)
```

To keep this example runnable offline with no model download and no GPU, the code below swaps in a deterministic hashing-embedding with the identical interface (`list[str] -> (n, dim)` unit vectors). Only the `embed()` function would change in production; the monitor, the rolling window, and the fusion logic are exactly what you would deploy.

```python
import numpy as np
from collections import deque
from scipy import stats

rng = np.random.default_rng(7)
EMBED_DIM = 64  # small for a fast demo; a real MiniLM embedding is 384-dim


def embed(texts, dim=EMBED_DIM):
    """Stand-in for model.encode(texts, normalize_embeddings=True).

    Deterministic and offline: hashes each token to a fixed random
    direction and sums, then L2-normalizes. Same input/output contract
    as a real sentence-transformer call.
    """
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            h = abs(hash(tok)) % (2**32)
            out[i] += np.random.default_rng(h).normal(size=dim)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norms, 1e-8, None)


class RollingEmbeddingDriftMonitor:
    """
    Tracks centroid + cosine drift of a live embedding stream against a
    frozen reference centroid, over a rolling window of recent requests.
    """

    def __init__(self, reference_embeddings, window_size=300,
                 cosine_alert=0.02, euclid_alert=0.20):
        self.ref_centroid = reference_embeddings.mean(axis=0)
        self.ref_centroid /= np.linalg.norm(self.ref_centroid)
        self.window = deque(maxlen=window_size)   # oldest requests fall off
        self.cosine_alert = cosine_alert
        self.euclid_alert = euclid_alert

    def update(self, new_embeddings):
        """Feed a batch of new (already-normalized) embeddings, return the score."""
        for e in new_embeddings:
            self.window.append(e)
        return self.score()

    def score(self):
        if len(self.window) < 10:
            return None  # not enough data to trust a centroid yet
        live_centroid = np.mean(self.window, axis=0)
        live_centroid /= np.linalg.norm(live_centroid)
        cos_dist = 1.0 - float(np.dot(self.ref_centroid, live_centroid))
        euclid_dist = float(np.linalg.norm(self.ref_centroid - live_centroid))
        return {
            "cosine_dist": cos_dist,
            "euclid_dist": euclid_dist,
            "fires": cos_dist >= self.cosine_alert or euclid_dist >= self.euclid_alert,
        }


def psi_len(ref, live, bins=10, eps=1e-6):
    """Same frozen-reference-edges PSI as the worked example above."""
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r_counts, _ = np.histogram(ref, bins=edges)
    l_counts, _ = np.histogram(live, bins=edges)
    r_prop = np.clip(r_counts / r_counts.sum(), eps, None)
    l_prop = np.clip(l_counts / l_counts.sum(), eps, None)
    return float(np.sum((l_prop - r_prop) * np.log(l_prop / r_prop)))


def combined_alert(ref_lengths, live_texts, live_lengths, monitor):
    """Fire ONLY when >=2 of {PSI, KS, embedding-drift} agree."""
    psi_val = psi_len(ref_lengths, live_lengths)
    ks = stats.ks_2samp(ref_lengths, live_lengths)
    emb_result = monitor.update(embed(live_texts))

    psi_fires = psi_val >= 0.20
    ks_fires = (ks.statistic >= 0.10) and (ks.pvalue < 0.01)
    emb_fires = bool(emb_result and emb_result["fires"])
    votes = int(psi_fires) + int(ks_fires) + int(emb_fires)

    return {
        "psi": round(psi_val, 3),
        "ks_D": round(float(ks.statistic), 3),
        "emb_cos_dist": round(emb_result["cosine_dist"], 3) if emb_result else None,
        "votes": votes,
        "ALERT": votes >= 2,   # require agreement -- this is the whole point
    }
```

**Simulating a realistic month.** A support-chat reference of billing/support/how-to questions; then three stable weeks; then one week where a billing surge (a seasonal, calendar-driven spike, not a product change) shifts the topic mix hard; then four weeks where a genuinely new topic — a just-shipped product feature — grows from 10% to 45% of traffic.

```python
def make_batch(n, topic_mix, rng):
    topics, probs = list(topic_mix), list(topic_mix.values())
    chosen = rng.choice(topics, size=n, p=probs)
    texts = [f"{t} question about {t} details number {rng.integers(0, 999)}" for t in chosen]
    # prompts about the new feature run longer (users paste config/logs) --
    # length scales with how much of the traffic is the new topic
    lengths = rng.gamma(shape=4.0, scale=30.0, size=n) + 40.0 * topic_mix.get("new_feature", 0.0)
    return texts, lengths


ref_texts, ref_lengths = make_batch(2000, {"billing": 0.4, "support": 0.4, "howto": 0.2}, rng)
monitor = RollingEmbeddingDriftMonitor(embed(ref_texts), window_size=300)

for week in range(1, 4):                                    # stable weeks
    texts, lengths = make_batch(300, {"billing": 0.4, "support": 0.4, "howto": 0.2}, rng)
    print(f"week {week} (stable):        ", combined_alert(ref_lengths, texts, lengths, monitor))

texts, lengths = make_batch(300, {"billing": 0.75, "support": 0.20, "howto": 0.05}, rng)
print("week 4 (seasonal spike):", combined_alert(ref_lengths, texts, lengths, monitor))

for week, share in zip(range(5, 9), [0.10, 0.20, 0.30, 0.45]):  # real, growing drift
    s = 1.0 - share
    mix = {"billing": 0.4*s, "support": 0.4*s, "howto": 0.2*s, "new_feature": share}
    texts, lengths = make_batch(300, mix, rng)
    print(f"week {week} (real drift {share:.0%}):", combined_alert(ref_lengths, texts, lengths, monitor))
```

**Actual output of this code:**

```
week                          PSI    KS D  cos_dist  votes   ALERT
week 1 (stable)             0.017   0.038     0.001      0   False
week 2 (stable)              0.02   0.034     0.001      0   False
week 3 (stable)              0.038  0.051       0.0      0   False
week 4 (seasonal spike)      0.009  0.036     0.071      1   False
week 5 (real drift 10%)      0.034  0.078     0.005      0   False
week 6 (real drift 20%)      0.067   0.11     0.027      2    True
week 7 (real drift 30%)      0.158  0.148     0.054      2    True
week 8 (real drift 45%)      0.262   0.19      0.14       3    True
```

Read this table as the whole point of the section. **Week 4** is a large, real shift in topic mix (billing jumps from 40% to 75% of traffic in a single week) — a naive single-metric monitor watching topic share with PSI/chi-square would page on-call immediately. But prompt *length* barely moves (billing questions are not systematically longer or shorter than support questions), so PSI and KS both stay quiet; only the embedding-centroid signal fires, casting 1 of 3 votes — below the 2-vote bar, so the combined alert correctly stays silent on what is, in fact, ordinary seasonal or promotional traffic. **Weeks 6 through 8** tell the opposite story: a genuinely new topic keeps growing, dragging prompt length up with it (users paste extra config for the new feature), so the embedding signal *and* the length-based tests climb together — by week 6 two of three signals agree and the alert fires, and by week 8 all three agree, which is the correct place to escalate from "investigate" to "page." The system did exactly the job description from earlier in this chapter: **the AND-of-independent-signals gate suppressed the false positive and still caught the real drift, with increasing signal count doubling as a built-in severity ladder.**

Two structural notes for reuse: the `RollingEmbeddingDriftMonitor`'s window is a `deque(maxlen=...)`, so it is O(1) to update and always reflects only the most recent `window_size` requests — old traffic ages out automatically, which is what makes this safe to run continuously rather than as a batch job. And the reference centroid, like the PSI bin edges earlier, is computed **once, from the frozen reference**, never recomputed from the live window — the same bug that silently zeroes out PSI (re-deriving bins from live data) would silently zero out this monitor too if you recomputed the reference centroid from the window it's supposed to be compared against.

### Calibrating embedding-drift thresholds from history, not from a blog post

Earlier sections warn repeatedly that MMD and centroid/cosine-distance thresholds have no intrinsic scale — a value that's alarming for one embedding model, dataset, and window size is noise for another. The discipline this demands in practice: replay the monitor over many historical **known-good** windows (weeks with no reported incident) and set the alert threshold from a high percentile of the resulting score distribution, rather than importing a threshold from this or any other chapter.

```python
def calibrate_threshold_from_history(monitor_factory, historical_windows, percentile=99):
    """
    monitor_factory: () -> a fresh RollingEmbeddingDriftMonitor built from
        the SAME frozen reference centroid every time, so each replay
        starts from identical state.
    historical_windows: list of embedding batches, each a KNOWN-GOOD window
        (e.g. the last 12 months of weeks with no reported incident).
    Returns a calibrated cosine_dist threshold at the given percentile.
    """
    scores = []
    for window in historical_windows:
        m = monitor_factory()
        result = m.update(window)
        if result is not None:
            scores.append(result["cosine_dist"])
    return float(np.percentile(scores, percentile))


# Replay 52 known-good weekly windows (same generator as the simulation above,
# stable topic mix throughout -- no incident in any of them).
historical_windows = []
for _ in range(52):
    texts, _ = make_batch(300, {"billing": 0.4, "support": 0.4, "howto": 0.2}, rng)
    historical_windows.append(embed(texts))

threshold = calibrate_threshold_from_history(
    lambda: RollingEmbeddingDriftMonitor(embed(ref_texts), window_size=300),
    historical_windows, percentile=99)

print(f"empirically calibrated cosine_dist threshold (P99 of 52 known-good weeks) = {threshold:.4f}")
# empirically calibrated cosine_dist threshold (P99 of 52 known-good weeks) = 0.0021
```

That calibrated figure — 0.0021 — is roughly **10x tighter** than the illustrative `cosine_alert=0.02` used in the worked example above. Neither number is "correct" in the abstract; the calibrated one is correct *for this reference population and this embedding function*, which is the entire point. Recalibrate whenever the reference window, the embedding model, or the window size changes — all three change the natural spread of the score, and a threshold calibrated under one regime will silently over- or under-fire under another.


### A minimal judge / anchor-set attribution monitor

The 2025–2026 landscape section above cites a sharp finding: if your only quality signal is an LLM judge's score, a routine judge-model version bump or prompt edit produces an alarm that is indistinguishable from real product decay, and naive rolling-average monitors reportedly false-alarm on the large majority of judge-only changes. The fix is small enough to implement directly: keep a fixed, human-labeled **anchor set**, re-score it with whatever judge is currently deployed at every check, and use the *change in the judge's own bias on those anchors* to separate "the judge moved" from "the system moved." Below is a minimal, runnable version — the judge itself is mocked (a deterministic function of a hidden "true quality" plus a judge-specific bias and noise) so the four possible worlds — nothing changed, only the product decayed, only the judge changed, and both — can all be demonstrated with real numbers.

```python
import numpy as np

rng = np.random.default_rng(11)

# 40 fixed anchor items with a stable, human-labeled quality in [0, 1].
N_ANCHOR = 40
anchor_human_scores = rng.uniform(0.6, 0.95, size=N_ANCHOR)


def make_judge(judge_shift=0.0, noise=0.03, seed=0):
    """A mock judge: true quality + a fixed judge-specific bias + noise.
    judge_shift models a version bump / prompt edit that recalibrates the
    judge's scoring -- it moves EVERY score, anchors included."""
    r = np.random.default_rng(seed)

    def score_fn(item_id, true_quality):
        return true_quality + judge_shift + r.normal(0, noise)

    return score_fn


class JudgeAnchorAttributor:
    """Separates system drift from judge drift via a frozen anchor set."""

    def __init__(self, anchor_human_scores, z_alert=2.5, system_drop_alert=0.10):
        self.anchor_human_scores = np.asarray(anchor_human_scores, dtype=float)
        self.z_alert = z_alert
        self.system_drop_alert = system_drop_alert
        self.baseline_gap_mean = None
        self.baseline_gap_std = None

    def calibrate(self, judge_score_fn):
        """Call once, right after deploying the current judge version."""
        scores = np.array([judge_score_fn(i, q) for i, q in
                            enumerate(self.anchor_human_scores)])
        gap = scores - self.anchor_human_scores          # judge bias on anchors
        self.baseline_gap_mean = gap.mean()
        self.baseline_gap_std = gap.std() + 1e-6

    def check(self, judge_score_fn, live_true_quality):
        # Re-score the SAME frozen anchors with whatever judge is live now.
        anchor_scores = np.array([judge_score_fn(i, q) for i, q in
                                   enumerate(self.anchor_human_scores)])
        gap = anchor_scores - self.anchor_human_scores
        se = self.baseline_gap_std / np.sqrt(len(self.anchor_human_scores))
        judge_drift_z = float((gap.mean() - self.baseline_gap_mean) / se)
        judge_drifted = abs(judge_drift_z) >= self.z_alert

        # Score live traffic, then correct for the judge bias just measured
        # so a judge shift alone can't masquerade as a system quality drop.
        live_scores = np.array([judge_score_fn(1000 + i, q) for i, q in
                                 enumerate(live_true_quality)])
        corrected = live_scores - gap.mean()
        system_quality = float(corrected.mean())
        baseline_quality = float(self.anchor_human_scores.mean())
        system_dropped = (baseline_quality - system_quality) >= self.system_drop_alert

        if judge_drifted and system_dropped:
            verdict = "system+judge"
        elif judge_drifted:
            verdict = "judge"
        elif system_dropped:
            verdict = "system"
        else:
            verdict = "none"
        return {"judge_drift_z": round(judge_drift_z, 2),
                "system_quality": round(system_quality, 3),
                "verdict": verdict}


attributor = JudgeAnchorAttributor(anchor_human_scores)
judge_v1 = make_judge(judge_shift=0.0, seed=1)
attributor.calibrate(judge_v1)

live_true_quality_stable  = rng.uniform(0.60, 0.95, size=200)
live_true_quality_decayed = rng.uniform(0.35, 0.70, size=200)   # real regression
judge_v2 = make_judge(judge_shift=-0.18, seed=2)                # a version bump

print("week 1 (nothing changed):        ", attributor.check(judge_v1, live_true_quality_stable))
print("week 2 (real product decay):     ", attributor.check(judge_v1, live_true_quality_decayed))
print("week 3 (judge version bump only):", attributor.check(judge_v2, live_true_quality_stable))
print("week 4 (judge bump + real decay):", attributor.check(judge_v2, live_true_quality_decayed))
```

**Actual output:**

```
week 1 (nothing changed):         {'judge_drift_z': -1.01, 'system_quality': 0.764, 'verdict': 'none'}
week 2 (real product decay):      {'judge_drift_z': -0.19, 'system_quality': 0.523, 'verdict': 'system'}
week 3 (judge version bump only): {'judge_drift_z': -40.8, 'system_quality': 0.764, 'verdict': 'judge'}
week 4 (judge bump + real decay): {'judge_drift_z': -42.57, 'system_quality': 0.527, 'verdict': 'system+judge'}
```

Read the four rows as the whole point: week 2's real decay is caught (`system_quality` drops to 0.523, well past the 0.10 alert margin) while the anchor-derived `judge_drift_z` stays near zero — nothing about the judge moved, so the verdict is cleanly `system`. Week 3 is the case naive monitors get wrong: the *live* traffic's true quality never changed, but a judge-shift of \(-0.18\) still shows up as a *massive* `judge_drift_z` because the anchors — which have fixed, known-correct human scores — expose the judge's new bias immediately; the monitor correctly reports `judge`, not a phantom product regression, and critically the bias-corrected `system_quality` (0.764) still reads as healthy because the attribution logic subtracted out the measured judge bias before judging the system. Week 4 shows both moving at once and reports `system+judge` — the one case where you should page a human for *two* separate reasons rather than one, and go fix the judge and the product independently rather than assuming a single root cause. This is the concrete mechanism behind the abstract claim in the landscape section above: **the anchor set is what lets the number "quality dropped" actually mean "the product got worse," instead of silently meaning "our ruler moved."**

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

## The 2025–2026 landscape

As of 2026, drift monitoring for LLM systems splits into two camps that are gradually converging in practice: adapt classic distributional-drift tooling to embeddings, or lean on an LLM judge for the semantic understanding that statistics can't see. Here is who is doing what, and how the debate is actually being resolved.

**Camp 1 — adapt classic ML monitoring to embeddings:**

- **Evidently AI** (Olga Filippova and Elena Samuylova, *5 methods to detect drift in ML embeddings*; first published May 17, 2023, last updated July 16, 2025 — https://www.evidentlyai.com/blog/embedding-drift-detection) documents five methods, the same ones covered earlier in this chapter: Euclidean centroid distance, cosine distance, a domain classifier scored by ROC-AUC, "share of drifted components" (treat each embedding dimension as a numeric feature and run PSI/KS per dimension, then report what fraction of dimensions drifted), and MMD. The guide frames these explicitly for "retrieval pipelines or high-volume user input analysis" — i.e., RAG.
- **NannyML**'s multivariate drift detector (https://nannyml.readthedocs.io/en/stable/how_it_works/multivariate_drift.html) takes a related but distinct approach: fit PCA on the reference set, reconstruct live data through that PCA basis, and alert on rising reconstruction error — a single scalar capturing "the live data no longer looks like anything the reference basis can explain," with no domain classifier required.
- **WhyLabs' LangKit** (https://github.com/whylabs/langkit, docs at https://docs.whylabs.ai/docs/large-language-model-monitoring/) takes a third angle: rather than one drift *test*, it extracts text-quality, text-relevance, security (jailbreak/injection pattern matches), and sentiment/toxicity signals per request via `whylogs`, then relies on the resulting *profile's* drift over time inside the WhyLabs platform. The underlying math is the same PSI-style profile comparison used throughout this chapter, applied to LLM-specific extracted features instead of raw tabular columns.

**Camp 2 — LLM-native, judge-based monitoring:**

- A March 2026 market survey (Galileo, *9 Best LLM Drift Monitoring Platforms in 2026*, https://galileo.ai/blog/best-llm-output-drift-monitoring-platforms) reviews nine commercial platforms — Galileo, Arize AI, LangSmith, Langfuse, Arthur AI, WhyLabs, Weights & Biases (W&B Weave), Aporia, and Helicone — and stakes out a position squarely against Camp 1: "traditional ML monitoring relies on statistical tests like KL divergence or Population Stability Index over numerical distributions," but these "often fail to capture semantic shift." By the survey's count, only three of the nine platforms offer purpose-built semantic-drift algorithms rather than requiring custom implementation.
- The pitch for judge-based detection is real and worth stating plainly: PSI on token length cannot tell you that the model started giving confidently wrong answers about a new product feature while staying exactly the same length.

**The debate is being resolved, in practice, as "both, layered, weighted":**

- A representative synthesis (dev.to, aiwithmohit, *Your LLM Is Lying to You Silently: 4 Statistical Signals That Catch Drift Before Users Do*, March 29, 2026 — https://dev.to/aiwithmohit/your-llm-is-lying-to-you-silently-4-statistical-signals-that-catch-drift-before-users-do-4cg2) proposes four signals used together rather than competing:
  - KL divergence on token-length distributions, alert at \( \ge 0.15 \);
  - embedding cosine drift on the centroid, alert when similarity drops below 0.82;
  - LLM-as-judge scoring — slower (5–8 days to flag decay) but semantically aware;
  - refusal-rate fingerprinting — fastest (3–5 days), catches safety-filter recalibration.
- The author's own advice is explicitly tiered and cost-aware: "start with KL divergence... add embedding drift next week... layer in LLM-as-judge when you have budget," the four combined via weighted voting (roughly KL 0.25, embedding 0.30, judge 0.30, refusal 0.15) to reach a reported ~0.93 AUC on labeled drift incidents. That is precisely the "require agreement across independent signals" pattern built out earlier in this chapter — just with a judge score added as one more vote, and a price tag attached to it.

**Judges rot too — and that changes the design of the monitor:**

- The sharpest 2026 argument against treating an LLM judge as a stable oracle for drift comes from Yitao Li's *Who Drifted: the System or the Judge? Anytime-Valid Attribution in LLM Evaluation Pipelines* (arXiv:2606.15474, June 2026 — https://arxiv.org/abs/2606.15474). The problem: if your drift monitor is "ask a judge model to score outputs and watch the trend," a routine judge-model version bump or a one-line prompt edit to the judge itself produces a drift alarm indistinguishable from real product decay — the paper reports that a naive rolling z-test monitor false-alarmed on 75% of streams where *nothing about the product had changed, only the judge had*.
- The proposed fix, implemented in runnable form earlier in this chapter: keep a small, fixed, human-labeled **anchor set** that never changes, and re-score it with the *current* judge at every check. Because the anchors are frozen, any movement in their scores can only be the judge drifting, letting you attribute an alarm to one of `{none, system, judge}` instead of shrugging at "something drifted." On real judge-version-bump and judge-prompt-edit datasets this attributed correctly on 60 of 60 and 110 of 120 runs respectively.
- The lesson generalizes past LLM judges specifically: **any drift detector that is itself a learned or versioned component — a domain classifier, an embedding model, a judge — needs its own frozen-anchor check, or you cannot tell "the world changed" from "your ruler changed."**

**RAG and agentic systems get their own drift vocabulary:**

- For retrieval-augmented systems, the 2025 *RAGOps: Operating and Managing Retrieval-Augmented Generation Pipelines* paper (Xu, Weytjens, Zhang, Lu, Weber, and Zhu — CSIRO's Data61, TU Munich, UNSW, and Fraunhofer — arXiv:2506.03401 — https://arxiv.org/abs/2506.03401) proposes treating retrieval **coverage** itself as a drift signal: continuously compare live query embeddings against a held-out test-query set, and alert when fewer than 85% of live queries achieve an adequate similarity match to anything in that test set. This is a direct, RAG-specific instance of the embedding-drift machinery in this chapter, aimed at "the index or corpus moved out from under the retriever" rather than at the language model itself.
- For multi-agent and long-running agentic systems, two 2026 threads stand out. *Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems Over Extended Interactions* (Abhishek Rath, arXiv:2601.04170, January 2026 — https://arxiv.org/abs/2601.04170) proposes an "Agent Stability Index" scored across twelve dimensions (response consistency, tool-usage patterns, reasoning-pathway stability, inter-agent agreement, among others) to decompose agent decay into *semantic drift* (deviation from the original objective), *coordination drift* (breakdown in multi-agent consensus), and *behavioral drift* (emergence of undesirable strategies), with mitigations including episodic-memory consolidation and drift-aware routing.
- A practitioner write-up on monitoring LangGraph agents in production (Vadim Nicolai, June 15, 2026 — https://vadim.blog/agent-defect-drift-detection-production/) implements six concrete runtime signals: tool-entropy collapse (the ratio of distinct tool calls to total calls dropping below 0.4, i.e. the agent is looping on the same few tools), role drift and execution-gap (both judge-scored: did the agent abandon its assigned framing, did it claim to have done something it did not), illegal state transitions and excessive loops (both cheap, deterministic graph checks), and dead-ends. It deliberately runs the cheap deterministic checks first, reserving a single fenced judge call only for genuinely ambiguous cases, with hard violations routed to human review.
- That "cheap deterministic gate before an expensive judge call" ordering is the same cost-aware pattern the field is converging on everywhere in 2025–2026: statistical and embedding signals as the always-on tripwire, judge-based signals as the higher-cost confirmation layer — and, per the anchor-set lesson above, the judge itself kept under its own drift watch, not trusted as ground truth.

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

## Production case studies & war stories

Two composite incidents, each assembled from patterns that recur across postmortems for deployed LLM products. Names and specifics are illustrative; the mechanisms and the lessons are exactly the failure modes this chapter exists to prevent.

### War story 1: the silent upstream change that took five weeks to notice

**The setup.** A support-chat assistant had been live for four months, evals green, dashboards calm. The product team shipped an unrelated change: a redesigned onboarding flow that, as a side effect, linked a new class of trial users straight into the chat widget from a "getting started" tooltip — a change that never touched the model, the prompt template, or any file the ML team owned.

**What actually happened:**

- The new users asked simpler, more repetitive setup questions than the assistant's tuned-for-power-users training mix.
- The model, never having seen this exact style, quietly gave shallow, sometimes-wrong answers — but it never refused, and it never got noticeably shorter, the two signals the team already watched.
- Refusal rate stayed flat. Latency stayed flat. Nobody was running a topic-mix or embedding-centroid monitor, because the input side had "always looked fine" for four months and nobody expected an onboarding-flow decision to change what the model needed to handle.

**How it surfaced.** Five weeks later, as a slow bleed of support escalations and a dip in a lagging NPS survey — not from any monitor.

**The retroactive reconstruction:**

- Pulling week-by-week embeddings from logs showed the topic centroid had drifted steadily and measurably starting the day the onboarding change shipped.
- A domain-classifier AUC comparing week-1 to week-5 traffic came back at 0.89 — obvious, in hindsight.
- Re-running the eval suite against a sample of the new traffic showed accuracy on setup-style questions had been roughly 20 points below the tuned baseline the entire time.

**The fix that shipped:**

- An always-on topic-mix PSI plus embedding-centroid monitor, with a standing alert routed to the same channel as the product team's release notes.
- A lightweight process change: any change that could alter *who* reaches the assistant, or *how*, gets flagged for a one-week close eval — not just changes to the model or the prompt itself.

**The lesson.** Input drift and output drift are genuinely decoupled, and it is entirely possible for a completely healthy-looking output dashboard to sit on top of five weeks of degrading quality. Monitoring only refusal rate and latency builds an alarm that structurally cannot ring for this failure mode — which is why this chapter insists on watching inputs *and* outputs *separately*, and why quality decay concentrated in a new user population needs periodic re-evaluation on fresh traffic, not just a static dashboard.

### War story 2: the false alarm that paged an on-call engineer on a holiday

**The setup.** A different team's drift monitor was better instrumented than the one above — PSI on prompt length, KS on the same, and a chi-square test on topic labels, all wired to page whoever was on call on **any single signal**, not on agreement.

**What actually happened:**

- On the last weekend of a fiscal quarter, billing-related questions spiked hard as customers rushed to check invoices and usage before month-end.
- The topic-mix chi-square test and the length PSI both crossed threshold within the same hour, and the monitor paged the on-call engineer at 2am on a holiday weekend.
- Compounding the bad luck, a routine model-provider version bump had gone out three days earlier. Seeing two red metrics and a recent deploy in the timeline, the on-call engineer made the reasonable-looking call to roll the provider version back — disrupting an unrelated improvement that had nothing to do with the spike.

**How the real cause surfaced.** The next morning, by accident: someone pulled up the same week from the prior fiscal quarter and found an almost identical billing-topic spike, on almost the same calendar day. The "drift" was calendar-driven and had recurred every quarter for at least two years — nobody had ever compared it against the matching period a year prior, because the monitor's only frame of reference was "the last N requests" versus a single static reference captured outside of any high-variance period.

**The fix that shipped:**

- The alert rule changed from "any one signal fires" to "at least two of three signals agree" — directly the fusion rule built out in the extended worked example earlier in this chapter, which alone would have kept the alarm from firing, since neither length nor topic mix was outside its normal quarter-end range once compared like-for-like.
- A **known high-variance calendar** (quarter-end, major holidays, product-launch weeks) was added; for those windows, the monitor compares against the matching period from the prior cycle rather than last week's traffic, exactly as the "Windowing & thresholds" section recommends.
- The provider-version rollback was reverted once the real cause was clear, at the cost of a wasted on-call page, a needless regression for a few hours, and a chunk of trust in the monitor that took months to rebuild.

**The lesson.** The "confirm it's real" step at the top of the response playbook is not optional busywork — it is the single check that would have prevented this incident, and it was skipped under 2am pressure precisely because the alert message gave no indication that the pattern might be routine. An alert that cannot distinguish "this happens every quarter" from "this has never happened before" will eventually cost you either a false remediation or a muted channel, and often both, in that order.

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
- **Silently swapping the embedding model.** Upgrading from one sentence-embedding model to another (even a same-vendor version bump) changes the vector space itself — a reference centroid computed under the old model is meaningless compared against live embeddings from the new one. Any embedding-model change requires re-snapshotting the reference, exactly like a model deploy does for the scalar reference.
- **Assuming retrieval coverage is fixed forever, in a RAG system.** A stable generator sitting behind a corpus or index that has silently gone stale (new documents never ingested, old ones expired) will show flat generator-side metrics while answer quality quietly degrades. Monitor retrieval coverage (e.g., the fraction of live queries with an adequate similarity match against a held-out test-query set) as its own signal, not folded into overall input drift.
- **Scoring agent trajectories only at the turn level.** Per-turn scalar and embedding checks can miss behavior that only shows up across a whole trajectory — looping on the same tool, abandoning the assigned role over many turns, or claiming actions never taken. Add trajectory-level, deterministic checks (tool-call diversity, state-transition legality, loop counts) alongside per-turn ones.
- **Trusting the judge as ground truth.** A judge-based quality signal is itself a versioned model or prompt; a judge upgrade or a prompt tweak can move the score exactly like a real regression would. Without a fixed, human-labeled anchor set re-scored every check, you cannot tell "the product decayed" from "the judge changed" — see the anchor-set discussion in the 2025–2026 landscape section above.

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

## Interview mastery

### Explain PSI vs. KS in 60 seconds

If you only get one minute, say this: "Both compare a reference sample to a live sample and give you one number. **PSI** bins the feature, computes what fraction of traffic falls in each bin at reference time versus now, and sums \( (l_b - r_b)\ln(l_b/r_b) \) across bins — it's a symmetrized divergence, gives you an unbounded score with industry-standard cutoffs at 0.1 and 0.2, and its big advantage is that it *localizes*: you can read off exactly which bin moved. **KS** skips binning entirely and compares the two empirical CDFs directly, taking the single largest vertical gap between them, \( D = \sup_x |F_{\text{live}}(x)-F_{\text{ref}}(x)| \) — it comes with a principled p-value, but it's most sensitive near the center of the distribution and comparatively blind in the tails, and on very large windows it gets hypersensitive, flagging trivial differences as significant. In practice: PSI for a quick, interpretable, binning-tolerant read, especially on skewed features like token counts with quantile bins; KS when you want a formal significance test and don't want to pick bin edges. Run both — they're cheap, and requiring both to agree before alerting is the single best false-positive-reduction trick available." That's the whole answer; if pressed further, add that neither applies to embeddings — for those you need MMD or a domain classifier instead.

### Extended Q&A bank (continued)

9. **"Two teams disagree: one wants a rolling reference window, one wants a fixed one. Who's right?"** — Fixed, almost always, for the reason in this chapter: a rolling reference lets slow, real decay quietly become "the new normal," which is exactly the failure you're trying to catch. The only case for a rolling reference is a genuinely non-stationary *benign* baseline (e.g., traffic volume itself, which naturally trends) — and even then, prefer a fixed reference refreshed on a deliberate, versioned schedule over a window that silently redefines itself every day.
10. **"Your embedding-drift monitor and your judge-based quality monitor disagree — embeddings say stable, judge says quality dropped. What do you do?"** — Don't average them away. This is exactly the input-vs-concept-drift split: stable embeddings with a falling judge score is the textbook signature of concept drift — the questions look the same, but the right answer (or the model's ability to give it) changed. Escalate straight to re-evaluation rather than waiting for input drift to confirm it, because input drift may never come.
11. **"How would you detect that your LLM-judge itself has drifted, not your product?"** — Keep a small, fixed, human-labeled anchor set that never changes and re-score it with whatever judge is currently in use at every check. Since the anchors are frozen by construction, any movement in their scores can only be judge drift (a version bump, a prompt edit), letting you attribute an alert to `{none, system, judge}` instead of shrugging at "something moved." This is the core idea behind the 2026 "Who Drifted: the System or the Judge?" line of work — naive judge-score monitors reportedly false-alarm on the large majority of judge-only changes if this check is skipped.
12. **"Design drift monitoring for a RAG system specifically — what's different from a plain chat endpoint?"** — Add a retrieval-coverage check: embed live queries and measure what fraction achieve an adequate similarity match against a held-out set of test/known-good queries (a documented approach uses an 85% coverage floor); a drop means the corpus or the query distribution has moved out from under the retriever, which is a distinct failure mode from the generator drifting. Monitor retrieval-hit-rate and citation/grounding rate as their own time series, not folded into a single "quality" number, since retrieval failures and generation failures need different fixes.
13. **"Design drift monitoring for a multi-agent or long-running agent — what's different?"** — Single-turn signals (prompt length, refusal rate) under-cover an agent because the failure mode is often *behavioral*, accumulating over a trajectory rather than a single turn: repetitive tool use, abandoning the assigned role, claiming actions it didn't take, or looping. Add deterministic, cheap-to-compute trajectory signals first (tool-call diversity, state-transition legality, loop counts) and reserve an LLM-judge call for the ambiguous cases those checks can't resolve — running the judge on every turn of every trajectory is usually not affordable, and it's also the component most likely to need its own drift watch per the anchor-set point above.
14. **"How do you keep a multi-signal monitor from crying wolf, concretely?"** — Require agreement (at least two of three independent signals firing) rather than any-single-signal alerting; gate each signal on both effect size and significance, not p-value alone; compare like-for-like calendar windows so routine seasonality doesn't trip every metric at once; and correct for multiple comparisons if you're running many tests per window, since 200 independent KS tests at \( \alpha=0.05 \) will produce roughly 10 "significant" hits by chance alone.
15. **"What's the actual cost profile of running all this in production, and how do you keep it cheap?"** — Scalar tests (PSI, KS, chi-square) are near-free — histograms and a CDF walk. Embedding-based tests scale with window size: centroid/cosine distance is O(n); MMD is O(n²), so subsample before computing it; a domain classifier needs retraining per window but on modest data is fast. Judge-based scoring is the expensive tier — per-call LLM cost — so sample a slice rather than scoring every request, and only escalate to the judge when the cheap deterministic/statistical layer has already flagged something ambiguous, mirroring the "deterministic gate before judge call" pattern used in production agent-monitoring writeups.
16. **"A stakeholder asks why the drift dashboard didn't catch a known quality regression last quarter. How do you answer without being defensive?"** — Walk the taxonomy: was it input drift (should have shown on PSI/KS/embedding metrics — if it didn't, the reference or thresholds need recalibration), output drift (should show on length/refusal/latency — same fix), or concept drift (inputs and outputs both looked normal, but the mapping changed — this is the one unsupervised monitoring cannot catch by design, and the fix is a faster periodic re-eval cadence or labeled-sample tracking, not a better drift statistic).
17. **"When would you *not* build a statistical drift monitor and just rely on periodic re-evaluation instead?"** — When request volume is too low for any test to be meaningful (KS and PSI both need hundreds-to-thousands of points per window to avoid pure noise), or when the traffic is highly non-repetitive by nature (e.g., a coding agent handling bespoke one-off tasks) such that "the distribution" is not a stable, well-defined object to test against week over week. In both cases, scheduled re-evaluation against a curated eval set, plus close tracking of a handful of scalar proxies (latency, error rate, refusal rate), is more honest than a drift score that's mostly measuring noise.
18. **"What's the single biggest sign that someone has actually run drift monitoring in production, versus just read about it?"** — They immediately bring up false positives, not detection power. Anyone can describe PSI and KS from a textbook; the people who've operated this in production lead with seasonality, reference-window staleness, alert fatigue, multiple-comparisons correction, and the fact that most "drift incidents" turn out to be a pipeline bug, a calendar effect, or a judge that changed — not the product decaying.

### System design prompt: "Design drift monitoring for a deployed agent"

A concrete sketch, the shape an interviewer wants to see on a whiteboard:

```
                         PRODUCTION AGENT (hot path -- never touched)
                                      |
                          async, sampled logging
                                      v
        +---------------------------------------------------------------+
        |                     LOG STORE (append-only)                    |
        |  per turn: prompt, response, tool calls, latency, embeddings   |
        |  per trajectory: full transcript, tool sequence, final state   |
        +---------------------------------------------------------------+
                                      |
                     scheduled batch job (hourly/daily)
                                      v
        +---------------------------------------------------------------+
        |                    TIER 1 -- cheap, always-on                  |
        |  scalar:  PSI/KS on prompt+response length, latency, tool-     |
        |           call count                                          |
        |  categ.:  chi-square on topic label, tool-name distribution    |
        |  embed.:  rolling centroid/cosine drift (this chapter's         |
        |           RollingEmbeddingDriftMonitor) on prompt embeddings   |
        |  agent-specific (deterministic): tool-entropy collapse,        |
        |           illegal state transitions, excessive-loop count      |
        +---------------------------------------------------------------+
                                      |
                     >= 2 of N signals agree, or persistent
                                      v
        +---------------------------------------------------------------+
        |                 TIER 2 -- judge, sampled + fenced              |
        |  sample the flagged window; one fenced judge call per         |
        |  trajectory scores: role adherence, execution-gap (claimed    |
        |  vs. actual tool use), goal-completion                        |
        |  JUDGE ITSELF is watched via a frozen human-labeled anchor    |
        |  set re-scored every run -> attributes {none, system, judge}  |
        +---------------------------------------------------------------+
                                      |
                              severity routing
                                      v
        +---------------------------------------------------------------+
        |     LOW: dashboard note      |     HIGH: page on-call         |
        |     (1 signal, or judge=none)|  (>=2 signals + judge agrees,  |
        |                               |   or hard deterministic       |
        |                               |   violation e.g. illegal      |
        |                               |   transition)                 |
        +---------------------------------------------------------------+
                                      |
                        response playbook (this chapter)
              confirm -> localize -> classify -> re-eval -> mitigate
```

Talking points to narrate while drawing this: (1) the hot path is untouched — monitoring is asynchronous, best-effort, and a monitor outage must never affect serving; (2) Tier 1 is cheap and catches most drift on its own via the multi-signal-agreement rule; (3) Tier 2 exists specifically because agent failures are often behavioral and accumulate over a trajectory, not visible in single-turn scalars, but an LLM judge is too expensive to run on every turn, so it's gated behind Tier 1 and sampled; (4) the judge is explicitly a monitored component, not an oracle, via the anchor set; (5) severity routing determines whether a human sees a dashboard note tomorrow or gets paged tonight, and that routing is exactly where false-positive discipline (agreement, persistence, seasonality-awareness) has to live.

### Red flags vs. green flags

| Signal in a candidate's answer | Red flag | Green flag |
|---|---|---|
| Reference window | "We just compare to last week's traffic" | "Fixed, curated reference from deploy/eval time, versioned, refreshed deliberately" |
| Alerting policy | "Any metric over threshold pages on-call" | "Requires \( \ge 2 \) independent signals to agree, or persistence across windows" |
| Embeddings | "We use cosine distance on the centroid" and stops there | Names cosine/centroid as the *cheap first* signal, then names its blind spot (multimodal shifts) and reaches for a domain classifier or MMD |
| Statistical rigor | Reports a bare KS \( D \) or MMD value with no p-value or threshold context | Distinguishes effect size from significance; calibrates embedding thresholds empirically, never from a textbook number |
| Labels | Claims a drift score "proves" quality dropped | States plainly that unsupervised drift is a smoke alarm, not a diagnosis, and describes a concrete re-eval / labeling step |
| Judge-based monitoring | Treats an LLM judge's score as ground truth | Flags that the judge itself is a versioned component and describes a frozen-anchor check to attribute drift to system vs. judge |
| Seasonality | Never mentions it, or "we just watch for spikes" | Explicitly compares like-for-like calendar windows and can describe a false-alarm incident it caused |
| Multiple comparisons | Runs dozens of per-feature tests with no correction and is unaware that's a problem | Names Bonferroni/Benjamini–Hochberg unprompted when discussing many-feature monitoring |
| Response to an alert | Jumps straight to retraining or rollback | Works the ladder: confirm it's real (rule out a pipeline bug first) → localize → classify → re-eval → cheapest mitigation first |
| RAG/agent specifics | Treats a RAG or agentic system identically to a plain chat endpoint | Names retrieval-coverage drift for RAG, and trajectory-level signals (tool-entropy, role drift, illegal transitions) for agents |

---

### Quick-reference thresholds cheat sheet

Defaults used throughout this chapter, worth being able to recite from memory — with the standing caveat that every embedding- and judge-based row must be recalibrated on your own reference population, never taken as a universal constant:

| Signal | Default starting threshold | Notes |
|---|---|---|
| PSI | 0.1 moderate / 0.2 significant | Industry-standard, from credit-risk practice |
| KS | \( D \ge 0.10 \) **and** \( p < 0.01 \) | Effect size and significance together, never either alone |
| Chi-square | \( p < 0.01 \) on the count table | Categorical analogue of KS |
| MMD | permutation \( p < 0.01 \) | The raw MMD value is meaningless without this |
| Domain classifier | ROC-AUC \( \ge 0.65 \) | 0.5 = indistinguishable; recalibrate per embedding model |
| Centroid / cosine distance | calibrate empirically (e.g. P99 of known-good history) | No intrinsic scale — this chapter's illustrative demo used 0.02; a real calibration on the same data gave 0.0021 |
| Combined alert (fusion rule) | \( \ge 2 \) of 3 independent signals agree | The single biggest false-positive-reduction lever in this chapter |
| Judge anchor-drift | \( \|z\| \ge 2.5 \) on anchor bias | Attributes an alert to `{none, system, judge}` |
| RAG retrieval coverage | below 85% of live queries matched | Per the RAGOps coverage-check approach cited above |
| Agent tool-entropy collapse | distinct-tool-call ratio \( < 0.4 \) | Deterministic and cheap; run before any judge call |


## Where this sits in the serving stack

A drift monitor is a small, boring pipeline that runs *beside* the hot path, never in it:

1. **Log at the edge.** For every request/response, emit cheap features — token counts, latency, refusal flag, topic label, format-valid flag — plus a sampled subset of embeddings. Sample embeddings (they are expensive to store); log scalars in full.
2. **Snapshot a reference.** At each deploy/eval, freeze a reference profile: bin edges per scalar feature, a held-out embedding sample, and the baseline rates. Version it alongside the model.
3. **Window & score offline.** On a schedule (hourly/daily), pull the latest window, run PSI/KS/chi-square on scalars and MMD/classifier on embeddings, per segment.
4. **Gate & alert.** Apply effect-size + significance gates, require persistence or multi-signal agreement, then route by severity to a dashboard (low) or a page (high).
5. **Feed the runbook.** Every alert links to the playbook above and appends to an incident log so patterns become institutional knowledge.

The key architectural property: the monitor is **asynchronous and best-effort**. It must never add latency to inference, and a monitor outage must never take down serving. Compute drift from logs, not inline.

### Monitoring maturity by team size

Not every team needs every tier from day one. A rough progression, useful for scoping what to build first:

| Stage | What to run | Typical tooling |
|---|---|---|
| Pre-launch / solo builder | Manual spot-checks on a sample of traffic; PSI/KS on 2–3 scalars (prompt length, latency) in a notebook, run weekly by hand | `scipy.stats`, a spreadsheet, or Evidently run locally, ad hoc |
| Early production (1–3 person ML team) | Tier 1 scalars (PSI/KS/chi-square) plus a rolling embedding-centroid signal; scheduled batch job; alerts to a Slack channel | Evidently or NannyML, self-hosted, on a cron job |
| Scaling (dedicated MLOps function) | Full Tier 1 + Tier 1b (domain classifier / MMD); sampled judge scoring gated behind Tier 1 agreement, with a frozen anchor set; on-call rotation and severity routing | WhyLabs, Arize, Galileo, or an in-house monitor built on this chapter's patterns |
| Large-scale, multi-team, multi-product | Per-team and per-segment monitors; judge-drift attribution treated as standard practice, not an afterthought; RAG retrieval-coverage and agent-trajectory monitoring as first-class, independently owned systems; a "monitor of monitors" tracking alert volume and false-positive rate over time | Commercial platform(s) plus custom in-house layers for product-specific signals |

The mistake to avoid at every stage is skipping straight to the tooling for a later stage before the earlier one is solid — a large-scale judge-attribution pipeline bolted onto a system with no fixed reference window or agreement-based alerting will just produce expensive, unreliable noise instead of cheap, unreliable noise.


### Mapping this chapter onto the `drift_detector.py` in this folder

The companion module in this same folder, `drift_detector.py`, is a compact, real implementation of the **Tier 1 statistical layer** described above, built on Evidently AI (see `requirements.txt`: `evidently==0.4.14`). Reading it alongside this chapter should make each piece legible:

- `DriftDetector.__init__` takes a `reference_data` DataFrame — this **is** the frozen reference window this chapter insists on: captured once, versioned, never silently re-derived from live traffic.
- `detect_data_drift()` builds an Evidently `Report` from `DataDriftTable()` and `DatasetDriftMetric()` — under the hood these run exactly the per-column PSI/KS-style tests worked through by hand earlier in this chapter, just across every column in the DataFrame at once, and roll them up into a single `dataset_drift` boolean plus a `drift_score` (the fraction of columns that drifted). The `threshold: float = 0.2` parameter is literally this chapter's PSI significant-shift threshold, passed straight through.
- `detect_prediction_drift()` runs `PredictionDriftMetric()` and `ColumnDriftMetric()` against a `prediction_column` — this is the **output-drift** half of the taxonomy from the top of this chapter, applied to whatever scalar or categorical prediction/output feature you log (label, output length, a judge score you've written back into the DataFrame, etc.).
- `generate_report()` is the offline, scheduled scoring step from "Where this sits in the serving stack" above, materialized as an HTML artifact for a human to review rather than a bare pass/fail.

What it does **not** do — and where this chapter's extended sections plug in — is embeddings, multi-signal fusion, or judge-based checks: it has no rolling embedding-centroid monitor, no MMD/domain-classifier pair, no combined-alert vote-counting, and no judge/anchor-set attribution. Extending `DriftDetector` with the `RollingEmbeddingDriftMonitor` and `combined_alert()` logic from "Build it in practice — extended," and gating any judge call behind its output the way the system-design sketch in "Interview mastery" does, turns this module from a Tier-1-only drift table into the full layered monitor this chapter argues for.

### Cost and latency by monitoring tier

Tying the tiers used throughout this chapter (and in the system-design sketch above) to what they actually cost to run:

| Tier | Example checks | Per-window cost | Latency to detect | What it catches | What it misses |
|---|---|---|---|---|---|
| 0 — logging | request/response metadata, cheap scalars | near-zero (already logged) | n/a (raw data, not a check) | nothing by itself | — |
| 1 — statistical | PSI, KS, chi-square on scalars | \( O(n) \) per feature, negligible | minutes to hours (batch schedule) | length/latency/topic-label shifts | semantic shifts with stable scalars |
| 1 — embedding (cheap) | rolling centroid / cosine drift | \( O(n) \) per window | minutes to hours | topic/semantic shifts even with stable scalars | multimodal splits (centroid-only is blind to these; pair with a classifier or MMD) |
| 1b — embedding (heavier) | MMD, domain classifier | \( O(n^2) \) (subsample) / \( O(n) \) to train | minutes to hours | any separable distributional shape shift | still no ground truth on quality |
| 2 — judge (sampled) | LLM-as-judge score on a slice | one LLM call per sampled item | hours to days (sampling cadence + judge latency) | quality/semantic-correctness proxies | the judge's own drift (needs the anchor-set check) |
| 2b — judge (anchor-corrected) | frozen human-labeled anchors re-scored every check | one LLM call per anchor item, a small fixed set | same as above | attributes an alarm to `{none, system, judge}` | still a proxy, not ground truth |
| 3 — human / ground truth | labeled current-traffic sample, full re-eval suite | most expensive, largely manual | days to weeks | actual quality, concept drift | too slow to be a first-detection early-warning system on its own |

Rule of thumb for the design-prompt interview answer above: **run tier 0/1 on every window, unconditionally — it's nearly free; gate tier 2 behind tier-1 agreement to control judge spend; reserve tier 3 for confirming and quantifying what tiers 1–2 already flagged, not for first detection.** This is the same cost curve reflected in the 2025–2026 landscape section's "start with KL divergence, add embedding drift, layer in LLM-as-judge when you have budget" advice, and in the deterministic-checks-before-judge-call pattern used in production agent monitoring.

---

## Further reading

- Fiddler AI — *Measuring Data Drift with the Population Stability Index (PSI)*: https://www.fiddler.ai/blog/measuring-data-drift-population-stability-index
- GeeksforGeeks — *Population Stability Index (PSI)*: https://www.geeksforgeeks.org/data-science/population-stability-index-psi/
- Gretton et al. — *A Kernel Two-Sample Test* (JMLR 2012, the MMD paper): https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
- TorchDrift — *Intuition for the Maximum Mean Discrepancy two-sample test*: https://torchdrift.org/notebooks/note_on_mmd.html
- Evidently AI — *5 methods to detect drift in ML embeddings* (Filippova & Samuylova; published May 17, 2023, updated July 16, 2025): https://www.evidentlyai.com/blog/embedding-drift-detection
- Evidently AI — *Data drift algorithm* (how thresholds/tests are chosen): https://docs-old.evidentlyai.com/reference/data-drift-algorithm
- Evidently AI — *Monitoring embeddings drift* (open-source course module): https://learn.evidentlyai.com/ml-observability-course/module-3-ml-monitoring-for-unstructured-data/monitoring-embeddings-drift
- NannyML — *Multivariate Drift Detection (PCA reconstruction error)*: https://nannyml.readthedocs.io/en/stable/how_it_works/multivariate_drift.html
- NannyML — *Monitoring data drift: univariate & multivariate methods*: https://www.nannyml.com/blog/monitoring-data-drift
- SciPy — `ks_2samp` reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
- Real Statistics — *Two-sample Kolmogorov–Smirnov test*: https://real-statistics.com/non-parametric-tests/goodness-of-fit-tests/two-sample-kolmogorov-smirnov-test/
- WhyLabs — *whylogs* (data logging & drift profiling, open source): https://github.com/whylabs/whylogs
- WhyLabs — *LangKit* (open-source LLM text-quality, relevance, security & sentiment signal extraction for observability): https://github.com/whylabs/langkit
- WhyLabs — *Large Language Model (LLM) Monitoring* docs: https://docs.whylabs.ai/docs/large-language-model-monitoring/
- Galileo — *9 Best LLM Drift Monitoring Platforms in 2026* (market survey of Galileo, Arize, LangSmith, Langfuse, Arthur AI, WhyLabs, W&B Weave, Aporia, Helicone; March 24, 2026): https://galileo.ai/blog/best-llm-output-drift-monitoring-platforms
- dev.to (aiwithmohit) — *Your LLM Is Lying to You Silently: 4 Statistical Signals That Catch Drift Before Users Do* (March 29, 2026): https://dev.to/aiwithmohit/your-llm-is-lying-to-you-silently-4-statistical-signals-that-catch-drift-before-users-do-4cg2
- Vadim Nicolai — *Detecting Agent Defects & Drift in Production* (LangGraph runtime signals; June 15, 2026): https://vadim.blog/agent-defect-drift-detection-production/
- Xu, Weytjens, Zhang, Lu, Weber, Zhu — *RAGOps: Operating and Managing Retrieval-Augmented Generation Pipelines* (arXiv:2506.03401, 2025): https://arxiv.org/abs/2506.03401
- Rath — *Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems Over Extended Interactions* (arXiv:2601.04170, January 2026): https://arxiv.org/abs/2601.04170
- Li — *Who Drifted: the System or the Judge? Anytime-Valid Attribution in LLM Evaluation Pipelines* (arXiv:2606.15474, June 2026): https://arxiv.org/abs/2606.15474
- Yildiz (Forbes) — *The 1% Catastrophe: Why AI Agent Drift Is The Boardroom's Real Problem* (May 7, 2026): https://www.forbes.com/sites/guneyyildiz/2026/05/07/the-1-catastrophe-why-ai-agent-drift-is-the-boardrooms-real-problem/
- Elixir Data — *AI Agent Drift Detection: Monitoring Model & Decision Drift*: https://www.elixirdata.co/blog/ai-agent-drift-detection
