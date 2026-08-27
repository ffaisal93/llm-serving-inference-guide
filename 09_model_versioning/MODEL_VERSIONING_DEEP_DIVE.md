# Model Versioning & Registry — A Deep Dive

> Tracking, promoting, and rolling back model versions in production serving.

## Why This Matters

A colleague pings you: *"Prod is giving different answers than last week, but nobody
deployed a new model."* You check the model name in the config. It reads
`chatbot-llm`. Same as always. You are certain nothing changed.

Something changed. Maybe a teammate re-ran the fine-tune and pushed to the same
Hugging Face repo. Maybe the base image bumped `transformers` from 4.44 to 4.46 and
the tokenizer now splits emoji differently. Maybe `latest` on your object store now
points at a different set of weights. Maybe the vLLM version changed and the sampling
RNG behaves differently. Each of these silently mutates behavior while the *name* you
pinned stays identical.

An un-pinned "same" model is not the same model. Reproducibility in LLM serving is not
a nice-to-have — it is the difference between "we can explain and roll back this
regression in five minutes" and "we have no idea what we are running." This chapter is
about making a served model version an **exact, immutable, auditable thing**: what it
comprises, how a registry tracks and promotes it, how the server resolves "current
prod," and how you roll back when it goes wrong.

> **Saying it out loud.** The line I'd lead with is: an unpinned "same" model
> is not the same model. Somebody re-runs the fine-tune and pushes to the same
> repo, or the base image bumps transformers and the tokenizer splits emoji
> differently, or latest starts pointing at different bytes — and the name in
> your config never changed, so nothing looks like a deploy. The cost isn't
> the regression itself, it's that you can't explain it or undo it, because
> you don't actually know what you're running. So the whole goal of versioning
> is to make a served model an exact, immutable, auditable thing, and the test
> of whether you've got it is simple: can you say, for a request three weeks
> ago, precisely which bytes answered it.

---

## Core Intuition

Think of a model version the way you think of a **container image digest**, not a tag.

- A tag (`myapp:latest`, `chatbot-llm`) is a *mutable pointer*. It can be repointed at
  any time. It tells you a name, not an identity.
- A digest (`sha256:9f86d0...`) is *content-addressed*. It names the bytes. If the
  bytes change, the digest changes. Two people who pull the same digest get the same
  thing, forever.

Good model versioning gives you both layers, and keeps them separate:

1. **Immutable identity** — a content hash or an append-only version number that never
   moves. This is what you record in logs, evals, and audit trails.
2. **Mutable pointers (aliases/stages)** — human-friendly names like `@champion`,
   `@production`, `staging` that *point at* an immutable version and can be repointed
   during a promotion or rollback.

The server should resolve a mutable pointer **once, at load time**, and then pin the
resolved immutable identity for the life of the process — logging it on every request.
Rollback then becomes "repoint the alias and reload," and provenance becomes "which
exact version answered request `X`."

> **Saying it out loud.** The mental model I use is container image tags
> versus digests. A tag like latest is a mutable pointer — it's a name, not an
> identity, and anyone can repoint it. A digest is content-addressed: it names
> the bytes, so if the bytes change the digest changes, and two people pulling
> the same digest get the same thing forever. Good model versioning keeps both
> layers but keeps them separate — an immutable version id you log and
> evaluate against, plus human-friendly aliases like @champion that point at
> one immutable version and can be moved. And the discipline that makes it
> work is that the server resolves the alias once, at load time, then pins and
> logs the resolved id for the life of the process — otherwise a promotion
> splits traffic mid-flight and your logs can't tell you which version
> answered.

---

## What Must Be Versioned Together

The single most common LLM-serving mistake is versioning *only the weights*. A served
model is a **bundle**. Change any component and outputs can move. Pin all of it or you
have not pinned anything.

| Component | Why it changes outputs | Failure if unpinned |
|---|---|---|
| **Weights** (safetensors/GGUF/etc.) | The model itself | Different answers; the obvious one |
| **Model config** (`config.json`, arch, rope/context, dtype) | Defines how weights are interpreted | Wrong context length, silent truncation |
| **Tokenizer** (vocab, merges, special tokens, chat template) | Maps text ↔ tokens | Prompt/formatting drift, off-by-one special tokens |
| **Generation config** (temp, top_p, stop, max_new_tokens defaults) | Shapes sampling | "Same prompt, different vibe" |
| **Serving/adapter code** (pre/post-processing, prompt template, LoRA merge) | Wraps the model | Prompt template drift is a top silent regression |
| **Inference engine + version** (vLLM, TGI, TensorRT-LLM, llama.cpp) | Kernels, sampling RNG, quant handling, batching | Numeric drift, different quant results |
| **Quantization recipe** (AWQ/GPTQ/FP8 params, calibration set) | Alters the effective weights | Quality cliff that "weights hash" alone won't catch |
| **Runtime deps** (`torch`, CUDA, `transformers`, flash-attn) | Kernel/numeric behavior | Reproducibility gaps across hosts |
| **Hardware/precision assumptions** (GPU arch, bf16 vs fp16) | Numeric results differ | Cross-host non-determinism |

Reproducibility checklist — a version is not reproducible unless you can answer all of:

- [ ] Exact weights identified by **content hash** (not a moving tag)
- [ ] `config.json` + `tokenizer.*` + chat template captured with the weights
- [ ] `generation_config.json` / default sampling params captured
- [ ] Inference engine **name and exact version** recorded
- [ ] Quantization recipe + calibration data recorded (if quantized)
- [ ] Serving-code commit SHA recorded
- [ ] Runtime deps pinned (lockfile / image digest)
- [ ] Training/lineage: source run, data snapshot, base model revision
- [ ] Eval results linked to *this* version id

> Practical shortcut: bake weights + tokenizer + config + engine into a **container
> image referenced by digest**, and register that digest as the version's artifact. The
> image digest content-addresses most of the bundle in one shot.

> **Agentic serving note:** once an LLM is wrapped in an agent, the bundle grows again —
> the *system prompt*, the *tool/function schemas*, and the *engine version* all shape
> behavior as much as the weights do, and each now has its own release cadence. See
> [The 2025–2026 Landscape](#the-20252026-landscape) below for how modern registries
> version prompts and tool schemas alongside the model itself.

> **Saying it out loud.** The most common mistake is versioning only the
> weights. A served model is a bundle: weights, config, tokenizer and chat
> template, generation defaults, the serving code that wraps it, the inference
> engine and its version, the quantization recipe, and the runtime
> dependencies. Change any one of those and the outputs move. The one that
> catches people most often is the tokenizer and chat template, because it
> drifts through a dependency bump nobody associates with the model at all. My
> shortcut is to bake weights plus tokenizer plus config plus engine into a
> container image and register that image digest as the version's artifact —
> one digest content-addresses most of the bundle in a single shot.

---

## Immutable, Content-Addressed Artifacts

An artifact is **content-addressed** when its identifier is a cryptographic hash of its
bytes. Two properties fall out for free:

1. **Integrity** — re-download and re-hash; if it matches, the bytes are intact.
2. **Deduplication & identity** — identical artifacts share an id; different bytes get
   different ids. You cannot accidentally overwrite version 5 with new content and keep
   the id.

You can express a version identity as the hash over the *ordered set* of component
digests:

\[
\text{version\_id} = H\big( H(\text{weights}) \,\|\, H(\text{tokenizer}) \,\|\, H(\text{config}) \,\|\, \text{engine\_ver} \,\|\, \text{code\_sha} \big)
\]

where \( H \) is a strong hash (SHA-256) and \( \| \) is concatenation. If any input
byte changes, \( \text{version\_id} \) changes. This is exactly how Docker image
digests, Git commit SHAs, and safetensors integrity checks work.

**Semantic version vs content hash — use both, for different jobs.**

- A **semantic/registry version** (`v3`, `2.1.0`, or MLflow's auto-incremented integer)
  is *human-ordered*: it tells you "newer than v2," carries release intent, and is what
  people talk about. It does **not** guarantee the bytes are unique or unchanged.
- A **content hash** is *machine-truth*: it guarantees identity but is unordered and
  unreadable. It is what you log and verify against.

Best practice: assign a monotonic registry version for humans, and record the content
hash(es) as immutable metadata/tags on that version. Never reuse a version number for
different bytes.

> **Saying it out loud.** Content-addressed just means the identifier is a
> hash of the bytes, and two useful things fall out for free. Integrity:
> re-download, re-hash, and if it matches, nothing was corrupted or swapped.
> And identity: you physically cannot overwrite version five with new content
> and keep the same id, because different bytes give a different hash. What
> I'd add is that you want both a content hash and a human version number,
> because they do different jobs — the integer tells a person "newer than v2"
> and carries release intent, the hash is machine truth. The rule that ties
> them together is: never reuse a version number for different bytes.

---

## Artifact Storage

LLM artifacts are big — a 70B model in bf16 is ~140 GB; even a 7B is ~14 GB. Storage
choices are shaped by size:

- **Object storage** (S3, GCS, Azure Blob) is the default backing store. It is cheap,
  durable, versioned, and content-addressable if you key objects by hash. Registries
  (MLflow, SageMaker, Vertex) all store *metadata* in a database and *artifacts* in
  object storage.
- **Enable object-versioning / immutability** (S3 Object Lock, GCS object versioning)
  so a bucket write cannot silently mutate an existing version's bytes.
- **Deduplicate** with content-addressed keys: `s3://models/by-hash/<sha256>`; the
  registry version just references the hash. Identical LoRA adapters, tokenizers, and
  base weights are stored once.
- **Mind egress and cold-start.** Pulling 140 GB per pod on autoscale is slow and
  expensive. Common mitigations: node-local caches, a shared read-only volume
  (EFS/Filestore), pre-warmed images, or a peer-to-peer distributor. The version id
  must stay stable regardless of *where* it is cached.
- **Git LFS** underpins the Hugging Face Hub: each repo is a Git repo, large files go to
  LFS, and every commit is a content-addressed revision (this is why HF pinning works —
  more below). Newer chunk-based, content-addressed backends (Hugging Face's Xet, OCI
  artifact registries) push the same idea further — see
  [The 2025–2026 Landscape](#the-20252026-landscape).

> **Saying it out loud.** The thing that shapes storage here is just size — a
> 70B model in bf16 is roughly 140 gigabytes, and even a 7B is around 14. So
> object storage is the default backing store, with object versioning or
> object lock turned on so a write can't silently mutate an existing version's
> bytes, and keys derived from the content hash so identical tokenizers and
> base weights get stored once. The operational pain isn't storage cost, it's
> cold start: pulling 140 gigabytes onto every new pod during an autoscale
> event is slow enough to break your scaling behavior. So you add node-local
> caches, a shared read-only volume, or pre-warmed images — but the version id
> has to stay identical no matter where the bytes were cached from.

---

## Registries & the Promotion Workflow

A **model registry** is the source of truth that maps human-facing pointers to
immutable versions, records lineage/metadata, and gates promotion. Two pointer models
exist; modern registries favor the second:

### Stages vs Aliases

- **Stages** (classic): a version lives in one of `None → Staging → Production →
  Archived`. Exactly one stage per version; transitions move a version between buckets.
  Simple, but coarse — you get one "Production" slot and rigid semantics. MLflow has
  **deprecated stages** in favor of aliases + tags.
- **Aliases** (modern): named, repointable pointers (`@champion`, `@challenger`,
  `@production`, `@canary`) that each point at exactly one version. A version can carry
  many aliases; you can have `@champion` and `@shadow` simultaneously. Aliases decouple
  "what code loads" from "which version is behind it." MLflow, Vertex, and (effectively)
  HF branches all use this model.

> **Saying it out loud.** Stages were the classic model: a version sits in
> exactly one of None, Staging, Production, or Archived. It's simple but
> coarse — you get one production slot. Aliases are the modern answer: named
> pointers like @champion, @challenger, @shadow, each pointing at exactly one
> version, and one version can carry several. That matters because in LLM
> serving the normal case is having a champion, a canary, and a shadow all
> live at once, which the single-slot model just can't express. MLflow
> deprecated stages in favor of aliases for exactly this reason, and Vertex
> and Hugging Face branches are effectively the same shape.

### Gated Promotion Workflow (dev → staging → prod)

A promotion is **a pointer move guarded by evidence**, not a rebuild:

```
                 register (immutable version N, content-hashed)
                        │
                        ▼
   [dev]  ──► automated evals + smoke tests ──► set alias @staging → N
                        │  (metrics logged and LINKED to version N)
                        ▼
   [staging]  ──► shadow / offline evals / human approval (gate)
                        │  approver signs off; validation_status=approved
                        ▼
   [prod]  ──► set alias @champion → N   (server reloads / picks up)
                        │
                        ▼
   rollback ──► set alias @champion → N-1   (previous version still intact)
```

The critical properties:

- **Promotion never mutates bytes.** It only moves a pointer to an already-registered,
  immutable version. This is what makes rollback trivial and instant.
- **Gates are enforced, not advisory.** A version should be blocked from `@champion`
  unless eval metrics on *that version id* pass thresholds and (for prod) an approver
  signed off. Encode gates in CI/CD, not tribal knowledge.
- **Approvals are recorded** on the version (who, when, against which eval run).
- **The previous prod version stays registered and warm-able**, so rollback is a
  pointer move back, not a rebuild-and-redeploy.

> **Saying it out loud.** The one-sentence version is that a promotion is a
> pointer move guarded by evidence — never a rebuild. You register an
> immutable, content-hashed version once; you run evals and attach the results
> to that exact version id; then a gate checks those results and moves the
> alias. Two properties follow. Rollback is instant, because the previous
> version was never destroyed and you're just moving the pointer back. And the
> gate has to be enforced in CI rather than being tribal knowledge — including
> failing closed when there are no eval results at all, because a gate that
> treats "missing data" as "pass" will eventually promote something nobody
> evaluated.

---

## Fully Worked Example: MLflow Registry Workflow

This is a real, correct MLflow (3.x) workflow: log + register a transformers model with
its tokenizer, link eval metrics, use aliases as gates, promote to `@champion`, and load
by alias in the server. It also shows how the server resolves "current prod version,"
walks through a **full dev→staging→prod pipeline gated by an eval threshold**, and ends
with a **rollback drill** that simulates a bad promotion and recovers from it.

### 1. Log the full bundle and register a version

```python
import mlflow
from mlflow import MlflowClient
from transformers import AutoModelForCausalLM, AutoTokenizer

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("chatbot-llm")

MODEL_NAME = "chatbot-llm"          # registered model (the "name")
BASE = "meta-llama/Llama-3.1-8B-Instruct"
BASE_REVISION = "0e9e39f"           # pin the base model commit (see HF section)

model = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION)
tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)

with mlflow.start_run() as run:
    # Log weights + tokenizer TOGETHER so they can never drift apart,
    # and register a new immutable version in one call.
    info = mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        name="model",
        registered_model_name=MODEL_NAME,          # -> creates/append version
        # Pin the runtime so the bundle is reproducible:
        pip_requirements=[
            "transformers==4.44.2",
            "torch==2.4.0",
            "accelerate==0.33.0",
        ],
    )
    # Capture lineage + the exact engine/code we intend to serve with.
    mlflow.set_tag("git_sha", "a1b2c3d")
    mlflow.set_tag("base_model_revision", BASE_REVISION)
    mlflow.set_tag("serving_engine", "vllm==0.6.2")

version = info.registered_model_version   # e.g. "7" — immutable, monotonic
print("registered", MODEL_NAME, "version", version)
```

### 2. Link eval results to *this* version, and gate with an alias

```python
client = MlflowClient()

# Run your eval harness against THIS version id, then record results
# ON the version so promotion decisions are auditable.
eval_exact_match = 0.712
eval_toxicity = 0.004

client.set_model_version_tag(MODEL_NAME, version, "eval_exact_match", str(eval_exact_match))
client.set_model_version_tag(MODEL_NAME, version, "eval_toxicity", str(eval_toxicity))
client.set_model_version_tag(MODEL_NAME, version, "validation_status", "pending")

# First gate: expose as challenger for staging/shadow traffic.
client.set_registered_model_alias(MODEL_NAME, "challenger", version)
```

### 3. Gated promotion to production

```python
def promote_to_champion(name: str, version: str,
                        min_em: float = 0.68, max_tox: float = 0.01) -> None:
    mv = client.get_model_version(name, version)
    em = float(mv.tags.get("eval_exact_match", "0"))
    tox = float(mv.tags.get("eval_toxicity", "1"))
    if em < min_em or tox > max_tox:
        raise RuntimeError(f"gate failed: em={em} tox={tox}")
    # (human approval would be checked here too, e.g. an approved tag)
    client.set_model_version_tag(name, version, "validation_status", "approved")
    # Atomically repoint the production pointer at the new version.
    client.set_registered_model_alias(name, "champion", version)
    print(f"{name} @champion -> v{version}")

promote_to_champion(MODEL_NAME, version)
```

### 4. The server loads BY ALIAS and pins the resolved version

```python
# --- serving process, at startup ---
import mlflow
from mlflow import MlflowClient

MODEL_NAME = "chatbot-llm"
ALIAS = "champion"

client = MlflowClient()

# Resolve the alias ONCE to an immutable version + source, and pin it.
mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
RESOLVED_VERSION = mv.version          # e.g. "7"
RESOLVED_SOURCE = mv.source            # artifact URI / storage location
RESOLVED_RUN = mv.run_id
print(f"serving {MODEL_NAME} @{ALIAS} = v{RESOLVED_VERSION} (run {RESOLVED_RUN})")

# Load the exact bundle (weights + tokenizer). Loading by @alias is convenient,
# but we resolved+logged the concrete version above so every request is auditable.
pipeline = mlflow.transformers.load_model(f"models:/{MODEL_NAME}@{ALIAS}")

def handle(request_text: str) -> dict:
    out = pipeline(request_text)
    # Stamp the immutable version on every response/log line.
    return {"model": MODEL_NAME, "version": RESOLVED_VERSION, "output": out}
```

**How the server resolves "current prod version":** it asks the registry for the version
behind the `@champion` alias (`get_model_version_by_alias`) at load time, records the
returned immutable version id, and serves that. It does **not** re-resolve per request —
otherwise a mid-flight promotion would split traffic across versions unpredictably.
Instead, a promotion signals a controlled reload (rolling restart, or a
watch-and-drain), and until then the process keeps serving its pinned version and logs
it on every request.

### 5. Rollback is a pointer move

```python
# Something regressed in v7. Point production back at the known-good v6.
client.set_registered_model_alias(MODEL_NAME, "champion", "6")
# Trigger the servers to reload (rolling restart). v7 stays registered for forensics.
```

> Load-by-version equivalent (fully pinned, no alias indirection):
> `mlflow.transformers.load_model("models:/chatbot-llm/7")`. Use aliases for
> operability; use explicit versions when you want the config file itself to be the pin.

> **Saying it out loud.** Rolling back is the part people get wrong in
> interviews, so I'd be concrete: it's one API call that repoints the alias to
> the last known-good version, plus whatever makes the servers actually reload
> — usually a rolling restart. The bad version stays registered, which you
> want, because you need it for forensics. And the pointer move is the easy
> half; the SLO you should quote is repoint plus drain plus serving the
> previous version in a few minutes with zero rebuild. If reverting requires
> re-running training, re-quantizing, or rebuilding an image, you don't have
> rollback — you have a second forward deploy, and it'll take hours.

### 6. Build it in practice, extended — a full promotion pipeline

The three-step gate above (`challenger` → check tags → `champion`) is correct but
minimal. A real pipeline runs unattended in CI, evaluates *every* candidate the same
way, keeps an explicit record of "what was champion before," and refuses to promote on
missing or stale data. Here is a complete, runnable gate function that drives a
candidate through `dev → staging → prod`, each hop guarded by a score threshold read
from the version's own tags — never from a human's memory:

```python
from dataclasses import dataclass
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

client = MlflowClient()
MODEL_NAME = "chatbot-llm"

@dataclass
class Gate:
    alias: str          # alias this stage promotes TO on success
    min_em: float        # exact-match threshold
    max_tox: float        # toxicity ceiling
    require_approval: bool = False   # prod requires a human tag, staging doesn't

PIPELINE = [
    Gate(alias="staging",  min_em=0.60, max_tox=0.02, require_approval=False),
    Gate(alias="champion", min_em=0.68, max_tox=0.01, require_approval=True),
]

def run_eval_harness(version: str) -> dict:
    """Stub: run your real eval suite against THIS registered version's
    artifact (never against a local checkpoint that might differ) and
    return metrics. In production this loads models:/{name}/{version}."""
    ...  # pretend this returns freshly computed numbers
    return {"exact_match": 0.712, "toxicity": 0.004}

def record_eval(name: str, version: str, metrics: dict) -> None:
    for k, v in metrics.items():
        client.set_model_version_tag(name, version, f"eval_{k}", str(v))
    client.set_model_version_tag(name, version, "eval_ts", str(mlflow.utils.time.get_current_time_millis()))

def gate_passes(name: str, version: str, gate: Gate) -> tuple[bool, str]:
    mv = client.get_model_version(name, version)
    tags = mv.tags
    if "eval_exact_match" not in tags or "eval_toxicity" not in tags:
        return False, "no eval recorded on this version — refusing to promote blind"
    em = float(tags["eval_exact_match"])
    tox = float(tags["eval_toxicity"])
    if em < gate.min_em:
        return False, f"exact_match {em:.3f} < required {gate.min_em}"
    if tox > gate.max_tox:
        return False, f"toxicity {tox:.3f} > allowed {gate.max_tox}"
    if gate.require_approval and tags.get("approved_by") is None:
        return False, "prod promotion requires a human 'approved_by' tag"
    return True, "ok"

def promote_through_pipeline(name: str, version: str) -> None:
    """Runs a candidate through every gate in order. Before EACH promotion
    to an alias, snapshot the alias's CURRENT target as 'last_known_good'
    so a rollback never has to guess what was there before."""
    for gate in PIPELINE:
        ok, reason = gate_passes(name, version, gate)
        if not ok:
            raise RuntimeError(f"blocked at @{gate.alias}: {reason}")

        # Snapshot the outgoing version for this alias, if one exists,
        # so rollback is a single lookup instead of an archaeology dig.
        try:
            previous = client.get_model_version_by_alias(name, gate.alias)
            client.set_model_version_tag(
                name, version, f"previous_{gate.alias}", previous.version
            )
        except MlflowException:
            pass  # first-ever promotion to this alias — nothing to snapshot

        client.set_registered_model_alias(name, gate.alias, version)
        print(f"{name} @{gate.alias} -> v{version}  ({reason})")

# --- usage ---
metrics = run_eval_harness(version)
record_eval(MODEL_NAME, version, metrics)
client.set_model_version_tag(MODEL_NAME, version, "approved_by", "ml-lead@company.com")
promote_through_pipeline(MODEL_NAME, version)
```

Two details do the real work here:

- **`gate_passes` refuses to promote a version with no eval tags at all**, rather than
  treating "missing" as "pass." A pipeline that fails open on missing data will
  eventually promote something nobody ever evaluated.
- **`previous_<alias>` is written on the incoming version, at promotion time**, not
  read from history after the fact. That single tag is what turns the rollback drill
  below from "grep the audit log" into "read one tag."

> **Saying it out loud.** The detail worth stealing from this pipeline is
> writing a previous_champion tag onto the incoming version at promotion time,
> rather than reconstructing history afterwards. That turns rollback from
> "grep the audit log and hope" into "read one tag," which matters at three in
> the morning when the person on call didn't do the promotion. The other half
> is the gate refusing to promote a version that has no eval tags at all.
> Failing open on missing data is the quiet killer — the pipeline looks like
> it's protecting you right up until the eval job silently didn't run.

### 7. Rollback drill — simulate a bad promotion and recover by alias

Run this as an actual drill (quarterly, or after onboarding a new on-call) so the first
time your team executes a rollback is not during a real incident.

```python
# --- STEP 0: baseline. v6 is a known-good champion. ---
client.set_registered_model_alias(MODEL_NAME, "champion", "6")

# --- STEP 1: a bad promotion. v7 passed the automated gate (its eval
#     harness ran on a stale eval set that didn't catch a regression) and
#     got promoted. This is the failure mode gates alone don't fully close —
#     which is why 'previous_champion' bookkeeping and monitoring both matter.
promote_through_pipeline(MODEL_NAME, "7")     # @champion now -> v7
assert client.get_model_version_by_alias(MODEL_NAME, "champion").version == "7"

# --- STEP 2: detection. Online monitoring (thumbs-down rate, error rate,
#     a canary eval re-run against live traffic) keyed by the version_id
#     stamped on each request shows v7's quality is worse than v6's.
#     This is the payoff of "stamp the version on every response" from
#     Step 4 above: the alert can say "v7 regressed" instead of "prod is bad."

# --- STEP 3: rollback. Read the snapshot this SAME promotion wrote,
#     don't rely on memory of what was running an hour ago.
def rollback(name: str, alias: str, bad_version: str) -> str:
    mv = client.get_model_version(name, bad_version)
    previous = mv.tags.get(f"previous_{alias}")
    if previous is None:
        raise RuntimeError(
            f"no previous_{alias} tag on v{bad_version} — cannot auto-rollback; "
            "fall back to the registry's version history for this model"
        )
    client.set_registered_model_alias(name, alias, previous)
    client.set_model_version_tag(name, bad_version, "validation_status", "rolled_back")
    return previous

restored = rollback(MODEL_NAME, "champion", "7")
assert restored == "6"
assert client.get_model_version_by_alias(MODEL_NAME, "champion").version == "6"
print(f"rolled back: @champion -> v{restored} (v7 kept registered for forensics)")

# --- STEP 4: verify the resolved artifact, not just the pointer. ---
mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
print(f"confirmed serving v{mv.version} from {mv.source}")
# In a real drill, also re-hash the artifact at mv.source and compare it
# against the content hash recorded when v6 was first registered — a
# pointer move is only a real rollback if the bytes behind it are the
# bytes you think they are.
```

What the drill is meant to prove, and what to time when you run it for real:

- **Time-to-detect** — how long between the bad promotion and monitoring flagging
  `v7` by version id (not "prod feels off").
- **Time-to-rollback** — how long from "roll back" decided to `@champion` pointing at
  `v6` again and servers actually serving it (this is the pointer move *plus* the
  reload, not just the API call).
- **No rebuild anywhere in the path.** If any step in your real rollback requires
  re-running training, re-quantizing, or rebuilding an image, the drill has found a gap
  — close it before you need it under pressure.

> **Saying it out loud.** The reason I'd run this as a real drill, quarterly,
> is that a rollback path you've never executed is a hypothesis. There are
> three numbers to time: time to detect — how long until monitoring names the
> bad version by id, not just "prod feels off"; time to roll back — from
> decision to servers actually serving the old version, which is the pointer
> move plus the reload; and whether any step in the path requires a rebuild.
> That last one is pass/fail. And the drill only counts if you verify the
> bytes behind the restored pointer, because a pointer move is only a rollback
> if what's behind it is what you think it is.

---

## Registry Comparison

| Feature | MLflow Model Registry | SageMaker Model Registry | Vertex AI Model Registry | Hugging Face Hub |
|---|---|---|---|---|
| Version unit | Registered model + integer version | Model Package Group + Model Package | Model resource + version id | Git repo + commit (revision) |
| Pointer mechanism | **Aliases** + tags (stages deprecated) | **Approval status** (`Pending/Approved/Rejected`) | **Version aliases** (incl. `default`) | Branches / tags / commit SHA |
| Immutable id | Version number + logged artifact hash | Model Package ARN | Version id (immutable) | **Commit hash** (content-addressed via Git/LFS) |
| Gated promotion | Alias move + tag gates in CI | Approval status flip (EventBridge-triggerable) | Alias reassignment | PR/branch merge; manual convention |
| Artifact store | Pluggable (S3/GCS/Azure/local) | S3 (+ ECR image) | Google Cloud Storage | Git LFS on the Hub |
| Lineage/metadata | Runs, params, metrics, tags | Metrics, data lineage, source pipeline | Metadata, eval, dataset links | Model card ([`README.md`](README.md) + YAML) |
| Approval/audit | Tags + external gate | Native approval workflow + audit | IAM + Cloud Audit Logs | Repo history / commits |
| Best when | Open-source, self-hosted MLOps | Deep AWS + Pipelines/EventBridge | Deep GCP + Vertex Endpoints | Public/OSS models, git-native pinning |

Key nuances:

- **MLflow** deprecated the `None/Staging/Production/Archived` **stages** in favor of
  named **aliases** + tags — repoint an alias to promote or roll back. Load with
  `models:/<name>@<alias>` or `models:/<name>/<version>`.
- **SageMaker** organizes versions under a **Model Package Group**; promotion is a flip
  of **approval status** (`PendingManualApproval → Approved`), which can trigger
  downstream deploys via EventBridge. The `ModelPackageArn` is the immutable handle.
- **Vertex AI** keeps versions under one model resource; **aliases** (e.g. the built-in
  `default`) point at versions. Reference `model@default` or a version id; update the
  alias target to promote without touching client code.
- **Hugging Face Hub** is git-native: every push is a commit, and `revision=` on
  `from_pretrained` pins a **commit hash, branch, or tag**. The commit hash is a true
  content address; a bare `main` is a *mutable* pointer — never rely on it in prod.

> **Saying it out loud.** If someone asks me to pick a registry I'd say the
> mechanics are the same everywhere and the choice follows your cloud. MLflow
> gives you aliases plus tags with a pluggable artifact store — the right
> default if you're self-hosting. SageMaker models it as approval status on a
> model package, which is nice because flipping to Approved can trigger a
> downstream deploy through EventBridge. Vertex uses version aliases including
> a built-in default. And Hugging Face is git-native, where the commit hash is
> a genuine content address. The trap is the same across all four: whatever
> the tool calls its mutable pointer — main, latest, default — never let
> production resolve it at runtime.

---

## The 2025–2026 Landscape

The mechanics above (content hash + alias + gate) are not just theory — they are
exactly what production registries converged on through 2025 and into 2026. Four
concrete developments are worth knowing cold, with real, checkable sources.

### Hugging Face Hub: `revision` is the reproducibility contract

Every Hugging Face Hub repo is a Git repo; every push is a commit, and every commit has
a hash. `AutoModel.from_pretrained(repo_id, revision=...)` accepts that hash, a tag, or
a branch name — but only the **commit hash** is a true content address. Passing `main`
(the default when `revision` is omitted) resolves to *whatever `main` currently points
at*, which is exactly the mutable-tag failure mode this chapter opened with. The
Transformers maintainers spell this out directly on the forum thread discussing
`commit_hash` in `from_pretrained`: the resolved commit hash is threaded through the
loading code specifically so a cached load and a fresh load agree on *which* files they
mean, even if `main` has since moved (Hugging Face Forums, "Purpose of commit_hash in
PreTrainedModel.from_pretrained"). Baseten's engineering blog makes the operational case
explicit: pin `revision` to an exact commit whenever you use `trust_remote_code` (which
executes arbitrary code from the repo) or need reproducible eval numbers, because an
upstream maintainer can push backwards-incompatible or malicious changes to `main`
without you doing anything at all ("Pinning ML model revisions for compatibility and
security," baseten.co, 2024–2025).

Underneath this, Hugging Face has been replacing plain Git LFS with **Xet**, a
content-defined-chunking (CDC) storage backend: files are split into ~64 KB
variable-length chunks by a rolling hash (so an edit only invalidates the chunks that
actually changed, unlike LFS's whole-file versioning), and chunks are stored in a
content-addressed store (CAS) keyed by their own hash. The result is deduplication
*across* repositories, not just across commits of one repo, and a stronger
content-addressing guarantee at the chunk level, not just the file level (Hugging Face,
"Xet Chunk-Level Deduplication Specification" and the "From Chunks to Blocks" engineering
blog post). For versioning purposes the practical upshot is the same lesson, reinforced
at finer grain: identity is a hash of bytes, and `main`/`latest` is not that.

> **Saying it out loud.** For Hugging Face specifically the one thing to know
> is the revision argument. from_pretrained takes a commit hash, a tag, or a
> branch, and only the commit hash is a real content address — omit it and you
> get main, which resolves to whatever main points at right now. That's the
> mutable-pointer failure mode, but with an extra edge: if you're using
> trust_remote_code you're executing arbitrary code from that repo, so an
> unpinned revision is a supply-chain exposure, not just a reproducibility
> one. Underneath, Hugging Face has been moving from Git LFS to Xet, which
> chunks files with a rolling hash so an edit only invalidates the chunks that
> changed and dedup works across repos, not just across commits. Same lesson
> at finer grain: identity is a hash of bytes.

### MLflow's Prompt Registry: versioning the other half of an agent

As of MLflow 3.x (the docs tree covers versions up to 3.15.0, released July 31, 2026),
MLflow ships a dedicated **Prompt Registry** alongside the Model Registry, because in an
agentic system the prompt is a first-class artifact that changes on its own schedule.
It uses the *same* mental model this chapter has been building for models: prompt
versions are immutable once created (Git-inspired, commit-message-per-version), and
named **aliases** move between them for promotion and rollback:

```python
import mlflow

# Register a new, immutable prompt version with a commit message.
mlflow.genai.register_prompt(
    name="agent-system-prompt",
    template="You are a support agent for {{product}}. Be concise...",
    commit_message="Tighten tone, add refund-policy clause",
)

# Promote it the same way you promote a model: repoint an alias.
mlflow.genai.set_prompt_alias("agent-system-prompt", alias="production", version=5)

# The server resolves the alias once, same discipline as @champion above.
prompt = mlflow.genai.load_prompt("prompts:/agent-system-prompt@production")
```

A reserved `@latest` alias always resolves to the newest version for convenience in
dev, but production code should pin `@production` (or an explicit version) for the same
reason it should never load a model by `latest` (MLflow docs, "Prompt Registry" and
"Manage Prompt Lifecycles with Aliases," mlflow.org/docs/latest/genai/prompt-registry/).
MLflow additionally lets you log the *generation parameters* (model name, temperature,
max_tokens) alongside the prompt version, so a prompt version and the sampling config it
was tuned against travel together — closing exactly the "generation config" gap flagged
in the versioning-together table earlier in this chapter.

> **Saying it out loud.** The point of a prompt registry is that in an agentic
> system the prompt is a first-class artifact that changes on its own schedule
> — usually faster than the weights do. MLflow 3 ships one, and it
> deliberately uses the same mental model as the model registry: each prompt
> version is immutable once created, with a commit message, and named aliases
> move between versions for promotion and rollback. There's a reserved @latest
> alias for convenience in dev, and you should no more ship that to production
> than you'd ship a model by latest. The nice extra is that MLflow lets you
> log the generation parameters next to the prompt version, so a prompt and
> the temperature it was tuned against travel together.

### Content-addressed, OCI-packaged models

A second, independent trend treats a model bundle as **an OCI artifact** — the same
content-addressed, layered, digest-referenced format container images use — rather than
inventing a bespoke model format. The CNCF's **ModelPack** project
(`github.com/modelpack/model-spec`) defines an open standard for packaging weights,
tokenizer, and config as OCI layers so a model can be pulled, cached, and run with the
same tooling (registries, signing, Kubernetes volume sources) already built for
containers; **`modctl`** (`github.com/modelpack/modctl`) is its reference CLI for
building and pushing these artifacts. The CNCF's own August 2025 write-up frames the
motivation plainly: "Models can be versioned, distributed, and tracked like container
images," gaining OCI's existing digest-based integrity guarantees and sigstore-based
signing for free instead of re-deriving them per-vendor ("How OCI Artifacts will drive
future AI use cases," cncf.io, August 27, 2025). VMware's Broadcom team has since
documented running **Harbor** — an existing OCI-compliant container registry — as an AI
model registry on top of this, i.e., production teams are already reusing container
infrastructure for model versioning rather than standing up a separate system
("Using Harbor as an AI Model Registry," blogs.vmware.com, March 2026).

> **Saying it out loud.** The bet here is that instead of inventing a bespoke
> model format, you package a model bundle as an OCI artifact — the same
> content-addressed, layered, digest-referenced format container images
> already use. You inherit the whole ecosystem for free: digest-based
> integrity, sigstore signing, existing registries, Kubernetes volume sources,
> multi-arch manifests. CNCF's ModelPack spec and its modctl CLI are the
> standardization effort, and teams are already running Harbor as a model
> registry on top of it. The tradeoff to name is that you give up ML-specific
> metadata — run linkage, eval results, lineage graphs — that a purpose-built
> registry like MLflow hands you natively, unless you layer it back on with
> tags.

### Versioning the whole agentic bundle

Put the three developments together and a pattern falls out for agentic serving: an
agent's observable behavior is now a function of (at least) the model version, the
prompt version, the tool/function-schema version, and the engine version — each
independently versioned in 2025–2026 tooling, each capable of drifting on its own. The
practical fix is the same content-hash-of-components idea from earlier in this chapter,
applied one level up — an explicit **bundle manifest** that a promotion pipeline treats
as a single unit to gate and roll back together:

```yaml
# agent_bundle_manifest.yaml — versions the WHOLE agent, not just the model.
# Hash this file's canonical form and register THAT as the agent's version id.
agent_bundle_version: "support-agent-2026.07.2"
model:
  registry: mlflow
  name: chatbot-llm
  version: 12
  content_hash: "sha256:9f86d0..."
prompt:
  registry: mlflow-prompts
  name: agent-system-prompt
  version: 5
  alias: production
tools:
  schema_hash: "sha256:1a2b3c..."   # hash over the tool/function-calling schema set
  count: 7
engine:
  name: vllm
  version: "0.11.2"
runtime:
  image_digest: "sha256:7c4a8d..."
```

Register this manifest itself as a version (an MLflow run's params/tags, or a
dedicated "agent" entry in whatever registry you use), gate promotion on the manifest
as a whole, and roll back by repointing the manifest's alias — not by repointing the
model, prompt, and tool schema independently and hoping they land on compatible
versions at the same time.

> **Saying it out loud.** Once you wrap a model in an agent, behavior is a
> function of at least four independently versioned things: the weights, the
> system prompt, the tool schemas, and the engine. Each one has its own
> release cadence and each can drift alone. So the fix is the same
> content-hash-of-components idea, applied one level up — write an explicit
> bundle manifest that names the model version, the prompt version, the
> tool-schema hash, the engine version and the runtime image digest, then hash
> that manifest and register it as the agent's version. You gate and roll back
> the manifest as one unit. The failure to avoid is repointing the model alias
> and the prompt alias in two separate uncoordinated steps and landing on a
> combination nobody ever evaluated.

---

## Reproducibility & Rollback Mechanics

**Reproducibility** means: given a version id, you can reconstruct the exact serving
behavior on a fresh host. Mechanically:

1. Resolve the version id → immutable artifact reference (hash/ARN/commit).
2. Fetch bytes from object storage; **re-hash and verify** against the recorded digest.
3. Load with the **pinned engine + pinned runtime** (image digest or lockfile).
4. Apply the **captured generation/serving config** (not the engine defaults).
5. Re-run the version's eval suite; confirm metrics match the recorded numbers within
   tolerance. If they don't, something in the bundle was not actually pinned.

**Rollback** works because the previous version was never destroyed and the pointer is
cheap to move:

- **Registry-level:** repoint `@champion` (MLflow), flip approval / redeploy previous
  `ModelPackageArn` (SageMaker), reassign `default` alias (Vertex), or pin the prior
  commit (HF). O(1) metadata operation.
- **Serving-level:** the server must actually *reload*. Options: rolling restart of
  pods, a sidecar that watches the alias and drains+reloads, or blue/green where the old
  version's replicas are kept warm until the new one is confirmed. Keeping N-1 warm
  turns rollback into a traffic shift measured in seconds.
- **Forensics:** because every request logged its immutable version id, you can bound
  the blast radius exactly — "requests between 14:03 and 14:31 hit v7" — and attach that
  to the incident.

Rollback SLO worth stating out loud: *repoint + drain + serve previous version in under
a few minutes, with zero rebuild.* If your rollback requires re-running a training or
build pipeline, you do not have rollback; you have a second forward deploy.

> **Saying it out loud.** Reproducibility here means: give me a version id and
> I can reconstruct the exact serving behavior on a fresh host. Mechanically
> that's resolve the id to an immutable artifact reference, fetch the bytes
> and re-hash them against the recorded digest, load with the pinned engine
> and pinned runtime, apply the captured generation config rather than engine
> defaults, then re-run that version's eval suite and confirm the numbers
> match. That last step is the real test — if the evals come out different,
> something in the bundle wasn't actually pinned, and you've just found which
> layer you're missing. Rollback is the same machinery run backwards, and it's
> cheap only because the old version was never destroyed.

---

## A/B and Shadow of Versions

Aliases make multi-version serving natural because several pointers can coexist:

- **A/B (canary):** split live traffic across `@champion` and `@challenger`. Users see
  responses from both; you compare online metrics (latency, thumbs-up, task success) by
  the **version id stamped on each request**. Promote the winner by repointing
  `@champion`. (See the Canary Deployments chapter for traffic-splitting mechanics.)
- **Shadow (mirror):** send a **copy** of production traffic to `@shadow` (the candidate)
  but **do not return its output** to the user. You capture the candidate's responses
  and latency for offline comparison with zero user risk — ideal for validating an
  engine upgrade or a re-quantized version before it ever touches a user.

Both require the same discipline: the response/log record must carry the exact version
that produced it, or the comparison is meaningless. Shadow is the safest way to catch
tokenizer/engine drift *before* promotion, because you diff the candidate against prod
on identical inputs.

> **Saying it out loud.** A/B and shadow are answering different questions and
> I'd keep them separate. A/B, or canary, splits real traffic between champion
> and challenger, so users see both and you compare online metrics — latency,
> thumbs-up rate, task success — sliced by the version id stamped on each
> request. Shadow mirrors a copy of production traffic to the candidate but
> never returns its output, so you get real-input comparison at zero user
> risk. Shadow is what I'd reach for before an engine upgrade or a
> re-quantization, because those break in ways offline evals miss. Both are
> worthless without one discipline: every log record carries the exact version
> that produced it, or the comparison means nothing.

---

## Model Cards & Lineage

A **model card** is the human-readable documentation of a version: intended use, training
data, eval results, limitations, biases, and license (Mitchell et al., 2019, *Model
Cards for Model Reporting*). On the Hugging Face Hub the card is the repo's [`README.md`](README.md)
with a YAML metadata header; it renders on the model page and is machine-parsable.

**Lineage** is the machine-readable provenance graph: this version came from *this
training run*, on *this data snapshot*, from *this base-model revision*, built by *this
pipeline commit*. Registries capture lineage as run links (MLflow), source pipeline
references (SageMaker), or metadata edges (Vertex).

Why both matter for serving: when an incident, a compliance request, or a
"which-data-did-this-see" question lands, the card answers *what and why*, and lineage
answers *from what*. A version with neither is unauditable — you cannot prove what it is
or where it came from, which is exactly the state the opening anecdote describes.

> **Saying it out loud.** A model card is the human-readable half — intended
> use, training data, evals, limitations, license — and lineage is the
> machine-readable half: this version came from this training run, on this
> data snapshot, from this base-model revision. You need both, and you need
> them for a boring reason: when an incident or a compliance request lands,
> the card answers what and why, and lineage answers from what. A version with
> neither is unauditable — you can't prove what it is or where it came from,
> which is exactly the situation this chapter opened with.

---

## Production Case Studies & War Stories

The failure modes in this chapter are not hypothetical. Here are three, in increasing
order of "the version number was technically correct the whole time."

### War story 1 — a tokenizer auto-upgrade silently shifted a special token id

In April 2026, users of Kimi K2.5's multimodal inference in vLLM hit a hard crash:
`AssertionError: Failed to apply prompt replacement for mm_items['vision_chunk'][0]`
(vLLM issue #39261). Nothing about the *registered model version* had changed. The root
cause: the model's `config.json` had been written assuming a slow, TikToken-style
tokenizer, with a hardcoded `media_placeholder_token_id = 163605`. When `transformers`
v5 loaded the same repo, it auto-converted the tokenizer to its faster backend — and
that backend compacts gaps in the special-token id space, which shifted the actual
`<|media_pad|>` token from id 163605 to id 163602. Token 163605 now decoded to
`[UNK]`. vLLM went looking for a token that, at runtime, no longer existed at that id.

The model weights were unchanged. The registered version number was unchanged. What
changed was a *loader-level auto-migration* one layer below the version the team
thought they had pinned. The lesson generalizes directly from this chapter's bundle
table: pinning "the tokenizer files" is not the same as pinning "tokenizer *behavior*."
A startup self-check that encodes a fixed probe string (including every special token)
and compares the resulting ids against a recorded golden fingerprint would have caught
this in a health check, not in production traffic.

> **Saying it out loud.** This is my favorite example because nothing anyone
> would call "the model" changed. A newer transformers auto-converted the
> tokenizer to its fast backend, that backend compacts gaps in the
> special-token id space, and a token the config had hardcoded by numeric id
> moved by three. The engine went looking for a token that no longer existed
> at that id and hard-crashed. Weights unchanged, registered version
> unchanged; a loader-level auto-migration one layer below what the team
> thought they'd pinned. The generalizable lesson: pinning the tokenizer files
> is not the same as pinning tokenizer behavior — so encode a fixed probe
> string, record the resulting token ids as a golden fingerprint, and check it
> at startup.

### War story 2 — an unrelated dependency bump broke chat formatting

In December 2025, a team deploying DeepSeek-V3.2 on vLLM 0.11.2 hit a `ValueError` at
*request* time, not at startup: `"As of transformers v4.44, default chat template is no
longer allowed, so you must provide a chat template if the tokenizer does not define
one"` (vLLM issue #29849). The server had started cleanly; the model loaded; the crash
only surfaced the moment a real chat-formatted request arrived, because the model had
been relying on an *implicit default* chat template that a `transformers` version bump
in the base image simply removed. Nothing in the registered model version changed —
the break was purely in an unpinned runtime dependency, exactly the "Runtime deps" row
this chapter's versioning-together table calls out. The practical guard this incident
argues for: a deploy-time canary that sends one real, chat-formatted request through
the new pod *before* it takes production traffic, so a broken template fails the
rollout gate instead of a live user's request.

> **Saying it out loud.** The nasty part of this one is the timing: the server
> started cleanly and the model loaded fine, and the failure only appeared
> when the first real chat-formatted request arrived. A transformers bump in
> the base image removed an implicit default chat template the model had been
> relying on. Nothing in the registered version changed — it was purely an
> unpinned runtime dependency, which is why the versioning-together table has
> a row for runtime deps. The guard it argues for is a deploy-time canary:
> push one real, fully formatted request through the new pod before it takes
> production traffic, so a broken template fails the rollout gate instead of a
> live user's request.

### War story 3 — the mutable `latest` tag (a composite, illustrative pattern)

This one is presented deliberately as a *pattern*, not a single sourced incident,
because it is the single most commonly reported shape of this failure across teams and
is worth naming precisely: a serving config points at `chatbot-llm:latest` (or an S3
prefix like `s3://models/chatbot-llm/latest/`, or an HF `main` branch with no
`revision=`). A teammate — in a different repo, a different team, a different
timezone — pushes a retrain, a quick fix, or even just a [`README.md`](README.md) metadata update
that happens to also touch the pointer. `latest` now resolves to different bytes. No
deploy fired. No CI ran. The next pod restart (an autoscale event, a node drain, a
routine rolling update — not even the same team's action) picks up the new bytes and
starts serving different behavior under a config line that has not changed in months.
The only way to notice is behavioral: a metric drifts, a user complains, or an eval
canary re-run against a fixed prompt set produces a different fingerprint than
yesterday. This is precisely why this chapter insists a served pointer be resolved
**once, at load, to an immutable id that gets logged** — with that discipline, this
class of incident becomes "check the resolved version id in the logs," not "trace
through everyone's recent commits."

**Common thread across all three:** in every case the thing a human would call "the
model" — the name, the registered version, the config line — stayed put. The behavior
changed because *something the bundle table lists as a separate row* moved
underneath it. That is the whole argument for versioning the bundle, not the weights.

> **Saying it out loud.** This one is a pattern rather than a single incident,
> and it's the most common shape of the whole failure class. Your config
> points at latest, or an S3 latest prefix, or the HF main branch with no
> revision pinned. Someone in a different team and a different timezone pushes
> something — a retrain, or honestly just a README update that touches the
> pointer — and now latest resolves to different bytes. No deploy fired, no CI
> ran. The next pod restart, which could be a routine autoscale event nobody
> initiated, starts serving different behavior under a config line that hasn't
> changed in months. With a resolved-once-and-logged version id, this becomes
> "check the version in the logs"; without it, it's archaeology across
> everyone's commit history.

---

## Failure Modes & Pitfalls

- **Mutable `latest`/`main` in prod.** Serving `chatbot-llm:latest` or HF `main` means
  behavior can change under you with no deploy and no diff. Pin a digest, alias→version,
  or commit hash. `latest` is for dev laptops, never production. (War story 3, above.)
- **Weights not content-addressed.** If a bucket write can overwrite version 5's bytes
  and keep the id, your "immutable" version is a lie. Key by hash and enable object
  immutability/versioning.
- **Tokenizer/engine drift.** Weights pinned, but the tokenizer or chat template comes
  from a different revision, or the base image bumped `transformers`/vLLM. Same weights,
  different tokens, different outputs. Version the *whole bundle* and record the engine
  version; catch it with shadow. This is not hypothetical — see War stories 1 and 2
  above, both real, dated incidents where the registered version never changed at all.
- **Prompt/serving-code drift.** The prompt template or post-processing lives in app
  code that deploys on its own cadence, decoupled from the model version. Pin the
  serving-code SHA into the version — or, for agentic systems, version the prompt itself
  in a prompt registry and gate it the same way you gate the model.
- **No link between served version and eval results.** Metrics logged against a run but
  not the *version id*, or evals run on a different bundle than what ships. Promotion
  gates then guard nothing. Attach eval numbers to the version, run them on the exact
  artifact.
- **Alias re-resolved per request.** Resolving `@champion` on every request makes a
  promotion split traffic mid-flight and makes logs ambiguous. Resolve once at load,
  pin, and reload on change.
- **Semantic version reused for different bytes.** Re-tagging `v3` after a hotfix
  silently changes identity. Versions are append-only; new bytes get a new number.
- **Rollback that rebuilds.** If reverting requires re-running the build/train pipeline,
  incidents last hours. Keep N-1 registered and warm-able.
- **Config drift between registry and serving.** The registry says v7 but the pod
  mounted a stale cached artifact. Verify the loaded hash against the registry at
  startup and fail closed on mismatch.
- **A gate that fails open on missing data.** An eval-gated promotion pipeline that
  treats "no eval tags recorded yet" as "pass" will eventually promote something nobody
  evaluated. Fail closed: no eval on the version means no promotion.

> **Saying it out loud.** If I'm naming the pitfalls that actually cause
> incidents: mutable pointers in production, bytes that aren't
> content-addressed so a bucket write can overwrite a version in place,
> tokenizer or engine drift where the weights are pinned but the layer
> underneath isn't, and eval results attached to a run instead of to the
> version id — which means your promotion gate is guarding nothing. Two more
> that are less obvious. Re-resolving an alias per request instead of once at
> load, which splits traffic mid-promotion and makes logs ambiguous. And a
> gate that fails open on missing eval data, which will eventually promote
> something nobody evaluated. Rollback that requires a rebuild belongs on the
> list too: that turns a five-minute incident into a multi-hour one.

---

## Interview Mastery

This section is the one to over-prepare. Interviewers use model versioning as a proxy
for "does this person actually understand production ML systems, or just training
runs" — the questions below go from fundamentals to a full system-design prompt.

1. **"What exactly is a model version to you?"** — Expect the full bundle: weights +
   config + tokenizer + generation config + serving code + engine version + runtime,
   all pinned. Naming only the weights is a red flag.
2. **"How does the server know which version is prod, right now?"** — Alias/approval
   resolved once at load, pinned, and stamped on every request; not `latest`, not
   per-request re-resolution.
3. **"Walk me through promoting dev → staging → prod."** — Immutable register, evals
   linked to the version, enforced gates, approval recorded, pointer move — never a
   rebuild.
4. **"A regression is in prod. Roll it back."** — Repoint alias / redeploy prior ARN /
   pin prior commit; drain + reload; previous version still registered and warm;
   target minutes, no rebuild. Bonus points for mentioning a `previous_<alias>`-style
   snapshot written *at promotion time*, so rollback is a lookup, not a memory test.
5. **"How do you guarantee the model is byte-for-byte what you think?"** — Content hash /
   commit digest, re-verified on load; object-store immutability; fail closed on
   mismatch.
6. **"Same weights but outputs changed — how did that happen and how do you prevent it?"**
   — Tokenizer/engine/config/prompt drift; version the whole bundle, record engine
   version, catch with shadow. Cite a concrete mechanism if you can — e.g. a
   tokenizer-backend auto-conversion silently remapping special-token ids (War story 1).
7. **"How do you compare two versions safely in production?"** — Shadow for zero-risk
   diffing, A/B/canary for online metrics, with the version id stamped on every record.
8. **"How would you audit which model answered a given request three weeks ago?"** —
   Immutable version id logged per request + model card + lineage back to run/data.
9. **"Explain, in about 60 seconds, why 'the same model' can silently change in
   production."** — A strong answer names *specific* mechanisms, fast, without
   rambling: (a) a mutable pointer — `latest`, `main`, an S3 `latest/` prefix — gets
   repointed by someone else's unrelated push; (b) a base-image/runtime-dependency bump
   (`transformers`, vLLM, CUDA) changes tokenizer behavior, chat-template defaults, or
   sampling kernels without anyone touching the model config; (c) a tokenizer loader
   auto-migrates formats (slow→fast) and silently remaps special-token ids; (d) a
   quantization or LoRA-merge step gets re-run with a different calibration set under
   the same artifact name; (e) prompt template or post-processing code deploys on its
   own cadence, decoupled from the model version. The unifying point to land on: *the
   human-facing name is not the identity* — only a content hash of the full bundle is,
   which is why every mechanism above is invisible until you check bytes, not names.
10. **System design: "Design the model registry + promotion workflow for a company
    running 5 models in prod."** A strong answer covers, roughly in this order:
    - **Requirements first:** how many teams own models, what's the rollback SLO, is
      there a compliance/audit requirement, do models share infra (GPUs, base images)?
    - **One registry, namespaced per model** — not five bespoke systems. MLflow (or
      SageMaker/Vertex if already on that cloud) with a Postgres metadata store and
      S3/GCS artifact backend, content-addressed by hash.
    - **Aliases per model, not global:** `chatbot-llm@champion`, `ranker@champion`,
      etc. — each model gets its own `@champion`/`@challenger`/`@shadow`, so a
      promotion on one model can't accidentally touch another.
    - **Promotion pipeline as code, shared across all 5 models** — one CI job template
      parameterized by model name, so gates (eval thresholds, approval requirement)
      are consistent and reviewable, not five diverging tribal processes.
    - **Serving layer:** each model's servers resolve their own alias once at startup,
      log the resolved version id on every request, and expose a `/version` endpoint
      for on-call sanity checks.
    - **Rollback SLO:** keep N-1 warm per model (accept the GPU-hours cost for the 5
      models that matter) so rollback is a pointer move + drain, not a cold rebuild.
    - **Monitoring tied to version id:** per-model dashboards sliced by the version
      stamped in logs, so a regression alert names a version, not just "the service."
    - **Ownership:** one on-call rotation per model team, but a single shared registry
      pattern and runbook, so a new hire only has to learn the pattern once.

    Sketch:

    ```
        ┌─────────────────────────────────────────────────────────────┐
        │                     Model Registry (MLflow)                  │
        │   metadata: Postgres     artifacts: S3 (content-addressed)   │
        │                                                                │
        │   chatbot-llm   @champion→v12  @challenger→v13  @shadow→v14   │
        │   ranker        @champion→v4   @challenger→v5                  │
        │   summarizer    @champion→v9                                   │
        │   classifier    @champion→v2   @challenger→v3                  │
        │   embedder      @champion→v6                                   │
        └───────────────┬────────────────────────────────────────────────┘
                        │ resolve alias ONCE at load; log version_id/req
            ┌───────────┼───────────┬───────────┬───────────┐
            ▼           ▼           ▼           ▼           ▼
        chatbot pods  ranker pods  summ. pods  class. pods  embed pods
            │
            └── shared CI promotion pipeline (eval gate → alias move),
                one template, parameterized by model name
    ```

11. **"How would you version an agentic system where the prompt, tools, and engine all
    change independently of the weights?"** — Version each independently (a prompt
    registry with its own aliases, a hash over the tool/function schema set, the engine
    version pinned per-image), then define an explicit **bundle manifest** that
    references all of them and gate/promote/roll back *the manifest* as one unit — see
    the agentic-bundle section above. The failure to avoid: repointing the model's
    alias and the prompt's alias in two separate, uncoordinated steps.
12. **"What's the difference between a semantic version and a content hash, and why do
    you need both?"** — Semver/registry integer is human-ordered and communicates
    intent ("newer than v2") but doesn't guarantee uniqueness of bytes; a content hash
    guarantees identity but is unordered and meaningless to a human. Use an
    auto-incrementing registry version for people, and record the content hash as
    immutable metadata on it — never let a version number get reused for different
    bytes.
13. **"When would you choose OCI artifacts / a container registry over something like
    MLflow for models?"** — When you want to reuse existing container infra (registry,
    signing via sigstore, Kubernetes volume sources, multi-arch manifests) rather than
    stand up a separate model-specific system — the CNCF ModelPack effort and
    Harbor-as-model-registry deployments are exactly this bet. Trade-off: you give up
    some ML-specific metadata (runs, eval linkage, lineage graphs) that a purpose-built
    registry like MLflow gives you natively, unless you layer it back on with tags.
14. **"How do you detect version drift automatically, before a human notices bad
    outputs?"** — Startup self-check: re-hash the loaded artifact against the recorded
    digest and fail closed on mismatch; a fixed probe-string tokenizer fingerprint
    checked against a golden value (this would have caught War story 1 at boot, not in
    live traffic); a deploy-time canary request through the full serving stack before
    taking traffic (this would have caught War story 2 at rollout, not at the first
    real user request); continuous shadow evaluation comparing `@shadow` against
    `@champion` on identical inputs.
15. **"What's the cost/latency tradeoff of keeping N-1 warm for instant rollback, and
    how do you decide how many versions to keep warm?"** — Warm N-1 costs roughly
    double the steady-state GPU footprint for that model during any promotion window;
    justify it by rollback SLO (if "minutes" is the requirement, cold-start-from-object-
    storage for a 70B model won't hit it) and by blast radius (a model with 5 dependent
    downstream services justifies the spend more than an internal experiment). Most
    teams keep exactly N-1 warm and rely on object storage + a fast loader for anything
    older, since rollback more than one hop back is rare and can tolerate a slower path.
16. **"Stages vs aliases — when, if ever, would you still want classic stages?"** —
    Stages are simpler when you genuinely have one linear lifecycle and only ever need
    one "current production" slot with no concurrent challenger/shadow traffic; they
    get in the way the moment you want two things pointing at production-adjacent
    versions simultaneously (canary + shadow + champion), which is the normal case for
    LLM serving — hence MLflow's move to aliases.

> **Saying it out loud.** For a design prompt like five models in production,
> the shape of a good answer is: one registry, namespaced per model, not five
> bespoke systems — metadata in Postgres, artifacts content-addressed in
> object storage. Each model gets its own aliases, so promoting the ranker
> can't accidentally touch the chatbot. One promotion pipeline as code,
> parameterized by model name, so the eval thresholds and approval
> requirements are consistent and reviewable rather than five diverging tribal
> processes. Each server resolves its own alias once at startup, logs the
> resolved version id on every request, and exposes a version endpoint for
> on-call. And I'd state the rollback SLO explicitly and pay for it — keeping
> N-1 warm roughly doubles that model's steady-state GPU footprint during a
> promotion window, and that's the honest tradeoff: you're buying
> minutes-not-hours rollback with GPU-hours.

### Red flags vs. green flags

| Signal | Red flag | Green flag |
|---|---|---|
| Naming a version | "the weights" only | Full bundle: weights + tokenizer + config + engine + serving code + runtime deps |
| Pointer semantics in prod | `latest` / `main` / an S3 `latest/` prefix | A pinned alias or commit, resolved once at load and logged |
| Promotion mechanism | Manual file copy, ad hoc rebuild | Pointer move over an already-registered, already-hashed version |
| Rollback time | "We'd redeploy / retrain" | Alias repoint + rolling restart; N-1 kept warm; minutes, not hours |
| Eval linkage | Metrics live in a spreadsheet or a run, untied to a version id | Eval results stored as tags/params on the exact served version |
| Drift detection | "We'd notice from user complaints" | Hash re-verification at load + tokenizer fingerprint + shadow diffing pre-promotion |
| Dependency pinning | "`requirements.txt`, roughly" | Lockfile or image digest pinned per version, engine version explicitly recorded |
| Auditability | "Check the deploy Slack channel" | Immutable version id on every request log + model card + lineage graph |
| Scaling to many models | Ad hoc process per team, reinvented each time | One registry pattern, namespaced aliases, one promotion pipeline as code |
| Agentic bundles | Prompt/tools versioned informally in app code | Prompt registry + tool-schema hash + engine version, gated as one manifest |

---

## Further Reading

- MLflow — Model Registry (aliases, tags, workflow): <https://mlflow.org/docs/latest/ml/model-registry/workflow/>
- MLflow — Registry concepts (stages deprecated in favor of aliases): <https://mlflow.org/docs/latest/model-registry/>
- MLflow — Load a registered model (`models:/name@alias`, `models:/name/version`): <https://mlflow.org/docs/latest/getting-started/registering-first-model/step3-load-model/>
- MLflow — Prompt Registry overview: <https://mlflow.org/docs/latest/genai/prompt-registry/>
- MLflow — Manage prompt lifecycles with aliases: <https://mlflow.org/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases/>
- MLflow — release notes (3.15.0, July 31, 2026, and history): <https://mlflow.org/releases/>
- Amazon SageMaker — Register a model & Model Package Groups: <https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html>
- Amazon SageMaker — Update model approval status: <https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-approve.html>
- Vertex AI — Model Registry introduction: <https://cloud.google.com/vertex-ai/docs/model-registry/introduction>
- Vertex AI — Model version aliases: <https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/model-registry/model-alias>
- Hugging Face — Sharing & the `revision` argument (pin a commit): <https://huggingface.co/docs/transformers/model_sharing>
- Hugging Face — Model Cards (format & metadata): <https://huggingface.co/docs/hub/en/model-cards>
- Hugging Face Forums — Purpose of `commit_hash` in `PreTrainedModel.from_pretrained`: <https://discuss.huggingface.co/t/purpose-of-commit-hash-in-pretrainedmodel-from-pretrained/174304>
- Hugging Face — Xet chunk-level deduplication specification: <https://huggingface.co/docs/hub/en/xet/deduplication>
- Hugging Face — "From Chunks to Blocks" (Xet storage engineering): <https://huggingface.co/blog/from-chunks-to-blocks>
- Baseten — Pinning ML model revisions for compatibility and security: <https://www.baseten.co/blog/pinning-ml-model-revisions-for-compatibility-and-security/>
- CNCF — How OCI Artifacts will drive future AI use cases (Aug 27, 2025): <https://www.cncf.io/blog/2025/08/27/how-oci-artifacts-will-drive-future-ai-use-cases/>
- CNCF ModelPack — open model packaging spec: <https://github.com/modelpack/model-spec>
- CNCF ModelPack — `modctl` reference CLI: <https://github.com/modelpack/modctl>
- VMware/Broadcom — Using Harbor as an AI Model Registry (Mar 2026): <https://blogs.vmware.com/cloud-foundation/2026/03/03/using-harbor-as-an-ai-model-registry/>
- vLLM issue #39261 — Kimi K2.5 tokenizer-backend token-id mismatch (Apr 2026): <https://github.com/vllm-project/vllm/issues/39261>
- vLLM issue #29849 — DeepSeek-V3.2 chat-template break from a `transformers` bump (Dec 2025): <https://github.com/vllm-project/vllm/issues/29849>
- Mitchell et al., 2019 — *Model Cards for Model Reporting*: <https://arxiv.org/abs/1810.03993>
- Docker — image digests vs tags (the mental model): <https://docs.docker.com/dhi/core-concepts/digests/>
