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
  more below).

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

---

## Fully Worked Example: MLflow Registry Workflow

This is a real, correct MLflow (3.x) workflow: log + register a transformers model with
its tokenizer, link eval metrics, use aliases as gates, promote to `@champion`, and load
by alias in the server. It also shows how the server resolves "current prod version."

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

---

## Registry Comparison

| Feature | MLflow Model Registry | SageMaker Model Registry | Vertex AI Model Registry | Hugging Face Hub |
|---|---|---|---|---|
| Version unit | Registered model + integer version | Model Package Group + Model Package | Model resource + version id | Git repo + commit (revision) |
| Pointer mechanism | **Aliases** + tags (stages deprecated) | **Approval status** (`Pending/Approved/Rejected`) | **Version aliases** (incl. `default`) | Branches / tags / commit SHA |
| Immutable id | Version number + logged artifact hash | Model Package ARN | Version id (immutable) | **Commit hash** (content-addressed via Git/LFS) |
| Gated promotion | Alias move + tag gates in CI | Approval status flip (EventBridge-triggerable) | Alias reassignment | PR/branch merge; manual convention |
| Artifact store | Pluggable (S3/GCS/Azure/local) | S3 (+ ECR image) | Google Cloud Storage | Git LFS on the Hub |
| Lineage/metadata | Runs, params, metrics, tags | Metrics, data lineage, source pipeline | Metadata, eval, dataset links | Model card (`README.md` + YAML) |
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

---

## Model Cards & Lineage

A **model card** is the human-readable documentation of a version: intended use, training
data, eval results, limitations, biases, and license (Mitchell et al., 2019, *Model
Cards for Model Reporting*). On the Hugging Face Hub the card is the repo's `README.md`
with a YAML metadata header; it renders on the model page and is machine-parsable.

**Lineage** is the machine-readable provenance graph: this version came from *this
training run*, on *this data snapshot*, from *this base-model revision*, built by *this
pipeline commit*. Registries capture lineage as run links (MLflow), source pipeline
references (SageMaker), or metadata edges (Vertex).

Why both matter for serving: when an incident, a compliance request, or a
"which-data-did-this-see" question lands, the card answers *what and why*, and lineage
answers *from what*. A version with neither is unauditable — you cannot prove what it is
or where it came from, which is exactly the state the opening anecdote describes.

---

## Failure Modes & Pitfalls

- **Mutable `latest`/`main` in prod.** Serving `chatbot-llm:latest` or HF `main` means
  behavior can change under you with no deploy and no diff. Pin a digest, alias→version,
  or commit hash. `latest` is for dev laptops, never production.
- **Weights not content-addressed.** If a bucket write can overwrite version 5's bytes
  and keep the id, your "immutable" version is a lie. Key by hash and enable object
  immutability/versioning.
- **Tokenizer/engine drift.** Weights pinned, but the tokenizer or chat template comes
  from a different revision, or the base image bumped `transformers`/vLLM. Same weights,
  different tokens, different outputs. Version the *whole bundle* and record the engine
  version; catch it with shadow.
- **Prompt/serving-code drift.** The prompt template or post-processing lives in app
  code that deploys on its own cadence, decoupled from the model version. Pin the
  serving-code SHA into the version.
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

---

## Production Checklist — What an Interviewer Probes

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
   pin prior commit; drain + reload; previous version still registered and warm; target
   minutes, no rebuild.
5. **"How do you guarantee the model is byte-for-byte what you think?"** — Content hash /
   commit digest, re-verified on load; object-store immutability; fail closed on
   mismatch.
6. **"Same weights but outputs changed — how did that happen and how do you prevent it?"**
   — Tokenizer/engine/config/prompt drift; version the whole bundle, record engine
   version, catch with shadow.
7. **"How do you compare two versions safely in production?"** — Shadow for zero-risk
   diffing, A/B/canary for online metrics, with the version id stamped on every record.
8. **"How would you audit which model answered a given request three weeks ago?"** —
   Immutable version id logged per request + model card + lineage back to run/data.

---

## Further Reading

- MLflow — Model Registry (aliases, tags, workflow): <https://mlflow.org/docs/latest/ml/model-registry/workflow/>
- MLflow — Registry concepts (stages deprecated in favor of aliases): <https://mlflow.org/docs/latest/model-registry/>
- MLflow — Load a registered model (`models:/name@alias`, `models:/name/version`): <https://mlflow.org/docs/latest/getting-started/registering-first-model/step3-load-model/>
- Amazon SageMaker — Register a model & Model Package Groups: <https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html>
- Amazon SageMaker — Update model approval status: <https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-approve.html>
- Vertex AI — Model Registry introduction: <https://cloud.google.com/vertex-ai/docs/model-registry/introduction>
- Vertex AI — Model version aliases: <https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/model-registry/model-alias>
- Hugging Face — Sharing & the `revision` argument (pin a commit): <https://huggingface.co/docs/transformers/model_sharing>
- Hugging Face — Model Cards (format & metadata): <https://huggingface.co/docs/hub/en/model-cards>
- Mitchell et al., 2019 — *Model Cards for Model Reporting*: <https://arxiv.org/abs/1810.03993>
- Docker — image digests vs tags (the mental model): <https://docs.docker.com/dhi/core-concepts/digests/>
