# Docker for GPU LLM Serving

> Containerizing an LLM inference service so it runs the same on your laptop, in CI, and on a rented H100 — without a 40 GB image, a leaked token, or a `CUDA driver version is insufficient` at 3 a.m.

## Why this matters

An LLM server is not a normal web app. It links against CUDA, cuDNN, NCCL, and a specific PyTorch build; it needs a physical GPU exposed into the container; and it depends on multi-gigabyte model weights that you do *not* want to redownload on every restart. Get the containerization wrong and you hit one of a dozen classic failures: the image balloons to tens of gigabytes, the container can't see the GPU, the CUDA version mismatches the host driver, your Hugging Face token ends up baked into a layer, or the server runs as root with no healthcheck and the orchestrator can't tell it's wedged.

Containers are also the unit that Kubernetes, Nomad, ECS, and every autoscaler schedule. A well-built image is the foundation for everything in later chapters (K8s, canary, autoscaling). This chapter is about getting that foundation right — and about what "right" means as of 2026, since this corner of the ecosystem (NVIDIA Container Toolkit, official framework images, and how weights get distributed) has moved substantially in the last two years.

The intuition first, then the exact mechanisms.

---

## Core intuition

Three ideas carry most of the weight.

**1. The container shares the host's GPU driver, not its own.** You never install an NVIDIA *driver* inside the image. The driver is a kernel module and lives on the host. The container ships *userspace* CUDA libraries. At `docker run` time, the NVIDIA Container Toolkit injects the host driver's device files and libraries into the container. So the contract is: **host driver must be new enough for the container's CUDA userspace.** This is the single most important mental model in GPU containers.

**2. Model weights are data, not code.** Code changes daily and is tiny; a 14 GB weights file changes rarely and is huge. Baking weights into an image layer couples the two lifecycles badly. The default for production is to keep weights *out* of the image and bring them in as a mounted volume, a startup download into a persistent cache, or — increasingly, as of 2025–2026 — a separate OCI artifact pulled alongside the image (see "The 2025–2026 landscape" below).

**3. Build image ≠ runtime image.** The tools you need to *compile* CUDA kernels (nvcc, headers, build-essential) are hundreds of MB you never need to *run* the server. Multi-stage builds let you compile in a fat stage and copy only the artifacts into a lean runtime stage.

Hold these three and the rest is detail.

---

## Mechanism 1 — GPU base images and the CUDA/driver contract

### The `nvidia/cuda` image family

NVIDIA publishes `nvidia/cuda` images on Docker Hub and NGC. Tags follow the pattern:

```
nvidia/cuda:<cuda_version>-<flavor>-<os>
# e.g.
nvidia/cuda:12.4.1-runtime-ubuntu22.04
nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
nvidia/cuda:12.4.1-devel-ubuntu22.04
```

There are three flavors, and picking the wrong one is a common source of bloat:

| Flavor    | Contains                                             | Size (approx) | Use it for                                  |
|-----------|------------------------------------------------------|---------------|---------------------------------------------|
| `base`    | CUDA runtime libs minimum                            | ~200 MB       | Rare; you usually need more                 |
| `runtime` | CUDA runtime + math libs (cuBLAS), optionally cuDNN  | ~2–3 GB       | **Final runtime stage** of an inference server |
| `devel`   | Everything in runtime **+ nvcc, headers, static libs** | ~5–7 GB     | **Build stage** when you compile kernels    |

Rule of thumb: build in `devel`, ship on `runtime`. If your framework ships prebuilt wheels (most do), you may not even need `devel`.

### Driver/runtime compatibility

The host driver exposes a maximum supported CUDA version. CUDA has *forward compatibility* within a major version and *minor-version compatibility* so that, e.g., a driver supporting CUDA 12.2 can generally run CUDA 12.4 userspace on data-center GPUs via the compat package — but do not rely on this casually. The safe posture:

- Check the host: `nvidia-smi` prints `Driver Version` and `CUDA Version` (the max CUDA the driver supports).
- Pick a container CUDA version **≤** that, or confirm forward-compat coverage.
- Pin the container CUDA minor version explicitly (`12.4.1`, not `12`).

The failure you're avoiding looks like:

```
CUDA driver version is insufficient for CUDA runtime version
```

That means the container's CUDA userspace is newer than the host driver can serve. Fix by upgrading the host driver or downgrading the image's CUDA version — you cannot fix it inside the image. (See the "driver skew across a mixed GPU fleet" case study later in this chapter — this exact error is what took down one team's rollout.)

### GPU partitioning: MIG and time-slicing

Two GPUs is not always the right unit of allocation — sometimes you want to run several smaller inference workloads on one physical GPU. NVIDIA data-center GPUs (A100/H100-class) support two partitioning mechanisms, and interviewers like to check you know the difference:

- **MIG (Multi-Instance GPU)** — hardware-level partitioning. A single A100/H100 can be split into up to 7 fully isolated instances, each with its own dedicated slice of SMs, memory, and memory bandwidth, exposed to the container runtime as a distinct device. Because the isolation is in hardware, one MIG instance's workload cannot starve or interfere with another's — the strongest isolation option, but the partition sizes are fixed at profile granularity and set outside the container (via `nvidia-smi mig -cgi ...` on the host) before containers ever start.
- **Time-slicing** — software-level sharing. Multiple containers share the same full GPU, and the driver time-multiplexes compute across them (similar in spirit to CPU time-sharing). No memory isolation: any container can, in principle, allocate all the GPU's VRAM and starve its neighbors. Simpler to set up (no MIG profile management) but a weaker isolation guarantee — appropriate for dev/test or trusted-tenant workloads, not for hard multi-tenant isolation.

For a single dedicated LLM server per GPU (the common case in this chapter's examples), neither matters — you're using the whole device. They become relevant once you're packing multiple smaller models or replicas onto shared GPUs, which is a natural follow-up question after "how do you containerize a model server" in a systems-design interview.

### The CUDA forward-compatibility package

The "pin container CUDA ≤ host driver" rule earlier in this section has one documented escape hatch worth knowing by name: NVIDIA ships a **CUDA forward-compatibility package** for data-center GPUs that lets a container built against a newer CUDA toolkit run on an older driver than would normally be required, by shipping a compatible driver shim inside the container image itself. It is intentionally scoped — it applies to data-center GPU driver branches, not consumer GPUs, and it does not make every combination of "any CUDA version on any driver" work. Treat it as a documented exception for a specific, narrow upgrade-sequencing problem (e.g., you need to ship a newer CUDA-versioned image before every node's driver has been upgraded yet), not as a reason to stop tracking the driver/CUDA contract explicitly.

---

## Mechanism 2 — The NVIDIA Container Toolkit and `--gpus`

A plain `docker run` gives the container no GPU. Two pieces make it work.

### The toolkit

The **NVIDIA Container Toolkit** is a set of host packages (`nvidia-container-toolkit`, the `nvidia-ctk` CLI, and a runtime shim) that automatically configure a container to use NVIDIA GPUs by mounting the driver libraries and device nodes at startup. You install it on the host, once:

```bash
# 1. Add NVIDIA's apt repo (see official install guide for the current key/URL)
sudo apt-get install -y nvidia-container-toolkit

# 2. Wire it into the Docker daemon (writes the "nvidia" runtime into
#    /etc/docker/daemon.json)
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Prerequisite: the NVIDIA **driver** is already installed on the host. You do **not** install the CUDA toolkit on the host — only the driver.

Verify the whole chain end-to-end:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If that prints your GPU table, host driver + toolkit + runtime are all good. This is the first thing to run when a GPU container "can't see the GPU."

### `--gpus`

Once the toolkit is installed, `--gpus` selects which GPUs to expose:

```bash
--gpus all                 # all GPUs
--gpus '"device=0,1"'      # only GPU 0 and 1 (note the nested quoting)
--gpus 2                   # any 2 GPUs
```

Under the hood this sets `NVIDIA_VISIBLE_DEVICES` and triggers the toolkit's injection hook. Older stacks used `--runtime=nvidia` plus that env var directly; `--gpus` is the modern, preferred flag. Some tools (vLLM's docs) still show `--runtime nvidia` — it's equivalent when the runtime is registered.

> **Forward pointer — CDI.** The legacy hook-based injection above is being superseded by the **Container Device Interface (CDI)**, a vendor-neutral, Kubernetes/Podman/Docker-portable way to describe "how to expose device X into a container." The Container Toolkit has generated CDI specs since v1.14 (2024) and it's now the recommended path for Podman and for rootless setups. Full detail, with commands, is in "The 2025–2026 landscape" below.

---

## Mechanism 3 — Handling large model weights

This is where architecture decisions bite hardest. You have three classic options, plus a fourth that emerged in 2025.

### Option A — Bake weights into the image

Copy the weights in during build (`COPY ./model /model`). The image is fully self-contained: pull it and run, no network, no external volume.

- Pro: hermetic, reproducible, air-gap friendly, one artifact to sign/scan.
- Con: the image is now 15–150 GB. Every push/pull moves all of it. Layer caching is useless once weights change. Registry storage costs balloon. Build context upload is slow.
- Verdict: reserve for **small models**, **air-gapped/regulated** deployments, or when the exact weights are part of your release contract.

### Option B — Mount weights as a volume

Keep weights on the host / network storage and bind-mount at runtime (`-v /data/models:/models`). This is what the vLLM and TGI official images assume.

- Pro: small image, weights shared across containers and versions, swap models without rebuilding, fast cold builds.
- Con: image is no longer self-contained; you must provision and pre-populate the volume; in K8s you need a `PersistentVolume` / `hostPath` / CSI mount.
- Verdict: the **default for most production** on a fixed node pool or shared filesystem.

### Option C — Download at startup into a cache

The container downloads weights from Hugging Face (or S3/GCS) on first boot into a persistent cache directory, then reuses the cache on restart.

- Pro: smallest image, model chosen by env var, trivially swappable.
- Con: cold start pays a multi-GB download; needs network egress + a token for gated models; if the cache volume isn't persistent you redownload every restart (a classic and expensive bug); registry outages ≠ HF outages now both matter.
- Verdict: great for **dev, experimentation, and autoscaling where a warm cache volume** (or a pre-baked node image) hides the download.

> **Option D (emerging, 2025+) — weights as a separate OCI artifact.** Docker's Model Runner and Hugging Face's OCI push path distribute weights as an OCI *artifact* (not an image layer) alongside the serving image, so the model can be `docker pull`-ed, versioned, and content-addressed independently of the engine — while staying uncompressed on disk for fast `mmap` loading. This is a genuine fourth point on the spectrum: image-registry ergonomics with volume-mount-like decoupling. Details and citations are in the landscape section below; treat it as complementary to, not a replacement for, Options B/C in most production fleets as of this writing.

### The Hugging Face cache — make it persist

The `huggingface_hub` library caches downloads under `HF_HOME` (default `~/.cache/huggingface`; the hub cache is `\(HF\_HOME\)/hub`). The env vars that matter:

- `HF_HOME` — root of all HF caches. Set this and everything follows.
- `HF_HUB_CACHE` (older: `HUGGINGFACE_HUB_CACHE`) — the model blob cache specifically.
- `HF_TOKEN` — auth for gated/private models.

The whole point of Options B/C is to mount a volume at the cache path so weights survive container restarts:

```bash
# vLLM: cache lives at /root/.cache/huggingface inside the image
-v ~/.cache/huggingface:/root/.cache/huggingface

# TGI: the official image sets HUGGINGFACE_HUB_CACHE=/data, so mount /data
-v $PWD/data:/data
```

Miss this mount and every `docker run` redownloads the model — slow, costly, and rate-limit-prone.

### Quantization formats and their effect on image and weight size

Weight size is not a fixed property of a model — the quantization format you choose shifts it by 2–4x, which feeds directly back into the bake/mount/download decision above. A 70B-parameter model is roughly 140 GB in fp16, roughly 70 GB in int8, and commonly 35–40 GB in 4-bit formats (GPTQ, AWQ, or GGUF's `Q4_K_M`-style schemes). This matters for containerization in three concrete ways:

- **Bake-into-image (Option A) becomes far more viable** for a 4-bit-quantized model than for the fp16 original — a 35 GB image is still large, but it is a very different operational proposition than a 140 GB one, and may cross the threshold into "acceptable for our registry and node bandwidth."
- **The serving engine constrains the format.** vLLM, TGI, and TensorRT-LLM each support a different subset of quantization schemes with different kernel-level performance, so "which quantization format" and "which base image" are coupled decisions — you can't pick a format your engine's image doesn't have kernels for and expect the speed benefit to materialize.
- **Quantized weights still deserve the same weights-vs-code separation** as the fp16 case — the file is smaller, but it is still large, slow-changing data, and belongs on a mounted volume or content-addressed artifact rather than baked into a layer that gets rebuilt every time the application code changes, for the same CI/registry-cost reasons covered in the war stories below.


---

## Mechanism 4 — Multi-stage builds, layer caching, and `.dockerignore`

### Multi-stage

A multi-stage build uses multiple `FROM` statements. Early stages compile; the final stage copies only what's needed. For an LLM server that means: compile custom kernels / install a heavy build toolchain in a `devel` stage, then `COPY --from=builder` the installed environment into a slim `runtime` stage. The `devel` layers never ship.

### Targeting specific GPU architectures: `TORCH_CUDA_ARCH_LIST`

When a build stage compiles CUDA kernels from source (custom ops, or building a framework from source rather than installing a prebuilt wheel), it by default may compile for a broad list of GPU compute capabilities so the resulting wheel works everywhere — L4, L40S, A100, H100, and older architectures alike. That breadth costs real build time and binary size, and most of it is wasted if you know exactly which GPU SKU the image will run on. Setting `TORCH_CUDA_ARCH_LIST` (PyTorch's build-time env var) to only the architectures you actually deploy on narrows the compiled kernel set accordingly:

```dockerfile
# Example: building only for A100 (8.0) and H100 (9.0) compute capabilities,
# instead of the full default list PyTorch would otherwise target.
ENV TORCH_CUDA_ARCH_LIST="8.0 9.0"
RUN pip install --no-binary :all: torch
```

This is a build-stage-only concern — it has no effect once you're installing prebuilt wheels (the common, recommended path from Mechanism 4) — but it's worth knowing by name for the case where you *are* compiling from source, since it's a direct, fast answer to "why is our custom-kernel build so slow" or "why is this wheel so large" in a design discussion.


### Layer caching order

Docker caches each instruction as a layer and reuses it until an input changes; **everything after a changed layer is rebuilt.** So order from least-to-most volatile:

1. Base image
2. System packages (`apt-get`)
3. Dependency manifests (`requirements.txt` / `pyproject.toml`) and `pip install`
4. Application source code

Copy `requirements.txt` and install *before* copying your source. Then a one-line code change reuses the (slow) dependency layer instead of reinstalling PyTorch every build. Use BuildKit cache mounts for the pip/uv cache to speed rebuilds further:

```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

### Registry-backed BuildKit cache in CI

Layer caching (above) only helps within a single machine's local Docker cache. CI runners are frequently ephemeral or shared across many jobs, so the local cache that made your dependency layer fast on your laptop may not exist on the runner that picks up the next PR — which is exactly the trap that made War story 2's 40 GB image so painful in CI. BuildKit's registry cache backend fixes this by pushing/pulling the build cache itself as a registry artifact, so any runner can warm its cache from the last successful build regardless of which machine ran it:

```bash
docker buildx build \
  --cache-from type=registry,ref=myregistry.example.com/my-llm-server:buildcache \
  --cache-to   type=registry,ref=myregistry.example.com/my-llm-server:buildcache,mode=max \
  -t my-llm-server:ci .
```

`mode=max` caches every intermediate layer (not just the final stage), which matters for multi-stage Dockerfiles like the one in this chapter — otherwise the builder stage's expensive `pip install` layer isn't reusable across runners even though the final runtime-stage layers are. Combined with keeping weights out of the build context entirely (Mechanism 3), this is what keeps CI build times in the range War story 2's fix achieved (under two minutes) rather than the 25-plus minutes the baked-weights version suffered.


### `.dockerignore`

The build context is everything Docker uploads to the daemon before building. Without a `.dockerignore`, a stray `./models`, `.git`, `__pycache__`, or a 30 GB checkpoint gets shipped into the build — slow, and a vector for secrets and bloat. A minimal one:

```gitignore
.git
__pycache__/
*.pyc
*.pt
*.safetensors
models/
data/
.env
*.log
.venv/
```

This is also your first line of defense against `COPY . .` accidentally baking weights or a `.env` file into a layer.

---

## Mechanism 5 — Reproducibility, security, healthchecks, config

**Pin everything.** `python:3.11-slim` is a moving target. Prefer a digest for the base and pinned versions for packages:

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:<digest>
```

Pin your Python deps (lockfile or `==`), and tag your own images with an immutable version, never rely on `:latest` in production.

**Run as non-root.** By default containers run as UID 0. A container escape from root is worse than from an unprivileged user. Create a user and drop to it:

```dockerfile
RUN useradd --create-home --uid 10001 appuser
USER appuser
```

Note some GPU stacks and cache paths assume `/root`; if you run non-root, point `HF_HOME` at a directory that user can write.

**Keep secrets out of layers.** Never `ENV HF_TOKEN=hf_xxx` or `COPY .env` — both persist in the image history for anyone who pulls it. Pass secrets at **runtime** (`-e HF_TOKEN=...`, Docker/K8s secrets) or use BuildKit `--secret` mounts for build-time-only credentials.

**Add a HEALTHCHECK.** Orchestrators restart containers that fail their health probe. For an OpenAI-compatible server, probe the health/models endpoint:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1
```

The long `start-period` matters: model load can take minutes, and you don't want the container killed during warmup.

**Config via env, not baked files.** Model id, tensor-parallel size, port, and dtype should be env vars / CLI args so one image serves many configs.

---

## Mechanism 6 — Operating the container: logs, exec, monitoring, and shutdown

Building the image is half the job; running it in production for months is the other half. A few operational mechanics are specific to GPU containers and worth knowing cold.

### Logs and exec

`docker logs -f <container>` streams stdout/stderr — make sure your server logs there (not to a file inside the container that nobody mounts out) and emits structured (JSON) lines so they're parseable by whatever aggregator sits downstream. For live debugging, `docker exec -it <container> bash` drops you inside the running container, where the single most useful diagnostic is running `nvidia-smi` **from inside the container**: it reflects the same driver and devices as the host (since the toolkit injected them), and its process list shows exactly which processes inside *this* container are holding GPU memory — invaluable when a server reports OOM but you're not sure if it's fragmentation, a leaked previous request's KV cache, or another process entirely.

```bash
docker exec -it my-llm-server nvidia-smi        # GPU state as seen by *this* container
docker exec -it my-llm-server nvidia-smi pmon   # per-process GPU utilization inside the container
```

### Monitoring GPU utilization and health from outside the container

For host-level and fleet-level observability, don't rely on shelling into every container. Two standard options:

- **`nvidia-smi dmon`** on the host gives a live per-GPU utilization/memory/temperature stream — useful for a quick manual check, not for durable metrics.
- **DCGM (Data Center GPU Manager) exporter** — NVIDIA's `dcgm-exporter` runs as a sidecar or daemonset and exposes GPU utilization, memory, ECC errors, power, and temperature as Prometheus metrics, scoped per physical GPU and (with the right labels) attributable back to the container/pod using it. This is the standard way to get GPU metrics into the same dashboards and alerting as the rest of your fleet, rather than parsing `nvidia-smi` text output on a cron job.

The metric to alert on that's easy to miss: **GPU memory utilization**, not just compute utilization. A server can show low SM utilization (looks "idle") while sitting at 95% VRAM usage from an oversized KV cache or a memory leak across requests — the next allocation OOMs with no warning if you were only watching compute.

### Resource limits: what cgroups do and do not control for GPUs

The `deploy.resources.limits`/`reservations` block shown earlier in this chapter's compose file controls **CPU and memory** through the host's cgroups — Docker genuinely enforces those. It does **not** give you an equivalent hard limit on GPU compute or GPU memory the way it does for CPU shares or RAM. `--gpus` controls *which* GPUs a container can see, not *how much* of a shared GPU's compute or memory it's capped at using cgroups semantics. Practically:

- If you run one container per GPU (the common pattern for LLM serving, since a large model typically wants a whole device or several via tensor-parallel), this limitation doesn't bite — the container has the whole GPU and there's nothing else to contend with.
- If you deliberately share a GPU across containers (time-slicing, or just running two processes on one device without MIG), nothing at the container-runtime layer stops one from allocating all the VRAM and OOM-ing the other. NVIDIA's **MPS (Multi-Process Service)** can give more predictable compute sharing between cooperating processes, and MIG (above) gives hardware-enforced isolation — but plain Docker resource limits do not extend to the GPU the way they do to CPU/RAM. Don't assume `mem_limit` or `cpus` in your compose file constrains GPU memory; it doesn't.

### Graceful shutdown and restart policy

`docker stop` sends SIGTERM, waits a grace period (default 10s, configurable with `docker stop -t <seconds>` or `stop_grace_period` in compose), then SIGKILLs. For an LLM server mid-generation, 10 seconds is often not enough to drain in-flight streaming requests cleanly. Two adjustments matter:

- Increase the grace period (`stop_grace_period: 60s` in compose, or the orchestrator's equivalent — e.g., Kubernetes `terminationGracePeriodSeconds`) to comfortably exceed your longest expected generation, so in-flight requests finish rather than being cut off mid-stream.
- Handle SIGTERM in the application itself if the framework allows it — stop accepting new requests immediately (so the load balancer's health/readiness check can flip and stop routing new traffic) while letting in-flight requests complete within the grace window, rather than dropping everything the instant SIGTERM arrives.

Restart policy (`restart: unless-stopped` used in this chapter's compose file, or `on-failure` with a max retry count) determines what happens after a crash. For a GPU server, prefer a policy with a capped retry count or backoff over unconditional restart-forever: a driver mismatch or an OOM that recurs deterministically on every restart will otherwise crash-loop indefinitely, burning GPU-node scheduling slots and generating alert noise, when what you actually want after N failures is for the orchestrator to mark the replica unhealthy and stop retrying until a human looks at it.

### A debugging playbook for "the container won't come up"

When a GPU container fails on a node and the cause isn't obvious from the first log line, work through this order — cheapest checks first:

1. **Does the toolkit even see the GPU?** `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` on the same host. If this fails, the problem is host-level (toolkit/runtime/driver), not your image — stop debugging the application.
2. **Does the driver support the image's CUDA version?** Compare `nvidia-smi`'s reported `CUDA Version` against the image's CUDA tag. A driver-insufficient error here means fix the host or ship a lower-CUDA image; it is never fixed inside the container.
3. **Is `/dev/shm` large enough?** If the crash is a `Bus error` or an NCCL hang rather than a CUDA-init failure, suspect the 64 MB default shared-memory size before suspecting the model or the framework — check with `docker exec <c> df -h /dev/shm`.
4. **Is the cache volume actually mounted, and writable by the running user?** A silent full redownload on every restart, or a permission-denied on first write, both trace back to a missing or misowned mount at the `HF_HOME`/cache path.
5. **Is the healthcheck timing out during legitimate model load, or is the process actually wedged?** Check `docker logs -f` for load-progress output and compare elapsed time against `start-period`; don't assume a failed healthcheck means a hung process without checking whether it's simply still loading.
6. **Is this node's driver actually the one you tested against?** On a mixed-generation fleet, confirm the specific node's driver version rather than assuming fleet-wide uniformity — this is the failure mode from War story 1 below, and it is easy to lose an hour to before checking it directly.



---

## The 2025–2026 landscape

The mechanics above (driver contract, toolkit, multi-stage) haven't changed. What has changed since roughly 2024 is *how much of this you have to build yourself* versus *what ships as a hardened, official artifact* — and how seriously the industry now treats GPU images as a supply-chain surface. Four threads matter for a working engineer today.

### 1. The NVIDIA Container Toolkit has moved to CDI, and rootless is now a real option

The toolkit itself keeps shipping — v1.19.0 was released March 12, 2026 (see the release list at the project's GitHub, cited below). The bigger shift is architectural: the legacy `nvidia-container-runtime` "hook" approach (patch the OCI runtime spec at container-start time) is being superseded by the **Container Device Interface (CDI)**, a CNCF-adjacent, vendor-neutral spec for describing "how to inject device X into this container" that Docker, Podman, containerd, and Kubernetes (via kubelet device plugins) all understand the same way. NVIDIA's toolkit has generated CDI specs since v1.14 (2024), and NVIDIA's own docs now present CDI as the forward-looking path, especially for Podman and rootless setups.

Rootless GPU containers were awkward for years because `/dev/nvidia*` device nodes and the injection hook both assumed a privileged (rootful) daemon. CDI plus `nvidia-ctk` closes that gap. The rootless recipe (Podman, but the same idea applies to Docker's rootless mode) is:

```bash
# Generate a CDI spec into user space (not /etc/cdi, which needs root)
mkdir -p ~/.config/cdi
nvidia-ctk cdi generate --output=$HOME/.config/cdi/nvidia.yaml

# Sanity-check what devices the spec exposes
nvidia-ctk cdi list

# Run rootless, referencing the CDI device by its vendor.com/class=name identifier
podman --cdi-spec-dir=$HOME/.config/cdi run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Two caveats that bite in practice: (1) rootless containers still need the invoking user to have read/write on the `/dev/nvidia*` device files — usually via membership in the host's `video` (or `render`) group, passed through with `--group-add keep-groups`; and (2) on SELinux hosts (Fedora/RHEL-family) you may need `sudo setsebool -P container_use_devices on` or GPU access is silently denied. Rootless matters for the same reason it always has — a compromised process inside the container can't leverage root-equivalent privileges on the host — and it is now something you can actually put in a hardening checklist rather than an aspiration.

### 2. Official, hardened images are the default path for the mainstream servers

Five years ago most teams hand-rolled a Dockerfile like the one in this chapter. Today, for the two most common serving stacks, you usually start from the vendor's image and only write a custom Dockerfile when you have a genuinely custom serving path:

- **vLLM** publishes `vllm/vllm-openai` and documents the `docker run --runtime nvidia --gpus all ...` invocation directly (docs.vllm.ai/en/latest/deployment/docker/). The known trade-off: the image is large — community reports and vLLM's own GitHub issue tracker put a recent tag at roughly 12.6 GB (vLLM Forums thread "Current vLLM docker image size is 12.64Gb"; GitHub issue vllm-project/vllm#27154, "How to reduce the vllm image"). A vLLM maintainer's response to a from-source slimming attempt was blunt: *"Probably not as of yet, too many changes to manually work. Even then not guaranteed to work yet."* Translation for your own builds: don't assume you can easily out-slim the officially supported image without maintenance burden — budget registry storage and pull-time accordingly, or accept the size as the cost of using the maintained artifact.
- **Hugging Face TGI** ships `ghcr.io/huggingface/text-generation-inference` with GPU install docs at huggingface.co/docs/text-generation-inference/en/installation_nvidia, following the same "official image + mounted volume" pattern shown earlier in this chapter.
- **NVIDIA NIM** (developer.nvidia.com/nim) goes a step further: it packages *pre-optimized inference engines* — TensorRT-LLM, vLLM, SGLang builds tuned per GPU SKU — behind an OpenAI-compatible API, distributed as containers you self-host (via NGC) or consume as managed endpoints on Hugging Face. NIM trades some flexibility (you're consuming a curated engine build, often gated behind an NGC API key / NVIDIA AI Enterprise entitlement for production use) for meaningfully less Dockerfile authoring and tuning work; it's worth evaluating before writing a bespoke image for a well-known model architecture.

The interview-relevant takeaway: know when to reach for an official/vendor image (the common case now) versus when a custom multi-stage build is actually warranted (custom serving logic, a framework without an official image, or hard constraints on image size/content that the vendor image doesn't meet).

### 3. Image-size reduction has two live approaches: harden the base, or stop shipping weights as layers

Two independent techniques are gaining traction for the "these images are enormous" problem:

- **Hardened, minimal base images.** Chainguard publishes CUDA/PyTorch-family images built to a zero-known-CVE target, rebuilt daily, each with an SBOM and reproducible from signed build configs. Their own comparison (chainguard.dev/unchained/securing-the-foundations-of-ai-applications-with-chainguard-images) found the official PyTorch Docker Hub image carried 1 critical, 23 high, 1,189 medium, and 72 low CVEs (as measured July 24, 2024) against zero in their equivalent image at the same time, driven mostly by stripping unnecessary OS packages rather than by removing CUDA/PyTorch functionality. This is the same "runtime not devel, `--no-install-recommends`, clean apt lists" discipline from Mechanism 4/5 in this chapter, taken to its logical extreme by a vendor who productizes it.
- **Stop treating weights as image layers at all.** Docker's rationale for packaging AI models as **OCI artifacts** rather than image layers (docker.com/blog/oci-artifacts-for-ai-model-packaging) is worth understanding even if you don't adopt Docker Model Runner: model weight files are high-entropy, so compressing them into a tar layer (the normal image-layer behavior) buys negligible size reduction while costing real (de)compression time, and it prevents the inference engine from `mmap`-ing the file directly off disk. Docker's model-artifact manifest instead stores the weights as an *uncompressed* layer under model-specific media types (e.g., `application/vnd.docker.ai.gguf.v3`) alongside a JSON config carrying architecture/quantization/parameter-count metadata — decoupling "which weights" from "which engine" so you don't duplicate a 15 GB checkpoint across every framework's image. It is early days for this pattern in mainstream LLM-serving production, but the direction — weights as a distinct, content-addressed, uncompressed OCI artifact rather than baked into `devel`/`runtime` layers — is the one to watch, and it's a clean answer to "is there a better option than bake/mount/download?" if it comes up in an interview.

### 4. Supply-chain scanning is now expected on AI images specifically, not just on your app images

Because a CUDA + PyTorch + framework image drags in an unusually large OS + Python dependency graph, it accumulates CVEs faster than a typical microservice image — which is exactly why the Chainguard comparison above is so stark. The practical response teams are standardizing on in 2025–2026:

- **Docker Scout** (`docker scout cves <image>`, docs.docker.com/reference/cli/docker/scout/cves/) or **Trivy** as a CI gate — fail the build (or at least the "promote to prod" step) above a critical/high CVE threshold. Docker's own "Docker Hardened Images" program (docs.docker.com/dhi/how-to/scan/) packages this scanning workflow for a curated base-image catalog.
- **SBOM generation** (`docker sbom`, `syft`, or Chainguard's built-in SBOMs) attached to the image so a security team can answer "are we exposed to CVE-XXXX" without re-scanning every running container.
- **Image signing** (cosign / Sigstore) so the deployment pipeline can verify the image it's about to run on a GPU node actually came from your CI, not a tampered registry mirror.
- Practically, because GPU images are large and slow to scan/pull, teams increasingly scan **once at build/push time** and **verify signature + digest** at deploy time, rather than re-scanning on every node — the same "shift left" idea as regular container security, adjusted for the fact these images are 10–100x the size of a typical service image.

None of this replaces the fundamentals earlier in this chapter (multi-stage, `.dockerignore`, non-root, pinned digests). It's the layer on top: assume your CUDA/PyTorch base has a nontrivial CVE surface, and have an explicit story — vendor-hardened base, CI scanning gate, or both — for managing it, rather than discovering it during a customer security questionnaire.

### 5. Kubernetes-adjacent: the GPU Operator abstracts the host-side setup

Everything in Mechanism 2 (install the toolkit, configure the Docker/containerd runtime, verify with `nvidia-smi`) is manual host configuration. On a Kubernetes fleet, the **NVIDIA GPU Operator** packages that entire host-side setup — driver installation/management, container toolkit, device plugin, DCGM monitoring, and (as of the 25.10 release line) CDI support — as a set of components the cluster manages itself, so individual node bootstrap scripts stop being where GPU readiness lives. This chapter deliberately stays at the single-host `docker run`/compose level; the GPU Operator (and the rest of the Kubernetes-specific device-plugin and scheduling story) belongs to this guide's Kubernetes chapter, but it's worth knowing the name and roughly what it replaces before that chapter goes deep, since an interviewer moving from "containerize this" to "now put it in a cluster" is testing whether you know the boundary between the two layers.

---

## Build it in practice — extended

The Dockerfile below is the same pattern shown earlier in this chapter (build in `devel`, ship on `runtime`, non-root, healthcheck, exec-form entrypoint), reproduced here as the anchor for the compose and CI additions that follow.

```dockerfile
# syntax=docker/dockerfile:1.7

###############################################################################
# Stage 1: builder — has nvcc + build tools, compiles/installs the env.
###############################################################################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=0 \
    PYTHONDONTWRITEBYTECODE=1

# System build deps. Pin, clean apt lists to keep the layer lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv so we can copy the whole thing to the runtime stage.
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependency layer FIRST — cached across code changes.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install -r requirements.txt

###############################################################################
# Stage 2: runtime — slim CUDA runtime, no compilers, non-root, healthcheck.
###############################################################################
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # Persist weights here; mount a volume at this path (see docker run).
    HF_HOME=/models/hf

# Only the runtime OS deps: python + curl (for healthcheck). No build-essential.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the fully-built virtualenv from the builder stage — no nvcc ships.
COPY --from=builder /opt/venv /opt/venv

# Non-root user that can write the cache dir.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /models/hf && chown -R appuser:appuser /models
COPY --chown=appuser:appuser ./app /app
WORKDIR /app
USER appuser

EXPOSE 8000

# Config comes from env / CLI at runtime, not baked in.
ENV MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
    TENSOR_PARALLEL_SIZE=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Exec form so signals reach the process (clean shutdown).
ENTRYPOINT ["python3.11", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
# Build (BuildKit on for cache mounts + syntax directive)
DOCKER_BUILDKIT=1 docker build -t my-llm-server:1.0.0 .

# Run: expose GPUs, mount the HF cache volume, pass secrets at runtime.
docker run --rm \
  --gpus all \                                  # expose all GPUs (toolkit required)
  --ipc=host \                                  # shared mem for NCCL / TP; see note
  -p 8000:8000 \
  -v $HOME/.cache/hf:/models/hf \               # persist weights across restarts
  -e HF_TOKEN=$HF_TOKEN \                        # gated-model auth, NOT baked in
  my-llm-server:1.0.0 \
  --model meta-llama/Llama-3.1-8B-Instruct \    # config via CLI
  --tensor-parallel-size 1
```

**Why `--ipc=host`?** vLLM (and PyTorch tensor-parallel generally) uses shared memory (`/dev/shm`) for inter-process/GPU communication. Docker's default `/dev/shm` is 64 MB, which causes cryptic crashes or hangs under load. `--ipc=host` (or `--shm-size=1g`) gives it room. TGI uses `--shm-size 1g` for the same reason.

Reference: the official one-liner most teams actually start from —

```bash
# vLLM official image
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env HF_TOKEN=$HF_TOKEN -p 8000:8000 --ipc=host \
  vllm/vllm-openai:latest --model mistralai/Mistral-7B-Instruct-v0.2

# TGI official image
docker run --gpus all --shm-size 1g -p 8080:80 \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-generation-inference:3.3.5 \
  --model-id teknium/OpenHermes-2.5-Mistral-7B
```

### The full stack: LLM server + reverse proxy + resource limits

A single `docker run` is fine for a dev box. A production compose file needs at minimum: the model server, a reverse proxy in front of it (TLS termination, request buffering, and a stable port even if you swap the backend image), and explicit resource limits so one runaway container can't starve its neighbors on a shared host. Here is a more complete `docker-compose.yaml`:

```yaml
services:
  llm:
    image: my-llm-server:1.0.0
    restart: unless-stopped
    expose:
      - "8000"                      # only reachable from other compose services, not the host
    ipc: host                       # equivalent to --ipc=host
    environment:
      - HF_TOKEN=${HF_TOKEN}        # sourced from host env / .env, not in image
      - MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
    volumes:
      - hf-cache:/models/hf         # persistent named volume for weights
    command: ["--model", "meta-llama/Llama-3.1-8B-Instruct"]
    deploy:
      resources:
        limits:
          cpus: "8"                 # cap host CPU this container can use
          memory: 32g                # cap host RAM (guards against OOM-killing neighbors)
        reservations:
          cpus: "4"
          memory: 16g
          devices:
            - driver: nvidia
              count: all             # or `device_ids: ["0","1"]`
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      start_period: 300s
      retries: 3

  proxy:
    image: caddy:2.8-alpine
    restart: unless-stopped
    depends_on:
      llm:
        condition: service_healthy   # don't take traffic until the model is loaded
    ports:
      - "443:443"                    # only the proxy is exposed to the host/internet
      - "80:80"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy-data:/data
    deploy:
      resources:
        limits:
          cpus: "1"
          memory: 512m

volumes:
  hf-cache:
  caddy-data:
```

A minimal `Caddyfile` fronting the model server — TLS, a request timeout longer than your typical generation latency, and a size cap on request bodies:

```
llm.example.com {
    reverse_proxy llm:8000 {
        # Streaming responses (SSE) need this off, or you buffer the whole stream.
        flush_interval -1
    }
    timeout 300s
    request_body {
        max_size 2MB
    }
}
```

Two details that matter more than they look: `expose` (not `ports`) on the `llm` service means the model server is reachable only from the `proxy` service on the compose network, not directly from the host or internet — the proxy is the only public entry point, which is where you'd add auth, rate limiting, and TLS. And `depends_on: condition: service_healthy` means the proxy won't route traffic to the model server until its `HEALTHCHECK` passes — closing the "requests arrive during the multi-minute model load and get 502s" gap.

### Adding GPU observability to the stack

The compose file above is functionally complete but operationally blind — nothing exports GPU metrics. Adding NVIDIA's DCGM exporter as a sidecar makes GPU utilization, memory, and temperature visible to Prometheus/Grafana without touching the `llm` service:

```yaml
  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04
    restart: unless-stopped
    ports:
      - "9400:9400"                 # scrape target for Prometheus
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Point Prometheus at `dcgm-exporter:9400/metrics` and alert on GPU memory utilization specifically, not just SM/compute utilization — a server can look idle on compute while sitting dangerously close to an out-of-memory KV-cache allocation, and compute-only dashboards miss that entirely (see Mechanism 6's monitoring note above).


### CI smoke-test stage

GPU-backed CI runners are expensive and often unavailable, so most teams cannot run a full inference smoke test on every commit. The pragmatic middle ground: build the image, verify it *starts and imports correctly* on a CPU-only runner (catching the large class of bugs that have nothing to do with the GPU — bad `pip install`, broken entrypoint, missing env var, syntax errors), and reserve full GPU inference smoke tests for a nightly job or a pre-deploy gate on a real GPU runner.

```yaml
# .github/workflows/docker-build.yml
name: build-and-smoke-test

on: [pull_request]

jobs:
  build:
    runs-on: ubuntu-latest         # CPU-only runner: no --gpus here
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: DOCKER_BUILDKIT=1 docker build -t my-llm-server:ci .

      - name: Scan image (fail on critical/high CVEs)
        run: docker scout cves my-llm-server:ci --exit-code --only-severity critical,high

      - name: Smoke test — container starts, imports resolve, CLI parses args
        run: |
          # No GPU on this runner: --help exits before touching CUDA, so this
          # catches broken installs / bad entrypoints without needing a GPU.
          docker run --rm my-llm-server:ci --help

      - name: Smoke test — process boots far enough to hit the arg parser
        run: |
          docker run --rm --entrypoint python3.11 my-llm-server:ci \
            -c "import vllm; print(vllm.__version__)"

  gpu-smoke-test:
    if: github.event_name == 'workflow_dispatch'   # manual / nightly, on a real GPU runner
    runs-on: [self-hosted, gpu]
    needs: build
    steps:
      - name: Full smoke test — load a tiny model, hit /health and /v1/completions
        run: |
          docker run -d --rm --gpus all --ipc=host -p 8000:8000 \
            --name smoke my-llm-server:ci --model facebook/opt-125m
          for i in $(seq 1 60); do
            curl -fsS http://localhost:8000/health && break
            sleep 5
          done
          curl -fsS http://localhost:8000/v1/models
          docker stop smoke
```

The split matters: the CPU job runs on every PR and catches most regressions cheaply; the GPU job runs on a real GPU runner against a *tiny* model (`opt-125m`, not your 70B production model) so the full request path — container start, healthcheck, an actual generation call — is exercised without needing an expensive GPU for every commit.


---

## Weights-handling comparison

| Dimension            | Bake into image           | Mount volume               | Download at startup          | OCI model artifact (emerging) |
|-----------------------|---------------------------|-----------------------------|-------------------------------|-------------------------------|
| Image size            | Huge (15–150 GB)          | Small                       | Smallest                      | Small (weights are a separate pull) |
| Cold start             | Fast (already present)    | Fast (local mount)          | Slow (multi-GB download)      | Fast once artifact is cached/mmap'd |
| Self-contained         | Yes (air-gap ok)          | No (needs volume)           | No (needs network + token)    | No (needs registry pull of artifact) |
| Swap models            | Rebuild image              | Change mount / env          | Change env var                | Change artifact tag/digest    |
| Registry cost / push   | High                       | Low                          | Low                            | Moderate (uncompressed but shared across engines) |
| Reproducibility        | Highest (weights pinned)   | Depends on volume contents  | Depends on HF tag/revision     | Highest (content-addressed digest) |
| Best for               | Air-gapped, small, regulated | Fixed nodes, shared FS    | Dev, autoscaling w/ warm cache | Multi-engine fleets sharing one checkpoint |

Pin the model **revision** (commit SHA) or artifact **digest**, not just the repo name or tag, when reproducibility matters for any of these options.

A note that cuts across every row of this table: the quantization format (see Mechanism 3's earlier note) shifts where each option's numbers land — a 4-bit-quantized 70B model bakes into an image at a size that would have been unthinkable in fp16, which is why "bake vs. mount vs. download" is a decision you should revisit per model/format combination, not settle once for the whole fleet.

---

## Failure modes and pitfalls

- **Driver/runtime mismatch** — `CUDA driver version is insufficient`. Container CUDA newer than host driver supports. Fix host driver or lower image CUDA; can't be patched in the image. On a mixed-node fleet, this can appear on *some* nodes and not others — see the case study below.
- **GPU invisible** — `torch.cuda.is_available()` is `False`, no `nvidia-smi` in container. Toolkit not installed, `--gpus` omitted, or runtime not registered. Test with `docker run --gpus all nvidia/cuda:...-base nvidia-smi`.
- **Giant images** — shipped the `devel` base, `apt` lists left behind, weights baked in, or `pip` cache retained. Use multi-stage, `--no-install-recommends`, clean `/var/lib/apt/lists`, and keep weights out. If you still land at 10+ GB (common with official framework images), that may simply be the cost of the maintained artifact — budget registry/pull time rather than fighting it alone.
- **Redownloading weights every restart** — cache path not mounted to a persistent volume, or `HF_HOME` points at an ephemeral dir. Mount a named volume at the cache path.
- **Secrets in layers** — `ENV HF_TOKEN=...` or `COPY .env` persists in image history forever. Pass at runtime or use BuildKit secrets.
- **Running as root** — default UID 0; a bad default for security and for shared-filesystem permissions. Create and drop to a non-root user. On rootless Podman/CDI setups, also check `video`/`render` group membership and SELinux `container_use_devices` — GPU access can silently fail even when the container itself is configured correctly.
- **No / bad healthcheck** — orchestrator can't detect a wedged server, or kills it during a 3-minute model load. Add a healthcheck with a generous `start-period`, and gate your reverse proxy on it (`depends_on: condition: service_healthy`) so requests don't 502 during warmup.
- **`/dev/shm` too small** — default 64 MB causes NCCL/tensor-parallel hangs and `Bus error`. Use `--ipc=host` or `--shm-size`.
- **`:latest` everywhere** — non-reproducible builds and surprise upgrades. Pin base by digest, deps by version, your image by semver.
- **Signals ignored** — shell-form `CMD` runs under `/bin/sh` which doesn't forward SIGTERM; use exec-form `ENTRYPOINT`/`CMD` so shutdown is graceful and requests drain.
- **Unscanned CVE surface** — CUDA/PyTorch images pull in a much larger OS + Python dependency graph than a typical service image, so they accumulate CVEs faster; shipping without a CI scanning gate (Docker Scout / Trivy) or a hardened base (e.g., Chainguard) is a common 2025–2026-era gap that shows up in security reviews, not in functional testing.
- **CI "passed" but the image never boots on GPU** — a CPU-only CI runner that only checks `--help`/import succeeds doesn't exercise CUDA initialization at all. Pair it with a periodic real-GPU smoke test (see the CI section above) or you'll ship a CUDA-init bug straight to prod.
- **`no kernel image is available for execution on the device`** — a from-source build's `TORCH_CUDA_ARCH_LIST` didn't include the compute capability of the GPU you're actually running on (e.g., built for 8.0/9.0, deployed on an older 7.5-class card). Fixed by rebuilding with the right architecture list, or by using a prebuilt wheel that already covers it — not a runtime-configurable option.
- **GPU memory creep goes unnoticed** — dashboards only track SM/compute utilization, which can look healthy (or idle) right up until an allocation fails. Track GPU memory utilization per container (DCGM exporter) as a first-class metric, not an afterthought.
- **Crash-loop storm from an unconditional restart policy** — a deterministic failure (driver mismatch, OOM on every boot) combined with `restart: always` and no backoff burns node scheduling slots and pages on-call repeatedly instead of failing fast. Cap retries or use `on-failure` with a backoff, and treat "restarted N times in M minutes" as its own alert.

---

## Tools and options comparison

| Option                                   | What it is                                   | When to reach for it                                     |
|-------------------------------------------|-----------------------------------------------|------------------------------------------------------------|
| `nvidia/cuda:*-runtime`                   | Slim CUDA userspace base                      | Final stage of a custom server                              |
| `nvidia/cuda:*-devel`                     | CUDA + nvcc + headers                         | Build stage compiling kernels                                |
| `vllm/vllm-openai`                        | Official vLLM OpenAI server image             | Fast path to production vLLM (accept the ~12 GB size)        |
| `ghcr.io/.../text-generation-inference`   | Official HF TGI image                         | HF-ecosystem serving, gated models                           |
| NVIDIA NIM                                | Prebuilt, GPU-SKU-tuned inference microservices (TensorRT-LLM/vLLM/SGLang) | Well-known model architectures, want less tuning work, OK with NGC entitlement |
| NVIDIA Container Toolkit                  | Host runtime that injects GPUs                | Required for any GPU container                                |
| CDI (`nvidia-ctk cdi generate`)           | Vendor-neutral device-injection spec           | Podman, rootless Docker, or any CDI-aware orchestrator        |
| Chainguard Images                         | Hardened, near-zero-CVE CUDA/PyTorch bases     | Security-sensitive deployments, want a maintained hardened base |
| Docker Model Runner / OCI model artifacts | Weights distributed as a separate OCI artifact | Multi-engine fleets sharing one checkpoint; early-adopter teams |
| BuildKit / `docker buildx`                | Modern builder: cache mounts, secrets          | Every build (faster, safer)                                    |
| `dive` / `docker history`                 | Inspect layers & size                          | Hunting image bloat                                             |
| `docker scout` / `trivy`                  | Image vulnerability scanning                   | CI gate before push, especially for large CUDA/PyTorch images    |
| `syft` / `docker sbom` / cosign           | SBOM generation and image signing              | Supply-chain attestation, verifying provenance before deploy    |
| `nvidia-smi mig` / MIG profiles           | Hardware GPU partitioning                      | Hard multi-tenant isolation on A100/H100-class GPUs             |
| DCGM / `dcgm-exporter`                    | GPU metrics exporter (Prometheus)              | Fleet-wide GPU utilization, memory, and health monitoring       |
| NVIDIA MPS                                | Software multi-process GPU sharing             | Sharing one GPU across cooperating processes with more predictable compute allocation than plain time-slicing |


---

## Production case studies & war stories

Three incidents that recur often enough in practice to be worth internalizing before you hit them yourself.

### War story 1 — the driver mismatch that only broke on one node type

**Setup.** A platform team ran inference on an autoscaling GPU node pool mixing two instance types: an older generation bought a year earlier (driver 535.x, installed when the nodes were provisioned) and a newer generation added last quarter (driver 550.x, from a newer base AMI). The serving image was built on `nvidia/cuda:12.4.1-*`, which needs a driver new enough to support CUDA 12.4's userspace.

**What happened.** A routine image bump (upgrading a Python dependency, nothing GPU-related) triggered a rolling redeploy across the whole node pool. Pods scheduled onto the newer-generation nodes (driver 550.x) came up fine. Pods scheduled onto the older-generation nodes (driver 535.x — new enough for CUDA 12.2, not comfortably for 12.4) crash-looped with:

```
CUDA driver version is insufficient for CUDA runtime version
```

Because the autoscaler load-balanced across both node types, roughly a third of pods were healthy and two-thirds were crash-looping — the deploy looked like a partial, confusing outage rather than a clean pass/fail, and the on-call's first instinct (check the app logs, check the model, check the request path) burned the first 40 minutes before someone ran `nvidia-smi` on a failing node and saw the driver version.

**Root cause.** The container's CUDA version was never validated against the *oldest* driver in the fleet — only against the engineer's own dev box, which happened to have the newer driver. Nothing in CI caught it, because CI didn't have GPU nodes of the older generation to test against.

**Fix and lesson.** The team added a driver-version floor check to node bootstrap (fail node registration if `nvidia-smi --query-gpu=driver_version --format=csv,noheader` is below a pinned minimum) so the *fleet's* minimum driver version becomes an explicit, enforced contract rather than an implicit assumption about whichever node an engineer happened to test on. They also added a one-line preflight in the container's entrypoint — run `nvidia-smi` before starting the server and exit with a clear error if it fails — so a driver mismatch produces an immediate, legible failure instead of a framework-level CUDA init stack trace three layers down. The generalizable lesson: **treat "minimum supported host driver version" as a versioned contract between infra and the serving image, checked at both node-bootstrap time and container-start time — not something that's implicitly whatever driver happened to be on the box someone tested on.**

### War story 2 — the 40 GB image that broke CI, not just the registry

**Setup.** An early version of a custom serving image baked model weights directly into the image (Option A from Mechanism 3) because it was simple and "worked on my machine." The weights were a 34B-parameter checkpoint in fp16, roughly 70 GB of safetensors, compressing to a ~40 GB image layer.

**What happened.** CI ran `docker build` on every PR to catch regressions. The build step alone took 25+ minutes once network transfer of that layer was involved, and the shared CI runner's local image cache (sized for typical services, a few GB) evicted the 40 GB layer between runs — so nearly every PR paid the full weights-copy cost again, rather than getting a cache hit. Beyond CI: every `docker push`/`pull` moved the full 40 GB, registry storage costs for versioned images climbed fast (each rebuild produced a new immutable tag), and rolling out a one-line code fix to production meant redistributing the entire checkpoint to every node again, turning what should have been a 30-second deploy into a 20+ minute one gated by network transfer.

**Root cause.** Weights (large, slow-changing, indifferent to code) and code (tiny, fast-changing) were coupled into one artifact and one lifecycle, so every code change paid the weights-transfer cost, and CI's cache assumptions (sized for normal service images) were simply wrong for this workload.

**Fix and lesson.** The team moved to Option B (mount a volume pre-populated with weights on each node, refreshed out-of-band from the deploy pipeline) for production, and kept a from-scratch build with a *tiny* stand-in model (`facebook/opt-125m`, a few hundred MB) for CI so the Dockerfile/dependency layers were still validated on every PR without moving real weights through the build pipeline at all. Deploy time for a code-only change dropped from ~20 minutes to under a minute; CI build time dropped from 25+ minutes to about 90 seconds. The generalizable lesson: **the moment your image is dominated by data rather than code, your CI and registry tooling need a data-vs-code split too — don't let a shared cache and registry sized for normal service images silently absorb a 40 GB workload; separate the lifecycles explicitly (Mechanism 3's bake/mount/download decision isn't just a runtime-architecture choice, it's a CI/registry-cost decision too).**

### War story 3 — the "idle" GPU that was actually one bad allocation from falling over

**Setup.** A dashboard tracked GPU SM/compute utilization per node as the primary GPU health signal, on the reasonable-sounding assumption that "low utilization = healthy, high utilization = busy."

**What happened.** A slow KV-cache memory leak, triggered only by a specific long-context request pattern, grew VRAM usage across days while compute utilization stayed unremarkable — the server was mostly waiting on generation, not compute-bound, so nothing on the compute dashboard moved. The first visible symptom was a hard CUDA out-of-memory crash under otherwise normal load, with no warning in the metrics anyone was watching.

**Fix and lesson.** The team added GPU memory utilization (not just compute) as an explicit, alerted metric via DCGM, with a warning threshold well below 100% so there was runway to intervene before a hard OOM. The generalizable lesson, and the reason it's paired with Mechanism 6's monitoring note earlier in this chapter: **compute utilization and memory utilization are different signals that fail independently — a GPU container can look "idle" by one measure while one allocation away from crashing by the other, so watch both, not just the one that's easiest to eyeball on `nvidia-smi`.**



---

## Interview mastery

### "Explain why GPU images are different from normal Docker images" — in 60 seconds

A normal Docker image is fully self-contained: the base image, the runtime, and the app all travel together, and the host just runs the kernel. A GPU image breaks that isolation on purpose. The container ships CUDA/cuDNN *userspace* libraries, but the GPU *driver* is a kernel module that must already be installed on the host — you never bake it into the image — so there's a version contract: the container's CUDA version has to be no newer than what the host driver supports, checked with `nvidia-smi`. Getting the GPU into the container at all requires a host-side component, the NVIDIA Container Toolkit, which injects the driver's libraries and device nodes at `docker run --gpus all` time — a plain container gets no GPU access. On top of that, these images are unusually large (multi-GB CUDA runtime, plus optionally multi-GB-to-hundreds-of-GB of model weights), which forces explicit decisions a normal service image never has to make: build vs. runtime base (multi-stage), and whether weights live in the image, a mounted volume, or a startup download. And because the OS + CUDA + Python dependency graph is so much bigger, these images carry a larger CVE surface than a typical microservice, which is why vulnerability scanning and hardened base images get called out specifically for AI workloads rather than being generic Docker hygiene.

### Q&A

1. **How does a container get access to the GPU?** Host driver + NVIDIA Container Toolkit + `--gpus`. You never install the driver in the image; the toolkit injects host driver libs at runtime. Verify with `docker run --gpus all ... nvidia-smi`.
2. **`runtime` vs `devel` base image — which do you ship?** Build in `devel`, ship on `runtime` via multi-stage. Shipping `devel` is a multi-GB mistake — nvcc, headers, and static libs you never need at inference time.
3. **Where do the weights live and why?** Articulate bake vs. mount vs. download (and the emerging OCI-model-artifact option) and the cold-start/size/reproducibility tradeoffs; know that a mounted, persistent HF cache (`HF_HOME`) is the usual production answer, with baking reserved for air-gapped/regulated cases.
4. **How do you keep the image small?** Multi-stage, `--no-install-recommends`, clean apt lists, `.dockerignore`, don't bake weights, BuildKit cache mounts. Also know the honest limit: official framework images (e.g., `vllm/vllm-openai`) are large (~12 GB) largely by design/maintenance tradeoff, not a bug you can always fix yourself.
5. **How do you handle the CUDA/driver version contract?** Pin container CUDA ≤ host-driver-supported; understand minor-version/forward compatibility; recognize the "driver insufficient" error and know it's fixed on the host or by lowering the image's CUDA version, never inside the container.
6. **How do secrets and config get in?** Runtime env / orchestrator secrets and BuildKit `--secret`, never `ENV`/`COPY .env` (persists in layer history forever, even if a later layer deletes the file).
7. **Non-root, healthcheck, signals?** Drop to an unprivileged UID, healthcheck with a long `start-period` for model load (and gate a reverse proxy on that healthcheck), exec-form entrypoint for graceful SIGTERM draining.
8. **Why `--ipc=host` / `--shm-size`?** Tensor-parallel / NCCL uses `/dev/shm`; the 64 MB default causes hangs and bus errors under load.
9. **What is the NVIDIA Container Toolkit actually doing under the hood?** It's a host-side runtime shim that, at container-start, mounts the host's GPU device nodes and matching driver userspace libraries into the container's filesystem/namespace, based on `NVIDIA_VISIBLE_DEVICES` (legacy) or a CDI spec (current). It does not install or virtualize a driver — the container always uses the exact host driver.
10. **What is CDI and why does it matter?** The Container Device Interface is a vendor-neutral spec (Docker/Podman/containerd/Kubernetes all understand it) for describing device injection, replacing NVIDIA's older proprietary hook mechanism. It matters practically because it's what makes rootless GPU containers (Podman, rootless Docker) workable — the legacy hook assumed a privileged daemon.
11. **How would you run a GPU container rootless, and what breaks if you don't set it up right?** Generate a user-space CDI spec (`nvidia-ctk cdi generate --output=~/.config/cdi/nvidia.yaml`), reference the device by its CDI name (`--device nvidia.com/gpu=all`), and ensure the invoking user has device-file permissions (`video`/`render` group) and, on SELinux hosts, `container_use_devices` enabled — otherwise the container starts but silently can't see the GPU.
12. **Why might you choose an official/vendor image (vLLM, TGI, NIM) over a hand-rolled Dockerfile?** Less Dockerfile/CUDA-version maintenance burden, a tested and (for NIM) per-GPU-SKU-tuned engine, faster time to a working server. Trade-offs: less control over exact image contents/size, and for NIM, potential licensing/entitlement gating.
13. **How do you defend an image's security posture in a review?** Name the concrete mechanisms: pinned digests, non-root user, no secrets in layers, a CI scanning gate (Docker Scout/Trivy) with a CVE-severity threshold, and — if asked about the current state of the art — hardened base images (e.g., Chainguard) that measurably cut CVE count versus the stock CUDA/PyTorch bases.
14. **A GPU container works on your dev box but crash-loops on some fleet nodes with a CUDA driver error — how do you debug and fix it, structurally, not just for this incident?** Diagnose: `nvidia-smi` on the failing node to read its driver version, compare against the image's CUDA version. Fix the immediate incident by aligning driver/CUDA. Fix it structurally by enforcing a minimum-driver-version check at node bootstrap and a preflight `nvidia-smi` check in the container entrypoint, so the fleet's driver floor is an explicit, tested contract instead of "whatever driver the last person's dev box had."
15. **Your image is 40+ GB because it bakes in the weights, and CI/registry costs are exploding — what do you change?** Split weights out of the image (mount or download), keep a tiny stand-in model for CI/Dockerfile validation, and separate the code-deploy lifecycle from the weights-distribution lifecycle so a one-line code fix doesn't require redistributing the checkpoint.
16. **What's the difference between baking weights into an image layer and packaging them as a separate OCI artifact?** An image layer is a compressed tarball; weight files are high-entropy so compression barely helps and costs (de)compression time, and the engine can't `mmap` a compressed layer directly. An OCI model artifact stores the weights uncompressed under a model-specific media type, decoupled from any particular serving engine's image, so multiple engines can reference the same weights without duplicating them and the engine can load it more directly.
17. **Why do healthcheck `start-period` and reverse-proxy `depends_on` matter together?** A model load can take minutes; without a long `start-period` the orchestrator may kill the container mid-load, and without gating the proxy on the healthcheck, requests arrive and get 502s during that window even if the container itself survives.
18. **When would you deliberately choose a larger, less-optimized image over a hand-tuned minimal one?** When the maintenance cost of hand-slimming exceeds the storage/pull-time cost — e.g., adopting the official `vllm/vllm-openai` image rather than fighting its size, because the vLLM maintainers themselves have indicated from-source slimming isn't a reliably supported path.
19. **What's the difference between MIG and time-slicing, and when would you reach for each?** MIG partitions a GPU in hardware into isolated instances (strong isolation, fixed profile sizes, set up outside the container); time-slicing shares a whole GPU across containers in software with no memory isolation (simpler, weaker guarantee). Reach for MIG when tenants must not be able to starve each other; time-slicing is fine for trusted, non-adversarial sharing.
20. **Do Docker/Kubernetes resource limits cap GPU usage the way they cap CPU and memory?** No — `cpus`/`memory` limits are enforced via host cgroups and are real; there's no equivalent cgroups-based cap on GPU compute or VRAM for a container. `--gpus`/device requests control *which* GPU(s) a container can see, not how much of a shared one it's capped at using. MIG or MPS are the actual mechanisms for bounding/sharing GPU resources between containers.
21. **How do you monitor a fleet of GPU containers in production?** DCGM exporter (or equivalent) scraped by Prometheus, tracking both compute *and* memory utilization per GPU/container — not compute alone, since a memory-bound failure can occur with unremarkable compute metrics right up until an OOM.
22. **A container gets SIGTERM mid-generation — what should happen, and what's the default risk?** Default Docker grace period (10s) is often too short to drain an in-flight streaming response; the fix is a longer grace period (`stop_grace_period`/`terminationGracePeriodSeconds`) plus application-level handling that stops accepting new requests immediately while letting in-flight ones finish within the window.
23. **A build stage that compiles kernels from source suddenly fails on the deploy GPU with `no kernel image is available for execution on the device` — what's wrong?** `TORCH_CUDA_ARCH_LIST` (or equivalent) didn't target that GPU's compute capability at build time; the fix is rebuilding for the right architecture list or switching to a prebuilt wheel that already covers it, not anything fixable at runtime.
24. **CI takes 25 minutes to build your image and the cache never seems to hit on a fresh runner — why, and what's the fix?** Local Docker layer caching only helps on the same machine; ephemeral/shared CI runners don't retain it between jobs. Push/pull the build cache itself through a registry (`docker buildx build --cache-to type=registry,mode=max --cache-from type=registry`) so any runner can warm from the last successful build.

### System design prompt: "Containerize a multi-GPU model server for production"

A common follow-up prompt: *design the containerization and rollout for a model server that needs 4 GPUs per replica (tensor-parallel), running across a fleet, with zero-downtime deploys.* A sketch of the answer, mapping back to mechanisms in this chapter:

```
                                 ┌─────────────────────────────┐
                                 │        Load balancer /       │
                                 │  reverse proxy (TLS, auth,   │
                                 │  rate limit, streaming SSE)  │
                                 └───────────────┬───────────────┘
                                                 │  routes only to
                                                 │  healthy replicas
              ┌──────────────────────────────────┼──────────────────────────────────┐
              │                                  │                                  │
   ┌──────────▼─────────┐            ┌───────────▼──────────┐            ┌──────────▼─────────┐
   │ Replica (canary,    │            │ Replica (stable, v1)  │            │ Replica (stable, v1) │
   │ v2, 5% of traffic)  │            │ 4x GPU, TP=4           │            │ 4x GPU, TP=4          │
   │ 4x GPU, TP=4        │            │ node affinity: pool-A  │            │ node affinity: pool-A │
   │ node affinity: pool-B│           │ driver >= floor pinned │            │ driver >= floor pinned │
   └──────────┬──────────┘            └───────────┬───────────┘            └──────────┬────────────┘
              │                                    │                                   │
              └───────────────┬────────────────────┴───────────────┬───────────────────┘
                              │                                    │
                   ┌──────────▼──────────┐                ┌────────▼─────────┐
                   │  Weights: mounted    │                │ Secrets: HF_TOKEN, │
                   │  volume / warm HF    │                │ TLS certs — from    │
                   │  cache per node pool │                │ orchestrator secret │
                   │  (shared, versioned) │                │ store, never baked  │
                   └──────────────────────┘                └────────────────────┘

   CI/CD gate (pre-deploy): docker build → docker scout/trivy scan (fail on
   critical/high) → CPU smoke test (import + --help) → nightly/pre-promote GPU
   smoke test (tiny model, real /v1/completions call) → sign + push digest →
   canary 5% on pool-B → automated rollback if error-rate/latency regress →
   promote to 100%.
```

Talking points an interviewer wants to hear, in roughly this order: (1) the image is built multi-stage (`devel` → `runtime`), non-root, healthchecked, exec-form entrypoint; (2) weights are *not* baked in — mounted or cached per node pool so a code-only rollout doesn't redistribute the checkpoint (tie this directly to the CI/registry-cost war story); (3) `--ipc=host`/`--shm-size` is set because TP=4 needs real shared memory for NCCL; (4) node pools have an enforced minimum driver version, checked at bootstrap and preflighted in the entrypoint, so a mixed-generation fleet can't silently crash-loop a subset of replicas; (5) the reverse proxy is the only externally exposed surface and only routes to replicas passing healthcheck; (6) rollout is canary-then-promote, gated by a CI pipeline that scans for CVEs and smoke-tests on both CPU (every PR) and GPU (pre-promote); (7) secrets come from the orchestrator's secret store, never from the image.

### Red flags vs. green flags

| Signal                                                              | Red flag                                                        | Green flag                                                                 |
|------------------------------------------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------|
| Base image choice                                                    | Ships `devel` to production, or `python:slim` with manual CUDA install | Multi-stage: `devel` to build, `runtime` (or a hardened base) to ship        |
| Model weights                                                        | Baked into the image "because it was simpler"                   | Explicit bake/mount/download decision tied to a stated tradeoff (air-gap, node pool, autoscaling) |
| Driver/CUDA versioning                                                | "It works on my machine," no stated minimum driver version       | A pinned, fleet-wide minimum driver version enforced at bootstrap + preflighted at container start |
| GPU access                                                            | Doesn't know the difference between `--gpus all` and `NVIDIA_VISIBLE_DEVICES`, never heard of CDI | Can explain the toolkit/CDI injection mechanism and when rootless matters   |
| Secrets                                                                | `ENV HF_TOKEN=...` or `COPY .env` in the Dockerfile              | Runtime env / orchestrator secrets / BuildKit `--secret`                     |
| User                                                                   | Runs as root, no justification                                   | Non-root UID, with cache/data dirs explicitly chowned                        |
| Healthcheck                                                            | None, or a short `start-period` that kills the container during model load | Healthcheck with a generous `start-period`, and downstream proxy gated on it |
| `/dev/shm`                                                             | Unaware TP/NCCL needs shared memory; hits mysterious bus errors under load | Sets `--ipc=host`/`--shm-size` deliberately, can explain why                  |
| CI                                                                     | No image build validation, or claims "GPU tested" from a CPU-only runner | CPU smoke test on every PR, real GPU smoke test with a tiny model pre-promote |
| Security posture                                                       | No scanning, no SBOM, "we haven't gotten to that yet"            | CI-gated CVE scanning, SBOM, ideally a hardened base or explicit CVE-budget policy |
| Talking about size                                                     | "The image is 40 GB, that's just how it is" with no plan          | Can name the specific driver of size (weights? `devel` base? apt cache?) and the fix |


---

---

## Quick reference: commands you'll actually run

A condensed cheat sheet of the commands from this chapter you'll reach for most often, in the order you'd typically use them.

```bash
# 1. Verify the whole host GPU chain before blaming your image
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# 2. Build with BuildKit, warming/publishing the cache through a registry (fast CI)
docker buildx build \
  --cache-from type=registry,ref=myregistry.example.com/my-llm-server:buildcache \
  --cache-to   type=registry,ref=myregistry.example.com/my-llm-server:buildcache,mode=max \
  -t my-llm-server:1.0.0 .

# 3. Scan before you ship it
docker scout cves my-llm-server:1.0.0 --exit-code --only-severity critical,high

# 4. Run it for real: GPUs, shared memory, persistent weights cache, runtime secrets
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v $HOME/.cache/hf:/models/hf -e HF_TOKEN=$HF_TOKEN \
  my-llm-server:1.0.0 --model meta-llama/Llama-3.1-8B-Instruct

# 5. Or bring up the whole stack (proxy + limits + healthchecks) via compose
docker compose up -d

# 6. Debug from inside a running container — GPU state as *this* container sees it
docker exec -it my-llm-server nvidia-smi

# 7. Rootless / Podman: generate a user-space CDI spec once per host
mkdir -p ~/.config/cdi && nvidia-ctk cdi generate --output=$HOME/.config/cdi/nvidia.yaml

# 8. Shut down cleanly, giving in-flight streaming requests time to drain
docker stop -t 60 my-llm-server
```

Keep this list next to the Dockerfile and compose file from earlier in this chapter — between the two, they cover build, ship, run, observe, and shut down for a single-node GPU deployment; the Kubernetes chapter picks up from here for multi-node scheduling.

---

## Further reading

**Core toolkit and base images**
- NVIDIA Container Toolkit — repo, releases (v1.19.0, March 12, 2026): https://github.com/NVIDIA/nvidia-container-toolkit
- NVIDIA Container Toolkit — install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- NVIDIA Container Toolkit — overview: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html
- `nvidia/cuda` image tags (Docker Hub): https://hub.docker.com/r/nvidia/cuda/tags
- CUDA container supported tags & flavors: https://gitlab.com/nvidia/container-images/cuda/-/blob/master/doc/supported-tags.md

**CDI and rootless GPU containers**
- Container Device Interface support (NVIDIA Container Toolkit docs): https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.16.2/cdi-support.html
- CDI support in the GPU Operator (25.10): https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/25.10/cdi.html
- Running NVIDIA GPU containers with Podman (rootless, CDI walkthrough, 2026-03-18): https://oneuptime.com/blog/post/2026-03-18-run-nvidia-gpu-containers-podman/view
- Using CDI with Podman: https://oneuptime.com/blog/post/2026-03-18-use-cdi-container-device-interface-podman/view

**Official framework images**
- vLLM — Using Docker: https://docs.vllm.ai/en/latest/deployment/docker/
- vLLM GitHub issue — reducing the official image size (maintainer response on from-source builds): https://github.com/vllm-project/vllm/issues/27154
- vLLM Forums — "Current vLLM docker image size is 12.64Gb, how to reduce it?": https://discuss.vllm.ai/t/current-vllm-docker-image-size-is-12-64gb-how-to-reduce-it/1204
- Hugging Face TGI — Nvidia GPU install: https://huggingface.co/docs/text-generation-inference/en/installation_nvidia
- Hugging Face TGI — Quick Tour: https://huggingface.co/docs/text-generation-inference/quicktour
- Hugging Face Hub — Understand caching (`HF_HOME`): https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
- NVIDIA NIM — overview and developer docs: https://developer.nvidia.com/nim
- NVIDIA NIM — microservices product page: https://www.nvidia.com/en-us/ai-data-science/products/nim-microservices/

**Image hardening, size, and weight packaging**
- Chainguard — Securing the foundations of AI applications (zero-CVE PyTorch/CUDA image comparison): https://www.chainguard.dev/unchained/securing-the-foundations-of-ai-applications-with-chainguard-images
- Chainguard Containers — overview: https://edu.chainguard.dev/chainguard/chainguard-images/overview/
- Docker — Why OCI Artifacts for AI Model Packaging: https://www.docker.com/blog/oci-artifacts-for-ai-model-packaging/

**Supply-chain scanning and security**
- Docker Scout — `docker scout cves` reference: https://docs.docker.com/reference/cli/docker/scout/cves/
- Docker Hardened Images — scanning how-to: https://docs.docker.com/dhi/how-to/scan/
- Vulnerability management with Trivy (2025-10-19): https://infrahouse.com/blog/2025-10-19-vulnerability-management-part2-trivy/

**Docker mechanics**
- Docker — Multi-stage builds: https://docs.docker.com/build/building/multi-stage/
- Docker — Building best practices: https://docs.docker.com/build/building/best-practices/
- Docker — Build cache & BuildKit: https://docs.docker.com/build/cache/
- Docker Compose — GPU support: https://docs.docker.com/compose/how-tos/gpu-support/
- NVIDIA GPU Operator — CDI support (25.10): https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/25.10/cdi.html
- NVIDIA — Multi-Instance GPU (MIG) user guide: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html
- NVIDIA DCGM — Data Center GPU Manager overview: https://developer.nvidia.com/dcgm
- Docker — `stop`/graceful-shutdown grace period reference: https://docs.docker.com/reference/cli/docker/container/stop/
- Docker Buildx — registry cache backend (`--cache-to`/`--cache-from`): https://docs.docker.com/build/cache/backends/registry/
- vLLM GitHub issue — earlier image-size discussion, `python-slim` attempt: https://github.com/vllm-project/vllm/issues/13112
- NVIDIA — CUDA compatibility guide (forward compatibility, driver/toolkit matrix): https://docs.nvidia.com/deploy/cuda-compatibility/index.html
