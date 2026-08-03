# Docker for GPU LLM Serving

> Containerizing an LLM inference service so it runs the same on your laptop, in CI, and on a rented H100 — without a 40 GB image, a leaked token, or a `CUDA driver version is insufficient` at 3 a.m.

## Why this matters

An LLM server is not a normal web app. It links against CUDA, cuDNN, NCCL, and a specific PyTorch build; it needs a physical GPU exposed into the container; and it depends on multi-gigabyte model weights that you do *not* want to redownload on every restart. Get the containerization wrong and you hit one of a dozen classic failures: the image balloons to tens of gigabytes, the container can't see the GPU, the CUDA version mismatches the host driver, your Hugging Face token ends up baked into a layer, or the server runs as root with no healthcheck and the orchestrator can't tell it's wedged.

Containers are also the unit that Kubernetes, Nomad, ECS, and every autoscaler schedule. A well-built image is the foundation for everything in later chapters (K8s, canary, autoscaling). This chapter is about getting that foundation right.

The intuition first, then the exact mechanisms.

---

## Core intuition

Three ideas carry most of the weight.

**1. The container shares the host's GPU driver, not its own.** You never install an NVIDIA *driver* inside the image. The driver is a kernel module and lives on the host. The container ships *userspace* CUDA libraries. At `docker run` time, the NVIDIA Container Toolkit injects the host driver's device files and libraries into the container. So the contract is: **host driver must be new enough for the container's CUDA userspace.** This is the single most important mental model in GPU containers.

**2. Model weights are data, not code.** Code changes daily and is tiny; a 14 GB weights file changes rarely and is huge. Baking weights into an image layer couples the two lifecycles badly. The default for production is to keep weights *out* of the image and bring them in as a mounted volume or a startup download into a persistent cache.

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

That means the container's CUDA userspace is newer than the host driver can serve. Fix by upgrading the host driver or downgrading the image's CUDA version — you cannot fix it inside the image.

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

---

## Mechanism 3 — Handling large model weights

This is where architecture decisions bite hardest. You have three options.

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

---

## Mechanism 4 — Multi-stage builds, layer caching, and `.dockerignore`

### Multi-stage

A multi-stage build uses multiple `FROM` statements. Early stages compile; the final stage copies only what's needed. For an LLM server that means: compile custom kernels / install a heavy build toolchain in a `devel` stage, then `COPY --from=builder` the installed environment into a slim `runtime` stage. The `devel` layers never ship.

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

## Fully worked example

A real, multi-stage GPU Dockerfile for a vLLM-based OpenAI-compatible server. (If you just want vLLM, the official `vllm/vllm-openai` image is usually the right call — this shows the *general* pattern you'd use for a custom server or a framework without an official image.)

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

### Compose snippet

`docker compose` needs the `deploy.resources.reservations.devices` block to request GPUs:

```yaml
services:
  llm:
    image: my-llm-server:1.0.0
    ports:
      - "8000:8000"
    ipc: host                       # equivalent to --ipc=host
    environment:
      - HF_TOKEN=${HF_TOKEN}        # sourced from host env / .env, not in image
      - MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
    volumes:
      - hf-cache:/models/hf         # persistent named volume for weights
    command: ["--model", "meta-llama/Llama-3.1-8B-Instruct"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all            # or `device_ids: ["0","1"]`
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      start_period: 300s
      retries: 3

volumes:
  hf-cache:
```

---

## Weights-handling comparison

| Dimension            | Bake into image           | Mount volume               | Download at startup          |
|----------------------|---------------------------|----------------------------|------------------------------|
| Image size           | Huge (15–150 GB)          | Small                      | Smallest                     |
| Cold start           | Fast (already present)    | Fast (local mount)         | Slow (multi-GB download)     |
| Self-contained       | Yes (air-gap ok)          | No (needs volume)          | No (needs network + token)   |
| Swap models          | Rebuild image             | Change mount / env         | Change env var               |
| Registry cost / push | High                      | Low                        | Low                          |
| Reproducibility      | Highest (weights pinned)  | Depends on volume contents | Depends on HF tag/revision   |
| Best for             | Air-gapped, small, regulated | Fixed nodes, shared FS  | Dev, autoscaling w/ warm cache |

Pin the model **revision** (commit SHA), not just the repo name, when reproducibility matters for Options B and C.

---

## Failure modes and pitfalls

- **Driver/runtime mismatch** — `CUDA driver version is insufficient`. Container CUDA newer than host driver supports. Fix host driver or lower image CUDA; can't be patched in the image.
- **GPU invisible** — `torch.cuda.is_available()` is `False`, no `nvidia-smi` in container. Toolkit not installed, `--gpus` omitted, or runtime not registered. Test with `docker run --gpus all nvidia/cuda:...-base nvidia-smi`.
- **Giant images** — shipped the `devel` base, `apt` lists left behind, weights baked in, or `pip` cache retained. Use multi-stage, `--no-install-recommends`, clean `/var/lib/apt/lists`, and keep weights out.
- **Redownloading weights every restart** — cache path not mounted to a persistent volume, or `HF_HOME` points at an ephemeral dir. Mount a named volume at the cache path.
- **Secrets in layers** — `ENV HF_TOKEN=...` or `COPY .env` persists in image history forever. Pass at runtime or use BuildKit secrets.
- **Running as root** — default UID 0; a bad default for security and for shared-filesystem permissions. Create and drop to a non-root user.
- **No / bad healthcheck** — orchestrator can't detect a wedged server, or kills it during a 3-minute model load. Add a healthcheck with a generous `start-period`.
- **`/dev/shm` too small** — default 64 MB causes NCCL/tensor-parallel hangs and `Bus error`. Use `--ipc=host` or `--shm-size`.
- **`:latest` everywhere** — non-reproducible builds and surprise upgrades. Pin base by digest, deps by version, your image by semver.
- **Signals ignored** — shell-form `CMD` runs under `/bin/sh` which doesn't forward SIGTERM; use exec-form `ENTRYPOINT`/`CMD` so shutdown is graceful and requests drain.

---

## Tools and options comparison

| Option                          | What it is                              | When to reach for it                          |
|---------------------------------|-----------------------------------------|-----------------------------------------------|
| `nvidia/cuda:*-runtime`         | Slim CUDA userspace base                | Final stage of a custom server                |
| `nvidia/cuda:*-devel`           | CUDA + nvcc + headers                   | Build stage compiling kernels                 |
| `vllm/vllm-openai`              | Official vLLM OpenAI server image       | Fast path to production vLLM                   |
| `ghcr.io/.../text-generation-inference` | Official HF TGI image           | HF-ecosystem serving, gated models            |
| NVIDIA Container Toolkit        | Host runtime that injects GPUs          | Required for any GPU container                 |
| BuildKit / `docker buildx`      | Modern builder: cache mounts, secrets   | Every build (faster, safer)                    |
| `dive` / `docker history`       | Inspect layers & size                   | Hunting image bloat                            |
| `docker scout` / `trivy`        | Image vulnerability scanning            | CI gate before push                            |

---

## Production checklist — what an interviewer probes

1. **"How does a container get access to the GPU?"** — Host driver + NVIDIA Container Toolkit + `--gpus`. You never install the driver in the image; the toolkit injects host driver libs at runtime. Verify with `docker run --gpus all ... nvidia-smi`.
2. **"runtime vs devel base image — which do you ship?"** — Build in `devel`, ship on `runtime` via multi-stage. Shipping `devel` is a multi-GB mistake.
3. **"Where do the weights live and why?"** — Articulate bake vs mount vs download and the cold-start/size/reproducibility tradeoffs; know that a mounted persistent HF cache (`HF_HOME`) is the usual answer.
4. **"How do you keep the image small?"** — Multi-stage, `--no-install-recommends`, clean apt lists, `.dockerignore`, don't bake weights, BuildKit cache mounts.
5. **"How do you handle the CUDA/driver version contract?"** — Pin container CUDA ≤ host-driver-supported; understand minor-version/forward compatibility; recognize the "driver insufficient" error.
6. **"How do secrets and config get in?"** — Runtime env / orchestrator secrets and BuildKit `--secret`, never `ENV`/`COPY .env` (persists in layer history).
7. **"Non-root, healthcheck, signals?"** — Drop to an unprivileged UID, healthcheck with a long `start-period` for model load, exec-form entrypoint for graceful SIGTERM draining.
8. **"Why `--ipc=host` / `--shm-size`?"** — Tensor-parallel / NCCL uses `/dev/shm`; the 64 MB default causes hangs and bus errors under load.

---

## Further reading

- NVIDIA Container Toolkit — repo and overview: https://github.com/NVIDIA/nvidia-container-toolkit
- NVIDIA Container Toolkit — install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- `nvidia/cuda` image tags (Docker Hub): https://hub.docker.com/r/nvidia/cuda/tags
- CUDA container supported tags & flavors: https://gitlab.com/nvidia/container-images/cuda/-/blob/master/doc/supported-tags.md
- vLLM — Using Docker: https://docs.vllm.ai/en/stable/deployment/docker/
- Hugging Face TGI — Nvidia GPU install: https://huggingface.co/docs/text-generation-inference/en/installation_nvidia
- Hugging Face TGI — Quick Tour: https://huggingface.co/docs/text-generation-inference/quicktour
- Hugging Face Hub — Understand caching (`HF_HOME`): https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
- Docker — Multi-stage builds: https://docs.docker.com/build/building/multi-stage/
- Docker — Building best practices: https://docs.docker.com/build/building/best-practices/
- Docker — Build cache & BuildKit: https://docs.docker.com/build/cache/
- Docker Compose — GPU support: https://docs.docker.com/compose/how-tos/gpu-support/
