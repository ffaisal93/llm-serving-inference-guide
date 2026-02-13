# Topic 2: Containerization with Docker

## What You'll Learn

This topic teaches you how to:
- Containerize LLM serving applications
- Create efficient Docker images for ML workloads
- Handle large model files
- Use multi-stage builds
- Configure environment variables
- Optimize image size

## Why We Need This

### Business Need
- **Consistency**: Same model behavior in dev, staging, production
- **Speed to market**: Deploy faster without environment setup
- **Cost reduction**: Standardized deployment reduces ops overhead
- **Compliance**: Reproducible deployments for audit trails

### Technical Need
- **Environment consistency**: Python version, dependencies, system libraries
- **Isolation**: No conflicts between different ML projects
- **Portability**: Run anywhere (local, cloud, edge)
- **Versioning**: Tag images with model versions for rollback

### Real-World Impact
Without containerization:
- ❌ "Works on my machine" problems
- ❌ Difficult to reproduce production issues
- ❌ Slow deployments (manual environment setup)
- ❌ Can't scale easily (each server needs manual setup)

## Industry Use Cases

### 1. **Multi-Cloud Deployment**
**Company**: Enterprise ML platforms
**Use Case**: 
- Same Docker image runs on AWS, GCP, Azure
- No vendor lock-in, easy migration

**Example:**
```bash
# Build once, deploy anywhere
docker build -t llm-model:v1.0 .
# Deploy to AWS
docker push llm-model:v1.0
# Deploy to GCP
docker push llm-model:v1.0
```

### 2. **CI/CD Pipelines**
**Company**: All tech companies
**Use Case**:
- Automated testing in containers
- Consistent environments across pipeline stages

**Example:**
```yaml
# GitHub Actions / GitLab CI
- name: Test model
  run: |
    docker build -t test-model .
    docker run test-model pytest
```

### 3. **Edge Deployment**
**Company**: IoT, autonomous vehicles
**Use Case**:
- Deploy models to edge devices
- Same container on server and edge

**Example:**
```bash
# Deploy to edge device
docker save llm-model:v1.0 | ssh edge-device docker load
```

### 4. **Model Versioning & Rollback**
**Company**: ML platforms (MLflow, Weights & Biases)
**Use Case**:
- Tag images with model versions
- Quick rollback to previous version

**Example:**
```bash
docker tag llm-model:latest llm-model:v1.2.3
docker tag llm-model:latest llm-model:v1.2.2  # Rollback
```

### 5. **Development Teams**
**Company**: All companies with ML teams
**Use Case**:
- New developers can run models immediately
- No "works on my machine" issues

**Example:**
```bash
# New developer setup
git clone repo
docker-compose up  # Everything works!
```

## Industry-Standard Boilerplate Code

### Production Dockerfile (Industry Standard)

```dockerfile
# Multi-stage build for production (used by: Google, AWS, Microsoft)
# Stage 1: Builder
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime (minimal image)
FROM python:3.9-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Create non-root user (security best practice)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Copy application
COPY --chown=appuser:appuser . .

# Health check (for K8s, load balancers)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app.py"]
```

### Docker Compose for Development (Industry Standard)

```yaml
# docker-compose.yml
# Used by: Development teams, local testing
version: '3.8'

services:
  llm-serving:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=gpt2
      - DEVICE=cpu
      - LOG_LEVEL=INFO
    volumes:
      # Mount code for development (hot reload)
      - .:/app
      # Mount models directory
      - ./models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # Optional: Add monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

### Build & Deploy Script (Industry Standard)

```bash
#!/bin/bash
# build-and-deploy.sh
# Used by: CI/CD pipelines, deployment automation

set -e  # Exit on error

IMAGE_NAME="llm-serving"
VERSION="${1:-latest}"
REGISTRY="your-registry.com"  # Docker Hub, ECR, GCR, etc.

echo "Building image: ${IMAGE_NAME}:${VERSION}"

# Build image
docker build -t ${IMAGE_NAME}:${VERSION} .

# Tag for registry
docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:${VERSION}
docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:latest

# Push to registry
docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}
docker push ${REGISTRY}/${IMAGE_NAME}:latest

echo "Image pushed: ${REGISTRY}/${IMAGE_NAME}:${VERSION}"

# Deploy (example for K8s)
# kubectl set image deployment/llm-serving llm-serving=${REGISTRY}/${IMAGE_NAME}:${VERSION}
```

### GPU Dockerfile (Industry Standard)

```dockerfile
# GPU-enabled Dockerfile
# Used by: Production ML serving, high-performance inference
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python3", "app.py"]
```

### Usage in Production

```bash
# Build for production
docker build -t llm-serving:prod -f Dockerfile.prod .

# Run with GPU
docker run --gpus all -p 8000:8000 llm-serving:prod

# Run with resource limits
docker run \
  --memory="4g" \
  --cpus="2.0" \
  -p 8000:8000 \
  llm-serving:prod

# Run in production (with restart policy)
docker run -d \
  --name llm-serving \
  --restart unless-stopped \
  -p 8000:8000 \
  llm-serving:prod
```

## Key Concepts

### Dockerfile Basics
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

### Multi-Stage Builds
Split build into stages to reduce final image size:
1. **Builder stage**: Install dependencies, download models
2. **Runtime stage**: Copy only what's needed

### Layer Caching
Order matters! Put frequently changing files last:
```dockerfile
# Good: Dependencies change less often
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .  # Code changes more often
```

## Dockerfile Examples

### Basic Dockerfile
See `Dockerfile.basic` for a simple example.

### Optimized Dockerfile
See `Dockerfile.optimized` for:
- Multi-stage build
- Model caching
- Smaller image size
- Better layer caching

### GPU Dockerfile
See `Dockerfile.gpu` for CUDA/GPU support.

## Building Images

### Basic Build
```bash
docker build -t llm-serving:latest .
```

### With Build Args
```bash
docker build \
  --build-arg MODEL_NAME=gpt2 \
  --build-arg PYTHON_VERSION=3.9 \
  -t llm-serving:gpt2 .
```

### Multi-stage Build
```bash
docker build -f Dockerfile.optimized -t llm-serving:optimized .
```

## Running Containers

### CPU Mode
```bash
docker run -p 8000:8000 llm-serving:latest
```

### GPU Mode (NVIDIA Docker)
```bash
docker run --gpus all -p 8000:8000 llm-serving:gpu
```

### With Environment Variables
```bash
docker run \
  -p 8000:8000 \
  -e MODEL_NAME=gpt2 \
  -e MAX_LENGTH=100 \
  llm-serving:latest
```

### With Volume Mounts (for models)
```bash
docker run \
  -p 8000:8000 \
  -v ./models:/app/models \
  llm-serving:latest
```

## Docker Compose

For multi-container setups, use `docker-compose.yml`:
- Application container
- Model storage
- Monitoring
- Load balancer

## Image Optimization

### Reduce Image Size
1. **Use slim base images**: `python:3.9-slim` instead of `python:3.9`
2. **Multi-stage builds**: Don't include build tools in final image
3. **Remove cache**: `pip install --no-cache-dir`
4. **Combine RUN commands**: Fewer layers = smaller image

### Speed Up Builds
1. **Layer caching**: Order Dockerfile commands by change frequency
2. **Build cache**: Use `--cache-from` for CI/CD
3. **Parallel builds**: Build multiple images simultaneously

## Handling Large Models

### Option 1: Download in Image
```dockerfile
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('gpt2')"
```
**Pros**: Self-contained
**Cons**: Large image, slow builds

### Option 2: Volume Mount
```bash
docker run -v ./models:/app/models ...
```
**Pros**: Small image, fast builds
**Cons**: Need models on host

### Option 3: Model Registry
Download from S3/GCS at runtime
**Pros**: Flexible, versioned
**Cons**: Requires network access

## Best Practices

1. **Use .dockerignore**: Exclude unnecessary files
2. **Tag images**: Use semantic versioning
3. **Health checks**: Add HEALTHCHECK instruction
4. **Non-root user**: Run as non-root for security
5. **Resource limits**: Set memory/CPU limits
6. **Logging**: Configure proper logging

## Exercises

1. **Build basic image**: Create Dockerfile for basic serving
2. **Optimize image**: Reduce image size by 50%
3. **Multi-stage build**: Create optimized multi-stage Dockerfile
4. **GPU support**: Add CUDA support to Dockerfile
5. **Docker Compose**: Create compose file for full stack

## Common Issues

### Out of Memory
- **Problem**: Container runs out of memory
- **Solution**: Increase Docker memory limit or use smaller model

### Slow Builds
- **Problem**: Building takes forever
- **Solution**: Use layer caching, download models in separate stage

### GPU Not Available
- **Problem**: `CUDA not available` in container
- **Solution**: Install nvidia-docker, use `--gpus all` flag

### Large Image Size
- **Problem**: Image is several GB
- **Solution**: Use multi-stage build, slim base images

## Next Steps

- **Topic 3**: Deploy containers to Kubernetes
- **Topic 4**: Load test your containerized app
- **Topic 5**: Optimize for production with vLLM

## Further Reading

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

