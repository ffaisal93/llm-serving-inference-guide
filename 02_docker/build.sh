#!/bin/bash
# Build script for LLM serving Docker images

set -e

IMAGE_NAME="llm-serving"
VERSION="${1:-latest}"

echo "Building Docker image: ${IMAGE_NAME}:${VERSION}"

# Build basic image
echo "Building basic image..."
docker build -f Dockerfile.basic -t ${IMAGE_NAME}:basic-${VERSION} .

# Build optimized image
echo "Building optimized image..."
docker build -f Dockerfile.optimized -t ${IMAGE_NAME}:optimized-${VERSION} .

# Build GPU image (if NVIDIA Docker available)
if command -v nvidia-docker &> /dev/null; then
    echo "Building GPU image..."
    docker build -f Dockerfile.gpu -t ${IMAGE_NAME}:gpu-${VERSION} .
else
    echo "NVIDIA Docker not available, skipping GPU image"
fi

echo "Build complete!"
echo "Images:"
docker images | grep ${IMAGE_NAME}

