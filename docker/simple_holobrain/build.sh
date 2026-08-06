#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROBOTICS_ROOT=$(cd -- "$DEPLOY_DIR/../../.." && pwd)

SIMPLE_BASE_IMAGE=${SIMPLE_BASE_IMAGE:-simple-holobrain-simple-base:latest}
IMAGE_NAME=${IMAGE_NAME:-simple-holobrain:latest}
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.6;12.0+PTX"}
SIMPLE_FULL_INSTALL=${SIMPLE_FULL_INSTALL:-0}

export DOCKER_BUILDKIT=1

if [[ "${REBUILD_SIMPLE_BASE:-0}" == 1 ]] || ! docker image inspect "$SIMPLE_BASE_IMAGE" >/dev/null 2>&1; then
    echo "Building SIMPLE base image: $SIMPLE_BASE_IMAGE"
    docker build \
        --network host \
        --build-arg SIMPLE_FULL_INSTALL="$SIMPLE_FULL_INSTALL" \
        --build-arg TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
        --tag "$SIMPLE_BASE_IMAGE" \
        --file "$ROBOTICS_ROOT/SIMPLE/Dockerfile" \
        "$ROBOTICS_ROOT/SIMPLE"
else
    echo "Reusing SIMPLE base image: $SIMPLE_BASE_IMAGE"
fi

echo "Building combined image: $IMAGE_NAME"
docker build \
    --network host \
    --build-arg SIMPLE_BASE_IMAGE="$SIMPLE_BASE_IMAGE" \
    --build-arg TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    --tag "$IMAGE_NAME" \
    --file "$DEPLOY_DIR/Dockerfile" \
    "$ROBOTICS_ROOT"

echo "Built $IMAGE_NAME"
