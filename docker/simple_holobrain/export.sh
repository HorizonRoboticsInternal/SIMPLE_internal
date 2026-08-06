#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE_NAME=${IMAGE_NAME:-simple-holobrain:latest}
ARCHIVE=${1:-"$DEPLOY_DIR/simple-holobrain.tar.zst"}

command -v zstd >/dev/null
docker image inspect "$IMAGE_NAME" >/dev/null

echo "Exporting $IMAGE_NAME to $ARCHIVE"
docker save "$IMAGE_NAME" | zstd -T0 -10 -o "$ARCHIVE"
echo "Model weights and evaluation data are volumes and are not included."
