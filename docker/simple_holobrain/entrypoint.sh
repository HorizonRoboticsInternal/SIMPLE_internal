#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-shell}" == "shell" ]]; then
    exec /bin/bash
fi

exec "$@"
