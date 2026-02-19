#!/usr/bin/env bash
# Convenience wrapper: run the existing single-node launcher with the tuned config.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TUNED_CONFIG="${PROJECT_ROOT}/configs/train/vitg16/driving-joint-256px-16f-8xa100-tuned.yaml"

exec "${SCRIPT_DIR}/run_phase1_training.sh" --config "$TUNED_CONFIG" "$@"
