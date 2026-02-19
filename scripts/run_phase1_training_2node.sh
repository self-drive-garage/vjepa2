#!/usr/bin/env bash
# Convenience wrapper: launch the torchrun launcher in 2-node mode.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_phase1_training_multinode.sh" --nnodes 2 "$@"
