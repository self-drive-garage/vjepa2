#!/usr/bin/env bash
# Start the preflight sweep in a detached tmux session.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SESSION_NAME="${SESSION_NAME:-vjepa-preflight}"
SWEEP_ROOT="${PROJECT_ROOT}/logs/vjepa_drive/sweeps"
LAUNCH_LOG="${SWEEP_ROOT}/launcher.log"

mkdir -p "${SWEEP_ROOT}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required but not installed." | tee -a "${LAUNCH_LOG}"
    exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session '${SESSION_NAME}' already exists." | tee -a "${LAUNCH_LOG}"
    exit 0
fi

SESSION_TAG="$(date +%Y%m%d-%H%M%S)"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting tmux session ${SESSION_NAME} (tag=${SESSION_TAG})" | tee -a "${LAUNCH_LOG}"

tmux new-session -d -s "${SESSION_NAME}" \
    "cd '${PROJECT_ROOT}'; SESSION_TAG='${SESSION_TAG}' bash '${PROJECT_ROOT}/scripts/run_preflight_sweep.sh'; rc=\$?; echo \"[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] sweep exited rc=\$rc\" | tee -a '${LAUNCH_LOG}'; exec bash"

echo "tmux session '${SESSION_NAME}' started." | tee -a "${LAUNCH_LOG}"
