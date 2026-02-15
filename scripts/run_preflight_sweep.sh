#!/usr/bin/env bash
# Run short phase-1 batch-size preflight jobs and record pass/fail results.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUNNER="${PROJECT_ROOT}/scripts/run_phase1_training.sh"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/train/vitg16/driving-joint-256px-16f-8xa100.yaml}"
DATASET_SPECS="${DATASET_SPECS:-nvidia_av:${PROJECT_ROOT}/data/nvidia_av/train.csv}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-${PROJECT_ROOT}/checkpoints/vitl.pt}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
BATCH_SIZES="${BATCH_SIZES:-4 5 6 7 8}"
EPOCHS="${EPOCHS:-1}"
IPE="${IPE:-20}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1800}"
READY_POLL_SEC="${READY_POLL_SEC:-15}"
SESSION_TAG="${SESSION_TAG:-$(date +%Y%m%d-%H%M%S)}"

SWEEP_DIR="${PROJECT_ROOT}/logs/vjepa_drive/sweeps/${SESSION_TAG}"
SWEEP_LOG="${SWEEP_DIR}/preflight_sweep.log"
RESULTS_CSV="${SWEEP_DIR}/results.csv"
RECOMMENDATION_FILE="${SWEEP_DIR}/recommended_batch_size.txt"

mkdir -p "${SWEEP_DIR}"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${SWEEP_LOG}"
}

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "${path}" ]]; then
        log "ERROR: missing ${label}: ${path}"
        exit 1
    fi
}

wait_for_fabric_manager() {
    local deadline=$((SECONDS + READY_TIMEOUT_SEC))
    local state="unknown"

    while (( SECONDS < deadline )); do
        state="$(systemctl is-active nvidia-fabricmanager 2>/dev/null || true)"
        if [[ "${state}" == "active" ]]; then
            log "nvidia-fabricmanager is active."
            return 0
        fi
        log "Waiting for nvidia-fabricmanager (state=${state})."
        sleep "${READY_POLL_SEC}"
    done

    log "ERROR: nvidia-fabricmanager did not become active before timeout."
    systemctl --no-pager --full status nvidia-fabricmanager 2>/dev/null | tail -n 40 | tee -a "${SWEEP_LOG}" || true
    return 1
}

check_cuda_once() {
    log "Checking CUDA visibility with nvidia-smi."
    if ! timeout 20 nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader >>"${SWEEP_LOG}" 2>&1; then
        log "ERROR: nvidia-smi failed or timed out."
        return 1
    fi

    log "Checking CUDA visibility with torch."
    if ! timeout 20 python3 - <<'PY' >>"${SWEEP_LOG}" 2>&1
import torch
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit(1)
PY
    then
        log "ERROR: torch could not see CUDA devices."
        return 1
    fi

    return 0
}

classify_failure() {
    local console_log="$1"
    if [[ ! -f "${console_log}" ]]; then
        echo "fail"
        return
    fi
    if rg -q -i "CUDA out of memory|CUDNN_STATUS_ALLOC_FAILED|device-side assert" "${console_log}"; then
        echo "oom"
        return
    fi
    if rg -q -i "CUDA is unavailable|no GPUs found|cudaGetDeviceCount" "${console_log}"; then
        echo "cuda_unavailable"
        return
    fi
    echo "fail"
}

main() {
    require_file "${RUNNER}" "runner script"
    require_file "${CONFIG_PATH}" "config"
    require_file "${PRETRAINED_CKPT}" "checkpoint"

    log "Starting preflight sweep."
    log "Sweep dir: ${SWEEP_DIR}"
    log "Batch sizes: ${BATCH_SIZES}"
    log "Runner: ${RUNNER}"
    log "Config: ${CONFIG_PATH}"
    log "Datasets: ${DATASET_SPECS}"
    log "Checkpoint: ${PRETRAINED_CKPT}"

    echo "batch_size,status,exit_code,start_utc,end_utc,run_name,run_dir" > "${RESULTS_CSV}"

    wait_for_fabric_manager
    check_cuda_once

    local best_bs=""
    local status="unknown"
    local rc=0

    for bs in ${BATCH_SIZES}; do
        local start_utc end_utc run_name run_dir console_log
        start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        run_name="preflight-bs${bs}-${SESSION_TAG}"
        run_dir="${PROJECT_ROOT}/logs/vjepa_drive/${run_name}"
        console_log="${run_dir}/console.log"

        log "Launching batch_size=${bs} run_name=${run_name}"
        set +e
        "${RUNNER}" \
            --config "${CONFIG_PATH}" \
            --run-name "${run_name}" \
            --gpus "${GPU_LIST}" \
            --datasets "${DATASET_SPECS}" \
            --pretrained "${PRETRAINED_CKPT}" \
            --epochs "${EPOCHS}" \
            --ipe "${IPE}" \
            --batch-size "${bs}" \
            --backend nccl
        rc=$?
        set -e

        end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        if [[ ${rc} -eq 0 ]]; then
            status="ok"
            best_bs="${bs}"
            log "PASS batch_size=${bs}"
        else
            status="$(classify_failure "${console_log}")"
            log "FAIL batch_size=${bs} status=${status} exit_code=${rc}"
        fi

        printf "%s,%s,%s,%s,%s,%s,%s\n" \
            "${bs}" "${status}" "${rc}" "${start_utc}" "${end_utc}" "${run_name}" "${run_dir}" \
            >> "${RESULTS_CSV}"

        if [[ "${status}" == "oom" ]]; then
            log "Stopping sweep after OOM at batch_size=${bs}."
            break
        fi
        if [[ "${status}" == "cuda_unavailable" ]]; then
            log "Stopping sweep because CUDA became unavailable."
            break
        fi
    done

    if [[ -n "${best_bs}" ]]; then
        echo "${best_bs}" > "${RECOMMENDATION_FILE}"
        log "Recommended batch size: ${best_bs}"
    else
        echo "none" > "${RECOMMENDATION_FILE}"
        log "No successful batch size found."
    fi

    log "Sweep finished. Results: ${RESULTS_CSV}"
}

main "$@"
