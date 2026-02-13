#!/usr/bin/env bash
set -euo pipefail

# Setup script for vjepa2 on a fresh host.
# Creates a local virtual environment, installs editable dependencies,
# and verifies required data/checkpoint directories exist.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HOME}/.hugging_face_token}"

echo "[vjepa2-setup] Repository root: ${REPO_ROOT}"

if [ ! -x "$(command -v "${PYTHON_BIN}")" ]; then
  echo "[vjepa2-setup] ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN to a valid interpreter." >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "[vjepa2-setup] Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "[vjepa2-setup] Reusing existing virtual environment at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo "[vjepa2-setup] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[vjepa2-setup] Installing vjepa2 (editable)"
python -m pip install -e "${REPO_ROOT}"

if [ -d "${REPO_ROOT}/physical_ai_av" ]; then
  echo "[vjepa2-setup] Installing physical_ai_av devkit (editable)"
  python -m pip install -e "${REPO_ROOT}/physical_ai_av"
fi

echo "[vjepa2-setup] Ensuring data/checkpoint directories exist"
mkdir -p "${REPO_ROOT}/data/nvidia_av/videos"
mkdir -p "${REPO_ROOT}/data/nvidia_av/egomotion"
mkdir -p "${REPO_ROOT}/checkpoints"
mkdir -p "${REPO_ROOT}/logs"

if [ ! -f "${HF_TOKEN_FILE}" ]; then
  cat <<EOF
[vjepa2-setup] WARNING: Hugging Face token not found at ${HF_TOKEN_FILE}.
Create this file with your token before running scripts/prepare_nvidia_av_data.py.
EOF
else
  echo "[vjepa2-setup] Hugging Face token found at ${HF_TOKEN_FILE}"
fi

cat <<'EOF'
[vjepa2-setup] Done.
Next steps:
  1) source .venv/bin/activate
  2) Run scripts/prepare_nvidia_av_data.py to download sample data
  3) Launch training via scripts/run_phase1_training.sh (see CODEX_ENVIRONMENT_SETUP.md)
EOF
