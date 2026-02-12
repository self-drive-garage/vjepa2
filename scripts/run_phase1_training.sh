#!/usr/bin/env bash
# Run Phase 1 (minADE) V-JEPA2-Drive training as defined in PLAN_VJEPA2_E2E_AD.md.
# The script wraps app/main.py, rewrites the driving config with the selected
# datasets/checkpoints, and launches distributed training on the requested GPUs.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/run_phase1_training.sh [options]

Options:
  -c, --config FILE          Base YAML config (default: configs/train/vitg16/driving-joint-256px-16f.yaml)
  -o, --output-root DIR      Root directory for run folders (default: <repo>/logs/vjepa_drive)
  -n, --run-name NAME        Name of this run (default: phase1-YYYYmmdd-HHMMSS)
  -g, --gpus LIST            Comma-separated GPU identifiers (e.g. 0,1 or cuda:0,cuda:1; default: 0)
  -d, --datasets SPECS       Comma-separated dataset specs name:path (default: nvidia_av:<repo>/data/nvidia_av/train.csv)
  -p, --pretrained CKPT      Path to pretrained V-JEPA2 checkpoint to initialize from
  -b, --batch-size N         Override batch size in config
  -w, --num-workers N        Override DataLoader worker count (disables pin_mem when set)
  -h, --help                 Show this message and exit

Examples:
  scripts/run_phase1_training.sh \\
      --gpus 0,1,2,3 \\
      --datasets \"nuscenes:/mnt/nuscenes/train.csv,waymo:/mnt/waymo/train.csv\" \\
      --pretrained /checkpoints/vit_giant.pt
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_TEMPLATE="${PROJECT_ROOT}/configs/train/vitg16/driving-joint-256px-16f.yaml"
OUTPUT_ROOT="${PROJECT_ROOT}/logs/vjepa_drive"
RUN_NAME="phase1-$(date +%Y%m%d-%H%M%S)"
GPU_STRING="0"
DATASET_SPECS="nvidia_av:${PROJECT_ROOT}/data/nvidia_av/train.csv"
PRETRAIN_CKPT=""
BATCH_SIZE_OVERRIDE=""
NUM_WORKERS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG_TEMPLATE="$(realpath "$2")"
            shift 2
            ;;
        -o|--output-root)
            OUTPUT_ROOT="$(realpath "$2")"
            shift 2
            ;;
        -n|--run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        -g|--gpus)
            GPU_STRING="$2"
            shift 2
            ;;
        -d|--datasets)
            DATASET_SPECS="$2"
            shift 2
            ;;
        -p|--pretrained)
            PRETRAIN_CKPT="$(realpath "$2")"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE_OVERRIDE="$2"
            shift 2
            ;;
        -w|--num-workers)
            NUM_WORKERS_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ! -f "$CONFIG_TEMPLATE" ]]; then
    echo "Config file not found: $CONFIG_TEMPLATE" >&2
    exit 1
fi

if [[ -n "$PRETRAIN_CKPT" && ! -f "$PRETRAIN_CKPT" ]]; then
    echo "Pretrained checkpoint not found: $PRETRAIN_CKPT" >&2
    exit 1
fi

IFS=',' read -ra DATASET_ARRAY <<< "$DATASET_SPECS"
if [[ ${#DATASET_ARRAY[@]} -eq 0 ]]; then
    echo "At least one dataset spec must be provided." >&2
    exit 1
fi

declare -a DATASET_CANONICAL=()
for spec in "${DATASET_ARRAY[@]}"; do
    if [[ "$spec" != *:* ]]; then
        echo "Dataset spec must be name:path -> '$spec'" >&2
        exit 1
    fi
    name="${spec%%:*}"
    path="${spec#*:}"
    if [[ -z "$name" || -z "$path" ]]; then
        echo "Dataset spec must include both name and path -> '$spec'" >&2
        exit 1
    fi
    if [[ ! -f "$path" ]]; then
        echo "Dataset CSV does not exist: $path" >&2
        exit 1
    fi
    abs_path="$(realpath "$path")"
    DATASET_CANONICAL+=("${name}:${abs_path}")
done

IFS=',' read -ra GPU_ARRAY <<< "$GPU_STRING"
declare -a DEVICE_ARGS=()
for gpu in "${GPU_ARRAY[@]}"; do
    trimmed="$(echo "$gpu" | xargs)"
    [[ -z "$trimmed" ]] && continue
    if [[ "$trimmed" == cpu ]]; then
        DEVICE_ARGS+=("cpu")
    elif [[ "$trimmed" == cuda:* ]]; then
        DEVICE_ARGS+=("$trimmed")
    else
        DEVICE_ARGS+=("cuda:${trimmed}")
    fi
done

if [[ ${#DEVICE_ARGS[@]} -eq 0 ]]; then
    echo "No valid GPUs provided." >&2
    exit 1
fi

RUN_FOLDER="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "$RUN_FOLDER"

RUN_CONFIG_PATH="${RUN_FOLDER}/phase1_config.yaml"
DATASET_SPEC_JOINED="$(IFS=','; printf '%s' "${DATASET_CANONICAL[*]}")"

export PHASE1_CONFIG_TEMPLATE="$CONFIG_TEMPLATE"
export PHASE1_CONFIG_OUTPUT="$RUN_CONFIG_PATH"
export PHASE1_RUN_FOLDER="$RUN_FOLDER"
export PHASE1_DATASETS="$DATASET_SPEC_JOINED"
export PHASE1_PRETRAIN="${PRETRAIN_CKPT}"
export PHASE1_BATCH_SIZE="${BATCH_SIZE_OVERRIDE}"
export PHASE1_NUM_WORKERS="${NUM_WORKERS_OVERRIDE}"

python3 - <<'PY'
import os
import pathlib

import yaml

config_template = pathlib.Path(os.environ["PHASE1_CONFIG_TEMPLATE"])
config_output = pathlib.Path(os.environ["PHASE1_CONFIG_OUTPUT"])
run_folder = os.environ["PHASE1_RUN_FOLDER"]
dataset_specs = [spec for spec in os.environ["PHASE1_DATASETS"].split(",") if spec]
pretrained = os.environ.get("PHASE1_PRETRAIN", "")
batch_override = os.environ.get("PHASE1_BATCH_SIZE")
workers_override = os.environ.get("PHASE1_NUM_WORKERS")

cfg = yaml.safe_load(config_template.read_text())
cfg["folder"] = run_folder

dataset_names = []
dataset_paths = []
for spec in dataset_specs:
    if ":" not in spec:
        raise SystemExit(f"Dataset spec missing ':' -> {spec}")
    name, path = spec.split(":", 1)
    dataset_names.append(name)
    dataset_paths.append(path)

num_datasets = len(dataset_paths)
weights = [round(1.0 / num_datasets, 5) for _ in dataset_paths]

cfg.setdefault("data", {})
cfg["data"]["datasets"] = dataset_paths
cfg["data"]["dataset_names"] = dataset_names
cfg["data"]["datasets_weights"] = weights
if batch_override:
    cfg["data"]["batch_size"] = int(batch_override)
if workers_override:
    cfg["data"]["num_workers"] = int(workers_override)
    cfg["data"]["persistent_workers"] = False
    cfg["data"]["pin_mem"] = False

cfg.setdefault("meta", {})
if pretrained:
    cfg["meta"]["load_checkpoint"] = True
    cfg["meta"]["read_checkpoint"] = pretrained
else:
    cfg["meta"]["load_checkpoint"] = False
    cfg["meta"]["read_checkpoint"] = None

config_output.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

echo "Launching V-JEPA2 Phase 1 training"
echo "  Config template : $CONFIG_TEMPLATE"
echo "  Run config      : $RUN_CONFIG_PATH"
echo "  Run folder      : $RUN_FOLDER"
echo "  Datasets        :"
for spec in "${DATASET_CANONICAL[@]}"; do
    echo "    - $spec"
done
[[ -n "$PRETRAIN_CKPT" ]] && echo "  Pretrained ckpt : $PRETRAIN_CKPT"
echo "  Devices         : ${DEVICE_ARGS[*]}"

cd "$PROJECT_ROOT"
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp
PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" python3 -m app.main --fname "$RUN_CONFIG_PATH" --devices "${DEVICE_ARGS[@]}"
