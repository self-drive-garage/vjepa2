#!/usr/bin/env bash
# Launch V-JEPA2-Drive Phase 1 with torchrun for 1-node or multi-node training.
# Run this script on every node with the same --run-name/--master-addr/--master-port.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/run_phase1_training_multinode.sh [options]

Core options:
  -c, --config FILE          Base YAML config (default: tuned 8xA100 config)
  -o, --output-root DIR      Root directory for run folders (default: <repo>/logs/vjepa_drive)
  -n, --run-name NAME        Run folder name (required for --nnodes > 1)
  -g, --gpus LIST            Local GPU ids on this node (default: 0,1,2,3,4,5,6,7)
  -d, --datasets SPECS       Comma-separated dataset specs name:path
                             (default: nvidia_av:<repo>/data/nvidia_av/train.csv)
  -p, --pretrained CKPT      Optional pretrained checkpoint path
  -e, --epochs N             Override total epochs in config
  --ipe N                    Override iterations-per-epoch
  -b, --batch-size N         Override per-process batch size
  -w, --num-workers N        Override DataLoader workers (disables pin_mem when set)

Distributed options:
  --nnodes N                 Number of nodes (default: 1)
  --node-rank N              Node rank in [0, nnodes-1] (default: 0)
  --master-addr HOST         Rendezvous host/IP (required for --nnodes > 1)
  --master-port PORT         Rendezvous port (default: 37129)
  --backend NAME             Process-group backend nccl|gloo|mpi (default: nccl)

Logging options:
  --tail                     Force tail of console log after launch
  --no-tail                  Do not tail console log after launch
  -h, --help                 Show this message and exit

Examples:
  # Single node (8 GPUs)
  scripts/run_phase1_training_multinode.sh --gpus 0,1,2,3,4,5,6,7

  # Two nodes (run once per node with different --node-rank)
  scripts/run_phase1_training_multinode.sh \
      --nnodes 2 --node-rank 0 \
      --master-addr 10.1.0.12 --master-port 37129 \
      --run-name phase1-2node-test
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_ACTIVATE="${PROJECT_ROOT}/../.venv/bin/activate"

if [[ -f "$VENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
else
    echo "WARNING: venv activate script not found at $VENV_ACTIVATE; using current Python environment."
fi

CONFIG_TEMPLATE="${PROJECT_ROOT}/configs/train/vitg16/driving-joint-256px-16f-8xa100-tuned.yaml"
OUTPUT_ROOT="${PROJECT_ROOT}/logs/vjepa_drive"
RUN_NAME="phase1-$(date +%Y%m%d-%H%M%S)"
RUN_NAME_USER_SET=0
GPU_STRING="0,1,2,3,4,5,6,7"
DATASET_SPECS="nvidia_av:${PROJECT_ROOT}/data/nvidia_av/train.csv"
PRETRAIN_CKPT=""
EPOCHS_OVERRIDE=""
IPE_OVERRIDE=""
BATCH_SIZE_OVERRIDE=""
NUM_WORKERS_OVERRIDE=""
DIST_BACKEND="${TORCH_DISTRIBUTED_BACKEND:-nccl}"
NNODES=1
NODE_RANK=0
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-37129}"
TAIL_MODE="auto"

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
            RUN_NAME_USER_SET=1
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
        -e|--epochs)
            EPOCHS_OVERRIDE="$2"
            shift 2
            ;;
        --ipe)
            IPE_OVERRIDE="$2"
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
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --backend)
            DIST_BACKEND="$2"
            shift 2
            ;;
        --tail)
            TAIL_MODE="yes"
            shift
            ;;
        --no-tail)
            TAIL_MODE="no"
            shift
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

if ! [[ "$NNODES" =~ ^[0-9]+$ ]] || [[ "$NNODES" -lt 1 ]]; then
    echo "--nnodes must be an integer >= 1." >&2
    exit 1
fi
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || [[ "$NODE_RANK" -lt 0 ]] || [[ "$NODE_RANK" -ge "$NNODES" ]]; then
    echo "--node-rank must be an integer in [0, nnodes-1]." >&2
    exit 1
fi

DIST_BACKEND="$(echo "$DIST_BACKEND" | tr '[:upper:]' '[:lower:]')"
case "$DIST_BACKEND" in
    nccl|gloo|mpi)
        ;;
    *)
        echo "Unsupported distributed backend '$DIST_BACKEND' (expected nccl|gloo|mpi)." >&2
        exit 1
        ;;
esac

if [[ "$NNODES" -gt 1 && "$RUN_NAME_USER_SET" -eq 0 ]]; then
    echo "For multi-node runs, provide an explicit --run-name shared by all nodes." >&2
    exit 1
fi

if [[ "$NNODES" -gt 1 && -z "$MASTER_ADDR" ]]; then
    echo "For multi-node runs, --master-addr is required." >&2
    exit 1
fi
if [[ -z "$MASTER_ADDR" ]]; then
    MASTER_ADDR="127.0.0.1"
fi

IFS=',' read -ra DATASET_ARRAY <<< "$DATASET_SPECS"
if [[ ${#DATASET_ARRAY[@]} -eq 0 ]]; then
    echo "At least one dataset spec must be provided." >&2
    exit 1
fi

declare -a DATASET_CANONICAL=()
declare -a DATASET_ROWS=()
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
    row_count="$(wc -l < "$abs_path" | xargs)"
    DATASET_ROWS+=("${row_count}")
done

IFS=',' read -ra GPU_ARRAY <<< "$GPU_STRING"
declare -a GPU_IDS=()
for gpu in "${GPU_ARRAY[@]}"; do
    trimmed="$(echo "$gpu" | xargs)"
    [[ -z "$trimmed" ]] && continue
    if [[ "$trimmed" == cpu ]]; then
        echo "CPU execution is not supported by this launcher." >&2
        exit 1
    elif [[ "$trimmed" == cuda:* ]]; then
        GPU_IDS+=("${trimmed#cuda:}")
    else
        GPU_IDS+=("$trimmed")
    fi
done

if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "No valid GPUs provided." >&2
    exit 1
fi

NPROC_PER_NODE="${#GPU_IDS[@]}"
CUDA_VISIBLE_DEVICES_VALUE="$(IFS=','; printf '%s' "${GPU_IDS[*]}")"

RUN_FOLDER="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "$RUN_FOLDER"
CONSOLE_LOG="${RUN_FOLDER}/console.node${NODE_RANK}.log"
PID_FILE="${RUN_FOLDER}/train.node${NODE_RANK}.pid"
RUN_CONFIG_PATH="${RUN_FOLDER}/phase1_config.node${NODE_RANK}.yaml"
DATASET_SPEC_JOINED="$(IFS=','; printf '%s' "${DATASET_CANONICAL[*]}")"

export PHASE1_CONFIG_TEMPLATE="$CONFIG_TEMPLATE"
export PHASE1_CONFIG_OUTPUT="$RUN_CONFIG_PATH"
export PHASE1_RUN_FOLDER="$RUN_FOLDER"
export PHASE1_DATASETS="$DATASET_SPEC_JOINED"
export PHASE1_PRETRAIN="${PRETRAIN_CKPT}"
export PHASE1_EPOCHS="${EPOCHS_OVERRIDE}"
export PHASE1_IPE="${IPE_OVERRIDE}"
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
epochs_override = os.environ.get("PHASE1_EPOCHS")
ipe_override = os.environ.get("PHASE1_IPE")
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
if epochs_override:
    cfg.setdefault("optimization", {})
    cfg["optimization"]["epochs"] = int(epochs_override)
if ipe_override:
    cfg.setdefault("optimization", {})
    cfg["optimization"]["ipe"] = int(ipe_override)

cfg.setdefault("meta", {})
if pretrained:
    cfg["meta"]["load_checkpoint"] = True
    cfg["meta"]["read_checkpoint"] = pretrained
else:
    cfg["meta"]["load_checkpoint"] = False
    cfg["meta"]["read_checkpoint"] = None

config_output.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

echo "Launching V-JEPA2 Phase 1 training (torchrun)"
echo "  Config template : $CONFIG_TEMPLATE"
echo "  Run config      : $RUN_CONFIG_PATH"
echo "  Run folder      : $RUN_FOLDER"
echo "  Datasets        :"
for idx in "${!DATASET_CANONICAL[@]}"; do
    spec="${DATASET_CANONICAL[$idx]}"
    rows="${DATASET_ROWS[$idx]}"
    echo "    - $spec (${rows} rows)"
    if [[ "$rows" -lt 1000 ]]; then
        echo "      WARNING: small dataset detected (<1000 rows)."
    fi
done
[[ -n "$PRETRAIN_CKPT" ]] && echo "  Pretrained ckpt : $PRETRAIN_CKPT"
echo "  GPUs (local)    : $CUDA_VISIBLE_DEVICES_VALUE"
echo "  Backend         : $DIST_BACKEND"
echo "  Nodes           : $NNODES"
echo "  Node rank       : $NODE_RANK"
echo "  Master addr     : $MASTER_ADDR"
echo "  Master port     : $MASTER_PORT"
echo "  Console log     : $CONSOLE_LOG"

cd "$PROJECT_ROOT"
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp
export TORCH_DISTRIBUTED_BACKEND="$DIST_BACKEND"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${HOME}/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${HOME}/.cache/triton}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"

TORCHRUN_CMD=(
    python3 -m torch.distributed.run
    --nnodes "$NNODES"
    --nproc_per_node "$NPROC_PER_NODE"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
    --max_restarts 0
    -m app.main_torchrun
    --fname "$RUN_CONFIG_PATH"
)

nohup "${TORCHRUN_CMD[@]}" > "$CONSOLE_LOG" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo "  PID file        : $PID_FILE"
echo "  Train PID       : $TRAIN_PID"

DO_TAIL=0
if [[ "$TAIL_MODE" == "yes" ]]; then
    DO_TAIL=1
elif [[ "$TAIL_MODE" == "auto" && "$NODE_RANK" -eq 0 ]]; then
    DO_TAIL=1
fi

if [[ "$DO_TAIL" -eq 1 ]]; then
    echo "Tailing console log (Ctrl-C stops tail only; training keeps running)..."
    sleep 1
    tail -n 200 -f "$CONSOLE_LOG"
else
    echo "Not tailing log (use: tail -n 200 -f $CONSOLE_LOG)"
fi
