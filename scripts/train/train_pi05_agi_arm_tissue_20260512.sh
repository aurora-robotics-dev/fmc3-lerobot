#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."

# ============================================================================
# Common settings: edit these first
# ============================================================================

POLICY_TYPE="pi05"
TASK_VARIANT="${TASK_VARIANT:-black_to_yellow}"

POLICY_PATH="${POLICY_PATH:-/home/phl/workspace/models/pi05_base}"
PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/home/phl/workspace/models/paligemma-tokenizer}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/phl/workspace/mymodels/agi_arm_bot}"
CONDA_ENV="${CONDA_ENV:-lerobot-pi0}"

STEPS="${STEPS:-60000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-50}"

# ============================================================================
# LoRA / PEFT settings (set LORA_ENABLE=false to do full fine-tuning)
# ============================================================================

LORA_ENABLE="${LORA_ENABLE:-true}"
LORA_METHOD_TYPE="${LORA_METHOD_TYPE:-LORA}"
LORA_R="${LORA_R:-16}"
# Leave LORA_TARGET_MODULES empty to use PI0.5's built-in default.
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"
LORA_FULL_TRAINING_MODULES="${LORA_FULL_TRAINING_MODULES:-}"

WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-agi_arm_pi05}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"

DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ============================================================================
# PI0.5 fine-tuning settings
# ============================================================================

MAX_STATE_DIM="${MAX_STATE_DIM:-32}"
MAX_ACTION_DIM="${MAX_ACTION_DIM:-32}"

COMPILE_MODEL="${COMPILE_MODEL:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
DTYPE="${DTYPE:-bfloat16}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
USE_TACTILE="${USE_TACTILE:-false}"

# ============================================================================
# Less common settings
# ============================================================================

DATASET_STREAMING="${DATASET_STREAMING:-false}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"

# Pass extra lerobot-train args after the script command.
# Examples:
#   ./scripts/train/train_pi05_agi_arm_tissue_20260512.sh --dry-run
#   TASK_VARIANT=yellow_to_black ./scripts/train/train_pi05_agi_arm_tissue_20260512.sh
#   ./scripts/train/train_pi05_agi_arm_tissue_20260512.sh --task-variant yellow_to_black --steps=30000

parse_cli_args() {
  DRY_RUN=false
  EXTRA_ARGS=()

  while (($# > 0)); do
    case "$1" in
      --dry-run)
        DRY_RUN=true
        ;;
      --task-variant)
        shift
        if (($# == 0)); then
          echo "--task-variant requires a value: black_to_yellow or yellow_to_black" >&2
          exit 1
        fi
        TASK_VARIANT="$1"
        ;;
      --task-variant=*)
        TASK_VARIANT="${1#*=}"
        ;;
      --black-to-yellow)
        TASK_VARIANT="black_to_yellow"
        ;;
      --yellow-to-black)
        TASK_VARIANT="yellow_to_black"
        ;;
      *)
        EXTRA_ARGS+=("$1")
        ;;
    esac
    shift
  done
}

configure_task_variant() {
  local dataset_base="/home/phl/workspace/dataset/Robot/agi_arm_bot/without_tactile"

  case "${TASK_VARIANT}" in
    black_to_yellow)
      DEFAULT_DATASET_ROOT="${dataset_base}/use_the_right_arm_to_move_the_tissue_from_the_black_paper_to_the_yellow_paper_20260512"
      DEFAULT_DATASET_REPO_ID="local/agi_arm_tissue_black_to_yellow_20260512"
      DEFAULT_RUN_NAME="pi05_agi_arm_tissue_black_to_yellow_20260512_lora"
      DEFAULT_TASK_TEXT="Use the right arm to move the tissue from the black paper to the yellow paper."
      ;;
    yellow_to_black)
      DEFAULT_DATASET_ROOT="${dataset_base}/use_the_right_arm_to_move_the_tissue_from_the_yellow_paper_to_the_black_paper_20260512"
      DEFAULT_DATASET_REPO_ID="local/agi_arm_tissue_yellow_to_black_20260512"
      DEFAULT_RUN_NAME="pi05_agi_arm_tissue_yellow_to_black_20260512_lora"
      DEFAULT_TASK_TEXT="Use the right arm to move the tissue from the yellow paper to the black paper."
      ;;
    *)
      echo "Unknown TASK_VARIANT='${TASK_VARIANT}'. Expected black_to_yellow or yellow_to_black." >&2
      exit 1
      ;;
  esac

  DATASET_ROOT="${DATASET_ROOT:-${DEFAULT_DATASET_ROOT}}"
  DATASET_REPO_ID="${DATASET_REPO_ID:-${DEFAULT_DATASET_REPO_ID}}"
  RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
  JOB_NAME="${JOB_NAME:-${RUN_NAME}}"
  TASK_TEXT="${TASK_TEXT:-${DEFAULT_TASK_TEXT}}"
}

require_file() {
  local path="$1"
  local description="$2"

  if [[ ! -f "${path}" ]]; then
    echo "${description} not found: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local description="$2"

  if [[ ! -d "${path}" ]]; then
    echo "${description} not found: ${path}" >&2
    exit 1
  fi
}

load_dataset_info() {
  eval "$(
    DATASET_ROOT="${DATASET_ROOT}" python - <<'PY'
import json
import os
import shlex
from pathlib import Path

root = Path(os.environ["DATASET_ROOT"])
info = json.loads((root / "meta" / "info.json").read_text())
features = info["features"]

values = {
    "DATASET_NAME": root.name,
    "ROBOT_TYPE": info.get("robot_type", "unknown"),
    "TOTAL_EPISODES": info.get("total_episodes", 0),
    "TOTAL_FRAMES": info.get("total_frames", 0),
    "TOTAL_TASKS": info.get("total_tasks", 0),
    "DATASET_STATE_DIM": features["observation.state"]["shape"][0],
    "DATASET_ACTION_DIM": features["action"]["shape"][0],
    "DATASET_IMAGE_KEYS": ",".join(
        sorted(key for key in features if key.startswith("observation.images."))
    ),
}

for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
  )"
}

ensure_task_text() {
  TASKS_PATH="${DATASET_ROOT}/meta/tasks.parquet"

  if [[ "${FIX_TASKS_METADATA:-true}" != "true" ]]; then
    return 0
  fi

  TASKS_PATH="${TASKS_PATH}" TASK_TEXT="${TASK_TEXT}" DRY_RUN="${DRY_RUN}" python - <<'PY'
import os
from pathlib import Path

import pandas as pd

path = Path(os.environ["TASKS_PATH"])
task_text = os.environ["TASK_TEXT"]
dry_run = os.environ["DRY_RUN"] == "true"

tasks = pd.read_parquet(path)
if len(tasks) != 1:
    raise SystemExit(f"Expected exactly one task row in {path}, got {len(tasks)}")

current_task = tasks.index[0]
if current_task == task_text:
    raise SystemExit(0)

task_index = int(tasks.iloc[0]["task_index"])
if dry_run:
    print(f"Would update {path} task text: {current_task!r} -> {task_text!r}")
    raise SystemExit(0)

fixed = pd.DataFrame({"task_index": [task_index]}, index=pd.Index([task_text]))
fixed.to_parquet(path)
print(f"Updated {path} task text: {current_task!r} -> {task_text!r}")
PY
}

validate_setup() {
  if [[ "${POLICY_TYPE}" != "pi05" ]]; then
    echo "This script is configured for PI0.5 training. Got POLICY_TYPE='${POLICY_TYPE}'." >&2
    exit 1
  fi

  if [[ -z "${DATASET_IMAGE_KEYS}" ]]; then
    echo "Dataset does not contain any observation.images.* features." >&2
    exit 1
  fi

  if (( DATASET_STATE_DIM > MAX_STATE_DIM )); then
    echo "Dataset state dim ${DATASET_STATE_DIM} exceeds PI0.5 max_state_dim ${MAX_STATE_DIM}." >&2
    exit 1
  fi

  if (( DATASET_ACTION_DIM > MAX_ACTION_DIM )); then
    echo "Dataset action dim ${DATASET_ACTION_DIM} exceeds PI0.5 max_action_dim ${MAX_ACTION_DIM}." >&2
    exit 1
  fi

  mkdir -p "${OUTPUT_BASE}" "${OUTPUT_BASE}/_logs"

  if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output directory already exists: ${OUTPUT_DIR}" >&2
    echo "Change RUN_NAME before starting a new training run." >&2
    exit 1
  fi
}

add_train_arg() {
  local key="$1"
  local value="$2"
  TRAIN_CMD+=("--${key}=${value}")
}

build_train_command() {
  TRAIN_CMD=(
    conda run
    --no-capture-output
    -n "${CONDA_ENV}"
    lerobot-train
  )

  add_train_arg "dataset.repo_id" "${DATASET_REPO_ID}"
  add_train_arg "dataset.root" "${DATASET_ROOT}"
  add_train_arg "dataset.streaming" "${DATASET_STREAMING}"
  add_train_arg "dataset.video_backend" "${VIDEO_BACKEND}"

  add_train_arg "policy.path" "${POLICY_PATH}"
  add_train_arg "policy.input_features" "null"
  add_train_arg "policy.output_features" "null"
  add_train_arg "policy.compile_model" "${COMPILE_MODEL}"
  add_train_arg "policy.gradient_checkpointing" "${GRADIENT_CHECKPOINTING}"
  add_train_arg "policy.dtype" "${DTYPE}"
  add_train_arg "policy.freeze_vision_encoder" "${FREEZE_VISION_ENCODER}"
  add_train_arg "policy.train_expert_only" "${TRAIN_EXPERT_ONLY}"
  add_train_arg "policy.max_state_dim" "${MAX_STATE_DIM}"
  add_train_arg "policy.max_action_dim" "${MAX_ACTION_DIM}"
  add_train_arg "policy.use_tactile" "${USE_TACTILE}"
  add_train_arg "policy.paligemma_tokenizer_name" "${PALIGEMMA_TOKENIZER}"
  add_train_arg "policy.device" "${DEVICE}"
  add_train_arg "policy.push_to_hub" "false"

  add_train_arg "output_dir" "${OUTPUT_DIR}"
  add_train_arg "job_name" "${JOB_NAME}"

  add_train_arg "steps" "${STEPS}"
  add_train_arg "save_freq" "${SAVE_FREQ}"
  add_train_arg "log_freq" "${LOG_FREQ}"
  add_train_arg "num_workers" "${NUM_WORKERS}"
  add_train_arg "batch_size" "${BATCH_SIZE}"

  add_train_arg "wandb.enable" "${WANDB_ENABLE}"
  add_train_arg "wandb.disable_artifact" "${WANDB_DISABLE_ARTIFACT}"
  add_train_arg "wandb.project" "${WANDB_PROJECT}"

  if [[ "${LORA_ENABLE}" == "true" ]]; then
    add_train_arg "peft.method_type" "${LORA_METHOD_TYPE}"
    add_train_arg "peft.r" "${LORA_R}"
    if [[ -n "${LORA_TARGET_MODULES}" ]]; then
      add_train_arg "peft.target_modules" "${LORA_TARGET_MODULES}"
    fi
    if [[ -n "${LORA_FULL_TRAINING_MODULES}" ]]; then
      add_train_arg "peft.full_training_modules" "[${LORA_FULL_TRAINING_MODULES}]"
    fi
  fi

  if (( ${#EXTRA_ARGS[@]} > 0 )); then
    TRAIN_CMD+=("${EXTRA_ARGS[@]}")
  fi
}

print_summary() {
  echo "=========================================="
  echo "PI0.5 LoRA fine-tuning summary"
  echo "=========================================="
  echo "Task variant        : ${TASK_VARIANT}"
  echo "Task text           : ${TASK_TEXT}"
  echo "Policy type         : ${POLICY_TYPE}"
  echo "Base policy path    : ${POLICY_PATH}"
  echo "Tokenizer path      : ${PALIGEMMA_TOKENIZER}"
  echo "Dataset root        : ${DATASET_ROOT}"
  echo "Dataset repo id     : ${DATASET_REPO_ID}"
  echo "Dataset name        : ${DATASET_NAME}"
  echo "Robot type          : ${ROBOT_TYPE}"
  echo "Episodes / frames   : ${TOTAL_EPISODES} / ${TOTAL_FRAMES}"
  echo "Task count          : ${TOTAL_TASKS}"
  echo "Dataset image keys  : ${DATASET_IMAGE_KEYS}"
  echo "Dataset dims        : state=${DATASET_STATE_DIM}, action=${DATASET_ACTION_DIM}"
  echo "PI0.5 max dims      : state=${MAX_STATE_DIM}, action=${MAX_ACTION_DIM}"
  echo "Fine-tune options   : dtype=${DTYPE}, gradient_checkpointing=${GRADIENT_CHECKPOINTING}, tactile=${USE_TACTILE}"
  if [[ "${LORA_ENABLE}" == "true" ]]; then
    echo "LoRA                : method=${LORA_METHOD_TYPE}, r=${LORA_R}, target_modules=${LORA_TARGET_MODULES:-<pi05-default>}, full_training_modules=${LORA_FULL_TRAINING_MODULES:-<pi05-default>}"
  else
    echo "LoRA                : disabled (full fine-tuning)"
  fi
  echo "Output dir          : ${OUTPUT_DIR}"
  echo "Log file            : ${LOG_FILE}"
  echo "=========================================="
  echo "Policy features are inferred from dataset metadata."
  echo "Use --dry-run to inspect the final lerobot-train command."
  echo "=========================================="
}

run_training() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run mode enabled. Command to execute:"
    printf ' %q' "${TRAIN_CMD[@]}"
    echo
    return 0
  fi

  env \
    LEROBOT_PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
}

main() {
  parse_cli_args "$@"
  configure_task_variant

  DATASET_INFO_PATH="${DATASET_ROOT}/meta/info.json"
  DATASET_TASKS_PATH="${DATASET_ROOT}/meta/tasks.parquet"
  POLICY_CONFIG_PATH="${POLICY_PATH}/config.json"
  TOKENIZER_CONFIG_PATH="${PALIGEMMA_TOKENIZER}/tokenizer_config.json"
  OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"
  LOG_FILE="${OUTPUT_BASE}/_logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

  require_file "${DATASET_INFO_PATH}" "Dataset metadata"
  require_file "${DATASET_TASKS_PATH}" "Dataset tasks metadata"
  require_dir "${POLICY_PATH}" "Base PI0.5 policy directory"
  require_file "${POLICY_CONFIG_PATH}" "Base PI0.5 policy config"
  require_dir "${PALIGEMMA_TOKENIZER}" "PaliGemma tokenizer directory"
  require_file "${TOKENIZER_CONFIG_PATH}" "PaliGemma tokenizer config"

  ensure_task_text
  load_dataset_info
  validate_setup
  build_train_command
  print_summary
  run_training
}

main "$@"
