#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."

# ============================================================================
# Common settings: edit these first
# ============================================================================

POLICY_TYPE="pi0"

DATASET_ROOT="/home/phl/workspace/dataset/Robot/agi_arm_bot/right_hand_picks_up_the_camera_left_hand_picks_up_the_pen_then_left_hand_uses_the_pen_to_touch_20260427_merged"
DATASET_REPO_ID="local/agi_arm_camera_pen_touch_20260427"
POLICY_PATH="/home/phl/workspace/models/agi_arm_bot/lerobot/pi0_base"
OUTPUT_BASE="/home/phl/workspace/mymodels/agi_arm_bot"

RUN_NAME="pi0_right_hand_camera_left_hand_pen_touch_20260427_lora"
JOB_NAME="${RUN_NAME}"
CONDA_ENV="lerobot-pi0"

STEPS=90000  # about 5.1 epochs for this dataset with batch_size=8
BATCH_SIZE=8
SAVE_FREQ=5000
LOG_FREQ=50

# ============================================================================
# LoRA / PEFT settings (set LORA_ENABLE=false to do full fine-tuning)
# ============================================================================

LORA_ENABLE=true
LORA_METHOD_TYPE="LORA"
LORA_R=16
# Leave LORA_TARGET_MODULES empty to use PI0's built-in default
# (gemma_expert q/v + state/action projections, see modeling_pi0.py).
# To override, set a regex string, e.g.
#   LORA_TARGET_MODULES='(.*\.gemma_expert\..*\.self_attn\.(q|k|v|o)_proj|model\.(state_proj|action_in_proj|action_out_proj))'
LORA_TARGET_MODULES=""
# Modules that stay fully trainable (saved alongside the adapter).
# PI0 defaults to [] when LoRA is enabled. Override with a comma-separated list, e.g.
#   LORA_FULL_TRAINING_MODULES="state_proj,action_out_proj"
LORA_FULL_TRAINING_MODULES=""

WANDB_ENABLE=true
WANDB_PROJECT="agi_arm_pi0"
WANDB_DISABLE_ARTIFACT=true

DEVICE="cuda"
NUM_WORKERS=4
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ============================================================================
# PI0 fine-tuning settings
# ============================================================================

MAX_STATE_DIM=32
MAX_ACTION_DIM=32

COMPILE_MODEL=false
GRADIENT_CHECKPOINTING=true
DTYPE="bfloat16"
FREEZE_VISION_ENCODER=false
TRAIN_EXPERT_ONLY=false

# ============================================================================
# Less common settings
# ============================================================================

DATASET_STREAMING=false
VIDEO_BACKEND="torchcodec"

# Pass extra lerobot-train args after the script command.
# Example:
#   ./scripts/train/train_pi0_agi_arm_camera_pen_touch_20260427.sh --policy.optimizer_lr=1e-5

parse_cli_args() {
  DRY_RUN=false
  EXTRA_ARGS=()

  for arg in "$@"; do
    case "$arg" in
      --dry-run)
        DRY_RUN=true
        ;;
      *)
        EXTRA_ARGS+=("$arg")
        ;;
    esac
  done
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

validate_setup() {
  if [[ "${POLICY_TYPE}" != "pi0" ]]; then
    echo "This script is configured for PI0 training. Got POLICY_TYPE='${POLICY_TYPE}'." >&2
    exit 1
  fi

  if [[ -z "${DATASET_IMAGE_KEYS}" ]]; then
    echo "Dataset does not contain any observation.images.* features." >&2
    exit 1
  fi

  if (( DATASET_STATE_DIM > MAX_STATE_DIM )); then
    echo "Dataset state dim ${DATASET_STATE_DIM} exceeds PI0 max_state_dim ${MAX_STATE_DIM}." >&2
    exit 1
  fi

  if (( DATASET_ACTION_DIM > MAX_ACTION_DIM )); then
    echo "Dataset action dim ${DATASET_ACTION_DIM} exceeds PI0 max_action_dim ${MAX_ACTION_DIM}." >&2
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
  echo "PI0 fine-tuning summary"
  echo "=========================================="
  echo "Policy type         : ${POLICY_TYPE}"
  echo "Base policy path    : ${POLICY_PATH}"
  echo "Dataset root        : ${DATASET_ROOT}"
  echo "Dataset repo id     : ${DATASET_REPO_ID}"
  echo "Dataset name        : ${DATASET_NAME}"
  echo "Robot type          : ${ROBOT_TYPE}"
  echo "Episodes / frames   : ${TOTAL_EPISODES} / ${TOTAL_FRAMES}"
  echo "Task count          : ${TOTAL_TASKS}"
  echo "Dataset image keys  : ${DATASET_IMAGE_KEYS}"
  echo "Dataset dims        : state=${DATASET_STATE_DIM}, action=${DATASET_ACTION_DIM}"
  echo "PI0 max dims        : state=${MAX_STATE_DIM}, action=${MAX_ACTION_DIM}"
  echo "Fine-tune options   : dtype=${DTYPE}, gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
  if [[ "${LORA_ENABLE}" == "true" ]]; then
    echo "LoRA                : method=${LORA_METHOD_TYPE}, r=${LORA_R}, target_modules=${LORA_TARGET_MODULES:-<pi0-default>}, full_training_modules=${LORA_FULL_TRAINING_MODULES:-<pi0-default>}"
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

  env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
}

main() {
  parse_cli_args "$@"

  DATASET_INFO_PATH="${DATASET_ROOT}/meta/info.json"
  POLICY_CONFIG_PATH="${POLICY_PATH}/config.json"
  OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"
  LOG_FILE="${OUTPUT_BASE}/_logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

  require_file "${DATASET_INFO_PATH}" "Dataset metadata"
  require_dir "${POLICY_PATH}" "Base PI0 policy directory"
  require_file "${POLICY_CONFIG_PATH}" "Base PI0 policy config"
  load_dataset_info
  validate_setup
  build_train_command
  print_summary
  run_training
}

main "$@"
