#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."

# ============================================================================
# Common settings: edit these first
# ============================================================================

POLICY_TYPE="act"

DATASET_ROOT="/home/phl/workspace/dataset/Robot/agi_arm_bot/right_hand_picks_up_the_camera_left_hand_picks_up_the_pen_then_left_hand_uses_the_pen_to_touch_merged_all/camera_pen_touch_clean_del_52_376"
DATASET_REPO_ID="local/camera_pen_touch_clean_del_52_376"
OUTPUT_BASE="/home/phl/workspace/mymodels/agi_arm_bot"

RUN_NAME="act_camera_pen_touch_clean_del_52_376_chunk50_te001"
JOB_NAME="${RUN_NAME}"
CONDA_ENV="lerobot-pi0"

STEPS=90000  # about 5.5 epochs for this cleaned dataset with batch_size=16
BATCH_SIZE=16
SAVE_FREQ=5000
LOG_FREQ=50

WANDB_ENABLE=true
WANDB_PROJECT="agi_arm_act"
WANDB_DISABLE_ARTIFACT=true

DEVICE="cuda"
NUM_WORKERS=4
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ============================================================================
# ACT settings
# ============================================================================

N_OBS_STEPS=1
CHUNK_SIZE=50
N_ACTION_STEPS=1

VISION_BACKBONE="resnet18"
TEMPORAL_ENSEMBLE_COEFF="0.01"

OPTIMIZER_LR="1e-5"
OPTIMIZER_LR_BACKBONE="1e-5"
OPTIMIZER_WEIGHT_DECAY="1e-4"
KL_WEIGHT="10.0"
DROPOUT="0.1"

# ============================================================================
# Less common settings
# ============================================================================

DATASET_STREAMING=false
VIDEO_BACKEND="torchcodec"

# Pass extra lerobot-train args after the script command.
# Example:
#   ./scripts/train/train_act_agi_arm_pick_and_place.sh --policy.optimizer_lr=5e-6

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
  if [[ "${POLICY_TYPE}" != "act" ]]; then
    echo "This script is configured for ACT training. Got POLICY_TYPE='${POLICY_TYPE}'." >&2
    exit 1
  fi

  if [[ -z "${DATASET_IMAGE_KEYS}" ]]; then
    echo "Dataset does not contain any observation.images.* features." >&2
    exit 1
  fi

  if (( N_OBS_STEPS != 1 )); then
    echo "ACT in this repo currently expects n_obs_steps=1. Got ${N_OBS_STEPS}." >&2
    exit 1
  fi

  if (( N_ACTION_STEPS > CHUNK_SIZE )); then
    echo "Invalid ACT rollout setup: n_action_steps=${N_ACTION_STEPS}, chunk_size=${CHUNK_SIZE}." >&2
    echo "Require: n_action_steps <= chunk_size" >&2
    exit 1
  fi

  if [[ -n "${TEMPORAL_ENSEMBLE_COEFF}" && "${TEMPORAL_ENSEMBLE_COEFF}" != "null" ]] && (( N_ACTION_STEPS != 1 )); then
    echo "ACT temporal ensembling requires n_action_steps=1. Got ${N_ACTION_STEPS}." >&2
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

  add_train_arg "policy.type" "${POLICY_TYPE}"
  add_train_arg "policy.device" "${DEVICE}"
  add_train_arg "policy.input_features" "null"
  add_train_arg "policy.output_features" "null"
  add_train_arg "policy.n_obs_steps" "${N_OBS_STEPS}"
  add_train_arg "policy.chunk_size" "${CHUNK_SIZE}"
  add_train_arg "policy.n_action_steps" "${N_ACTION_STEPS}"
  add_train_arg "policy.vision_backbone" "${VISION_BACKBONE}"
  add_train_arg "policy.temporal_ensemble_coeff" "${TEMPORAL_ENSEMBLE_COEFF}"
  add_train_arg "policy.optimizer_lr" "${OPTIMIZER_LR}"
  add_train_arg "policy.optimizer_lr_backbone" "${OPTIMIZER_LR_BACKBONE}"
  add_train_arg "policy.optimizer_weight_decay" "${OPTIMIZER_WEIGHT_DECAY}"
  add_train_arg "policy.kl_weight" "${KL_WEIGHT}"
  add_train_arg "policy.dropout" "${DROPOUT}"
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

  if (( ${#EXTRA_ARGS[@]} > 0 )); then
    TRAIN_CMD+=("${EXTRA_ARGS[@]}")
  fi
}

print_summary() {
  echo "=========================================="
  echo "ACT training summary"
  echo "=========================================="
  echo "Policy type         : ${POLICY_TYPE}"
  echo "Dataset root        : ${DATASET_ROOT}"
  echo "Dataset repo id     : ${DATASET_REPO_ID}"
  echo "Dataset name        : ${DATASET_NAME}"
  echo "Robot type          : ${ROBOT_TYPE}"
  echo "Episodes / frames   : ${TOTAL_EPISODES} / ${TOTAL_FRAMES}"
  echo "Task count          : ${TOTAL_TASKS}"
  echo "Dataset image keys  : ${DATASET_IMAGE_KEYS}"
  echo "Dataset dims        : state=${DATASET_STATE_DIM}, action=${DATASET_ACTION_DIM}"
  echo "ACT rollout         : n_obs=${N_OBS_STEPS}, chunk=${CHUNK_SIZE}, n_action=${N_ACTION_STEPS}"
  echo "Temporal ensemble   : ${TEMPORAL_ENSEMBLE_COEFF}"
  echo "Vision backbone     : ${VISION_BACKBONE}"
  echo "Optimizer           : lr=${OPTIMIZER_LR}, lr_backbone=${OPTIMIZER_LR_BACKBONE}, wd=${OPTIMIZER_WEIGHT_DECAY}"
  echo "Loss regularization : kl_weight=${KL_WEIGHT}, dropout=${DROPOUT}"
  echo "Output dir          : ${OUTPUT_DIR}"
  echo "Log file            : ${LOG_FILE}"
  echo "=========================================="
  echo "Policy features are inferred from dataset metadata."
  echo "ACT does not use language task text as training input in this script."
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
  OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"
  LOG_FILE="${OUTPUT_BASE}/_logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

  require_file "${DATASET_INFO_PATH}" "Dataset metadata"
  load_dataset_info
  validate_setup
  build_train_command
  print_summary
  run_training
}

main "$@"
