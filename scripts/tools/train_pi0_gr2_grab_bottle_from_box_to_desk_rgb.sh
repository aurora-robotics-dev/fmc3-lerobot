#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-lerobot-pi0}"
DATASET_ROOT="${DATASET_ROOT:-/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_from_box_to_desk_lerobot_ds_rgb}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/fmc3_gr2_grab_bottle_from_box_to_desk_rgb}"
POLICY_PATH="${POLICY_PATH:-/home/phl/workspace/models/pi0}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/phl/workspace/mymodels/gr2/pi0_gr2_grab_bottle_from_box_to_desk_rgb}"
JOB_NAME="${JOB_NAME:-pi0_gr2_grab_bottle_from_box_to_desk_rgb}"
RESUME_CONFIG_PATH="${RESUME_CONFIG_PATH:-}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-Lerobot_Phl_Project}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-50000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-50}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/tools/train_pi0_gr2_grab_bottle_from_box_to_desk_rgb.sh

Optional environment overrides:
  CONDA_ENV
  DATASET_ROOT
  DATASET_REPO_ID
  POLICY_PATH
  OUTPUT_DIR
  JOB_NAME
  RESUME_CONFIG_PATH
  WANDB_ENABLE
  WANDB_DISABLE_ARTIFACT
  WANDB_PROJECT
  BATCH_SIZE
  STEPS
  SAVE_FREQ
  LOG_FREQ
  VIDEO_BACKEND
  DRY_RUN
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -n "${RESUME_CONFIG_PATH}" ]]; then
  if [[ ! -f "${RESUME_CONFIG_PATH}" ]]; then
    echo "[ERROR] Resume config not found: ${RESUME_CONFIG_PATH}" >&2
    exit 1
  fi
  echo "[INFO] mode=resume"
  echo "[INFO] resume_config_path=${RESUME_CONFIG_PATH}"
else
  if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "[ERROR] Dataset root not found: ${DATASET_ROOT}" >&2
    exit 1
  fi

  if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "[ERROR] Missing dataset metadata: ${DATASET_ROOT}/meta/info.json" >&2
    exit 1
  fi

  if [[ ! -e "${POLICY_PATH}" ]]; then
    echo "[ERROR] Policy path not found: ${POLICY_PATH}" >&2
    exit 1
  fi

  if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "[ERROR] Output directory already exists: ${OUTPUT_DIR}" >&2
    echo "        Remove it first, override OUTPUT_DIR, or set RESUME_CONFIG_PATH." >&2
    exit 1
  fi

  echo "[INFO] mode=fresh"
  echo "[INFO] dataset_root=${DATASET_ROOT}"
  echo "[INFO] dataset_repo_id=${DATASET_REPO_ID}"
  echo "[INFO] policy_path=${POLICY_PATH}"
  echo "[INFO] output_dir=${OUTPUT_DIR}"
  echo "[INFO] job_name=${JOB_NAME}"
  echo "[INFO] policy_features=infer_from_dataset"
fi
echo "[INFO] wandb_enable=${WANDB_ENABLE}"
echo "[INFO] wandb_disable_artifact=${WANDB_DISABLE_ARTIFACT}"
if [[ "${WANDB_ENABLE}" == "true" ]]; then
  echo "[INFO] wandb_project=${WANDB_PROJECT}"
  echo "[INFO] Reminder: run 'wandb login' first if needed."
fi

if [[ -n "${RESUME_CONFIG_PATH}" ]]; then
  TRAIN_ARGS=(
    "--config_path=${RESUME_CONFIG_PATH}"
    "--resume=true"
    "--steps=${STEPS}"
    "--save_freq=${SAVE_FREQ}"
    "--log_freq=${LOG_FREQ}"
    "--wandb.enable=${WANDB_ENABLE}"
    "--wandb.disable_artifact=${WANDB_DISABLE_ARTIFACT}"
    "--batch_size=${BATCH_SIZE}"
  )
else
  TRAIN_ARGS=(
    "--dataset.repo_id=${DATASET_REPO_ID}"
    "--dataset.root=${DATASET_ROOT}"
    "--dataset.streaming=false"
    "--dataset.video_backend=${VIDEO_BACKEND}"
    "--policy.path=${POLICY_PATH}"
    "--policy.input_features=null"
    "--policy.output_features=null"
    "--policy.compile_model=false"
    "--policy.gradient_checkpointing=true"
    "--policy.dtype=bfloat16"
    "--policy.freeze_vision_encoder=false"
    "--policy.train_expert_only=false"
    "--policy.max_state_dim=45"
    "--policy.max_action_dim=35"
    "--policy.device=cuda"
    "--policy.push_to_hub=false"
    "--output_dir=${OUTPUT_DIR}"
    "--job_name=${JOB_NAME}"
    "--steps=${STEPS}"
    "--save_freq=${SAVE_FREQ}"
    "--log_freq=${LOG_FREQ}"
    "--wandb.enable=${WANDB_ENABLE}"
    "--wandb.disable_artifact=${WANDB_DISABLE_ARTIFACT}"
    "--batch_size=${BATCH_SIZE}"
  )
fi

if [[ "${WANDB_ENABLE}" == "true" ]]; then
  TRAIN_ARGS+=("--wandb.project=${WANDB_PROJECT}")
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
  TRAIN_CMD=(lerobot-train)
else
  TRAIN_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" lerobot-train)
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  printf '%q ' "${TRAIN_CMD[@]}" "${TRAIN_ARGS[@]}"
  echo
  exit 0
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
"${TRAIN_CMD[@]}" "${TRAIN_ARGS[@]}"
