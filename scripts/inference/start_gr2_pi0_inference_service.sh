#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/phl/workspace/lerobot-versions/lerobot"
CONDA_ENV="${CONDA_ENV:-lerobot-pi0}"

UNIX_SOCKET_PATH="${UNIX_SOCKET_PATH:-/tmp/gr2_pi0_inference_service.sock}"
ROBOT_NAME="${ROBOT_NAME:-gr2}"
DOMAIN_ID="${DOMAIN_ID:-123}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/phl/workspace/lerobot-versions/lerobot/outputs/train/pi0_gr2_pick_3_4_20260306_185911/checkpoints/070000/pretrained_model}"
TASK="${TASK:-pick bottle and place into box}"
ROBOT_TYPE="${ROBOT_TYPE:-fourier_gr2}"

FPS="${FPS:-15}"
FSM_STATE="${FSM_STATE:-11}"
CAMERA_KEY="${CAMERA_KEY:-observation.images.camera_top}"
DEVICE="${DEVICE:-auto}"
DRY_RUN="${DRY_RUN:-0}"
CLEAN_OLD="${CLEAN_OLD:-1}"
STOP_LEGACY_DEPLOY="${STOP_LEGACY_DEPLOY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

cd "${ROOT_DIR}"

if [[ ! -e "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

stop_pids() {
  local pattern="$1"
  local title="$2"
  mapfile -t pids < <(pgrep -f "${pattern}" || true)
  if [[ "${#pids[@]}" -eq 0 ]]; then
    return 0
  fi
  echo "[INFO] Stopping ${title} pids: ${pids[*]}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  sleep 1
  mapfile -t remain < <(pgrep -f "${pattern}" || true)
  if [[ "${#remain[@]}" -gt 0 ]]; then
    echo "[WARN] Force killing ${title} pids: ${remain[*]}"
    kill -KILL "${remain[@]}" 2>/dev/null || true
  fi
}

if [[ "${CLEAN_OLD}" == "1" ]]; then
  stop_pids "python scripts/gr2_pi0_inference_service.py" "PI0 service"
fi

if [[ "${STOP_LEGACY_DEPLOY}" == "1" ]]; then
  stop_pids "python scripts/deploy_gr2_pi0_rgbd.py" "legacy deploy"
fi

if [[ -S "${UNIX_SOCKET_PATH}" || -e "${UNIX_SOCKET_PATH}" ]]; then
  echo "[INFO] Removing stale socket: ${UNIX_SOCKET_PATH}"
  rm -f "${UNIX_SOCKET_PATH}"
fi

EXTRA_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

set -x
exec conda run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/gr2_pi0_inference_service.py \
  --unix-socket-path "${UNIX_SOCKET_PATH}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --task "${TASK}" \
  --robot-type "${ROBOT_TYPE}" \
  --robot-name "${ROBOT_NAME}" \
  --domain-id "${DOMAIN_ID}" \
  --fps "${FPS}" \
  --fsm-state "${FSM_STATE}" \
  --camera-key "${CAMERA_KEY}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"
