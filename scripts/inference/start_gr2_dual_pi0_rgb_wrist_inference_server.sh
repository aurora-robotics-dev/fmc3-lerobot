#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/phl/workspace/lerobot-versions/lerobot"
CONDA_ENV="${CONDA_ENV:-lerobot-pi0}"

UNIX_SOCKET_PATH="${UNIX_SOCKET_PATH:-/tmp/gr2_dual_pi0_rgb_wrist.sock}"
ROBOT_NAME="${ROBOT_NAME:-gr2}"
DOMAIN_ID="${DOMAIN_ID:-123}"
ROBOT_TYPE="${ROBOT_TYPE:-fourier_gr2}"

TAKE_OUT_CHECKPOINT_PATH="${TAKE_OUT_CHECKPOINT_PATH:-/home/phl/workspace/mymodels/gr2/pi0_gr2_grab_bottle_from_box_to_desk_rgb_3_cam/checkpoints/035000/pretrained_model}"
TAKE_OUT_TASK="${TAKE_OUT_TASK:-take the bottle out of the box and place it on the desk}"
PUT_IN_CHECKPOINT_PATH="${PUT_IN_CHECKPOINT_PATH:-/home/phl/workspace/mymodels/gr2/pi0_gr2_grab_bottle_desk_to_box_rgb_3_cam/checkpoints/050000/pretrained_model}"
PUT_IN_TASK="${PUT_IN_TASK:-pick up the bottle from the grid cell and place it into the box}"

FPS="${FPS:-15}"
FSM_STATE="${FSM_STATE:-11}"
CAMERA_KEY="${CAMERA_KEY:-observation.images.camera_top}"
DEVICE="${DEVICE:-auto}"
DRY_RUN="${DRY_RUN:-0}"
MOVE_TO_INIT_POSE_ON_START="${MOVE_TO_INIT_POSE_ON_START:-0}"
CLEAN_OLD="${CLEAN_OLD:-1}"

WRIST_LEFT_SERIAL="${WRIST_LEFT_SERIAL:-420222072816}"
WRIST_RIGHT_SERIAL="${WRIST_RIGHT_SERIAL:-349522072801}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

cd "${ROOT_DIR}"

if [[ -z "${TAKE_OUT_CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] TAKE_OUT_CHECKPOINT_PATH is required."
  exit 1
fi
if [[ -z "${PUT_IN_CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] PUT_IN_CHECKPOINT_PATH is required."
  exit 1
fi
if [[ -z "${TAKE_OUT_TASK}" ]]; then
  echo "[ERROR] TAKE_OUT_TASK is required."
  exit 1
fi
if [[ -z "${PUT_IN_TASK}" ]]; then
  echo "[ERROR] PUT_IN_TASK is required."
  exit 1
fi

if [[ ! -e "${TAKE_OUT_CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] take-out checkpoint not found: ${TAKE_OUT_CHECKPOINT_PATH}"
  exit 1
fi
if [[ ! -e "${PUT_IN_CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] put-in checkpoint not found: ${PUT_IN_CHECKPOINT_PATH}"
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
  stop_pids "python scripts/inference/gr2_dual_pi0_rgb_wrist_inference_server.py" "dual PI0 service"
fi

if [[ -S "${UNIX_SOCKET_PATH}" || -e "${UNIX_SOCKET_PATH}" ]]; then
  echo "[INFO] Removing stale socket: ${UNIX_SOCKET_PATH}"
  rm -f "${UNIX_SOCKET_PATH}"
fi

EXTRA_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi
if [[ "${MOVE_TO_INIT_POSE_ON_START}" == "1" ]]; then
  EXTRA_ARGS+=(--move-to-init-pose-on-start)
fi

set -x
exec conda run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/inference/gr2_dual_pi0_rgb_wrist_inference_server.py \
  --unix-socket-path "${UNIX_SOCKET_PATH}" \
  --take-out-checkpoint-path "${TAKE_OUT_CHECKPOINT_PATH}" \
  --take-out-task "${TAKE_OUT_TASK}" \
  --put-in-checkpoint-path "${PUT_IN_CHECKPOINT_PATH}" \
  --put-in-task "${PUT_IN_TASK}" \
  --robot-type "${ROBOT_TYPE}" \
  --robot-name "${ROBOT_NAME}" \
  --domain-id "${DOMAIN_ID}" \
  --fps "${FPS}" \
  --fsm-state "${FSM_STATE}" \
  --camera-key "${CAMERA_KEY}" \
  --device "${DEVICE}" \
  --wrist-left-serial "${WRIST_LEFT_SERIAL}" \
  --wrist-right-serial "${WRIST_RIGHT_SERIAL}" \
  "${EXTRA_ARGS[@]}"
