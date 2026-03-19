#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/inference/infer_lerobot_generic.sh [options] [-- extra lerobot-record args]

Purpose:
  Run a trained LeRobot policy on a real robot through `lerobot-record`.
  This is the common "inference on robot + optionally record eval episodes" workflow.

Examples:

  Run a local pi0 checkpoint on SO101 and save 10 eval episodes:
    bash scripts/inference/infer_lerobot_generic.sh \
      --policy-path /home/phl/workspace/mymodels/pi0_step30000/030000/pretrained_model \
      --robot-type so101_follower \
      --robot-port /dev/ttyACM2 \
      --robot-id fmc3_robotics_follower_arm \
      --robot-cameras '{"top":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30}}' \
      --dataset-repo-id puheliang/pi0_lerobot_fmc3_gr2_grab_box_v2_eval \
      --single-task "Pick up the tape and place it in the box" \
      --num-episodes 10 \
      --episode-time-s 90 \
      --reset-time-s 3 \
      --display-data true \
      --policy-device cuda \
      --policy-dtype bfloat16

  Run policy inference and keep teleop connected for resets between episodes:
    bash scripts/inference/infer_lerobot_generic.sh \
      --policy-path /path/to/pretrained_model \
      --robot-type so101_follower \
      --robot-port /dev/ttyACM2 \
      --robot-id fmc3_robotics_follower_arm \
      --robot-cameras '{"top":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30}}' \
      --teleop-type so101_leader \
      --teleop-port /dev/ttyACM1 \
      --teleop-id fmc3_robotics_leader_arm \
      --dataset-repo-id local/pi0_eval_demo \
      --single-task "Pick up the tape and place it in the box"

  Preview command only:
    bash scripts/inference/infer_lerobot_generic.sh \
      --policy-path /path/to/pretrained_model \
      --robot-type so101_follower \
      --robot-port /dev/ttyACM2 \
      --robot-id fmc3_robotics_follower_arm \
      --robot-cameras '{"top":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30}}' \
      --dataset-repo-id local/pi0_eval_demo \
      --single-task "Pick up the tape and place it in the box" \
      --dry-run

Options:
  --policy-path <path_or_hf_id>     Local pretrained_model dir or Hub model id.
  --robot-type <type>               Robot type, e.g. so101_follower.
  --robot-port <port>               Robot serial port for simple single-arm setups.
  --robot-id <id>                   Robot id used by calibration files.
  --robot-cameras <json>            Robot camera config JSON string.
  --teleop-type <type>              Optional teleop type, e.g. so101_leader.
  --teleop-port <port>              Optional teleop port for simple single-arm setups.
  --teleop-id <id>                  Optional teleop id.
  --dataset-repo-id <id>            Eval dataset name / logical id.
  --dataset-root <path>             Optional explicit dataset output directory.
  --single-task <text>              Task instruction text used during inference.
  --resume <true|false>             Continue writing into an existing eval dataset. Default: false
  --num-episodes <n>                Default: 10
  --episode-time-s <n>              Default: 90
  --reset-time-s <n>                Default: 0
  --fps <n>                         Default: 30
  --display-data <true|false>       Default: true
  --display-compressed-images <v>   Default: false
  --push-to-hub <true|false>        Default: false
  --policy-device <device>          Optional policy device override, e.g. cuda / cpu.
  --policy-dtype <dtype>            Optional policy dtype override, e.g. bfloat16.
  --play-sounds <true|false>        Default: true
  --conda-env <env>                 Conda env to run in. Default: lerobot
  --dry-run                         Print command only.
  -h, --help                        Show this help.

Notes:
  - For policy inference, the underlying command is `lerobot-record --policy.path=...`.
  - Any args after `--` are forwarded to `lerobot-record` unchanged.
  - For more complex robots, pass extra nested config flags after `--`.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot}"
POLICY_PATH=""
ROBOT_TYPE=""
ROBOT_PORT=""
ROBOT_ID=""
ROBOT_CAMERAS=""
TELEOP_TYPE=""
TELEOP_PORT=""
TELEOP_ID=""
DATASET_REPO_ID=""
DATASET_ROOT=""
SINGLE_TASK=""
RESUME="false"
NUM_EPISODES="10"
EPISODE_TIME_S="90"
RESET_TIME_S="0"
FPS="30"
DISPLAY_DATA="true"
DISPLAY_COMPRESSED_IMAGES="false"
PUSH_TO_HUB="false"
POLICY_DEVICE=""
POLICY_DTYPE=""
PLAY_SOUNDS="true"
DRY_RUN="false"

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path)
      POLICY_PATH="$2"
      shift 2
      ;;
    --robot-type)
      ROBOT_TYPE="$2"
      shift 2
      ;;
    --robot-port)
      ROBOT_PORT="$2"
      shift 2
      ;;
    --robot-id)
      ROBOT_ID="$2"
      shift 2
      ;;
    --robot-cameras)
      ROBOT_CAMERAS="$2"
      shift 2
      ;;
    --teleop-type)
      TELEOP_TYPE="$2"
      shift 2
      ;;
    --teleop-port)
      TELEOP_PORT="$2"
      shift 2
      ;;
    --teleop-id)
      TELEOP_ID="$2"
      shift 2
      ;;
    --dataset-repo-id)
      DATASET_REPO_ID="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --single-task)
      SINGLE_TASK="$2"
      shift 2
      ;;
    --resume)
      RESUME="$2"
      shift 2
      ;;
    --num-episodes)
      NUM_EPISODES="$2"
      shift 2
      ;;
    --episode-time-s)
      EPISODE_TIME_S="$2"
      shift 2
      ;;
    --reset-time-s)
      RESET_TIME_S="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --display-data)
      DISPLAY_DATA="$2"
      shift 2
      ;;
    --display-compressed-images)
      DISPLAY_COMPRESSED_IMAGES="$2"
      shift 2
      ;;
    --push-to-hub)
      PUSH_TO_HUB="$2"
      shift 2
      ;;
    --policy-device)
      POLICY_DEVICE="$2"
      shift 2
      ;;
    --policy-dtype)
      POLICY_DTYPE="$2"
      shift 2
      ;;
    --play-sounds)
      PLAY_SOUNDS="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      echo
      usage
      exit 1
      ;;
  esac
done

require_arg() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    echo "[ERROR] ${name} is required."
    exit 1
  fi
}

is_local_like_path() {
  local value="$1"
  [[ "${value}" == /* || "${value}" == ./* || "${value}" == ../* || "${value}" == ~* ]]
}

require_arg "--policy-path" "${POLICY_PATH}"
require_arg "--robot-type" "${ROBOT_TYPE}"
require_arg "--robot-id" "${ROBOT_ID}"
require_arg "--robot-cameras" "${ROBOT_CAMERAS}"
require_arg "--dataset-repo-id" "${DATASET_REPO_ID}"
require_arg "--single-task" "${SINGLE_TASK}"

if [[ -z "${ROBOT_PORT}" ]]; then
  echo "[WARN] --robot-port is empty. This is okay only if you pass robot port config through extra args."
fi

if [[ -n "${TELEOP_PORT}" || -n "${TELEOP_ID}" ]]; then
  require_arg "--teleop-type" "${TELEOP_TYPE}"
fi

if is_local_like_path "${POLICY_PATH}" && [[ ! -e "${POLICY_PATH}" ]]; then
  echo "[ERROR] Local policy path not found: ${POLICY_PATH}"
  exit 1
fi

if [[ -n "${DATASET_ROOT}" ]]; then
  TARGET_DATASET_DIR="${DATASET_ROOT}"
else
  TARGET_DATASET_DIR="${HOME}/.cache/huggingface/lerobot/${DATASET_REPO_ID}"
fi

if [[ "${RESUME}" == "true" ]]; then
  if [[ ! -d "${TARGET_DATASET_DIR}" ]]; then
    echo "[ERROR] --resume=true but dataset dir does not exist: ${TARGET_DATASET_DIR}"
    exit 1
  fi
else
  # Avoid the known failure path where dataset creation crashes if the dir already exists.
  if [[ -e "${TARGET_DATASET_DIR}" ]]; then
    echo "[ERROR] Target dataset dir already exists: ${TARGET_DATASET_DIR}"
    echo "[ERROR] Use one of the following:"
    echo "  1. Change --dataset-repo-id"
    echo "  2. Add --resume true"
    echo "  3. Delete the old eval dataset dir first"
    exit 1
  fi
fi

LOG_DIR="${REPO_ROOT}/outputs/inference_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/infer_$(date +%Y%m%d_%H%M%S).log"

cd "${REPO_ROOT}"

echo "[INFO] repo_root=${REPO_ROOT}"
echo "[INFO] env=${CONDA_DEFAULT_ENV:-base} target_env=${CONDA_ENV}"
echo "[INFO] policy_path=${POLICY_PATH}"
echo "[INFO] robot_type=${ROBOT_TYPE}"
echo "[INFO] robot_port=${ROBOT_PORT:-<via_extra_args>}"
echo "[INFO] robot_id=${ROBOT_ID}"
echo "[INFO] teleop_type=${TELEOP_TYPE:-<none>}"
echo "[INFO] teleop_port=${TELEOP_PORT:-<none>}"
echo "[INFO] teleop_id=${TELEOP_ID:-<none>}"
echo "[INFO] dataset_repo_id=${DATASET_REPO_ID}"
echo "[INFO] dataset_dir=${TARGET_DATASET_DIR}"
echo "[INFO] resume=${RESUME}"
echo "[INFO] log_file=${LOG_FILE}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
  RECORD_CMD=(lerobot-record)
else
  RECORD_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" lerobot-record)
fi

RECORD_ARGS=(
  "--robot.type=${ROBOT_TYPE}"
  "--robot.id=${ROBOT_ID}"
  "--robot.cameras=${ROBOT_CAMERAS}"
  "--policy.path=${POLICY_PATH}"
  "--display_data=${DISPLAY_DATA}"
  "--display_compressed_images=${DISPLAY_COMPRESSED_IMAGES}"
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.single_task=${SINGLE_TASK}"
  "--dataset.num_episodes=${NUM_EPISODES}"
  "--dataset.episode_time_s=${EPISODE_TIME_S}"
  "--dataset.reset_time_s=${RESET_TIME_S}"
  "--dataset.fps=${FPS}"
  "--dataset.push_to_hub=${PUSH_TO_HUB}"
  "--resume=${RESUME}"
  "--play_sounds=${PLAY_SOUNDS}"
)

if [[ -n "${ROBOT_PORT}" ]]; then
  RECORD_ARGS+=("--robot.port=${ROBOT_PORT}")
fi

if [[ -n "${DATASET_ROOT}" ]]; then
  RECORD_ARGS+=("--dataset.root=${DATASET_ROOT}")
fi

if [[ -n "${TELEOP_TYPE}" ]]; then
  RECORD_ARGS+=("--teleop.type=${TELEOP_TYPE}")
fi

if [[ -n "${TELEOP_PORT}" ]]; then
  RECORD_ARGS+=("--teleop.port=${TELEOP_PORT}")
fi

if [[ -n "${TELEOP_ID}" ]]; then
  RECORD_ARGS+=("--teleop.id=${TELEOP_ID}")
fi

if [[ -n "${POLICY_DEVICE}" ]]; then
  RECORD_ARGS+=("--policy.device=${POLICY_DEVICE}")
fi

if [[ -n "${POLICY_DTYPE}" ]]; then
  RECORD_ARGS+=("--policy.dtype=${POLICY_DTYPE}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  RECORD_ARGS+=("${EXTRA_ARGS[@]}")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[INFO] DRY_RUN=true, command preview:"
  printf '%q ' PYTHONUNBUFFERED=1 stdbuf -oL -eL "${RECORD_CMD[@]}" "${RECORD_ARGS[@]}"
  echo
  exit 0
fi

set -x
PYTHONUNBUFFERED=1 stdbuf -oL -eL "${RECORD_CMD[@]}" "${RECORD_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
