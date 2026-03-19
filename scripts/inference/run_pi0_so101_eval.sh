#!/usr/bin/env bash
set -euo pipefail

# Pi0 evaluation script for the current SO101 setup.
#
# Default usage:
#   bash scripts/inference/run_pi0_so101_eval.sh
#
# Override task text:
#   TASK='Pick up the tape and place it in the box' \
#   bash scripts/inference/run_pi0_so101_eval.sh
#
# Override model path:
#   MODEL_PATH=/path/to/pretrained_model \
#   bash scripts/inference/run_pi0_so101_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/home/phl/workspace/mymodels/pi0_via_middle_finetune_20260310/checkpoints/last/pretrained_model}"

ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM2}"
ROBOT_ID="${ROBOT_ID:-fmc3_robotics_follower_arm}"

if [[ -z "${ROBOT_CAMERAS:-}" ]]; then
  ROBOT_CAMERAS='{"top":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30}}'
fi

TASK="${TASK:-Pick up the tape, place it at the middle waypoint, then place it in the box}"

DATASET_REPO_ID="${DATASET_REPO_ID:-local/eval_pi0_lerobot_fmc3_gr2_grab_box_via_middle_v1}"
DATASET_ROOT="${DATASET_ROOT:-/tmp/eval_pi0_lerobot_fmc3_gr2_grab_box_via_middle_v1}"

NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-90}"
RESET_TIME_S="${RESET_TIME_S:-0}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}"
  exit 1
fi

echo "[INFO] repo_root=${REPO_ROOT}"
echo "[INFO] model_path=${MODEL_PATH}"
echo "[INFO] robot_port=${ROBOT_PORT}"
echo "[INFO] robot_id=${ROBOT_ID}"
echo "[INFO] dataset_repo_id=${DATASET_REPO_ID}"
echo "[INFO] dataset_root=${DATASET_ROOT}"
echo "[INFO] task=${TASK}"

rm -rf "${DATASET_ROOT}"

cd "${REPO_ROOT}"

lerobot-record \
  --robot.type="${ROBOT_TYPE}" \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --policy.path="${MODEL_PATH}" \
  --policy.device="${POLICY_DEVICE}" \
  --policy.dtype="${POLICY_DTYPE}" \
  --display_data="${DISPLAY_DATA}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.single_task="${TASK}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub=false
