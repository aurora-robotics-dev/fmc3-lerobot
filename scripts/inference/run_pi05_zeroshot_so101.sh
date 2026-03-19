#!/usr/bin/env bash
set -euo pipefail

# Pi0.5 zero-shot test for the current SO101 setup.
#
# This script avoids common shell mistakes from long multi-line commands:
# - broken '\' line continuations
# - JSON strings split across lines
# - stale eval dataset directories causing FileExistsError
#
# Expected current hardware mapping:
# - follower: /dev/ttyACM2
# - leader is not used in this script
# - cameras:
#   - top   -> OpenCV camera 0
#   - wrist -> OpenCV camera 2
#
# Important:
# pi05_base expects different camera names from this SO101 setup, so we remap:
# - observation.images.top   -> observation.images.base_0_rgb
# - observation.images.wrist -> observation.images.right_wrist_0_rgb
#
# The remaining expected pi05 camera stream is left missing on purpose.
# The pi05 model code will pad the missing stream with an empty image.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Model path. Change this if you later want to test another pi05 checkpoint.
POLICY_PATH="${POLICY_PATH:-/home/phl/workspace/models/pi05_base}"

# Robot connection settings for the current SO101 follower arm.
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM2}"
ROBOT_ID="${ROBOT_ID:-fmc3_robotics_follower_arm}"

# Camera configuration for the current setup.
if [[ -z "${ROBOT_CAMERAS:-}" ]]; then
  ROBOT_CAMERAS='{"top":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30}}'
fi

# Task text given to the policy. You can override it with:
# TASK='Pick up the tape, place it at the middle waypoint, then place it in the box' bash ...
TASK="${TASK:-Pick up the tape and place it in the box}"

# Keep this as an eval_* name because lerobot-record requires that when a policy is provided.
DATASET_REPO_ID="${DATASET_REPO_ID:-local/eval_pi05_lerobot_fmc3_gr2_grab_box_zeroshot_v1}"

# Store temporary eval data under /tmp so it is easy to discard.
DATASET_ROOT="${DATASET_ROOT:-/tmp/eval_pi05_lerobot_fmc3_gr2_grab_box_zeroshot_v1}"

# Runtime controls.
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-90}"
RESET_TIME_S="${RESET_TIME_S:-3}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"

# Camera-key remapping required by pi05_base.
RENAME_MAP='{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.right_wrist_0_rgb"}'

if [[ ! -e "${POLICY_PATH}" ]]; then
  echo "[ERROR] POLICY_PATH does not exist: ${POLICY_PATH}"
  exit 1
fi

echo "[INFO] repo_root=${REPO_ROOT}"
echo "[INFO] policy_path=${POLICY_PATH}"
echo "[INFO] robot_port=${ROBOT_PORT}"
echo "[INFO] robot_id=${ROBOT_ID}"
echo "[INFO] dataset_repo_id=${DATASET_REPO_ID}"
echo "[INFO] dataset_root=${DATASET_ROOT}"
echo "[INFO] task=${TASK}"

# lerobot-record refuses to create a dataset if the target directory already exists.
# Remove the previous temporary eval directory so repeated tests work without manual cleanup.
rm -rf "${DATASET_ROOT}"

cd "${REPO_ROOT}"

lerobot-record \
  --robot.type="${ROBOT_TYPE}" \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --policy.path="${POLICY_PATH}" \
  --policy.device="${POLICY_DEVICE}" \
  --policy.dtype="${POLICY_DTYPE}" \
  --display_data="${DISPLAY_DATA}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.rename_map="${RENAME_MAP}" \
  --dataset.single_task="${TASK}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub=false
