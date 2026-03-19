#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-lerobot-pi0}"
SRC_DATASET_ROOT="${SRC_DATASET_ROOT:-/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_from_box_to_desk_lerobot_ds}"
DST_DATASET_ROOT="${DST_DATASET_ROOT:-${SRC_DATASET_ROOT}_rgb}"
SRC_REPO_ID="${SRC_REPO_ID:-local/fmc3_gr2_grab_bottle_from_box_to_desk}"
DST_REPO_ID="${DST_REPO_ID:-local/fmc3_gr2_grab_bottle_from_box_to_desk_rgb}"
FORCE="${FORCE:-false}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/tools/remove_dataset_depth_features.sh [options]

Options:
  --src <path>          Source LeRobot dataset root.
  --dst <path>          Output dataset root. Defaults to "<src>_rgb".
  --repo-id <id>        Logical repo id for the source dataset.
  --new-repo-id <id>    Logical repo id for the output dataset.
  --env <name>          Conda env to use when the target env is not active.
  --force               Remove the output directory first if it already exists.
  -h, --help            Show this help.

Examples:
  bash scripts/tools/remove_dataset_depth_features.sh

  bash scripts/tools/remove_dataset_depth_features.sh \
    --src /path/to/dataset \
    --dst /path/to/dataset_rgb \
    --repo-id local/my_dataset \
    --new-repo-id local/my_dataset_rgb
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)
      SRC_DATASET_ROOT="$2"
      shift 2
      ;;
    --dst)
      DST_DATASET_ROOT="$2"
      shift 2
      ;;
    --repo-id)
      SRC_REPO_ID="$2"
      shift 2
      ;;
    --new-repo-id)
      DST_REPO_ID="$2"
      shift 2
      ;;
    --env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --force)
      FORCE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "${SRC_DATASET_ROOT}" ]]; then
  echo "[ERROR] Source dataset root not found: ${SRC_DATASET_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${SRC_DATASET_ROOT}/meta/info.json" ]]; then
  echo "[ERROR] Missing metadata file: ${SRC_DATASET_ROOT}/meta/info.json" >&2
  exit 1
fi

if [[ "${SRC_DATASET_ROOT}" == "${DST_DATASET_ROOT}" ]]; then
  echo "[ERROR] Source and destination dataset roots must differ." >&2
  exit 1
fi

if [[ -e "${DST_DATASET_ROOT}" ]]; then
  if [[ "${FORCE}" != "true" ]]; then
    echo "[ERROR] Output directory already exists: ${DST_DATASET_ROOT}" >&2
    echo "        Re-run with --force to replace it." >&2
    exit 1
  fi
  rm -rf "${DST_DATASET_ROOT}"
fi

mkdir -p "$(dirname "${DST_DATASET_ROOT}")"

echo "[INFO] src=${SRC_DATASET_ROOT}"
echo "[INFO] dst=${DST_DATASET_ROOT}"
echo "[INFO] src_repo_id=${SRC_REPO_ID}"
echo "[INFO] dst_repo_id=${DST_REPO_ID}"
echo "[INFO] target_env=${CONDA_ENV}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
  PYTHON_CMD=(python)
else
  PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)
fi

export SRC_DATASET_ROOT
export DST_DATASET_ROOT
export SRC_REPO_ID
export DST_REPO_ID

"${PYTHON_CMD[@]}" - <<'PY'
import os
from pathlib import Path

from lerobot.datasets.dataset_tools import remove_feature
from lerobot.datasets.lerobot_dataset import LeRobotDataset

src_root = Path(os.environ["SRC_DATASET_ROOT"])
dst_root = Path(os.environ["DST_DATASET_ROOT"])
src_repo_id = os.environ["SRC_REPO_ID"]
dst_repo_id = os.environ["DST_REPO_ID"]

dataset = LeRobotDataset(repo_id=src_repo_id, root=src_root)

depth_features = [
    key
    for key in dataset.meta.features
    if key.startswith("observation.images.") and "depth" in key.lower()
]

if not depth_features:
    raise SystemExit("[INFO] No depth image features found. Nothing to remove.")

print("[INFO] Removing depth features:")
for key in depth_features:
    print(f"  - {key}")

remove_feature(
    dataset=dataset,
    feature_names=depth_features,
    output_dir=dst_root,
    repo_id=dst_repo_id,
)

print(f"[INFO] Saved RGB-only dataset to: {dst_root}")
PY
