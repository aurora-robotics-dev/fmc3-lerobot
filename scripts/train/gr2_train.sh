#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="/home/phl/workspace/mymodels/gr2/act_gr2_grab_bottle_from_box_to_desk_rgb_${TIMESTAMP}"

echo "dataset_root=/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_from_box_to_desk_lerobot_ds_rgb"
echo "output_dir=${OUTPUT_DIR}"
echo "policy_type=act"
echo "batch_size=8"

cd "${REPO_ROOT}"

PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=local/fmc3_gr2_grab_bottle_from_box_to_desk_lerobot_ds_rgb \
    --dataset.root=/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_from_box_to_desk_lerobot_ds_rgb \
    --dataset.streaming=false \
    --dataset.video_backend=torchcodec \
    --policy.type=act \
    --policy.input_features=null \
    --policy.output_features=null \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.use_amp=true \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=act_gr2_grab_bottle_from_box_to_desk_rgb \
    --steps=100000 \
    --save_freq=5000 \
    --log_freq=50 \
    --wandb.enable=true \
    --wandb.project=Lerobot_Phl_Project_to_desk \
    --batch_size=8
