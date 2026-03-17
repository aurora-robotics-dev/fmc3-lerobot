#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/home/phl/workspace/mymodels/gr2/pi0_gr2_grab_bottle_into_box_rgb_${TIMESTAMP}"

echo "dataset_root=/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_into_box_lerobot_ds_rgb"
echo "output_dir=${OUTPUT_DIR}"
echo "batch_size=8"

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
lerobot-train \
    --dataset.repo_id=local/fmc3_gr2_grab_bottle_into_box_lerobot_ds_rgb \
    --dataset.root=/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_into_box_lerobot_ds_rgb \
    --dataset.streaming=false \
    --dataset.video_backend=torchcodec \
    --policy.type=pi0 \
    --policy.pretrained_path=/home/phl/workspace/models/pi0 \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.max_state_dim=45 \
    --policy.max_action_dim=35 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=pi0_gr2_grab_bottle_into_box_rgb \
    --steps=100000 \
    --save_freq=10000 \
    --log_freq=50 \
    --wandb.enable=true \
    --wandb.project=Lerobot_Phl_Project_grap_box_into_box \
    --batch_size=8
