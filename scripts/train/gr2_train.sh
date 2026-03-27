#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ "${1:-}" == "--resume" ]]; then
  conda run --no-capture-output -n lerobot-pi0 \
    lerobot-train \
    --config_path=/home/phl/workspace/mymodels/gr2/pi0_gr2_black_capped_bottle_yellow_to_green/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
  exit 0
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n lerobot-pi0 \
  lerobot-train \
  --dataset.repo_id=local/fmc3_gr2_green_yellow_merged \
  --dataset.root=/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_green_yellow_merged \
  --dataset.streaming=false \
  --dataset.video_backend=torchcodec \
  --policy.path=/home/phl/workspace/models/pi0 \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.max_state_dim=45 \
  --policy.max_action_dim=35 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=/home/phl/workspace/mymodels/gr2/fmc3_gr2_green_yellow_merged_3_26 \
  --job_name=fmc3_gr2_green_yellow_merged \
  --steps=100000 \
  --save_freq=10000 \
  --log_freq=50 \
  --wandb.enable=false\
  --wandb.disable_artifact=false \
  --wandb.project=Lerobot_Phl_Project \
  --batch_size=8
