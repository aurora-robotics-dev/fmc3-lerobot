#!/usr/bin/env bash
# GR2 机器人 Pi0 策略训练脚本
# 用途：训练 LeRobot Pi0 策略模型用于 GR2 机器人操作任务
# 作者：FMC3 Robotics Team
# 最后更新：2026-03-27

set -euo pipefail  # 遇到错误立即退出，未定义变量报错

cd "$(dirname "$0")/../.."  # 切换到项目根目录

# ============================================================================
# 【常用配置】- 每次训练最常修改的参数
# ============================================================================

# --- 数据集 ---
DATASET_NAME="fmc3_gr2_green_yellow_merged"  # 数据集名称

# --- 输出 ---
RUN_NAME="pi0_green_to_yellow_0327"  # 本次训练名称（建议包含日期版本）

# --- 训练参数 ---
STEPS=100000        # 总训练步数
BATCH_SIZE=4        # 批次大小（根据显存调整）
SAVE_FREQ=10000     # 每隔多少步保存检查点
LOG_FREQ=50         # 每隔多少步记录日志

# --- Wandb 日志 ---
WANDB_ENABLE=true   # 是否启用 Wandb（true/false）
WANDB_PROJECT="Lerobot_Phl_Project_green2yellow"  # Wandb 项目名

# ============================================================================
# 【路径配置】- 根据环境设置
# ============================================================================

# 数据集路径
DATASET_ROOT="/home/phl/workspace/dataset/Robot/fourier/gr2/muticams/lerobot/${DATASET_NAME}"

# 模型路径
MODEL_BASE="/home/phl/workspace/models/pi0"

# 输出路径
OUTPUT_BASE="/home/phl/workspace/mymodels/gr2"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"

# ============================================================================
# 【模型配置】- 较少修改
# ============================================================================

POLICY_TYPE="pi0"
MAX_STATE_DIM=45    # GR2 机器人状态维度
MAX_ACTION_DIM=35   # GR2 机器人动作维度

# 优化设置
COMPILE_MODEL=false              # 是否编译模型
GRADIENT_CHECKPOINTING=true      # 梯度检查点（省显存）
DTYPE=bfloat16                   # 训练精度
FREEZE_VISION_ENCODER=false      # 是否冻结视觉编码器
TRAIN_EXPERT_ONLY=false          # 是否只训练动作头

# ============================================================================
# 【数据配置】- 较少修改
# ============================================================================

DATASET_STREAMING=false  # 是否流式加载
VIDEO_BACKEND=torchcodec # 视频解码后端

# ============================================================================
# 【硬件配置】- 较少修改
# ============================================================================

DEVICE=cuda
NUM_WORKERS=4
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# 【其他配置】
# ============================================================================

JOB_NAME="${DATASET_NAME}"
WANDB_DISABLE_ARTIFACT=false

# ============================================================================
# 训练执行
# ============================================================================

echo "=========================================="
echo "训练配置摘要"
echo "=========================================="
echo "数据集: ${DATASET_NAME}"
echo "模型: ${POLICY_TYPE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "训练步数: ${STEPS}"
echo "批次大小: ${BATCH_SIZE}"
echo "=========================================="

# 执行训练（训练程序会创建输出目录，然后我们才能写日志）
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} \
conda run --no-capture-output -n lerobot-pi0 \
  lerobot-train \
  --dataset.repo_id=local/${DATASET_NAME} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.streaming=${DATASET_STREAMING} \
  --dataset.video_backend=${VIDEO_BACKEND} \
  --policy.path=${MODEL_BASE} \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.compile_model=${COMPILE_MODEL} \
  --policy.gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
  --policy.dtype=${DTYPE} \
  --policy.freeze_vision_encoder=${FREEZE_VISION_ENCODER} \
  --policy.train_expert_only=${TRAIN_EXPERT_ONLY} \
  --policy.max_state_dim=${MAX_STATE_DIM} \
  --policy.max_action_dim=${MAX_ACTION_DIM} \
  --policy.device=${DEVICE} \
  --policy.push_to_hub=false \
  --output_dir=${OUTPUT_DIR} \
  --job_name=${JOB_NAME} \
  --steps=${STEPS} \
  --save_freq=${SAVE_FREQ} \
  --log_freq=${LOG_FREQ} \
  --wandb.enable=${WANDB_ENABLE} \
  --wandb.disable_artifact=${WANDB_DISABLE_ARTIFACT} \
  --wandb.project=${WANDB_PROJECT} \
  --batch_size=${BATCH_SIZE} 2>&1 | tee "${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
