# GR2 训练脚本

## 快速开始

### 1. 使用 screen 后台运行训练

```bash
# 创建 screen 会话
screen -S gr2_train

# 运行训练脚本
./gr2_train.sh

# 按 Ctrl+A+D 分离会话
# 重新连接: screen -r gr2_train
```

### 2. 配置训练参数

编辑 `gr2_train.sh` 文件顶部的【常用配置】区域：

```bash
# --- 数据集 ---
DATASET_NAME="fmc3_gr2_green_yellow_merged"  # 修改数据集名称

# --- 输出 ---
RUN_NAME="pi0_green_to_yellow_0327_v3"  # 修改运行名称（每次训练需要不同）

# --- 训练参数 ---
STEPS=100000        # 总训练步数
BATCH_SIZE=8        # 批次大小（根据显存调整）
SAVE_FREQ=10000     # 保存频率
LOG_FREQ=50         # 日志频率

# --- Wandb 日志 ---
WANDB_ENABLE=true   # 是否启用 Wandb
WANDB_PROJECT="Lerobot_Phl_Project_green2yellow"  # Wandb 项目名
```

## 输出文件

训练输出保存在：
- 模型检查点: `/home/phl/workspace/mymodels/gr2/{RUN_NAME}/`
- 训练日志: `/home/phl/workspace/mymodels/gr2/{RUN_NAME}/train_*.log`

## 常见问题

### 1. 输出目录已存在错误

```
FileExistsError: Output directory ... already exists
```

**解决方法**: 修改 `RUN_NAME` 为新的名称，或删除旧目录

### 2. 数据集路径不存在

检查 `DATASET_ROOT` 路径是否正确，确保包含 `meta/info.json` 文件

### 3. 显存不足

调整以下参数：
- 减小 `BATCH_SIZE`
- 设置 `GRADIENT_CHECKPOINTING=true`
- 设置 `FREEZE_VISION_ENCODER=true`

## Screen 常用命令

```bash
# 查看所有 screen 会话
screen -ls

# 重新连接会话
screen -r gr2_train

# 分离会话（在 screen 内）
Ctrl+A+D

# 终止会话（在 screen 内）
exit
```
