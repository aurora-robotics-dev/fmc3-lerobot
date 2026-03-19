# Pi0 机器人策略训练指南

基于 [LeRobot](https://github.com/huggingface/lerobot) 框架，在 AutoDL 上 fine-tune `lerobot/pi0` 官方基座模型。

---

## 环境说明

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX PRO 6000 Blackwell (97GB VRAM) |
| 系统 | Linux (AutoDL 容器) |
| Python 环境 | conda `lerobot` |
| 系统盘 | `/root` (空间有限，约 30GB) |
| 数据盘 | `/root/autodl-tmp` (大容量，存模型和缓存) |

---

## 第一步：环境准备

### 1.1 激活 conda 环境

```bash
conda activate lerobot
```

### 1.2 开启学术加速（AutoDL 专用，访问 HuggingFace/GitHub）

```bash
source /etc/network_turbo
```

### 1.3 设置 HuggingFace 缓存到数据盘（避免系统盘爆满）

```bash
export HF_HOME=/root/autodl-tmp/huggingface_cache
```

> **重要**：pi0 基座模型约 14GB，必须存到数据盘，否则系统盘会满。

---

## 第二步：HuggingFace 登录

### 2.1 登录账号

```bash
huggingface-cli login
# 输入 HF token（从 https://huggingface.co/settings/tokens 获取）
```

### 2.2 将 token 复制到数据盘缓存目录

```bash
mkdir -p /root/autodl-tmp/huggingface_cache
cp /root/.cache/huggingface/token /root/autodl-tmp/huggingface_cache/token
```

> **原因**：设置了 `HF_HOME` 后，token 需要在新目录下才能被识别。

### 2.3 申请 PaliGemma 访问权限

pi0 模型依赖 `google/paligemma-3b-pt-224`，这是一个受限模型，需要在 HuggingFace 网站手动申请：

1. 访问 https://huggingface.co/google/paligemma-3b-pt-224
2. 点击 "Agree and access repository"
3. 等待审核通过（通常即时）

---

## 第三步：下载数据集

```bash
export HF_HOME=/root/autodl-tmp/huggingface_cache
source /etc/network_turbo

huggingface-cli download \
    --repo-type dataset \
    puheliang/lerobot_fmc3_grab_box_v2 \
    --local-dir /root/autodl-tmp/huggingface_cache/hub/datasets--puheliang--lerobot_fmc3_grab_box_v2
```

> 替换 `puheliang/lerobot_fmc3_grab_box_v2` 为你自己的数据集 repo_id。

---

## 第四步：下载 pi0 基座模型

```bash
export HF_HOME=/root/autodl-tmp/huggingface_cache
source /etc/network_turbo

huggingface-cli download lerobot/pi0
```

模型会自动缓存到 `$HF_HOME/hub/models--lerobot--pi0`，约 14GB。

---

## 第五步：启动训练

### 5.1 使用训练脚本（推荐）

```bash
# 新训练（会清空旧输出目录）
bash /root/autodl-tmp/scripts/train_pi0_grab_box.sh

# 恢复训练
bash /root/autodl-tmp/scripts/train_pi0_grab_box.sh --resume
```

### 5.2 手动命令（完整参数）

```bash
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/root/autodl-tmp/huggingface_cache
source /etc/network_turbo

lerobot-train \
    --dataset.repo_id=puheliang/lerobot_fmc3_grab_box_v2 \
    --dataset.streaming=false \
    --policy.type=pi0 \
    --output_dir=/root/output_lerobot_train/grab_box/pi0_fmc3 \
    --job_name=grab_box_pi0_fmc3 \
    --policy.pretrained_path=lerobot/pi0 \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=false \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --steps=50000 \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=puheliang/pi0_fmc3_grab_box_v2 \
    --batch_size=32 \
    --save_freq=20000 \
    --wandb.enable=true \
    --wandb.project=Lerobot_Fmc_Project
```

### 关键参数说明

| 参数 | 说明 |
|------|------|
| `--policy.pretrained_path=lerobot/pi0` | **必须设置**，加载官方基座模型做 fine-tune |
| `--policy.compile_model=true` | 开启 torch.compile 加速（首次启动需 10-30 分钟编译） |
| `--policy.gradient_checkpointing=false` | 97GB 显存足够，不需要节省显存 |
| `--policy.freeze_vision_encoder=false` | 解冻视觉编码器，全参数训练 |
| `--policy.dtype=bfloat16` | 使用 bfloat16 精度 |
| `--batch_size=32` | 根据显存调整，97GB 可用 32 |
| `--steps=50000` | 总训练步数 |
| `--save_freq=20000` | 每 20000 步保存一次 checkpoint |

---

## 第六步：后台运行（screen）

```bash
# 创建新 screen 会话
screen -S train_pi0

# 在 screen 里执行训练命令
conda activate lerobot
bash /root/autodl-tmp/scripts/train_pi0_grab_box.sh

# 脱离 screen（训练继续在后台运行）
# 按 Ctrl+A，然后按 D
```

### 常用 screen 命令

```bash
# 查看所有 screen 会话
screen -ls

# 重新连接到训练会话
screen -r train_pi0

# 停止训练（在 screen 内）
# 按 Ctrl+C
```

---

## 监控训练

### 查看 GPU 显存占用

```bash
nvidia-smi
```

### 查看训练日志

```bash
# 连接到 screen 查看实时输出
screen -r train_pi0

# 或查看日志文件
tail -f /root/output_lerobot_train/grab_box/pi0_fmc3/logs/train.log
```

### WandB 监控

训练启动后会输出 WandB 链接，例如：
```
Track this run --> https://wandb.ai/your-entity/Lerobot_Fmc_Project/runs/xxxxx
```

---

## 常见问题

### Q: 训练启动后长时间卡在 AUTOTUNE

**正常现象**。`torch.compile max-autotune` 会对每个矩阵运算测试最优 kernel，首次编译需要 10-30 分钟，之后会缓存。

### Q: 出现 `OutOfResources: shared memory` 错误

**不影响训练**。这是 autotune 在跳过不适合当前 GPU 的 kernel 配置，属于正常日志。

### Q: `GatedRepoError: 401 Unauthorized` for paligemma

原因：HF token 未找到，或未申请 PaliGemma 访问权限。
解决：
1. 确认已在 https://huggingface.co/google/paligemma-3b-pt-224 申请访问
2. 执行：`cp /root/.cache/huggingface/token /root/autodl-tmp/huggingface_cache/token`

### Q: `FileExistsError: Output directory already exists`

```bash
rm -rf /root/output_lerobot_train/grab_box/pi0_fmc3
```

或使用 `--resume=true` 继续训练。

### Q: 系统盘空间不足

```bash
# 清理 pip 缓存
pip cache purge

# 清理 conda 缓存
conda clean --all -y

# 确保 HF_HOME 指向数据盘
export HF_HOME=/root/autodl-tmp/huggingface_cache
```

### Q: 网络无法访问 HuggingFace

```bash
source /etc/network_turbo
```

---

## 目录结构

```
/root/autodl-tmp/
├── huggingface_cache/          # HF 模型和数据集缓存（数据盘）
│   ├── hub/
│   │   ├── models--lerobot--pi0/       # pi0 基座模型 (~14GB)
│   │   └── datasets--puheliang--*/     # 数据集
│   └── token                           # HF 登录 token
└── scripts/
    ├── train_pi0_grab_box.sh   # 训练启动脚本
    ├── README.md               # 脚本说明
    └── TRAINING_GUIDE.md       # 本文件

/root/output_lerobot_train/
└── grab_box/
    └── pi0_fmc3/               # 训练输出（checkpoints、日志）
```
