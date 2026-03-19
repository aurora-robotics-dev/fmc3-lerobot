# LeRobot for Fourier GR-2

基于 [HuggingFace LeRobot](https://github.com/huggingface/lerobot) 的 Fourier GR-2 人形机器人策略训练与部署工具链。

核心流程：**遥操作采集数据 → LeRobot 数据集 → 训练策略（PI0 / ACT / GR00T） → 部署到 GR-2 机器人**

## 项目结构

```
├── src/lerobot/                    # LeRobot 核心库（editable install）
│   ├── policies/
│   │   ├── pi0/                    # PI0 策略
│   │   ├── act/                    # ACT 策略
│   │   ├── groot/                  # GR00T N1.5 策略（本分支新增）
│   │   └── ...
│   └── scripts/                    # CLI 入口脚本
├── scripts/
│   ├── train/                      # 训练脚本
│   │   ├── train_pi0_gr2_pick.sh       # PI0 微调 GR2 抓取
│   │   ├── train_act_grab_box.sh       # ACT 训练
│   │   ├── train_groot_n1p5_gr2_pick.sh # GR00T N1.5 训练
│   │   └── train_lerobot_generic.sh    # 通用训练模板
│   ├── inference/                  # 推理 & 部署启动脚本
│   │   └── start_gr2_pi0_inference_service.sh
│   ├── deploy_gr2_pi0.py          # PI0 单视觉部署（RGB）
│   ├── deploy_gr2_pi0_rgbd.py     # PI0 多视觉部署（RGB + Depth）
│   ├── deploy_gr2_act.py          # ACT 部署
│   ├── gr2_pi0_inference_service.py  # PI0 推理服务（Unix Socket IPC）
│   └── review_episodes.py         # 数据集清洗：交互式 episode 筛选
└── outputs/                        # 训练输出 & checkpoints
```

## 环境配置

```bash
# 基础环境
conda create -n lerobot-pi0 python=3.10
conda activate lerobot-pi0
pip install -e ".[pi0]"

# GR00T N1.5（需要 flash-attn）
pip install -e ".[groot]"
```

## 训练

### PI0 微调（GR-2 抓取任务）

```bash
conda activate lerobot-pi0
bash scripts/train/train_pi0_gr2_pick.sh
```

### ACT 训练

```bash
bash scripts/train/train_act_grab_box.sh
```

### GR00T N1.5 训练

```bash
bash scripts/train/train_groot_n1p5_gr2_pick.sh
```

### 断点续训

```bash
bash scripts/train/train_pi0_gr2_pick.sh --resume
```

## 部署

### 直接部署（单次推理循环）

```bash
# PI0 单视觉（RGB）
python scripts/deploy_gr2_pi0.py \
    --checkpoint outputs/train/.../pretrained_model \
    --domain-id 123 --robot-name gr2

# PI0 多视觉（RGB + Depth）
python scripts/deploy_gr2_pi0_rgbd.py \
    --checkpoint outputs/train/.../pretrained_model \
    --domain-id 123 --robot-name gr2

# ACT
python scripts/deploy_gr2_act.py \
    --checkpoint outputs/train/.../pretrained_model \
    --domain-id 123 --robot-name gr2

# 调试模式（不连接机器人）
python scripts/deploy_gr2_pi0.py --checkpoint ... --dry-run
```

### 推理服务（RoboOS 对接）

通过 Unix Domain Socket 提供 IPC 推理服务，供 RoboOS Slaver 调用：

```bash
bash scripts/inference/start_gr2_pi0_inference_service.sh
```

支持命令：`health` / `status` / `start` / `stop` / `reload`，默认 socket 路径 `/tmp/gr2_pi0_inference_service.sock`。

## 数据集清洗

交互式逐个回放 episode，通过 Rerun GUI 可视化审核，标记保留或删除：

```bash
python scripts/review_episodes.py \
    --repo-id fmc3_gr2_grab_bottle_into_box_lerobot_ds \
    --root /path/to/dataset
```

方向键操作：`→` 保留下一个，`←` 上一个，`↓` 标记删除，`↑` 重播。审核结果保存到 `episodes_to_delete.json`，确认后执行：

```bash
lerobot-edit-dataset delete_episodes \
    --dataset.repo_id=<repo_id> \
    --dataset.root=/path/to/dataset \
    --episode_indices='[3, 7, 15]' \
    --output_dir=/path/to/cleaned_dataset
```

详见 [scripts/README.md](scripts/README.md)。

## GR-2 机器人关节配置

| 控制组 | 关节数 | 说明 |
|--------|--------|------|
| `left/right_manipulator` | 7 × 2 | 肩部俯仰/横滚/偏航、肘部俯仰、腕部偏航/俯仰/横滚 |
| `left/right_hand` | 6 × 2 | 小指/无名指/中指/食指/拇指近端 |
| `head` | 2 | 偏航、俯仰 |
| `waist` | 1-3 | 偏航（始终）、横滚/俯仰（仅 action） |

数据维度：State 45D，Action 35D。FSM 状态：11（upper_body_cmd）。

## 分支说明

| 分支 | 说明 |
|------|------|
| `main` | 上游 LeRobot 主分支 |
| `lerobot-gr2` | GR-2 适配：GR00T 策略集成、训练/部署脚本、数据集工具 |
