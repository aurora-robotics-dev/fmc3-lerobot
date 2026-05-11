# fmc3-lerobot / Fourier GR2 LeRobot

本仓库基于 Hugging Face LeRobot，面向真实机器人学习、数据集转换、策略训练、推理部署和 Fourier GR2 机器人运行做了本地化扩展。当前项目重点支持 GR2 上身控制、PI0 策略微调、RGB/RGB-D/腕部相机输入、LeRobot v3 数据集回放，以及基于 Unix Socket 的在线推理服务。

> 说明：本仓库根目录保留 LeRobot 原始工程结构，同时新增了 GR2 相关脚本、环境快照和部署文档。实际训练或部署前，请先确认本机 conda 环境、机器人 DDS domain、相机设备和 checkpoint 路径。

### 主要功能

- LeRobot 核心库：策略、数据集、机器人接口、环境、训练和评估入口。
- Fourier GR2 数据集转换：将 Dora-Record 风格数据转换为 LeRobot v3 格式。
- GR2 数据集回放：支持本地或 Hugging Face Hub 数据集 episode 回放。
- PI0 训练：支持 GR2 数据集上的 PI0 微调流程。
- PI0 部署：支持单模型、RGB-D、RGB 腕部相机和双模型切换推理。
- 推理服务：提供 Unix Domain Socket 服务，方便 RoboOS 或外部控制进程对接。
- 环境记录：保存 `lerobot` 与 `lerobot-pi0` 两套环境的参考说明和 `pip freeze` 快照。

### 目录结构

```text
.
├── src/lerobot/                  # LeRobot 核心库代码
├── tests/                        # pytest 测试
├── docs/                         # 文档源码
├── examples/                     # 示例
├── Robot/fouier/                 # Fourier/GR2 数据转换相关工具
├── scripts/                      # GR2 训练、部署、推理、回放脚本
├── scripts/train/                # 训练脚本
├── scripts/inference/            # 推理服务与客户端脚本
├── gr2_env/                      # GR2 环境说明与配置总结
├── fourier_aurora_sdk/           # Fourier Aurora SDK 相关内容
├── outputs/                      # 本地输出目录
└── Log/                          # 本地日志目录
```

### 环境安装

推荐使用 Python 3.10。

通用 LeRobot 开发、测试和 GR2 数据集回放环境：

```bash
conda create -n lerobot python=3.10
conda activate lerobot
pip install -e ".[dev,test]"
pip install fourier-aurora-client
```

PI0 训练、PI0 部署和推理服务环境：

```bash
conda create -n lerobot-pi0 python=3.10
conda activate lerobot-pi0
pip install -e ".[pi,dev,test]"
pip install fourier-aurora-client
```

如果只需要 PI0 运行能力，可简化为：

```bash
pip install -e ".[pi]"
pip install fourier-aurora-client
```

环境验证：

```bash
python -c "import lerobot; print('lerobot import ok')"
python -c "import fourier_aurora_client; print('aurora import ok')"
lerobot-train --help
```

PI0 环境额外验证：

```bash
python -c "from lerobot.policies.pi0.modeling_pi0 import PI0Policy; print('pi0 import ok')"
```

### GR2 关键约定

当前仓库内 GR2 训练、部署和回放脚本建议保持以下约定一致：

- `robot_type=fourier_gr2`
- `robot_name=gr2`
- `domain_id=123`
- `fsm_state=11`
- 控制频率通常为 `30 FPS`
- 相机分辨率通常为 `640x480`
- 当前标准状态维度：`state_dim=45`
- 当前标准动作维度：`action_dim=35`

当前 35 维动作布局：

| 索引 | 关节组 | 维度 |
| --- | --- | --- |
| `0:7` | left_manipulator | 7 |
| `7:14` | right_manipulator | 7 |
| `14:20` | left_hand | 6 |
| `20:26` | right_hand | 6 |
| `26:28` | head | 2 |
| `28:29` | waist yaw | 1 |
| `29:35` | base | 6 |

常用观测键：

```text
observation.state
observation.images.camera_top
observation.images.camera_top_depth
```

部分 RGB 腕部相机脚本还会使用 left/right wrist camera 相关键。

### 数据集转换

GR2 数据转换脚本：

```bash
Robot/fouier/convert_dora_to_lerobot.py
```

示例：

```bash
python Robot/fouier/convert_dora_to_lerobot.py \
  --input /path/to/dora_episode_root \
  --output /path/to/lerobot_output \
  --task "teleoperation task" \
  --fps 30 \
  --video-codec libx264 \
  --robot-type fourier_gr2
```

转换后建议保持 LeRobot v3 数据集结构：

```text
dataset_name/
├── meta/
│   ├── info.json
│   ├── stats.json
│   └── tasks.parquet
├── data/
│   └── chunk-000/
│       └── file-000.parquet
└── videos/
    └── camera_top/
        └── chunk-000/
            └── file-000.mp4
```

### 数据集回放

建议在 `lerobot` 环境中使用回放工具。

Dry-run 验证数据集：

```bash
conda run --no-capture-output -n lerobot \
  python scripts/replay_gr2_dataset.py \
  --dataset-path /path/to/dataset \
  --episode 0 \
  --dry-run \
  --verbose
```

真实机器人回放：

```bash
conda run --no-capture-output -n lerobot \
  python scripts/replay_gr2_dataset.py \
  --dataset-path /path/to/dataset \
  --episode 0 \
  --domain-id 123 \
  --fps 30 \
  --transition-time 3.0
```

首次上真机前建议先使用 `--dry-run`，并设置较小的 `--max-joint-delta`。

### PI0 训练

建议在 `lerobot-pi0` 环境中训练。仓库中可参考：

```bash
scripts/train/gr2_train.sh
scripts/tools/train_pi0_gr2_black_capped_bottle_yellow_to_green.sh
scripts/tools/train_pi0_gr2_grab_bottle_from_box_to_desk_rgb.sh
```

通用训练命令示例：

```bash
conda run --no-capture-output -n lerobot-pi0 \
  lerobot-train \
  --dataset.repo_id=local/my_gr2_dataset \
  --dataset.root=/path/to/my_gr2_dataset \
  --dataset.streaming=false \
  --dataset.video_backend=torchcodec \
  --policy.path=/path/to/pi0 \
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
  --output_dir=/path/to/output \
  --job_name=my_gr2_run \
  --steps=100000 \
  --save_freq=10000 \
  --log_freq=50 \
  --batch_size=4 \
  --wandb.enable=true \
  --wandb.disable_artifact=false \
  --wandb.project=my_gr2_project
```

显存不足时可优先降低 `batch_size`，或开启/保持 `gradient_checkpointing=true`。

### PI0.5 触觉与本地 PaliGemma

PI0.5 支持可选触觉输入。不开触觉时保持默认：

```bash
--policy.type=pi05 \
--policy.use_tactile=false
```

使用 O10 触觉热力图时，数据集特征可命名为 `observation.tactile.left`、
`observation.tactile.right` 或 `observation.tactile`。设置 `--policy.use_tactile=true`
后会自动判断左手、右手或双手；默认触觉图尺寸为 `[12,32]`。

```bash
--policy.type=pi05 \
--policy.use_tactile=true \
--policy.tactile_input_shape='[12,32]' \
--policy.tactile_encoder_type=cnn
```

如果本地已有 PaliGemma tokenizer，建议直接指定本地路径，避免访问 gated Hub：

```bash
export LEROBOT_PALIGEMMA_TOKENIZER=/home/phl/workspace/models/paligemma-tokenizer
# 或者：
--policy.paligemma_tokenizer_name=/home/phl/workspace/models/paligemma-tokenizer
```

### 部署与推理

常用部署入口：

```text
scripts/deploy_gr2_pi0.py
scripts/deploy_gr2_pi0_rgbd.py
scripts/deploy_gr2_pi0_rgb_right_wrist.py
scripts/gr2_pi0_inference_service.py
scripts/inference/gr2_dual_pi0_rgb_wrist_inference_server.py
```

单模型部署示例：

```bash
conda run --no-capture-output -n lerobot-pi0 \
  python scripts/deploy_gr2_pi0.py \
  --checkpoint-path /path/to/pretrained_model \
  --task "pick bottle" \
  --robot-type fourier_gr2 \
  --robot-name gr2 \
  --domain-id 123 \
  --fsm-state 11 \
  --fps 30 \
  --device auto
```

Unix Socket 推理服务示例：

```bash
CONDA_ENV=lerobot-pi0 \
CHECKPOINT_PATH=/path/to/pretrained_model \
TASK="pick bottle and place into box" \
bash scripts/inference/start_gr2_pi0_inference_service.sh
```

双模型 RGB 腕部相机推理服务：

```bash
bash scripts/inference/start_gr2_dual_pi0_rgb_wrist_inference_server.sh
```

客户端控制示例：

```bash
CLIENT="python scripts/inference/gr2_dual_pi0_rgb_wrist_client.py"
$CLIENT start-take-out --fps 15 --max-steps 0
$CLIENT start-put-in --fps 15 --max-steps 0
$CLIENT status
$CLIENT stop
```

### 测试与代码质量

安装开发依赖后：

```bash
pre-commit install
pre-commit run --all-files
pytest tests -vv --maxfail=10
make test-end-to-end DEVICE=cpu
```

快速迭代时建议运行聚焦测试：

```bash
pytest tests/datasets/test_dataset_tools.py -vv
```

### 安全注意事项

- 不要提交密钥、数据集凭证、模型 checkpoint 或大体积生成文件。
- 真实机器人部署前先使用 `--dry-run` 检查数据和推理流程。
- 首次运行保持较低速度和较小动作增量，确认急停可用。
- 保持数据集转换、训练配置和部署脚本的 `state_dim/action_dim` 一致。
- 确认 `domain_id`、`robot_name`、`fsm_state` 与现场机器人配置一致。

### 更多文档

- `gr2_env/ENVIRONMENT_USAGE_GUIDE.md`
- `gr2_env/FOURIER_GR2_CONFIG_SUMMARY.md`
- `gr2_env/FOURIER_GR2_TRAINING_CONFIG.md`
- `gr2_env/FOURIER_GR2_DEPLOYMENT_CONFIG.md`
- `gr2_env/FOURIER_GR2_DATASET_CONFIG.md`
- `scripts/README_replay_gr2.md`
- `scripts/inference/README.md`
- `scripts/train/README.md`
