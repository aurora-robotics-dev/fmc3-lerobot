# 训练脚本使用说明

## 环境要求

- conda 环境: `lerobot`
- GPU: 建议显存 >= 24GB
- AutoDL 学术加速（脚本内已自动开启）

## 策略列表

### 1. pi0 微调 - 抓取物体放入盒子

- 脚本: `scripts/train_pi0_grab_box.sh`
- 基座模型: `lerobot/pi0`（HuggingFace 自动下载）
- 数据集: `puheliang/lerobot_fmc3_grab_box_v2`（401 episodes, SO-101 机械臂, top + wrist 双相机）
- 任务: Pick up the tape and place it in the box

**首次训练:**
```bash
conda activate lerobot
bash scripts/train_pi0_grab_box.sh
```

**断点续训:**
```bash
conda activate lerobot
bash scripts/train_pi0_grab_box.sh --resume
```

**后台运行（推荐）:**
```bash
conda activate lerobot
screen -S train
bash scripts/train_pi0_grab_box.sh
# 按 Ctrl+A D 脱离 screen
# 回来查看: screen -r train
```

**主要参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| STEPS | 50000 | 总训练步数 |
| BATCH_SIZE | 8 | 批大小 |
| DTYPE | bfloat16 | 训练精度 |
| SAVE_FREQ | 20000 | checkpoint 保存频率 |
| PUSH_TO_HUB | true | 训练完成后推送到 HuggingFace |
| WANDB_ENABLE | true | 启用 WandB 日志 |

**训练输出:** `/root/output_lerobot_train/grab_box/pi0_fmc3`

**WandB 看板:** 项目名 `Lerobot_Zihao_Project`

启动

## GR2 PI0 推理服务（RoboOS 对接）

新增脚本：`scripts/gr2_pi0_inference_service.py`

启动示例：

```bash
conda run --no-capture-output -n lerobot-pi0 \
  python scripts/gr2_pi0_inference_service.py \
  --unix-socket-path /tmp/gr2_pi0_inference_service.sock \
  --checkpoint-path /home/phl/workspace/lerobot-versions/lerobot/outputs/train/pi0_gr2_pick_3_4_20260306_185911/checkpoints/070000/pretrained_model \
  --robot-name gr2 \
  --domain-id 123
```

通信方式：
- Unix Domain Socket（进程间通信）
- 默认 socket: `/tmp/gr2_pi0_inference_service.sock`

## 数据集清洗 - 交互式 Episode 筛选

脚本: `scripts/review_episodes.py`

逐个回放 episode，通过 Rerun GUI 可视化查看，在终端标记保留或删除。支持上一个/下一个导航，进度自动保存，中断后可继续。

**启动:**
```bash
conda activate lerobot-pi0
python scripts/review_episodes.py \
    --repo-id fmc3_gr2_grab_bottle_into_box_lerobot_ds \
    --root /home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_into_box_lerobot_ds
```

**键盘操作:**

| 按键 | 功能 |
|------|------|
| `→` 或 `n` 或回车 | 保留当前 episode，跳到下一个 |
| `↓` 或 `d` | 标记删除当前 episode，跳到下一个 |
| `←` 或 `p` | 回到上一个 episode 重新审核 |
| `↑` 或 `r` | 重播当前 episode |
| `j` + 数字 | 跳转到第 N 个 episode |
| `q` | 退出并保存 |

**参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --repo-id | (必填) | 数据集 repo id |
| --root | (必填) | 数据集本地路径 |
| --start-episode | 0 | 起始 episode 索引 |
| --output | episodes_to_delete.json | 审核结果保存路径 |

**Rerun GUI 多视角显示:**
首次打开 Rerun GUI 可能只显示一个相机视角，在左侧 Blueprint 面板展开 `observation` → `images`，勾选其他相机即可同时查看多视角。

**审核结果:** 保存在 `episodes_to_delete.json`，格式如下:
```json
{
  "delete": [3, 7, 15],
  "keep": [0, 1, 2, 4, 5, 6]
}
```

**确认后执行清洗:**
```bash
lerobot-edit-dataset delete_episodes \
    --dataset.repo_id=fmc3_gr2_grab_bottle_into_box_lerobot_ds \
    --dataset.root=/path/to/dataset \
    --episode_indices='[3, 7, 15]' \
    --output_dir=/path/to/cleaned_dataset
```
