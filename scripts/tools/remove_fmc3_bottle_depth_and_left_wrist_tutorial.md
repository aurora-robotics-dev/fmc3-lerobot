# 删除两个 FMC3 瓶子数据集的深度信息和左手腕相机

本文记录如何使用 `lerobot-edit-dataset` 清理下面两个本地 LeRobot 数据集：

- `fmc3_gr2_black_capped_bottle_green_to_yellow_lerobot`
- `fmc3_gr2_black_capped_bottle_yellow_to_green_lerobot`

目标：

- 删除所有深度相机特征
- 删除左手腕 RGB 相机特征 `observation.images.camera_left_wrist`

最终会删除这 4 个特征：

- `observation.images.camera_top_depth`
- `observation.images.camera_left_wrist`
- `observation.images.camera_left_wrist_depth`
- `observation.images.camera_right_wrist_depth`

## 1. 环境

使用 conda 环境：

```bash
conda activate lerobot-pi0
```

或者直接用 `conda run` 执行命令。

## 2. 执行命令

### 数据集 1

```bash
conda run --no-capture-output -n lerobot-pi0 lerobot-edit-dataset \
  --repo_id . \
  --root /home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_yellow-to-green_3_25 \
  --operation.type remove_feature \
  --operation.feature_names "['observation.images.camera_top_depth','observation.images.camera_left_wrist','observation.images.camera_left_wrist_depth','observation.images.camera_right_wrist_depth']"
```

### 数据集 2

```bash
conda run --no-capture-output -n lerobot-pi0 lerobot-edit-dataset \
  --repo_id . \
  --root /home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_black_capped_bottle_yellow_to_green_lerobot \
  --operation.type remove_feature \
  --operation.feature_names "['observation.images.camera_top_depth','observation.images.camera_left_wrist','observation.images.camera_left_wrist_depth','observation.images.camera_right_wrist_depth']"
```

## 3. 重要说明

- 这两个命令都要执行。
- `--repo_id .` 这一点很重要。这里处理的是本地目录数据集，不是 Hugging Face 上的 repo。
- `--root` 必须直接指向数据集根目录，也就是包含 `data/`、`meta/`、`videos/` 的那一层目录。
- 该操作会原地改写数据集。

## 4. 删除后检查

可以用下面的命令检查剩余特征：

```bash
conda run --no-capture-output -n lerobot-pi0 python - <<'PY'
from pathlib import Path
import json

for path in [
    "/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_black_capped_bottle_green_to_yellow_lerobot",
    "/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_black_capped_bottle_yellow_to_green_lerobot",
]:
    info = json.loads((Path(path) / "meta/info.json").read_text())
    print(path)
    for key in info["features"].keys():
        print(" ", key)
    print()
PY
```

清理完成后，理论上不应再看到下面这些特征：

- `observation.images.camera_top_depth`
- `observation.images.camera_left_wrist`
- `observation.images.camera_left_wrist_depth`
- `observation.images.camera_right_wrist_depth`
