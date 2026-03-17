# GR-2 数据集回放脚本使用指南

## 概述

`replay_gr2_dataset.py` 是专为 Fourier GR-2 机器人设计的数据集回放工具，可以从 LeRobot v3.0 格式的数据集中读取 episode 并在真实机器人上重现动作序列。

## 功能特性

✅ **多数据源支持**
- 本地数据集路径
- HuggingFace Hub 数据集

✅ **安全保护**
- 关节限位自动夹紧
- 步间动作平滑（防止突变）
- 首帧平滑过渡（避免机器人跳变）

✅ **灵活控制**
- 可调节回放帧率
- 支持部分帧回放（起始帧 + 最大帧数）
- Dry-run 模式（无需连接机器人）

✅ **详细日志**
- 实时回放进度
- 动作范数监控
- 异常诊断信息

---

## 安装依赖

### 1. LeRobot 环境
```bash
conda activate lerobot
cd /home/phl/workspace/lerobot-versions/lerobot
pip install -e .
```

### 2. Aurora SDK（仅真实机器人需要）
```bash
# 安装 Fourier Aurora Python 客户端
pip install fourier-aurora-client
```

---

## 使用方法

### 基础用法

#### 回放本地数据集
```bash
python scripts/replay_gr2_dataset.py \
    --dataset-path ./outputs/dataset/gr2_pick_bottle \
    --episode 0 \
    --domain-id 123
```

#### 回放 HuggingFace 数据集
```bash
python scripts/replay_gr2_dataset.py \
    --repo-id username/gr2-pick-bottle \
    --episode 0 \
    --domain-id 123
```

#### Dry-run 模式（测试数据集加载）
```bash
python scripts/replay_gr2_dataset.py \
    --dataset-path ./outputs/dataset/gr2_pick_bottle \
    --episode 0 \
    --dry-run \
    --verbose
```

---

### 高级参数

#### 回放控制
```bash
# 自定义帧率（默认 30fps）
--fps 20.0

# 从第 100 帧开始回放
--start-frame 100

# 最多回放 500 帧
--max-frames 500

# 首帧平滑过渡时间（默认 3 秒）
--transition-time 5.0
```

#### 安全参数
```bash
# 单步最大关节变化量（默认 0.1 rad）
--max-joint-delta 0.05

# 禁用关节限位夹紧（危险！仅用于调试）
--disable-clamp
```

#### 机器人连接
```bash
# 自定义 DDS 域 ID（默认 123）
--domain-id 456

# 自定义机器人名称（默认 gr2）
--robot-name gr2_test
```

#### 调试选项
```bash
# 详细日志输出
--verbose

# Dry-run 模式（不连接机器人）
--dry-run
```

---

## 完整示例

### 示例 1：回放训练数据集的第一个 episode
```bash
python scripts/replay_gr2_dataset.py \
    --dataset-path /home/phl/workspace/lerobot-versions/lerobot/outputs/dataset/gr2_pick_3_4 \
    --episode 0 \
    --domain-id 123 \
    --fps 30.0 \
    --transition-time 3.0 \
    --verbose
```

### 示例 2：快速验证数据集格式（无需机器人）
```bash
python scripts/replay_gr2_dataset.py \
    --dataset-path ./outputs/dataset/gr2_pick_bottle \
    --episode 0 \
    --dry-run \
    --verbose
```

### 示例 3：慢速回放（便于观察）
```bash
python scripts/replay_gr2_dataset.py \
    --dataset-path ./outputs/dataset/gr2_pick_bottle \
    --episode 0 \
    --domain-id 123 \
    --fps 10.0 \
    --max-joint-delta 0.05
```

### 示例 4：回放 episode 的中间片段
```bash
python scripts/replay_gr2_dataset.py \
    --dataset-path ./outputs/dataset/gr2_pick_bottle \
    --episode 0 \
    --domain-id 123 \
    --start-frame 50 \
    --max-frames 200
```

---

## 数据集格式要求

脚本支持 **LeRobot v3.0** 格式的数据集，目录结构如下：

```
dataset_name/
├── meta/
│   ├── info.json          # 特征定义、fps、robot_type
│   ├── stats.json         # 归一化统计
│   └── tasks.parquet      # 任务描述
├── data/
│   └── chunk-000/
│       └── file-000.parquet   # 帧级数据（action, state, timestamps）
└── videos/
    └── camera_top/
        └── chunk-000/
            └── file-000.mp4
```

### 动作维度（35D URDF 空间）
| 索引 | 关节组 | 维度 | 说明 |
|------|--------|------|------|
| 0:7 | left_manipulator | 7 | 左臂关节 |
| 7:14 | right_manipulator | 7 | 右臂关节 |
| 14:20 | left_hand | 6 | 左手关节 |
| 20:26 | right_hand | 6 | 右手关节 |
| 26:28 | head | 2 | 头部关节（yaw, pitch） |
| 28:29 | waist | 1 | 腰部关节（yaw） |
| 29:35 | base | 6 | 底盘（x, y, z, roll, pitch, yaw） |

---

## 安全注意事项

⚠️ **在真实机器人上回放前，请务必：**

1. **检查数据集质量**
   ```bash
   # 先用 dry-run 模式验证数据集可以正常加载
   python scripts/replay_gr2_dataset.py --dataset-path <path> --episode 0 --dry-run --verbose
   ```

2. **确认机器人状态**
   - 机器人已上电且处于安全位置
   - 周围无障碍物和人员
   - 急停按钮触手可及

3. **使用平滑过渡**
   - 保持 `--transition-time` 至少 3 秒
   - 首次回放时使用较小的 `--max-joint-delta`（如 0.05）

4. **逐步测试**
   - 先回放少量帧（`--max-frames 50`）
   - 确认无异常后再回放完整 episode

5. **监控日志**
   - 使用 `--verbose` 查看详细动作信息
   - 注意 "Arm" 和 "Hand" 范数是否异常

---

## 故障排查

### 问题 1：`fourier_aurora_client not found`
**原因**：未安装 Aurora SDK
**解决**：
```bash
pip install fourier-aurora-client
# 或使用 dry-run 模式
python scripts/replay_gr2_dataset.py --dry-run ...
```

### 问题 2：`Dataset not found`
**原因**：数据集路径错误或数据集格式不正确
**解决**：
```bash
# 检查路径是否存在
ls -la /path/to/dataset

# 验证数据集格式
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset('/path/to/dataset'); print(ds)"
```

### 问题 3：`Episode X not found in dataset`
**原因**：指定的 episode 索引不存在
**解决**：
```bash
# 查看数据集中有哪些 episode
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset('/path/to/dataset'); print(f'Episodes: {ds.num_episodes}')"
```

### 问题 4：机器人连接失败
**原因**：DDS 域 ID 不匹配或网络问题
**解决**：
```bash
# 检查机器人服务端的 domain_id 配置
# 确保客户端和服务端使用相同的 domain_id

# 尝试不同的 domain_id
python scripts/replay_gr2_dataset.py --domain-id 456 ...
```

### 问题 5：机器人动作异常（抖动/突变）
**原因**：动作平滑参数不合适
**解决**：
```bash
# 增加平滑约束
python scripts/replay_gr2_dataset.py \
    --max-joint-delta 0.03 \
    --transition-time 5.0 \
    ...
```

---

## 与其他工具的对比

| 工具 | 用途 | 适用场景 |
|------|------|----------|
| `lerobot-replay` | 通用回放脚本 | 所有 LeRobot 支持的机器人 |
| `replay_gr2_dataset.py` | GR-2 专用回放 | Fourier GR-2 机器人 |
| `deploy_gr2_pi0.py` | PI0 策略部署 | 在线推理 + 执行 |
| `lerobot-dataset-viz` | 数据集可视化 | 离线查看数据集内容 |

---

## 开发者信息

### 代码结构
```python
# 主要函数
load_dataset()           # 加载 LeRobot 数据集
filter_episode()         # 提取指定 episode 的动作
clamp_joint_action()     # 关节限位夹紧
stabilize_action()       # 步间动作平滑
send_action_to_robot()   # 发送动作到机器人
smooth_transition()      # 首帧平滑过渡
replay_episode()         # 主回放循环
```

### 扩展建议
- 添加可视化窗口（显示当前帧的 RGB 图像）
- 支持多 episode 连续回放
- 添加动作录制功能（对比回放效果）
- 集成底盘控制（目前仅支持上半身）

---

## 相关文档

- [LeRobot 数据集格式](../docs/lerobot-dataset-v3.mdx)
- [GR-2 关节对齐文档](../docs/fourier_gr2_joint_alignment.mdx)
- [Aurora SDK 文档](../Robot/fouier/README.md)
- [数据集转换工具](../Robot/fouier/convert_dora_to_lerobot.py)

---

## 许可证

本脚本遵循 LeRobot 项目的 Apache 2.0 许可证。
