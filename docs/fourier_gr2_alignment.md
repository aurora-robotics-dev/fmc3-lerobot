# Fourier GR-2 模型对齐指南

本文档用于指导在 LeRobot 框架下训练各种策略时，如何与傅里叶 GR-2 机器人对齐。

---

## 1. GR-2 硬件规格

- 身高 1.75m，体重 63kg
- 总自由度：最高 53 DOF（含 12-DOF 灵巧手）
- 控制频率：30 Hz
- 单臂负载：3 kg

---

## 2. 关节组定义（官方 6 组 + 灵巧手）

来源：[傅里叶官方文档](https://support.fftai.com/en/docs/GR-X-Humanoid-Robot/GR2/GR-2_Introduction/)

| 控制组 | 关节名称 | DOF |
|---|---|---|
| **left_manipulator** | left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow_pitch, left_wrist_yaw, left_wrist_pitch, left_wrist_roll | 7 |
| **right_manipulator** | right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow_pitch, right_wrist_yaw, right_wrist_pitch, right_wrist_roll | 7 |
| **left_hand** | L_pinky_proximal, L_ring_proximal, L_middle_proximal, L_index_proximal, L_thumb_proximal_pitch, L_thumb_proximal_yaw | 6 |
| **right_hand** | R_pinky_proximal, R_ring_proximal, R_middle_proximal, R_index_proximal, R_thumb_proximal_pitch, R_thumb_proximal_yaw | 6 |
| **head** | head_yaw, head_pitch | 2 |
| **waist** | waist_yaw, waist_roll, waist_pitch | 3 (数据集) / 1 (官方SDK仅 yaw) |
| **left_leg** | left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee_pitch, left_ankle_pitch, left_ankle_roll | 6 |
| **right_leg** | right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee_pitch, right_ankle_pitch, right_ankle_roll | 6 |

> 注意：当前数据集采集方案未包含腿部关节，用底盘速度指令替代。

---

## 3. 数据集 Action 空间（37 维）

索引 → 关节名 → 控制组 的完整映射：

```
索引   关节名称                        控制组
────────────────────────────────────────────────────
 0     left_shoulder_pitch_joint       left_manipulator
 1     left_shoulder_roll_joint        left_manipulator
 2     left_shoulder_yaw_joint         left_manipulator
 3     left_elbow_pitch_joint          left_manipulator
 4     left_wrist_yaw_joint            left_manipulator
 5     left_wrist_pitch_joint          left_manipulator
 6     left_wrist_roll_joint           left_manipulator
 7     right_shoulder_pitch_joint      right_manipulator
 8     right_shoulder_roll_joint       right_manipulator
 9     right_shoulder_yaw_joint        right_manipulator
10     right_elbow_pitch_joint         right_manipulator
11     right_wrist_yaw_joint           right_manipulator
12     right_wrist_pitch_joint         right_manipulator
13     right_wrist_roll_joint          right_manipulator
14     L_pinky_proximal_joint          left_hand
15     L_ring_proximal_joint           left_hand
16     L_middle_proximal_joint         left_hand
17     L_index_proximal_joint          left_hand
18     L_thumb_proximal_pitch_joint    left_hand
19     L_thumb_proximal_yaw_joint      left_hand
20     R_pinky_proximal_joint          right_hand
21     R_ring_proximal_joint           right_hand
22     R_middle_proximal_joint         right_hand
23     R_index_proximal_joint          right_hand
24     R_thumb_proximal_pitch_joint    right_hand
25     R_thumb_proximal_yaw_joint      right_hand
26     head_yaw_joint                  head
27     head_pitch_joint                head
28     waist_yaw_joint                 waist
29     waist_roll_joint                waist
30     waist_pitch_joint               waist
31     base_vel_x                      base（速度指令）
32     base_vel_y                      base
33     base_vel_yaw                    base
34     base_vel_height                 base
35     base_vel_pitch                  base
36     base_base_yaw                   base
```

**分组切片（用于部署脚本）：**

```python
ACTION_SLICES = {
    "left_manipulator":  slice(0, 7),    # 7 DOF
    "right_manipulator": slice(7, 14),   # 7 DOF
    "left_hand":         slice(14, 20),  # 6 DOF
    "right_hand":        slice(20, 26),  # 6 DOF
    "head":              slice(26, 28),  # 2 DOF
    "waist":             slice(28, 31),  # 3 DOF
}
BASE_VEL_SLICE = slice(31, 37)           # 6 维速度指令
```

---

## 4. 数据集 State 空间（45 维）

```
索引   关节名称                        说明
────────────────────────────────────────────────────
 0-6   left_shoulder/elbow/wrist       左臂 7 关节位置
 7-13  right_shoulder/elbow/wrist      右臂 7 关节位置
14-15  head_yaw, head_pitch            头部 2 关节位置
16     waist_yaw_joint                 腰部关节位置
17-22  L_index/middle/ring/pinky/thumb 左手 6 关节位置
23-28  R_index/middle/ring/pinky/thumb 右手 6 关节位置
29-31  base_pos_x/y/z                  底盘位置
32-35  base_quat_x/y/z/w              底盘四元数姿态
36-38  base_rpy_roll/pitch/yaw         底盘欧拉角
39-41  imu_acc_x/y/z                   IMU 加速度
42-44  imu_omega_x/y/z                 IMU 角速度
```

> 注意：State 中关节排列顺序与 Action 不同！State 的手部关节（17-28）排在底盘信息之前，且腰部只有 1 维。

---

## 5. 相机配置

| 相机 | 分辨率 | FPS | 编码 | 说明 |
|---|---|---|---|---|
| `observation.images.camera_top` | 640×480 | 30 | H.264 | RGB 相机 |
| `observation.images.camera_top_depth` | 640×480 | 30 | H.264 | 深度图（伪彩色） |

硬件：Orbbec Gemini 335Lg，通过 OpenCV 采集。

---

## 6. 各策略对齐配置

### 通用参数（所有策略共用）

```bash
--dataset.repo_id=fourier_gr2_pick_place
--dataset.root=/home/phl/workspace/dataset/fourier/pick_and_place
--dataset.video_backend=torchcodec
```

### 6.1 轻量策略（从头训练，无需预训练模型）

| 策略 | action_dim | state_dim | 需要设置 | 推荐 batch_size |
|---|---|---|---|---|
| **ACT** | 自动从数据集推断 | 自动 | `--policy.type=act` | 256 |
| **Diffusion** | 自动 | 自动 | `--policy.type=diffusion` | 128 |
| **VQ-BeT** | 自动 | 自动 | `--policy.type=vqbet` | 128 |
| **TD-MPC** | 自动 | 自动 | `--policy.type=tdmpc` | 256 |
| **SAC** | 自动 | 自动 | `--policy.type=sac` | 256 |

这些策略从头训练，LeRobot 会自动从数据集读取维度，**不需要手动设置 action/state 维度**。

### 6.2 VLA 策略（需要预训练模型 + 手动维度对齐）

这些策略使用预训练 VLM 骨干，**必须手动指定 `max_action_dim` 和 `max_state_dim`**，因为模型内部会将实际维度 pad 到这两个值。

| 策略 | max_action_dim | max_state_dim | 预训练模型 | 环境 |
|---|---|---|---|---|
| **PI0** | 37 | 45 | `/home/phl/workspace/models/pi0` | lerobot-pi0 |
| **PI0.5** | 37 | 45 | 需下载 pi05 权重 | lerobot-pi0 |
| **PI0-Fast** | 37 | 45 | 需下载 pi0_fast 权重 | lerobot-pi0 |
| **SmolVLA** | 37 | 45 | `/home/phl/workspace/models/SmolVLM2-500M-Video-Instruct` | lerobot |
| **Wall-X** | 37 | 45 | 需下载 wall-oss-flow 权重 | lerobot |
| **XVLA** | 37 | 45 | 不需要（Florence2 自动下载）| lerobot |
| **GR00T** | 37 | 45 | 需下载 GR00T-N1.5-3B | lerobot |

**关键参数示例（以 PI0 为例）：**

```bash
lerobot-train \
    --policy.path=/home/phl/workspace/models/pi0 \
    --policy.max_action_dim=37 \    # ← 必须手动指定！
    --policy.max_state_dim=45 \     # ← 必须手动指定！
    --policy.use_amp=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    ...
```

### 6.3 预训练权重适配说明

当预训练模型的 action/state 维度与 GR-2 不匹配时（几乎所有公开权重都是这样），以下投影层会**随机初始化**：

| 层 | 说明 |
|---|---|
| `state_proj` | 将 state 投射到模型隐层，形状 [hidden, max_state_dim] |
| `action_in_proj` | 将 action 投射到模型隐层，形状 [hidden, max_action_dim] |
| `action_out_proj` | 将隐层投射回 action，形状 [max_action_dim, hidden] |

其他层（VLM 骨干、Transformer、注意力层等，占参数量 99%+）正常加载。

**影响**：训练初期 loss 会偏高，需要 1000-3000 步让投影层收敛。整体训练效果取决于数据量。

---

## 7. 部署时的关节映射

部署脚本 (`scripts/deploy_gr2_act.py`) 中，模型输出的 37 维 action 需要映射到 GR-2 SDK 的控制接口：

```python
# 模型输出 → GR2Robot SDK
for group_name, s in ACTION_SLICES.items():
    joint_targets[group_name] = action[s].tolist()

# 底盘速度单独处理
base_vel = action[31:37]
robot.set_velocity(vx=base_vel[0], vy=base_vel[1], vyaw=base_vel[2])
```

部署时的 State 采集顺序必须与训练数据一致（参考第 4 节）。

---

## 8. 换模型检查清单

换用新策略时，按以下步骤检查对齐：

- [ ] **维度对齐**：确认 `max_action_dim=37`，`max_state_dim=45`（VLA 策略必设）
- [ ] **相机名称**：确认策略的 `input_features` 包含 `observation.images.camera_top`
- [ ] **归一化模式**：确认 ACTION 和 STATE 使用 `MEAN_STD`（PI0/SmolVLA），或 `QUANTILES`（PI0.5）
- [ ] **chunk_size**：PI0/SmolVLA/PI0.5 默认 50，ACT 默认 100，按需调整
- [ ] **conda 环境**：PI0/PI0.5/PI0-Fast 需要 `lerobot-pi0`，其他用 `lerobot`
- [ ] **预训练路径**：确认模型文件存在（config.json + model.safetensors）
- [ ] **部署脚本**：非 ACT 类的 VLA 策略需要传 `--task` 参数（语言指令）

---

## 9. 参考链接

- 傅里叶 GR-2 官方文档：https://support.fftai.com/en/docs/GR-X-Humanoid-Robot/GR2/GR-2_Introduction/
- URDF 模型文件：https://github.com/FFTAI/Wiki-GRx-Models/tree/master/GRX/GR2
- fourier-grx-client SDK：https://github.com/FFTAI/fourier-grx-client
- openpi (PI0 原始实现)：https://github.com/Physical-Intelligence/openpi
- LeRobot 框架：https://github.com/huggingface/lerobot
