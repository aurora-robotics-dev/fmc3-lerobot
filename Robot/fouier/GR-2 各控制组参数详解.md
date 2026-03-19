# GR-2 各控制组参数详解

GR-2 共有 **8 个控制组**，合计 **41 个自由度**。SDK 以控制组为最小控制单位，每次需对整个控制组的所有关节一起下发指令。

所有关节位置单位为 **弧度（rad）**。

---

## 1. `left_leg` — 左腿（6 DOF）

| 索引 | 关节名 | 含义 | 运动方向 | 下限 (rad) | 上限 (rad) | 速度上限 (rad/s) | 扭矩上限 (Nm) |
|------|------|------|---------|-----------|-----------|----------------|-------------|
| 0 | `left_hip_pitch_joint` | 髋关节俯仰 | 前踢 / 后蹬 | -2.618 | 2.618 | 6.50 | 366.05 |
| 1 | `left_hip_roll_joint` | 髋关节侧摆 | 腿外展 / 内收 | -0.593 | 1.571 | 12.36 | 95.47 |
| 2 | `left_hip_yaw_joint` | 髋关节旋转 | 腿内旋 / 外旋 | -0.698 | 1.571 | 14.75 | 54.33 |
| 3 | `left_knee_pitch_joint` | 膝关节俯仰 | 屈膝 / 伸膝 | -0.087 | 2.356 | 6.50 | 366.05 |
| 4 | `left_ankle_pitch_joint` | 踝关节俯仰 | 脚尖上翘 / 下压 | -0.785 | 0.785 | 14.75 | 54.33 |
| 5 | `left_ankle_roll_joint` | 踝关节侧摆 | 脚内翻 / 外翻 | -0.384 | 0.384 | 16.76 | 29.84 |

```python
client.set_joint_positions({
    "left_leg": [hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll]
})
```

---

## 2. `right_leg` — 右腿（6 DOF）

| 索引 | 关节名 | 含义 | 运动方向 | 下限 (rad) | 上限 (rad) | 速度上限 (rad/s) | 扭矩上限 (Nm) |
|------|------|------|---------|-----------|-----------|----------------|-------------|
| 0 | `right_hip_pitch_joint` | 髋关节俯仰 | 前踢 / 后蹬 | -2.618 | 2.618 | 6.50 | 366.05 |
| 1 | `right_hip_roll_joint` | 髋关节侧摆 | 腿外展 / 内收 | -1.571 | 0.593 | 12.36 | 95.47 |
| 2 | `right_hip_yaw_joint` | 髋关节旋转 | 腿内旋 / 外旋 | -1.571 | 0.698 | 14.75 | 54.33 |
| 3 | `right_knee_pitch_joint` | 膝关节俯仰 | 屈膝 / 伸膝 | -0.087 | 2.356 | 6.50 | 366.05 |
| 4 | `right_ankle_pitch_joint` | 踝关节俯仰 | 脚尖上翘 / 下压 | -0.785 | 0.785 | 14.75 | 54.33 |
| 5 | `right_ankle_roll_joint` | 踝关节侧摆 | 脚内翻 / 外翻 | -0.384 | 0.384 | 16.76 | 29.84 |

```python
client.set_joint_positions({
    "right_leg": [hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll]
})
```

> **注意**：左右腿的 `hip_roll` 和 `hip_yaw` 限位是镜像对称的。

---

## 3. `waist` — 腰部（1 DOF）

| 索引 | 关节名 | 含义 | 运动方向 | 下限 (rad) | 上限 (rad) | 速度上限 (rad/s) | 扭矩上限 (Nm) |
|------|------|------|---------|-----------|-----------|----------------|-------------|
| 0 | `waist_yaw_joint` | 腰部旋转 | 向左转 / 向右转 | -2.618 | 2.618 | 7.76 | 74.45 |

```python
client.set_joint_positions({
    "waist": [yaw]
})
```

---

## 4. `head` — 头部（2 DOF）

| 索引 | 关节名 | 含义 | 运动方向 | 下限 (rad) | 上限 (rad) | 速度上限 (rad/s) | 扭矩上限 (Nm) |
|------|------|------|---------|-----------|-----------|----------------|-------------|
| 0 | `head_yaw_joint` | 头部旋转 | 向左转头 / 向右转头 | -1.396 | 1.396 | 9.16 | 17.33 |
| 1 | `head_pitch_joint` | 头部俯仰 | 低头 / 抬头 | -0.524 | 0.524 | 9.16 | 17.33 |

```python
client.set_joint_positions({
    "head": [yaw, pitch]
})
```

---

## 5. `left_manipulator` — 左臂（7 DOF）

| 索引 | 关节名 | 含义 | 运动方向 | 下限 (rad) | 上限 (rad) | 速度上限 (rad/s) | 扭矩上限 (Nm) |
|------|------|------|---------|-----------|-----------|----------------|-------------|
| 0 | `left_shoulder_pitch_joint` | 肩关节俯仰 | 手臂前举 / 后伸 | -2.967 | 2.967 | 7.76 | 74.45 |
| 1 | `left_shoulder_roll_joint` | 肩关节侧摆 | 手臂侧举 / 内收 | -0.524 | 2.793 | 7.76 | 74.45 |
| 2 | `left_shoulder_yaw_joint` | 肩关节旋转 | 大臂内旋 / 外旋 | -1.833 | 1.833 | 6.28 | 42.75 |
| 3 | `left_elbow_pitch_joint` | 肘关节俯仰 | 屈肘 / 伸肘 | -1.527 | 0.480 | 6.28 | 42.75 |
| 4 | `left_wrist_yaw_joint` | 腕关节旋转 | 手腕左右转 | -1.833 | 1.833 | 9.16 | 17.33 |
| 5 | `left_wrist_pitch_joint` | 腕关节俯仰 | 手腕上下翻 | -0.611 | 0.611 | 9.16 | 17.33 |
| 6 | `left_wrist_roll_joint` | 腕关节翻滚 | 掌心朝上 / 朝下 | -0.960 | 0.960 | 9.16 | 17.33 |

```python
client.set_joint_positions({
    "left_manipulator": [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, wrist_yaw, wrist_pitch, wrist_roll]
})
```

---

## 6. `right_manipulator` — 右臂（7 DOF）

| 索引 | 关节名 | 含义 | 运动方向 | 下限 (rad) | 上限 (rad) | 速度上限 (rad/s) | 扭矩上限 (Nm) |
|------|------|------|---------|-----------|-----------|----------------|-------------|
| 0 | `right_shoulder_pitch_joint` | 肩关节俯仰 | 手臂前举 / 后伸 | -2.967 | 2.967 | 7.76 | 74.45 |
| 1 | `right_shoulder_roll_joint` | 肩关节侧摆 | 手臂侧举 / 内收 | -2.793 | 0.524 | 7.76 | 74.45 |
| 2 | `right_shoulder_yaw_joint` | 肩关节旋转 | 大臂内旋 / 外旋 | -1.833 | 1.833 | 6.28 | 42.75 |
| 3 | `right_elbow_pitch_joint` | 肘关节俯仰 | 屈肘 / 伸肘 | -1.527 | 0.480 | 6.28 | 42.75 |
| 4 | `right_wrist_yaw_joint` | 腕关节旋转 | 手腕左右转 | -1.833 | 1.833 | 9.16 | 17.33 |
| 5 | `right_wrist_pitch_joint` | 腕关节俯仰 | 手腕上下翻 | -0.611 | 0.611 | 9.16 | 17.33 |
| 6 | `right_wrist_roll_joint` | 腕关节翻滚 | 掌心朝上 / 朝下 | -0.960 | 0.960 | 9.16 | 17.33 |

```python
client.set_joint_positions({
    "right_manipulator": [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, wrist_yaw, wrist_pitch, wrist_roll]
})
```

> **注意**：左右臂的 `shoulder_roll` 限位是镜像对称的。

---

## 7. `left_hand` — 左手（6 DOF）

| 索引 | 含义 | 张开值 | 握紧值 |
|------|------|-------|-------|
| 0 | 食指弯曲 | 0.2 | 1.7 |
| 1 | 中指弯曲 | 0.2 | 1.7 |
| 2 | 无名指弯曲 | 0.2 | 1.7 |
| 3 | 小指弯曲 | 0.2 | 1.7 |
| 4 | 拇指弯曲 | 1.2 | 0.0 |
| 5 | 拇指旋转 | 0.0 | 0.0 |

```python
# 张开手掌
client.set_joint_positions({"left_hand": [0.2, 0.2, 0.2, 0.2, 1.2, 0.0]})

# 握拳
client.set_joint_positions({"left_hand": [1.7, 1.7, 1.7, 1.7, 0.0, 0.0]})
```

---

## 8. `right_hand` — 右手（6 DOF）

参数含义与 `left_hand` 相同。

```python
# 张开手掌
client.set_joint_positions({"right_hand": [0.2, 0.2, 0.2, 0.2, 1.2, 0.0]})

# 握拳
client.set_joint_positions({"right_hand": [1.7, 1.7, 1.7, 1.7, 0.0, 0.0]})
```

---

## 完整示例：同时控制所有控制组

```python
import time
from fourier_aurora_client import AuroraClient

client = AuroraClient.get_instance(domain_id=123, robot_name="gr2t2v2", serial_number=None)
time.sleep(1)

# 切换到 PdStand 状态
client.set_fsm_state(2)
time.sleep(1.0)

# 读取所有控制组当前状态
init_pos = {
    "left_leg":          client.get_group_state("left_leg"),           # 6 DOF
    "right_leg":         client.get_group_state("right_leg"),          # 6 DOF
    "waist":             client.get_group_state("waist"),              # 1 DOF
    "head":              client.get_group_state("head"),               # 2 DOF
    "left_manipulator":  client.get_group_state("left_manipulator"),   # 7 DOF
    "right_manipulator": client.get_group_state("right_manipulator"),  # 7 DOF
    "left_hand":         [0.2, 0.2, 0.2, 0.2, 1.2, 0.0],             # 6 DOF
    "right_hand":        [0.2, 0.2, 0.2, 0.2, 1.2, 0.0],             # 6 DOF
}

# 设置目标位置（全部回零位）
target_pos = {
    "left_leg":          [0, 0, 0, 0, 0, 0],
    "right_leg":         [0, 0, 0, 0, 0, 0],
    "waist":             [0],
    "head":              [0, 0],
    "left_manipulator":  [0, 0, 0, 0, 0, 0, 0],
    "right_manipulator": [0, 0, 0, 0, 0, 0, 0],
    "left_hand":         [0.2, 0.2, 0.2, 0.2, 1.2, 0.0],
    "right_hand":        [0.2, 0.2, 0.2, 0.2, 1.2, 0.0],
}

# 使用线性插值平滑运动（100Hz，持续 2 秒）
frequency = 100
duration = 2.0
total_steps = int(frequency * duration)

for step in range(total_steps + 1):
    positions = {}
    for group in init_pos:
        positions[group] = [
            i + (t - i) * step / total_steps
            for i, t in zip(init_pos[group], target_pos[group])
        ]
    client.set_joint_positions(positions)
    time.sleep(1 / frequency)

client.close()
```

---

## 安全提示

1. **务必使用插值**：直接跳变到目标位置可能导致关节冲击，始终使用线性插值或更平滑的轨迹规划。
2. **控制频率**：推荐 100Hz（`time.sleep(0.01)`）。
3. **先读后写**：发送指令前先用 `get_group_state()` 获取当前状态，避免从未知位置开始运动。
4. **遵守限位**：超出关节限位的指令不会被执行，请参照上表中的上下限。
5. **FSM 状态**：
   - `2`（PdStand）：上半身控制，适合手臂和手部操作。
   - `10`（UserCmd）：全身关节控制。
   - `11`（UpperBodyUserCmd）：仅上半身，腿部禁用。
