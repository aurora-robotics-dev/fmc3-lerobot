# GR2 推理适配指南

本文档是 Fourier GR2 机器人策略模型推理部署的**完整参考**。阅读本文档后，你应该能够：
1. 理解模型输入/输出每一维的含义
2. 知道如何从机器人读取观测、构造模型输入
3. 知道如何将模型输出转换为 SDK 指令下发给机器人

## 背景

- 使用外骨骼遥操采集数据（Dora-Record 格式）
- 通过 `convert_dora_to_lerobot.py` 转换为 LeRobot v3.0 格式
- 使用 LeRobot 框架训练策略模型（ACT / Diffusion / PI0 / SmolVLA 等）
- 转换时关节顺序已对齐 GR2 SDK 控制组，部署时可直接按维度切片下发

## SDK 基础

```python
from fourier_aurora_client import AuroraClient

client = AuroraClient.get_instance(domain_id=123, robot_name="gr2t2v2", serial_number=None)
```

- `client.get_group_state(group_name)` → 读取控制组当前关节位置（list[float]，单位弧度）
- `client.set_joint_positions(dict)` → 下发关节位置指令
- `client.set_fsm_state(state_id)` → 切换状态机（2=PdStand 上半身, 10=UserCmd 全身, 11=UpperBodyUserCmd）

## 模型输出：action（35维）

模型每一步输出一个 35 维向量，含义如下：

### 关节位置指令（29维，维度 0-28，单位：弧度）

| 切片 | 维度 | SDK 控制组 | 关节名 | 关节限位 (rad) |
|------|------|-----------|--------|---------------|
| `[0:7]` | 0 | `left_manipulator` | left_shoulder_pitch_joint | [-2.967, 2.967] |
| | 1 | | left_shoulder_roll_joint | [-0.524, 2.793] |
| | 2 | | left_shoulder_yaw_joint | [-1.833, 1.833] |
| | 3 | | left_elbow_pitch_joint | [-1.527, 0.480] |
| | 4 | | left_wrist_yaw_joint | [-1.833, 1.833] |
| | 5 | | left_wrist_pitch_joint | [-0.611, 0.611] |
| | 6 | | left_wrist_roll_joint | [-0.960, 0.960] |
| `[7:14]` | 7 | `right_manipulator` | right_shoulder_pitch_joint | [-2.967, 2.967] |
| | 8 | | right_shoulder_roll_joint | [-2.793, 0.524] |
| | 9 | | right_shoulder_yaw_joint | [-1.833, 1.833] |
| | 10 | | right_elbow_pitch_joint | [-1.527, 0.480] |
| | 11 | | right_wrist_yaw_joint | [-1.833, 1.833] |
| | 12 | | right_wrist_pitch_joint | [-0.611, 0.611] |
| | 13 | | right_wrist_roll_joint | [-0.960, 0.960] |
| `[14:20]` | 14 | `left_hand` | L_index（食指） | [0.2, 1.7] |
| | 15 | | L_middle（中指） | [0.2, 1.7] |
| | 16 | | L_ring（无名指） | [0.2, 1.7] |
| | 17 | | L_pinky（小指） | [0.2, 1.7] |
| | 18 | | L_thumb_pitch（拇指弯曲） | [0.0, 1.2] |
| | 19 | | L_thumb_yaw（拇指旋转） | - |
| `[20:26]` | 20 | `right_hand` | R_index（食指） | [0.2, 1.7] |
| | 21 | | R_middle（中指） | [0.2, 1.7] |
| | 22 | | R_ring（无名指） | [0.2, 1.7] |
| | 23 | | R_pinky（小指） | [0.2, 1.7] |
| | 24 | | R_thumb_pitch（拇指弯曲） | [0.0, 1.2] |
| | 25 | | R_thumb_yaw（拇指旋转） | - |
| `[26:28]` | 26 | `head` | head_yaw_joint | [-1.396, 1.396] |
| | 27 | | head_pitch_joint | [-0.524, 0.524] |
| `[28:29]` | 28 | `waist` | waist_yaw_joint | [-2.618, 2.618] |

### 底盘速度指令（6维，维度 29-34）

| 维度 | 名称 | 说明 |
|------|------|------|
| 29 | base_vel_x | 前后线速度 |
| 30 | base_vel_y | 左右线速度 |
| 31 | base_vel_yaw | 偏航角速度 |
| 32 | base_vel_height | 升降速度 |
| 33 | base_vel_pitch | 俯仰角速度 |
| 34 | base_base_yaw | 底盘偏航角 |

### action 下发代码

```python
import numpy as np

def send_action(client, action: np.ndarray):
    """将模型输出的 35 维 action 下发给 GR2。"""
    # 关节位置（直接切片，顺序已对齐 SDK）
    client.set_joint_positions({
        "left_manipulator":  action[0:7].tolist(),
        "right_manipulator": action[7:14].tolist(),
        "left_hand":         action[14:20].tolist(),
        "right_hand":        action[20:26].tolist(),
        "head":              action[26:28].tolist(),
        "waist":             action[28:29].tolist(),
    })

    # 底盘速度
    client.set_velocity(
        vel_x=float(action[29]),
        vel_y=float(action[30]),
        vel_yaw=float(action[31]),
        vel_height=float(action[32]),
        vel_pitch=float(action[33]),
    )
```

## 模型输入：observation.state（45维）

模型输入的状态观测向量，从机器人实时读取。

### 关节状态（29维，维度 0-28，单位：弧度）

与 action 的维度 0-28 **完全相同的顺序和含义**：
- `[0:7]` → `left_manipulator` 当前关节位置
- `[7:14]` → `right_manipulator` 当前关节位置
- `[14:20]` → `left_hand` 当前关节位置
- `[20:26]` → `right_hand` 当前关节位置
- `[26:28]` → `head` 当前关节位置
- `[28:29]` → `waist` 当前关节位置

### 底盘状态（16维，维度 29-44）

| 维度 | 名称 | 说明 |
|------|------|------|
| 29-31 | base_pos_x/y/z | 底盘位置（米） |
| 32-35 | base_quat_x/y/z/w | 底盘四元数 |
| 36-38 | base_rpy_roll/pitch/yaw | 底盘欧拉角（弧度） |
| 39-41 | imu_acc_x/y/z | IMU 加速度（m/s²） |
| 42-44 | imu_omega_x/y/z | IMU 角速度（rad/s） |

### observation.state 构造代码

```python
import numpy as np

def get_observation_state(client) -> np.ndarray:
    """从 GR2 读取 45 维状态向量，顺序与训练数据一致。"""
    state = np.zeros(45, dtype=np.float32)

    # 关节状态（29维，维度 0-28）
    state[0:7]   = client.get_group_state("left_manipulator")
    state[7:14]  = client.get_group_state("right_manipulator")
    state[14:20] = client.get_group_state("left_hand")
    state[20:26] = client.get_group_state("right_hand")
    state[26:28] = client.get_group_state("head")
    state[28:29] = client.get_group_state("waist")

    # 底盘状态（16维，维度 29-44）
    # 需要根据实际 SDK 接口获取底盘位姿和 IMU 数据
    base_state = client.get_base_state()  # 具体接口视 SDK 版本而定
    state[29:32] = base_state["position"]       # x, y, z
    state[32:36] = base_state["quat"]           # qx, qy, qz, qw
    state[36:39] = base_state["rpy"]            # roll, pitch, yaw
    state[39:42] = base_state["imu_acc"]        # ax, ay, az
    state[42:45] = base_state["imu_omega"]      # wx, wy, wz

    return state
```

## 模型输入：图像观测

| 观测键名 | 来源 | 分辨率 | 格式 |
|----------|------|--------|------|
| `observation.images.camera_top` | 头顶 RGB 摄像头 | 480 x 640 x 3 | uint8, RGB |
| `observation.images.camera_top_depth` | 头顶深度摄像头 | 480 x 640 | uint16 深度图 |

## 关节限位表（用于 clamp 安全保护）

```python
# action 维度 0-28 的关节限位 [下限, 上限]（弧度）
JOINT_LIMITS = np.array([
    # left_manipulator (0-6)
    [-2.967,  2.967],  # left_shoulder_pitch
    [-0.524,  2.793],  # left_shoulder_roll
    [-1.833,  1.833],  # left_shoulder_yaw
    [-1.527,  0.480],  # left_elbow_pitch
    [-1.833,  1.833],  # left_wrist_yaw
    [-0.611,  0.611],  # left_wrist_pitch
    [-0.960,  0.960],  # left_wrist_roll
    # right_manipulator (7-13)
    [-2.967,  2.967],  # right_shoulder_pitch
    [-2.793,  0.524],  # right_shoulder_roll
    [-1.833,  1.833],  # right_shoulder_yaw
    [-1.527,  0.480],  # right_elbow_pitch
    [-1.833,  1.833],  # right_wrist_yaw
    [-0.611,  0.611],  # right_wrist_pitch
    [-0.960,  0.960],  # right_wrist_roll
    # left_hand (14-19)
    [ 0.200,  1.700],  # L_index
    [ 0.200,  1.700],  # L_middle
    [ 0.200,  1.700],  # L_ring
    [ 0.200,  1.700],  # L_pinky
    [ 0.000,  1.200],  # L_thumb_pitch
    [-1.570,  1.570],  # L_thumb_yaw
    # right_hand (20-25)
    [ 0.200,  1.700],  # R_index
    [ 0.200,  1.700],  # R_middle
    [ 0.200,  1.700],  # R_ring
    [ 0.200,  1.700],  # R_pinky
    [ 0.000,  1.200],  # R_thumb_pitch
    [-1.570,  1.570],  # R_thumb_yaw
    # head (26-27)
    [-1.396,  1.396],  # head_yaw
    [-0.524,  0.524],  # head_pitch
    # waist (28)
    [-2.618,  2.618],  # waist_yaw
], dtype=np.float32)

def clamp_action(action: np.ndarray) -> np.ndarray:
    """将关节部分限制在安全范围内。"""
    action = action.copy()
    action[:29] = np.clip(action[:29], JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    return action
```

## 完整推理循环示例

```python
import time
import numpy as np
from fourier_aurora_client import AuroraClient

# ---- 初始化 ----
client = AuroraClient.get_instance(domain_id=123, robot_name="gr2t2v2", serial_number=None)
time.sleep(1)
client.set_fsm_state(2)  # PdStand 上半身控制
time.sleep(1)

# 加载策略模型（以 LeRobot ACT 为例）
policy = load_policy("path/to/checkpoint")  # 根据实际框架替换

# ---- 推理参数 ----
CONTROL_FREQ = 30       # 控制频率（Hz），与训练 fps 一致
TASK = "grab the bottle on the table"

# ---- 主循环 ----
try:
    while True:
        t_start = time.time()

        # 1. 读取观测
        obs_state = get_observation_state(client)        # (45,)
        obs_image = get_camera_image()                   # (480, 640, 3) uint8
        # obs_depth = get_depth_image()                  # 如果模型用到深度

        # 2. 构造模型输入
        observation = {
            "observation.state": obs_state,
            "observation.images.camera_top": obs_image,
            # "observation.images.camera_top_depth": obs_depth,
            "task": TASK,
        }

        # 3. 模型推理
        action = policy.predict(observation)             # (35,) 或 (T, 35) action chunk

        # 如果模型输出 action chunk（多步），取第一步
        if action.ndim == 2:
            action = action[0]

        # 4. 安全限位
        action = clamp_action(action)

        # 5. 下发指令
        send_action(client, action)

        # 6. 控制频率
        elapsed = time.time() - t_start
        sleep_time = 1.0 / CONTROL_FREQ - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("停止推理")
finally:
    client.close()
```

## 安全注意事项

1. **关节限位**：模型输出可能超出安全范围，务必用 `clamp_action()` 裁剪后再下发
2. **控制频率**：推理频率应与训练 fps（默认 30Hz）一致，过快或过慢都会导致行为失真
3. **先读后写**：启动推理前先读一次当前状态，确认机器人在合理姿态
4. **平滑过渡**：从待机姿态到推理姿态之间，应使用插值过渡（100Hz，1-2秒），避免关节突变
5. **急停机制**：推理循环中应有急停逻辑（如键盘中断、力矩异常检测）
6. **FSM 状态**：
   - `2`（PdStand）：上半身控制，腿部由底层平衡控制器管理
   - `10`（UserCmd）：全身关节控制（需要自行处理平衡）
   - `11`（UpperBodyUserCmd）：仅上半身关节，腿部禁用
