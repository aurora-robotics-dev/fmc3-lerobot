# Dora-Record 数据格式说明

## 目录结构

```
dora-record/<session_id>/
├── episode_000000000/
│   ├── metadata.json                               # episode 元信息
│   ├── action.parquet                              # 手臂关节动作（31维，~100Hz）
│   ├── action.base.parquet                         # 底盘动作（6维，~100Hz）
│   ├── observation.state.parquet                   # 手臂关节状态（29维，~60Hz）
│   ├── observation.base_state.parquet              # 底盘状态 + IMU（16维，~60Hz）
│   ├── observation.images.camera_top.parquet       # RGB 图像（JPEG 编码，~30Hz）
│   └── observation.images.camera_top_depth.parquet # 深度图像（PNG 编码，~30Hz）
├── episode_000000001/
│   └── ...
└── episode_000000002/
    └── ...
```

## Parquet 文件公共列

每个 parquet 文件都包含以下公共列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `trace_id` | string | 分布式追踪 ID |
| `span_id` | string | 追踪 span ID |
| `timestamp_uhlc` | uint64 | UHLC 时间戳 |
| `timestamp_utc` | timestamp[ns] | UTC 时间戳（纳秒精度） |
| `user_timestamp_utc` | timestamp[ns] | 用户时间戳 |
| `parameters` | string | 附加参数 |

## 各数据文件详情

### action.parquet — 手臂关节动作

- **数据列**: `action`
- **类型**: `list<struct<name: string, value: double>>`
- **采样频率**: ~100Hz
- **维度**: 31

每行是一个字典列表，示例：

```python
[
    {"name": "left_shoulder_pitch_joint",  "value": -0.6680},
    {"name": "left_shoulder_roll_joint",   "value": -0.0809},
    ...  # 共 31 个关节
]
```

**31 个关节维度映射：**

| 维度 | 关节名 | 所属部位 |
|------|--------|---------|
| 0 | left_shoulder_pitch_joint | 左臂 |
| 1 | left_shoulder_roll_joint | 左臂 |
| 2 | left_shoulder_yaw_joint | 左臂 |
| 3 | left_elbow_pitch_joint | 左臂 |
| 4 | left_wrist_yaw_joint | 左臂 |
| 5 | left_wrist_pitch_joint | 左臂 |
| 6 | left_wrist_roll_joint | 左臂 |
| 7 | right_shoulder_pitch_joint | 右臂 |
| 8 | right_shoulder_roll_joint | 右臂 |
| 9 | right_shoulder_yaw_joint | 右臂 |
| 10 | right_elbow_pitch_joint | 右臂 |
| 11 | right_wrist_yaw_joint | 右臂 |
| 12 | right_wrist_pitch_joint | 右臂 |
| 13 | right_wrist_roll_joint | 右臂 |
| 14 | L_pinky_proximal_joint | 左手 |
| 15 | L_ring_proximal_joint | 左手 |
| 16 | L_middle_proximal_joint | 左手 |
| 17 | L_index_proximal_joint | 左手 |
| 18 | L_thumb_proximal_pitch_joint | 左手 |
| 19 | L_thumb_proximal_yaw_joint | 左手 |
| 20 | R_pinky_proximal_joint | 右手 |
| 21 | R_ring_proximal_joint | 右手 |
| 22 | R_middle_proximal_joint | 右手 |
| 23 | R_index_proximal_joint | 右手 |
| 24 | R_thumb_proximal_pitch_joint | 右手 |
| 25 | R_thumb_proximal_yaw_joint | 右手 |
| 26 | head_yaw_joint | 头部 |
| 27 | head_pitch_joint | 头部 |
| 28 | waist_yaw_joint | 腰部 |
| 29 | waist_roll_joint | 腰部（GR2 无此自由度） |
| 30 | waist_pitch_joint | 腰部（GR2 无此自由度） |

> **注意**: Fourier GR2 腰部实际只有 `waist_yaw` 一个自由度，转换时会自动过滤掉 `waist_roll_joint` 和 `waist_pitch_joint`，action 从 31 维变为 29 维。

### action.base.parquet — 底盘动作

- **数据列**: `action.base`
- **类型**: `list<struct<name: string, value: double>>`
- **采样频率**: ~100Hz
- **维度**: 6

| 维度 | 名称 | 说明 |
|------|------|------|
| 0 | vel_x | 前后线速度 |
| 1 | vel_y | 左右线速度 |
| 2 | vel_yaw | 偏航角速度 |
| 3 | vel_height | 升降速度 |
| 4 | vel_pitch | 俯仰角速度 |
| 5 | base_yaw | 底盘偏航角 |

### observation.state.parquet — 手臂关节状态

- **数据列**: `observation.state`
- **类型**: `list<struct<name: string, value: double>>`
- **采样频率**: ~60Hz
- **维度**: 29（比 action 少腰部的 roll 和 pitch）

### observation.base_state.parquet — 底盘状态

- **数据列**: `observation.base_state`
- **类型**: `list<struct<base: struct<position, quat, rpy>, imu: struct<acc_B, omega_B>, stand_pose, state>>`
- **采样频率**: ~60Hz

每行是一个嵌套结构体：

```python
[{
    "base": {
        "position": [x, y, z],           # 位置（米）
        "quat": [qx, qy, qz, qw],       # 四元数
        "rpy": [roll, pitch, yaw]         # 欧拉角（弧度）
    },
    "imu": {
        "acc_B": [ax, ay, az],            # 加速度（m/s²）
        "omega_B": [wx, wy, wz]          # 角速度（rad/s）
    },
    "stand_pose": [4个值],
    "state": 11                           # 状态码
}]
```

展平后为 16 维向量：

| 维度 | 名称 | 说明 |
|------|------|------|
| 0-2 | base_pos_x/y/z | 底盘位置 |
| 3-6 | base_quat_x/y/z/w | 底盘四元数 |
| 7-9 | base_rpy_roll/pitch/yaw | 底盘欧拉角 |
| 10-12 | imu_acc_x/y/z | IMU 加速度 |
| 13-15 | imu_omega_x/y/z | IMU 角速度 |

### observation.images.camera_top.parquet — RGB 图像

- **数据列**: `observation.images.camera_top`
- **类型**: `list<uint8>`
- **采样频率**: ~30Hz
- **图像格式**: JPEG 编码的字节流（约 67KB/帧）
- **分辨率**: 480 x 640 x 3

### observation.images.camera_top_depth.parquet — 深度图像

- **数据列**: `observation.images.camera_top_depth`
- **类型**: `list<uint8>`
- **采样频率**: ~30Hz
- **图像格式**: PNG 编码的深度图字节流（约 106KB/帧）
- **分辨率**: 480 x 640

## metadata.json

每个 episode 目录下有一个 `metadata.json`，记录该 episode 的元信息：

```json
{
    "episode_index": 0,
    "task_id": null,
    "start_time": "2026-02-12T02:29:11.715201951Z",
    "end_time": "2026-02-12T02:29:25.814816087Z",
    "session_id": "019c4fad-86f2-7017-bc8c-35744b1de20d",
    "session_start_time": "2026-02-12T02:28:20.039273399Z",
    "notes": "grxtest",
    "pilot": "-1",
    "operator": "-1",
    "machine_id": "GR3",
    "station_id": "-1",
    "equipment": "t5d",
    "camera_type": ""
}
```

| 字段 | 说明 |
|------|------|
| `episode_index` | episode 序号 |
| `task_id` | 任务 ID（可为 null） |
| `start_time` / `end_time` | episode 起止时间（ISO 8601） |
| `session_id` | 采集 session 的唯一 ID |
| `machine_id` | 机器人型号（如 GR3） |
| `equipment` | 遥操设备标识（如 t5d） |
| `notes` | 备注信息 |

## 各传感器采样频率

| 数据 | 频率 | 说明 |
|------|------|------|
| action | ~100Hz | 手臂关节动作指令 |
| action.base | ~100Hz | 底盘动作指令 |
| observation.state | ~60Hz | 手臂关节状态反馈 |
| observation.base_state | ~60Hz | 底盘状态 + IMU 反馈 |
| camera_top (RGB) | ~30Hz | RGB 摄像头 |
| camera_top_depth | ~30Hz | 深度摄像头 |

> **注意**: 各传感器采样频率不同，转换为 LeRobot 格式时需要重采样对齐到统一帧率（默认 30fps）。详见 [CONVERT_GUIDE.md](CONVERT_GUIDE.md)。
