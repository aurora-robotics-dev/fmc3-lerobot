#!/usr/bin/env python3
"""
将 Dora-Record 格式的 Fourier GR3 数据集转换为 LeRobot v3.0 格式。

用法:
    python convert_dora_to_lerobot.py \
        --input ./019c4a76-1e53-7ad6-a112-590a9a7c5b21 \
        --output ./lerobot_output \
        --task "遥操作测试" \
        --fps 30 \
        --video-codec libx264 \
        --robot-type fourier_gr2
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ===========================================================================
# GR2 SDK 控制组对齐的关节顺序（29个关节）
# 顺序: left_manipulator(7) → right_manipulator(7) → left_hand(6) →
#        right_hand(6) → head(2) → waist(1)
# ===========================================================================

GR2_JOINT_ORDER = [
    # left_manipulator (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    # right_manipulator (7)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    # left_hand (6) — SDK 顺序: 食指→中指→无名指→小指→拇指
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_ring_proximal_joint",
    "L_pinky_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
    # right_hand (6)
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
    # head (2)
    "head_yaw_joint",
    "head_pitch_joint",
    # waist (1)
    "waist_yaw_joint",
]


def reorder_to_target(names, values, target_order):
    """按目标顺序重排关节列。源数据必须包含 target_order 中的所有关节。"""
    idx_map = {name: i for i, name in enumerate(names)}
    missing = [n for n in target_order if n not in idx_map]
    if missing:
        raise ValueError(
            f"源数据缺少以下关节: {missing}\n"
            f"源数据关节: {names}\n"
            f"目标顺序: {list(target_order)}"
        )
    reorder_idx = [idx_map[n] for n in target_order]
    return list(target_order), values[:, reorder_idx]


# 根据机器人类型定义需要过滤的 action 关节
ACTION_FILTER_JOINTS = {
    "fourier_gr2": {"waist_roll_joint", "waist_pitch_joint"},
    "fourier_gr3": set(),
}

# 根据机器人类型定义目标关节顺序（对齐 SDK 控制组）
JOINT_ORDER_MAP = {
    "fourier_gr2": GR2_JOINT_ORDER,
}


# ===========================================================================
# Dora-Record 数据读取
# ===========================================================================

def read_dora_episode_dir(episode_dir: Path) -> dict | None:
    """读取单个 episode 目录下所有 parquet + metadata。"""
    meta_path = episode_dir / "metadata.json"
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"    [WARN] {meta_path} 为空或损坏, 跳过此 episode")
        return None

    result = {"metadata": meta}

    parquet_files = [
        "action",
        "action.base",
        "observation.state",
        "observation.base_state",
        "observation.images.camera_top",
        "observation.images.camera_top_depth",
    ]
    for name in parquet_files:
        fpath = episode_dir / f"{name}.parquet"
        if fpath.exists():
            result[name] = fpath
    return result


def read_named_list_column(parquet_path: Path, column_name: str):
    """
    读取 Dora 的 list<struct<name, value>> 格式列。
    返回 (names: list[str], values: np.ndarray[N, D])。
    """
    table = pq.read_table(str(parquet_path), columns=[column_name, "timestamp_utc"])
    col = table.column(column_name)
    ts_col = table.column("timestamp_utc").cast(pa.int64())

    # 从第一行提取关节名称
    first_row = col[0].as_py()
    names = [item["name"] for item in first_row]
    n_joints = len(names)

    # 提取所有值
    n_rows = len(col)
    values = np.zeros((n_rows, n_joints), dtype=np.float64)
    for i in range(n_rows):
        row = col[i].as_py()
        for j, item in enumerate(row):
            values[i, j] = item["value"]

    timestamps_ns = np.array([ts_col[i].as_py() for i in range(n_rows)], dtype=np.int64)
    return names, values, timestamps_ns


def read_image_column(parquet_path: Path, column_name: str):
    """
    读取图像列（list<uint8>），返回 (raw_bytes_list, timestamps_ns)。
    每个元素是一个 bytes 对象（原始像素数据或编码图像）。
    使用 iter_batches 避免大文件的 int32 list offset 溢出。
    """
    pf = pq.ParquetFile(str(parquet_path))
    images = []
    timestamps = []

    for batch in pf.iter_batches(
        batch_size=100, columns=[column_name, "timestamp_utc"]
    ):
        ts_arr = batch.column("timestamp_utc").cast(pa.int64())
        img_col = batch.column(column_name)
        for i in range(len(img_col)):
            images.append(bytes(img_col[i].as_py()))
            timestamps.append(ts_arr[i].as_py())

    timestamps_ns = np.array(timestamps, dtype=np.int64)
    return images, timestamps_ns


def read_base_state_column(parquet_path: Path):
    """
    读取 observation.base_state 的嵌套结构列。
    返回 (field_names, values, timestamps_ns)。
    """
    table = pq.read_table(
        str(parquet_path),
        columns=["observation.base_state", "timestamp_utc"],
    )
    col = table.column("observation.base_state")
    ts_col = table.column("timestamp_utc").cast(pa.int64())

    n_rows = len(col)
    # 解析嵌套结构：base(position[3], quat[4], rpy[3]), imu(acc_B[3], omega_B[3])
    # 展平为: pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w,
    #          rpy_r, rpy_p, rpy_y, acc_x, acc_y, acc_z, omega_x, omega_y, omega_z
    field_names = [
        "base_pos_x", "base_pos_y", "base_pos_z",
        "base_quat_x", "base_quat_y", "base_quat_z", "base_quat_w",
        "base_rpy_roll", "base_rpy_pitch", "base_rpy_yaw",
        "imu_acc_x", "imu_acc_y", "imu_acc_z",
        "imu_omega_x", "imu_omega_y", "imu_omega_z",
    ]
    values = np.zeros((n_rows, len(field_names)), dtype=np.float64)

    for i in range(n_rows):
        row = col[i].as_py()
        if row is None or len(row) == 0:
            continue
        entry = row[0]
        base = entry.get("base", {})
        imu = entry.get("imu", {})
        pos = base.get("position", [0, 0, 0])
        quat = base.get("quat", [0, 0, 0, 0])
        rpy = base.get("rpy", [0, 0, 0])
        acc = imu.get("acc_B", [0, 0, 0])
        omega = imu.get("omega_B", [0, 0, 0])
        flat = list(pos) + list(quat) + list(rpy) + list(acc) + list(omega)
        values[i, :len(flat)] = flat[:len(field_names)]

    timestamps_ns = np.array([ts_col[i].as_py() for i in range(n_rows)], dtype=np.int64)
    return field_names, values, timestamps_ns


# ===========================================================================
# 时间对齐和重采样
# ===========================================================================

def resample_to_timestamps(
    source_values: np.ndarray,
    source_ts: np.ndarray,
    target_ts: np.ndarray,
) -> np.ndarray:
    """
    将 source 数据按照 target 的时间戳进行最近邻重采样。
    source_values: (N_src, D)
    source_ts: (N_src,) int64 nanoseconds
    target_ts: (N_tgt,) int64 nanoseconds
    返回: (N_tgt, D)
    """
    indices = np.searchsorted(source_ts, target_ts, side="right") - 1
    indices = np.clip(indices, 0, len(source_ts) - 1)

    # 如果下一个点更近，则选择它
    next_indices = np.minimum(indices + 1, len(source_ts) - 1)
    diff_left = np.abs(source_ts[indices] - target_ts)
    diff_right = np.abs(source_ts[next_indices] - target_ts)
    use_next = diff_right < diff_left
    indices[use_next] = next_indices[use_next]

    return source_values[indices]


def resample_images_to_timestamps(
    images: list,
    source_ts: np.ndarray,
    target_ts: np.ndarray,
) -> list:
    """对图像列表进行最近邻时间重采样。"""
    indices = np.searchsorted(source_ts, target_ts, side="right") - 1
    indices = np.clip(indices, 0, len(source_ts) - 1)

    next_indices = np.minimum(indices + 1, len(source_ts) - 1)
    diff_left = np.abs(source_ts[indices] - target_ts)
    diff_right = np.abs(source_ts[next_indices] - target_ts)
    use_next = diff_right < diff_left
    indices[use_next] = next_indices[use_next]

    return [images[idx] for idx in indices]


def generate_uniform_timestamps(start_ns: int, end_ns: int, fps: int) -> np.ndarray:
    """生成均匀时间戳序列（纳秒）。"""
    duration_s = (end_ns - start_ns) / 1e9
    n_frames = max(1, int(math.floor(duration_s * fps)))
    step_ns = int(1e9 / fps)
    return np.array([start_ns + i * step_ns for i in range(n_frames)], dtype=np.int64)


# ===========================================================================
# 图像检测和视频编码
# ===========================================================================

def detect_image_shape(raw_bytes: bytes):
    """
    检测图像格式。返回 (height, width, channels) 或 None。
    支持原始像素（根据大小推断）和编码格式（JPEG/PNG）。
    """
    # 尝试 PIL 解码
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(raw_bytes))
        w, h = img.size
        c = len(img.getbands())
        return h, w, c
    except Exception:
        pass

    # 原始像素数据：猜测常见分辨率
    n = len(raw_bytes)
    candidates = [
        (960, 640, 3),
        (480, 640, 3),
        (720, 1280, 3),
        (1080, 1920, 3),
        (480, 848, 3),
        (480, 1280, 3),
    ]
    for h, w, c in candidates:
        if h * w * c == n:
            return h, w, c

    # 通用搜索
    for c in [3, 4, 1]:
        if n % c != 0:
            continue
        pixels = n // c
        sqrt_p = int(math.sqrt(pixels))
        for h in range(sqrt_p - 200, sqrt_p + 200):
            if h > 0 and pixels % h == 0:
                w = pixels // h
                if 100 < w < 4000 and 100 < h < 4000 and 0.3 < w / h < 3.0:
                    return h, w, c

    return None


def raw_to_rgb_array(raw_bytes: bytes, height: int, width: int, channels: int) -> np.ndarray:
    """将原始字节转换为 RGB numpy 数组。"""
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(raw_bytes))
        return np.array(img.convert("RGB"))
    except Exception:
        pass

    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(height, width, channels)
    if channels == 4:
        arr = arr[:, :, :3]  # 丢弃 alpha
    elif channels == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def raw_to_depth_array(raw_bytes: bytes) -> np.ndarray:
    """将深度图像字节（PNG编码）转换为 uint16 深度数组 (H, W)。"""
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(raw_bytes))
    return np.array(img)


def _find_ffmpeg() -> str:
    """查找 ffmpeg 可执行文件路径。"""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return "ffmpeg"


def encode_video_from_frames(
    frames: list[np.ndarray],
    output_path: Path,
    fps: int,
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
):
    """
    用 ffmpeg 将一组 RGB numpy 帧编码为 mp4 视频。
    frames: list of (H, W, 3) uint8 arrays。
    """
    if not frames:
        return

    h, w = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = _find_ffmpeg()
    print(f"    使用 ffmpeg: {ffmpeg_bin}")
    cmd = [
        ffmpeg_bin, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", codec,
        "-pix_fmt", pix_fmt,
        "-an",
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        pass
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
    proc.wait()
    stderr = proc.stderr.read()
    if proc.returncode != 0:
        print(f"[ERROR] ffmpeg 编码失败 (returncode={proc.returncode}):\n{stderr.decode()}", file=sys.stderr)
        sys.exit(1)


def get_video_codec_name(codec_arg: str) -> str:
    """将 ffmpeg 编码器名映射到 LeRobot info 中的 codec 名称。"""
    mapping = {
        "libx264": "h264",
        "libopenh264": "h264",
        "libx265": "hevc",
        "libsvtav1": "av1",
        "libaom-av1": "av1",
        "libvpx-vp9": "vp9",
    }
    return mapping.get(codec_arg, codec_arg)


# ===========================================================================
# LeRobot v3.0 输出
# ===========================================================================

def compute_stats(values: np.ndarray) -> dict:
    """计算一组数值的 min/max/mean/std/count 统计，结果始终为列表。"""
    def _to_list(x):
        result = x.tolist()
        return result if isinstance(result, list) else [result]

    return {
        "min": _to_list(values.min(axis=0)),
        "max": _to_list(values.max(axis=0)),
        "mean": _to_list(values.mean(axis=0)),
        "std": _to_list(values.std(axis=0)),
        "count": [len(values)],
    }


def compute_image_stats(frames: list[np.ndarray]) -> dict:
    """计算图像统计（归一化到 [0,1]）。"""
    if not frames:
        c = 3
        placeholder = [[[0.0]]] * c
        return {"min": placeholder, "max": placeholder, "mean": placeholder, "std": placeholder, "count": [0]}

    # 采样最多 100 张图像计算统计
    sample_indices = np.linspace(0, len(frames) - 1, min(100, len(frames)), dtype=int)
    sample = [frames[i].astype(np.float32) / 255.0 for i in sample_indices]
    stacked = np.stack(sample)  # (N, H, W, C) or (N, H, W)

    if stacked.ndim == 3:
        # 灰度/深度图像 (N, H, W) -> 视为单通道
        c = 1
        channel_min = [stacked.min()]
        channel_max = [stacked.max()]
        channel_mean = [stacked.mean()]
        channel_std = [stacked.std()]
    else:
        c = stacked.shape[3]
        channel_min = stacked.min(axis=(0, 1, 2))
        channel_max = stacked.max(axis=(0, 1, 2))
        channel_mean = stacked.mean(axis=(0, 1, 2))
        channel_std = stacked.std(axis=(0, 1, 2))

    return {
        "min": [[[float(channel_min[i])]] for i in range(c)],
        "max": [[[float(channel_max[i])]] for i in range(c)],
        "mean": [[[float(channel_mean[i])]] for i in range(c)],
        "std": [[[float(channel_std[i])]] for i in range(c)],
        "count": [len(sample_indices)],
    }


def build_info_json(
    robot_type: str,
    fps: int,
    total_episodes: int,
    total_frames: int,
    action_names: list[str],
    state_names: list[str],
    video_key: str | None,
    video_height: int | None,
    video_width: int | None,
    video_codec: str | None,
) -> dict:
    """构建 LeRobot v3.0 的 info.json。"""
    features = {
        "action": {
            "dtype": "float32",
            "shape": [len(action_names)],
            "names": action_names,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [len(state_names)],
            "names": state_names,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }

    if video_key and video_height and video_width:
        features[video_key] = {
            "dtype": "video",
            "shape": [video_height, video_width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": video_height,
                "video.width": video_width,
                "video.codec": video_codec or "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    return {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        if video_key else None,
        "features": features,
    }


# ===========================================================================
# 主转换流程
# ===========================================================================

def convert(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    fps = args.fps
    task_name = args.task
    robot_type = args.robot_type
    video_codec = args.video_codec
    no_video = args.no_video
    video_key = "observation.images.camera_top"
    depth_key = "observation.images.camera_top_depth"

    # 发现所有 episode 目录
    episode_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")],
        key=lambda d: d.name,
    )
    if not episode_dirs:
        print(f"[ERROR] 在 {input_dir} 中未找到 episode_* 目录", file=sys.stderr)
        sys.exit(1)

    print(f"找到 {len(episode_dirs)} 个 episode")

    filter_joints = ACTION_FILTER_JOINTS.get(robot_type, set())
    if filter_joints:
        print(f"机器人类型 {robot_type}: 将从 action 中过滤关节 {filter_joints}")

    target_joint_order = JOINT_ORDER_MAP.get(robot_type)
    if target_joint_order:
        print(f"机器人类型 {robot_type}: 将按 SDK 控制组顺序重排关节")

    # ==== 第一遍：读取所有数据并对齐 ====
    all_action_names = None
    all_state_names = None
    all_base_action_names = None
    all_base_state_names = None
    episode_data_list = []  # 每个 episode 的对齐后数据
    video_height, video_width, video_channels = None, None, None
    depth_height, depth_width = None, None
    has_images = False
    has_depth = False
    has_base_action = False
    has_base_state = False

    for ep_idx, ep_dir in enumerate(episode_dirs):
        print(f"  读取 episode {ep_idx}: {ep_dir.name} ...")
        ep_info = read_dora_episode_dir(ep_dir)
        if ep_info is None:
            continue
        meta = ep_info["metadata"]

        # 读取 action
        action_path = ep_info.get("action")
        if action_path is None:
            print(f"    [WARN] episode {ep_idx} 缺少 action.parquet, 跳过")
            continue
        action_names, action_vals, action_ts = read_named_list_column(action_path, "action")

        # 过滤掉不需要的关节
        if filter_joints:
            keep_idx = [i for i, n in enumerate(action_names) if n not in filter_joints]
            action_names = [action_names[i] for i in keep_idx]
            action_vals = action_vals[:, keep_idx]

        # 按 SDK 控制组顺序重排 action 关节
        if target_joint_order:
            action_names, action_vals = reorder_to_target(action_names, action_vals, target_joint_order)

        if all_action_names is None:
            all_action_names = action_names

        # 读取 action.base
        base_action_path = ep_info.get("action.base")
        base_action_vals, base_action_ts = None, None
        if base_action_path:
            base_action_names, base_action_vals, base_action_ts = read_named_list_column(
                base_action_path, "action.base"
            )
            if all_base_action_names is None:
                all_base_action_names = base_action_names
                has_base_action = True

        # 读取 observation.state
        state_path = ep_info.get("observation.state")
        if state_path:
            state_names, state_vals, state_ts = read_named_list_column(state_path, "observation.state")
            # 按 SDK 控制组顺序重排 state 关节
            if target_joint_order:
                state_names, state_vals = reorder_to_target(state_names, state_vals, target_joint_order)
            if all_state_names is None:
                all_state_names = state_names
        else:
            state_names, state_vals, state_ts = action_names, action_vals.copy(), action_ts.copy()
            if all_state_names is None:
                all_state_names = action_names

        # 读取 observation.base_state
        base_state_path = ep_info.get("observation.base_state")
        base_state_vals, base_state_ts = None, None
        if base_state_path:
            base_state_names, base_state_vals, base_state_ts = read_base_state_column(base_state_path)
            if all_base_state_names is None:
                all_base_state_names = base_state_names
                has_base_state = True

        # 读取 RGB 图像
        img_path = ep_info.get("observation.images.camera_top")
        images_raw, img_ts = None, None
        if img_path and not no_video:
            images_raw, img_ts = read_image_column(img_path, "observation.images.camera_top")
            if images_raw and not has_images:
                shape = detect_image_shape(images_raw[0])
                if shape:
                    video_height, video_width, video_channels = shape
                    has_images = True
                    print(f"    检测到 RGB 图像尺寸: {video_height}x{video_width}x{video_channels}")

        # 读取深度图像
        depth_path = ep_info.get("observation.images.camera_top_depth")
        depth_raw, depth_ts = None, None
        if depth_path and not no_video:
            depth_raw, depth_ts = read_image_column(depth_path, "observation.images.camera_top_depth")
            if depth_raw and not has_depth:
                depth_shape = detect_image_shape(depth_raw[0])
                if depth_shape:
                    depth_height, depth_width = depth_shape[0], depth_shape[1]
                    has_depth = True
                    print(f"    检测到深度图像尺寸: {depth_height}x{depth_width}")

        # 确定时间范围（从 action 时间戳）
        start_ns = action_ts[0]
        end_ns = action_ts[-1]

        # 生成统一时间轴
        target_ts = generate_uniform_timestamps(start_ns, end_ns, fps)
        n_frames = len(target_ts)

        if n_frames == 0:
            print(f"    [WARN] episode {ep_idx} 帧数为 0, 跳过")
            continue

        # 重采样 action 和 state 到统一时间轴
        action_resampled = resample_to_timestamps(action_vals, action_ts, target_ts).astype(np.float32)
        state_resampled = resample_to_timestamps(state_vals, state_ts, target_ts).astype(np.float32)

        # 重采样 action.base
        base_action_resampled = None
        if base_action_vals is not None and base_action_ts is not None:
            base_action_resampled = resample_to_timestamps(
                base_action_vals, base_action_ts, target_ts
            ).astype(np.float32)

        # 重采样 observation.base_state
        base_state_resampled = None
        if base_state_vals is not None and base_state_ts is not None:
            base_state_resampled = resample_to_timestamps(
                base_state_vals, base_state_ts, target_ts
            ).astype(np.float32)

        # 重采样 RGB 图像
        frames_rgb = None
        if has_images and images_raw is not None and img_ts is not None:
            resampled_imgs = resample_images_to_timestamps(images_raw, img_ts, target_ts)
            frames_rgb = [
                raw_to_rgb_array(img_bytes, video_height, video_width, video_channels)
                for img_bytes in resampled_imgs
            ]

        # 重采样深度图像
        frames_depth = None
        if has_depth and depth_raw is not None and depth_ts is not None:
            resampled_depth = resample_images_to_timestamps(depth_raw, depth_ts, target_ts)
            frames_depth = [raw_to_depth_array(d) for d in resampled_depth]

        episode_data_list.append({
            "episode_index": ep_idx,
            "n_frames": n_frames,
            "action": action_resampled,
            "state": state_resampled,
            "base_action": base_action_resampled,
            "base_state": base_state_resampled,
            "frames_rgb": frames_rgb,
            "frames_depth": frames_depth,
            "metadata": meta,
        })

    if not episode_data_list:
        print("[ERROR] 没有有效的 episode 数据", file=sys.stderr)
        sys.exit(1)

    # 合并 action = arm_action + base_action（如果有 base_action）
    if has_base_action:
        combined_action_names = all_action_names + [f"base_{n}" for n in all_base_action_names]
        for ep in episode_data_list:
            if ep["base_action"] is not None:
                ep["action"] = np.concatenate([ep["action"], ep["base_action"]], axis=1)
            else:
                # 用零填充缺失的 base_action
                pad = np.zeros((ep["n_frames"], len(all_base_action_names)), dtype=np.float32)
                ep["action"] = np.concatenate([ep["action"], pad], axis=1)
        all_action_names = combined_action_names

    # 合并 state = arm_state + base_state（如果有 base_state）
    if has_base_state:
        combined_state_names = all_state_names + all_base_state_names
        for ep in episode_data_list:
            if ep["base_state"] is not None:
                ep["state"] = np.concatenate([ep["state"], ep["base_state"]], axis=1)
            else:
                pad = np.zeros((ep["n_frames"], len(all_base_state_names)), dtype=np.float32)
                ep["state"] = np.concatenate([ep["state"], pad], axis=1)
        all_state_names = combined_state_names

    total_episodes = len(episode_data_list)
    total_frames = sum(ep["n_frames"] for ep in episode_data_list)
    n_action = len(all_action_names)
    n_state = len(all_state_names)

    print(f"\n总计: {total_episodes} episodes, {total_frames} frames")
    print(f"Action 维度: {n_action} ({', '.join(all_action_names[:5])}...)")
    print(f"State 维度: {n_state} ({', '.join(all_state_names[:5])}...)")

    # ==== 构建输出目录结构 ====
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    if has_images:
        (output_dir / "videos" / video_key / "chunk-000").mkdir(parents=True, exist_ok=True)
    if has_depth:
        (output_dir / "videos" / depth_key / "chunk-000").mkdir(parents=True, exist_ok=True)

    # ==== 写 data parquet ====
    print("\n生成 data parquet ...")
    data_rows = []
    global_index = 0
    for ep in episode_data_list:
        for frame_i in range(ep["n_frames"]):
            data_rows.append({
                "action": ep["action"][frame_i].tolist(),
                "observation.state": ep["state"][frame_i].tolist(),
                "timestamp": float(frame_i) / fps,
                "frame_index": frame_i,
                "episode_index": ep["episode_index"],
                "index": global_index,
                "task_index": 0,
            })
            global_index += 1

    # 写入 parquet（使用 pyarrow 确保列表类型正确）
    action_type = pa.list_(pa.float64())
    state_type = pa.list_(pa.float64())
    schema = pa.schema([
        ("action", action_type),
        ("observation.state", state_type),
        ("timestamp", pa.float64()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ])

    arrays = {
        "action": [row["action"] for row in data_rows],
        "observation.state": [row["observation.state"] for row in data_rows],
        "timestamp": [row["timestamp"] for row in data_rows],
        "frame_index": [row["frame_index"] for row in data_rows],
        "episode_index": [row["episode_index"] for row in data_rows],
        "index": [row["index"] for row in data_rows],
        "task_index": [row["task_index"] for row in data_rows],
    }
    table = pa.table(arrays, schema=schema)
    data_path = output_dir / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(table, str(data_path))
    print(f"  写入: {data_path} ({total_frames} 行)")

    # ==== 编码视频 ====
    video_file_index = 0
    episode_video_info = {}  # ep_idx -> {chunk, file, from_ts, to_ts}
    if has_images:
        print("\n编码 RGB 视频 ...")
        cumulative_time = 0.0
        for ep in episode_data_list:
            if ep["frames_rgb"] is None:
                episode_video_info[ep["episode_index"]] = {
                    "chunk": 0, "file": video_file_index,
                    "from_ts": 0.0, "to_ts": 0.0,
                }
                continue

            from_ts = cumulative_time
            to_ts = from_ts + (ep["n_frames"] - 1) / fps

            episode_video_info[ep["episode_index"]] = {
                "chunk": 0, "file": video_file_index,
                "from_ts": from_ts, "to_ts": to_ts,
            }
            cumulative_time = from_ts + ep["n_frames"] / fps

        # 将所有帧合并写入一个视频文件
        all_frames = []
        for ep in episode_data_list:
            if ep["frames_rgb"]:
                all_frames.extend(ep["frames_rgb"])

        if all_frames:
            video_path = (
                output_dir / "videos" / video_key / "chunk-000" / f"file-{video_file_index:03d}.mp4"
            )
            print(f"  编码 {len(all_frames)} 帧 -> {video_path}")
            encode_video_from_frames(all_frames, video_path, fps, codec=video_codec)
            print(f"  完成")

    # 编码深度视频
    depth_file_index = 0
    episode_depth_info = {}
    if has_depth:
        print("\n编码深度视频 ...")
        cumulative_time = 0.0
        all_depth_frames = []
        for ep in episode_data_list:
            if ep["frames_depth"] is None:
                episode_depth_info[ep["episode_index"]] = {
                    "chunk": 0, "file": depth_file_index,
                    "from_ts": 0.0, "to_ts": 0.0,
                }
                continue

            from_ts = cumulative_time
            to_ts = from_ts + (ep["n_frames"] - 1) / fps

            episode_depth_info[ep["episode_index"]] = {
                "chunk": 0, "file": depth_file_index,
                "from_ts": from_ts, "to_ts": to_ts,
            }
            cumulative_time = from_ts + ep["n_frames"] / fps

            # 深度图转灰度 RGB（16bit -> 8bit 可视化，或保持原样编码）
            for d in ep["frames_depth"]:
                if d.ndim == 2:
                    # 16bit 深度 -> 归一化到 8bit 灰度 -> 伪 RGB
                    d8 = (d.astype(np.float32) / d.max() * 255).astype(np.uint8) if d.max() > 0 else d.astype(np.uint8)
                    all_depth_frames.append(np.stack([d8, d8, d8], axis=2))
                elif d.ndim == 3 and d.shape[2] == 1:
                    d8 = (d[:, :, 0].astype(np.float32) / d.max() * 255).astype(np.uint8) if d.max() > 0 else d[:, :, 0].astype(np.uint8)
                    all_depth_frames.append(np.stack([d8, d8, d8], axis=2))
                else:
                    all_depth_frames.append(d[:, :, :3] if d.shape[2] >= 3 else np.repeat(d, 3, axis=2))

        if all_depth_frames:
            depth_video_path = (
                output_dir / "videos" / depth_key / "chunk-000" / f"file-{depth_file_index:03d}.mp4"
            )
            print(f"  编码 {len(all_depth_frames)} 帧 -> {depth_video_path}")
            encode_video_from_frames(all_depth_frames, depth_video_path, fps, codec=video_codec)
            print(f"  完成")

    # ==== 写 tasks parquet ====
    print("\n生成 meta/tasks.parquet ...")
    tasks_df = pd.DataFrame({"task_index": [0]}, index=pd.Index([task_name], name="__index_level_0__"))
    tasks_path = output_dir / "meta" / "tasks.parquet"
    tasks_df.to_parquet(str(tasks_path))

    # ==== 写 episodes parquet ====
    print("生成 meta/episodes parquet ...")
    episode_records = []
    dataset_from = 0
    for ep in episode_data_list:
        ep_idx = ep["episode_index"]
        n = ep["n_frames"]
        dataset_to = dataset_from + n

        # 每 episode 统计
        ep_action_stats = compute_stats(ep["action"])
        ep_state_stats = compute_stats(ep["state"])
        ep_img_stats = compute_image_stats(ep["frames_rgb"] or [])

        timestamps = [float(i) / fps for i in range(n)]
        frame_indices = list(range(n))
        global_indices = list(range(dataset_from, dataset_to))

        ts_stats = compute_stats(np.array(timestamps))
        fi_stats = compute_stats(np.array(frame_indices, dtype=np.float64))

        ep_idx_stats = {
            "min": [ep_idx], "max": [ep_idx],
            "mean": [float(ep_idx)], "std": [0.0], "count": [n],
        }
        idx_stats = compute_stats(np.array(global_indices, dtype=np.float64))
        task_stats = {"min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [n]}

        rec = {
            "episode_index": ep_idx,
            "tasks": [task_name],
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": dataset_from,
            "dataset_to_index": dataset_to,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }

        # RGB 视频信息
        if has_images:
            vinfo = episode_video_info.get(ep_idx, {"chunk": 0, "file": 0, "from_ts": 0.0, "to_ts": 0.0})
            rec[f"videos/{video_key}/chunk_index"] = vinfo["chunk"]
            rec[f"videos/{video_key}/file_index"] = vinfo["file"]
            rec[f"videos/{video_key}/from_timestamp"] = vinfo["from_ts"]
            rec[f"videos/{video_key}/to_timestamp"] = vinfo["to_ts"]

        # 深度视频信息
        if has_depth:
            dinfo = episode_depth_info.get(ep_idx, {"chunk": 0, "file": 0, "from_ts": 0.0, "to_ts": 0.0})
            rec[f"videos/{depth_key}/chunk_index"] = dinfo["chunk"]
            rec[f"videos/{depth_key}/file_index"] = dinfo["file"]
            rec[f"videos/{depth_key}/from_timestamp"] = dinfo["from_ts"]
            rec[f"videos/{depth_key}/to_timestamp"] = dinfo["to_ts"]

        # 统计信息
        stats_list = [
            ("stats/action", ep_action_stats),
            ("stats/observation.state", ep_state_stats),
            (f"stats/{video_key}", ep_img_stats),
            ("stats/timestamp", ts_stats),
            ("stats/frame_index", fi_stats),
            ("stats/episode_index", ep_idx_stats),
            ("stats/index", idx_stats),
            ("stats/task_index", task_stats),
        ]
        if has_depth:
            ep_depth_stats = compute_image_stats(ep["frames_depth"] or [])
            stats_list.append((f"stats/{depth_key}", ep_depth_stats))

        for prefix, stats in stats_list:
            for k in ["min", "max", "mean", "std", "count"]:
                rec[f"{prefix}/{k}"] = stats[k]

        episode_records.append(rec)
        dataset_from = dataset_to

    ep_df = pd.DataFrame(episode_records)
    ep_path = output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_df.to_parquet(str(ep_path), index=False)

    # ==== 写 stats.json ====
    print("生成 meta/stats.json ...")
    all_actions = np.concatenate([ep["action"] for ep in episode_data_list], axis=0)
    all_states = np.concatenate([ep["state"] for ep in episode_data_list], axis=0)
    all_timestamps = []
    all_frame_indices = []
    all_ep_indices = []
    all_global_indices = []
    gi = 0
    for ep in episode_data_list:
        n = ep["n_frames"]
        all_timestamps.extend([float(i) / fps for i in range(n)])
        all_frame_indices.extend(range(n))
        all_ep_indices.extend([ep["episode_index"]] * n)
        all_global_indices.extend(range(gi, gi + n))
        gi += n

    global_stats = {
        "action": compute_stats(all_actions),
        "observation.state": compute_stats(all_states),
        "timestamp": compute_stats(np.array(all_timestamps)),
        "frame_index": compute_stats(np.array(all_frame_indices, dtype=np.float64)),
        "episode_index": compute_stats(np.array(all_ep_indices, dtype=np.float64)),
        "index": compute_stats(np.array(all_global_indices, dtype=np.float64)),
        "task_index": {"min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [total_frames]},
    }

    # RGB 图像全局统计
    if has_images:
        all_rgb = []
        for ep in episode_data_list:
            if ep["frames_rgb"]:
                all_rgb.extend(ep["frames_rgb"])
        global_stats[video_key] = compute_image_stats(all_rgb)

    # 深度图像全局统计
    if has_depth:
        all_depth = []
        for ep in episode_data_list:
            if ep["frames_depth"]:
                all_depth.extend(ep["frames_depth"])
        global_stats[depth_key] = compute_image_stats(all_depth)

    stats_path = output_dir / "meta" / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(global_stats, f, indent=4)

    # ==== 写 info.json ====
    print("生成 meta/info.json ...")
    info = build_info_json(
        robot_type=robot_type,
        fps=fps,
        total_episodes=total_episodes,
        total_frames=total_frames,
        action_names=all_action_names,
        state_names=all_state_names,
        video_key=video_key if has_images else None,
        video_height=video_height,
        video_width=video_width,
        video_codec=get_video_codec_name(video_codec) if has_images else None,
    )

    # 添加深度视频到 features
    if has_depth:
        info["features"][depth_key] = {
            "dtype": "video",
            "shape": [depth_height, depth_width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": depth_height,
                "video.width": depth_width,
                "video.codec": get_video_codec_name(video_codec),
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": True,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    info_path = output_dir / "meta" / "info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"\n转换完成！输出目录: {output_dir}")
    print(f"  Episodes: {total_episodes}")
    print(f"  总帧数: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Action 维度: {n_action}")
    print(f"  State 维度: {n_state}")
    if has_images:
        print(f"  RGB 视频: {video_height}x{video_width}, codec={video_codec}")
    if has_depth:
        print(f"  深度视频: {depth_height}x{depth_width}, codec={video_codec}")


def main():
    parser = argparse.ArgumentParser(
        description="将 Dora-Record 数据集转换为 LeRobot v3.0 格式"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="输入的 Dora-Record session 目录（包含 episode_* 子目录）",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="输出的 LeRobot 数据集目录",
    )
    parser.add_argument(
        "--task", "-t", default="teleoperation",
        help="任务描述文本 (默认: teleoperation)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="输出数据集帧率 (默认: 30)",
    )
    parser.add_argument(
        "--video-codec", default="libx264",
        help="视频编码器 (默认: libx264, 可选: libsvtav1, libx265)",
    )
    parser.add_argument(
        "--robot-type", default="fourier_gr3",
        help="机器人类型 (默认: fourier_gr3)",
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="跳过图像/视频处理（无摄像头数据时使用）",
    )
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
