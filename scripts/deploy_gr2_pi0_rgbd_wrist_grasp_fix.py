#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""在 Fourier GR2 机器人上部署 PI0 策略 —— 多视觉输入版本（Orbbec RGB-D + 双腕部 RealSense RGB-D）。

基于 `deploy_gr2_pi0_rgbd.py`，在 Orbbec RGB-D 头顶相机的基础上，
新增左右腕部两路 Intel RealSense RGB-D 相机，共 6 路视觉输入：
  - observation.images.camera_top              → Orbbec RGB
  - observation.images.camera_top_depth        → Orbbec 深度伪彩色
  - observation.images.camera_left_wrist       → RealSense 左腕 RGB
  - observation.images.camera_left_wrist_depth → RealSense 左腕深度伪彩色
  - observation.images.camera_right_wrist      → RealSense 右腕 RGB
  - observation.images.camera_right_wrist_depth→ RealSense 右腕深度伪彩色

整体控制循环：
  1. 从 Orbbec 读取 RGB + 深度帧
  2. 从左右 RealSense 各读取 RGB + 深度帧
  3. 从 Aurora SDK 获取 GR2 机器人状态
  4. 将 6 路图像 + 状态按策略要求打包成 observation
  5. preprocessor → PI0 推理 → postprocessor 得到动作
  6. clamp / stabilize 后下发到机器人
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path
from threading import Event, Lock, Thread

import cv2
import numpy as np
import torch

import deploy_gr2_pi0 as base
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import prepare_observation_for_inference

LOGGER = logging.getLogger(__name__)
_RECEIVED_STOP_SIGNAL = False

# ---------------------------------------------------------------------------
# PD 增益配置（与 demo_joint_command.py 对齐）
# Kp 控制刚度（越大关节越"硬"），Kd 控制阻尼（抑制振荡）
# ---------------------------------------------------------------------------
PD_KP_CONFIG = {
    'left_manipulator': [300, 300, 100, 100, 50, 50, 50],
    'right_manipulator': [300, 300, 100, 100, 50, 50, 50],
    'waist': [200],
    'head': [100, 100],
}
PD_KD_CONFIG = {
    'left_manipulator': [10, 10, 5, 5, 5, 5, 5],
    'right_manipulator': [10, 10, 5, 5, 5, 5, 5],
    'waist': [10],
    'head': [10, 10],
}

# ---------------------------------------------------------------------------
# 初始姿态（从数据集 task 2 第一帧提取，URDF 空间，弧度）
# 部署前先把机器人移到这个姿态，减少首帧跳变
# ---------------------------------------------------------------------------
INIT_POSE_URDF = np.array([
    # left_arm (7)
    -0.2774,  0.047, -1.7668,  0.0163,  0.4624,  0.5922, -0.4454,
    # right_arm (7)
     0.0262, -0.06,  -0.3199, -1.4412,  0.0116, -0.0373,  0.3151,
    # left_hand (6)
     0.0, 0.0, -1.57, 0.0, 0.0, 0.0,
    # right_hand (6)
     0.0, 0.0, -1.57, 0.0, 0.0, 0.0,
    # head (2)
     0.0, 0.0,
    # waist (1)
     0.0,
    # base (6) — 不下发
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Threaded camera reader
# ---------------------------------------------------------------------------


class ThreadedFrameGrabber:
    """后台线程持续抓帧，主循环通过 get_latest() 零阻塞获取最新帧。

    设计思路：
      - 每个相机实例对应一个 daemon 线程，以相机自身帧率持续调用 read_fn()
      - 最新帧通过 Lock 保护写入 _latest，主循环随时可读
      - 内置帧计数器 _frame_id，主循环可通过 get_new_frame() 判断是否有新帧，
        避免对同一帧重复做 GPU 推理（PI0 单次推理 ~15-30ms，跳过重复帧可显著降负载）

    线程安全：
      - _latest / _frame_id 的读写均在 _lock 保护下完成
      - read_fn 本身不需要线程安全（每个相机独占一个线程）

    Args:
        read_fn: 可调用对象，返回帧数据或 None（读取失败时）。
                 Orbbec 返回 (rgb, depth_rgb, depth_u16, depth_vis_bgr) 四元组；
                 RealSense 返回 (rgb, depth_rgb) 二元组。
        name:    相机标识名，用于日志输出。
    """

    def __init__(self, read_fn, name: str) -> None:
        self._read_fn = read_fn
        self._name = name
        self._lock = Lock()
        self._latest = None
        self._frame_id: int = 0          # 每次成功读取递增，用于判断帧是否更新
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        """启动后台抓帧线程。"""
        self._stop_event.clear()
        self._thread = Thread(target=self._loop, name=f"grabber_{self._name}", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        """后台线程主循环：持续读帧并更新 _latest。

        - 读取成功：更新 _latest 和 _frame_id
        - 读取返回 None：短暂 sleep 避免空转烧 CPU（相机可能在初始化中）
        - 读取抛异常：记录警告后继续，不中断线程（相机偶发错误不应终止部署）
        """
        while not self._stop_event.is_set():
            try:
                frame = self._read_fn()
            except Exception as exc:
                LOGGER.warning("ThreadedFrameGrabber[%s] read error: %s", self._name, exc)
                frame = None
            if frame is not None:
                with self._lock:
                    self._latest = frame
                    self._frame_id += 1
            else:
                # 读取失败时短暂等待，避免无意义的高频重试
                time.sleep(0.005)

    def get_latest(self):
        """返回最新帧（可能为 None，如果后台线程还没拿到第一帧）。"""
        with self._lock:
            return self._latest

    def get_new_frame(self, last_seen_id: int) -> tuple[int, object | None]:
        """仅在有新帧时返回，避免主循环对同一帧重复推理。

        Args:
            last_seen_id: 主循环上次处理的 frame_id

        Returns:
            (current_frame_id, frame) — 如果 frame_id > last_seen_id 则返回新帧，
            否则 frame 为 None 表示没有更新。
        """
        with self._lock:
            if self._frame_id > last_seen_id:
                return self._frame_id, self._latest
            return self._frame_id, None

    def stop(self) -> None:
        """停止后台线程并等待其退出（最多 2 秒）。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ---------------------------------------------------------------------------
# 新 SDK API 封装（替代 base 中的旧接口）
# ---------------------------------------------------------------------------


def configure_pd_gains(client) -> None:
    """在部署开始前配置电机 PD 增益，提升关节跟踪精度。

    不设置的话用控制器默认值，可能偏软导致跟踪延迟。
    配置后回读左臂 Kp 做一次验证。
    """
    client.set_motor_cfg(kp_config=PD_KP_CONFIG, kd_config=PD_KD_CONFIG)
    LOGGER.info("PD gains configured. left_manipulator Kp: %s", PD_KP_CONFIG['left_manipulator'])


def send_action_to_robot(client, action_urdf: np.ndarray, send_base: bool) -> None:
    """用新 SDK 的 set_group_cmd 替代旧的 set_joint_positions 下发关节位置。

    流程与 base.send_action_to_robot 相同：
      1. 手部关节从 URDF 空间转换到 SDK 空间
      2. 按 group 切片打包成 dict
      3. 调用 set_group_cmd(position_cmd=dict) 下发

    Args:
        client:      AuroraClient 实例
        action_urdf: 35D URDF 空间动作向量
        send_base:   是否同时下发底盘速度指令
    """
    action_send = action_urdf.copy()
    action_send[14:20] = base.hand_urdf_to_sdk(action_send[14:20], clip=True)
    action_send[20:26] = base.hand_urdf_to_sdk(action_send[20:26], clip=True)

    position_dict = {
        group_name: action_send[group_slice].tolist()
        for group_name, group_slice in base.ACTION_GROUP_SLICES.items()
    }
    # 旧 SDK API：set_joint_positions
    client.set_joint_positions(position_dict)

    if send_base:
        base._try_send_base_velocity(client, action_urdf)


def smooth_transition_move_cmd(client, target_action_urdf: np.ndarray, duration_s: float = 2.0, frequency_hz: int = 100) -> None:
    """首帧平滑过渡：线性插值从当前位置到目标位置。

    兼容旧版 SDK (0.1.1)，使用 set_joint_positions 逐步下发。

    Args:
        client:            AuroraClient 实例
        target_action_urdf: 目标 35D URDF 动作向量
        duration_s:        过渡时间（秒），默认 2.0
        frequency_hz:      插值频率（Hz），默认 100
    """
    if duration_s <= 0:
        send_action_to_robot(client, target_action_urdf, send_base=False)
        return

    init_joints = base.get_robot_state_urdf(client)[:29]
    target_joints = target_action_urdf[:29]
    total_steps = max(1, int(duration_s * frequency_hz))
    LOGGER.info("Running smooth transition for %.2fs (%d steps).", duration_s, total_steps)

    for step in range(total_steps + 1):
        alpha = step / total_steps
        interp_action = init_joints * (1.0 - alpha) + target_joints * alpha
        send_action_to_robot(client, interp_action, send_base=False)
        time.sleep(1.0 / frequency_hz)

    LOGGER.info("Smooth transition complete.")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _ordered_visual_keys(visual_keys: list[str], preferred_key: str) -> list[str]:
    """对视觉键列表排序，确保 preferred_key 排在最前面。"""
    if preferred_key in visual_keys:
        return [preferred_key] + [k for k in visual_keys if k != preferred_key]
    return list(visual_keys)


def _depth_u16_to_rgb(depth_u16: np.ndarray) -> np.ndarray:
    """将 uint16 深度图转换为 3 通道伪彩色 RGB 图像，供模型输入使用。

    转换流程：
      1. 使用固定量程 MAX_DEPTH_MM=1500mm，将有效深度线性映射到 [0, 255]
      2. 深度为 0 的无效像素保持为 0；超过 MAX_DEPTH_MM 的像素截断为 255
      3. 用 JET colormap 上色，转为 RGB 格式

    Args:
        depth_u16: 原始深度图 (H, W)，dtype=uint16，单位 mm。值为 0 表示无效。

    Returns:
        伪彩色 RGB 图像 (H, W, 3)，dtype=uint8
    """
    MAX_DEPTH_MM = 1500.0

    if not np.any(depth_u16):
        return np.zeros((*depth_u16.shape, 3), dtype=np.uint8)

    norm = np.zeros_like(depth_u16, dtype=np.uint8)
    valid_mask = depth_u16 > 0
    clipped_depth = np.clip(depth_u16.astype(np.float32), 0.0, MAX_DEPTH_MM)
    norm[valid_mask] = np.rint(clipped_depth[valid_mask] / MAX_DEPTH_MM * 255.0).astype(np.uint8)

    colored_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colored_bgr[depth_u16 == 0] = 0
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def _build_visual_observation(
    visual_keys: list[str],
    top_rgb: np.ndarray,
    top_depth_rgb: np.ndarray,
    left_wrist_rgb: np.ndarray,
    left_wrist_depth_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
    right_wrist_depth_rgb: np.ndarray,
) -> dict[str, np.ndarray]:
    """根据策略要求的视觉键，构建视觉观测字典。

    分配规则（按优先级从高到低匹配）：
      - "left_wrist" + "depth" → 左腕深度伪彩色
      - "right_wrist" + "depth"→ 右腕深度伪彩色
      - "left_wrist"           → 左腕 RGB
      - "right_wrist"          → 右腕 RGB
      - "depth"                → 头顶深度伪彩色
      - 其他                    → 头顶 RGB
    """
    obs: dict[str, np.ndarray] = {}
    for key in visual_keys:
        kl = key.lower()
        if "left_wrist" in kl and "depth" in kl:
            obs[key] = left_wrist_depth_rgb
        elif "right_wrist" in kl and "depth" in kl:
            obs[key] = right_wrist_depth_rgb
        elif "left_wrist" in kl:
            obs[key] = left_wrist_rgb
        elif "right_wrist" in kl:
            obs[key] = right_wrist_rgb
        elif "depth" in kl:
            obs[key] = top_depth_rgb
        else:
            obs[key] = top_rgb
    return obs


def _create_realsense_camera(
    serial: str,
    width: int,
    height: int,
    fps: int,
) -> RealSenseCamera:
    """创建并连接一个 RealSense RGB-D 相机实例。

    Args:
        serial: 相机序列号或设备名（如 "420222072816"）
        width:  图像宽度（像素）
        height: 图像高度（像素）
        fps:    目标帧率

    Returns:
        已连接的 RealSenseCamera 实例（use_depth=True）
    """
    cfg = RealSenseCameraConfig(
        serial_number_or_name=serial,
        fps=fps,
        width=width,
        height=height,
        use_depth=True,
    )
    cam = RealSenseCamera(cfg)
    cam.connect()
    return cam


def _auto_detect_realsense_pair() -> tuple[str, str]:
    """自动发现恰好两个 RealSense 相机，返回 (left_serial, right_serial)。

    按序列号排序，较小的分配给左腕，较大的分配给右腕。
    如果不是恰好两个设备则抛出异常。
    """
    cameras = RealSenseCamera.find_cameras()
    if len(cameras) != 2:
        sns = [c["id"] for c in cameras]
        raise RuntimeError(
            f"自动发现需要恰好 2 个 RealSense 相机，但找到 {len(cameras)} 个: {sns}。"
            f"请用 --wrist-left-serial / --wrist-right-serial 手动指定。"
        )
    # 按序列号排序：小的 → 左腕，大的 → 右腕
    sorted_cams = sorted(cameras, key=lambda c: c["id"])
    left_sn = sorted_cams[0]["id"]
    right_sn = sorted_cams[1]["id"]
    LOGGER.info(
        "Auto-detected RealSense pair: left=%s (%s), right=%s (%s)",
        left_sn, sorted_cams[0].get("name", "?"),
        right_sn, sorted_cams[1].get("name", "?"),
    )
    return left_sn, right_sn


def _apply_right_hand_grasp_bias(action_urdf: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """测试用：对右手抓取关节施加额外偏置，验证是否因抓取闭合过弱导致任务卡住。"""
    close_bias = float(args.right_hand_close_bias)
    thumb_bias = float(args.right_hand_thumb_bias)
    if close_bias == 0.0 and thumb_bias == 0.0:
        return action_urdf

    action_urdf = action_urdf.copy()
    action_urdf[20:24] -= close_bias
    action_urdf[24] += thumb_bias
    return action_urdf


def _read_realsense_rgbd(cam: RealSenseCamera, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """从单次 frameset 中同时读取 RGB + 深度帧，返回 (rgb, depth_rgb) 或 None。

    为什么不用 cam.read() + cam.read_depth()：
      这两个方法各自调用一次 try_wait_for_frames()，拿到的是两个不同的 frameset。
      第二次调用会阻塞等下一帧（~33ms@30fps），且 RGB/深度不对齐。
      直接访问 rs_pipeline 可以从同一个 frameset 中提取两路数据，零额外延迟。

    注意：_depth_u16_to_rgb 的 CPU 开销（~1-2ms）在后台线程中完成，不占主循环时间。

    Args:
        cam:  已连接的 RealSenseCamera 实例
        name: 相机标识名，用于日志

    Returns:
        (rgb, depth_rgb) 元组，或 None（读取失败时）
    """
    try:
        if cam.rs_pipeline is None:
            raise RuntimeError(f"RealSense {name}: pipeline not initialized.")
        ret, frames = cam.rs_pipeline.try_wait_for_frames(timeout_ms=200)
        if not ret or frames is None:
            LOGGER.warning("RealSense %s: no frame available.", name)
            return None

        # 从同一个 frameset 中提取 color 和 depth
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            LOGGER.warning("RealSense %s: incomplete frameset (color=%s depth=%s).", name, bool(color_frame), bool(depth_frame))
            return None

        rgb = np.asanyarray(color_frame.get_data())
        # 应用 RealSenseCamera 的后处理（颜色转换、旋转）
        rgb = cam._postprocess_image(rgb)

        depth_u16 = np.asanyarray(depth_frame.get_data())
        depth_u16 = cam._postprocess_image(depth_u16, depth_frame=True)
        depth_rgb = _depth_u16_to_rgb(depth_u16)

        return rgb, depth_rgb
    except Exception as exc:
        LOGGER.warning("RealSense %s read failed: %s", name, exc)
        return None


# ---------------------------------------------------------------------------
# Warmup & Inference
# ---------------------------------------------------------------------------


def warmup_policy(
    policy: PI0Policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
    robot_type: str,
    visual_keys: list[str],
    state_key: str,
    camera_height: int,
    camera_width: int,
    wrist_height: int,
    wrist_width: int,
    state_dim: int,
) -> None:
    """用全零假数据做一次完整前向传播，触发 JIT/CUDA 懒初始化。

    为什么需要区分 camera_height/width 和 wrist_height/width：
      头顶 Orbbec 和腕部 RealSense 可能使用不同分辨率（如 640x480 vs 320x240），
      PI0 的 image encoder 会根据输入尺寸分配不同的 CUDA 内存。
      warmup 时必须用实际尺寸，否则首帧推理仍会触发重新分配。
    """
    h, w = camera_height, camera_width
    wh, ww = wrist_height, wrist_width

    dummy_obs: dict[str, np.ndarray] = {}
    for key in visual_keys:
        kl = key.lower()
        is_wrist = "left_wrist" in kl or "right_wrist" in kl
        ch, cw = (wh, ww) if is_wrist else (h, w)
        dummy_obs[key] = np.zeros((ch, cw, 3), dtype=np.uint8)
    dummy_obs[state_key] = np.zeros((state_dim,), dtype=np.float32)

    with torch.inference_mode():
        batch = prepare_observation_for_inference(
            observation=dummy_obs, device=device, task=task, robot_type=robot_type,
        )
        _ = postprocessor(policy.select_action(preprocessor(batch)))


def infer_single_action(
    policy: PI0Policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
    robot_type: str,
    visual_keys: list[str],
    state_key: str,
    top_rgb: np.ndarray,
    top_depth_rgb: np.ndarray,
    left_wrist_rgb: np.ndarray,
    left_wrist_depth_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
    right_wrist_depth_rgb: np.ndarray,
    state_model: np.ndarray,
    action_dim: int,
) -> np.ndarray:
    """执行一次 PI0 六路视觉输入推理，返回单步动作向量。

    推理管线：
      1. _build_visual_observation 按键名规则将 6 路图像映射到策略要求的视觉键
      2. 加入状态向量，构成完整 observation dict
      3. prepare_observation_for_inference 将 numpy → batch tensor（含 task 文本编码）
      4. preprocessor 归一化 → policy.select_action 前向推理 → postprocessor 反归一化
      5. 从输出 tensor 中提取第一个动作（PI0 输出 action chunk），裁剪到 action_dim

    Returns:
        action 向量 (action_dim,)，dtype=float32
    """
    observation = _build_visual_observation(
        visual_keys, top_rgb, top_depth_rgb,
        left_wrist_rgb, left_wrist_depth_rgb,
        right_wrist_rgb, right_wrist_depth_rgb,
    )
    observation[state_key] = state_model

    with torch.inference_mode():
        batch = prepare_observation_for_inference(
            observation=observation, device=device, task=task, robot_type=robot_type,
        )
        action_tensor = postprocessor(policy.select_action(preprocessor(batch)))

    # 返回完整 action chunk (N, action_dim)，由主循环逐步消费
    all_actions = action_tensor.reshape(-1, action_tensor.shape[-1]).detach().cpu().numpy().astype(np.float32)
    return np.array([base.fit_vector(a, action_dim) for a in all_actions])


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """在 base 参数基础上追加腕部 RealSense 相机参数。"""
    parser = argparse.ArgumentParser(
        description="Deploy Pi0 on GR2 (Orbbec RGB-D + dual wrist RealSense RGB-D).",
    )

    # --- 直接从 base 脚本复制核心参数（保持兼容） ---
    parser.add_argument(
        "--checkpoint-path", type=str,
        default="outputs/train/pi0_gr2_grab_bottle_multicam_ft_20260314_bs4_stable/checkpoints/100000/pretrained_model",
    )
    parser.add_argument(
        "--dataset-root", type=str,
        default="/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/gr2_grab_bottle_merged",
        help="数据集根目录，用于读取可选 task 列表。",
    )
    parser.add_argument("--task", type=str, default=None, help="Task 文本。留空则交互式选择。")
    parser.add_argument("--robot-type", type=str, default=base.DEFAULT_ROBOT_TYPE)
    parser.add_argument("--domain-id", type=int, default=123)
    parser.add_argument("--robot-name", type=str, default="gr2")
    parser.add_argument("--client-init-retries", type=int, default=4)
    parser.add_argument("--client-retry-interval-s", type=float, default=2.0)
    parser.add_argument("--fsm-state", type=int, default=11)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--transition-time-s", type=float, default=6.0)
    parser.add_argument("--transition-freq", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--send-base", action="store_true")
    parser.add_argument("--disable-clamp", action="store_true")
    parser.add_argument("--action-ema-alpha", type=float, default=0.35)
    parser.add_argument("--arm-ema-alpha", type=float, default=-1.0)
    parser.add_argument("--hand-ema-alpha", type=float, default=-1.0)
    parser.add_argument("--max-arm-delta", type=float, default=0.06)
    parser.add_argument("--max-hand-delta", type=float, default=0.12)
    parser.add_argument("--max-head-waist-delta", type=float, default=0.08)
    parser.add_argument("--max-base-delta", type=float, default=0.15)
    parser.add_argument("--slow-loop-warn-ms", type=float, default=120.0)
    parser.add_argument("--skip-confirm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--camera-key", type=str, default="observation.images.camera_top")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-timeout-ms", type=int, default=200)
    parser.add_argument("--camera-warmup-frames", type=int, default=15)
    parser.add_argument("--camera-init-retries", type=int, default=6)
    parser.add_argument("--camera-retry-interval-s", type=float, default=1.0)
    parser.add_argument("--state-dim", type=int, default=0)
    parser.add_argument("--action-dim", type=int, default=0)
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--save-dir", type=str, default="scripts/outputs/deploy_gr2_pi0_wrist")
    parser.add_argument("--save-every", type=int, default=0)

    test_group = parser.add_argument_group("Test hooks")
    test_group.add_argument(
        "--right-hand-close-bias", type=float, default=0.0,
        help="测试用：对右手前四个抓取关节施加额外闭合偏置（URDF 空间，越大闭合越强）。",
    )
    test_group.add_argument(
        "--right-hand-thumb-bias", type=float, default=0.0,
        help="测试用：对右手拇指 pitch 施加额外偏置（URDF 空间）。",
    )

    # --- 腕部 RealSense 相机参数 ---
    wrist = parser.add_argument_group("Wrist RealSense cameras")
    wrist.add_argument(
        "--wrist-left-serial", type=str, default="420222072816",
        help="左腕 RealSense 序列号或设备名。留空则自动发现。",
    )
    wrist.add_argument(
        "--wrist-right-serial", type=str, default="349522072801",
        help="右腕 RealSense 序列号或设备名。留空则自动发现。",
    )
    wrist.add_argument("--wrist-width", type=int, default=640, help="腕部相机图像宽度。")
    wrist.add_argument("--wrist-height", type=int, default=480, help="腕部相机图像高度。")
    wrist.add_argument("--wrist-fps", type=int, default=30, help="腕部相机帧率。")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main deployment loop
# ---------------------------------------------------------------------------


def _resolve_task(args) -> str:
    """根据 --task 和 --dataset-root 解析最终 task 文本。

    如果 --task 已指定，直接返回。
    否则从数据集 meta/tasks.parquet 读取可选 task 列表，交互式让用户选择。
    """
    if args.task:
        return args.task

    tasks_path = Path(args.dataset_root) / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        LOGGER.warning("未找到 %s，使用默认 task。", tasks_path)
        return base.DEFAULT_TASK

    import pyarrow.parquet as pq
    table = pq.read_table(str(tasks_path))
    df = table.to_pandas()
    # 兼容两种格式：
    #   格式 A: 列 "task" 存放文本（标准 LeRobot v3.0）
    #   格式 B: 文本在 index 中，列只有 "task_index"（某些转换工具生成）
    if "task" in df.columns:
        tasks = df["task"].tolist()
    else:
        tasks = df.index.tolist()

    if len(tasks) == 0:
        LOGGER.warning("数据集中没有 task，使用默认 task。")
        return base.DEFAULT_TASK
    if len(tasks) == 1:
        LOGGER.info("数据集中只有 1 个 task: %s", tasks[0])
        return tasks[0]

    # 交互式选择（编号从 1 开始）
    print("\n可选 task:")
    for i, t in enumerate(tasks, 1):
        print(f"  [{i}] {t}")
    while True:
        choice = input(f"\n请选择 task 编号 [1-{len(tasks)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(tasks):
            selected = tasks[int(choice) - 1]
            LOGGER.info("选择 task: %s", selected)
            return selected
        print(f"无效输入，请输入 1 到 {len(tasks)} 之间的数字。")


def run(args) -> None:
    """主部署流程：初始化 → 控制循环 → 清理。"""

    # ===== (A) 初始化 =====
    args.task = _resolve_task(args)
    pretrained_dir = base.resolve_pretrained_model_dir(args.checkpoint_path)
    train_cfg = base.load_train_config(pretrained_dir)
    device = base.select_device(args.device)

    LOGGER.info("Model: %s | Device: %s", pretrained_dir, device)
    if train_cfg:
        ds = train_cfg.get("dataset", {})
        LOGGER.info("Train dataset: repo_id=%s root=%s", ds.get("repo_id"), ds.get("root"))
    LOGGER.info(
        "Top camera: %dx%d@%d | Wrist cameras: %dx%d@%d",
        args.camera_width, args.camera_height, args.camera_fps,
        args.wrist_width, args.wrist_height, args.wrist_fps,
    )
    if args.right_hand_close_bias != 0.0 or args.right_hand_thumb_bias != 0.0:
        LOGGER.warning(
            "Test hook enabled: right_hand_close_bias=%.4f right_hand_thumb_bias=%.4f",
            args.right_hand_close_bias,
            args.right_hand_thumb_bias,
        )

    # --- 加载 PI0 策略 ---
    policy = PI0Policy.from_pretrained(str(pretrained_dir), strict=False).to(device)
    policy.eval()
    # 关闭 gradient checkpointing（训练配置残留，推理时拖慢速度）
    if hasattr(policy.model, 'gradient_checkpointing_disable'):
        policy.model.gradient_checkpointing_disable()
        LOGGER.info("Disabled gradient checkpointing for inference.")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # --- 推断 I/O 规格 ---
    state_key, cfg_state_dim, visual_keys, action_key, cfg_action_dim = base.infer_policy_io(policy)
    state_dim = args.state_dim if args.state_dim > 0 else cfg_state_dim
    action_dim = args.action_dim if args.action_dim > 0 else cfg_action_dim
    visual_keys = _ordered_visual_keys(visual_keys, args.camera_key)

    LOGGER.info(
        "Policy IO: state=%s(%d) action=%s(%d) visual_keys=%s",
        state_key, state_dim, action_key, action_dim, visual_keys,
    )

    # --- GUI & 快照 ---
    gui_enabled = base.setup_gui(args)
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    last_vis_ts = time.time()
    frame_idx = 0
    periodic_save_every = max(0, int(args.save_every))
    if not gui_enabled and not args.no_gui and periodic_save_every <= 0:
        periodic_save_every = 30
        LOGGER.info("GUI unavailable. Saving snapshots every %d steps to %s", periodic_save_every, save_dir)

    # --- 初始化 Orbbec 头顶相机 ---
    orbbec = base.OrbbecRGBDCamera(
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        timeout_ms=args.camera_timeout_ms,
        warmup_frames=args.camera_warmup_frames,
        init_retries=args.camera_init_retries,
        retry_interval_s=args.camera_retry_interval_s,
    )
    cam_left: RealSenseCamera | None = None
    cam_right: RealSenseCamera | None = None
    client = None
    grabbers: list[ThreadedFrameGrabber] = []

    try:
        orbbec.connect()

        # --- 初始化左右腕部 RealSense 相机 ---
        left_sn = args.wrist_left_serial
        right_sn = args.wrist_right_serial

        if not left_sn and not right_sn:
            LOGGER.info("Auto-detecting RealSense cameras...")
            left_sn, right_sn = _auto_detect_realsense_pair()
        elif not left_sn or not right_sn:
            raise ValueError("必须同时指定左右腕序列号，或同时留空以自动发现。")

        LOGGER.info("Connecting left wrist RealSense (SN=%s)...", left_sn)
        cam_left = _create_realsense_camera(
            left_sn, args.wrist_width, args.wrist_height, args.wrist_fps,
        )
        LOGGER.info("Connecting right wrist RealSense (SN=%s)...", right_sn)
        cam_right = _create_realsense_camera(
            right_sn, args.wrist_width, args.wrist_height, args.wrist_fps,
        )

        # --- 启动后台抓帧线程（3 路相机并行读取） ---
        # 每个相机一个独立 daemon 线程，以相机自身帧率持续读取。
        # 主循环只取最新帧，相机 I/O 开销从串行 ~100ms 降到 ~0ms。
        # 注意：lambda 捕获的 cam_left / cam_right 此时已完成赋值，闭包安全。
        grabber_orbbec = ThreadedFrameGrabber(orbbec.read, "orbbec")
        grabber_left = ThreadedFrameGrabber(lambda: _read_realsense_rgbd(cam_left, "left_wrist"), "left_wrist")
        grabber_right = ThreadedFrameGrabber(lambda: _read_realsense_rgbd(cam_right, "right_wrist"), "right_wrist")
        grabbers = [grabber_orbbec, grabber_left, grabber_right]
        for g in grabbers:
            g.start()

        # 等待所有相机至少拿到第一帧（超时 10 秒，覆盖 RealSense 的 warmup 时间）
        LOGGER.info("Waiting for first frames from all cameras...")
        _wait_start = time.time()
        while time.time() - _wait_start < 10.0:
            if all(g.get_latest() is not None for g in grabbers):
                break
            time.sleep(0.05)
        else:
            missing = [g._name for g in grabbers if g.get_latest() is None]
            raise RuntimeError(f"Timeout waiting for first frame from: {missing}")
        LOGGER.info("All cameras streaming.")

        # --- 连接机器人 ---
        client = base.setup_robot_if_needed(args)

        # --- 配置 PD 增益（暂时禁用） ---
        # if client is not None:
        #     configure_pd_gains(client)

        # --- 移动到初始姿态 ---
        # 从数据集提取的初始姿态，减少首帧跳变
        if client is not None:
            LOGGER.info("Moving to initial pose (from dataset)...")
            smooth_transition_move_cmd(client, INIT_POSE_URDF, duration_s=3.0, frequency_hz=100)
            LOGGER.info("Initial pose reached.")

        # --- 模型预热 ---
        # 用全零假数据跑一次完整推理管线，触发 CUDA kernel 编译和内存分配。
        # 不做 warmup 的话首帧推理延迟可能 >500ms，导致机器人抖动。
        LOGGER.info("Warming up policy...")
        warmup_policy(
            policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
            device=device, task=args.task, robot_type=args.robot_type,
            visual_keys=visual_keys, state_key=state_key,
            camera_height=args.camera_height, camera_width=args.camera_width,
            wrist_height=args.wrist_height, wrist_width=args.wrist_width,
            state_dim=state_dim,
        )
        LOGGER.info("Warm-up complete.")

        # ===== (B) 主控制循环 =====
        # 帧 ID 追踪：只有当任一相机产生新帧时才执行推理，
        # 避免对同一帧重复做 GPU 推理（PI0 单次 ~15-30ms，跳过可显著降负载）。
        step = 0
        first_action = True
        policy.reset()
        prev_action_urdf: np.ndarray | None = None
        slow_loop_last_log_ts = time.time()
        last_orbbec_id = last_left_id = last_right_id = 0
        action_chunk: np.ndarray | None = None  # 缓存的 action chunk (N, 35)
        chunk_idx = 0  # 当前消费到 chunk 的第几步

        while True:
            loop_start = time.perf_counter()

            # --- Step 1: 从后台线程获取最新帧（零阻塞） ---
            # get_new_frame() 返回 (frame_id, frame)，frame 为 None 表示没有新帧
            orbbec_id, orbbec_frame = grabber_orbbec.get_new_frame(last_orbbec_id)
            left_id, left_result = grabber_left.get_new_frame(last_left_id)
            right_id, right_result = grabber_right.get_new_frame(last_right_id)
            has_new_frame = any(frame is not None for frame in (orbbec_frame, left_result, right_result))

            latest_orbbec = grabber_orbbec.get_latest()
            latest_left = grabber_left.get_latest()
            latest_right = grabber_right.get_latest()

            # 任一相机尚未就绪（首帧前）→ 短暂等待
            if latest_orbbec is None or latest_left is None or latest_right is None:
                time.sleep(0.001)
                continue

            # 三路都没有新帧且没有历史 chunk 可补帧 → 等待下一轮
            if not has_new_frame and (action_chunk is None or len(action_chunk) == 0):
                time.sleep(0.001)
                continue

            # 部分相机有新帧、部分没有 → 用上一次的帧补齐（get_latest 兜底）
            if orbbec_frame is None:
                orbbec_frame = latest_orbbec
            if left_result is None:
                left_result = latest_left
            if right_result is None:
                right_result = latest_right

            # 更新帧 ID 水位线
            last_orbbec_id = orbbec_id
            last_left_id = left_id
            last_right_id = right_id

            top_rgb, top_depth_rgb, depth_u16, depth_vis_bgr = orbbec_frame
            lw_rgb, lw_depth_rgb = left_result
            rw_rgb, rw_depth_rgb = right_result

            # --- Step 2: 读取机器人状态 ---
            # dry_run 模式下用全零向量代替，不连接真实机器人
            if args.dry_run:
                state_full = np.zeros((45,), dtype=np.float32)
            else:
                if client is None:
                    raise RuntimeError("Robot client is not initialized.")
                # 获取 URDF 空间下的 45D 状态向量（7 左臂 + 7 右臂 + 6 左手 + 6 右手 + ...）
                state_full = base.get_robot_state_urdf(client)
            # 裁剪/填充到模型期望的 state_dim 维度
            state_model = base.fit_vector(state_full, state_dim)

            # --- Step 3: 六路视觉推理（Receding Horizon / Sliding Window） ---
            # 任一相机有新帧 → 立即重规划并执行 chunk 第 0 步。
            # 若当前 tick 没有新帧 → 顺延执行历史 chunk 的后续动作补帧。
            if has_new_frame:
                action_chunk = infer_single_action(
                    policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
                    device=device, task=args.task, robot_type=args.robot_type,
                    visual_keys=visual_keys, state_key=state_key,
                    top_rgb=top_rgb, top_depth_rgb=top_depth_rgb,
                    left_wrist_rgb=lw_rgb, left_wrist_depth_rgb=lw_depth_rgb,
                    right_wrist_rgb=rw_rgb, right_wrist_depth_rgb=rw_depth_rgb,
                    state_model=state_model, action_dim=action_dim,
                )
                chunk_idx = 0
                LOGGER.info("New action chunk generated (%d steps).", len(action_chunk))
                action_model = action_chunk[0]
            else:
                if action_chunk is None or len(action_chunk) == 0:
                    time.sleep(0.001)
                    continue
                chunk_idx = min(chunk_idx + 1, len(action_chunk) - 1)
                action_model = action_chunk[chunk_idx]

            # --- Step 4: 动作后处理 ---
            # 将模型输出适配到 GR2 的 35D URDF 动作空间：
            #   [0:7] 左臂, [7:14] 右臂, [14:20] 左手, [20:26] 右手,
            #   [26:28] 头, [28:29] 腰, [29:35] 底盘
            raw_action_urdf = base.fit_vector(action_model, 35)
            prev_action_for_diag = prev_action_urdf
            action_urdf = raw_action_urdf.copy()
            if not args.disable_clamp:
                # 夹紧非手部关节到安全范围，防止超出关节限位
                action_urdf = base.clamp_non_hand_joint_action(action_urdf)
            if prev_action_urdf is not None:
                # EMA 平滑 + 步间变化量限制，减少抖动
                action_urdf = base.stabilize_action(action_urdf, prev_action_urdf, args)
            # 手部保留原始阶跃闭合信号，不参与 EMA 平滑。
            action_urdf[14:20] = raw_action_urdf[14:20]
            action_urdf[20:26] = raw_action_urdf[20:26]
            action_urdf = _apply_right_hand_grasp_bias(action_urdf, args)

            # --- Step 5: 下发动作 ---
            if not args.dry_run:
                if client is None:
                    raise RuntimeError("Robot client is not initialized.")
                if first_action and args.transition_time_s > 0:
                    # 首帧：线性插值平滑过渡到目标位置
                    smooth_transition_move_cmd(
                        client,
                        action_urdf,
                        duration_s=args.transition_time_s,
                        frequency_hz=args.transition_freq,
                    )
                    policy.reset()
                    action_chunk = None
                    chunk_idx = 0
                    first_action = False
                else:
                    # 正常帧：用 set_joint_positions 下发关节位置
                    send_action_to_robot(client, action_urdf, args.send_base)
                    first_action = False

            # --- Step 6: 日志 ---
            if step % max(1, args.log_every) == 0:
                # 计算手臂关节动作与当前状态的偏差，用于判断执行效果
                arm_dim = min(14, state_full.shape[0], action_urdf.shape[0])
                arm_delta = float(np.linalg.norm(action_urdf[:arm_dim] - state_full[:arm_dim]))
                valid_depth = depth_u16[depth_u16 > 0]
                depth_range = "empty" if valid_depth.size == 0 else f"{int(valid_depth.min())}..{int(valid_depth.max())}"
                base.log_action_diagnostics(
                    step, arm_delta, depth_range,
                    raw_action_urdf, action_urdf, prev_action_for_diag, args,
                )

                obs_left_arm = state_full[0:7]
                obs_right_arm = state_full[7:14]
                obs_left_hand = state_full[14:20]
                obs_right_hand = state_full[20:26]

                tgt_left_arm = raw_action_urdf[0:7]
                tgt_right_arm = raw_action_urdf[7:14]
                tgt_left_hand = raw_action_urdf[14:20]
                tgt_right_hand = raw_action_urdf[20:26]

                err_left_arm = tgt_left_arm - obs_left_arm
                err_right_arm = tgt_right_arm - obs_right_arm
                err_left_hand = tgt_left_hand - obs_left_hand
                err_right_hand = tgt_right_hand - obs_right_hand

                LOGGER.info(
                    "Obs/Target/Error | left_arm  obs=%s | target=%s | error=%s",
                    np.round(obs_left_arm, 4).tolist(),
                    np.round(tgt_left_arm, 4).tolist(),
                    np.round(err_left_arm, 4).tolist(),
                )
                LOGGER.info(
                    "Obs/Target/Error | right_arm obs=%s | target=%s | error=%s",
                    np.round(obs_right_arm, 4).tolist(),
                    np.round(tgt_right_arm, 4).tolist(),
                    np.round(err_right_arm, 4).tolist(),
                )
                LOGGER.info(
                    "Obs/Target/Error | LEFT_HAND  obs=%s | target=%s | error=%s",
                    np.round(obs_left_hand, 4).tolist(),
                    np.round(tgt_left_hand, 4).tolist(),
                    np.round(err_left_hand, 4).tolist(),
                )
                LOGGER.info(
                    "Obs/Target/Error | RIGHT_HAND obs=%s | target=%s | error=%s",
                    np.round(obs_right_hand, 4).tolist(),
                    np.round(tgt_right_hand, 4).tolist(),
                    np.round(err_right_hand, 4).tolist(),
                )

            # 记录本帧动作供下一帧平滑使用
            prev_action_urdf = action_urdf.copy()

            # --- Step 7: 可视化 ---
            if gui_enabled:
                should_quit, last_vis_ts = base.render_gui_frame(
                    args=args, rgb=top_rgb, depth_u16=depth_u16,
                    depth_vis_bgr=depth_vis_bgr,
                    last_vis_ts=last_vis_ts, save_dir=save_dir, frame_idx=frame_idx,
                )
                if should_quit:
                    break
            elif periodic_save_every > 0 and step % periodic_save_every == 0:
                # 无 GUI 模式：定期保存快照到磁盘供离线查看
                base.save_snapshot(
                    save_dir=save_dir, frame_idx=frame_idx,
                    rgb=top_rgb, depth_vis_bgr=depth_vis_bgr, depth_u16=depth_u16,
                )

            step += 1
            frame_idx += 1
            if args.max_steps > 0 and step >= args.max_steps:
                LOGGER.info("Reached max steps: %d", args.max_steps)
                break

            # --- 频率控制 ---
            # sleep 补齐到目标帧率（如 30fps → 每帧 33.3ms）
            elapsed = time.perf_counter() - loop_start
            if args.slow_loop_warn_ms > 0 and elapsed * 1000.0 > args.slow_loop_warn_ms:
                now_ts = time.time()
                if now_ts - slow_loop_last_log_ts > 2.0:
                    LOGGER.warning(
                        "Slow loop: %.1f ms (target %.1f ms). "
                        "Consider --no-gui, --save-every 0, or lower --fps.",
                        elapsed * 1000.0, 1000.0 / max(1e-6, args.fps),
                    )
                    slow_loop_last_log_ts = now_ts
            time.sleep(max(0.0, 1.0 / args.fps - elapsed))


    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")

    # ===== (C) 清理 =====
    # 顺序：先停抓帧线程 → 关相机 → 关 GUI → 断开机器人
    finally:
        for g in grabbers:
            g.stop()
        orbbec.close()
        for cam, name in [(cam_left, "left_wrist"), (cam_right, "right_wrist")]:
            if cam is not None:
                try:
                    cam.disconnect()
                except Exception as exc:
                    LOGGER.warning("Failed to disconnect %s: %s", name, exc)
        if gui_enabled:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass
        if client is not None:
            client.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """脚本入口：配置日志、注册信号处理、解析参数、启动部署。

    信号处理策略：
      - SIGINT (Ctrl+C) / SIGTERM (kill) → 转为 KeyboardInterrupt 触发优雅退出
      - SIGTSTP (Ctrl+Z) → 忽略，防止意外挂起导致机器人失控
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)

    def _handle_stop(signum, frame):
        del frame
        global _RECEIVED_STOP_SIGNAL
        _RECEIVED_STOP_SIGNAL = True
        LOGGER.warning("Received signal %s, exiting cleanly.", signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGTSTP, signal.SIG_IGN)

    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        if _RECEIVED_STOP_SIGNAL:
            LOGGER.info("Stopped by signal.")
        else:
            LOGGER.info("Interrupted by user.")


if __name__ == "__main__":
    main()
