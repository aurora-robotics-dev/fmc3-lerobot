#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deploy a Pi0 policy on Fourier GR2 using Aurora SDK + Orbbec RGB-D camera.

This script is tailored to the trained Pi0 checkpoint in this repo:
- Input: `observation.images.camera_top` + `observation.state` (45D)
- Output: `action` (35D)
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import signal
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.aurora_compat import patch_fourier_aurora_client_enums
from lerobot.utils.gr2_hand_conversion import hand_sdk_to_urdf, hand_urdf_to_sdk

LOGGER = logging.getLogger(__name__)
WINDOW_NAME = "GR2 PI0 Deploy | RGB + Depth (q/ESC quit, s save)"
_RECEIVED_STOP_SIGNAL = False
DEFAULT_CHECKPOINT_PATH = (
    "/home/phl/workspace/lerobot-versions/lerobot/outputs/train/"
    "pi0_gr2_pick_3_4_20260306_185911/checkpoints/070000/pretrained_model"
)
DEFAULT_TASK = "pick bottle"
DEFAULT_ROBOT_TYPE = "fourier_gr2"
DEFAULT_DEPLOY_FPS = 30.0

# GR2 observation order for 45D state vector.
STATE_GROUP_ORDER = [
    ("left_manipulator", "position"),
    ("right_manipulator", "position"),
    ("left_hand", "position"),
    ("right_hand", "position"),
    ("head", "position"),
    ("waist", "position"),
]

STATE_BASE_DATA_ORDER = [
    ("pos_W", 3),
    ("quat_xyzw", 4),
    ("rpy", 3),
    ("acc_B", 3),
    ("omega_B", 3),
]

# GR2 action layout (29 joints + optional base velocities).
ACTION_GROUP_SLICES = {
    "left_manipulator": slice(0, 7),
    "right_manipulator": slice(7, 14),
    "left_hand": slice(14, 20),
    "right_hand": slice(20, 26),
    "head": slice(26, 28),
    "waist": slice(28, 29),
}

GROUP_DIMS = {name: group_slice.stop - group_slice.start for name, group_slice in ACTION_GROUP_SLICES.items()}

# Keep clipping only on non-hand joints. Hand values stay in URDF space and are converted later.
JOINT_LIMITS = {
    "left_manipulator": [
        (-2.9671, 2.9671),
        (-0.5236, 2.7925),
        (-1.8326, 1.8326),
        (-1.5272, 0.47997),
        (-1.8326, 1.8326),
        (-0.61087, 0.61087),
        (-0.95993, 0.95993),
    ],
    "right_manipulator": [
        (-2.9671, 2.9671),
        (-2.7925, 0.5236),
        (-1.8326, 1.8326),
        (-1.5272, 0.47997),
        (-1.8326, 1.8326),
        (-0.61087, 0.61087),
        (-0.95993, 0.95993),
    ],
    "head": [(-1.3963, 1.3963), (-0.5236, 0.5236)],
    "waist": [(-2.618, 2.618)],
}

FSM_STATE_NAME_HINTS = {
    0: "Default",
    1: "JointStand",
    2: "PdStand",
    10: "UserCmd",
    11: "UpperBodyUserCmd",
}

LIFT_DIAGNOSTIC_JOINTS = (0, 1, 2, 3, 7, 8, 9, 10)
_MISSING_STATE_COUNTS: dict[str, int] = {}
_MISSING_STATE_LAST_LOG_TS = 0.0
_MISSING_STATE_LOG_INTERVAL_S = 2.0


def _fsm_state_matches(observed: str, expected: str) -> bool:
    if not observed:
        return False
    obs = str(observed)
    return obs == expected or obs.endswith(f".{expected}")


def check_opencv_gui_available() -> None:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("No DISPLAY/WAYLAND_DISPLAY found. Use `--no-gui`.")
    try:
        cv2.namedWindow("__cv2_gui_check__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__cv2_gui_check__")
    except Exception as exc:
        raise RuntimeError(
            "OpenCV GUI is unavailable (likely headless build). "
            "Use `--no-gui` or install a GUI-capable OpenCV."
        ) from exc


def depth_to_colormap(depth_u16: np.ndarray) -> np.ndarray:
    nonzero = depth_u16[depth_u16 > 0]
    if nonzero.size == 0:
        return np.zeros((*depth_u16.shape, 3), dtype=np.uint8)

    near = float(np.percentile(nonzero, 1))
    far = float(np.percentile(nonzero, 99))
    if far <= near:
        far = near + 1.0

    depth_clipped = np.clip(depth_u16.astype(np.float32), near, far)
    depth_8u = ((depth_clipped - near) / (far - near) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(255 - depth_8u, cv2.COLORMAP_JET)


class OrbbecRGBDCamera:
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        timeout_ms: int,
        warmup_frames: int,
        init_retries: int,
        retry_interval_s: float,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.timeout_ms = timeout_ms
        self.warmup_frames = warmup_frames
        self.init_retries = init_retries
        self.retry_interval_s = retry_interval_s
        self.ob = None
        self.pipeline = None

    def connect(self) -> None:
        self.ob = importlib.import_module("pyorbbecsdk")
        retries = max(1, self.init_retries)
        last_exc: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                self.pipeline = self._init_pipeline()
                for _ in range(max(0, self.warmup_frames)):
                    try:
                        self.pipeline.wait_for_frames(self.timeout_ms)
                    except Exception:
                        pass
                LOGGER.info("Orbbec stream is ready.")
                return
            except Exception as exc:
                last_exc = exc
                LOGGER.warning("Orbbec init attempt %d/%d failed: %s", attempt, retries, exc)
                self.close()
                if attempt < retries:
                    time.sleep(max(0.0, self.retry_interval_s))

        hint = (
            "Camera may be busy (for example another process was suspended with Ctrl+Z). "
            "Run `jobs -l`, then `fg %<job>` and press Ctrl+C, or kill that PID."
        )
        raise RuntimeError(f"Failed to initialize Orbbec after {retries} attempts. {hint}") from last_exc

    def close(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if self.pipeline is None:
            raise RuntimeError("Orbbec pipeline is not started.")

        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None

        color_rgb = self._decode_color_to_rgb(color_frame)
        depth_u16 = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            depth_frame.get_height(), depth_frame.get_width()
        )

        if color_rgb.shape[:2] != (self.height, self.width):
            color_rgb = cv2.resize(color_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if depth_u16.shape[:2] != (self.height, self.width):
            depth_u16 = cv2.resize(depth_u16, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        depth_vis_bgr = depth_to_colormap(depth_u16)
        depth_rgb = self._depth_to_rgb(depth_u16)
        return color_rgb, depth_rgb, depth_u16, depth_vis_bgr

    def _init_pipeline(self):
        if self.ob is None:
            raise RuntimeError("`pyorbbecsdk` is not imported.")

        pipeline = self.ob.Pipeline()
        config = self.ob.Config()

        candidates: list[tuple[object, object]] = []
        sensor_enum = getattr(self.ob, "OBSensorType", None)
        if sensor_enum is not None:
            candidates.append(
                (
                    getattr(sensor_enum, "DEPTH_SENSOR", None),
                    getattr(sensor_enum, "COLOR_SENSOR", None),
                )
            )
        stream_enum = getattr(self.ob, "StreamType", None)
        if stream_enum is not None:
            candidates.append((getattr(stream_enum, "DEPTH", None), getattr(stream_enum, "COLOR", None)))
        ob_stream_enum = getattr(self.ob, "OBStreamType", None)
        if ob_stream_enum is not None:
            candidates.append(
                (
                    getattr(ob_stream_enum, "DEPTH_STREAM", None),
                    getattr(ob_stream_enum, "COLOR_STREAM", None),
                )
            )

        depth_profiles = color_profiles = None
        errors: list[str] = []
        for depth_key, color_key in candidates:
            if depth_key is None or color_key is None:
                continue
            try:
                depth_profiles = pipeline.get_stream_profile_list(depth_key)
                color_profiles = pipeline.get_stream_profile_list(color_key)
                break
            except Exception as exc:
                errors.append(str(exc))

        if depth_profiles is None or color_profiles is None:
            err_msg = "; ".join(errors) if errors else "unknown error"
            raise RuntimeError(f"Cannot query Orbbec stream profiles: {err_msg}")

        color_profile = self._select_video_profile(
            color_profiles,
            target_width=self.width,
            target_height=self.height,
            target_fps=self.fps,
            prefer_formats={"OBFormat.MJPG", "OBFormat.YUYV", "OBFormat.RGB", "OBFormat.BGR"},
        )
        depth_profile = self._select_video_profile(
            depth_profiles,
            target_width=self.width,
            target_height=self.height,
            target_fps=self.fps,
            prefer_formats={"OBFormat.Y16"},
        )
        config.enable_stream(depth_profile)
        config.enable_stream(color_profile)
        pipeline.start(config)
        LOGGER.info("Orbbec color profile: %s", color_profile)
        LOGGER.info("Orbbec depth profile: %s", depth_profile)
        return pipeline

    @staticmethod
    def _get_default_profile(profile_list):
        if hasattr(profile_list, "get_default_video_stream_profile"):
            return profile_list.get_default_video_stream_profile()
        if hasattr(profile_list, "get_default_profile"):
            return profile_list.get_default_profile()
        raise RuntimeError("No default stream profile method available.")

    @staticmethod
    def _iter_video_profiles(profile_list):
        count = profile_list.get_count() if hasattr(profile_list, "get_count") else 0
        for i in range(int(count)):
            try:
                profile = profile_list.get_stream_profile_by_index(i)
            except Exception:
                continue
            if hasattr(profile, "as_video_stream_profile"):
                try:
                    profile = profile.as_video_stream_profile()
                except Exception:
                    pass
            yield profile

    def _select_video_profile(
        self,
        profile_list,
        target_width: int,
        target_height: int,
        target_fps: int,
        prefer_formats: set[str] | None = None,
    ):
        best = None
        best_score = None
        prefer_formats = prefer_formats or set()

        for profile in self._iter_video_profiles(profile_list):
            try:
                width = int(profile.get_width())
                height = int(profile.get_height())
                fps = int(profile.get_fps())
                fmt = str(profile.get_format())
            except Exception:
                continue

            exact_res = int(width == target_width and height == target_height)
            exact_fps = int(fps == target_fps)
            fmt_pref = int(fmt in prefer_formats)
            fps_penalty = abs(fps - target_fps)
            res_penalty = abs(width - target_width) + abs(height - target_height)
            score = (-exact_res, -exact_fps, -fmt_pref, fps_penalty, res_penalty)
            if best_score is None or score < best_score:
                best = profile
                best_score = score

        if best is not None:
            return best
        return self._get_default_profile(profile_list)

    @staticmethod
    def _decode_color_to_rgb(color_frame) -> np.ndarray:
        h, w = color_frame.get_height(), color_frame.get_width()
        raw = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        pixels = h * w

        if raw.size == pixels * 3:
            return raw.reshape(h, w, 3)
        if raw.size == pixels * 2:
            yuyv = raw.reshape(h, w, 2)
            bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if raw.size == pixels * 4:
            bgra = raw.reshape(h, w, 4)
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)

        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.zeros((h, w, 3), dtype=np.uint8)

    @staticmethod
    def _depth_to_rgb(depth_u16: np.ndarray) -> np.ndarray:
        max_depth = float(depth_u16.max())
        if max_depth <= 0:
            depth_8u = np.zeros_like(depth_u16, dtype=np.uint8)
        else:
            depth_8u = np.clip(depth_u16.astype(np.float32) / max_depth * 255.0, 0, 255).astype(np.uint8)
        return np.repeat(depth_8u[..., None], 3, axis=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy Pi0 checkpoint on GR2 (Aurora SDK + Orbbec RGB-D).")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint root or pretrained_model directory.",
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task string passed to inference.")
    parser.add_argument("--robot-type", type=str, default=DEFAULT_ROBOT_TYPE)
    parser.add_argument("--domain-id", type=int, default=123)
    parser.add_argument("--robot-name", type=str, default="gr2")
    parser.add_argument(
        "--client-init-retries",
        type=int,
        default=4,
        help="AuroraClient initialization retries before failing.",
    )
    parser.add_argument(
        "--client-retry-interval-s",
        type=float,
        default=2.0,
        help="Seconds to wait between AuroraClient initialization retries.",
    )
    parser.add_argument("--fsm-state", type=int, default=11, help="GR2 UpperBodyUserCmd state is typically 11.")
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_DEPLOY_FPS,
        help="Control loop target frequency. Keep this aligned with dataset fps (default 30).",
    )
    parser.add_argument("--transition-time-s", type=float, default=6.0, help="First-action smooth alignment time.")
    parser.add_argument("--transition-freq", type=int, default=100, help="Smooth alignment interpolation frequency.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--send-base", action="store_true", help="Also send base velocity action dims 29:34.")
    parser.add_argument("--disable-clamp", action="store_true", help="Disable non-hand joint clipping.")
    parser.add_argument(
        "--action-ema-alpha",
        type=float,
        default=0.35,
        help="EMA blend factor for action smoothing in [0,1]. 1.0 disables EMA blending.",
    )
    parser.add_argument(
        "--arm-ema-alpha",
        type=float,
        default=-1.0,
        help="EMA alpha for arm joints. Negative means using --action-ema-alpha.",
    )
    parser.add_argument(
        "--hand-ema-alpha",
        type=float,
        default=-1.0,
        help="EMA alpha for hand joints. Negative means using --action-ema-alpha.",
    )
    parser.add_argument(
        "--max-arm-delta",
        type=float,
        default=0.06,
        help="Max per-step delta (rad) for manipulator joints. <=0 disables limiting.",
    )
    parser.add_argument(
        "--max-hand-delta",
        type=float,
        default=0.12,
        help="Max per-step delta (rad) for dexterous hand joints. <=0 disables limiting.",
    )
    parser.add_argument(
        "--max-head-waist-delta",
        type=float,
        default=0.08,
        help="Max per-step delta (rad) for head/waist joints. <=0 disables limiting.",
    )
    parser.add_argument(
        "--max-base-delta",
        type=float,
        default=0.15,
        help="Max per-step delta for base action dims when --send-base is enabled. <=0 disables limiting.",
    )
    parser.add_argument(
        "--slow-loop-warn-ms",
        type=float,
        default=120.0,
        help="Warn when one control iteration exceeds this latency in ms. <=0 disables warning.",
    )
    parser.add_argument("--skip-confirm", action="store_true", help="Skip interactive safety confirmation.")
    parser.add_argument("--dry-run", action="store_true", help="Run model inference without commanding the robot.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--camera-key",
        type=str,
        default="observation.images.camera_top",
        help="Visual key expected by Pi0 policy input.",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30, help="Preferred camera stream FPS.")
    parser.add_argument("--camera-timeout-ms", type=int, default=200)
    parser.add_argument("--camera-warmup-frames", type=int, default=15)
    parser.add_argument("--camera-init-retries", type=int, default=6)
    parser.add_argument("--camera-retry-interval-s", type=float, default=1.0)
    parser.add_argument("--state-dim", type=int, default=0, help="Override state dim (0 means use policy config).")
    parser.add_argument("--action-dim", type=int, default=0, help="Override action dim (0 means use policy config).")
    parser.add_argument("--no-gui", action="store_true", help="Disable RGB+depth visualization window.")
    parser.add_argument("--save-dir", type=str, default="scripts/outputs/deploy_gr2_pi0")
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save RGB/depth snapshots every N steps (works in both GUI and headless). 0 disables periodic save.",
    )
    return parser.parse_args()


def resolve_pretrained_model_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    direct_model = path / "model.safetensors"
    nested_model = path / "pretrained_model" / "model.safetensors"
    if direct_model.exists():
        return path
    if nested_model.exists():
        return path / "pretrained_model"
    raise FileNotFoundError(
        "Cannot locate `model.safetensors`. Use a path containing `model.safetensors` "
        f"or a checkpoint root containing `pretrained_model/`. Got: {path}"
    )


def load_train_config(pretrained_dir: Path) -> dict[str, Any] | None:
    cfg_path = pretrained_dir / "train_config.json"
    if not cfg_path.exists():
        return None
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        LOGGER.warning("Failed to read train_config.json: %s", exc)
        return None


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("`--device cuda` was set, but CUDA is unavailable.")
    return torch.device(device_arg)


def _feature_type_name(feature: Any) -> str:
    feature_type = getattr(feature, "type", None)
    if hasattr(feature_type, "value"):
        return str(feature_type.value)
    return str(feature_type) if feature_type is not None else ""


def _feature_shape(feature: Any) -> list[int]:
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape", [])
    if shape is None:
        return []
    return [int(x) for x in shape]


def infer_policy_io(policy: PI0Policy) -> tuple[str, int, list[str], str, int]:
    input_features = policy.config.input_features
    output_features = policy.config.output_features

    state_key = "observation.state"
    state_dim = 45
    visual_keys: list[str] = []

    for key, feature in input_features.items():
        type_name = _feature_type_name(feature).upper()
        if "STATE" in type_name:
            state_key = key
            state_shape = _feature_shape(feature)
            if state_shape:
                state_dim = int(state_shape[0])
        if "VISUAL" in type_name:
            visual_keys.append(key)

    if not visual_keys:
        visual_keys = [k for k in input_features if "observation.images" in k]
    if not visual_keys:
        raise RuntimeError("No visual input key found in policy config.")

    action_key = "action"
    action_dim = 35
    for key, feature in output_features.items():
        action_key = key
        out_shape = _feature_shape(feature)
        if out_shape:
            action_dim = int(out_shape[0])
        break

    return state_key, state_dim, visual_keys, action_key, action_dim


def fit_vector(x: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.shape[0] == dim:
        return arr
    if arr.shape[0] > dim:
        return arr[:dim]
    out = np.zeros((dim,), dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out


def clamp_non_hand_joint_action(action_urdf: np.ndarray) -> np.ndarray:
    action = action_urdf.copy()
    for group_name, limits in JOINT_LIMITS.items():
        group_slice = ACTION_GROUP_SLICES[group_name]
        group_values = action[group_slice]
        for i, (lower, upper) in enumerate(limits):
            group_values[i] = float(np.clip(group_values[i], lower, upper))
        action[group_slice] = group_values
    return action


def _clip_delta_inplace(
    out: np.ndarray,
    prev: np.ndarray,
    group_slice: slice,
    max_delta: float,
) -> None:
    if max_delta <= 0:
        return
    delta = out[group_slice] - prev[group_slice]
    out[group_slice] = prev[group_slice] + np.clip(delta, -max_delta, max_delta)


def stabilize_action(action_urdf: np.ndarray, prev_action_urdf: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Apply action smoothing and per-step rate limiting in URDF space."""
    if action_urdf.shape != prev_action_urdf.shape:
        return action_urdf

    out = action_urdf.copy()

    default_alpha = float(np.clip(args.action_ema_alpha, 0.0, 1.0))
    arm_alpha = default_alpha if float(args.arm_ema_alpha) < 0 else float(np.clip(args.arm_ema_alpha, 0.0, 1.0))
    hand_alpha = default_alpha if float(args.hand_ema_alpha) < 0 else float(np.clip(args.hand_ema_alpha, 0.0, 1.0))

    if arm_alpha < 1.0:
        out[0:14] = prev_action_urdf[0:14] * (1.0 - arm_alpha) + out[0:14] * arm_alpha
    if hand_alpha < 1.0:
        out[14:26] = prev_action_urdf[14:26] * (1.0 - hand_alpha) + out[14:26] * hand_alpha
    if default_alpha < 1.0:
        out[26:29] = prev_action_urdf[26:29] * (1.0 - default_alpha) + out[26:29] * default_alpha
        if args.send_base:
            out[29:35] = prev_action_urdf[29:35] * (1.0 - default_alpha) + out[29:35] * default_alpha

    _clip_delta_inplace(out, prev_action_urdf, slice(0, 14), float(args.max_arm_delta))
    _clip_delta_inplace(out, prev_action_urdf, slice(14, 26), float(args.max_hand_delta))
    _clip_delta_inplace(out, prev_action_urdf, slice(26, 29), float(args.max_head_waist_delta))
    if args.send_base:
        _clip_delta_inplace(out, prev_action_urdf, slice(29, 35), float(args.max_base_delta))
    return out.astype(np.float32)


def _note_missing_state(source: str, dim: int) -> None:
    global _MISSING_STATE_LAST_LOG_TS
    _MISSING_STATE_COUNTS[source] = _MISSING_STATE_COUNTS.get(source, 0) + 1
    now = time.time()
    if now - _MISSING_STATE_LAST_LOG_TS >= _MISSING_STATE_LOG_INTERVAL_S:
        details = ", ".join(f"{key}:{count}" for key, count in sorted(_MISSING_STATE_COUNTS.items()))
        LOGGER.warning("Missing robot state fallback to zeros (dim=%d). counts=%s", dim, details)
        _MISSING_STATE_LAST_LOG_TS = now


def _safe_array(values: list[float] | None, dim: int, source: str) -> np.ndarray:
    if values is None:
        _note_missing_state(source, dim)
        return np.zeros(dim, dtype=np.float32)
    return fit_vector(np.asarray(values, dtype=np.float32), dim)


def get_robot_state_urdf(client: Any) -> np.ndarray:
    parts = []
    for group_name, key in STATE_GROUP_ORDER:
        pos_array = _safe_array(
            client.get_group_state(group_name, key),
            GROUP_DIMS[group_name],
            source=f"{group_name}.{key}",
        )
        if "hand" in group_name and pos_array.shape[0] == 6:
            pos_array = hand_sdk_to_urdf(pos_array)
        parts.append(pos_array)

    for data_key, dim in STATE_BASE_DATA_ORDER:
        parts.append(_safe_array(client.get_base_data(data_key), dim, source=f"base.{data_key}"))

    return np.concatenate(parts).astype(np.float32)


def _compute_arm_limit_stats(
    raw_action_urdf: np.ndarray,
    prev_action_urdf: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[float, float]:
    if prev_action_urdf is None:
        return 0.0, 0.0

    max_delta = float(args.max_arm_delta)
    if max_delta <= 0:
        return 0.0, 0.0

    alpha = (
        float(np.clip(args.action_ema_alpha, 0.0, 1.0))
        if float(args.arm_ema_alpha) < 0
        else float(np.clip(args.arm_ema_alpha, 0.0, 1.0))
    )
    raw_arm = raw_action_urdf[:14]
    prev_arm = prev_action_urdf[:14]
    ema_arm = prev_arm * (1.0 - alpha) + raw_arm * alpha if alpha < 1.0 else raw_arm
    delta = ema_arm - prev_arm

    hit_mask = np.abs(delta) > max_delta
    hit_ratio = float(hit_mask.mean())
    if hit_mask.any():
        max_cut = float(np.max(np.abs(delta[hit_mask]) - max_delta))
    else:
        max_cut = 0.0
    return hit_ratio, max_cut


def _format_key_joint_transitions(raw_action_urdf: np.ndarray, final_action_urdf: np.ndarray) -> str:
    parts = []
    for idx in LIFT_DIAGNOSTIC_JOINTS:
        raw_v = float(raw_action_urdf[idx])
        final_v = float(final_action_urdf[idx])
        parts.append(f"{idx}:{raw_v:.3f}->{final_v:.3f}({final_v - raw_v:+.3f})")
    return " ".join(parts)


def log_action_diagnostics(
    step: int,
    arm_delta: float,
    depth_range: str,
    raw_action_urdf: np.ndarray,
    final_action_urdf: np.ndarray,
    prev_action_urdf: np.ndarray | None,
    args: argparse.Namespace,
) -> None:
    arm_raw = raw_action_urdf[:14]
    arm_final = final_action_urdf[:14]
    arm_raw_norm = float(np.linalg.norm(arm_raw))
    compress_ratio = float(np.linalg.norm(arm_raw - arm_final) / max(arm_raw_norm, 1e-6))
    hit_ratio, max_cut = _compute_arm_limit_stats(raw_action_urdf, prev_action_urdf, args)
    LOGGER.info(
        "step=%d arm_delta=%.4f depth(mm)=%s raw_head=%s final_head=%s arm_compress=%.4f arm_limit_hit=%.2f "
        "arm_limit_max_cut=%.4f lift(raw->final)=%s",
        step,
        arm_delta,
        depth_range,
        np.round(raw_action_urdf[:8], 4).tolist(),
        np.round(final_action_urdf[:8], 4).tolist(),
        compress_ratio,
        hit_ratio,
        max_cut,
        _format_key_joint_transitions(raw_action_urdf, final_action_urdf),
    )


def _try_send_base_velocity(client: Any, action_urdf: np.ndarray) -> None:
    if action_urdf.shape[0] < 32:
        return
    vel_x, vel_y, vel_yaw = map(float, action_urdf[29:32])
    try:
        height = float(action_urdf[32]) if action_urdf.shape[0] > 32 else 0.0
        pitch = float(action_urdf[33]) if action_urdf.shape[0] > 33 else 0.0
        client.set_velocity(vel_x, vel_y, vel_yaw, height, pitch)
        return
    except TypeError:
        pass
    client.set_velocity(vel_x, vel_y, vel_yaw)


def send_action_to_robot(client: Any, action_urdf: np.ndarray, send_base: bool) -> None:
    action_send = action_urdf.copy()
    action_send[14:20] = hand_urdf_to_sdk(action_send[14:20], clip=True)
    action_send[20:26] = hand_urdf_to_sdk(action_send[20:26], clip=True)

    position_dict = {
        group_name: action_send[group_slice].tolist() for group_name, group_slice in ACTION_GROUP_SLICES.items()
    }
    client.set_joint_positions(position_dict)

    if send_base:
        _try_send_base_velocity(client, action_urdf)


def smooth_transition(client: Any, target_action_urdf: np.ndarray, duration_s: float, frequency_hz: int) -> None:
    if duration_s <= 0:
        send_action_to_robot(client, target_action_urdf, send_base=False)
        return

    init_joints = get_robot_state_urdf(client)[:29]
    target_joints = target_action_urdf[:29]
    total_steps = max(1, int(duration_s * frequency_hz))
    LOGGER.info("Running smooth transition for %.2fs (%d steps).", duration_s, total_steps)

    for step in range(total_steps + 1):
        alpha = step / total_steps
        interp_action = init_joints * (1.0 - alpha) + target_joints * alpha
        send_action_to_robot(client, interp_action, send_base=False)
        time.sleep(1.0 / frequency_hz)


def load_robot_client(domain_id: int, robot_name: str, retries: int, retry_interval_s: float) -> Any:
    patch_fourier_aurora_client_enums()
    from fourier_aurora_client import AuroraClient

    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            client = AuroraClient.get_instance(domain_id=domain_id, robot_name=robot_name)
            if client is None:
                raise RuntimeError("AuroraClient.get_instance returned None.")
            if attempt > 1:
                LOGGER.info("AuroraClient connected on retry %d/%d.", attempt, retries)
            return client
        except Exception as exc:
            last_exc = exc
            LOGGER.warning(
                "AuroraClient init failed (%d/%d): %s",
                attempt,
                max(1, retries),
                exc,
            )
            if attempt < max(1, retries):
                time.sleep(max(0.0, retry_interval_s))
    raise RuntimeError("Failed to create AuroraClient instance.") from last_exc


def setup_gui(args: argparse.Namespace) -> bool:
    gui_enabled = not args.no_gui
    if not gui_enabled:
        return False
    try:
        check_opencv_gui_available()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, args.camera_width * 2, args.camera_height)
        return True
    except Exception as exc:
        LOGGER.warning("Disable GUI because it is unavailable: %s", exc)
        return False


def setup_robot_if_needed(args: argparse.Namespace) -> Any | None:
    if args.dry_run:
        return None
    client = load_robot_client(
        args.domain_id,
        args.robot_name,
        retries=args.client_init_retries,
        retry_interval_s=args.client_retry_interval_s,
    )
    if not args.skip_confirm:
        LOGGER.info("Waiting for manual confirmation. Press Enter in terminal to continue.")
        input("确认机器人和周围环境安全后按回车，脚本将切换 FSM 并开始推理...")
    expected_state_name = FSM_STATE_NAME_HINTS.get(int(args.fsm_state))
    fsm_state = "Unknown"
    upper_fsm_state = "Unknown"
    for attempt in range(1, 7):
        client.set_fsm_state(args.fsm_state)
        time.sleep(0.5)
        try:
            fsm_state = client.get_fsm_state()
            upper_fsm_state = client.get_upper_fsm_state()
        except Exception as exc:
            LOGGER.warning("Failed to query Aurora FSM after switch attempt %d: %s", attempt, exc)
            continue
        if expected_state_name is None or _fsm_state_matches(fsm_state, expected_state_name):
            break

    try:
        LOGGER.info("Aurora FSM after switch: whole=%s upper=%s", fsm_state, upper_fsm_state)
        if expected_state_name is not None and not _fsm_state_matches(fsm_state, expected_state_name):
            raise RuntimeError(
                "FSM switch failed: expected "
                f"{expected_state_name}(id={args.fsm_state}) but got {fsm_state}. "
                "请先确认手柄/急停/控制权限状态，再重试；上身控制可优先尝试 --fsm-state 11。"
            )
        if int(args.fsm_state) == 2:
            stand_pose = client.get_stand_pose()
            if isinstance(stand_pose, (list, tuple)) and len(stand_pose) >= 4:
                dz = float(stand_pose[0])
                dp = float(stand_pose[1])
                dy = float(stand_pose[2])
                stable_level = int(stand_pose[3])
                LOGGER.info(
                    "PdStand pose: delta_z=%.4f delta_pitch=%.4f delta_yaw=%.4f stable_level=%d",
                    dz,
                    dp,
                    dy,
                    stable_level,
                )
                if not all(math.isfinite(v) for v in [dz, dp, dy]):
                    LOGGER.warning("PdStand pose contains non-finite values, DDS state stream may be abnormal.")
    except Exception as exc:
        LOGGER.warning("Failed to query Aurora FSM/stand state after switch: %s", exc)
        raise
    return client


def warmup_policy(
    policy: PI0Policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
    robot_type: str,
    camera_key: str,
    state_key: str,
    camera_height: int,
    camera_width: int,
    state_dim: int,
) -> None:
    dummy_obs = {
        camera_key: np.zeros((camera_height, camera_width, 3), dtype=np.uint8),
        state_key: np.zeros((state_dim,), dtype=np.float32),
    }
    with torch.inference_mode():
        dummy_batch = prepare_observation_for_inference(
            observation=dummy_obs,
            device=device,
            task=task,
            robot_type=robot_type,
        )
        _ = postprocessor(policy.select_action(preprocessor(dummy_batch)))


def infer_single_action(
    policy: PI0Policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
    robot_type: str,
    camera_key: str,
    state_key: str,
    rgb: np.ndarray,
    state_model: np.ndarray,
    action_dim: int,
) -> np.ndarray:
    observation = {camera_key: rgb, state_key: state_model}
    with torch.inference_mode():
        batch = prepare_observation_for_inference(
            observation=observation,
            device=device,
            task=task,
            robot_type=robot_type,
        )
        action_tensor = postprocessor(policy.select_action(preprocessor(batch)))
    raw_action = action_tensor.reshape(-1, action_tensor.shape[-1])[0].detach().cpu().numpy().astype(np.float32)
    return fit_vector(raw_action, action_dim)


def render_gui_frame(
    args: argparse.Namespace,
    rgb: np.ndarray,
    depth_u16: np.ndarray,
    depth_vis_bgr: np.ndarray,
    last_vis_ts: float,
    save_dir: Path,
    frame_idx: int,
) -> tuple[bool, float]:
    panel = np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth_vis_bgr])
    now = time.time()
    vis_fps = 1.0 / max(1e-6, now - last_vis_ts)
    valid_depth = depth_u16[depth_u16 > 0]
    depth_text = (
        "Depth valid range: empty"
        if valid_depth.size == 0
        else f"Depth valid range: {int(valid_depth.min())}..{int(valid_depth.max())} mm"
    )
    cv2.putText(panel, "RGB", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(
        panel,
        "Depth (colormap)",
        (args.camera_width + 10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        panel,
        f"FPS: {vis_fps:.1f}",
        (10, args.camera_height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        panel,
        depth_text,
        (args.camera_width + 10, args.camera_height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.imshow(WINDOW_NAME, panel)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord("q")):
        LOGGER.info("Quit requested from visualization window.")
        return True, now
    if key == ord("s"):
        save_snapshot(save_dir=save_dir, frame_idx=frame_idx, rgb=rgb, depth_vis_bgr=depth_vis_bgr, depth_u16=depth_u16)
    return False, now


def save_snapshot(
    *,
    save_dir: Path,
    frame_idx: int,
    rgb: np.ndarray,
    depth_vis_bgr: np.ndarray,
    depth_u16: np.ndarray,
) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    rgb_path = save_dir / f"rgb_{ts}_{frame_idx:06d}.png"
    depth_vis_path = save_dir / f"depth_vis_{ts}_{frame_idx:06d}.png"
    depth_raw_path = save_dir / f"depth_raw_{ts}_{frame_idx:06d}.npy"
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(depth_vis_path), depth_vis_bgr)
    np.save(str(depth_raw_path), depth_u16)
    LOGGER.info("Saved snapshot: %s | %s | %s", rgb_path, depth_vis_path, depth_raw_path)


def run(args: argparse.Namespace) -> None:
    pretrained_dir = resolve_pretrained_model_dir(args.checkpoint_path)
    train_cfg = load_train_config(pretrained_dir)
    device = select_device(args.device)

    LOGGER.info("Model: %s", pretrained_dir)
    LOGGER.info("Device: %s", device)
    if train_cfg:
        train_ds = train_cfg.get("dataset", {})
        LOGGER.info("Train dataset from config: repo_id=%s root=%s", train_ds.get("repo_id"), train_ds.get("root"))
    LOGGER.info("Camera target profile: %dx%d@%d", args.camera_width, args.camera_height, args.camera_fps)

    policy = PI0Policy.from_pretrained(str(pretrained_dir), strict=False).to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    state_key, cfg_state_dim, visual_keys, action_key, cfg_action_dim = infer_policy_io(policy)
    state_dim = args.state_dim if args.state_dim > 0 else cfg_state_dim
    action_dim = args.action_dim if args.action_dim > 0 else cfg_action_dim
    camera_key = args.camera_key if args.camera_key in visual_keys else visual_keys[0]

    LOGGER.info(
        "Policy IO: camera_key=%s state_key=%s state_dim=%d action_key=%s action_dim=%d visual_keys=%s",
        camera_key,
        state_key,
        state_dim,
        action_key,
        action_dim,
        visual_keys,
    )

    gui_enabled = setup_gui(args)

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    last_vis_ts = time.time()
    frame_idx = 0
    periodic_save_every = max(0, int(args.save_every))
    if not gui_enabled and not args.no_gui and periodic_save_every <= 0:
        # GUI is requested but unavailable in this environment; keep observability by saving frames periodically.
        periodic_save_every = 30
        LOGGER.info(
            "GUI is unavailable in this environment. Fallback: save RGB/depth snapshots every %d steps to %s",
            periodic_save_every,
            save_dir,
        )

    camera = OrbbecRGBDCamera(
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        timeout_ms=args.camera_timeout_ms,
        warmup_frames=args.camera_warmup_frames,
        init_retries=args.camera_init_retries,
        retry_interval_s=args.camera_retry_interval_s,
    )
    camera.connect()

    client = setup_robot_if_needed(args)

    LOGGER.info("Warming up policy...")
    warmup_policy(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        task=args.task,
        robot_type=args.robot_type,
        camera_key=camera_key,
        state_key=state_key,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        state_dim=state_dim,
    )
    LOGGER.info("Warm-up complete.")

    step = 0
    first_action = True
    policy.reset()
    prev_action_urdf: np.ndarray | None = None
    no_frame_count = 0
    no_frame_last_log_ts = time.time()
    slow_loop_last_log_ts = time.time()

    try:
        while True:
            loop_start = time.perf_counter()
            try:
                frame = camera.read()
            except Exception as exc:
                LOGGER.warning("Camera read failed: %s. Reconnecting...", exc)
                camera.close()
                camera.connect()
                continue

            if frame is None:
                no_frame_count += 1
                now_ts = time.time()
                if now_ts - no_frame_last_log_ts > 2.0:
                    LOGGER.warning(
                        "No camera frame yet (count=%d). Check camera stream or timeout setting.",
                        no_frame_count,
                    )
                    no_frame_last_log_ts = now_ts
                continue

            no_frame_count = 0
            rgb, _, depth_u16, depth_vis_bgr = frame

            if args.dry_run:
                state_full = np.zeros((45,), dtype=np.float32)
            else:
                if client is None:
                    raise RuntimeError("Robot client is not initialized.")
                state_full = get_robot_state_urdf(client)
            state_model = fit_vector(state_full, state_dim)

            action_model = infer_single_action(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                task=args.task,
                robot_type=args.robot_type,
                camera_key=camera_key,
                state_key=state_key,
                rgb=rgb,
                state_model=state_model,
                action_dim=action_dim,
            )
            raw_action_urdf = fit_vector(action_model, 35)
            prev_action_for_diag = prev_action_urdf
            action_urdf = raw_action_urdf.copy()
            if not args.disable_clamp:
                action_urdf = clamp_non_hand_joint_action(action_urdf)
            if prev_action_urdf is not None:
                action_urdf = stabilize_action(action_urdf, prev_action_urdf, args)

            if not args.dry_run:
                if client is None:
                    raise RuntimeError("Robot client is not initialized.")
                if first_action and args.transition_time_s > 0:
                    smooth_transition(client, action_urdf, args.transition_time_s, args.transition_freq)
                    policy.reset()
                    first_action = False
                else:
                    send_action_to_robot(client, action_urdf, args.send_base)
                    first_action = False

            if step % max(1, args.log_every) == 0:
                arm_dim = min(14, state_full.shape[0], action_urdf.shape[0])
                arm_delta = float(np.linalg.norm(action_urdf[:arm_dim] - state_full[:arm_dim]))
                valid_depth = depth_u16[depth_u16 > 0]
                depth_range = "empty" if valid_depth.size == 0 else f"{int(valid_depth.min())}..{int(valid_depth.max())}"
                log_action_diagnostics(
                    step,
                    arm_delta,
                    depth_range,
                    raw_action_urdf,
                    action_urdf,
                    prev_action_for_diag,
                    args,
                )

            prev_action_urdf = action_urdf.copy()

            if gui_enabled:
                should_quit, last_vis_ts = render_gui_frame(
                    args=args,
                    rgb=rgb,
                    depth_u16=depth_u16,
                    depth_vis_bgr=depth_vis_bgr,
                    last_vis_ts=last_vis_ts,
                    save_dir=save_dir,
                    frame_idx=frame_idx,
                )
                if should_quit:
                    break
            elif periodic_save_every > 0 and step % periodic_save_every == 0:
                save_snapshot(
                    save_dir=save_dir,
                    frame_idx=frame_idx,
                    rgb=rgb,
                    depth_vis_bgr=depth_vis_bgr,
                    depth_u16=depth_u16,
                )

            step += 1
            frame_idx += 1
            if args.max_steps > 0 and step >= args.max_steps:
                LOGGER.info("Reached max steps: %d", args.max_steps)
                break

            elapsed = time.perf_counter() - loop_start
            if args.slow_loop_warn_ms > 0 and elapsed * 1000.0 > args.slow_loop_warn_ms:
                now_ts = time.time()
                if now_ts - slow_loop_last_log_ts > 2.0:
                    LOGGER.warning(
                        "Slow control loop: %.1f ms (target %.1f ms). "
                        "Consider `--no-gui`, `--save-every 0`, or lowering `--fps`.",
                        elapsed * 1000.0,
                        1000.0 / max(1e-6, args.fps),
                    )
                    slow_loop_last_log_ts = now_ts
            time.sleep(max(0.0, 1.0 / args.fps - elapsed))
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        camera.close()
        if gui_enabled:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass
        if client is not None:
            client.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)

    def _handle_stop(signum, frame):
        del frame
        global _RECEIVED_STOP_SIGNAL
        _RECEIVED_STOP_SIGNAL = True
        LOGGER.warning("Received signal %s, exiting cleanly.", signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGTSTP, signal.SIG_IGN)  # Avoid suspended jobs holding camera resources.

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
