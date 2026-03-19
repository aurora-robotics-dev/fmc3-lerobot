#!/usr/bin/env python3

"""Deploy a Pi0 policy on Fourier GR2 with RGB-only multi-camera input.

This variant is derived from `deploy_gr2_pi0_rgbd_wrist.py`, but it removes all
depth observations and builds policy inputs from three RGB streams only:
  - observation.images.camera_top
  - observation.images.camera_left_wrist
  - observation.images.camera_right_wrist

The default dataset root matches the RGB-only dataset prepared in this repo:
`fmc3_gr2_grab_bottle_from_box_to_desk_lerobot_ds_rgb`.
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
WINDOW_NAME = "GR2 PI0 Deploy | RGB Wrist (q/ESC quit, s save)"
_RECEIVED_STOP_SIGNAL = False

DEFAULT_CHECKPOINT_PATH = (
    "/home/phl/workspace/mymodels/gr2/"
    "pi0_gr2_grab_bottle_from_box_to_desk_rgb/checkpoints/050000/pretrained_model"
)
DEFAULT_DATASET_ROOT = (
    "/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/"
    "fmc3_gr2_grab_bottle_from_box_to_desk_lerobot_ds_rgb"
)
DEFAULT_LEFT_WRIST_SERIAL = "420222072816"
DEFAULT_RIGHT_WRIST_SERIAL = "349522072801"

# PD gains for GR2 joint position control.
PD_KP_CONFIG = {
    "left_manipulator": [300, 300, 100, 100, 50, 50, 50],
    "right_manipulator": [270, 250, 95, 95, 45, 45, 45],
    "waist": [200],
    "head": [100, 100],
}
PD_KD_CONFIG = {
    "left_manipulator": [10, 10, 5, 5, 5, 5, 5],
    "right_manipulator": [10, 10, 5, 5, 5, 5, 5],
    "waist": [10],
    "head": [10, 10],
}

# First-frame joint pose extracted from the RGB dataset's first observation.state.
# Layout: 29 joint dims + 6 zero base-action dims.
INIT_POSE_URDF = np.array(
    [
        0.04899011,
        0.13595560,
        -1.76930320,
        0.13200505,
        0.47436661,
        -0.45774499,
        0.93427992,
        -0.27377611,
        -0.10182220,
        0.11617941,
        -1.37683356,
        -0.13093603,
        0.30303562,
        -0.32032081,
        -0.00458619,
        -0.00671026,
        -0.01201849,
        -0.01197735,
        0.00885655,
        -1.56070650,
        -0.00806141,
        -0.00798318,
        -0.01938905,
        -0.01500668,
        0.00638821,
        -1.57624388,
        0.00296034,
        0.15725957,
        -0.04507174,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


class ThreadedFrameGrabber:
    """Continuously grabs frames in a background thread and stores the latest one."""

    def __init__(self, read_fn, name: str) -> None:
        self._read_fn = read_fn
        self._name = name
        self._lock = Lock()
        self._latest = None
        self._frame_id = 0
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = Thread(target=self._loop, name=f"grabber_{self._name}", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
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
                time.sleep(0.005)

    def get_latest(self):
        with self._lock:
            return self._latest

    def get_new_frame(self, last_seen_id: int) -> tuple[int, object | None]:
        with self._lock:
            if self._frame_id > last_seen_id:
                return self._frame_id, self._latest
            return self._frame_id, None

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class OrbbecRGBCamera(base.OrbbecRGBDCamera):
    """Color-only Orbbec wrapper for RGB-trained checkpoints."""

    def read(self) -> np.ndarray | None:
        if self.pipeline is None:
            raise RuntimeError("Orbbec pipeline is not started.")

        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None

        color_rgb = self._decode_color_to_rgb(color_frame)
        if color_rgb.shape[:2] != (self.height, self.width):
            color_rgb = cv2.resize(color_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return color_rgb

    def _init_pipeline(self):
        if self.ob is None:
            raise RuntimeError("`pyorbbecsdk` is not imported.")

        pipeline = self.ob.Pipeline()
        config = self.ob.Config()

        color_candidates: list[object] = []
        sensor_enum = getattr(self.ob, "OBSensorType", None)
        if sensor_enum is not None:
            color_candidates.append(getattr(sensor_enum, "COLOR_SENSOR", None))
        stream_enum = getattr(self.ob, "StreamType", None)
        if stream_enum is not None:
            color_candidates.append(getattr(stream_enum, "COLOR", None))
        ob_stream_enum = getattr(self.ob, "OBStreamType", None)
        if ob_stream_enum is not None:
            color_candidates.append(getattr(ob_stream_enum, "COLOR_STREAM", None))

        color_profiles = None
        errors: list[str] = []
        for color_key in color_candidates:
            if color_key is None:
                continue
            try:
                color_profiles = pipeline.get_stream_profile_list(color_key)
                break
            except Exception as exc:
                errors.append(str(exc))

        if color_profiles is None:
            err_msg = "; ".join(errors) if errors else "unknown error"
            raise RuntimeError(f"Cannot query Orbbec color stream profiles: {err_msg}")

        color_profile = self._select_video_profile(
            color_profiles,
            target_width=self.width,
            target_height=self.height,
            target_fps=self.fps,
            prefer_formats={"OBFormat.MJPG", "OBFormat.YUYV", "OBFormat.RGB", "OBFormat.BGR"},
        )
        config.enable_stream(color_profile)
        pipeline.start(config)
        LOGGER.info("Orbbec color profile: %s", color_profile)
        return pipeline


def configure_pd_gains(client) -> None:
    client.set_motor_cfg(kp_config=PD_KP_CONFIG, kd_config=PD_KD_CONFIG)
    LOGGER.info("PD gains configured. left_manipulator Kp: %s", PD_KP_CONFIG["left_manipulator"])


def _ordered_visual_keys(visual_keys: list[str], preferred_key: str) -> list[str]:
    if preferred_key in visual_keys:
        return [preferred_key] + [k for k in visual_keys if k != preferred_key]
    return list(visual_keys)


def _build_visual_observation(
    visual_keys: list[str],
    top_rgb: np.ndarray,
    left_wrist_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
) -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {}
    for key in visual_keys:
        key_lower = key.lower()
        if "left_wrist" in key_lower:
            obs[key] = left_wrist_rgb
        elif "right_wrist" in key_lower:
            obs[key] = right_wrist_rgb
        else:
            obs[key] = top_rgb
    return obs


def _create_realsense_camera(serial: str, width: int, height: int, fps: int) -> RealSenseCamera:
    cfg = RealSenseCameraConfig(
        serial_number_or_name=serial,
        fps=fps,
        width=width,
        height=height,
        use_depth=False,
    )
    cam = RealSenseCamera(cfg)
    cam.connect()
    return cam


def _auto_detect_realsense_pair() -> tuple[str, str]:
    cameras = RealSenseCamera.find_cameras()
    if len(cameras) != 2:
        serials = [camera["id"] for camera in cameras]
        raise RuntimeError(
            f"Expected exactly 2 RealSense cameras for wrist deployment, found {len(cameras)}: {serials}. "
            "Provide --wrist-left-serial and --wrist-right-serial explicitly."
        )
    sorted_cameras = sorted(cameras, key=lambda camera: camera["id"])
    left_sn = sorted_cameras[0]["id"]
    right_sn = sorted_cameras[1]["id"]
    LOGGER.info(
        "Auto-detected RealSense pair: left=%s (%s), right=%s (%s)",
        left_sn,
        sorted_cameras[0].get("name", "?"),
        right_sn,
        sorted_cameras[1].get("name", "?"),
    )
    return left_sn, right_sn


def _read_realsense_rgb(camera: RealSenseCamera, name: str) -> np.ndarray | None:
    try:
        return camera.read(timeout_ms=200)
    except Exception as exc:
        LOGGER.warning("RealSense %s read failed: %s", name, exc)
        return None


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
    dummy_obs: dict[str, np.ndarray] = {}
    for key in visual_keys:
        key_lower = key.lower()
        if "wrist" in key_lower:
            dummy_obs[key] = np.zeros((wrist_height, wrist_width, 3), dtype=np.uint8)
        else:
            dummy_obs[key] = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
    dummy_obs[state_key] = np.zeros((state_dim,), dtype=np.float32)

    with torch.inference_mode():
        batch = prepare_observation_for_inference(
            observation=dummy_obs,
            device=device,
            task=task,
            robot_type=robot_type,
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
    left_wrist_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
    state_model: np.ndarray,
    action_dim: int,
) -> np.ndarray:
    observation = _build_visual_observation(
        visual_keys=visual_keys,
        top_rgb=top_rgb,
        left_wrist_rgb=left_wrist_rgb,
        right_wrist_rgb=right_wrist_rgb,
    )
    observation[state_key] = state_model

    with torch.inference_mode():
        batch = prepare_observation_for_inference(
            observation=observation,
            device=device,
            task=task,
            robot_type=robot_type,
        )
        action_tensor = postprocessor(policy.select_action(preprocessor(batch)))

    raw_action = action_tensor.reshape(-1, action_tensor.shape[-1])[0].detach().cpu().numpy().astype(np.float32)
    return base.fit_vector(raw_action, action_dim)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy Pi0 on GR2 with RGB-only top + dual wrist cameras.")
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="RGB-only dataset root, used to resolve the task prompt when --task is omitted.",
    )
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--robot-type", type=str, default=base.DEFAULT_ROBOT_TYPE)
    parser.add_argument("--domain-id", type=int, default=123)
    parser.add_argument("--robot-name", type=str, default="gr2")
    parser.add_argument("--client-init-retries", type=int, default=4)
    parser.add_argument("--client-retry-interval-s", type=float, default=2.0)
    parser.add_argument("--fsm-state", type=int, default=11)
    parser.add_argument("--fps", type=float, default=10.0)
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
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--camera-timeout-ms", type=int, default=200)
    parser.add_argument("--camera-warmup-frames", type=int, default=15)
    parser.add_argument("--camera-init-retries", type=int, default=6)
    parser.add_argument("--camera-retry-interval-s", type=float, default=1.0)
    parser.add_argument("--state-dim", type=int, default=0)
    parser.add_argument("--action-dim", type=int, default=0)
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--save-dir", type=str, default="scripts/outputs/deploy_gr2_pi0_rgb_wrist")
    parser.add_argument("--save-every", type=int, default=0)

    wrist = parser.add_argument_group("Wrist RealSense cameras")
    wrist.add_argument("--wrist-width", type=int, default=640)
    wrist.add_argument("--wrist-height", type=int, default=480)
    wrist.add_argument("--wrist-fps", type=int, default=30)

    return parser.parse_args()


def _resolve_task(args: argparse.Namespace) -> str:
    if args.task:
        return args.task

    tasks_path = Path(args.dataset_root) / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        LOGGER.warning("Task metadata not found at %s. Falling back to default task.", tasks_path)
        return base.DEFAULT_TASK

    import pyarrow.parquet as pq

    table = pq.read_table(str(tasks_path))
    df = table.to_pandas()
    if "task" in df.columns:
        tasks = df["task"].tolist()
    else:
        tasks = df.index.tolist()

    if not tasks:
        LOGGER.warning("No task found in %s. Falling back to default task.", tasks_path)
        return base.DEFAULT_TASK
    if len(tasks) == 1:
        LOGGER.info("Using the dataset's only task: %s", tasks[0])
        return tasks[0]

    print("\nAvailable tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"  [{i}] {task}")
    while True:
        choice = input(f"\nSelect task [1-{len(tasks)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(tasks):
            selected = tasks[int(choice) - 1]
            LOGGER.info("Selected task: %s", selected)
            return selected
        print(f"Invalid input. Please enter a number between 1 and {len(tasks)}.")


def setup_gui(args: argparse.Namespace) -> bool:
    if args.no_gui:
        return False
    try:
        base.check_opencv_gui_available()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, args.camera_width * 3, args.camera_height)
        return True
    except Exception as exc:
        LOGGER.warning("Disable GUI because it is unavailable: %s", exc)
        return False


def _resize_for_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def save_snapshot(
    *,
    save_dir: Path,
    frame_idx: int,
    top_rgb: np.ndarray,
    left_wrist_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    top_path = save_dir / f"top_rgb_{ts}_{frame_idx:06d}.png"
    left_path = save_dir / f"left_wrist_rgb_{ts}_{frame_idx:06d}.png"
    right_path = save_dir / f"right_wrist_rgb_{ts}_{frame_idx:06d}.png"
    cv2.imwrite(str(top_path), cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(left_path), cv2.cvtColor(left_wrist_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(right_path), cv2.cvtColor(right_wrist_rgb, cv2.COLOR_RGB2BGR))
    LOGGER.info("Saved snapshot: %s | %s | %s", top_path, left_path, right_path)


def render_gui_frame(
    args: argparse.Namespace,
    top_rgb: np.ndarray,
    left_wrist_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
    last_vis_ts: float,
    save_dir: Path,
    frame_idx: int,
) -> tuple[bool, float]:
    panel_images = [
        _resize_for_panel(top_rgb, args.camera_width, args.camera_height),
        _resize_for_panel(left_wrist_rgb, args.camera_width, args.camera_height),
        _resize_for_panel(right_wrist_rgb, args.camera_width, args.camera_height),
    ]
    panel = np.hstack([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in panel_images])

    now = time.time()
    vis_fps = 1.0 / max(1e-6, now - last_vis_ts)
    labels = ["Top RGB", "Left Wrist RGB", "Right Wrist RGB"]
    for idx, label in enumerate(labels):
        x = idx * args.camera_width + 10
        cv2.putText(panel, label, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(
        panel,
        f"FPS: {vis_fps:.1f}",
        (10, args.camera_height - 12),
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
        save_snapshot(
            save_dir=save_dir,
            frame_idx=frame_idx,
            top_rgb=top_rgb,
            left_wrist_rgb=left_wrist_rgb,
            right_wrist_rgb=right_wrist_rgb,
        )
    return False, now


def run(args: argparse.Namespace) -> None:
    args.task = _resolve_task(args)
    pretrained_dir = base.resolve_pretrained_model_dir(args.checkpoint_path)
    train_cfg = base.load_train_config(pretrained_dir)
    device = base.select_device(args.device)

    LOGGER.info("Model: %s | Device: %s", pretrained_dir, device)
    if train_cfg:
        dataset_cfg = train_cfg.get("dataset", {})
        LOGGER.info("Train dataset: repo_id=%s root=%s", dataset_cfg.get("repo_id"), dataset_cfg.get("root"))
    LOGGER.info(
        "Top camera: %dx%d@%d | Wrist cameras: %dx%d@%d",
        args.camera_width,
        args.camera_height,
        args.camera_fps,
        args.wrist_width,
        args.wrist_height,
        args.wrist_fps,
    )

    policy = PI0Policy.from_pretrained(str(pretrained_dir), strict=False).to(device)
    policy.eval()
    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
        LOGGER.info("Disabled gradient checkpointing for inference.")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    state_key, cfg_state_dim, visual_keys, action_key, cfg_action_dim = base.infer_policy_io(policy)
    state_dim = args.state_dim if args.state_dim > 0 else cfg_state_dim
    action_dim = args.action_dim if args.action_dim > 0 else cfg_action_dim
    visual_keys = _ordered_visual_keys(visual_keys, args.camera_key)
    need_left_wrist = any("left_wrist" in key.lower() for key in visual_keys)
    need_right_wrist = any("right_wrist" in key.lower() for key in visual_keys)
    rgbd_keys = [key for key in visual_keys if "depth" in key.lower()]
    if rgbd_keys:
        raise ValueError(
            "This RGB-only deploy script cannot satisfy checkpoints that expect depth keys. "
            f"Found depth visual keys: {rgbd_keys}. Use the RGB-only checkpoint or fall back to "
            "`deploy_gr2_pi0_rgbd_wrist.py`."
        )

    LOGGER.info(
        "Policy IO: state=%s(%d) action=%s(%d) visual_keys=%s",
        state_key,
        state_dim,
        action_key,
        action_dim,
        visual_keys,
    )
    if not need_left_wrist and not need_right_wrist:
        LOGGER.warning("Checkpoint does not use wrist cameras. Deployment will run with top camera only.")

    gui_enabled = setup_gui(args)
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    periodic_save_every = max(0, int(args.save_every))
    last_vis_ts = time.time()
    frame_idx = 0
    blank_left_rgb = np.zeros((args.wrist_height, args.wrist_width, 3), dtype=np.uint8)
    blank_right_rgb = np.zeros((args.wrist_height, args.wrist_width, 3), dtype=np.uint8)
    if not gui_enabled and periodic_save_every <= 0:
        periodic_save_every = 30
        LOGGER.info("GUI unavailable. Saving RGB snapshots every %d steps to %s", periodic_save_every, save_dir)

    orbbec = OrbbecRGBCamera(
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
    grabber_left: ThreadedFrameGrabber | None = None
    grabber_right: ThreadedFrameGrabber | None = None
    client = None
    grabbers: list[ThreadedFrameGrabber] = []

    try:
        orbbec.connect()

        left_sn = DEFAULT_LEFT_WRIST_SERIAL
        right_sn = DEFAULT_RIGHT_WRIST_SERIAL

        if need_left_wrist:
            LOGGER.info("Connecting left wrist RealSense (SN=%s)...", left_sn)
            cam_left = _create_realsense_camera(left_sn, args.wrist_width, args.wrist_height, args.wrist_fps)
        if need_right_wrist:
            LOGGER.info("Connecting right wrist RealSense (SN=%s)...", right_sn)
            cam_right = _create_realsense_camera(right_sn, args.wrist_width, args.wrist_height, args.wrist_fps)

        grabber_top = ThreadedFrameGrabber(orbbec.read, "camera_top")
        grabbers = [grabber_top]
        if need_left_wrist:
            grabber_left = ThreadedFrameGrabber(
                lambda: _read_realsense_rgb(cam_left, "left_wrist"),
                "camera_left_wrist",
            )
            grabbers.append(grabber_left)
        if need_right_wrist:
            grabber_right = ThreadedFrameGrabber(
                lambda: _read_realsense_rgb(cam_right, "right_wrist"),
                "camera_right_wrist",
            )
            grabbers.append(grabber_right)
        for grabber in grabbers:
            grabber.start()

        LOGGER.info("Waiting for first frames from required cameras...")
        wait_start = time.time()
        while time.time() - wait_start < 10.0:
            if all(grabber.get_latest() is not None for grabber in grabbers):
                break
            time.sleep(0.05)
        else:
            missing = [grabber._name for grabber in grabbers if grabber.get_latest() is None]
            raise RuntimeError(f"Timeout waiting for first frame from: {missing}")
        LOGGER.info("All RGB cameras are streaming.")

        client = base.setup_robot_if_needed(args)
        if client is not None:
            configure_pd_gains(client)
            LOGGER.info("Moving robot to the dataset's initial pose...")
            base.smooth_transition(client, INIT_POSE_URDF, duration_s=3.0, frequency_hz=100)
            LOGGER.info("Initial pose reached.")

        LOGGER.info("Warming up policy...")
        warmup_policy(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            task=args.task,
            robot_type=args.robot_type,
            visual_keys=visual_keys,
            state_key=state_key,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            wrist_height=args.wrist_height,
            wrist_width=args.wrist_width,
            state_dim=state_dim,
        )
        LOGGER.info("Warm-up complete.")

        step = 0
        first_action = True
        prev_action_urdf: np.ndarray | None = None
        slow_loop_last_log_ts = time.time()
        last_top_id = last_left_id = last_right_id = 0
        policy.reset()

        while True:
            loop_start = time.perf_counter()

            top_id, top_rgb = grabber_top.get_new_frame(last_top_id)
            if grabber_left is not None:
                left_id, left_rgb = grabber_left.get_new_frame(last_left_id)
            else:
                left_id, left_rgb = last_left_id, blank_left_rgb
            if grabber_right is not None:
                right_id, right_rgb = grabber_right.get_new_frame(last_right_id)
            else:
                right_id, right_rgb = last_right_id, blank_right_rgb

            if top_rgb is None and last_top_id == 0:
                time.sleep(0.001)
                continue
            if top_rgb is None and left_rgb is None and right_rgb is None:
                time.sleep(0.001)
                continue

            if top_rgb is None:
                top_rgb = grabber_top.get_latest()
            if left_rgb is None and grabber_left is not None:
                left_rgb = grabber_left.get_latest()
            if right_rgb is None and grabber_right is not None:
                right_rgb = grabber_right.get_latest()
            if left_rgb is None:
                left_rgb = blank_left_rgb
            if right_rgb is None:
                right_rgb = blank_right_rgb

            last_top_id = top_id
            last_left_id = left_id
            last_right_id = right_id

            if args.dry_run:
                state_full = np.zeros((45,), dtype=np.float32)
            else:
                if client is None:
                    raise RuntimeError("Robot client is not initialized.")
                state_full = base.get_robot_state_urdf(client)
            state_model = base.fit_vector(state_full, state_dim)

            raw_action_urdf = infer_single_action(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                task=args.task,
                robot_type=args.robot_type,
                visual_keys=visual_keys,
                state_key=state_key,
                top_rgb=top_rgb,
                left_wrist_rgb=left_rgb,
                right_wrist_rgb=right_rgb,
                state_model=state_model,
                action_dim=action_dim,
            )
            raw_action_urdf = base.fit_vector(raw_action_urdf, 35)

            prev_action_for_diag = prev_action_urdf
            action_urdf = raw_action_urdf.copy()
            if not args.disable_clamp:
                action_urdf = base.clamp_non_hand_joint_action(action_urdf)
            if prev_action_urdf is not None:
                action_urdf = base.stabilize_action(action_urdf, prev_action_urdf, args)

            if not args.dry_run:
                if client is None:
                    raise RuntimeError("Robot client is not initialized.")
                if first_action and args.transition_time_s > 0:
                    base.smooth_transition(client, action_urdf, args.transition_time_s, args.transition_freq)
                    policy.reset()
                    first_action = False
                else:
                    base.send_action_to_robot(client, action_urdf, args.send_base)
                    first_action = False

            if step % max(1, args.log_every) == 0:
                arm_dim = min(14, state_full.shape[0], action_urdf.shape[0])
                arm_delta = float(np.linalg.norm(action_urdf[:arm_dim] - state_full[:arm_dim]))
                base.log_action_diagnostics(
                    step=step,
                    arm_delta=arm_delta,
                    depth_range="rgb-only",
                    raw_action_urdf=raw_action_urdf,
                    final_action_urdf=action_urdf,
                    prev_action_urdf=prev_action_for_diag,
                    args=args,
                )

            prev_action_urdf = action_urdf.copy()

            if gui_enabled:
                should_quit, last_vis_ts = render_gui_frame(
                    args=args,
                    top_rgb=top_rgb,
                    left_wrist_rgb=left_rgb,
                    right_wrist_rgb=right_rgb,
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
                    top_rgb=top_rgb,
                    left_wrist_rgb=left_rgb,
                    right_wrist_rgb=right_rgb,
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
                        "Slow loop: %.1f ms (target %.1f ms). Consider --no-gui or lower --fps.",
                        elapsed * 1000.0,
                        1000.0 / max(1e-6, args.fps),
                    )
                    slow_loop_last_log_ts = now_ts
            time.sleep(max(0.0, 1.0 / args.fps - elapsed))

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        for grabber in grabbers:
            grabber.stop()
        orbbec.close()
        for cam, name in [(cam_left, "left_wrist"), (cam_right, "right_wrist")]:
            if cam is not None:
                try:
                    cam.disconnect()
                except Exception as exc:
                    LOGGER.warning("Failed to disconnect %s camera: %s", name, exc)
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
