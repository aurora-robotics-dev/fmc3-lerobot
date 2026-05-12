#!/usr/bin/env python3

"""Deploy a Pi0 policy on Fourier GR2 with top + right-wrist RGB cameras only.

This script is adapted from `deploy_gr2_pi0_rgb_wrist.py` for the dataset:
`fmc3_gr2_black_capped_bottle_yellow_to_green_lerobot`

Expected observations:
  - observation.images.camera_top
  - observation.images.camera_right_wrist
  - observation.state
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import deploy_gr2_pi0 as base
import deploy_gr2_pi0_rgb_wrist as wrist_base
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import prepare_observation_for_inference

LOGGER = logging.getLogger(__name__)
WINDOW_NAME = "GR2 PI0 Deploy | Top + Right Wrist (q/ESC quit, s save)"
_RECEIVED_STOP_SIGNAL = False

DEFAULT_CHECKPOINT_PATH = (
    "/home/phl/workspace/mymodels/gr2/"
    "pi0_gr2_black_capped_bottle_yellow_to_green/checkpoints/last/pretrained_model"
)
DEFAULT_DATASET_ROOT = (
    "/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/"
    "fmc3_gr2_black_capped_bottle_yellow_to_green_lerobot"
)
# DEFAULT_CHECKPOINT_PATH = (
#     "/home/phl/workspace/mymodels/gr2/"
#     "/pi0_gr2_black_capped_bottle_green_to_yellow/010000/pretrained_model"
# )
# DEFAULT_DATASET_ROOT = (
#     "/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/"
#     "fmc3_gr2_black_capped_bottle_green_to_yellow_lerobot"
# )
DEFAULT_RIGHT_WRIST_SERIAL = "349522072801"
LEFT_MANIPULATOR_SLICE = slice(0, 7)
LEFT_HAND_SLICE = slice(14, 20)
RIGHT_HAND_FINGER_MIN_URDF = -1.9226667
RIGHT_HAND_FINGER_MAX_URDF = 0.0
RIGHT_HAND_FINGER_SLICE = slice(20, 24)
RIGHT_HAND_THUMB_PITCH_INDEX = 24


# First-frame joint pose extracted from the dataset's first observation.state.
# Layout: 29 joint dims + 6 zero base-action dims.
FALLBACK_INIT_POSE_URDF = np.array(
    [
        -0.33063921,
        -0.26062882,
        -1.80515277,
        -0.14196779,
        -0.33864361,
        -0.14060171,
        0.08505822,
        -0.34313941,
        -0.50384259,
        0.55984002,
        -1.24582672,
        -0.68907523,
        0.01295744,
        -0.62775779,
        -0.00458619,
        -0.00890377,
        -0.01201849,
        -0.01197735,
        0.00762257,
        -1.56070650,
        -0.00806141,
        -0.00798318,
        -0.01722117,
        -0.01064423,
        0.00638821,
        -1.57819879,
        0.01420687,
        0.19575508,
        -0.13117069,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

ThreadedFrameGrabber = wrist_base.ThreadedFrameGrabber
OrbbecRGBCamera = wrist_base.OrbbecRGBCamera


def _auto_detect_single_realsense() -> str:
    cameras = RealSenseCamera.find_cameras()
    if len(cameras) != 1:
        serials = [camera["id"] for camera in cameras]
        raise RuntimeError(
            "Expected exactly 1 RealSense camera for right-wrist deployment, "
            f"found {len(cameras)}: {serials}. Provide --wrist-right-serial explicitly."
        )

    serial = cameras[0]["id"]
    LOGGER.info(
        "Auto-detected right wrist RealSense: %s (%s)",
        serial,
        cameras[0].get("name", "?"),
    )
    return serial


def _build_visual_observation(
    visual_keys: list[str],
    top_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
) -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {}
    for key in visual_keys:
        if "right_wrist" in key.lower():
            obs[key] = right_wrist_rgb
        else:
            obs[key] = top_rgb
    return obs


def _resolve_init_pose(dataset_root: str) -> np.ndarray:
    data_root = Path(dataset_root).expanduser() / "data"
    parquet_files = sorted(data_root.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        LOGGER.warning("No parquet data found under %s. Using fallback initial pose.", data_root)
        return FALLBACK_INIT_POSE_URDF.copy()

    try:
        import pyarrow.parquet as pq

        table = pq.read_table(str(parquet_files[0]), columns=["observation.state"])
        if table.num_rows == 0:
            raise ValueError(f"Empty parquet file: {parquet_files[0]}")

        state = np.asarray(table.column(0)[0].as_py(), dtype=np.float32)
        if state.shape[0] < 29:
            raise ValueError(f"Expected observation.state dim >= 29, got {state.shape[0]}")

        init_pose = np.concatenate([state[:29], np.zeros((6,), dtype=np.float32)], axis=0)
        LOGGER.info("Initial pose loaded from %s", parquet_files[0])
        return init_pose
    except Exception as exc:
        LOGGER.warning("Failed to derive initial pose from dataset. Using fallback pose. Error: %s", exc)
        return FALLBACK_INIT_POSE_URDF.copy()


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
        if "right_wrist" in key.lower():
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
    right_wrist_rgb: np.ndarray,
    state_model: np.ndarray,
    action_dim: int,
) -> np.ndarray:
    observation = _build_visual_observation(
        visual_keys=visual_keys,
        top_rgb=top_rgb,
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

    all_actions = action_tensor.reshape(-1, action_tensor.shape[-1]).detach().cpu().numpy().astype(np.float32)
    return np.array([base.fit_vector(action, action_dim) for action in all_actions], dtype=np.float32)


def apply_right_hand_trigger_coupling(action_urdf: np.ndarray) -> np.ndarray:
    """Match the dataset's right-hand action pattern: fingers = -2 * thumb_pitch."""
    if action_urdf.shape[0] <= RIGHT_HAND_THUMB_PITCH_INDEX:
        return action_urdf

    coupled = action_urdf.copy()
    thumb_pitch = float(coupled[RIGHT_HAND_THUMB_PITCH_INDEX])
    coupled[RIGHT_HAND_FINGER_SLICE] = float(
        np.clip(-2.0 * thumb_pitch, RIGHT_HAND_FINGER_MIN_URDF, RIGHT_HAND_FINGER_MAX_URDF)
    )
    return coupled


def apply_left_side_motion_mask(action_urdf: np.ndarray, state_urdf: np.ndarray) -> np.ndarray:
    """Keep left arm and left hand fixed at the robot's current joint state."""
    masked = action_urdf.copy()
    masked[LEFT_MANIPULATOR_SLICE] = state_urdf[LEFT_MANIPULATOR_SLICE]
    masked[LEFT_HAND_SLICE] = state_urdf[LEFT_HAND_SLICE]
    return masked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy Pi0 on GR2 with RGB-only top camera + right wrist camera."
    )
    parser.add_argument(
        "--checkpoint-path",
        "--weights",
        "--weights-path",
        "--ckpt",
        dest="checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the checkpoint directory or pretrained_model directory to load at runtime.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root used to resolve the task prompt and initial pose when omitted.",
    )
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--robot-type", type=str, default=base.DEFAULT_ROBOT_TYPE)
    parser.add_argument("--domain-id", type=int, default=123)
    parser.add_argument("--robot-name", type=str, default="gr2")
    parser.add_argument("--client-init-retries", type=int, default=4)
    parser.add_argument("--client-retry-interval-s", type=float, default=2.0)
    parser.add_argument("--fsm-state", type=int, default=11)
    parser.add_argument("--fps", type=float, default=30.0)
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
    parser.add_argument("--save-dir", type=str, default="scripts/outputs/deploy_gr2_pi0_rgb_right_wrist")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument(
        "--right-side-only",
        "--freeze-left-arm-hand",
        dest="right_side_only",
        action="store_true",
        default=True,
        help="Freeze left_manipulator and left_hand at current state so only the right side is driven.",
    )
    parser.add_argument(
        "--allow-left-side-motion",
        dest="right_side_only",
        action="store_false",
        help="Allow the deploy script to command left_manipulator and left_hand again.",
    )
    parser.add_argument(
        "--disable-right-hand-trigger-coupling",
        action="store_true",
        help="Disable the dataset-specific right-hand coupling (fingers = -2 * thumb_pitch).",
    )

    wrist = parser.add_argument_group("Right wrist RealSense camera")
    wrist.add_argument(
        "--wrist-right-serial",
        type=str,
        default=DEFAULT_RIGHT_WRIST_SERIAL,
        help="Right wrist RealSense serial number or device name. Leave empty to auto-detect.",
    )
    wrist.add_argument("--wrist-width", type=int, default=640)
    wrist.add_argument("--wrist-height", type=int, default=480)
    wrist.add_argument("--wrist-fps", type=int, default=30)

    return parser.parse_args()


def setup_gui(args: argparse.Namespace) -> bool:
    if args.no_gui:
        return False
    try:
        base.check_opencv_gui_available()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, args.camera_width * 2, args.camera_height)
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
    right_wrist_rgb: np.ndarray,
) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    top_path = save_dir / f"top_rgb_{ts}_{frame_idx:06d}.png"
    right_path = save_dir / f"right_wrist_rgb_{ts}_{frame_idx:06d}.png"
    cv2.imwrite(str(top_path), cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(right_path), cv2.cvtColor(right_wrist_rgb, cv2.COLOR_RGB2BGR))
    LOGGER.info("Saved snapshot: %s | %s", top_path, right_path)


def render_gui_frame(
    args: argparse.Namespace,
    top_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
    last_vis_ts: float,
    save_dir: Path,
    frame_idx: int,
) -> tuple[bool, float]:
    panel_images = [
        _resize_for_panel(top_rgb, args.camera_width, args.camera_height),
        _resize_for_panel(right_wrist_rgb, args.camera_width, args.camera_height),
    ]
    panel = np.hstack([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in panel_images])

    now = time.time()
    vis_fps = 1.0 / max(1e-6, now - last_vis_ts)
    labels = ["Top RGB", "Right Wrist RGB"]
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
            right_wrist_rgb=right_wrist_rgb,
        )
    return False, now


def run(args: argparse.Namespace) -> None:
    args.task = wrist_base._resolve_task(args)
    pretrained_dir = base.resolve_pretrained_model_dir(args.checkpoint_path)
    train_cfg = base.load_train_config(pretrained_dir)
    device = base.select_device(args.device)

    LOGGER.info("Model: %s | Device: %s", pretrained_dir, device)
    if train_cfg:
        dataset_cfg = train_cfg.get("dataset", {})
        LOGGER.info("Train dataset: repo_id=%s root=%s", dataset_cfg.get("repo_id"), dataset_cfg.get("root"))
    LOGGER.info(
        "Top camera: %dx%d@%d | Right wrist camera: %dx%d@%d",
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
    visual_keys = wrist_base._ordered_visual_keys(visual_keys, args.camera_key)

    need_left_wrist = any("left_wrist" in key.lower() for key in visual_keys)
    need_right_wrist = any("right_wrist" in key.lower() for key in visual_keys)
    rgbd_keys = [key for key in visual_keys if "depth" in key.lower()]

    if need_left_wrist:
        raise ValueError(
            "This deploy script only supports top + right wrist RGB observations. "
            f"Checkpoint expects left wrist keys: {[key for key in visual_keys if 'left_wrist' in key.lower()]}"
        )
    if rgbd_keys:
        raise ValueError(
            "This RGB-only deploy script cannot satisfy checkpoints that expect depth keys. "
            f"Found depth visual keys: {rgbd_keys}."
        )

    LOGGER.info(
        "Policy IO: state=%s(%d) action=%s(%d) visual_keys=%s",
        state_key,
        state_dim,
        action_key,
        action_dim,
        visual_keys,
    )
    if args.right_side_only:
        LOGGER.info("Motion mask: enabled (freeze left_manipulator + left_hand)")
    else:
        LOGGER.info("Motion mask: disabled")
    if args.disable_right_hand_trigger_coupling:
        LOGGER.info("Right-hand trigger coupling: disabled")
    else:
        LOGGER.info("Right-hand trigger coupling: enabled (fingers = -2 * thumb_pitch)")
    if not need_right_wrist:
        LOGGER.warning("Checkpoint does not use right wrist camera. Deployment will run with top camera only.")

    gui_enabled = setup_gui(args)
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    periodic_save_every = max(0, int(args.save_every))
    last_vis_ts = time.time()
    frame_idx = 0
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
    cam_right: RealSenseCamera | None = None
    grabber_right: ThreadedFrameGrabber | None = None
    client = None
    grabbers: list[ThreadedFrameGrabber] = []

    try:
        orbbec.connect()

        right_sn = args.wrist_right_serial.strip()
        if need_right_wrist:
            if not right_sn:
                LOGGER.info("Auto-detecting right wrist RealSense...")
                right_sn = _auto_detect_single_realsense()
            LOGGER.info("Connecting right wrist RealSense (SN=%s)...", right_sn)
            cam_right = wrist_base._create_realsense_camera(
                right_sn,
                args.wrist_width,
                args.wrist_height,
                args.wrist_fps,
            )

        grabber_top = ThreadedFrameGrabber(orbbec.read, "camera_top")
        grabbers = [grabber_top]
        if need_right_wrist:
            grabber_right = ThreadedFrameGrabber(
                lambda: wrist_base._read_realsense_rgb(cam_right, "right_wrist"),
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
        LOGGER.info("All required RGB cameras are streaming.")

        client = base.setup_robot_if_needed(args)
        if client is not None:
            wrist_base.configure_pd_gains(client)
            init_pose = _resolve_init_pose(args.dataset_root)
            if args.right_side_only:
                current_state_urdf = base.get_robot_state_urdf(client)
                init_pose = apply_left_side_motion_mask(init_pose, current_state_urdf)
            LOGGER.info("Moving robot to the dataset's initial pose...")
            base.smooth_transition(client, init_pose, duration_s=3.0, frequency_hz=100)
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
        last_top_id = 0
        last_right_id = 0
        policy.reset()
        action_chunk: np.ndarray | None = None
        chunk_idx = 0

        while True:
            loop_start = time.perf_counter()

            top_id, top_rgb = grabber_top.get_new_frame(last_top_id)
            if grabber_right is not None:
                right_id, right_rgb = grabber_right.get_new_frame(last_right_id)
            else:
                right_id, right_rgb = last_right_id, None

            has_new_frame = top_rgb is not None
            if grabber_right is not None and right_rgb is not None:
                has_new_frame = True

            latest_top_rgb = grabber_top.get_latest()
            latest_right_rgb = grabber_right.get_latest() if grabber_right is not None else blank_right_rgb

            if latest_top_rgb is None:
                time.sleep(0.001)
                continue
            if not has_new_frame and (action_chunk is None or len(action_chunk) == 0):
                time.sleep(0.001)
                continue

            if top_rgb is None:
                top_rgb = latest_top_rgb
            if right_rgb is None and grabber_right is not None:
                right_rgb = latest_right_rgb
            if right_rgb is None:
                right_rgb = blank_right_rgb

            last_top_id = top_id
            last_right_id = right_id

            if args.dry_run:
                state_full = np.zeros((45,), dtype=np.float32)
            else:
                if client is None:
                    raise RuntimeError("Robot client is not initialized.")
                state_full = base.get_robot_state_urdf(client)
            state_model = base.fit_vector(state_full, state_dim)

            if has_new_frame:
                action_chunk = infer_single_action(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    device=device,
                    task=args.task,
                    robot_type=args.robot_type,
                    visual_keys=visual_keys,
                    state_key=state_key,
                    top_rgb=top_rgb,
                    right_wrist_rgb=right_rgb,
                    state_model=state_model,
                    action_dim=action_dim,
                )
                chunk_idx = 0
                action_model = action_chunk[0]
            else:
                if action_chunk is None or len(action_chunk) == 0:
                    time.sleep(0.001)
                    continue
                chunk_idx = min(chunk_idx + 1, len(action_chunk) - 1)
                action_model = action_chunk[chunk_idx]

            raw_action_urdf = base.fit_vector(action_model, 35)
            if not args.disable_right_hand_trigger_coupling:
                raw_action_urdf = apply_right_hand_trigger_coupling(raw_action_urdf)
            if args.right_side_only:
                raw_action_urdf = apply_left_side_motion_mask(raw_action_urdf, state_full)

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
                    action_chunk = None
                    chunk_idx = 0
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
        if cam_right is not None:
            try:
                cam_right.disconnect()
            except Exception as exc:
                LOGGER.warning("Failed to disconnect right_wrist camera: %s", exc)
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
