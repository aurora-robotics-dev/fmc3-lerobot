#!/usr/bin/env python3

"""Deploy an ACT policy on Fourier GR2 with RGB-only multi-camera input.

This script follows the same deployment flow as `deploy_gr2_pi0_rgb_wrist.py`:
  - observation.images.camera_top
  - observation.images.camera_left_wrist
  - observation.images.camera_right_wrist
  - observation.state

Unlike Pi0, ACT does not require a language task input. The checkpoint is expected
to be trained on the RGB-only GR2 dataset from this repo.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import deploy_gr2_pi0 as base
import deploy_gr2_pi0_rgb_wrist as wrist_base
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

LOGGER = logging.getLogger(__name__)
_RECEIVED_STOP_SIGNAL = False

DEFAULT_CHECKPOINT_PATH = (
    "/home/phl/workspace/mymodels/gr2/act_gr2_grab_bottle_from_box_to_desk_rgb_20260321_183147/checkpoints/065000/pretrained_model"
)
DEFAULT_LEFT_WRIST_SERIAL = wrist_base.DEFAULT_LEFT_WRIST_SERIAL
DEFAULT_RIGHT_WRIST_SERIAL = wrist_base.DEFAULT_RIGHT_WRIST_SERIAL
DEFAULT_WINDOW_NAME = "GR2 ACT Deploy | RGB Wrist (q/ESC quit, s save)"


def _build_act_observation(
    *,
    visual_keys: list[str],
    state_key: str,
    top_rgb: np.ndarray,
    left_wrist_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
    state_model: np.ndarray,
) -> dict[str, torch.Tensor]:
    observation_np = wrist_base._build_visual_observation(
        visual_keys=visual_keys,
        top_rgb=top_rgb,
        left_wrist_rgb=left_wrist_rgb,
        right_wrist_rgb=right_wrist_rgb,
    )

    observation: dict[str, torch.Tensor] = {
        state_key: torch.from_numpy(state_model).float(),
    }
    for key, image in observation_np.items():
        observation[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return observation


def warmup_policy(
    *,
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    visual_keys: list[str],
    state_key: str,
    camera_height: int,
    camera_width: int,
    wrist_height: int,
    wrist_width: int,
    state_dim: int,
) -> None:
    dummy_top = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
    dummy_left = np.zeros((wrist_height, wrist_width, 3), dtype=np.uint8)
    dummy_right = np.zeros((wrist_height, wrist_width, 3), dtype=np.uint8)
    dummy_state = np.zeros((state_dim,), dtype=np.float32)
    observation = _build_act_observation(
        visual_keys=visual_keys,
        state_key=state_key,
        top_rgb=dummy_top,
        left_wrist_rgb=dummy_left,
        right_wrist_rgb=dummy_right,
        state_model=dummy_state,
    )
    with torch.inference_mode():
        _ = postprocessor(policy.select_action(preprocessor(observation)))


def infer_single_action(
    *,
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    visual_keys: list[str],
    state_key: str,
    top_rgb: np.ndarray,
    left_wrist_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
    state_model: np.ndarray,
    action_dim: int,
) -> np.ndarray:
    observation = _build_act_observation(
        visual_keys=visual_keys,
        state_key=state_key,
        top_rgb=top_rgb,
        left_wrist_rgb=left_wrist_rgb,
        right_wrist_rgb=right_wrist_rgb,
        state_model=state_model,
    )
    with torch.inference_mode():
        action_tensor = postprocessor(policy.select_action(preprocessor(observation)))
    raw_action = action_tensor.reshape(-1, action_tensor.shape[-1])[0].detach().cpu().numpy().astype(np.float32)
    return base.fit_vector(raw_action, action_dim)


def _apply_hand_only_postprocess(
    action_urdf: np.ndarray,
    prev_action_urdf: np.ndarray | None,
    args: argparse.Namespace,
) -> np.ndarray:
    """Keep arm/head/base raw, but optionally smooth and rate-limit hand joints only."""
    if prev_action_urdf is None or action_urdf.shape != prev_action_urdf.shape:
        return action_urdf
    if args.disable_clamp or args.disable_hand_postprocess:
        return action_urdf

    stabilized = base.stabilize_action(action_urdf, prev_action_urdf, args)
    out = action_urdf.copy()
    out[14:26] = stabilized[14:26]
    return out.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy ACT on GR2 with RGB-only top + dual wrist cameras.")
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint",
        "--weights",
        "--weights-path",
        "--ckpt",
        dest="checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the checkpoint directory or pretrained_model directory to load at runtime.",
    )
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
    parser.add_argument(
        "--disable-hand-postprocess",
        action="store_true",
        help="Send dexterous hand joints directly without EMA or per-step delta limiting.",
    )
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
    parser.add_argument("--save-dir", type=str, default="scripts/outputs/deploy_gr2_act_rgb_wrist")
    parser.add_argument("--save-every", type=int, default=0)

    wrist = parser.add_argument_group("Wrist RealSense cameras")
    wrist.add_argument(
        "--wrist-left-serial",
        type=str,
        default=DEFAULT_LEFT_WRIST_SERIAL,
        help="Left wrist RealSense serial number or device name. Leave empty to auto-detect both wrists.",
    )
    wrist.add_argument(
        "--wrist-right-serial",
        type=str,
        default=DEFAULT_RIGHT_WRIST_SERIAL,
        help="Right wrist RealSense serial number or device name. Leave empty to auto-detect both wrists.",
    )
    wrist.add_argument("--wrist-width", type=int, default=640)
    wrist.add_argument("--wrist-height", type=int, default=480)
    wrist.add_argument("--wrist-fps", type=int, default=30)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    wrist_base.WINDOW_NAME = DEFAULT_WINDOW_NAME

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

    policy = ACTPolicy.from_pretrained(str(pretrained_dir)).to(device)
    policy.eval()

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
    if rgbd_keys:
        raise ValueError(
            "This RGB-only ACT deploy script cannot satisfy checkpoints that expect depth keys. "
            f"Found depth visual keys: {rgbd_keys}."
        )

    LOGGER.info(
        "Policy IO: state=%s(%d) action=%s(%d) visual_keys=%s n_action_steps=%s chunk_size=%s",
        state_key,
        state_dim,
        action_key,
        action_dim,
        visual_keys,
        getattr(policy.config, "n_action_steps", "?"),
        getattr(policy.config, "chunk_size", "?"),
    )
    if not need_left_wrist and not need_right_wrist:
        LOGGER.warning("Checkpoint does not use wrist cameras. Deployment will run with top camera only.")

    gui_enabled = wrist_base.setup_gui(args)
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

    orbbec = wrist_base.OrbbecRGBCamera(
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        timeout_ms=args.camera_timeout_ms,
        warmup_frames=args.camera_warmup_frames,
        init_retries=args.camera_init_retries,
        retry_interval_s=args.camera_retry_interval_s,
    )
    cam_left = None
    cam_right = None
    grabber_left: wrist_base.ThreadedFrameGrabber | None = None
    grabber_right: wrist_base.ThreadedFrameGrabber | None = None
    client = None
    grabbers: list[wrist_base.ThreadedFrameGrabber] = []

    try:
        orbbec.connect()

        left_sn = args.wrist_left_serial.strip()
        right_sn = args.wrist_right_serial.strip()

        if need_left_wrist or need_right_wrist:
            if not left_sn and not right_sn:
                LOGGER.info("Auto-detecting RealSense wrist cameras...")
                left_sn, right_sn = wrist_base._auto_detect_realsense_pair()
            elif need_left_wrist and not left_sn:
                raise ValueError(
                    "`--wrist-left-serial` is required for checkpoints that use the left wrist camera. "
                    "Leave both wrist serials empty to auto-detect the pair."
                )
            elif need_right_wrist and not right_sn:
                raise ValueError(
                    "`--wrist-right-serial` is required for checkpoints that use the right wrist camera. "
                    "Leave both wrist serials empty to auto-detect the pair."
                )

        if need_left_wrist:
            LOGGER.info("Connecting left wrist RealSense (SN=%s)...", left_sn)
            cam_left = wrist_base._create_realsense_camera(left_sn, args.wrist_width, args.wrist_height, args.wrist_fps)
        if need_right_wrist:
            LOGGER.info("Connecting right wrist RealSense (SN=%s)...", right_sn)
            cam_right = wrist_base._create_realsense_camera(
                right_sn, args.wrist_width, args.wrist_height, args.wrist_fps
            )

        grabber_top = wrist_base.ThreadedFrameGrabber(orbbec.read, "camera_top")
        grabbers = [grabber_top]
        if need_left_wrist:
            grabber_left = wrist_base.ThreadedFrameGrabber(
                lambda: wrist_base._read_realsense_rgb(cam_left, "left_wrist"),
                "camera_left_wrist",
            )
            grabbers.append(grabber_left)
        if need_right_wrist:
            grabber_right = wrist_base.ThreadedFrameGrabber(
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
        LOGGER.info("All RGB cameras are streaming.")

        client = base.setup_robot_if_needed(args)
        if client is not None:
            wrist_base.configure_pd_gains(client)
            LOGGER.info("Moving robot to the dataset's initial pose...")
            base.smooth_transition(client, wrist_base.INIT_POSE_URDF, duration_s=3.0, frequency_hz=100)
            LOGGER.info("Initial pose reached.")

        LOGGER.info("Warming up policy...")
        warmup_policy(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            visual_keys=visual_keys,
            state_key=state_key,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            wrist_height=args.wrist_height,
            wrist_width=args.wrist_width,
            state_dim=state_dim,
        )
        policy.reset()
        LOGGER.info("Warm-up complete.")

        step = 0
        first_action = True
        prev_action_urdf: np.ndarray | None = None
        slow_loop_last_log_ts = time.time()
        last_top_id = last_left_id = last_right_id = 0

        while True:
            loop_start = time.perf_counter()

            top_id, top_rgb = grabber_top.get_new_frame(last_top_id)
            if grabber_left is not None:
                left_id, left_rgb = grabber_left.get_new_frame(last_left_id)
            else:
                left_id, left_rgb = last_left_id, None
            if grabber_right is not None:
                right_id, right_rgb = grabber_right.get_new_frame(last_right_id)
            else:
                right_id, right_rgb = last_right_id, None

            latest_top_rgb = grabber_top.get_latest()
            latest_left_rgb = grabber_left.get_latest() if grabber_left is not None else blank_left_rgb
            latest_right_rgb = grabber_right.get_latest() if grabber_right is not None else blank_right_rgb

            if latest_top_rgb is None:
                time.sleep(0.001)
                continue

            if top_rgb is None:
                top_rgb = latest_top_rgb
            if left_rgb is None and grabber_left is not None:
                left_rgb = latest_left_rgb
            if right_rgb is None and grabber_right is not None:
                right_rgb = latest_right_rgb
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

            action_model = infer_single_action(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                visual_keys=visual_keys,
                state_key=state_key,
                top_rgb=top_rgb,
                left_wrist_rgb=left_rgb,
                right_wrist_rgb=right_rgb,
                state_model=state_model,
                action_dim=action_dim,
            )

            raw_action_urdf = base.fit_vector(action_model, 35)
            prev_action_for_diag = prev_action_urdf
            action_urdf = raw_action_urdf.copy()
            action_urdf = _apply_hand_only_postprocess(action_urdf, prev_action_urdf, args)

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
                should_quit, last_vis_ts = wrist_base.render_gui_frame(
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
                wrist_base.save_snapshot(
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
                wrist_base.cv2.destroyAllWindows()
                wrist_base.cv2.waitKey(1)
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
