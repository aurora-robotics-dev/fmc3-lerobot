#!/usr/bin/env python3

"""Two-camera RGB demo for GR2 camera bring-up.

This tool is derived from `scripts/deploy_gr2_pi0_rgb_wrist.py`, but removes
all policy and robot-control logic. It only brings up:

1. the top Orbbec RGB camera
2. one wrist RealSense RGB camera

and shows the two streams side by side.

Examples:
  python scripts/tools/gr2_two_camera_rgb_demo.py
  python scripts/tools/gr2_two_camera_rgb_demo.py --wrist-side left
  python scripts/tools/gr2_two_camera_rgb_demo.py --wrist-serial 349522072801
  python scripts/tools/gr2_two_camera_rgb_demo.py --no-gui --save-every 30
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

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import deploy_gr2_pi0 as base
import deploy_gr2_pi0_rgb_wrist as wrist_base
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera

LOGGER = logging.getLogger(__name__)
WINDOW_NAME = "GR2 Two-Camera RGB Demo (q/ESC quit, s save)"
_RECEIVED_STOP_SIGNAL = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize top Orbbec RGB + one wrist RealSense RGB stream.")
    parser.add_argument(
        "--wrist-side",
        type=str,
        default="right",
        choices=["left", "right"],
        help="Which wrist camera to use in the two-camera demo.",
    )
    parser.add_argument(
        "--wrist-serial",
        type=str,
        default="",
        help="Wrist RealSense serial number or device name. If empty, auto-detect from connected devices.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Visualization/update loop frequency.")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many iterations. 0 means infinite.")
    parser.add_argument("--no-gui", action="store_true", help="Disable the OpenCV window and save periodically.")
    parser.add_argument("--save-dir", type=str, default="scripts/outputs/gr2_two_camera_rgb_demo")
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save every N iterations. If GUI is disabled and this is 0, it defaults to 30.",
    )

    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-timeout-ms", type=int, default=200)
    parser.add_argument("--camera-warmup-frames", type=int, default=15)
    parser.add_argument("--camera-init-retries", type=int, default=6)
    parser.add_argument("--camera-retry-interval-s", type=float, default=1.0)

    parser.add_argument("--wrist-width", type=int, default=640)
    parser.add_argument("--wrist-height", type=int, default=480)
    parser.add_argument("--wrist-fps", type=int, default=30)
    return parser.parse_args()


def _resolve_wrist_serial(wrist_side: str, explicit_serial: str) -> str:
    if explicit_serial.strip():
        return explicit_serial.strip()

    cameras = RealSenseCamera.find_cameras()
    if len(cameras) == 1:
        serial = cameras[0]["id"]
        LOGGER.info(
            "Auto-detected a single RealSense for %s wrist: %s (%s)",
            wrist_side,
            serial,
            cameras[0].get("name", "?"),
        )
        return serial

    if len(cameras) == 2:
        sorted_cameras = sorted(cameras, key=lambda camera: camera["id"])
        index = 0 if wrist_side == "left" else 1
        serial = sorted_cameras[index]["id"]
        LOGGER.info(
            "Auto-selected %s wrist RealSense: %s (%s)",
            wrist_side,
            serial,
            sorted_cameras[index].get("name", "?"),
        )
        return serial

    serials = [camera["id"] for camera in cameras]
    raise RuntimeError(
        f"Cannot auto-detect wrist camera for side={wrist_side}. Found {len(cameras)} RealSense devices: {serials}. "
        "Pass --wrist-serial explicitly."
    )


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
    wrist_side: str,
    top_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    top_path = save_dir / f"top_rgb_{ts}_{frame_idx:06d}.png"
    wrist_path = save_dir / f"{wrist_side}_wrist_rgb_{ts}_{frame_idx:06d}.png"
    cv2.imwrite(str(top_path), cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(wrist_path), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
    LOGGER.info("Saved snapshot: %s | %s", top_path, wrist_path)


def render_gui_frame(
    *,
    args: argparse.Namespace,
    wrist_side: str,
    top_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    last_vis_ts: float,
    save_dir: Path,
    frame_idx: int,
) -> tuple[bool, float]:
    panel_images = [
        _resize_for_panel(top_rgb, args.camera_width, args.camera_height),
        _resize_for_panel(wrist_rgb, args.camera_width, args.camera_height),
    ]
    panel = np.hstack([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in panel_images])

    now = time.time()
    vis_fps = 1.0 / max(1e-6, now - last_vis_ts)
    labels = ["Top RGB", f"{wrist_side.capitalize()} Wrist RGB"]
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
            wrist_side=wrist_side,
            top_rgb=top_rgb,
            wrist_rgb=wrist_rgb,
        )
    return False, now


def run(args: argparse.Namespace) -> None:
    gui_enabled = setup_gui(args)
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    periodic_save_every = max(0, int(args.save_every))
    if not gui_enabled and periodic_save_every <= 0:
        periodic_save_every = 30
        LOGGER.info("GUI unavailable. Saving RGB snapshots every %d steps to %s", periodic_save_every, save_dir)

    last_vis_ts = time.time()
    frame_idx = 0

    wrist_serial = _resolve_wrist_serial(args.wrist_side, args.wrist_serial)
    LOGGER.info(
        "Starting two-camera demo with top Orbbec + %s wrist RealSense (SN=%s)",
        args.wrist_side,
        wrist_serial,
    )

    orbbec = wrist_base.OrbbecRGBCamera(
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        timeout_ms=args.camera_timeout_ms,
        warmup_frames=args.camera_warmup_frames,
        init_retries=args.camera_init_retries,
        retry_interval_s=args.camera_retry_interval_s,
    )
    wrist_cam = None
    grabber_top = None
    grabber_wrist = None

    try:
        orbbec.connect()
        wrist_cam = wrist_base._create_realsense_camera(
            wrist_serial,
            args.wrist_width,
            args.wrist_height,
            args.wrist_fps,
        )

        grabber_top = wrist_base.ThreadedFrameGrabber(orbbec.read, "camera_top")
        grabber_wrist = wrist_base.ThreadedFrameGrabber(
            lambda: wrist_base._read_realsense_rgb(wrist_cam, f"{args.wrist_side}_wrist"),
            f"camera_{args.wrist_side}_wrist",
        )
        grabber_top.start()
        grabber_wrist.start()

        LOGGER.info("Waiting for first frames from both cameras...")
        wait_start = time.time()
        while time.time() - wait_start < 10.0:
            if grabber_top.get_latest() is not None and grabber_wrist.get_latest() is not None:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("Timeout waiting for first frame from top/wrist cameras.")
        LOGGER.info("Both RGB cameras are streaming.")

        while True:
            loop_start = time.perf_counter()
            top_rgb = grabber_top.get_latest()
            wrist_rgb = grabber_wrist.get_latest()
            if top_rgb is None or wrist_rgb is None:
                time.sleep(0.005)
                continue

            if gui_enabled:
                should_quit, last_vis_ts = render_gui_frame(
                    args=args,
                    wrist_side=args.wrist_side,
                    top_rgb=top_rgb,
                    wrist_rgb=wrist_rgb,
                    last_vis_ts=last_vis_ts,
                    save_dir=save_dir,
                    frame_idx=frame_idx,
                )
                if should_quit:
                    break
            elif periodic_save_every > 0 and frame_idx % periodic_save_every == 0:
                save_snapshot(
                    save_dir=save_dir,
                    frame_idx=frame_idx,
                    wrist_side=args.wrist_side,
                    top_rgb=top_rgb,
                    wrist_rgb=wrist_rgb,
                )

            frame_idx += 1
            if args.max_steps > 0 and frame_idx >= args.max_steps:
                LOGGER.info("Reached max steps: %d", args.max_steps)
                break

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, 1.0 / max(1e-6, args.fps) - elapsed))
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        if grabber_top is not None:
            grabber_top.stop()
        if grabber_wrist is not None:
            grabber_wrist.stop()
        orbbec.close()
        if wrist_cam is not None:
            try:
                wrist_cam.disconnect()
            except Exception as exc:
                LOGGER.warning("Failed to disconnect wrist camera: %s", exc)
        if gui_enabled:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass


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
