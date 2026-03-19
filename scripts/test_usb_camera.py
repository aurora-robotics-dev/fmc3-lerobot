#!/usr/bin/env python3
"""USB camera visualization demo.

Usage examples:
  python scripts/test_usb_camera.py
  python scripts/test_usb_camera.py --index 2 --width 1280 --height 720 --fps 30
  python scripts/test_usb_camera.py --list-only --max-index 8

Keyboard:
  q / ESC : quit
  s       : save current frame
  m       : toggle mirror
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a USB camera stream with OpenCV.")
    parser.add_argument("--index", type=int, default=0, help="Camera device index (e.g., /dev/video<index>).")
    parser.add_argument("--width", type=int, default=640, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=480, help="Requested frame height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested FPS.")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "v4l2", "any", "ffmpeg", "gstreamer", "dshow", "msmf"],
        help="OpenCV VideoCapture backend preference.",
    )
    parser.add_argument("--mjpg", action="store_true", help="Request MJPG pixel format when supported.")
    parser.add_argument("--warmup", type=int, default=8, help="Number of warmup frames to drop.")
    parser.add_argument("--max-index", type=int, default=10, help="Max index (inclusive) for --list-only scan.")
    parser.add_argument("--list-only", action="store_true", help="List available camera indices and exit.")
    parser.add_argument("--no-mirror", action="store_true", help="Disable initial mirror mode.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="scripts/outputs/usb_camera_demo",
        help="Directory for saved snapshots.",
    )
    parser.add_argument(
        "--reopen-fail-threshold",
        type=int,
        default=20,
        help="Reopen camera after this many consecutive read failures. <=0 disables reopen.",
    )
    parser.add_argument("--reopen-wait-s", type=float, default=0.5, help="Wait time before camera reopen.")
    return parser.parse_args()


def _backend_flag(name: str) -> int:
    mapping = {
        "auto": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
        "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
        "any": cv2.CAP_ANY,
        "ffmpeg": getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY),
        "gstreamer": getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY),
        "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
        "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    }
    return mapping[name]


def _check_gui_available() -> None:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("No DISPLAY/WAYLAND_DISPLAY found. Use an environment with GUI.")
    try:
        cv2.namedWindow("__usb_cam_gui_check__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__usb_cam_gui_check__")
    except Exception as exc:
        raise RuntimeError("OpenCV GUI backend is unavailable in this environment.") from exc


def _open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    preferred = _backend_flag(args.backend)
    cap = cv2.VideoCapture(args.index, preferred)
    if not cap.isOpened() and preferred != cv2.CAP_ANY:
        cap.release()
        cap = cv2.VideoCapture(args.index, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open USB camera index={args.index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if args.mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    for _ in range(max(0, args.warmup)):
        cap.read()

    return cap


def _list_cameras(max_index: int, backend_name: str) -> list[int]:
    found: list[int] = []
    backend = _backend_flag(backend_name)
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, backend)
        ok = cap.isOpened()
        if ok:
            ret, _ = cap.read()
            ok = bool(ret)
        cap.release()
        if ok:
            found.append(idx)
    return found


def main() -> None:
    args = parse_args()

    if args.list_only:
        cams = _list_cameras(args.max_index, args.backend)
        if cams:
            print("Available camera indices:", ", ".join(str(x) for x in cams))
        else:
            print("No camera found in scanned range.")
        return

    _check_gui_available()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    mirror = not args.no_mirror
    cap = _open_camera(args)
    fail_count = 0
    t_last = time.perf_counter()
    fps_smooth = 0.0

    print(
        f"Camera opened: index={args.index}, requested={args.width}x{args.height}@{args.fps:.1f}, "
        f"mirror={int(mirror)}"
    )
    print("Press q/ESC to quit, s to save frame, m to toggle mirror.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if args.reopen_fail_threshold > 0 and fail_count >= args.reopen_fail_threshold:
                    cap.release()
                    time.sleep(max(0.0, args.reopen_wait_s))
                    cap = _open_camera(args)
                    fail_count = 0
                continue

            fail_count = 0

            if mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = max(1e-6, now - t_last)
            t_last = now
            fps_inst = 1.0 / dt
            fps_smooth = fps_inst if fps_smooth == 0.0 else (0.90 * fps_smooth + 0.10 * fps_inst)

            h, w = frame.shape[:2]
            msg1 = f"index={args.index} frame={w}x{h} fps={fps_smooth:.1f}"
            msg2 = "q/ESC:quit  s:save  m:mirror"
            cv2.putText(frame, msg1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, msg2, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("USB Camera Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                mirror = not mirror
            if key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                out = save_dir / f"usb_cam_{args.index}_{ts}.jpg"
                cv2.imwrite(str(out), frame)
                print(f"Saved: {out}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            cv2.waitKey(1)
        except Exception:
            pass


if __name__ == "__main__":
    main()
