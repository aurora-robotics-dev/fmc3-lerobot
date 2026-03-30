#!/usr/bin/env python3

"""Dual-model PI0 inference server for GR2 RGB wrist deployment.

The server preloads two PI0 checkpoints on startup and exposes a local Unix
socket JSON API. Only one model can drive the robot at a time. When a start
request arrives for the other model, the current worker is stopped first and
the requested model takes over.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import signal
import socketserver
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import deploy_gr2_pi0 as base  # noqa: E402
import deploy_gr2_pi0_rgb_wrist as rgb_wrist  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi0.modeling_pi0 import PI0Policy  # noqa: E402

LOGGER = logging.getLogger("gr2_dual_pi0_rgb_wrist_inference_server")

DEFAULT_UNIX_SOCKET_PATH = "/tmp/gr2_dual_pi0_rgb_wrist.sock"
MODEL_GREEN_TO_YELLOW = "green_to_yellow"
MODEL_YELLOW_TO_GREEN = "yellow_to_green"

LEFT_MANIPULATOR_SLICE = slice(0, 7)
LEFT_HAND_SLICE = slice(14, 20)
RIGHT_HAND_FINGER_MIN_URDF = -1.9226667
RIGHT_HAND_FINGER_MAX_URDF = 0.0
RIGHT_HAND_FINGER_SLICE = slice(20, 24)
RIGHT_HAND_THUMB_PITCH_INDEX = 24

DEFAULT_GREEN_TO_YELLOW_CHECKPOINT_PATH = (
    "/home/phl/workspace/mymodels/gr2/pi0/pi0_green_to_yellow_0327/checkpoints/last/pretrained_model"
)
DEFAULT_GREEN_TO_YELLOW_TASK = (
    "move the black-capped bottle from the green area to the yellow area"
)
DEFAULT_YELLOW_TO_GREEN_CHECKPOINT_PATH = (
    "/home/phl/workspace/mymodels/gr2/pi0/pi0_gr2_black_capped_bottle_yellow_to_green/checkpoints/last/pretrained_model"
)
DEFAULT_YELLOW_TO_GREEN_TASK = (
    "move the black-capped bottle from the yellow area to the green area"
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint_path: str
    task: str


@dataclass
class LoadedModel:
    spec: ModelSpec
    pretrained_dir: Path
    policy: PI0Policy
    preprocessor: Any
    postprocessor: Any
    state_key: str
    state_dim: int
    action_dim: int
    visual_keys: list[str]


@dataclass(frozen=True)
class WorkerConfig:
    model_name: str
    task: str
    max_steps: int
    fps: float
    fsm_state: int


@dataclass
class SharedRuntimeResources:
    orbbec: rgb_wrist.OrbbecRGBCamera | None = None
    cam_left: Any = None
    cam_right: Any = None
    grabber_top: rgb_wrist.ThreadedFrameGrabber | None = None
    grabber_left: rgb_wrist.ThreadedFrameGrabber | None = None
    grabber_right: rgb_wrist.ThreadedFrameGrabber | None = None
    client: Any = None
    left_serial: str = ""
    right_serial: str = ""

    def grabbers(self) -> list[rgb_wrist.ThreadedFrameGrabber]:
        return [
            grabber
            for grabber in (self.grabber_top, self.grabber_left, self.grabber_right)
            if grabber is not None
        ]


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


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


class DualModelInferenceRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._active_model_name: str | None = None

        self.device = base.select_device(args.device)
        self.model_specs = {
            MODEL_GREEN_TO_YELLOW: ModelSpec(
                name=MODEL_GREEN_TO_YELLOW,
                checkpoint_path=args.green_to_yellow_checkpoint_path,
                task=args.green_to_yellow_task,
            ),
            MODEL_YELLOW_TO_GREEN: ModelSpec(
                name=MODEL_YELLOW_TO_GREEN,
                checkpoint_path=args.yellow_to_green_checkpoint_path,
                task=args.yellow_to_green_task,
            ),
        }
        self.models: dict[str, LoadedModel] = {}
        self._shared = SharedRuntimeResources()
        self._shared_need_left_wrist = False
        self._shared_need_right_wrist = False
        self._did_move_to_init_pose = False
        self._last_fsm_state: int | None = None
        self._had_previous_run = False
        self._window_name = "GR2 Dual PI0 | RGB Wrist"
        self._gui_enabled = False
        self._gui_quit_requested = threading.Event()
        self._save_dir = Path(self.args.save_dir).expanduser().resolve()
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._periodic_save_every = max(0, int(self.args.save_every))
        self._frame_idx = 0

        self._status: dict[str, Any] = {
            "ok": True,
            "state": "loading",
            "message": "initializing",
            "active_model": None,
            "last_model": None,
            "task": None,
            "fps": args.fps,
            "fsm_state": args.fsm_state,
            "step": 0,
            "started_at": None,
            "updated_at": time.time(),
            "last_error": "",
            "last_error_traceback": "",
            "models": {
                name: {
                    "name": name,
                    "loaded": False,
                    "checkpoint_path": spec.checkpoint_path,
                    "task": spec.task,
                    "pretrained_dir": None,
                    "state_dim": None,
                    "action_dim": None,
                    "visual_keys": [],
                }
                for name, spec in self.model_specs.items()
            },
        }

        self._load_all_models()
        self._shared_need_left_wrist = any(
            "left_wrist" in key.lower() for model in self.models.values() for key in model.visual_keys
        )
        self._shared_need_right_wrist = any(
            "right_wrist" in key.lower() for model in self.models.values() for key in model.visual_keys
        )
        self._setup_visualization()
        self._mark(ok=True, state="idle", message="all models loaded")

    def _mark(self, **kwargs: Any) -> None:
        with self._lock:
            self._status.update(kwargs)
            self._status["updated_at"] = time.time()

    def _mark_model_loaded(self, model: LoadedModel) -> None:
        with self._lock:
            self._status["models"][model.spec.name].update(
                {
                    "loaded": True,
                    "checkpoint_path": model.spec.checkpoint_path,
                    "task": model.spec.task,
                    "pretrained_dir": str(model.pretrained_dir),
                    "state_dim": model.state_dim,
                    "action_dim": model.action_dim,
                    "visual_keys": list(model.visual_keys),
                }
            )
            self._status["updated_at"] = time.time()

    def _mark_model_failed(self, spec: ModelSpec, exc: Exception) -> None:
        with self._lock:
            self._status["models"][spec.name].update(
                {
                    "loaded": False,
                    "checkpoint_path": spec.checkpoint_path,
                    "task": spec.task,
                    "pretrained_dir": None,
                    "state_dim": None,
                    "action_dim": None,
                    "visual_keys": [],
                }
            )
            self._status.update(
                {
                    "ok": False,
                    "state": "error",
                    "message": f"failed to load model {spec.name}",
                    "last_error": str(exc),
                    "last_error_traceback": traceback.format_exc(),
                    "updated_at": time.time(),
                }
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            out = copy.deepcopy(self._status)
            out["worker_alive"] = bool(self._worker and self._worker.is_alive())
            out["device"] = str(self.device)
            return out

    def health(self) -> dict[str, Any]:
        status = self.status()
        all_loaded = all(model_info["loaded"] for model_info in status["models"].values())
        with self._lock:
            camera_connected = self._shared.grabber_top is not None and self._shared.orbbec is not None
            robot_connected = self._shared.client is not None
        return {
            "ok": all_loaded and status.get("state") != "error",
            "state": status.get("state"),
            "device": status.get("device"),
            "worker_alive": status.get("worker_alive"),
            "active_model": status.get("active_model"),
            "camera_connected": camera_connected,
            "robot_connected": robot_connected or self.args.dry_run,
            "models": {
                name: {
                    "loaded": info["loaded"],
                    "checkpoint_path": info["checkpoint_path"],
                    "task": info["task"],
                }
                for name, info in status["models"].items()
            },
            "message": status.get("message", ""),
        }

    def _load_model(self, spec: ModelSpec) -> LoadedModel:
        pretrained_dir = base.resolve_pretrained_model_dir(spec.checkpoint_path)
        LOGGER.info("Loading model '%s' from %s on %s", spec.name, pretrained_dir, self.device)

        policy = PI0Policy.from_pretrained(str(pretrained_dir), strict=False).to(self.device)
        policy.eval()
        if hasattr(policy.model, "gradient_checkpointing_disable"):
            policy.model.gradient_checkpointing_disable()

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(pretrained_dir),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        state_key, cfg_state_dim, visual_keys, _action_key, cfg_action_dim = base.infer_policy_io(policy)
        state_dim = self.args.state_dim if self.args.state_dim > 0 else cfg_state_dim
        action_dim = self.args.action_dim if self.args.action_dim > 0 else cfg_action_dim
        ordered_visual_keys = rgb_wrist._ordered_visual_keys(visual_keys, self.args.camera_key)

        rgbd_keys = [key for key in ordered_visual_keys if "depth" in key.lower()]
        if rgbd_keys:
            raise ValueError(
                "This dual-model server is based on deploy_gr2_pi0_rgb_wrist.py and only supports RGB inputs. "
                f"Model '{spec.name}' expects depth keys: {rgbd_keys}"
            )

        rgb_wrist.warmup_policy(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=self.device,
            task=spec.task,
            robot_type=self.args.robot_type,
            visual_keys=ordered_visual_keys,
            state_key=state_key,
            camera_height=self.args.camera_height,
            camera_width=self.args.camera_width,
            wrist_height=self.args.wrist_height,
            wrist_width=self.args.wrist_width,
            state_dim=state_dim,
        )
        policy.reset()

        return LoadedModel(
            spec=spec,
            pretrained_dir=pretrained_dir,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            state_key=state_key,
            state_dim=state_dim,
            action_dim=action_dim,
            visual_keys=ordered_visual_keys,
        )

    def _load_all_models(self) -> None:
        for spec in self.model_specs.values():
            try:
                model = self._load_model(spec)
            except Exception as exc:
                self._mark_model_failed(spec, exc)
                for loaded in self.models.values():
                    try:
                        loaded.policy.cpu()
                        del loaded.policy
                    except Exception:
                        pass
                self.models.clear()
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            self.models[spec.name] = model
            self._mark_model_loaded(model)

    def _reset_all_policies(self) -> None:
        for model in self.models.values():
            try:
                model.policy.reset()
            except Exception:
                LOGGER.debug("Failed to reset policy '%s'", model.spec.name, exc_info=True)

    def _build_robot_args(self, run_cfg: WorkerConfig) -> argparse.Namespace:
        return argparse.Namespace(
            dry_run=self.args.dry_run,
            domain_id=self.args.domain_id,
            robot_name=self.args.robot_name,
            client_init_retries=self.args.client_init_retries,
            client_retry_interval_s=self.args.client_retry_interval_s,
            skip_confirm=True,
            fsm_state=run_cfg.fsm_state,
        )

    def _build_algo_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            action_ema_alpha=self.args.action_ema_alpha,
            arm_ema_alpha=self.args.arm_ema_alpha,
            hand_ema_alpha=self.args.hand_ema_alpha,
            max_arm_delta=self.args.max_arm_delta,
            max_hand_delta=self.args.max_hand_delta,
            max_head_waist_delta=self.args.max_head_waist_delta,
            max_base_delta=self.args.max_base_delta,
            send_base=self.args.send_base,
        )

    def _setup_visualization(self) -> None:
        if self.args.no_gui:
            return
        try:
            base.check_opencv_gui_available()
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_name, self.args.camera_width * 3, self.args.camera_height)
            self._gui_enabled = True
        except Exception as exc:
            LOGGER.warning("Disable GUI because it is unavailable: %s", exc)
            self._gui_enabled = False

    def _close_visualization(self) -> None:
        if not self._gui_enabled:
            return
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            LOGGER.debug("Failed to close GUI cleanly", exc_info=True)

    def _start_gui_thread(self, stop_event: threading.Event) -> threading.Thread:
        """Start a daemon thread that shows live camera feeds while idle.

        NOTE: On Linux, OpenCV highgui (imshow/waitKey) MUST run on the main
        thread.  Use ``run_gui_loop`` from the main thread instead.  This
        helper is kept only for platforms where a background GUI thread works.
        """
        t = threading.Thread(
            target=self._gui_loop,
            args=(stop_event,),
            daemon=True,
            name="gui-renderer",
        )
        t.start()
        return t

    def run_gui_loop(self, stop_event: threading.Event) -> None:
        """Run the unified GUI loop on the **main thread**.

        On Linux, OpenCV highgui requires the main thread.  Call this instead
        of ``_start_gui_thread`` when running on Linux.  The method returns
        when *stop_event* is set (e.g. via signal handler or 'q' key).
        """
        self._gui_loop(stop_event)

    def _gui_loop(self, stop_event: threading.Event) -> None:
        """Unified GUI thread: renders camera feeds in both idle and running states.

        This is the sole thread that calls ``cv2.imshow`` / ``cv2.waitKey``,
        eliminating cross-thread contention on the OpenCV window.
        """
        if not self._gui_enabled:
            return

        cameras_ready = False
        grabber_top = grabber_left = grabber_right = None
        blank = np.zeros((self.args.wrist_height, self.args.wrist_width, 3), dtype=np.uint8)
        last_vis_ts = time.time()

        while not stop_event.is_set():
            # Lazily connect cameras.
            if not cameras_ready:
                try:
                    grabber_top, grabber_left, grabber_right = self._ensure_camera_resources()
                    required = [g for g in (grabber_top, grabber_left, grabber_right) if g is not None]
                    self._wait_for_required_frames(required, timeout_s=10.0)
                    cameras_ready = True
                    LOGGER.info("GUI thread: cameras connected.")
                except InterruptedError:
                    break
                except Exception as exc:
                    LOGGER.warning("GUI thread: failed to connect cameras: %s", exc)
                    stop_event.wait(timeout=2.0)
                    continue

            top_rgb = grabber_top.get_latest() if grabber_top is not None else None
            if top_rgb is None:
                stop_event.wait(timeout=0.02)
                continue

            left_rgb = grabber_left.get_latest() if grabber_left is not None else blank
            right_rgb = grabber_right.get_latest() if grabber_right is not None else blank

            # Build the 3-panel image.
            panel_images = [
                rgb_wrist._resize_for_panel(left_rgb, self.args.camera_width, self.args.camera_height),
                rgb_wrist._resize_for_panel(top_rgb, self.args.camera_width, self.args.camera_height),
                rgb_wrist._resize_for_panel(right_rgb, self.args.camera_width, self.args.camera_height),
            ]
            panel = np.hstack([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in panel_images])

            for idx, label in enumerate(["Left Wrist RGB", "Top RGB", "Right Wrist RGB"]):
                x = idx * self.args.camera_width + 10
                cv2.putText(panel, label, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Read runtime status to decide overlay.
            now = time.time()
            vis_fps = 1.0 / max(1e-6, now - last_vis_ts)
            last_vis_ts = now

            with self._lock:
                state = self._status.get("state")
                active_model = self._status.get("active_model")
                task = self._status.get("task", "")
                step = self._status.get("step", 0)

            if active_model and state == "running":
                task_label = task if len(task) <= 72 else f"{task[:69]}..."
                cv2.putText(
                    panel,
                    f"Model: {active_model} | Step: {step} | FPS: {vis_fps:.1f}",
                    (10, self.args.camera_height - 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    panel,
                    f"Task: {task_label}",
                    (10, self.args.camera_height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
            elif state == "starting":
                cv2.putText(
                    panel,
                    f"STARTING {active_model or ''}...",
                    (10, self.args.camera_height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 200),
                    2,
                )
            else:
                cv2.putText(
                    panel,
                    f"IDLE - waiting for command | FPS: {vis_fps:.1f}",
                    (10, self.args.camera_height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 255),
                    2,
                )

            cv2.imshow(self._window_name, panel)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                LOGGER.info("Quit requested from GUI window.")
                self._gui_quit_requested.set()
                stop_event.set()
                break
            if key == ord("s"):
                self._save_snapshot(
                    top_rgb=top_rgb,
                    left_wrist_rgb=left_rgb,
                    right_wrist_rgb=right_rgb,
                )

    def _save_snapshot(
        self,
        *,
        top_rgb: np.ndarray,
        left_wrist_rgb: np.ndarray,
        right_wrist_rgb: np.ndarray,
    ) -> None:
        rgb_wrist.save_snapshot(
            save_dir=self._save_dir,
            frame_idx=self._frame_idx,
            top_rgb=top_rgb,
            left_wrist_rgb=left_wrist_rgb,
            right_wrist_rgb=right_wrist_rgb,
        )

    def _smooth_transition_interruptible(
        self,
        client: Any,
        target_action_urdf: np.ndarray,
        duration_s: float,
        frequency_hz: int,
    ) -> bool:
        if duration_s <= 0:
            if self._stop_event.is_set():
                return False
            base.send_action_to_robot(client, target_action_urdf, send_base=False)
            return not self._stop_event.is_set()

        init_joints = base.get_robot_state_urdf(client)[:29]
        target_joints = target_action_urdf[:29]
        total_steps = max(1, int(duration_s * frequency_hz))
        LOGGER.info("Running smooth transition for %.2fs (%d steps).", duration_s, total_steps)

        for step_idx in range(total_steps + 1):
            if self._stop_event.is_set():
                LOGGER.info("Stop requested during smooth transition.")
                return False
            alpha = step_idx / total_steps
            interp_action = init_joints * (1.0 - alpha) + target_joints * alpha
            base.send_action_to_robot(client, interp_action, send_base=False)
            if step_idx < total_steps and self._stop_event.wait(timeout=1.0 / max(1, frequency_hz)):
                LOGGER.info("Stop requested during smooth transition.")
                return False
        return True

    def _resolve_wrist_serials(self) -> tuple[str, str]:
        left_sn = self._shared.left_serial or self.args.wrist_left_serial.strip()
        right_sn = self._shared.right_serial or self.args.wrist_right_serial.strip()

        if self._shared_need_left_wrist or self._shared_need_right_wrist:
            if not left_sn and not right_sn:
                LOGGER.info("Auto-detecting RealSense wrist cameras...")
                left_sn, right_sn = rgb_wrist._auto_detect_realsense_pair()
            elif self._shared_need_left_wrist and not left_sn:
                raise ValueError(
                    "`--wrist-left-serial` is required for checkpoints that use the left wrist camera. "
                    "Leave both wrist serials empty to auto-detect the pair."
                )
            elif self._shared_need_right_wrist and not right_sn:
                raise ValueError(
                    "`--wrist-right-serial` is required for checkpoints that use the right wrist camera. "
                    "Leave both wrist serials empty to auto-detect the pair."
                )

        self._shared.left_serial = left_sn
        self._shared.right_serial = right_sn
        return left_sn, right_sn

    def _close_camera_resources(self) -> None:
        with self._lock:
            shared = self._shared
            grabbers = shared.grabbers()
            shared.grabber_top = None
            shared.grabber_left = None
            shared.grabber_right = None

        for grabber in grabbers:
            try:
                grabber.stop()
            except Exception:
                LOGGER.debug("Failed to stop grabber", exc_info=True)

        with self._lock:
            if shared.orbbec is not None:
                try:
                    shared.orbbec.close()
                except Exception:
                    LOGGER.debug("Failed to close Orbbec camera", exc_info=True)
                finally:
                    shared.orbbec = None

            for attr_name, camera_name in (("cam_left", "left_wrist"), ("cam_right", "right_wrist")):
                cam = getattr(shared, attr_name)
                if cam is not None:
                    try:
                        cam.disconnect()
                    except Exception:
                        LOGGER.warning("Failed to disconnect %s camera.", camera_name, exc_info=True)
                    finally:
                        setattr(shared, attr_name, None)

    def _close_robot_client(self) -> None:
        with self._lock:
            if self._shared.client is None:
                return
            client = self._shared.client
            self._shared.client = None
            self._did_move_to_init_pose = False
            self._last_fsm_state = None
        try:
            client.close()
        except Exception:
            LOGGER.debug("Failed to close robot client", exc_info=True)

    def _ensure_camera_resources(
        self,
    ) -> tuple[
        rgb_wrist.ThreadedFrameGrabber,
        rgb_wrist.ThreadedFrameGrabber | None,
        rgb_wrist.ThreadedFrameGrabber | None,
    ]:
        with self._lock:
            shared = self._shared
            if shared.grabber_top is not None and shared.orbbec is not None:
                LOGGER.info("Reusing shared RGB camera streams.")
                return shared.grabber_top, shared.grabber_left, shared.grabber_right
            if shared.grabber_top is not None or shared.orbbec is not None:
                self._close_camera_resources()

        orbbec = None
        cam_left = None
        cam_right = None
        grabber_top = None
        grabber_left = None
        grabber_right = None

        try:
            orbbec = rgb_wrist.OrbbecRGBCamera(
                width=self.args.camera_width,
                height=self.args.camera_height,
                fps=self.args.camera_fps,
                timeout_ms=self.args.camera_timeout_ms,
                warmup_frames=self.args.camera_warmup_frames,
                init_retries=self.args.camera_init_retries,
                retry_interval_s=self.args.camera_retry_interval_s,
            )
            orbbec.connect()

            left_sn, right_sn = self._resolve_wrist_serials()
            if self._shared_need_left_wrist:
                LOGGER.info("Connecting left wrist RealSense (SN=%s)...", left_sn)
                cam_left = rgb_wrist._create_realsense_camera(
                    left_sn,
                    self.args.wrist_width,
                    self.args.wrist_height,
                    self.args.wrist_fps,
                )
            if self._shared_need_right_wrist:
                LOGGER.info("Connecting right wrist RealSense (SN=%s)...", right_sn)
                cam_right = rgb_wrist._create_realsense_camera(
                    right_sn,
                    self.args.wrist_width,
                    self.args.wrist_height,
                    self.args.wrist_fps,
                )

            grabber_top = rgb_wrist.ThreadedFrameGrabber(orbbec.read, "camera_top")
            if cam_left is not None:
                grabber_left = rgb_wrist.ThreadedFrameGrabber(
                    lambda c=cam_left: rgb_wrist._read_realsense_rgb(c, "left_wrist"),
                    "camera_left_wrist",
                )
            if cam_right is not None:
                grabber_right = rgb_wrist.ThreadedFrameGrabber(
                    lambda c=cam_right: rgb_wrist._read_realsense_rgb(c, "right_wrist"),
                    "camera_right_wrist",
                )

            for grabber in [grabber_top, grabber_left, grabber_right]:
                if grabber is not None:
                    grabber.start()

            with self._lock:
                shared.orbbec = orbbec
                shared.cam_left = cam_left
                shared.cam_right = cam_right
                shared.grabber_top = grabber_top
                shared.grabber_left = grabber_left
                shared.grabber_right = grabber_right
            return grabber_top, grabber_left, grabber_right
        except Exception:
            with self._lock:
                shared.orbbec = orbbec
                shared.cam_left = cam_left
                shared.cam_right = cam_right
                shared.grabber_top = grabber_top
                shared.grabber_left = grabber_left
                shared.grabber_right = grabber_right
            self._close_camera_resources()
            raise

    def _wait_for_required_frames(
        self,
        grabbers: list[rgb_wrist.ThreadedFrameGrabber],
        timeout_s: float = 10.0,
    ) -> None:
        LOGGER.info("Waiting for first frames from required cameras...")
        wait_start = time.time()
        while time.time() - wait_start < timeout_s:
            if all(grabber.get_latest() is not None for grabber in grabbers):
                LOGGER.info("All required RGB cameras are streaming.")
                return
            if self._stop_event.wait(timeout=0.05):
                LOGGER.info("Stop requested while waiting for first frames.")
                raise InterruptedError("stop requested while waiting for first frames")

        missing = [grabber._name for grabber in grabbers if grabber.get_latest() is None]
        self._close_camera_resources()
        raise RuntimeError(f"Timeout waiting for first frame from: {missing}")

    def _switch_robot_fsm_if_needed(self, client: Any, fsm_state: int) -> None:
        # Skip if we already switched to this state.
        if self._last_fsm_state == fsm_state:
            LOGGER.info("FSM state %d unchanged, skipping switch.", fsm_state)
            return

        expected_state_name = base.FSM_STATE_NAME_HINTS.get(int(fsm_state))
        observed_fsm_state = "Unknown"
        upper_fsm_state = "Unknown"

        for attempt in range(1, 7):
            if self._stop_event.is_set():
                raise InterruptedError("stop requested while switching robot FSM")
            client.set_fsm_state(fsm_state)
            time.sleep(0.5)
            try:
                observed_fsm_state = client.get_fsm_state()
                upper_fsm_state = client.get_upper_fsm_state()
            except Exception as exc:
                LOGGER.warning("Failed to query Aurora FSM after switch attempt %d: %s", attempt, exc)
                continue
            if expected_state_name is None or base._fsm_state_matches(observed_fsm_state, expected_state_name):
                break

        LOGGER.info("Aurora FSM after switch: whole=%s upper=%s", observed_fsm_state, upper_fsm_state)
        if expected_state_name is not None and not base._fsm_state_matches(observed_fsm_state, expected_state_name):
            raise RuntimeError(
                "FSM switch failed: expected "
                f"{expected_state_name}(id={fsm_state}) but got {observed_fsm_state}. "
                "请先确认手柄/急停/控制权限状态，再重试；上身控制可优先尝试 --fsm-state 11。"
            )
        self._last_fsm_state = fsm_state

    def _ensure_robot_client(self, run_cfg: WorkerConfig) -> Any | None:
        if self.args.dry_run:
            return None

        with self._lock:
            client = self._shared.client
        try:
            if client is None:
                client = base.setup_robot_if_needed(self._build_robot_args(run_cfg))
                with self._lock:
                    self._shared.client = client
                self._last_fsm_state = run_cfg.fsm_state
            else:
                LOGGER.info("Reusing Aurora client for model '%s'.", run_cfg.model_name)
                self._switch_robot_fsm_if_needed(client, run_cfg.fsm_state)

            if client is not None:
                rgb_wrist.configure_pd_gains(client)
                LOGGER.info("PD gains configured for model '%s'.", run_cfg.model_name)
            return client
        except Exception:
            self._close_robot_client()
            raise

    def _worker_loop(self, run_cfg: WorkerConfig) -> None:
        blank_left_rgb = np.zeros((self.args.wrist_height, self.args.wrist_width, 3), dtype=np.uint8)
        blank_right_rgb = np.zeros((self.args.wrist_height, self.args.wrist_width, 3), dtype=np.uint8)
        algo_args = self._build_algo_args()

        try:
            model = self.models[run_cfg.model_name]
            policy = model.policy
            policy.reset()

            need_left_wrist = any("left_wrist" in key.lower() for key in model.visual_keys)
            need_right_wrist = any("right_wrist" in key.lower() for key in model.visual_keys)

            LOGGER.info(
                "Starting worker for model='%s' task='%s' fps=%.2f fsm_state=%d",
                run_cfg.model_name,
                run_cfg.task,
                run_cfg.fps,
                run_cfg.fsm_state,
            )

            with self._lock:
                self._active_model_name = run_cfg.model_name
                self._status.update(
                    {
                        "ok": True,
                        "state": "running",
                        "message": "running",
                        "active_model": run_cfg.model_name,
                        "last_model": run_cfg.model_name,
                        "task": run_cfg.task,
                        "fps": run_cfg.fps,
                        "fsm_state": run_cfg.fsm_state,
                        "step": 0,
                        "started_at": time.time(),
                        "last_error": "",
                        "last_error_traceback": "",
                        "updated_at": time.time(),
                    }
                )

            grabber_top, grabber_left, grabber_right = self._ensure_camera_resources()
            required_grabbers = [grabber_top]
            if need_left_wrist and grabber_left is not None:
                required_grabbers.append(grabber_left)
            if need_right_wrist and grabber_right is not None:
                required_grabbers.append(grabber_right)
            self._wait_for_required_frames(required_grabbers)

            client = self._ensure_robot_client(run_cfg)
            if client is not None:
                if self.args.move_to_init_pose_on_start and not self._did_move_to_init_pose:
                    LOGGER.info("Moving robot to initial pose before running policy...")
                    if not self._smooth_transition_interruptible(
                        client,
                        rgb_wrist.INIT_POSE_URDF,
                        self.args.init_pose_transition_time_s,
                        self.args.init_pose_transition_freq,
                    ):
                        raise InterruptedError("stop requested during init-pose transition")
                    self._did_move_to_init_pose = True
                elif self.args.move_to_init_pose_on_start:
                    LOGGER.info("Skipping init-pose transition on model switch; initial pose was already reached.")

            step = 0
            first_action = True
            prev_action_urdf: np.ndarray | None = None
            slow_loop_last_log_ts = time.time()
            last_top_id = last_left_id = last_right_id = 0
            action_chunk: np.ndarray | None = None
            chunk_idx = 0
            end_reason = "completed"

            while not self._stop_event.is_set():
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

                has_new_frame = top_rgb is not None
                if grabber_left is not None and left_rgb is not None:
                    has_new_frame = True
                if grabber_right is not None and right_rgb is not None:
                    has_new_frame = True

                latest_top_rgb = grabber_top.get_latest()
                latest_left_rgb = grabber_left.get_latest() if grabber_left is not None else blank_left_rgb
                latest_right_rgb = grabber_right.get_latest() if grabber_right is not None else blank_right_rgb

                if latest_top_rgb is None:
                    if self._stop_event.wait(timeout=0.001):
                        end_reason = "stopped"
                        break
                    continue
                if not has_new_frame and (action_chunk is None or len(action_chunk) == 0):
                    if self._stop_event.wait(timeout=0.001):
                        end_reason = "stopped"
                        break
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

                if self.args.dry_run:
                    state_full = np.zeros((45,), dtype=np.float32)
                else:
                    if client is None:
                        raise RuntimeError("Robot client is not initialized.")
                    state_full = base.get_robot_state_urdf(client)
                state_model = base.fit_vector(state_full, model.state_dim)

                if has_new_frame:
                    action_chunk = rgb_wrist.infer_single_action(
                        policy=policy,
                        preprocessor=model.preprocessor,
                        postprocessor=model.postprocessor,
                        device=self.device,
                        task=run_cfg.task,
                        robot_type=self.args.robot_type,
                        visual_keys=model.visual_keys,
                        state_key=model.state_key,
                        top_rgb=top_rgb,
                        left_wrist_rgb=left_rgb,
                        right_wrist_rgb=right_rgb,
                        state_model=state_model,
                        action_dim=model.action_dim,
                    )
                    chunk_idx = 0
                    action_model = action_chunk[0]
                else:
                    if action_chunk is None or len(action_chunk) == 0:
                        if self._stop_event.wait(timeout=0.001):
                            end_reason = "stopped"
                            break
                        continue
                    chunk_idx += 1
                    if chunk_idx >= len(action_chunk):
                        action_chunk = None
                        continue
                    action_model = action_chunk[chunk_idx]

                raw_action_urdf = base.fit_vector(action_model, 35)
                prev_action_for_diag = prev_action_urdf
                action_urdf = raw_action_urdf.copy()

                if not self.args.disable_clamp:
                    action_urdf = base.clamp_non_hand_joint_action(action_urdf)
                # Apply right hand trigger coupling: fingers = -2 * thumb_pitch
                action_urdf = apply_right_hand_trigger_coupling(action_urdf)
                # Freeze left arm and left hand: overwrite with current state
                action_urdf[LEFT_MANIPULATOR_SLICE] = state_full[LEFT_MANIPULATOR_SLICE]
                action_urdf[LEFT_HAND_SLICE] = state_full[LEFT_HAND_SLICE]
                if prev_action_urdf is not None:
                    action_urdf = base.stabilize_action(action_urdf, prev_action_urdf, algo_args)

                if not self.args.dry_run:
                    if client is None:
                        raise RuntimeError("Robot client is not initialized.")
                    if first_action:
                        # Use shorter transition on model switch since robot is
                        # already in a reasonable pose from the previous run.
                        t_s = (
                            self.args.switch_transition_time_s
                            if self._had_previous_run
                            else self.args.transition_time_s
                        )
                        if t_s > 0:
                            if not self._smooth_transition_interruptible(
                                client, action_urdf, t_s, self.args.transition_freq,
                            ):
                                end_reason = "stopped"
                                break
                            policy.reset()
                            action_chunk = None
                            chunk_idx = 0
                        first_action = False
                    else:
                        base.send_action_to_robot(client, action_urdf, self.args.send_base)
                        first_action = False

                if step % max(1, self.args.log_every) == 0:
                    arm_dim = min(14, state_full.shape[0], action_urdf.shape[0])
                    arm_delta = float(np.linalg.norm(action_urdf[:arm_dim] - state_full[:arm_dim]))
                    base.log_action_diagnostics(
                        step=step,
                        arm_delta=arm_delta,
                        depth_range="rgb-only",
                        raw_action_urdf=raw_action_urdf,
                        final_action_urdf=action_urdf,
                        prev_action_urdf=prev_action_for_diag,
                        args=algo_args,
                    )
                    self._mark(step=step)

                if self._gui_quit_requested.is_set():
                    end_reason = "gui quit requested"
                    break

                # Periodic save when GUI is disabled (GUI thread handles 's' key).
                if (
                    not self._gui_enabled
                    and self._periodic_save_every > 0
                    and step % self._periodic_save_every == 0
                ):
                    self._save_snapshot(
                        top_rgb=top_rgb,
                        left_wrist_rgb=left_rgb,
                        right_wrist_rgb=right_rgb,
                    )

                prev_action_urdf = action_urdf.copy()
                step += 1

                if run_cfg.max_steps > 0 and step >= run_cfg.max_steps:
                    end_reason = "max_steps reached"
                    break

                elapsed = time.perf_counter() - loop_start
                if self.args.slow_loop_warn_ms > 0 and elapsed * 1000.0 > self.args.slow_loop_warn_ms:
                    now_ts = time.time()
                    if now_ts - slow_loop_last_log_ts > 2.0:
                        LOGGER.warning(
                            "Slow loop for model '%s': %.1f ms (target %.1f ms).",
                            run_cfg.model_name,
                            elapsed * 1000.0,
                            1000.0 / max(1e-6, run_cfg.fps),
                        )
                        slow_loop_last_log_ts = now_ts

                sleep_s = max(0.0, 1.0 / max(1e-6, run_cfg.fps) - elapsed)
                if self._stop_event.wait(timeout=sleep_s):
                    end_reason = "stopped"
                    break

            LOGGER.info("Worker finished for model '%s': %s (step=%d)", run_cfg.model_name, end_reason, step)
            self._mark(
                ok=True,
                state="idle",
                message=end_reason,
                active_model=None,
                step=step,
            )
        except InterruptedError:
            LOGGER.info("Worker startup/loop interrupted for model '%s'.", run_cfg.model_name)
            self._mark(
                ok=True,
                state="idle",
                message="stopped",
                active_model=None,
            )
        except Exception as exc:
            LOGGER.exception("Worker failed for model '%s': %s", run_cfg.model_name, exc)
            self._mark(
                ok=False,
                state="error",
                message=f"worker failed for model {run_cfg.model_name}",
                active_model=None,
                last_error=str(exc),
                last_error_traceback=traceback.format_exc(),
            )
        finally:
            # Only reset the policy that was running, not all policies.
            try:
                self.models[run_cfg.model_name].policy.reset()
            except Exception:
                LOGGER.debug("Failed to reset policy '%s'", run_cfg.model_name, exc_info=True)
            self._had_previous_run = True
            with self._lock:
                if (
                    self._status.get("state") not in {"idle", "error"}
                    and self._status.get("active_model") == run_cfg.model_name
                ):
                    self._status.update(
                        {
                            "ok": True,
                            "state": "idle",
                            "message": "stopped" if self._stop_event.is_set() else "worker exited",
                            "active_model": None,
                            "updated_at": time.time(),
                        }
                    )
                self._worker = None
                self._active_model_name = None
            self._stop_event.clear()

    def _build_worker_config(self, model_name: str, payload: dict[str, Any]) -> WorkerConfig:
        model = self.models[model_name]
        task = str(payload.get("task", model.spec.task)).strip() or model.spec.task
        max_steps = int(payload.get("max_steps", self.args.max_steps))
        fps = float(payload.get("fps", self.args.fps))
        fsm_state = int(payload.get("fsm_state", self.args.fsm_state))
        return WorkerConfig(
            model_name=model_name,
            task=task,
            max_steps=max_steps,
            fps=fps,
            fsm_state=fsm_state,
        )

    def start_model(self, model_name: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        payload = payload or {}
        if model_name not in self.models:
            return HTTPStatus.NOT_FOUND, {
                "ok": False,
                "message": f"unknown model: {model_name}",
                "status": self.status(),
            }

        restart = _coerce_bool(payload.get("restart", False), default=False)
        stop_timeout_s = float(payload.get("stop_timeout_s", self.args.switch_stop_timeout_s))

        with self._lock:
            worker = self._worker
            active_model = self._active_model_name

        if worker is not None and worker.is_alive():
            if active_model == model_name and not restart:
                return HTTPStatus.CONFLICT, {
                    "ok": False,
                    "message": f"model '{model_name}' is already running",
                    "status": self.status(),
                }
            LOGGER.info(
                "Switch requested from model '%s' to '%s'. Stopping current worker first.",
                active_model,
                model_name,
            )
            _stop_code, stop_data = self.stop(timeout_s=stop_timeout_s)
            if stop_data.get("status", {}).get("worker_alive"):
                return HTTPStatus.CONFLICT, {
                    "ok": False,
                    "message": "failed to stop current worker before switching models",
                    "status": self.status(),
                }

        run_cfg = self._build_worker_config(model_name, payload)
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return HTTPStatus.CONFLICT, {
                    "ok": False,
                    "message": "another worker started while processing the request",
                    "status": self.status(),
                }
            self._stop_event.clear()
            self._status.update(
                {
                    "ok": True,
                    "state": "starting",
                    "message": f"starting model '{model_name}'",
                    "active_model": model_name,
                    "last_model": model_name,
                    "task": run_cfg.task,
                    "fps": run_cfg.fps,
                    "fsm_state": run_cfg.fsm_state,
                    "step": 0,
                    "started_at": time.time(),
                    "last_error": "",
                    "last_error_traceback": "",
                    "updated_at": time.time(),
                }
            )
            self._worker = threading.Thread(
                target=self._worker_loop,
                args=(run_cfg,),
                daemon=True,
                name=f"pi0-worker-{model_name}",
            )
            self._worker.start()

        return HTTPStatus.ACCEPTED, {
            "ok": True,
            "message": f"started model '{model_name}'",
            "status": self.status(),
        }

    def stop(self, timeout_s: float = 5.0) -> tuple[int, dict[str, Any]]:
        with self._lock:
            worker = self._worker
            if worker is None or not worker.is_alive():
                return HTTPStatus.OK, {
                    "ok": True,
                    "message": "service idle",
                    "status": self.status(),
                }
            self._status.update(
                {
                    "state": "stopping",
                    "message": "stop requested",
                    "updated_at": time.time(),
                }
            )
            self._stop_event.set()

        worker.join(timeout=max(0.1, timeout_s))
        still_alive = worker.is_alive()
        if not still_alive:
            self._reset_all_policies()

        return HTTPStatus.OK, {
            "ok": not still_alive,
            "message": "stopped" if not still_alive else "stop requested, worker still shutting down",
            "status": self.status(),
        }

    def set_pd(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """热更新 PD 增益，不需要重启 server。

        payload 示例（可只传部分 group）:
        {
            "kp": {"right_manipulator": [270, 250, 95, 95, 45, 45, 45]},
            "kd": {"right_manipulator": [10, 10, 5, 5, 5, 5, 5]},
            "apply": true                 # 默认 true，立即下发到机器人
        }
        """
        kp_patch = payload.get("kp", {})
        kd_patch = payload.get("kd", {})
        if not kp_patch and not kd_patch:
            return HTTPStatus.BAD_REQUEST, {
                "ok": False,
                "message": "至少提供 kp 或 kd 中的一项",
            }

        apply_now = _coerce_bool(payload.get("apply", True), default=True)

        applied = False
        if apply_now:
            with self._lock:
                client = self._shared.client
            if client is not None:
                if kp_patch or kd_patch:
                    client.set_motor_cfg(kp_config=kp_patch or {}, kd_config=kd_patch or {})
                applied = True
                LOGGER.info("PD gains applied to robot. kp_groups=%s kd_groups=%s",
                            list(kp_patch.keys()), list(kd_patch.keys()))
            else:
                LOGGER.warning("No robot client connected, PD gains not applied.")

        return HTTPStatus.OK, {
            "ok": True,
            "message": "PD gains updated" + (" and applied" if applied else " (not applied, no robot client)"),
            "applied": applied,
            "kp_groups_updated": list(kp_patch.keys()),
            "kd_groups_updated": list(kd_patch.keys()),
        }

    def get_pd(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """查询当前 PD 增益（直接从机器人读取不再可用，仅返回状态）。"""
        return HTTPStatus.OK, {
            "ok": True,
            "message": "PD gains are managed by rgb_wrist defaults; per-model presets removed.",
        }

    def shutdown(self) -> None:
        self.stop(timeout_s=5.0)
        with self._lock:
            worker = self._worker
        if worker is None or not worker.is_alive():
            self._close_camera_resources()
            self._close_robot_client()
            self._close_visualization()

    def handle_command(self, method: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        payload = payload or {}
        normalized_method = str(method).strip().lower()
        LOGGER.info("IPC request: method=%s payload=%s", normalized_method, payload)

        if normalized_method == "health":
            return HTTPStatus.OK, self.health()
        if normalized_method == "status":
            return HTTPStatus.OK, self.status()
        if normalized_method == "stop":
            timeout_s = float(payload.get("timeout_s", 5.0))
            return self.stop(timeout_s=timeout_s)
        if normalized_method in {"start_green_to_yellow", "start_greentoyellow"}:
            return self.start_model(MODEL_GREEN_TO_YELLOW, payload)
        if normalized_method in {"start_yellow_to_green", "start_yellowtogreen"}:
            return self.start_model(MODEL_YELLOW_TO_GREEN, payload)
        if normalized_method == "start_model":
            model_name = str(payload.get("model_name", payload.get("model", ""))).strip().lower()
            return self.start_model(model_name, payload)
        if normalized_method in {"set_pd", "setpd", "update_pd"}:
            return self.set_pd(payload)
        if normalized_method in {"get_pd", "getpd"}:
            return self.get_pd(payload)

        return HTTPStatus.NOT_FOUND, {
            "ok": False,
            "message": f"unknown method: {method}",
        }


class ThreadingUnixSocketServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class DualPi0UnixRequestHandler(socketserver.StreamRequestHandler):
    runtime: DualModelInferenceRuntime | None = None

    def _safe_write_json(self, out: dict[str, Any]) -> None:
        try:
            self.wfile.write((json.dumps(out, ensure_ascii=False) + "\n").encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.warning("IPC client disconnected before response was sent.")

    def handle(self) -> None:
        runtime = self.runtime
        if runtime is None:
            self._safe_write_json(
                {
                    "code": int(HTTPStatus.INTERNAL_SERVER_ERROR),
                    "data": {"ok": False, "message": "runtime missing"},
                }
            )
            return

        raw = self.rfile.readline(1024 * 1024).decode("utf-8").strip()
        if not raw:
            self._safe_write_json(
                {
                    "code": int(HTTPStatus.BAD_REQUEST),
                    "data": {"ok": False, "message": "empty request"},
                }
            )
            return

        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            method = str(request.get("method", "")).strip()
            payload = request.get("payload", {})
            if not isinstance(payload, dict):
                raise ValueError("payload must be a JSON object")
            code, data = runtime.handle_command(method, payload)
            response = {"code": int(code), "data": data}
        except Exception as exc:
            LOGGER.exception("Unix socket request handling failed: %s", exc)
            response = {
                "code": int(HTTPStatus.INTERNAL_SERVER_ERROR),
                "data": {
                    "ok": False,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        self._safe_write_json(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-model PI0 RGB wrist inference server for GR2")
    parser.add_argument("--unix-socket-path", type=str, default=DEFAULT_UNIX_SOCKET_PATH)

    parser.add_argument("--green-to-yellow-checkpoint-path", type=str, default=DEFAULT_GREEN_TO_YELLOW_CHECKPOINT_PATH)
    parser.add_argument("--green-to-yellow-task", type=str, default=DEFAULT_GREEN_TO_YELLOW_TASK)
    parser.add_argument("--yellow-to-green-checkpoint-path", type=str, default=DEFAULT_YELLOW_TO_GREEN_CHECKPOINT_PATH)
    parser.add_argument("--yellow-to-green-task", type=str, default=DEFAULT_YELLOW_TO_GREEN_TASK)

    parser.add_argument("--robot-type", type=str, default=base.DEFAULT_ROBOT_TYPE)
    parser.add_argument("--domain-id", type=int, default=123)
    parser.add_argument("--robot-name", type=str, default="gr2")
    parser.add_argument("--fsm-state", type=int, default=11)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--camera-key", type=str, default="observation.images.camera_top")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-timeout-ms", type=int, default=200)
    parser.add_argument("--camera-warmup-frames", type=int, default=15)
    parser.add_argument("--camera-init-retries", type=int, default=6)
    parser.add_argument("--camera-retry-interval-s", type=float, default=1.0)

    parser.add_argument("--wrist-left-serial", type=str, default=rgb_wrist.DEFAULT_LEFT_WRIST_SERIAL)
    parser.add_argument("--wrist-right-serial", type=str, default=rgb_wrist.DEFAULT_RIGHT_WRIST_SERIAL)
    parser.add_argument("--wrist-width", type=int, default=640)
    parser.add_argument("--wrist-height", type=int, default=480)
    parser.add_argument("--wrist-fps", type=int, default=30)

    parser.add_argument("--state-dim", type=int, default=0)
    parser.add_argument("--action-dim", type=int, default=0)
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--save-dir", type=str, default="scripts/outputs/gr2_dual_pi0_rgb_wrist")
    parser.add_argument("--save-every", type=int, default=0)

    parser.add_argument("--client-init-retries", type=int, default=10)
    parser.add_argument("--client-retry-interval-s", type=float, default=2.0)
    parser.add_argument("--switch-stop-timeout-s", type=float, default=5.0)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-clamp", action="store_true")
    parser.add_argument("--send-base", action="store_true")
    parser.add_argument("--transition-time-s", type=float, default=0.0)
    parser.add_argument("--switch-transition-time-s", type=float, default=0.0,
                        help="Shorter transition time used when switching between models (default: 0.0s)")
    parser.add_argument("--transition-freq", type=int, default=100)

    parser.add_argument(
        "--move-to-init-pose-on-start",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--init-pose-transition-time-s", type=float, default=3.0)
    parser.add_argument("--init-pose-transition-freq", type=int, default=100)

    parser.add_argument("--action-ema-alpha", type=float, default=0.35)
    parser.add_argument("--arm-ema-alpha", type=float, default=-1.0)
    parser.add_argument("--hand-ema-alpha", type=float, default=-1.0)
    parser.add_argument("--max-arm-delta", type=float, default=0.06)
    parser.add_argument("--max-hand-delta", type=float, default=0.12)
    parser.add_argument("--max-head-waist-delta", type=float, default=0.08)
    parser.add_argument("--max-base-delta", type=float, default=0.15)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--slow-loop-warn-ms", type=float, default=120.0)
    parser.add_argument("--log-level", type=str, default="INFO")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s | %(levelname)s | %(message)s", force=True)

    runtime = DualModelInferenceRuntime(args)
    DualPi0UnixRequestHandler.runtime = runtime

    socket_path = Path(args.unix_socket_path).expanduser().resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    server = ThreadingUnixSocketServer(str(socket_path), DualPi0UnixRequestHandler)
    server_thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.5},
        daemon=True,
        name="dual-pi0-unix-server",
    )
    server_thread.start()

    LOGGER.info("Unix transport ready at %s", socket_path)
    LOGGER.info(
        "Dual PI0 inference server started | green_to_yellow=%s | yellow_to_green=%s",
        args.green_to_yellow_checkpoint_path,
        args.yellow_to_green_checkpoint_path,
    )

    stop_event = threading.Event()

    def _handle_stop(signum, frame) -> None:
        del frame
        LOGGER.warning("Received signal %s, shutting down service...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    if hasattr(signal, "SIGTSTP"):
        signal.signal(signal.SIGTSTP, signal.SIG_IGN)

    # Run the GUI loop on the main thread (required by OpenCV on Linux).
    # When --no-gui, fall back to a simple sleep loop.
    try:
        if runtime._gui_enabled:
            runtime.run_gui_loop(stop_event)
        else:
            while not stop_event.is_set():
                time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        runtime.shutdown()
        try:
            server.shutdown()
        except Exception:
            LOGGER.debug("Failed to shutdown server cleanly", exc_info=True)
        try:
            server.server_close()
        except Exception:
            LOGGER.debug("Failed to close server cleanly", exc_info=True)
        try:
            if socket_path.exists():
                socket_path.unlink()
        except Exception:
            LOGGER.debug("Failed to remove socket path cleanly", exc_info=True)
        LOGGER.info("Dual PI0 inference server stopped.")


if __name__ == "__main__":
    main()
