#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""GR2 PI0 inference service (Unix socket IPC only).

This service preloads PI0 model on startup and accepts local IPC requests
through a Unix domain socket. Supported commands:
- health
- status
- start
- stop
- reload
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import socketserver
import threading
import time
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

import numpy as np

import deploy_gr2_pi0 as base
import deploy_gr2_pi0_rgbd as rgbd
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

LOGGER = logging.getLogger("gr2_pi0_inference_service")

DEFAULT_CHECKPOINT_PATH = (
    "/home/phl/workspace/lerobot-versions/lerobot/outputs/train/"
    "pi0_gr2_pick_3_4_20260306_185911/checkpoints/070000/pretrained_model"
)
DEFAULT_TASK = "pick bottle and place into box"
DEFAULT_ROBOT_TYPE = "fourier_gr2"
DEFAULT_UNIX_SOCKET_PATH = "/tmp/gr2_pi0_inference_service.sock"


@dataclass
class WorkerConfig:
    task: str
    max_steps: int
    fps: float
    fsm_state: int


class InferenceRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

        self.device = None
        self.pretrained_dir: Path | None = None
        self.policy: PI0Policy | None = None
        self.preprocessor = None
        self.postprocessor = None
        self.state_key = "observation.state"
        self.state_dim = 45
        self.action_dim = 35
        self.visual_keys: list[str] = [args.camera_key]

        self._status: dict[str, Any] = {
            "ok": True,
            "state": "loading",
            "model_loaded": False,
            "checkpoint_path": args.checkpoint_path,
            "task": args.task,
            "fps": args.fps,
            "fsm_state": args.fsm_state,
            "step": 0,
            "started_at": None,
            "updated_at": time.time(),
            "last_error": "",
            "last_error_traceback": "",
            "message": "initializing",
        }

        self.reload_model(args.checkpoint_path)

    def _mark(self, **kwargs: Any) -> None:
        with self._lock:
            self._status.update(kwargs)
            self._status["updated_at"] = time.time()

    def status(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self._status)
            out["worker_alive"] = bool(self._worker and self._worker.is_alive())
            return out

    def health(self) -> dict[str, Any]:
        st = self.status()
        return {
            "ok": bool(st.get("model_loaded")),
            "state": st.get("state"),
            "model_loaded": st.get("model_loaded"),
            "checkpoint_path": st.get("checkpoint_path"),
            "worker_alive": st.get("worker_alive"),
            "message": st.get("message", ""),
        }

    def _load_model(self, checkpoint_path: str) -> None:
        pretrained_dir = base.resolve_pretrained_model_dir(checkpoint_path)
        device = base.select_device(self.args.device)

        LOGGER.info("Loading PI0 model from %s on %s", pretrained_dir, device)
        policy = PI0Policy.from_pretrained(str(pretrained_dir), strict=False).to(device)
        policy.eval()

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(pretrained_dir),
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

        state_key, cfg_state_dim, visual_keys, _action_key, cfg_action_dim = base.infer_policy_io(policy)
        state_dim = self.args.state_dim if self.args.state_dim > 0 else cfg_state_dim
        action_dim = self.args.action_dim if self.args.action_dim > 0 else cfg_action_dim
        ordered_visual_keys = rgbd._ordered_visual_keys(visual_keys, self.args.camera_key)

        rgbd.warmup_policy_multivis(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            task=self.args.task,
            robot_type=self.args.robot_type,
            visual_keys=ordered_visual_keys,
            state_key=state_key,
            camera_height=self.args.camera_height,
            camera_width=self.args.camera_width,
            state_dim=state_dim,
        )
        policy.reset()

        self.pretrained_dir = pretrained_dir
        self.device = device
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.state_key = state_key
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.visual_keys = ordered_visual_keys

    def reload_model(self, checkpoint_path: str) -> dict[str, Any]:
        with self._lock:
            if self._worker and self._worker.is_alive():
                raise RuntimeError("Cannot reload model while worker is running.")
            self._status.update(
                {
                    "state": "loading",
                    "model_loaded": False,
                    "checkpoint_path": checkpoint_path,
                    "message": "loading model",
                    "last_error": "",
                    "last_error_traceback": "",
                    "updated_at": time.time(),
                }
            )

        try:
            self._load_model(checkpoint_path)
        except Exception as exc:
            LOGGER.exception("Model load failed: %s", exc)
            self._mark(
                ok=False,
                state="error",
                model_loaded=False,
                last_error=str(exc),
                last_error_traceback=traceback.format_exc(),
                message="model load failed",
            )
            raise

        self.args.checkpoint_path = checkpoint_path
        self._mark(
            ok=True,
            state="idle",
            model_loaded=True,
            checkpoint_path=checkpoint_path,
            message="model loaded",
            last_error="",
            last_error_traceback="",
            step=0,
        )
        return self.status()

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

    def _worker_loop(self, run_cfg: WorkerConfig) -> None:
        camera = None
        client = None
        algo_args = self._build_algo_args()

        LOGGER.info(
            "Worker starting: task=%s fps=%.2f fsm_state=%d max_steps=%d dry_run=%s",
            run_cfg.task,
            run_cfg.fps,
            run_cfg.fsm_state,
            run_cfg.max_steps,
            self.args.dry_run,
        )

        with self._lock:
            self._status.update(
                {
                    "state": "running",
                    "task": run_cfg.task,
                    "fps": run_cfg.fps,
                    "fsm_state": run_cfg.fsm_state,
                    "step": 0,
                    "started_at": time.time(),
                    "message": "running",
                    "last_error": "",
                    "last_error_traceback": "",
                    "updated_at": time.time(),
                }
            )

        try:
            if self.policy is None or self.preprocessor is None or self.postprocessor is None or self.device is None:
                raise RuntimeError("Model is not loaded.")

            LOGGER.info("Initializing camera stream...")
            camera = base.OrbbecRGBDCamera(
                width=self.args.camera_width,
                height=self.args.camera_height,
                fps=self.args.camera_fps,
                timeout_ms=self.args.camera_timeout_ms,
                warmup_frames=self.args.camera_warmup_frames,
                init_retries=self.args.camera_init_retries,
                retry_interval_s=self.args.camera_retry_interval_s,
            )
            camera.connect()

            robot_args = self._build_robot_args(run_cfg)
            LOGGER.info(
                "Initializing robot client: domain_id=%s robot_name=%s retries=%s",
                robot_args.domain_id,
                robot_args.robot_name,
                robot_args.client_init_retries,
            )
            client = base.setup_robot_if_needed(robot_args)
            LOGGER.info("Robot client initialized.")

            policy = self.policy
            policy.reset()

            step = 0
            first_action = True
            prev_action_urdf: np.ndarray | None = None
            no_frame_count = 0
            no_frame_last_log_ts = time.time()
            slow_loop_last_log_ts = time.time()
            end_reason = "completed"

            while not self._stop_event.is_set():
                loop_start = time.perf_counter()

                try:
                    frame = camera.read()
                except Exception as exc:
                    LOGGER.warning("Camera read failed: %s. Reconnecting...", exc)
                    camera.close()
                    if self._stop_event.is_set():
                        break
                    camera.connect()
                    continue

                if frame is None:
                    no_frame_count += 1
                    now_ts = time.time()
                    if now_ts - no_frame_last_log_ts > 2.0:
                        LOGGER.warning("No camera frame yet (count=%d).", no_frame_count)
                        no_frame_last_log_ts = now_ts
                    if self._stop_event.wait(timeout=0.01):
                        end_reason = "stopped"
                        break
                    continue

                no_frame_count = 0
                rgb, depth_rgb, depth_u16, _depth_vis_bgr = frame

                if self.args.dry_run:
                    state_full = np.zeros((45,), dtype=np.float32)
                else:
                    if client is None:
                        raise RuntimeError("Robot client is not initialized.")
                    state_full = base.get_robot_state_urdf(client)

                state_model = base.fit_vector(state_full, self.state_dim)
                action_model = rgbd.infer_single_action_multivis(
                    policy=policy,
                    preprocessor=self.preprocessor,
                    postprocessor=self.postprocessor,
                    device=self.device,
                    task=run_cfg.task,
                    robot_type=self.args.robot_type,
                    visual_keys=self.visual_keys,
                    state_key=self.state_key,
                    rgb=rgb,
                    depth_rgb=depth_rgb,
                    state_model=state_model,
                    action_dim=self.action_dim,
                )

                raw_action_urdf = base.fit_vector(action_model, 35)
                prev_action_for_diag = prev_action_urdf
                action_urdf = raw_action_urdf.copy()

                if not self.args.disable_clamp:
                    action_urdf = base.clamp_non_hand_joint_action(action_urdf)
                if prev_action_urdf is not None:
                    action_urdf = base.stabilize_action(action_urdf, prev_action_urdf, algo_args)

                if not self.args.dry_run:
                    if client is None:
                        raise RuntimeError("Robot client is not initialized.")
                    if first_action and self.args.transition_time_s > 0:
                        base.smooth_transition(client, action_urdf, self.args.transition_time_s, self.args.transition_freq)
                        policy.reset()
                        first_action = False
                    else:
                        base.send_action_to_robot(client, action_urdf, self.args.send_base)
                        first_action = False

                if step % max(1, self.args.log_every) == 0:
                    arm_dim = min(14, state_full.shape[0], action_urdf.shape[0])
                    arm_delta = float(np.linalg.norm(action_urdf[:arm_dim] - state_full[:arm_dim]))
                    valid_depth = depth_u16[depth_u16 > 0]
                    depth_range = "empty" if valid_depth.size == 0 else f"{int(valid_depth.min())}..{int(valid_depth.max())}"
                    base.log_action_diagnostics(
                        step,
                        arm_delta,
                        depth_range,
                        raw_action_urdf,
                        action_urdf,
                        prev_action_for_diag,
                        algo_args,
                    )

                prev_action_urdf = action_urdf.copy()
                step += 1

                if step % max(1, self.args.log_every) == 0:
                    self._mark(step=step)

                if run_cfg.max_steps > 0 and step >= run_cfg.max_steps:
                    end_reason = "max_steps reached"
                    break

                elapsed = time.perf_counter() - loop_start
                if self.args.slow_loop_warn_ms > 0 and elapsed * 1000.0 > self.args.slow_loop_warn_ms:
                    now_ts = time.time()
                    if now_ts - slow_loop_last_log_ts > 2.0:
                        LOGGER.warning(
                            "Slow control loop: %.1f ms (target %.1f ms).",
                            elapsed * 1000.0,
                            1000.0 / max(1e-6, run_cfg.fps),
                        )
                        slow_loop_last_log_ts = now_ts
                sleep_s = max(0.0, 1.0 / max(1e-6, run_cfg.fps) - elapsed)
                if self._stop_event.wait(timeout=sleep_s):
                    end_reason = "stopped"
                    break

            LOGGER.info("Worker finished: %s (step=%d)", end_reason, step)
            self._mark(ok=True, state="idle", message=end_reason, step=step)
        except Exception as exc:
            LOGGER.exception("Worker failed: %s", exc)
            self._mark(
                ok=False,
                state="error",
                message="worker failed",
                last_error=str(exc),
                last_error_traceback=traceback.format_exc(),
            )
        finally:
            try:
                if camera is not None:
                    camera.close()
            except Exception:
                pass
            try:
                if client is not None:
                    client.close()
            except Exception:
                pass

            with self._lock:
                self._worker = None
            self._stop_event.clear()

    def start(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        task = str(payload.get("task", self.args.task)).strip() or self.args.task
        max_steps = int(payload.get("max_steps", self.args.max_steps))
        fps = float(payload.get("fps", self.args.fps))
        fsm_state = int(payload.get("fsm_state", self.args.fsm_state))
        LOGGER.info(
            "Start requested: task='%s' max_steps=%d fps=%.2f fsm_state=%d",
            task,
            max_steps,
            fps,
            fsm_state,
        )

        with self._lock:
            if not self._status.get("model_loaded", False):
                return HTTPStatus.SERVICE_UNAVAILABLE, {
                    "ok": False,
                    "message": "model not loaded",
                    "status": self.status(),
                }

            if self._worker and self._worker.is_alive():
                return HTTPStatus.CONFLICT, {
                    "ok": False,
                    "message": "inference already running",
                    "status": self.status(),
                }

            run_cfg = WorkerConfig(task=task, max_steps=max_steps, fps=fps, fsm_state=fsm_state)
            self._stop_event.clear()
            self._worker = threading.Thread(target=self._worker_loop, args=(run_cfg,), daemon=True, name="pi0-worker")
            self._worker.start()

        return HTTPStatus.ACCEPTED, {
            "ok": True,
            "message": "inference started",
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
            self._status.update({"state": "stopping", "message": "stop requested", "updated_at": time.time()})
            self._stop_event.set()

        worker.join(timeout=max(0.1, timeout_s))
        still_alive = worker.is_alive()
        return HTTPStatus.OK, {
            "ok": not still_alive,
            "message": "stopped" if not still_alive else "stop requested, worker still shutting down",
            "status": self.status(),
        }

    def shutdown(self) -> None:
        self.stop(timeout_s=5.0)

    def handle_command(self, method: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        payload = payload or {}
        method = str(method).strip().lower()
        LOGGER.info("IPC request: method=%s payload=%s", method, payload)

        if method == "health":
            return HTTPStatus.OK, self.health()
        if method == "status":
            return HTTPStatus.OK, self.status()
        if method == "start":
            return self.start(payload)
        if method == "stop":
            timeout_s = float(payload.get("timeout_s", 5.0))
            return self.stop(timeout_s=timeout_s)
        if method == "reload":
            checkpoint_path = str(payload.get("checkpoint_path", self.args.checkpoint_path)).strip()
            if not checkpoint_path:
                checkpoint_path = self.args.checkpoint_path
            self.reload_model(checkpoint_path)
            return HTTPStatus.OK, {"ok": True, "message": "model reloaded", "status": self.status()}

        return HTTPStatus.NOT_FOUND, {"ok": False, "message": f"unknown method: {method}"}


class ThreadingUnixSocketServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class Pi0UnixRequestHandler(socketserver.StreamRequestHandler):
    runtime: InferenceRuntime | None = None

    def _safe_write_json(self, out: dict[str, Any]) -> None:
        try:
            self.wfile.write((json.dumps(out, ensure_ascii=False) + "\n").encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.warning("IPC client disconnected before response was sent.")

    def handle(self) -> None:
        runtime = self.runtime
        if runtime is None:
            out = {"code": int(HTTPStatus.INTERNAL_SERVER_ERROR), "data": {"ok": False, "message": "runtime missing"}}
            self._safe_write_json(out)
            return

        raw = self.rfile.readline(1024 * 1024).decode("utf-8").strip()
        if not raw:
            out = {"code": int(HTTPStatus.BAD_REQUEST), "data": {"ok": False, "message": "empty request"}}
            self._safe_write_json(out)
            return

        try:
            req = json.loads(raw)
            if not isinstance(req, dict):
                raise ValueError("request must be JSON object")
            method = str(req.get("method", "")).strip()
            payload = req.get("payload", {})
            if not isinstance(payload, dict):
                raise ValueError("payload must be JSON object")
            code, data = runtime.handle_command(method, payload)
            out = {"code": int(code), "data": data}
        except Exception as exc:
            LOGGER.exception("Unix socket request handling failed: %s", exc)
            out = {
                "code": int(HTTPStatus.INTERNAL_SERVER_ERROR),
                "data": {"ok": False, "message": str(exc), "traceback": traceback.format_exc()},
            }
        self._safe_write_json(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GR2 PI0 inference Unix socket service")
    parser.add_argument("--unix-socket-path", type=str, default=DEFAULT_UNIX_SOCKET_PATH)

    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--robot-type", type=str, default=DEFAULT_ROBOT_TYPE)
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

    parser.add_argument("--state-dim", type=int, default=0)
    parser.add_argument("--action-dim", type=int, default=0)

    parser.add_argument("--client-init-retries", type=int, default=10)
    parser.add_argument("--client-retry-interval-s", type=float, default=2.0)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-clamp", action="store_true")
    parser.add_argument("--send-base", action="store_true")
    parser.add_argument("--transition-time-s", type=float, default=0.0)
    parser.add_argument("--transition-freq", type=int, default=100)

    parser.add_argument("--action-ema-alpha", type=float, default=0.6)
    parser.add_argument("--arm-ema-alpha", type=float, default=-1.0)
    parser.add_argument("--hand-ema-alpha", type=float, default=0.95)
    parser.add_argument("--max-arm-delta", type=float, default=0.10)
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

    runtime = InferenceRuntime(args)
    Pi0UnixRequestHandler.runtime = runtime

    socket_path = Path(args.unix_socket_path).expanduser().resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    server = ThreadingUnixSocketServer(str(socket_path), Pi0UnixRequestHandler)
    server_thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.5},
        daemon=True,
        name="pi0-unix-server",
    )
    server_thread.start()

    LOGGER.info("Unix transport ready at %s", socket_path)
    LOGGER.info("PI0 inference service started (unix socket only, model=%s)", args.checkpoint_path)

    stop_event = threading.Event()

    def _handle_stop(signum, frame) -> None:
        del frame
        LOGGER.warning("Received signal %s, shutting down service...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        runtime.shutdown()
        try:
            server.shutdown()
        except Exception:
            pass
        try:
            server.server_close()
        except Exception:
            pass
        try:
            if socket_path.exists():
                socket_path.unlink()
        except Exception:
            pass
        LOGGER.info("PI0 inference service stopped.")


if __name__ == "__main__":
    main()
