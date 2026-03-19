#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Print PI0 model outputs with readable formatting (RGB + depth + state)."""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch

import deploy_gr2_pi0 as base
import deploy_gr2_pi0_rgbd as rgbd
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import prepare_observation_for_inference

LOGGER = logging.getLogger(__name__)

# Same 35D action layout as deploy scripts.
ACTION_GROUPS = [
    ("left_arm", slice(0, 7)),
    ("right_arm", slice(7, 14)),
    ("left_hand", slice(14, 20)),
    ("right_hand", slice(20, 26)),
    ("head", slice(26, 28)),
    ("waist", slice(28, 29)),
    ("base", slice(29, 35)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print readable PI0 model outputs without sending actions.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=(
            "/home/phl/workspace/lerobot-versions/lerobot/outputs/train/"
            "pi0_gr2_pick_3_4_20260306_185911/checkpoints/070000/pretrained_model"
        ),
        help="Checkpoint root or pretrained_model directory.",
    )
    parser.add_argument("--task", type=str, default="pick bottle and place into box")
    parser.add_argument("--robot-type", type=str, default="fourier_gr2")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--camera-key", type=str, default="observation.images.camera_top")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-timeout-ms", type=int, default=200)
    parser.add_argument("--camera-warmup-frames", type=int, default=15)
    parser.add_argument("--camera-init-retries", type=int, default=6)
    parser.add_argument("--camera-retry-interval-s", type=float, default=1.0)

    parser.add_argument("--domain-id", type=int, default=123)
    parser.add_argument("--robot-name", type=str, default="gr2")
    parser.add_argument("--client-init-retries", type=int, default=4)
    parser.add_argument("--client-retry-interval-s", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true", help="Use zero state instead of real robot state.")
    parser.add_argument("--no-camera", action="store_true", help="Use dummy RGB/depth frames.")

    parser.add_argument("--steps", type=int, default=20, help="How many iterations to print.")
    parser.add_argument("--interval-s", type=float, default=0.0, help="Extra sleep per iteration.")
    parser.add_argument("--precision", type=int, default=4, help="Float print precision.")
    parser.add_argument(
        "--topk-varying-dims",
        type=int,
        default=8,
        help="Top-K action dims with largest std across chunk horizon.",
    )
    parser.add_argument(
        "--print-full-vector",
        action="store_true",
        help="Print full 35D first action vector in addition to grouped summary.",
    )
    return parser.parse_args()


def _fmt_vec(values: np.ndarray, precision: int) -> str:
    rounded = np.round(values.astype(np.float32), precision)
    return np.array2string(rounded, separator=", ", max_line_width=160)


def _print_grouped_action(action: np.ndarray, precision: int) -> None:
    print("First action (35D) grouped summary:")
    for name, group_slice in ACTION_GROUPS:
        vec = action[group_slice]
        print(
            f"  - {name:<10} idx[{group_slice.start:02d}:{group_slice.stop:02d}] "
            f"mean={vec.mean(): .4f} min={vec.min(): .4f} max={vec.max(): .4f} l2={np.linalg.norm(vec): .4f}"
        )
        print(f"    values={_fmt_vec(vec, precision)}")


def _print_chunk_summary(chunk: np.ndarray, topk: int, precision: int) -> None:
    # chunk: (T, D)
    horizon, dim = chunk.shape
    first = chunk[0]
    mid = chunk[horizon // 2]
    last = chunk[-1]
    std_by_dim = chunk.std(axis=0)
    k = max(1, min(topk, dim))
    top_idx = np.argsort(std_by_dim)[::-1][:k]

    print(f"Chunk summary: horizon={horizon}, action_dim={dim}")
    print(f"  - delta_l2(first->last) = {np.linalg.norm(last - first):.4f}")
    print(f"  - mean_std_over_dims    = {std_by_dim.mean():.4f}")
    print(f"  - first[0:8]            = {_fmt_vec(first[:8], precision)}")
    print(f"  - middle[0:8]           = {_fmt_vec(mid[:8], precision)}")
    print(f"  - last[0:8]             = {_fmt_vec(last[:8], precision)}")
    print("  - top varying dims (idx -> std, first -> last):")
    for idx in top_idx:
        print(f"      {idx:02d} -> {std_by_dim[idx]:.5f}, {first[idx]: .4f} -> {last[idx]: .4f}")


def _depth_range_str(depth_u16: np.ndarray) -> str:
    valid_depth = depth_u16[depth_u16 > 0]
    if valid_depth.size == 0:
        return "empty"
    return f"{int(valid_depth.min())}..{int(valid_depth.max())}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    args = parse_args()

    pretrained_dir = base.resolve_pretrained_model_dir(args.checkpoint_path)
    device = base.select_device(args.device)

    LOGGER.info("Model: %s", pretrained_dir)
    LOGGER.info("Device: %s", device)

    policy = PI0Policy.from_pretrained(str(pretrained_dir), strict=False).to(device)
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    state_key, cfg_state_dim, visual_keys, action_key, cfg_action_dim = base.infer_policy_io(policy)
    state_dim = cfg_state_dim
    action_dim = cfg_action_dim
    visual_keys = rgbd._ordered_visual_keys(visual_keys, args.camera_key)
    chunk_size = int(getattr(policy.config, "chunk_size", -1))
    n_action_steps = int(getattr(policy.config, "n_action_steps", -1))

    LOGGER.info(
        "Policy IO: state_key=%s state_dim=%d action_key=%s action_dim=%d visual_keys=%s",
        state_key,
        state_dim,
        action_key,
        action_dim,
        visual_keys,
    )
    LOGGER.info("Policy action chunk config: chunk_size=%d n_action_steps=%d", chunk_size, n_action_steps)

    camera = None
    client = None
    try:
        if not args.no_camera:
            camera = base.OrbbecRGBDCamera(
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
                timeout_ms=args.camera_timeout_ms,
                warmup_frames=args.camera_warmup_frames,
                init_retries=args.camera_init_retries,
                retry_interval_s=args.camera_retry_interval_s,
            )
            camera.connect()

        if not args.dry_run:
            client = base.load_robot_client(
                args.domain_id,
                args.robot_name,
                retries=args.client_init_retries,
                retry_interval_s=args.client_retry_interval_s,
            )

        for step in range(args.steps):
            if camera is None:
                rgb = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
                depth_rgb = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
                depth_u16 = np.zeros((args.camera_height, args.camera_width), dtype=np.uint16)
            else:
                frame = camera.read()
                if frame is None:
                    print(f"\n=== step {step} ===")
                    print("No camera frame yet, skip.")
                    continue
                rgb, depth_rgb, depth_u16, _ = frame

            if client is None:
                state_full = np.zeros((45,), dtype=np.float32)
            else:
                state_full = base.get_robot_state_urdf(client)
            state_model = base.fit_vector(state_full, state_dim)

            observation = rgbd._build_visual_observation(visual_keys, rgb, depth_rgb)
            observation[state_key] = state_model

            with torch.inference_mode():
                batch = prepare_observation_for_inference(
                    observation=observation,
                    device=device,
                    task=args.task,
                    robot_type=args.robot_type,
                )
                batch = preprocessor(batch)
                # Use predict_action_chunk so we can inspect the whole model output horizon.
                action_chunk_tensor = postprocessor(policy.predict_action_chunk(batch))

            chunk = action_chunk_tensor.detach().cpu().numpy().astype(np.float32)
            # shape: (B, T, D) -> (T, D)
            chunk = chunk.reshape(-1, chunk.shape[-2], chunk.shape[-1])[0]
            first_action = base.fit_vector(chunk[0], action_dim)

            print(f"\n=== step {step} ===")
            print(
                f"depth(mm)={_depth_range_str(depth_u16)} | "
                f"state_l2={np.linalg.norm(state_model):.4f} | "
                f"chunk_shape={tuple(chunk.shape)}"
            )
            _print_chunk_summary(chunk, args.topk_varying_dims, args.precision)
            _print_grouped_action(first_action, args.precision)

            if args.print_full_vector:
                print(f"First action full 35D = {_fmt_vec(first_action, args.precision)}")

            if args.interval_s > 0:
                time.sleep(max(0.0, args.interval_s))

    finally:
        if camera is not None:
            camera.close()
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
