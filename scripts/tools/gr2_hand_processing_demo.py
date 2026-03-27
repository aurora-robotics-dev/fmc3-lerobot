#!/usr/bin/env python3

"""Standalone demo for GR2 hand processing used during deployment.

This script demonstrates three pieces of logic that are otherwise embedded in
the deploy path:

1. SDK hand state -> URDF hand state
2. Dataset-specific right-hand trigger coupling
3. URDF hand action -> SDK hand command

The demo is intentionally offline. It does not connect to the robot or any
camera. It only prints the intermediate arrays so the hand-processing path can
be inspected in isolation.

Examples:
    python scripts/tools/gr2_hand_processing_demo.py
    python scripts/tools/gr2_hand_processing_demo.py --thumb-pitch 0.46 --right-side-only
    python scripts/tools/gr2_hand_processing_demo.py --sdk-hand 0.1 0.2 0.3 0.4 0.5 -0.6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from collections.abc import Sequence

# These slices intentionally match the deploy scripts:
# - scripts/deploy_gr2_pi0.py
# - scripts/deploy_gr2_pi0_rgb_right_wrist.py
LEFT_MANIPULATOR_SLICE = slice(0, 7)
RIGHT_MANIPULATOR_SLICE = slice(7, 14)
LEFT_HAND_SLICE = slice(14, 20)
RIGHT_HAND_SLICE = slice(20, 26)
RIGHT_HAND_FINGER_SLICE = slice(20, 24)
RIGHT_HAND_THUMB_PITCH_INDEX = 24

RIGHT_HAND_FINGER_MIN_URDF = -1.9226667
RIGHT_HAND_FINGER_MAX_URDF = 0.0

# ---------------------------------------------------------------------------
# Hand conversion copied directly into this demo.
# ---------------------------------------------------------------------------
# These constants match the main deploy path:
# - src/lerobot/utils/gr2_hand_conversion.py
# They are duplicated here on purpose so this demo stays fully self-contained.
DEX_FINGER_OFFSET = 0.17
DEX_THUMB_PITCH_OFFSET = 0.12

# SDK-side safety range used when converting URDF actions back to SDK commands.
# The first four dimensions are the four fingers, the fifth is thumb pitch, and
# the sixth is thumb yaw.
DEX_HAND_SDK_MIN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -np.inf], dtype=np.float32)
DEX_HAND_SDK_MAX = np.array([1.9226667, 1.9226667, 1.9226667, 1.9226667, 1.5, np.inf], dtype=np.float32)

LEFT_HAND_NAMES = [
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_ring_proximal_joint",
    "L_pinky_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
]
RIGHT_HAND_NAMES = [
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
]


def _to_hand_array(hand: Sequence[float]) -> np.ndarray:
    """Normalize hand input to a fixed 6D float32 vector.

    The deploy code expects every hand vector to have exactly 6 dimensions:
    4 fingers + thumb_pitch + thumb_yaw.
    """

    hand_array = np.asarray(hand, dtype=np.float32)
    if hand_array.shape != (6,):
        raise ValueError(f"Expected a 6-DoF hand vector, got shape {hand_array.shape}")
    return hand_array


def hand_sdk_to_urdf(hand_sdk: Sequence[float]) -> np.ndarray:
    """Convert a GR2 hand vector from SDK space into URDF space.

    This is the exact logic used when reading robot state for the policy.
    """

    hand_sdk = _to_hand_array(hand_sdk)
    return np.array(
        [
            DEX_FINGER_OFFSET - hand_sdk[0],
            DEX_FINGER_OFFSET - hand_sdk[1],
            DEX_FINGER_OFFSET - hand_sdk[2],
            DEX_FINGER_OFFSET - hand_sdk[3],
            hand_sdk[4] - DEX_THUMB_PITCH_OFFSET,
            -hand_sdk[5],
        ],
        dtype=np.float32,
    )


def hand_urdf_to_sdk(hand_urdf: Sequence[float], clip: bool = False) -> np.ndarray:
    """Convert a GR2 hand vector from URDF space back into SDK space.

    This is the exact logic used before sending hand commands to the robot.
    """

    hand_urdf = _to_hand_array(hand_urdf)
    hand_sdk = np.array(
        [
            DEX_FINGER_OFFSET - hand_urdf[0],
            DEX_FINGER_OFFSET - hand_urdf[1],
            DEX_FINGER_OFFSET - hand_urdf[2],
            DEX_FINGER_OFFSET - hand_urdf[3],
            DEX_THUMB_PITCH_OFFSET + hand_urdf[4],
            -hand_urdf[5],
        ],
        dtype=np.float32,
    )
    if clip:
        hand_sdk = np.clip(hand_sdk, DEX_HAND_SDK_MIN, DEX_HAND_SDK_MAX)
    return hand_sdk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline demo for GR2 hand processing.")
    parser.add_argument(
        "--sdk-hand",
        type=float,
        nargs=6,
        metavar=("F0", "F1", "F2", "F3", "THUMB_PITCH", "THUMB_YAW"),
        default=[0.10, 0.20, 0.30, 0.40, 0.50, -0.60],
        help="Example 6D hand state in SDK space used for the round-trip demo.",
    )
    parser.add_argument(
        "--thumb-pitch",
        type=float,
        default=0.461,
        help="Example right-thumb pitch in URDF space used for the trigger-coupling demo.",
    )
    parser.add_argument(
        "--thumb-yaw",
        type=float,
        default=-1.57,
        help="Example right-thumb yaw in URDF space used for the trigger-coupling demo.",
    )
    parser.add_argument(
        "--right-side-only",
        action="store_true",
        help="Also demonstrate the deploy-side mask that freezes left arm and left hand.",
    )
    return parser.parse_args()


def apply_right_hand_trigger_coupling(action_urdf: np.ndarray) -> np.ndarray:
    """Same dataset-specific coupling used in deploy_gr2_pi0_rgb_right_wrist.py.

    The dataset driving the right-wrist deployment effectively uses the right
    thumb pitch as the single control variable for the four fingers:

        right_fingers = clip(-2 * right_thumb_pitch, -1.9226667, 0.0)

    That means the model's right-hand fingers are not treated as independent in
    deployment for this task. Instead, they are reconstructed from thumb pitch.
    """

    coupled = action_urdf.copy()
    thumb_pitch = float(coupled[RIGHT_HAND_THUMB_PITCH_INDEX])
    coupled[RIGHT_HAND_FINGER_SLICE] = float(
        np.clip(-2.0 * thumb_pitch, RIGHT_HAND_FINGER_MIN_URDF, RIGHT_HAND_FINGER_MAX_URDF)
    )
    return coupled


def apply_left_side_motion_mask(action_urdf: np.ndarray, state_urdf: np.ndarray) -> np.ndarray:
    """Same masking strategy used by --right-side-only in the deploy script.

    This does not zero the left side. Instead, it copies the current robot
    state into the left-arm and left-hand slots. In practice this means the
    deploy loop keeps commanding the current pose, so the left side stays still.
    """

    masked = action_urdf.copy()
    masked[LEFT_MANIPULATOR_SLICE] = state_urdf[LEFT_MANIPULATOR_SLICE]
    masked[LEFT_HAND_SLICE] = state_urdf[LEFT_HAND_SLICE]
    return masked


def format_named_vector(values: np.ndarray, names: list[str]) -> str:
    return "\n".join(f"  {name:<30} {float(value):>8.4f}" for name, value in zip(names, values, strict=True))


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def demo_sdk_urdf_roundtrip(sdk_hand: np.ndarray) -> None:
    print_section("1. SDK -> URDF -> SDK Round Trip")

    # Step A: Convert a raw hand state from the Aurora SDK convention into the
    # URDF convention used by the policy and dataset.
    urdf_hand = hand_sdk_to_urdf(sdk_hand)

    # Step B: Convert it back to SDK space. For a correct implementation, this
    # should recover the original values up to numerical precision.
    sdk_roundtrip = hand_urdf_to_sdk(urdf_hand, clip=False)

    print("Input SDK hand:")
    print(format_named_vector(sdk_hand, RIGHT_HAND_NAMES))
    print()
    print("Converted URDF hand:")
    print(format_named_vector(urdf_hand, RIGHT_HAND_NAMES))
    print()
    print("Back-converted SDK hand:")
    print(format_named_vector(sdk_roundtrip, RIGHT_HAND_NAMES))
    print()
    print(f"max_abs_roundtrip_error = {float(np.max(np.abs(sdk_roundtrip - sdk_hand))):.8f}")


def demo_right_hand_trigger_coupling(thumb_pitch: float, thumb_yaw: float) -> np.ndarray:
    print_section("2. Right-Hand Trigger Coupling")

    # This 35D action is just a demo container. Only the right-hand slice is
    # populated because this section focuses on the dataset-specific hand logic.
    raw_action_urdf = np.zeros((35,), dtype=np.float32)
    raw_action_urdf[RIGHT_HAND_SLICE] = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            thumb_pitch,
            thumb_yaw,
        ],
        dtype=np.float32,
    )

    coupled_action_urdf = apply_right_hand_trigger_coupling(raw_action_urdf)

    print("Raw right-hand action in URDF space:")
    print(format_named_vector(raw_action_urdf[RIGHT_HAND_SLICE], RIGHT_HAND_NAMES))
    print()
    print("Coupled right-hand action in URDF space:")
    print(format_named_vector(coupled_action_urdf[RIGHT_HAND_SLICE], RIGHT_HAND_NAMES))
    print()
    print("Expected behavior:")
    print("  right fingers = clip(-2 * right_thumb_pitch, -1.9226667, 0.0)")

    return coupled_action_urdf


def demo_full_deploy_side_path(coupled_action_urdf: np.ndarray, right_side_only: bool) -> None:
    print_section("3. Full Deploy-Side Hand Processing")

    # This example state mimics the current robot state in URDF space. The left
    # arm/hand values are intentionally non-zero so the masking effect is easy
    # to see when --right-side-only is enabled.
    state_urdf = np.zeros((45,), dtype=np.float32)
    state_urdf[LEFT_MANIPULATOR_SLICE] = np.array([0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17], dtype=np.float32)
    state_urdf[LEFT_HAND_SLICE] = np.array([-0.01, -0.02, -0.03, -0.04, 0.05, -1.57], dtype=np.float32)

    # Start from the action after the right-hand coupling stage.
    action_before_mask = coupled_action_urdf.copy()

    # Give the left side obviously different values so the masking effect is
    # visible in the printed diff.
    action_before_mask[LEFT_MANIPULATOR_SLICE] = np.array(
        [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07], dtype=np.float32
    )
    action_before_mask[LEFT_HAND_SLICE] = np.array(
        [-1.00, -1.10, -1.20, -1.30, 0.40, -0.50], dtype=np.float32
    )

    if right_side_only:
        action_after_mask = apply_left_side_motion_mask(action_before_mask, state_urdf)
    else:
        action_after_mask = action_before_mask.copy()

    # This is the final hand-specific step done before sending commands to the
    # robot: URDF hand actions are converted back into SDK command space.
    action_send = action_after_mask.copy()
    action_send[LEFT_HAND_SLICE] = hand_urdf_to_sdk(action_send[LEFT_HAND_SLICE], clip=True)
    action_send[RIGHT_HAND_SLICE] = hand_urdf_to_sdk(action_send[RIGHT_HAND_SLICE], clip=True)

    print("Left hand before mask (URDF action):")
    print(format_named_vector(action_before_mask[LEFT_HAND_SLICE], LEFT_HAND_NAMES))
    print()
    print("Left hand after mask (URDF action):")
    print(format_named_vector(action_after_mask[LEFT_HAND_SLICE], LEFT_HAND_NAMES))
    print()
    print("Right hand after coupling/mask (URDF action):")
    print(format_named_vector(action_after_mask[RIGHT_HAND_SLICE], RIGHT_HAND_NAMES))
    print()
    print("Left hand final SDK command:")
    print(format_named_vector(action_send[LEFT_HAND_SLICE], LEFT_HAND_NAMES))
    print()
    print("Right hand final SDK command:")
    print(format_named_vector(action_send[RIGHT_HAND_SLICE], RIGHT_HAND_NAMES))
    print()
    print(f"right_side_only = {right_side_only}")


def main() -> None:
    args = parse_args()

    sdk_hand = np.asarray(args.sdk_hand, dtype=np.float32)

    print("GR2 hand processing demo")
    print(f"Script path: {Path(__file__).resolve()}")
    print("This demo is offline. It only prints intermediate arrays.")

    demo_sdk_urdf_roundtrip(sdk_hand)
    coupled_action_urdf = demo_right_hand_trigger_coupling(args.thumb_pitch, args.thumb_yaw)
    demo_full_deploy_side_path(coupled_action_urdf, args.right_side_only)


if __name__ == "__main__":
    main()
