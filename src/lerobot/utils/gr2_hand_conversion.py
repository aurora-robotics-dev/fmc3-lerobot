#!/usr/bin/env python

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEX_FINGER_OFFSET = 0.17
DEX_THUMB_PITCH_OFFSET = 0.12

DEX_HAND_SDK_MIN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -np.inf], dtype=np.float32)
DEX_HAND_SDK_MAX = np.array([1.9226667, 1.9226667, 1.9226667, 1.9226667, 1.5, np.inf], dtype=np.float32)


def _to_hand_array(hand: Sequence[float]) -> np.ndarray:
    hand_array = np.asarray(hand, dtype=np.float32)
    if hand_array.shape != (6,):
        raise ValueError(f"Expected a 6-DoF hand vector, got shape {hand_array.shape}")
    return hand_array


def hand_sdk_to_urdf(hand_sdk: Sequence[float]) -> np.ndarray:
    """Convert one GR2 dexterous-hand vector from SDK space to URDF space."""
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
    """Convert one GR2 dexterous-hand vector from URDF space to SDK space."""
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
