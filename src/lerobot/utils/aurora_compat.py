#!/usr/bin/env python

from __future__ import annotations

import logging
from enum import IntEnum

LOGGER = logging.getLogger(__name__)

_PATCHED = False


def _enum_accepts_value(enum_cls: type, value: int) -> bool:
    try:
        enum_cls(value)
    except ValueError:
        return False
    return True


def patch_fourier_aurora_client_enums() -> None:
    """Patch Aurora enums so newer controller values do not crash DDS callbacks."""

    global _PATCHED
    if _PATCHED:
        return

    try:
        import fourier_aurora_client.aurora_state_map as state_map
        import fourier_aurora_client.client as client_mod
    except ModuleNotFoundError:
        return

    upper_body_enum = getattr(state_map, "UpperBodyFsmState", None)
    if upper_body_enum is None or _enum_accepts_value(upper_body_enum, 4):
        _PATCHED = True
        return

    seen_unknown_values: set[int] = set()

    class SafeUpperBodyFsmState(IntEnum):
        Default = 0
        UpperBodyActState = 1
        RemoteState = 2

        @classmethod
        def _missing_(cls, value: object):
            try:
                raw_value = int(value)
            except (TypeError, ValueError):
                return None

            member = int.__new__(cls, raw_value)
            member._name_ = f"Unknown_{raw_value}"
            member._value_ = raw_value
            if raw_value not in seen_unknown_values:
                seen_unknown_values.add(raw_value)
                LOGGER.warning(
                    "Observed unsupported Aurora upper body FSM state %s; treating it as %s.",
                    raw_value,
                    member._name_,
                )
            return member

    state_map.UpperBodyFsmState = SafeUpperBodyFsmState
    client_mod.UpperBodyFsmState = SafeUpperBodyFsmState
    _PATCHED = True
    LOGGER.warning("Patched fourier_aurora_client UpperBodyFsmState to tolerate unknown values.")
