#!/usr/bin/env python

from __future__ import annotations

import importlib
import sys
import types
from enum import Enum


def test_patch_fourier_aurora_client_enums_accepts_unknown_upper_body_states(monkeypatch):
    class OriginalUpperBodyFsmState(Enum):
        Default = 0
        UpperBodyActState = 1
        RemoteState = 2

    fake_pkg = types.ModuleType("fourier_aurora_client")
    fake_pkg.__path__ = []

    fake_state_map = types.ModuleType("fourier_aurora_client.aurora_state_map")
    fake_state_map.UpperBodyFsmState = OriginalUpperBodyFsmState

    fake_client = types.ModuleType("fourier_aurora_client.client")
    fake_client.UpperBodyFsmState = OriginalUpperBodyFsmState

    monkeypatch.setitem(sys.modules, "fourier_aurora_client", fake_pkg)
    monkeypatch.setitem(sys.modules, "fourier_aurora_client.aurora_state_map", fake_state_map)
    monkeypatch.setitem(sys.modules, "fourier_aurora_client.client", fake_client)

    import lerobot.utils.aurora_compat as aurora_compat

    importlib.reload(aurora_compat)
    aurora_compat.patch_fourier_aurora_client_enums()

    patched_enum = fake_client.UpperBodyFsmState
    assert fake_state_map.UpperBodyFsmState is patched_enum
    assert patched_enum(0).name == "Default"
    assert patched_enum(4).name == "Unknown_4"
