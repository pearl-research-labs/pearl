"""Regression tests for the wheel-level CUTLASS compatibility alias."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from vllm_miner import cutlass_compat


def _install_fake_cutlass(
    monkeypatch: pytest.MonkeyPatch,
    *,
    make_fragment: object | None = None,
) -> tuple[ModuleType, object]:
    cutlass = ModuleType("cutlass")
    cutlass.__path__ = []
    cute = ModuleType("cutlass.cute")
    replacement = object()
    cute.make_rmem_tensor = replacement
    if make_fragment is not None:
        cute.make_fragment = make_fragment
    cutlass.cute = cute
    monkeypatch.setitem(sys.modules, "cutlass", cutlass)
    monkeypatch.setitem(sys.modules, "cutlass.cute", cute)
    return cute, replacement


def test_compat_installs_alias_once(monkeypatch: pytest.MonkeyPatch) -> None:
    cute, replacement = _install_fake_cutlass(monkeypatch)
    info = MagicMock()
    monkeypatch.setattr(cutlass_compat._LOGGER, "info", info)

    cutlass_compat.apply_cutlass_46_compat()
    cutlass_compat.apply_cutlass_46_compat()

    assert cute.make_fragment is replacement
    assert getattr(cute, cutlass_compat._COMPAT_APPLIED_ATTR) is True
    info.assert_called_once_with("Applied cutlass-dsl 4.6 alias: cute.make_fragment")


def test_compat_preserves_existing_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = object()
    cute, _ = _install_fake_cutlass(monkeypatch, make_fragment=existing)
    info = MagicMock()
    monkeypatch.setattr(cutlass_compat._LOGGER, "info", info)

    cutlass_compat.apply_cutlass_46_compat()

    assert cute.make_fragment is existing
    assert getattr(cute, cutlass_compat._COMPAT_APPLIED_ATTR) is True
    info.assert_not_called()
