from __future__ import annotations

import pytest

from src.benchmark.cli import _safe_run_id


@pytest.mark.parametrize("value", ["smoke-20260811", "run_01", "A.b-c"])
def test_safe_run_id_accepts_portable_names(value):
    assert _safe_run_id(value) == value


@pytest.mark.parametrize("value", ["../escape", "space name", "", "a" * 81])
def test_safe_run_id_rejects_paths_and_unsafe_names(value):
    with pytest.raises(ValueError):
        _safe_run_id(value)
