"""Stress/soak tests (DEVELOPMENT_PLAN.md §12.4). Opt-in: LAYA_APPLE_STRESS=1, with
LAYA_APPLE_STRESS_SECONDS for the sustained-load duration (default 60, 600 for a release)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

STRESS_ENABLED = os.environ.get("LAYA_APPLE_STRESS") == "1"
HERE = Path(__file__).parent


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="stress tests only run with LAYA_APPLE_STRESS=1")
    for item in items:
        if HERE in Path(item.fspath).parents:
            item.add_marker(pytest.mark.stress)
            if not STRESS_ENABLED:
                item.add_marker(skip)
