from __future__ import annotations

import pytest

from laya_apple.artifacts import ComputeUnitMismatchError, check_ane_placement


def _summary(**overrides):
    base = {"ops": {"ane": 10, "cpu": 0, "gpu": 0}, "transitions": 0}
    base.update(overrides)
    return base


def test_fully_ane_placement_passes():
    check_ane_placement(_summary())


def test_cpu_ops_raise_compute_unit_mismatch():
    with pytest.raises(ComputeUnitMismatchError):
        check_ane_placement(_summary(ops={"ane": 8, "cpu": 2, "gpu": 0}))


def test_gpu_ops_raise_compute_unit_mismatch():
    with pytest.raises(ComputeUnitMismatchError):
        check_ane_placement(_summary(ops={"ane": 8, "cpu": 0, "gpu": 2}))


def test_transitions_raise_compute_unit_mismatch():
    with pytest.raises(ComputeUnitMismatchError):
        check_ane_placement(_summary(transitions=1))


def test_zero_ane_ops_raise_compute_unit_mismatch():
    with pytest.raises(ComputeUnitMismatchError):
        check_ane_placement(_summary(ops={"ane": 0, "cpu": 0, "gpu": 0}))
