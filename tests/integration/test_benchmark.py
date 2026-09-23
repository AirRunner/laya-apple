from __future__ import annotations

import pytest

from laya_apple.benchmark import run

pytestmark = pytest.mark.integration


def test_benchmark_run_returns_ok_records(cached_laya, model_name):
    laya = cached_laya(model_name, device="gpu")
    records = run(model_name, device="gpu", lengths=[64], questions=1, warmup=1, iters=3, laya=laya)
    assert len(records) == 1
    rec = records[0]
    assert rec["status"] == "ok"
    assert rec["forward"]["n"] == 3
    assert rec["predict"]["n"] == 3
    assert rec["backend"] == "mlx"
