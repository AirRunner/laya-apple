from __future__ import annotations

import json

import pytest

from laya_apple.parity import evaluate

pytestmark = pytest.mark.parity


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_mlx_passes_parity_gate(model_name, dtype, cached_laya):
    laya = cached_laya(model_name, device="gpu", dtype=dtype)
    summary = evaluate(
        model_name,
        laya.config,
        laya.mlx.forward,
        precision=dtype,
        prepare=lambda s, q: laya.prepare(s, q).items,
    )
    if not summary["passed"]:
        pytest.fail(f"{model_name} MLX {dtype} failed the parity gate:\n{json.dumps(summary, indent=1)}")
