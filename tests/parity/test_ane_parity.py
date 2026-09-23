from __future__ import annotations

import json

import pytest

from laya_apple import Laya
from laya_apple.parity import evaluate
from laya_apple.registry import models

pytestmark = [pytest.mark.parity, pytest.mark.ane]


def test_ane_passes_parity_gate_for_each_present_bucket(model_name, cached_laya):
    spec = models()[model_name]
    if not spec.ane_buckets:
        pytest.skip(f"{model_name} has no ANE buckets")
    try:
        laya = Laya.from_pretrained(model_name, device="ane", local_files_only=True)
    except Exception:
        pytest.skip(f"{model_name}: no validated ANE artifacts present")

    for bucket in laya.ane.buckets:
        summary = evaluate(
            model_name,
            laya.config,
            laya.ane.forward,
            precision="float16",
            max_len=bucket,
            prepare=lambda s, q: laya.prepare(s, q).items,
        )
        if not summary["passed"]:
            pytest.fail(f"{model_name} ANE L{bucket} failed the parity gate:\n{json.dumps(summary, indent=1)}")
