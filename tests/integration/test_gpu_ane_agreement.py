from __future__ import annotations

import pytest

from laya_apple import Laya
from laya_apple.registry import models
from laya_apple.workload import make_request

pytestmark = [pytest.mark.integration, pytest.mark.ane]


def test_gpu_and_ane_answers_agree_within_tolerance(cached_laya, model_name):
    spec = models()[model_name]
    if not spec.ane_buckets:
        pytest.skip(f"{model_name} has no ANE buckets")
    bucket = spec.ane_buckets[0]
    try:
        ane_laya = Laya.from_pretrained(model_name, device="ane", local_files_only=True)
    except Exception:
        pytest.skip(f"{model_name}: no validated ANE artifact for L{bucket} present")

    gpu_laya = cached_laya(model_name, device="gpu")
    state, qs = make_request(gpu_laya.tokenizer, gpu_laya.config, bucket, n_questions=1)
    qid = next(iter(qs))

    gpu_result = gpu_laya.predict(context=state, questions=qs)
    ane_result = ane_laya.predict(context=state, questions=qs)

    ga, aa = gpu_result.answers[qid], ane_result.answers[qid]
    assert set(ga) == set(aa)
    assert ga["type"] == aa["type"]
    for label in ga["probabilities"]:
        assert abs(ga["probabilities"][label] - aa["probabilities"][label]) <= 0.02
