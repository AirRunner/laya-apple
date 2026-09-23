from __future__ import annotations

from laya_apple.result import Result, RuntimeInfo


def _runtime(**overrides):
    base = dict(
        backend="mlx",
        device="gpu",
        model="laya",
        model_revision="abc123",
        sequence_length=64,
        question_count=1,
        routing_reason="gpu_requested",
        artifact_revision="mlx:deadbeef:float16",
        latency_ms=1.23,
    )
    base.update(overrides)
    return RuntimeInfo(**base)


def test_runtime_info_str_contains_key_fields():
    r = _runtime()
    s = str(r)
    assert "backend=mlx" in s
    assert "device=gpu" in s
    assert "reason=gpu_requested" in s


def test_result_to_dict_has_upstream_and_runtime_keys():
    r = Result(
        answers={"q1": {"type": "noul", "noul": 0.5}},
        usage={"input_tokens": 10, "output_tokens": 0},
        runtime=_runtime(),
    )
    d = r.to_dict()
    assert set(d) >= {"model", "answers", "usage", "runtime"}
    assert d["answers"] == {"q1": {"type": "noul", "noul": 0.5}}
    assert d["runtime"]["backend"] == "mlx"
    assert isinstance(d["runtime"]["buckets"], list)


def test_result_to_dict_buckets_are_a_list_not_tuple():
    r = Result(answers={}, usage={}, runtime=_runtime(buckets=(64, 96)))
    d = r.to_dict()
    assert d["runtime"]["buckets"] == [64, 96]
