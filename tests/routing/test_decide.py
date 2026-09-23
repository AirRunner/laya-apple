from __future__ import annotations

import pytest

from laya_apple import routing
from laya_apple.registry import models


@pytest.fixture(params=list(models().values()), ids=lambda s: s.name)
def spec(request):
    return request.param


def test_gpu_requested(spec):
    d = routing.decide("gpu", spec, 1, 10, routing.AneState())
    assert d == routing.Decision("gpu", routing.GPU_REQUESTED)


def test_ane_requested(spec):
    d = routing.decide("ane", spec, 1, 10, routing.AneState())
    assert d == routing.Decision("ane", routing.ANE_REQUESTED)


def test_invalid_device_raises(spec):
    with pytest.raises(ValueError):
        routing.decide("tpu", spec, 1, 10, routing.AneState())


def test_auto_ane_unavailable_runtime(spec):
    ane = routing.AneState(routing.RUNTIME_UNAVAILABLE)
    d = routing.decide("auto", spec, 1, 10, ane)
    assert d.target == "gpu"
    assert d.reason == routing.RUNTIME_UNAVAILABLE


def test_auto_platform_not_validated(spec):
    ane = routing.AneState(routing.PLATFORM_NOT_VALIDATED)
    d = routing.decide("auto", spec, 1, 10, ane)
    assert d.target == "gpu"
    assert d.reason == routing.PLATFORM_NOT_VALIDATED


def test_auto_multiple_questions(spec):
    ane = routing.AneState(None, spec.auto_ane_buckets)
    d = routing.decide("auto", spec, spec.auto_ane_max_questions + 1, 10, ane)
    assert d.target == "gpu"
    assert d.reason == routing.MULTIPLE_QUESTIONS


def test_auto_exceeds_range(spec):
    ane = routing.AneState(None, spec.auto_ane_buckets)
    d = routing.decide("auto", spec, 1, spec.auto_ane_max_len + 1, ane)
    assert d.target == "gpu"
    assert d.reason == routing.EXCEEDS_AUTO_RANGE


def test_auto_artifact_unavailable(spec):
    # buckets exist for the shape, but ane.buckets is empty -> not offered
    if not spec.auto_ane_buckets:
        pytest.skip(f"{spec.name} has no auto ANE buckets")
    ane = routing.AneState(None, ())
    d = routing.decide("auto", spec, 1, spec.auto_ane_buckets[0], ane)
    assert d.target == "gpu"
    assert d.reason == routing.ARTIFACT_UNAVAILABLE


def test_auto_validated_path(spec):
    if not spec.auto_ane_buckets:
        pytest.skip(f"{spec.name} has no auto ANE buckets")
    ane = routing.AneState(None, spec.auto_ane_buckets)
    d = routing.decide("auto", spec, 1, spec.auto_ane_buckets[0], ane)
    assert d.target == "ane"
    assert d.reason == routing.ANE_AUTO
    assert d.bucket == spec.auto_ane_buckets[0]


def test_auto_boundary_at_max_len_routes_ane(spec):
    if not spec.auto_ane_buckets:
        pytest.skip(f"{spec.name} has no auto ANE buckets")
    ane = routing.AneState(None, spec.auto_ane_buckets)
    d = routing.decide("auto", spec, 1, spec.auto_ane_max_len, ane)
    assert d.target == "ane"


def test_auto_boundary_past_max_len_routes_gpu(spec):
    if not spec.auto_ane_buckets:
        pytest.skip(f"{spec.name} has no auto ANE buckets")
    ane = routing.AneState(None, spec.auto_ane_buckets)
    d = routing.decide("auto", spec, 1, spec.auto_ane_max_len + 1, ane)
    assert d.target == "gpu"
    assert d.reason == routing.EXCEEDS_AUTO_RANGE


def test_auto_two_questions_is_multiple_questions_reason(spec):
    ane = routing.AneState(None, spec.auto_ane_buckets)
    d = routing.decide("auto", spec, 2, 10, ane)
    if spec.auto_ane_max_questions >= 2:
        pytest.skip("this model's auto max questions allows 2")
    assert d.reason == routing.MULTIPLE_QUESTIONS


def test_multilingual_256_is_explicit_only():
    spec = models()["laya-multilingual"]
    assert 256 not in spec.auto_ane_buckets
    assert 256 in spec.ane_buckets
    ane = routing.AneState(None, spec.auto_ane_buckets)
    d = routing.decide("auto", spec, 1, 256, ane)
    assert d.target == "gpu"
    assert d.reason == routing.EXCEEDS_AUTO_RANGE
    # but explicit ane device still routes there regardless of length
    d2 = routing.decide("ane", spec, 1, 256, ane)
    assert d2.target == "ane"


def test_determinism_same_inputs_same_decision(spec):
    ane = routing.AneState(None, spec.auto_ane_buckets)
    results = {routing.decide("auto", spec, 1, 10, ane) for _ in range(20)}
    assert len(results) == 1


def test_auto_never_returns_ane_when_unavailable(spec):
    ane = routing.AneState(routing.RUNTIME_UNAVAILABLE, spec.auto_ane_buckets)
    for length in (1, spec.auto_ane_max_len or 1, spec.max_len):
        d = routing.decide("auto", spec, 1, length, ane)
        assert d.target != "ane"


@pytest.mark.parametrize("reason", routing.REASONS)
def test_every_reason_code_is_reachable(reason, spec):
    """Each reason in routing.REASONS must be producible by some call to decide()."""
    ane_ok = routing.AneState(None, spec.auto_ane_buckets)
    ane_runtime = routing.AneState(routing.RUNTIME_UNAVAILABLE)
    ane_platform = routing.AneState(routing.PLATFORM_NOT_VALIDATED)
    ane_empty = routing.AneState(None, ())

    checks = {
        routing.GPU_REQUESTED: lambda: routing.decide("gpu", spec, 1, 10, ane_ok),
        routing.ANE_REQUESTED: lambda: routing.decide("ane", spec, 1, 10, ane_ok),
        routing.ANE_AUTO: lambda: (
            routing.decide("auto", spec, 1, spec.auto_ane_buckets[0], ane_ok) if spec.auto_ane_buckets else None
        ),
        routing.MULTIPLE_QUESTIONS: lambda: routing.decide("auto", spec, spec.auto_ane_max_questions + 1, 10, ane_ok),
        routing.EXCEEDS_AUTO_RANGE: lambda: routing.decide("auto", spec, 1, spec.max_len, ane_ok),
        routing.ARTIFACT_UNAVAILABLE: lambda: (
            routing.decide("auto", spec, 1, spec.auto_ane_buckets[0], ane_empty) if spec.auto_ane_buckets else None
        ),
        routing.RUNTIME_UNAVAILABLE: lambda: routing.decide("auto", spec, 1, 10, ane_runtime),
        routing.PLATFORM_NOT_VALIDATED: lambda: routing.decide("auto", spec, 1, 10, ane_platform),
        routing.ANE_STARTING: lambda: routing.decide("auto", spec, 1, 10, routing.AneState(routing.ANE_STARTING)),
    }
    fn = checks[reason]
    if (
        spec.name != "laya-multilingual"
        and reason
        in (
            routing.ANE_AUTO,
            routing.ARTIFACT_UNAVAILABLE,
        )
        and not spec.auto_ane_buckets
    ):
        pytest.skip(f"{spec.name} has no auto ANE buckets")
    d = fn()
    if d is None:
        pytest.skip(f"{spec.name} cannot reach {reason}")
    assert d.reason == reason
