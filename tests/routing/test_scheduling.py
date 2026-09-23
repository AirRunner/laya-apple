"""v0.2 queue-aware routing: idle equals v0.1, backlog-aware otherwise, never long work on the ANE."""

from __future__ import annotations

import itertools

import pytest

from laya_apple import routing, scheduling
from laya_apple.registry import models, routing_table


@pytest.fixture(params=list(models().values()), ids=lambda s: s.name)
def spec(request):
    return request.param


def service(spec):
    return scheduling.ServiceModel(routing_table()["models"][spec.name]["service_ms"])


def states(spec):
    tie = tuple(b for b in spec.ane_buckets if b not in spec.auto_ane_buckets)
    return [
        (routing.AneState(None, spec.auto_ane_buckets), tie),
        (routing.AneState(None, spec.auto_ane_buckets), ()),
        (routing.AneState(None, ()), ()),
        (routing.AneState(routing.RUNTIME_UNAVAILABLE), ()),
        (routing.AneState(routing.PLATFORM_NOT_VALIDATED), ()),
    ]


LENGTHS = [1, 40, 64, 65, 96, 97, 128, 129, 200, 256, 257, 512, 1024]


def test_idle_machine_matches_v01_rule(spec):
    sv = service(spec)
    for (ane, tie), q, L, device in itertools.product(states(spec), (1, 2, 4), LENGTHS, ("auto", "gpu", "ane")):
        queued = scheduling.decide_queued(
            device, spec, q, L, ane, service=sv, gpu_backlog_ms=0.0, ane_backlog_ms=0.0, tie_buckets=tie
        )
        assert queued == routing.decide(device, spec, q, L, ane), (device, q, L, ane, tie)


def test_never_routes_beyond_offered_buckets_to_ane(spec):
    sv = service(spec)
    top = max(spec.ane_buckets)
    tie = tuple(b for b in spec.ane_buckets if b not in spec.auto_ane_buckets)
    ane = routing.AneState(None, spec.auto_ane_buckets)
    for L, gb in itertools.product([top + 1, 512, 1024], (0.0, 1e6)):
        d = scheduling.decide_queued(
            "auto", spec, 1, L, ane, service=sv, gpu_backlog_ms=gb, ane_backlog_ms=0.0, tie_buckets=tie
        )
        assert d.target == "gpu"


def test_multi_question_never_auto_ane_even_when_gpu_busy(spec):
    ane = routing.AneState(None, spec.auto_ane_buckets)
    d = scheduling.decide_queued(
        "auto", spec, 2, 64, ane, service=service(spec), gpu_backlog_ms=1e6, ane_backlog_ms=0.0
    )
    assert d == routing.Decision("gpu", routing.MULTIPLE_QUESTIONS)


def test_ane_backlog_overflows_short_requests_to_gpu(spec):
    ane = routing.AneState(None, spec.auto_ane_buckets)
    d = scheduling.decide_queued(
        "auto", spec, 1, 64, ane, service=service(spec), gpu_backlog_ms=0.0, ane_backlog_ms=1000.0
    )
    assert d == routing.Decision("gpu", scheduling.GPU_ANE_BACKLOG)


def test_small_ane_backlog_keeps_short_requests_on_ane(spec):
    sv = service(spec)
    ane = routing.AneState(None, spec.auto_ane_buckets)
    b = spec.auto_ane_buckets[0]
    slack = sv.gpu_ms(b) - sv.ane_ms(b)
    assert slack > 0
    d = scheduling.decide_queued("auto", spec, 1, b, ane, service=sv, gpu_backlog_ms=0.0, ane_backlog_ms=slack * 0.5)
    assert d.target == "ane" and d.reason == routing.ANE_AUTO


def test_tie_band_uses_ane_only_when_gpu_is_busy():
    spec = models()["laya-multilingual"]
    assert 256 in spec.ane_buckets and 256 not in spec.auto_ane_buckets
    sv = service(spec)
    ane = routing.AneState(None, spec.auto_ane_buckets)
    idle = scheduling.decide_queued(
        "auto", spec, 1, 200, ane, service=sv, gpu_backlog_ms=0.0, ane_backlog_ms=0.0, tie_buckets=(256,)
    )
    busy = scheduling.decide_queued(
        "auto", spec, 1, 200, ane, service=sv, gpu_backlog_ms=100.0, ane_backlog_ms=0.0, tie_buckets=(256,)
    )
    assert idle == routing.Decision("gpu", routing.EXCEEDS_AUTO_RANGE)
    assert busy == routing.Decision("ane", scheduling.ANE_GPU_BACKLOG, 256)
    # without the tie-band artifact loaded, a busy GPU changes nothing
    no_art = scheduling.decide_queued(
        "auto", spec, 1, 200, ane, service=sv, gpu_backlog_ms=100.0, ane_backlog_ms=0.0, tie_buckets=()
    )
    assert no_art.target == "gpu"


def test_deterministic_for_a_snapshot(spec):
    sv = service(spec)
    ane = routing.AneState(None, spec.auto_ane_buckets)
    for L, g, a in itertools.product(LENGTHS, (0, 5, 50), (0, 5, 50)):
        runs = {
            scheduling.decide_queued("auto", spec, 1, L, ane, service=sv, gpu_backlog_ms=g, ane_backlog_ms=a)
            for _ in range(5)
        }
        assert len(runs) == 1


def test_service_model_matches_measurements(spec):
    table = routing_table()["models"][spec.name]["service_ms"]
    sv = service(spec)
    for L, v in table["gpu"]["1"].items():
        assert sv.gpu_ms(int(L)) == pytest.approx(v)
    for L, v in table["gpu"]["4"].items():
        assert sv.gpu_ms(int(L), 4) == pytest.approx(v)
    for b in spec.ane_buckets:
        assert sv.ane_ms(b, 3) == pytest.approx(3 * table["ane"]["1"][str(b)])
    # monotone in length and question count
    assert sv.gpu_ms(300) < sv.gpu_ms(600) and sv.gpu_ms(128, 2) < sv.gpu_ms(128, 3)
