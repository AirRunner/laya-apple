"""v0.2 heterogeneous execution: execution="workers" (MLX worker process + ANE thread or process)."""

from __future__ import annotations

import asyncio
import os
import signal
from concurrent.futures import ThreadPoolExecutor

import pytest

from laya_apple import Laya, routing, scheduling
from laya_apple.errors import BackendUnavailableError, UnsupportedShapeError
from laya_apple.model import ane_placement_for
from laya_apple.workload import make_request

pytestmark = [pytest.mark.integration, pytest.mark.ane]
MODEL = "laya-typed-decisions"


@pytest.fixture(scope="module")
def inline():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return {
        "gpu": Laya.from_pretrained(MODEL, device="gpu", local_files_only=True),
        "ane": Laya.from_pretrained(MODEL, device="ane", local_files_only=True),
    }


@pytest.fixture(scope="module")
def auto_workers():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    laya = Laya.from_pretrained(MODEL, device="auto", execution="workers", local_files_only=True)
    if not laya.ane_state.buckets:
        laya.close()
        pytest.skip("no validated ANE artifacts for the workers test model")
    yield laya
    laya.close()


@pytest.fixture(scope="module")
def requests(inline):
    tok, cfg = inline["gpu"].tokenizer, inline["gpu"].config
    return {
        "short": make_request(tok, cfg, 128, 1, seed=1),
        "long": make_request(tok, cfg, 1024, 1, seed=2),
        "multi": make_request(tok, cfg, 96, 3, seed=3),
    }


def test_idle_routing_and_answers_match_inline(auto_workers, inline, requests):
    for name, (state, qs) in requests.items():
        r = auto_workers.predict(context=state, questions=qs)
        expected_device = "ane" if name == "short" else "gpu"
        assert r.runtime.device == expected_device, (name, r.runtime)
        assert r.runtime.execution == "workers"
        assert r.runtime.queue_wait_ms is not None
        ref = inline[expected_device].predict(context=state, questions=qs)
        assert r.answers == ref.answers  # same device, same bits
        assert r.runtime.artifact_revision == ref.runtime.artifact_revision


def test_concurrent_submissions_are_all_correct(auto_workers, inline, requests):
    refs = {
        n: {d: inline[d].predict(context=s, questions=q).answers for d in ("gpu", "ane") if n == "short" or d == "gpu"}
        for n, (s, q) in requests.items()
    }
    jobs = [n for n in ("short", "long", "multi") for _ in range(12)]
    with ThreadPoolExecutor(8) as pool:
        results = list(
            pool.map(lambda n: (n, auto_workers.predict(context=requests[n][0], questions=requests[n][1])), jobs)
        )
    for n, r in results:
        assert r.answers == refs[n][r.runtime.device], (n, r.runtime)
        assert r.runtime.routing_reason in routing.REASONS + scheduling.QUEUE_REASONS
        if n != "short":
            assert r.runtime.device == "gpu"  # long and multi-question work never goes to the ANE


def test_submit_and_apredict(auto_workers, requests):
    state, qs = requests["short"]
    fut = auto_workers.submit(context=state, questions=qs)
    assert fut.result(timeout=60).runtime.device in ("ane", "gpu")

    async def many():
        return await asyncio.gather(*(auto_workers.apredict(context=state, questions=qs) for _ in range(5)))

    out = asyncio.run(many())
    assert len({str(r.answers) for r in out}) <= 2  # at most one answer per device


def test_explicit_ane_workers_refuses_long_rows_before_queueing(inline, requests):
    laya = Laya.from_pretrained(MODEL, device="ane", execution="workers", local_files_only=True)
    try:
        assert laya.mlx is None and "gpu" not in laya._workers
        state, qs = requests["long"]
        with pytest.raises(UnsupportedShapeError):
            laya.submit(context=state, questions=qs)
        state, qs = requests["multi"]  # explicit ANE runs multi-question rows sequentially
        r = laya.predict(context=state, questions=qs)
        assert r.runtime.device == "ane" and r.answers == inline["ane"].predict(context=state, questions=qs).answers
    finally:
        laya.close()


def test_dead_gpu_worker_fails_loudly_and_never_falls_back(requests):
    laya = Laya.from_pretrained(MODEL, device="auto", execution="workers", local_files_only=True)
    try:
        gpu = laya._workers["gpu"]
        assert gpu.placement == "process" and laya._workers["ane"].placement == ane_placement_for(MODEL)
        os.kill(gpu.pid, signal.SIGKILL)
        gpu._proc.wait(timeout=10)
        state, qs = requests["long"]  # GPU-only work
        with pytest.raises(BackendUnavailableError):
            laya.predict(context=state, questions=qs)
        with pytest.raises(BackendUnavailableError):  # and it stays failed; nothing moves it to the ANE
            laya.predict(context=state, questions=qs)
        state, qs = requests["short"]  # the ANE path is unaffected
        assert laya.predict(context=state, questions=qs).runtime.device == "ane"
    finally:
        laya.close()


def test_close_is_idempotent_and_refuses_new_work(requests):
    laya = Laya.from_pretrained(MODEL, device="gpu", execution="workers", local_files_only=True)
    procs = [w._proc for w in laya._workers.values() if w.placement == "process"]
    assert procs
    laya.close()
    laya.close()
    assert all(p.poll() is not None for p in procs)
    with pytest.raises(BackendUnavailableError):
        laya.submit(context="x", questions=["q?"])


def test_context_manager_and_info():
    with Laya.from_pretrained(MODEL, device="gpu", execution="workers", local_files_only=True) as laya:
        info = laya.info()
        assert info["execution"] == "workers" and info["mlx"]
        r = laya.predict(context="The invoice was charged twice.", questions=["Is a refund requested?"])
        assert r.runtime.backend == "mlx" and r.runtime.execution == "workers"


def test_invalid_execution_raises():
    with pytest.raises(ValueError):
        Laya.from_pretrained(MODEL, device="gpu", execution="threads", local_files_only=True)


@pytest.mark.parametrize("placement", ["thread", "process"])
def test_both_ane_placements_give_identical_answers(placement, inline, requests):
    state, qs = requests["short"]
    with Laya.from_pretrained(
        MODEL, device="ane", execution="workers", ane_placement=placement, local_files_only=True
    ) as laya:
        assert laya._workers["ane"].placement == placement
        assert (
            laya.predict(context=state, questions=qs).answers
            == inline["ane"].predict(context=state, questions=qs).answers
        )


def test_invalid_ane_placement_raises():
    with pytest.raises(ValueError):
        Laya.from_pretrained(MODEL, device="gpu", execution="workers", ane_placement="gpu", local_files_only=True)


def test_background_ane_startup_serves_mlx_first_then_ane(monkeypatch, inline, requests):
    import time as _time

    from laya_apple import executor

    real_warm = executor.warm

    def slow_warm(kind, backend, pad_id):
        if kind == "ane":
            _time.sleep(3.0)  # stands in for an evicted on-device ANE compile
        return real_warm(kind, backend, pad_id)

    monkeypatch.setattr(executor, "warm", slow_warm)
    with Laya.from_pretrained(
        MODEL, execution="workers", ane_placement="thread", ane_startup="background", local_files_only=True
    ) as laya:
        state, qs = requests["short"]
        r = laya.predict(context=state, questions=qs)
        assert r.runtime.device == "gpu" and r.runtime.routing_reason == routing.ANE_STARTING
        assert r.answers == inline["gpu"].predict(context=state, questions=qs).answers
        assert laya.wait_for_ane(60) is True
        r = laya.predict(context=state, questions=qs)
        assert r.runtime.device == "ane" and r.runtime.routing_reason == routing.ANE_AUTO


@pytest.mark.parametrize(
    "kwargs",
    [
        {"execution": "inline", "ane_startup": "background"},
        {"execution": "workers", "device": "ane", "ane_startup": "background"},
        {"execution": "workers", "ane_startup": "later"},
    ],
)
def test_invalid_ane_startup_raises(kwargs):
    with pytest.raises(ValueError):
        Laya.from_pretrained(MODEL, local_files_only=True, **{"device": "auto", **kwargs})


@pytest.mark.parametrize("placement", ["thread", "process"])
def test_close_during_background_startup_is_clean(monkeypatch, placement):
    import threading
    import time as _time
    import warnings

    from laya_apple import executor

    real_warm = executor.warm

    def slow_warm(kind, backend, pad_id):
        if kind == "ane":
            _time.sleep(2.0)
        return real_warm(kind, backend, pad_id)

    monkeypatch.setattr(executor, "warm", slow_warm)  # affects the thread placement only
    before = set(threading.enumerate())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        laya = Laya.from_pretrained(
            MODEL, execution="workers", ane_placement=placement, ane_startup="background", local_files_only=True
        )
        laya.close()
        _time.sleep(3.0 if placement == "thread" else 0.5)
    assert not any("ANE startup failed" in str(w.message) for w in caught)
    new = [t for t in set(threading.enumerate()) - before if t.name.startswith("laya-")]
    assert not new, new
    assert laya._workers == {}
