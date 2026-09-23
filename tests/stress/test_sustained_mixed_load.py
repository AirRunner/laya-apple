"""DEVELOPMENT_PLAN.md §12.4: sustained mixed load through execution="workers",
device="auto" — multiple client threads issuing short (ANE), long (GPU) and
multi-question (GPU) requests concurrently for LAYA_APPLE_STRESS_SECONDS (default 60,
600 for the release run).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import pytest

from laya_apple import Laya
from laya_apple.workload import make_request

STRESS_SECONDS = int(os.environ.get("LAYA_APPLE_STRESS_SECONDS", "60"))
MODEL = "laya-typed-decisions"
ROOT = Path(__file__).resolve().parents[2]
THREADS = 6


def _therm():
    try:
        r = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


@pytest.fixture(scope="module")
def laya():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    inst = Laya.from_pretrained(MODEL, device="auto", execution="workers", local_files_only=True)
    if not inst.ane_state.buckets:
        inst.close()
        pytest.skip("no validated ANE artifacts for the sustained-load stress test")
    yield inst
    inst.close()


def test_sustained_mixed_load(laya):
    tok, cfg = laya.tokenizer, laya.config
    requests = {
        "short": make_request(tok, cfg, 128, 1, seed=11),
        "long": make_request(tok, cfg, 1024, 1, seed=12),
        "multi": make_request(tok, cfg, 96, 3, seed=13),
    }
    # Answers are bit-identical per device, and equal only within the parity tolerance across
    # devices, so each result is checked against the inline answer from the device that served it.
    references = {}
    # Only the short single-question request can be served by the ANE under auto.
    for device, names in (("gpu", ("short", "long", "multi")), ("ane", ("short",))):
        with Laya.from_pretrained(MODEL, device=device, local_files_only=True) as ref:
            for name in names:
                s, q = requests[name]
                references[name, device] = ref.predict(context=s, questions=q).answers

    therm_before = _therm()
    parent = psutil.Process(os.getpid())
    baseline_rss = parent.memory_info().rss + sum(
        psutil.Process(c.pid).memory_info().rss for c in parent.children(recursive=True) if c.is_running()
    )

    errors: list[str] = []
    short_latencies_ms: list[float] = []
    lock_free_counter = {"n": 0}
    devices: dict = {}
    counts_lock = threading.Lock()
    stop_at = time.monotonic() + STRESS_SECONDS

    def worker():
        i = 0
        while time.monotonic() < stop_at:
            name = ("short", "long", "multi")[i % 3]
            i += 1
            state, qs = requests[name]
            t0 = time.perf_counter()
            try:
                r = laya.predict(context=state, questions=qs)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {type(e).__name__}: {e}")
                continue
            dt_ms = (time.perf_counter() - t0) * 1000
            if r.answers != references.get((name, r.runtime.device)):
                errors.append(f"{name}: answer mismatch on device {r.runtime.device}")
            with counts_lock:
                devices[name, r.runtime.device] = devices.get((name, r.runtime.device), 0) + 1
                if name == "short":
                    short_latencies_ms.append(dt_ms)
            lock_free_counter["n"] += 1

    with ThreadPoolExecutor(THREADS) as pool:
        futs = [pool.submit(worker) for _ in range(THREADS)]
        for f in futs:
            f.result()

    therm_after = _therm()
    end_rss = parent.memory_info().rss + sum(
        psutil.Process(c.pid).memory_info().rss for c in parent.children(recursive=True) if c.is_running()
    )

    assert not errors, f"{len(errors)} errors, first: {errors[:5]}"
    assert lock_free_counter["n"] > 0

    n = len(short_latencies_ms)
    p99_first = p99_last = None
    if n >= 30:
        third = n // 3
        first_third = sorted(short_latencies_ms[:third])
        last_third = sorted(short_latencies_ms[-third:])

        def p99(xs):
            return xs[max(0, int(len(xs) * 0.99) - 1)]

        p99_first, p99_last = p99(first_third), p99(last_third)
        assert p99_last <= p99_first * 1.25, f"P99 short-class latency drifted: {p99_first:.1f} -> {p99_last:.1f} ms"

    growth = end_rss - baseline_rss
    assert growth < 200 * 1024 * 1024, f"parent+worker RSS grew {growth / 1e6:.1f} MB over the run"

    out_dir = ROOT / "benchmarks" / "v0.3"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "duration_s": STRESS_SECONDS,
        "threads": THREADS,
        "requests_completed": lock_free_counter["n"],
        "errors": errors,
        "served": {f"{k[0]}@{k[1]}": v for k, v in sorted(devices.items())},
        "short_latency_p99_first_third_ms": p99_first,
        "short_latency_p99_last_third_ms": p99_last,
        "rss_growth_bytes": growth,
        "thermal_before": therm_before,
        "thermal_after": therm_after,
    }
    (out_dir / f"sustained-mixed-load-{int(time.time())}.json").write_text(json.dumps(report, indent=1))
