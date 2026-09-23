"""`laya-apple benchmark`: warm latency on exact-length requests (DEVELOPMENT_PLAN.md §11).

Two boundaries per configuration: `forward` (backend only, synchronised) and `predict`
(end to end: prompt build, routing, forward, answer formatting). Load time is reported
separately and excluded. Raw samples are kept in every record.
"""

from __future__ import annotations

import platform
import time

import numpy as np

from . import __version__


def stats(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def run(
    model_id,
    *,
    device="auto",
    lengths=(64, 128, 256),
    questions=1,
    warmup=5,
    iters=30,
    dtype="float16",
    local_files_only=False,
    laya=None,
):
    from .artifacts import platform_profile
    from .model import Laya
    from .workload import make_request

    t = time.perf_counter()
    if laya is None:
        laya = Laya.from_pretrained(model_id, device=device, dtype=dtype, local_files_only=local_files_only)
    load_s = time.perf_counter() - t
    records = []
    for length in lengths:
        rec = {
            "tool": "laya-apple benchmark",
            "laya_apple": __version__,
            "model": laya.spec.name,
            "device_requested": device,
            "dtype": dtype,
            "length": length,
            "questions": questions,
            "load_s": load_s,
            "platform": platform_profile() | {"python": platform.python_version()},
        }
        if length > laya.spec.max_len:
            rec["status"] = "skipped: exceeds checkpoint max_len"
            records.append(rec)
            continue
        state, qs = make_request(laya.tokenizer, laya.config, length, n_questions=questions)
        try:
            first = laya.predict(context=state, questions=qs)
        except Exception as e:  # e.g. UnsupportedShapeError on device="ane"; recorded, not hidden
            rec["status"] = f"error: {type(e).__name__}: {e}"
            records.append(rec)
            continue
        assert first.runtime.sequence_length == length, "workload did not hit the exact length"
        prep = laya.prepare(state, qs)
        backend = laya.ane if first.runtime.device == "ane" else laya.mlx
        for _ in range(warmup):
            backend.forward(prep.items)
            laya.predict(context=state, questions=qs)
        fwd, e2e = [], []
        for _ in range(iters):
            t = time.perf_counter()
            backend.forward(prep.items)
            fwd.append((time.perf_counter() - t) * 1000)
        for _ in range(iters):
            t = time.perf_counter()
            r = laya.predict(context=state, questions=qs)
            e2e.append((time.perf_counter() - t) * 1000)
        rec.update(
            status="ok",
            backend=r.runtime.backend,
            device=r.runtime.device,
            routing_reason=r.runtime.routing_reason,
            artifact_revision=r.runtime.artifact_revision,
            buckets=list(r.runtime.buckets),
            forward=stats(fwd),
            predict=stats(e2e),
            samples={"forward_ms": fwd, "predict_ms": e2e},
        )
        records.append(rec)
    return records
