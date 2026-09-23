"""v0.2 step-1 gate: can ONE process drive the MLX GPU and the ANE concurrently?

Phase -1 measured GPU+ANE concurrency only across separate processes (3.10-3.32x the best
single device). This experiment repeats the same short/long closed-loop mix through the
product runtime (laya_apple backends) under different execution models:

  solo            each stream alone, in this process
  threads         ane_short + gpu_long, one Python thread each, one process
  threads_gpu2    gpu_short + gpu_long, one thread each (same-device control)
  processes       ane_short + gpu_long, one spawned worker process each (Phase -1 model)

Every stream is closed-loop: the next request starts as soon as the previous returns.
Streams issue full `Laya.predict` calls (prompt build, routing, forward, formatting), since
that is what an in-process scheduler would execute. Conditions alternate order per cycle.

    LAYA_APPLE_CACHE=... .venv/bin/python research/v0.2-concurrency/scripts/inproc.py \
        --model laya-typed-decisions --short 128 --long 1024 --seconds 20 --cycles 3
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import resource
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"


def stats(values) -> dict:
    a = np.asarray(values, np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
        "max_ms": float(a.max()),
    }


def load_stream(model, device, length):
    from laya_apple import Laya
    from laya_apple.workload import make_request

    laya = Laya.from_pretrained(model, device=device, local_files_only=True)
    state, qs = make_request(laya.tokenizer, laya.config, length, n_questions=1, seed=0)
    r = laya.predict(context=state, questions=qs)
    assert r.runtime.device == device and r.runtime.sequence_length == length, r.runtime
    for _ in range(10):
        laya.predict(context=state, questions=qs)
    return laya, state, qs, r


def closed_loop(laya, state, qs, start_at, end_at, out, answers):
    while time.monotonic() < start_at:
        pass
    lat = []
    cpu0 = time.thread_time()
    first = None
    while True:
        t0 = time.monotonic()
        if t0 >= end_at:
            break
        r = laya.predict(context=state, questions=qs)
        lat.append((time.monotonic() - t0) * 1e3)
        if first is None:
            first = r.answers
        elif r.answers != first:
            answers["changed"] = True
    out.update(latency_ms=lat, wall_s=time.monotonic() - start_at, thread_cpu_s=time.thread_time() - cpu0,
               answers=first)


def worker(model, device, length, conn):
    laya, state, qs, r = load_stream(model, device, length)
    conn.send({"ready": True, "runtime": str(r.runtime)})
    while True:
        cmd = conn.recv()
        if cmd["op"] == "stop":
            return
        out, answers = {}, {}
        closed_loop(laya, state, qs, cmd["start_at"], cmd["start_at"] + cmd["seconds"], out, answers)
        out["answers_changed"] = bool(answers.get("changed"))
        out["process_cpu_s"] = time.process_time()
        conn.send(out)


def summarize(name, r):
    return {
        **stats(r["latency_ms"]),
        "requests": len(r["latency_ms"]),
        "requests_per_s": len(r["latency_ms"]) / r["wall_s"],
        "cpu_cores_used": r.get("thread_cpu_s", 0.0) / r["wall_s"],
        "answers_changed_under_load": bool(r.get("answers_changed")),
        "latency_ms": r["latency_ms"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya-typed-decisions")
    ap.add_argument("--short", type=int, default=128)
    ap.add_argument("--long", type=int, default=1024)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--output")
    args = ap.parse_args()

    import laya_apple
    from laya_apple.artifacts import platform_profile

    t = time.perf_counter()
    streams = {
        "ane_short": load_stream(args.model, "ane", args.short),
        "gpu_long": load_stream(args.model, "gpu", args.long),
        "gpu_short": load_stream(args.model, "gpu", args.short),
    }
    load_s = time.perf_counter() - t
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    refs = {n: s[3].answers for n, s in streams.items()}

    ctx = mp.get_context("spawn")
    workers = {}
    for name, dev, L in (("ane_short", "ane", args.short), ("gpu_long", "gpu", args.long)):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=worker, args=(args.model, dev, L, child), daemon=True)
        p.start()
        workers[name] = (parent, p)
    for name, (conn, _) in workers.items():
        conn.recv()

    conds = [
        ("solo", ["ane_short"]),
        ("solo", ["gpu_long"]),
        ("solo", ["gpu_short"]),
        ("threads", ["ane_short", "gpu_long"]),
        ("threads_gpu2", ["gpu_short", "gpu_long"]),
        ("processes", ["ane_short", "gpu_long"]),
    ]
    windows = []
    for cycle in range(args.cycles):
        for mode, names in conds if cycle % 2 == 0 else list(reversed(conds)):
            time.sleep(2.0)
            start_at = time.monotonic() + 0.5
            end_at = start_at + args.seconds
            load0 = os.getloadavg()[0]
            results = {}
            if mode == "processes":
                for n in names:
                    workers[n][0].send({"op": "run", "start_at": start_at, "seconds": args.seconds})
                results = {n: workers[n][0].recv() for n in names}
            else:
                outs = {n: {} for n in names}
                flags = {n: {} for n in names}
                cpu0 = time.process_time()
                threads = [
                    threading.Thread(target=closed_loop, args=(*streams[n][:3], start_at, end_at, outs[n], flags[n]))
                    for n in names
                ]
                for th in threads:
                    th.start()
                for th in threads:
                    th.join()
                proc_cpu = time.process_time() - cpu0
                for n in names:
                    outs[n]["answers_changed"] = bool(flags[n].get("changed")) or outs[n]["answers"] != refs[n]
                results = outs
            w = {
                "cycle": cycle,
                "mode": mode,
                "condition": "+".join(names),
                "loadavg_before": load0,
                "streams": {n: summarize(n, r) for n, r in results.items()},
            }
            if mode != "processes":
                w["process_cpu_cores_used"] = proc_cpu / args.seconds
            windows.append(w)
            print(cycle, mode, w["condition"],
                  {n: (round(s["requests_per_s"], 1), round(s["p50_ms"], 2), round(s["p99_ms"], 2))
                   for n, s in w["streams"].items()}, flush=True)
    for conn, p in workers.values():
        conn.send({"op": "stop"})
        p.join(timeout=30)

    record = {
        "experiment": "v0.2 in-process concurrency gate",
        "args": vars(args),
        "laya_apple": laya_apple.__version__,
        "platform": platform_profile() | {"python": platform.python_version()},
        "time": datetime.now(timezone.utc).isoformat(),
        "load_s": load_s,
        "peak_rss_bytes_after_load": rss_after_load,
        "peak_rss_bytes_end": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "streams": {n: str(s[3].runtime) for n, s in streams.items()},
        "windows": windows,
    }
    out = Path(args.output or RAW / f"inproc-{args.model}-S{args.short}-L{args.long}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
