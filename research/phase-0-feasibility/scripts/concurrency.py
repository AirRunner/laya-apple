"""GPU + ANE request-level concurrency experiment (Experiment 5).

Persistent worker processes each own one backend and one request stream; the orchestrator
runs timed windows in which a chosen subset of workers issue closed-loop requests (next
request as soon as the previous one returns). Conditions alternate over several cycles.

    uv run python scripts/concurrency.py --model laya-typed-decisions --short 128 --long 1024 \
        --seconds 20 --cycles 3 --output raw/concurrency/typed-decisions.json

Streams:
  ane_short   Core ML BC1S graph, CPU_AND_NE, short request (L=--short)
  gpu_long    MLX FP16, long request (L=--long)
  gpu_short   MLX FP16, short request (second MLX process)
  ane_long    Core ML BC1S graph, CPU_AND_NE, long request (only with --ane-long)
Conditions (each a window of --seconds):
  solo:        every stream alone
  hetero:      ane_short + gpu_long          (the proposed heterogeneous split)
  gpu_gpu:     gpu_short + gpu_long          (same-device control, two processes)
  ane_ane:     ane_short + ane_long          (same-device control, optional)
Separate processes model the best case for a runtime (no GIL, independent queues).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def worker(name, spec, length, questions, conn):
    sys.path.insert(0, str(Path(__file__).parent))
    import numpy as np  # noqa: F401

    from backends import make_backend
    from common import agent_config, checkpoint, make_request, rss_bytes
    from laya_coreml.tokenizer import Tokenizer

    t = time.perf_counter()
    b = make_backend({**spec, "length": length} if spec["backend"] == "ane" else spec)
    load_s = time.perf_counter() - t
    tok = Tokenizer(checkpoint(spec["model"]) / "tokenizer")
    state, qs, items = make_request(tok, agent_config(spec["model"]), length, questions, seed=0)
    for _ in range(10):
        b.predict(state, qs)
    conn.send({"ready": name, "load_s": load_s, "describe": b.describe(), "rss": rss_bytes()})
    while True:
        cmd = conn.recv()
        if cmd["op"] == "stop":
            break
        # all participants start at the same absolute monotonic time
        start_at, seconds = cmd["start_at"], cmd["seconds"]
        while time.monotonic() < start_at:
            pass
        end_at = start_at + seconds
        lat, stamps = [], []
        cpu0 = time.process_time()
        while True:
            t0 = time.monotonic()
            if t0 >= end_at:
                break
            b.predict(state, qs)
            t1 = time.monotonic()
            lat.append((t1 - t0) * 1e3)
            stamps.append(t0 - start_at)
        wall = time.monotonic() - start_at
        conn.send(
            {
                "name": name,
                "latency_ms": lat,
                "start_offsets_s": stamps,
                "wall_s": wall,
                "cpu_s": time.process_time() - cpu0,
                "rss": rss_bytes(),
            }
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya-typed-decisions")
    ap.add_argument("--short", type=int, default=128)
    ap.add_argument("--long", type=int, default=1024)
    ap.add_argument("--questions", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--ane-long", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    from common import conditions, environment, save_json, stats

    m = args.model
    streams = {
        "ane_short": ({"backend": "ane", "units": "cpu_ne", "model": m}, args.short),
        "gpu_long": ({"backend": "mlx", "dtype": "float16", "model": m}, args.long),
        "gpu_short": ({"backend": "mlx", "dtype": "float16", "model": m}, args.short),
    }
    if args.ane_long:
        streams["ane_long"] = ({"backend": "ane", "units": "cpu_ne", "model": m}, args.long)
    conditions_list = [[s] for s in streams] + [["ane_short", "gpu_long"], ["gpu_short", "gpu_long"]]
    if args.ane_long:
        conditions_list.append(["ane_short", "ane_long"])

    ctx = mp.get_context("spawn")
    pipes, procs, ready = {}, {}, {}
    for name, (spec, L) in streams.items():
        parent, child = ctx.Pipe()
        p = ctx.Process(target=worker, args=(name, spec, L, args.questions, child), daemon=True)
        p.start()
        pipes[name], procs[name] = parent, p
    for name in streams:
        ready[name] = pipes[name].recv()
        print("ready", ready[name]["ready"], f"load {ready[name]['load_s']:.1f}s", flush=True)

    windows = []
    for cycle in range(args.cycles):
        order = conditions_list if cycle % 2 == 0 else list(reversed(conditions_list))
        for cond in order:
            time.sleep(2.0)  # settle between windows
            start_at = time.monotonic() + 0.5
            before = conditions()
            for name in cond:
                pipes[name].send({"op": "run", "start_at": start_at, "seconds": args.seconds})
            results = {name: pipes[name].recv() for name in cond}
            w = {"cycle": cycle, "condition": "+".join(cond), "streams": {}, "conditions_before": before}
            for name, r in results.items():
                w["streams"][name] = {
                    **stats(r["latency_ms"]),
                    "requests": len(r["latency_ms"]),
                    "requests_per_s": len(r["latency_ms"]) / r["wall_s"],
                    "cpu_cores_used": r["cpu_s"] / r["wall_s"],
                    "rss": r["rss"],
                    "latency_ms": r["latency_ms"],
                    "start_offsets_s": r["start_offsets_s"],
                }
            windows.append(w)
            print(
                cycle,
                w["condition"],
                {n: (round(s["requests_per_s"], 1), round(s["p50_ms"], 2), round(s["p99_ms"], 2)) for n, s in w["streams"].items()},
                flush=True,
            )
    for name in streams:
        pipes[name].send({"op": "stop"})
    for p in procs.values():
        p.join(timeout=30)

    save_json(
        args.output,
        {
            "experiment": "concurrency",
            "args": vars(args),
            "streams": {n: {"spec": s, "length": L, **ready[n]} for n, (s, L) in streams.items()},
            "windows": windows,
            "environment": environment(),
        },
    )


if __name__ == "__main__":
    main()
