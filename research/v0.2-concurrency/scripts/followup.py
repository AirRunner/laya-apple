"""Follow-up to inproc.py: where does the in-process thread penalty come from?

inproc.py showed that one process with two threads (ANE short + GPU long) costs the GPU
stream ~11% throughput and ~12% P99, while two processes cost nothing. Conditions here:

  solo_*            references, same run
  threads           both streams full predict(), threads (repeat of inproc.py)
  threads_forward   both streams backend.forward() only on pre-built rows (no prompt
                    building, routing or answer formatting in the loop)
  hybrid_gpu_proc   ANE stream in this process (thread), GPU stream in a worker process
  hybrid_ane_proc   GPU stream in this process (thread), ANE stream in a worker process

    LAYA_APPLE_CACHE=... .venv/bin/python research/v0.2-concurrency/scripts/followup.py --model laya-typed-decisions
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from inproc import RAW, closed_loop, load_stream, summarize, worker


def forward_loop(backend, items, start_at, end_at, out, _flags):
    while time.monotonic() < start_at:
        pass
    lat = []
    cpu0 = time.thread_time()
    while True:
        t0 = time.monotonic()
        if t0 >= end_at:
            break
        backend.forward(items)
        lat.append((time.monotonic() - t0) * 1e3)
    out.update(latency_ms=lat, wall_s=time.monotonic() - start_at, thread_cpu_s=time.thread_time() - cpu0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya-typed-decisions")
    ap.add_argument("--short", type=int, default=128)
    ap.add_argument("--long", type=int, default=1024)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--cycles", type=int, default=3)
    args = ap.parse_args()

    import laya_apple
    from laya_apple.artifacts import platform_profile

    local = {"ane_short": load_stream(args.model, "ane", args.short), "gpu_long": load_stream(args.model, "gpu", args.long)}
    fwd = {}
    for n, (laya, state, qs, _) in local.items():
        backend = laya.ane if n.startswith("ane") else laya.mlx
        fwd[n] = (backend, laya.prepare(state, qs).items)

    ctx = mp.get_context("spawn")
    workers = {}
    for name, dev, L in (("ane_short", "ane", args.short), ("gpu_long", "gpu", args.long)):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=worker, args=(args.model, dev, L, child), daemon=True)
        p.start()
        workers[name] = (parent, p)
    for conn, _ in workers.values():
        conn.recv()

    conds = [
        ("solo_ane", {"ane_short": "thread"}),
        ("solo_gpu", {"gpu_long": "thread"}),
        ("solo_ane_forward", {"ane_short": "forward"}),
        ("solo_gpu_forward", {"gpu_long": "forward"}),
        ("threads", {"ane_short": "thread", "gpu_long": "thread"}),
        ("threads_forward", {"ane_short": "forward", "gpu_long": "forward"}),
        ("hybrid_gpu_proc", {"ane_short": "thread", "gpu_long": "proc"}),
        ("hybrid_ane_proc", {"ane_short": "proc", "gpu_long": "thread"}),
    ]
    windows = []
    for cycle in range(args.cycles):
        for mode, plan in conds if cycle % 2 == 0 else list(reversed(conds)):
            time.sleep(2.0)
            start_at = time.monotonic() + 0.5
            end_at = start_at + args.seconds
            outs, flags, threads = {}, {}, []
            for n, how in plan.items():
                outs[n], flags[n] = {}, {}
                if how == "proc":
                    workers[n][0].send({"op": "run", "start_at": start_at, "seconds": args.seconds})
                elif how == "thread":
                    threads.append(threading.Thread(target=closed_loop, args=(*local[n][:3], start_at, end_at, outs[n], flags[n])))
                else:
                    threads.append(threading.Thread(target=forward_loop, args=(*fwd[n], start_at, end_at, outs[n], flags[n])))
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            for n, how in plan.items():
                if how == "proc":
                    outs[n] = workers[n][0].recv()
            w = {"cycle": cycle, "mode": mode, "plan": plan, "streams": {n: summarize(n, r) for n, r in outs.items()}}
            windows.append(w)
            print(cycle, mode, {n: (round(s["requests_per_s"], 1), round(s["p50_ms"], 2), round(s["p99_ms"], 2))
                                for n, s in w["streams"].items()}, flush=True)
    for conn, p in workers.values():
        conn.send({"op": "stop"})
        p.join(timeout=30)
    out = RAW / f"followup-{args.model}-S{args.short}-L{args.long}.json"
    out.write_text(json.dumps({
        "experiment": "v0.2 concurrency follow-up: source of the thread penalty",
        "args": vars(args),
        "laya_apple": laya_apple.__version__,
        "platform": platform_profile() | {"python": platform.python_version()},
        "time": datetime.now(timezone.utc).isoformat(),
        "windows": windows,
    }, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
