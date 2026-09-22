"""Latency benchmark for one (model, backend configuration) in a fresh process.

    uv run python scripts/bench.py --model laya-typed-decisions \
        --spec '{"backend":"mlx","dtype":"float16"}' --lengths 64 128 256 --questions 1 4 \
        --tag pass-a --output raw/bench/latency.jsonl

For every (length, questions) cell: build an exact-length request, 10 warmup calls, then
a time-budgeted number of samples (>=50, <=300, ~6 s) of BOTH timing boundaries:
  forward   prepared rows -> synchronised logits (model-only, see methodology §4)
  predict   public end-to-end call
Fixed-shape Core ML backends load the package for that exact length (`length` in spec).
Cold init (load seconds, first call) is recorded separately per loaded model.
One JSON line per cell with every raw sample.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from backends import make_backend  # noqa: E402
from common import (  # noqa: E402
    agent_config,
    append_jsonl,
    checkpoint,
    conditions,
    environment,
    make_request,
    peak_rss_bytes,
    rss_bytes,
    stats,
    valid_lengths,
)


def timed(fn, warmup, budget_s, lo, hi):
    for _ in range(warmup):
        fn()
    first = []
    t_end = time.perf_counter() + budget_s
    while True:
        t = time.perf_counter_ns()
        fn()
        first.append((time.perf_counter_ns() - t) / 1e6)
        if len(first) >= hi or (len(first) >= lo and time.perf_counter() > t_end):
            break
    return first


def memory_of(b):
    m = {"rss": rss_bytes(), "peak_rss": peak_rss_bytes()}
    if hasattr(b, "memory"):
        m.update(b.memory())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--lengths", type=int, nargs="*")
    ap.add_argument("--questions", type=int, nargs="*", default=[1])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--budget", type=float, default=6.0)
    ap.add_argument("--min-samples", type=int, default=50)
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--modes", nargs="*", default=["forward", "predict"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    spec = {**json.loads(args.spec), "model": args.model}
    lengths = args.lengths or valid_lengths(args.model)
    lengths = [L for L in lengths if L in valid_lengths(args.model, lengths)]
    from laya_coreml.tokenizer import Tokenizer

    tok = Tokenizer(checkpoint(args.model) / "tokenizer")
    cfg = agent_config(args.model)
    env = environment()
    per_length_backend = spec["backend"] in ("ane",) or (
        spec["backend"] == "coreml" and not spec.get("enumerated")
    )
    rss0 = rss_bytes()
    shared = None
    shared_init = None
    if not per_length_backend:
        t = time.perf_counter()
        shared = make_backend({k: v for k, v in spec.items() if k != "enumerated"})
        shared_init = {"load_s": time.perf_counter() - t}
    for L in lengths:
        if per_length_backend:
            gc.collect()
            t = time.perf_counter()
            b = make_backend({**spec, "length": L})
            init = {"load_s": time.perf_counter() - t}
        else:
            b, init = shared, dict(shared_init)
        for nq in args.questions:
            state, questions, items = make_request(tok, cfg, L, nq, seed=0)
            if hasattr(b, "reset_peak"):
                b.reset_peak()
            t = time.perf_counter_ns()
            b.forward(items)
            first_ms = (time.perf_counter_ns() - t) / 1e6
            rec = {
                "experiment": "latency",
                "tag": args.tag,
                "model": args.model,
                "spec": spec,
                "describe": b.describe(),
                "length": L,
                "questions": nq,
                "row_lengths": [len(i["ids"]) for i in items],
                "cold": {**init, "first_forward_ms": first_ms},
                "conditions_before": conditions(),
            }
            for mode in args.modes:
                if mode == "forward":
                    fn = lambda: b.forward(items)  # noqa: E731
                else:
                    fn = lambda: b.predict(state, questions)  # noqa: E731
                samples = timed(fn, args.warmup, args.budget, args.min_samples, args.max_samples)
                rec[mode] = {**stats(samples), "samples_ms": samples}
                rec[mode]["requests_per_s"] = 1000 * len(samples) / sum(samples)
                rec[mode]["decisions_per_s"] = rec[mode]["requests_per_s"] * nq
            lg, _ = b.forward(items)
            rec["logits_row0"] = lg[0][: len(items[0]["markers"])].tolist()
            rec["memory"] = {**memory_of(b), "rss_before_load": rss0}
            rec["conditions_after"] = conditions()
            rec["environment"] = env if L == lengths[0] and nq == args.questions[0] else None
            append_jsonl(args.output, rec)
            f, p = rec.get("forward", {}), rec.get("predict", {})
            print(
                f"{args.model} {b.describe().get('graph', spec['backend'])} "
                f"{spec.get('units', spec.get('dtype', spec.get('device', '')))} L{L} q{nq}: "
                f"fwd p50 {f.get('p50_ms', float('nan')):.2f} p95 {f.get('p95_ms', float('nan')):.2f} "
                f"(n={f.get('n')}) | e2e p50 {p.get('p50_ms', float('nan')):.2f} | load {init['load_s']:.1f}s",
                flush=True,
            )
        if per_length_backend:
            del b


if __name__ == "__main__":
    main()
