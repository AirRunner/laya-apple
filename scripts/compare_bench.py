"""Reproducibility check: compare a benchmark re-run with the recorded release data.

    # single-request latency (scripts/release_bench.py outputs)
    uv run python scripts/compare_bench.py latency benchmarks/v0.1/raw.jsonl benchmarks/v1.0/raw.jsonl

    # concurrency exit-gate mix, part A (scripts/bench_concurrency.py --part a outputs)
    uv run python scripts/compare_bench.py concurrency benchmarks/v0.2 benchmarks/v1.0

Prints Markdown tables with the relative change of each headline figure, then the largest
absolute change.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("laya", "laya-multilingual", "laya-typed-decisions")


def _latency_table(path: Path) -> dict:
    """{(model, device, L, q): (forward P50, predict P50)}, each the mean over passes."""
    acc = defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        key = (r["model"], r["device_requested"], r["length"], r["questions"])
        acc[key].append((r["forward"]["p50_ms"], r["predict"]["p50_ms"]))
    return {k: tuple(sum(x) / len(v) for x in zip(*v)) for k, v in acc.items()}


def latency(base: Path, new: Path) -> float:
    a, b = _latency_table(base), _latency_table(new)
    print("| Model | Device | L | q | forward P50 base → new (ms) | Δ | predict P50 base → new (ms) | Δ |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    worst = 0.0
    for key in sorted(set(a) & set(b)):
        (fa, pa), (fb, pb) = a[key], b[key]
        df, dp = fb / fa - 1, pb / pa - 1
        worst = max(worst, abs(df), abs(dp))
        print(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {fa:.2f} → {fb:.2f} | {df:+.1%} | {pa:.2f} → {pb:.2f} | {dp:+.1%} |"
        )
    missing = sorted(set(a) ^ set(b))
    if missing:
        print(f"\nIn only one run: {missing}")
    print(f"\nLargest absolute change: {worst:.1%} over {len(set(a) & set(b))} configurations")
    return worst


def _part_a(d: dict) -> dict:
    a, s = d["part_a"], d["part_a"]["summary"]
    gpu_only = sum(v["req_s"] for v in s["gpu_only"].values())
    return {
        "aggregate_req_s": a["gate"]["aggregate_hetero_req_s"],
        "vs_gpu_only": a["gate"]["aggregate_hetero_req_s"] / gpu_only,
        "short_p99_ms": s["hetero"]["short"]["p99_ms"],
        "long_req_s": s["hetero"]["long"]["req_s"],
        "mismatches": a["gate"]["mismatches"],
    }


def concurrency(base_dir: Path, new_dir: Path) -> float:
    placement = json.loads((ROOT / "laya_apple/data/placement.json").read_text())["models"]
    print(
        "| Model | ANE placement | aggregate req/s base → new | vs GPU-only base → new | short P99 ms base → new | long req/s base → new | mismatches (new) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")
    worst = 0.0
    for model in MODELS:
        pl = placement[model]["ane"]
        new_path = new_dir / f"placement-{model}-{pl}.json"
        if not new_path.exists():
            continue
        x = _part_a(json.loads((base_dir / f"placement-{model}-{pl}.json").read_text()))
        y = _part_a(json.loads(new_path.read_text()))
        d = y["aggregate_req_s"] / x["aggregate_req_s"] - 1
        worst = max(worst, abs(d))
        print(
            f"| {model} | {pl} | {x['aggregate_req_s']:.1f} → {y['aggregate_req_s']:.1f} ({d:+.1%}) "
            f"| {x['vs_gpu_only']:.2f}× → {y['vs_gpu_only']:.2f}× | {x['short_p99_ms']:.2f} → {y['short_p99_ms']:.2f} "
            f"| {x['long_req_s']:.1f} → {y['long_req_s']:.1f} | {y['mismatches']} |"
        )
    print(f"\nLargest aggregate-throughput change: {worst:.1%}")
    return worst


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="kind", required=True)
    s = sub.add_parser("latency")
    s.add_argument("base", type=Path)
    s.add_argument("new", type=Path)
    s = sub.add_parser("concurrency")
    s.add_argument("base", type=Path)
    s.add_argument("new", type=Path)
    a = p.parse_args(argv)
    (latency if a.kind == "latency" else concurrency)(a.base, a.new)


if __name__ == "__main__":
    main()
