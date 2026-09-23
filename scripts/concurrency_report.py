"""Render the v0.2 benchmark tables (markdown) from scripts/bench_concurrency.py outputs.

    uv run python scripts/concurrency_report.py

Part A (closed loop, exit gate): benchmarks/v0.2/placement-<model>-{thread,process}.json.
Part B (open loop): benchmarks/v0.2/concurrency-<model>.json.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B2 = ROOT / "benchmarks/v0.2"
MODELS = ("laya", "laya-multilingual", "laya-typed-decisions")
# Best GPU-only aggregate measured anywhere for the same mix (MLX streams on separate
# threads or processes; research/v0.2-concurrency, Phase -1): the stricter baseline.
RESEARCH_GPU_ONLY = {"laya-typed-decisions": 34.4, "laya-multilingual": 92.3}


def main():
    placement = json.loads((ROOT / "laya_apple/data/placement.json").read_text())["models"]
    print("### Part A: closed-loop exit-gate mix\n")
    print(
        "| Model (short/long L) | ANE placement | chosen | short req/s (solo) | short P99 ms (solo) | Δ P99 "
        "| long req/s (solo) | long P99 ms (solo) | Δ P99 | aggregate req/s | vs GPU-only (product) "
        "| vs best GPU-only | answer mismatches |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODELS:
        for pl in ("thread", "process"):
            d = json.loads((B2 / f"placement-{model}-{pl}.json").read_text())
            a, s = d["part_a"], d["part_a"]["summary"]
            h, ss, sl, g = s["hetero"], s["solo_short"]["short"], s["solo_long"]["long"], s["gpu_only"]
            gpu_only = sum(v["req_s"] for v in g.values())
            best = max(gpu_only, RESEARCH_GPU_ONLY.get(model, 0.0))
            agg = a["gate"]["aggregate_hetero_req_s"]
            print(
                f"| {model} ({d['args']['short']}/{d['args']['long']}) | {pl} | {'**yes**' if placement[model]['ane'] == pl else ''} "
                f"| {h['short']['req_s']:.1f} ({ss['req_s']:.1f}) | {h['short']['p99_ms']:.2f} ({ss['p99_ms']:.2f}) "
                f"| {a['gate']['p99_vs_solo']['short']:+.0%} | {h['long']['req_s']:.1f} ({sl['req_s']:.1f}) "
                f"| {h['long']['p99_ms']:.1f} ({sl['p99_ms']:.1f}) | {a['gate']['p99_vs_solo']['long']:+.0%} "
                f"| {agg:.1f} | {agg / gpu_only:.2f}× | {agg / best:.2f}× | {a['gate']['mismatches']} |"
            )
    print("\n### Part B: open-loop mixed workload (latency from arrival, queueing included)\n")
    for model in MODELS:
        d = json.loads((B2 / f"concurrency-{model}.json").read_text())
        print(f"#### {model} (ANE placement: {d['auto_info'].get('ane_placement')})\n")
        print("| arrivals | offered req/s | config | class | n | P50 ms | P95 ms | P99 ms | devices | mismatches |")
        print("|---|---:|---|---|---:|---:|---:|---:|---|---:|")
        for key, runs in d["part_b"].items():
            pattern, rate = key.split("@")
            for cfg, r in runs.items():
                for cls, c in r["classes"].items():
                    print(
                        f"| {pattern} | {r['offered_req_s']:.1f} | {cfg} | {cls} | {c['n']} | {c['p50_ms']:.1f} "
                        f"| {c['p95_ms']:.1f} | {c['p99_ms']:.1f} | {c['devices']} | {c['mismatches']} |"
                    )
        print()


if __name__ == "__main__":
    main()
