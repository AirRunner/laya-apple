"""Derive the per-model ANE placement for execution="workers" from v0.2 measurements.

    uv run python scripts/derive_placement.py            # writes laya_apple/data/placement.json
    uv run python scripts/derive_placement.py --check    # fails if the committed file is stale

Inputs: benchmarks/v0.2/placement-<model>-{thread,process}.json, each the exit-gate mix
(scripts/bench_concurrency.py --part a) with the ANE on a thread in the caller's process or
in a worker process. The GPU is always a worker process.

Rule, per model: the placement with the lower short-stream P99 under the heterogeneous mix
(the isolation objective); ties within 5% go to the higher
aggregate throughput, then to "thread" (one process fewer).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "laya_apple/data/placement.json"
MODELS = ("laya", "laya-multilingual", "laya-typed-decisions")


def derive() -> dict:
    out = {
        "generated_by": "scripts/derive_placement.py",
        "rule": __doc__.split("Rule, per model: ")[1].strip(),
        "evidence": {},
        "models": {},
    }
    for model in MODELS:
        cand = {}
        for placement in ("thread", "process"):
            path = ROOT / f"benchmarks/v0.2/placement-{model}-{placement}.json"
            data = json.loads(path.read_text())
            out["evidence"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            a = data["part_a"]
            hetero = a["summary"]["hetero"]
            cand[placement] = {
                "short_p99_ms": hetero["short"]["p99_ms"],
                "short_req_s": hetero["short"]["req_s"],
                "long_p99_ms": hetero["long"]["p99_ms"],
                "long_req_s": hetero["long"]["req_s"],
                "aggregate_req_s": a["gate"]["aggregate_hetero_req_s"],
                "mismatches": a["gate"]["mismatches"],
            }
        t, p = cand["thread"], cand["process"]
        if abs(t["short_p99_ms"] - p["short_p99_ms"]) <= 0.05 * min(t["short_p99_ms"], p["short_p99_ms"]):
            choice = "thread" if t["aggregate_req_s"] >= p["aggregate_req_s"] else "process"
        else:
            choice = "thread" if t["short_p99_ms"] < p["short_p99_ms"] else "process"
        out["models"][model] = {"ane": choice, "gpu": "process", "measured": cand}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = json.dumps(derive(), indent=1) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f"{OUT} is stale; rerun scripts/derive_placement.py")
        print("placement.json is up to date")
        return
    OUT.write_text(text)
    for m, d in json.loads(text)["models"].items():
        print(
            m,
            "ane:",
            d["ane"],
            {k: (round(v["short_p99_ms"], 2), round(v["aggregate_req_s"], 1)) for k, v in d["measured"].items()},
        )


if __name__ == "__main__":
    main()
