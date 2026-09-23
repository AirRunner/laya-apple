"""Summarise inproc/followup raw windows: per condition, the median over cycles of each
stream's req/s and P99, the change against the same stream solo, aggregate req/s, and the
v0.2 exit-gate checks (aggregate >= 2.5x best single device; each stream's P99 <= +10% of solo).

    .venv/bin/python research/v0.2-concurrency/scripts/summarize.py raw/inproc-*.json
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SOLO = {"ane_short": ("solo", "solo_ane"), "gpu_long": ("solo", "solo_gpu"), "gpu_short": ("solo",)}


def summarize(path: Path) -> dict:
    d = json.loads(path.read_text())
    by = defaultdict(lambda: defaultdict(list))
    for w in d["windows"]:
        mode = w["mode"]
        cond = mode if mode not in ("solo",) else f"solo:{'+'.join(w['streams'])}"
        for n, s in w["streams"].items():
            by[cond][n].append((s["requests_per_s"], s["p99_ms"], s["p50_ms"], s.get("answers_changed_under_load", False)))
    med = {
        c: {
            n: {
                "req_s": statistics.median(v[0] for v in vals),
                "p99_ms": statistics.median(v[1] for v in vals),
                "p50_ms": statistics.median(v[2] for v in vals),
                "answers_changed": any(v[3] for v in vals),
                "cycles": len(vals),
            }
            for n, vals in streams.items()
        }
        for c, streams in by.items()
    }
    solo = {}
    for c, streams in med.items():
        if c.startswith("solo:") or c in ("solo_ane", "solo_gpu"):
            for n, s in streams.items():
                solo.setdefault(n, s)
    out = {"file": path.name, "model": d["args"]["model"], "conditions": {}}
    gpu_only = None
    if "threads_gpu2" in med:
        gpu_only = sum(s["req_s"] for s in med["threads_gpu2"].values())
    for c, streams in med.items():
        row = {}
        for n, s in streams.items():
            ref = solo.get(n)
            row[n] = dict(s)
            if ref and not c.startswith("solo"):
                row[n]["req_s_vs_solo"] = s["req_s"] / ref["req_s"] - 1
                row[n]["p99_vs_solo"] = s["p99_ms"] / ref["p99_ms"] - 1
        entry = {"streams": row, "aggregate_req_s": sum(s["req_s"] for s in streams.values())}
        if gpu_only and len(streams) == 2 and "ane_short" in streams:
            entry["vs_gpu_only"] = entry["aggregate_req_s"] / gpu_only
            entry["gate_throughput"] = entry["vs_gpu_only"] >= 2.5
            entry["gate_p99"] = all(r.get("p99_vs_solo", 0) <= 0.10 for r in row.values())
            entry["gate_correctness"] = not any(r["answers_changed"] for r in row.values())
        out["conditions"][c] = entry
    out["gpu_only_aggregate_req_s"] = gpu_only
    return out


def main():
    for p in sys.argv[1:]:
        s = summarize(Path(p))
        print(f"\n## {s['file']}  (GPU-only aggregate {s['gpu_only_aggregate_req_s']})")
        for c, e in s["conditions"].items():
            parts = []
            for n, r in e["streams"].items():
                extra = ""
                if "req_s_vs_solo" in r:
                    extra = f" ({r['req_s_vs_solo']:+.1%} req/s, {r['p99_vs_solo']:+.1%} P99)"
                parts.append(f"{n} {r['req_s']:.1f} req/s P50 {r['p50_ms']:.2f} P99 {r['p99_ms']:.2f}{extra}")
            gate = ""
            if "vs_gpu_only" in e:
                gate = (f" | agg {e['aggregate_req_s']:.1f} = {e['vs_gpu_only']:.2f}x GPU-only; "
                        f"throughput {'PASS' if e['gate_throughput'] else 'FAIL'}, "
                        f"P99 {'PASS' if e['gate_p99'] else 'FAIL'}, "
                        f"answers {'PASS' if e['gate_correctness'] else 'FAIL'}")
            print(f"- {c}: " + "; ".join(parts) + gate)
        Path(p).with_name(Path(p).stem + ".summary.json").write_text(json.dumps(s, indent=1) + "\n")


if __name__ == "__main__":
    main()
