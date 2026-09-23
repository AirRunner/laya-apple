"""Render release-benchmark tables (markdown) from `laya-apple benchmark` JSONL records.

    uv run python scripts/bench_report.py benchmarks/v0.1/raw.jsonl

Per configuration: forward and predict P50/P99 from pass 1 and pass 2 (reversed order),
the mean of the two P50s, and the pass-to-pass difference as an order/noise indicator.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def main():
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") != "ok"]
    by = defaultdict(dict)
    for r in ok:
        by[(r["model"], r["device_requested"], r["length"], r["questions"])][r["pass"]] = r
    plat = ok[0]["platform"] if ok else {}
    print(
        f"Platform: {plat.get('soc')}, macOS {plat.get('macos')} ({plat.get('macos_build')}), "
        f"coremltools {plat.get('coremltools')}, Python {plat.get('python')}; laya-apple {ok[0]['laya_apple'] if ok else '?'}"
    )
    print(f"Records: {len(ok)} ok, {len(bad)} not ok; samples per record: {ok[0]['forward']['n'] if ok else 0}\n")
    for model in sorted({k[0] for k in by}):
        print(f"### {model}\n")
        print(
            "| device | L | q | ran on | reason | forward P50 ms | forward P99 ms | predict P50 ms | predict P99 ms "
            "| e2e overhead ms | pass Δ P50 |"
        )
        print("|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|")
        order = {"gpu": 0, "ane": 1, "auto": 2}
        for key in sorted((k for k in by if k[0] == model), key=lambda k: (k[3], k[2], order[k[1]])):
            passes = by[key]
            rs = [passes[p] for p in sorted(passes)]
            f50 = sum(r["forward"]["p50_ms"] for r in rs) / len(rs)
            f99 = max(r["forward"]["p99_ms"] for r in rs)
            p50 = sum(r["predict"]["p50_ms"] for r in rs) / len(rs)
            p99 = max(r["predict"]["p99_ms"] for r in rs)
            delta = ""
            if len(rs) == 2:
                a, b = rs[0]["forward"]["p50_ms"], rs[1]["forward"]["p50_ms"]
                delta = f"{(b - a) / a:+.1%}"
            r0 = rs[0]
            print(
                f"| {key[1]} | {key[2]} | {key[3]} | {r0['device']} | `{r0['routing_reason']}` | {f50:.2f} | {f99:.2f} "
                f"| {p50:.2f} | {p99:.2f} | {p50 - f50:.2f} | {delta} |"
            )
        print()
    if bad:
        print("### Not ok\n")
        for r in bad:
            print(f"- {r['model']} {r['device_requested']} L{r['length']} q{r['questions']}: {r['status']}")


if __name__ == "__main__":
    main()
