"""Render the v1.0 benchmark tables (Markdown) from the raw v1.0 data in benchmarks/v1.0/.

    uv run python scripts/v1_report.py [latency|e2e|parity|hetero|openloop|all]

Every number in benchmarks/v1.0.md and in the README's benchmark tables is produced here
from these raw files:
- raw.jsonl: laya-apple latency (scripts/release_bench.py);
- comparators-latency.jsonl and parity/: the comparators (scripts/bench_v1.sh);
- placement-*.json: the closed-loop mix;
- concurrency-*.json: the open-loop mix (scripts/bench_concurrency.py).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "benchmarks" / "v1.0"
MODELS = ("laya", "laya-multilingual", "laya-typed-decisions")
LENGTHS = (64, 128, 256, 512, 1024)
PLACEMENT = {"laya": "thread", "laya-multilingual": "process", "laya-typed-decisions": "thread"}


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def laya_apple_latency():
    """{(model, device_requested, L, q): {forward, predict: {p50,p95,p99}}}, P50 = mean over passes,
    P95/P99 = max over passes (as in the v0.1 report)."""
    acc = defaultdict(list)
    for r in _jsonl(V1 / "raw.jsonl"):
        acc[(r["model"], r["device_requested"], r["length"], r["questions"])].append(r)
    out = {}
    for k, rs in acc.items():
        out[k] = {
            b: {
                "p50": sum(r[b]["p50_ms"] for r in rs) / len(rs),
                "p95": max(r[b]["p95_ms"] for r in rs),
                "p99": max(r[b]["p99_ms"] for r in rs),
            }
            for b in ("forward", "predict")
        }
    return out


def comparator_latency():
    """{(model, label, L): forward P50} for the comparators."""
    out = {}
    for r in _jsonl(V1 / "comparators-latency.jsonl"):
        d = r["describe"]
        if d["backend"] == "torch":
            label = f"PyTorch {r['spec']['device'].upper()} FP32"
        elif d.get("graph") == "ordinary-enumerated":
            label = "Core ML ordinary, enumerated · CPU_AND_GPU"
        else:
            label = f"Core ML ordinary, fixed · {d['compute_units']}"
        out[(r["model"], label, r["length"])] = r["forward"]["p50_ms"]
    return out


def _parity(model: str, name: str):
    p = V1 / "parity" / model / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


COMPARATOR_ROWS = (
    ("PyTorch CPU FP32", None),
    ("PyTorch MPS FP32", "torch_device-mps"),
    ("Core ML ordinary, fixed · CPU_AND_NE", "coreml_units-cpu_ne"),
    ("Core ML ordinary, fixed · CPU_AND_GPU", "coreml_units-cpu_gpu"),
    ("Core ML ordinary, enumerated · CPU_AND_GPU", None),
)


def section_latency():
    la, cmp_ = laya_apple_latency(), comparator_latency()
    print("Forward P50, ms, one question per request (the model call only; see `e2e` for user-visible latency).\n")
    for m in MODELS:
        lengths = [L for L in LENGTHS if any(k[0] == m and k[2] == L for k in la)]
        print(f"#### {m}\n")
        print("| Implementation | " + " | ".join(f"L{L}" for L in lengths) + " | Correct? |")
        print("|---|" + "---:|" * len(lengths) + "---|")
        for label, parity_name in COMPARATOR_ROWS:
            cells = [f"{cmp_[(m, label, L)]:.1f}" if (m, label, L) in cmp_ else "—" for L in lengths]
            ok = (
                "reference"
                if label.startswith("PyTorch CPU")
                else _verdict(_parity(m, parity_name))
                if parity_name
                else "see note"
            )
            print(f"| {label} | " + " | ".join(cells) + f" | {ok} |")
        mlx = [f"{la[(m, 'gpu', L, 1)]['forward']['p50']:.1f}" if (m, "gpu", L, 1) in la else "—" for L in lengths]
        ane = [f"**{la[(m, 'ane', L, 1)]['forward']['p50']:.1f}**" if (m, "ane", L, 1) in la else "—" for L in lengths]
        print(
            "| laya-apple MLX FP16 (GPU) | "
            + " | ".join(mlx)
            + f" | {_verdict(_parity(m, 'laya-apple-gpu-float16'))} |"
        )
        print(
            "| laya-apple ANE (BC1S, CPU_AND_NE) | "
            + " | ".join(ane)
            + f" | {_verdict(_parity(m, 'laya-apple-ane-float16'))} |"
        )
        print()


def _verdict(p) -> str:
    if p is None:
        return "not measured"
    s = p.get("summary", p)
    hard, near = _mismatches(p)
    mark = "✅" if p.get("passed") else "❌"
    return f"{mark} {hard} hard, {near} near-tie, prob err {_num(s['prob_max_abs'])}"


def _mismatches(p) -> tuple[int, int]:
    """(hard mismatches, near-tie flips). The research harness records decision_mismatches
    (all) and ..._outside_near_tie (hard); laya-apple records hard_mismatches and the flips."""
    s = p.get("summary", p)
    if "decision_mismatches_outside_near_tie" in s:
        hard = s["decision_mismatches_outside_near_tie"]
        return hard, s["decision_mismatches"] - hard
    return p["hard_mismatches"], len(p.get("near_tie_flips", []))


def _num(x) -> str:
    """Probability errors are floats, except where the backend produced NaN (recorded as "nan")."""
    return f"{x:.3g}" if isinstance(x, (int, float)) else str(x)


def section_e2e():
    la = laya_apple_latency()
    print('laya-apple `device="auto"`, one fresh process per configuration. P50 / P99 in ms.')
    print("`forward` is the model call; `predict` is end to end: prompt build, tokenization, routing,")
    print("tensor creation, inference, calibration and answer formatting.\n")
    print("| Model | L | q | device chosen | forward P50 | predict P50 | predict P99 |")
    print("|---|---:|---:|---|---:|---:|---:|")
    raw = {
        (r["model"], r["length"], r["questions"]): r["device"]
        for r in _jsonl(V1 / "raw.jsonl")
        if r["device_requested"] == "auto"
    }
    for (m, dev, L, q), v in sorted(la.items()):
        if dev != "auto":
            continue
        print(
            f"| {m} | {L} | {q} | {raw[(m, L, q)]} | {v['forward']['p50']:.2f} | {v['predict']['p50']:.2f} | {v['predict']['p99']:.2f} |"
        )
    print()


def section_parity():
    print("Against upstream Laya on PyTorch CPU FP32 (the golden rows shipped in `laya_apple/parity/goldens`).\n")
    print("| Model | Implementation | Verdict | Rows | Hard mismatches | Near-tie flips | Max prob error |")
    print("|---|---|---|---:|---:|---:|---:|")
    names = (
        ("PyTorch MPS FP32", "torch_device-mps"),
        ("Core ML ordinary · CPU_AND_NE", "coreml_units-cpu_ne"),
        ("Core ML ordinary · ALL", "coreml_units-all"),
        ("Core ML ordinary · CPU_AND_GPU", "coreml_units-cpu_gpu"),
        ("laya-apple MLX FP32", "laya-apple-gpu-float32"),
        ("laya-apple MLX FP16", "laya-apple-gpu-float16"),
        ("laya-apple ANE FP16", "laya-apple-ane-float16"),
    )
    for m in MODELS:
        for label, name in names:
            p = _parity(m, name)
            if p is None:
                continue
            s = p.get("summary", p)
            hard, near = _mismatches(p)
            rows = s["rows"] if isinstance(s.get("rows"), int) else p.get("rows")
            print(
                f"| {m} | {label} | {'✅ pass' if p.get('passed') else '❌ fail'} | {rows} | {hard} | {near} "
                f"| {_num(s['prob_max_abs'])} |"
            )
    print()


def section_hetero():
    print(
        "| Model | GPU-only req/s | GPU + ANE req/s | Multiplier | Short P99 ms (GPU-only → GPU + ANE) | Long req/s (GPU-only → GPU + ANE) | Answer mismatches |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        d = json.loads((V1 / f"placement-{m}-{PLACEMENT[m]}.json").read_text())
        s, g = d["part_a"]["summary"], d["part_a"]["gate"]
        gpu = sum(v["req_s"] for v in s["gpu_only"].values())
        print(
            f"| {m} | {gpu:.1f} | {g['aggregate_hetero_req_s']:.1f} | {g['aggregate_hetero_req_s'] / gpu:.2f}× "
            f"| {s['gpu_only']['short']['p99_ms']:.1f} → {s['hetero']['short']['p99_ms']:.1f} "
            f"| {s['gpu_only']['long']['req_s']:.1f} → {s['hetero']['long']['req_s']:.1f} | {g['mismatches']} |"
        )
    print()


def section_openloop():
    print("Latency from arrival to result (queueing included), ms. `auto` = GPU + ANE; `gpu_only` = the same")
    print('arrival sequence with `device="gpu"`.\n')
    for m in MODELS:
        p = V1 / f"concurrency-{m}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        print(f"#### {m}\n")
        print(
            "| Arrivals | Offered req/s | Class | n | GPU-only P50 / P95 / P99 | GPU + ANE P50 / P95 / P99 | Mismatches |"
        )
        print("|---|---:|---|---:|---:|---:|---:|")
        for key, runs in d["part_b"].items():
            pattern, _ = key.split("@")
            a, g = runs["auto"], runs["gpu_only"]
            for cls in a["classes"]:
                ca, cg = a["classes"][cls], g["classes"][cls]
                print(
                    f"| {pattern} | {a['offered_req_s']:.1f} | {cls} | {ca['n']} "
                    f"| {cg['p50_ms']:.1f} / {cg['p95_ms']:.1f} / {cg['p99_ms']:.1f} "
                    f"| {ca['p50_ms']:.1f} / {ca['p95_ms']:.1f} / {ca['p99_ms']:.1f} | {ca['mismatches'] + cg['mismatches']} |"
                )
        print()


SECTIONS = {
    "latency": section_latency,
    "e2e": section_e2e,
    "parity": section_parity,
    "hetero": section_hetero,
    "openloop": section_openloop,
}


def main(argv=None):
    which = (argv or sys.argv[1:] or ["all"])[0]
    for name, fn in SECTIONS.items():
        if which in ("all", name):
            print(f"### {name}\n")
            fn()


if __name__ == "__main__":
    main()
