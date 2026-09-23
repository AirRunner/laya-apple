"""Derive the v0.1 `device="auto"` thresholds from Phase -1 measurements.

    uv run python scripts/derive_routing.py            # writes laya_apple/data/routing.json
    uv run python scripts/derive_routing.py --check    # fails if the committed file is stale

Inputs (committed Phase -1 evidence):
  research/phase-0-feasibility/raw/bench/latency.jsonl   tags pass-a + refine, 1 question
  research/phase-0-feasibility/raw/parity/<model>/ane_units-cpu_ne.json

Rule, per model, over the ANE buckets in the registry (sorted b1 < b2 < ...):
  a bucket b_i joins auto routing iff
    (1) the BC1S ANE graph passed parity at b_i on CPU_AND_NE, and
    (2) ANE forward P50 at b_i  <  MLX FP16 forward P50 at b_{i-1}
        (a request just above b_{i-1} is padded to b_i on the ANE but costs about MLX(b_{i-1})
         on the GPU, so this is the conservative comparison; for b_1 the comparison is
         against MLX(b_1) because no shorter MLX length was measured),
  and routing stops at the first bucket that fails.
Multi-question requests are never auto-routed to the ANE: with 4 and 8 questions MLX was
faster at every measured length, and 2-3 questions were not measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laya_apple.derivation import derive_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "research/phase-0-feasibility"
OUT = ROOT / "laya_apple/data/routing.json"
# Buckets for which ANE artifacts are offered (DEVELOPMENT_PLAN.md §6.3).
CANDIDATE_BUCKETS = {
    "laya": [64, 96, 128],
    "laya-multilingual": [64, 96, 128, 256],
    "laya-typed-decisions": [64, 96, 128],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def p50(rows, model, pred, length):
    hit = [
        r
        for r in rows
        if r["model"] == model
        and pred(r["spec"])
        and r["length"] == length
        and r["questions"] == 1
        and r["tag"] in ("pass-a", "refine")
        and "forward" in r
    ]
    return hit[0]["forward"]["p50_ms"] if hit else None


def service_table(rows, model, pred) -> dict:
    """{questions: {length: forward P50 ms}} from pass-a/refine, for queue-aware routing (v0.2)."""
    out: dict = {}
    for r in rows:
        if r["model"] == model and pred(r["spec"]) and r["tag"] in ("pass-a", "refine") and "forward" in r:
            out.setdefault(str(r["questions"]), {})[str(r["length"])] = round(r["forward"]["p50_ms"], 4)
    return {
        q: dict(sorted(v.items(), key=lambda kv: int(kv[0]))) for q, v in sorted(out.items(), key=lambda kv: int(kv[0]))
    }


def derive() -> dict:
    latency = EVIDENCE / "raw/bench/latency.jsonl"
    rows = [json.loads(line) for line in latency.read_text().splitlines() if line.strip()]
    is_mlx = lambda s: s["backend"] == "mlx" and s.get("dtype") == "float16"  # noqa: E731
    is_ane = lambda s: (  # noqa: E731
        s["backend"] == "ane"
        and s.get("units") == "cpu_ne"
        and s.get("batch", 1) == 1
        and s.get("variant", "masked") == "masked"
    )
    env_file = EVIDENCE / "raw/environment.json"
    env = json.loads(env_file.read_text())
    out = {
        "generated_by": "scripts/derive_routing.py",
        "rule": __doc__.split("Rule, per model")[1].strip(),
        "evidence": {"latency.jsonl": sha256(latency), "environment.json": sha256(env_file)},
        "hardware": "Apple M4 Max, macOS 26.6.2 (Phase -1); thresholds are not validated elsewhere",
        # auto uses the ANE only on a matching profile: same SoC, macOS major, coremltools
        "validated_profiles": [
            {"soc": env["soc"], "macos": env["macos"], "coremltools": env["packages"]["coremltools"]}
        ],
        "models": {},
    }
    for model, buckets in CANDIDATE_BUCKETS.items():
        parity_file = EVIDENCE / f"raw/parity/{model}/ane_units-cpu_ne.json"
        parity = json.loads(parity_file.read_text())
        out["evidence"][f"parity/{model}"] = sha256(parity_file)
        by_len = parity["summary"]["by_length"]
        tol = parity["tolerance"]["probability"]
        cells = {b: by_len.get(str(b), {}) for b in buckets}
        par = {
            b: (bool(c) and c["hard_mismatches"] == 0 and c["prob_max_abs"] <= tol, c.get("prob_max_abs"))
            for b, c in cells.items()
        }
        ane = {b: p50(rows, model, is_ane, b) for b in buckets}
        mlx = {b: p50(rows, model, is_mlx, b) for b in buckets}
        derived = derive_model(buckets, par, ane, mlx)
        out["models"][model] = {
            **derived,
            # Phase -1 forward P50s; v0.2 routing uses them as service-time estimates only.
            "service_ms": {"gpu": service_table(rows, model, is_mlx), "ane": service_table(rows, model, is_ane)},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    derived = derive()
    text = json.dumps(derived, indent=1) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f"{OUT} is stale; rerun scripts/derive_routing.py")
        print("routing.json is up to date")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    for m, d in derived["models"].items():
        print(m, "auto:", d["auto_ane_buckets"], "explicit:", d["ane_buckets"])


if __name__ == "__main__":
    main()
