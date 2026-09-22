"""Compare one backend configuration against the upstream PyTorch CPU FP32 goldens.

    uv run python scripts/parity.py --model laya-typed-decisions --spec '{"backend":"mlx","dtype":"float16"}'

Runs in its own process. Tolerances are fixed in methodology.md (set before any parity run):
  fp32 backends: calibrated-probability |diff| <= 1e-4,  action-probability |diff| <= 1e-4
  fp16 backends: calibrated-probability |diff| <= 0.02,  action-probability |diff| <= 0.02
  decisions:     0 argmax mismatches on rows whose reference top-1/top-2 calibrated margin
                 is >= 2x the probability tolerance; mismatches below that margin are
                 reported as near-tie flips and listed individually.
  tokens:        prompt ids / markers must equal upstream exactly.
Fixed-shape Core ML exports are served through `Bucketed` (pad to next length, never truncate).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from backends import Bucketed, make_backend  # noqa: E402
from common import MODELS, RAW, agent_config, environment, save_json, valid_lengths  # noqa: E402

TOL = {"float32": 1e-4, "float16": 0.02}


def calibrated(cfg, logits, k, qtype):
    from laya_mlx.common import clamp_temperature, temp_bucket

    temps = [clamp_temperature(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]
    by = {kk: clamp_temperature(v) for kk, v in cfg.get("temperature_by_options", {}).items()}
    t = by.get(temp_bucket(qtype, k), temps[qtype])
    z = np.asarray(logits[:k], np.float64) / t
    p = np.exp(z - z.max())
    return p / p.sum()


def softmax(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def precision_of(spec):
    if spec["backend"] == "torch":
        return "float32"
    if spec["backend"] == "mlx":
        return spec.get("dtype", "float16")
    return "float16"


def label_of(spec):
    parts = [spec["backend"]] + [f"{k}={v}" for k, v in spec.items() if k not in ("backend", "model")]
    return ",".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    spec = {**json.loads(args.spec), "model": args.model}
    ref = json.loads((RAW / "reference" / f"{args.model}.json").read_text())
    cfg = agent_config(args.model)
    precision = precision_of(spec)
    tol = TOL[precision]
    label = label_of(spec)
    out = {
        "model": args.model,
        "spec": spec,
        "label": label,
        "precision": precision,
        "tolerance": {"probability": tol, "action_probability": tol, "near_tie_margin": 2 * tol},
        "environment": environment(),
    }
    rows = []
    t0 = time.perf_counter()
    try:
        if spec["backend"] in ("coreml", "ane") and spec.get("bucketed", True) and not spec.get("enumerated"):
            b = Bucketed({k: v for k, v in spec.items() if k not in ("bucketed",)}, valid_lengths(args.model))
        else:
            b = make_backend({k: v for k, v in spec.items() if k != "enumerated"})
        out["describe"] = b.describe()
        token_mismatch = 0
        for case in ref["cases"]:
            items = b.prepare(case["state"], case["questions"])
            if items != case["items"]:
                token_mismatch += 1
                items = case["items"]  # still compare the model on the reference tokens
            logits, act = b.forward(items)
            buckets = getattr(b, "last_buckets", [None] * len(items))
            for r, it in enumerate(items):
                k, qt = len(it["markers"]), it["qtype"]
                rl = np.asarray(case["logits"][r], np.float64)
                gl = np.asarray(logits[r][:k], np.float64)
                rp, gp = calibrated(cfg, rl, k, qt), calibrated(cfg, gl, k, qt)
                srt = np.sort(rp)
                margin = float(srt[-1] - srt[-2]) if k > 1 else 1.0
                ra, ga = softmax(case["action_logits"][r]), softmax(act[r])
                d = np.abs(gl - rl)
                rows.append(
                    {
                        "case": case["name"],
                        "row": r,
                        "tokens": len(it["ids"]),
                        "bucket": buckets[r],
                        "k": k,
                        "qtype": qt,
                        "logit_max_abs": float(d.max()),
                        "logit_mean_abs": float(d.mean()),
                        "logit_rel": float(d.max() / max(np.abs(rl).max(), 1e-6)),
                        "prob_max_abs": float(np.abs(gp - rp).max()),
                        "action_prob_max_abs": float(np.abs(ga - ra).max()),
                        "action_logit_max_abs": float(np.abs(np.asarray(act[r], np.float64) - np.asarray(case["action_logits"][r])).max()),
                        "ref_argmax": int(rp.argmax()),
                        "argmax": int(gp.argmax()),
                        "ref_margin": margin,
                        "finite": bool(np.isfinite(gl).all() and np.isfinite(act[r]).all()),
                    }
                )
        # determinism: rerun the first and a long case, outputs must be bitwise identical
        rep = []
        longest = max(ref["cases"], key=lambda c: max(len(i["ids"]) for i in c["items"]))
        for case in (ref["cases"][0], longest):
            a1 = b.forward(case["items"])[0]
            a2 = b.forward(case["items"])[0]
            rep.append(bool(np.array_equal(a1, a2)))
        out["repeat_identical"] = all(rep)
        out["token_mismatch_cases"] = token_mismatch
        out["status"] = "ran"
        if isinstance(b, Bucketed):
            out["bucket_load_seconds"] = b.load_seconds
    except Exception as e:
        out["status"] = "failed"
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()[-4000:]
    out["seconds"] = time.perf_counter() - t0
    out["rows"] = rows
    if rows:
        mism = [r for r in rows if r["argmax"] != r["ref_argmax"]]
        hard = [r for r in mism if r["ref_margin"] >= 2 * tol]
        out["summary"] = {
            "rows": len(rows),
            "all_finite": all(r["finite"] for r in rows),
            "decision_mismatches": len(mism),
            "decision_mismatches_outside_near_tie": len(hard),
            "near_tie_flips": [
                {k: r[k] for k in ("case", "row", "tokens", "ref_margin", "prob_max_abs")} for r in mism if r not in hard
            ],
            "logit_max_abs": max(r["logit_max_abs"] for r in rows),
            "logit_mean_abs": float(np.mean([r["logit_mean_abs"] for r in rows])),
            "logit_rel_max": max(r["logit_rel"] for r in rows),
            "prob_max_abs": max(r["prob_max_abs"] for r in rows),
            "prob_p99_abs": float(np.percentile([r["prob_max_abs"] for r in rows], 99)),
            "action_prob_max_abs": max(r["action_prob_max_abs"] for r in rows),
            "action_logit_max_abs": max(r["action_logit_max_abs"] for r in rows),
            "by_length": {},
        }
        for L in sorted({r["bucket"] or r["tokens"] for r in rows}):
            sel = [r for r in rows if (r["bucket"] or r["tokens"]) == L]
            out["summary"]["by_length"][str(L)] = {
                "rows": len(sel),
                "prob_max_abs": max(r["prob_max_abs"] for r in sel),
                "mismatches": sum(r["argmax"] != r["ref_argmax"] for r in sel),
                "hard_mismatches": sum(r["argmax"] != r["ref_argmax"] and r["ref_margin"] >= 2 * tol for r in sel),
            }
        s = out["summary"]
        out["passed"] = bool(
            out.get("status") == "ran"
            and s["all_finite"]
            and s["decision_mismatches_outside_near_tie"] == 0
            and s["prob_max_abs"] <= tol
            and s["action_prob_max_abs"] <= tol
            and out.get("token_mismatch_cases", 1) == 0
            and out.get("repeat_identical", False)
        )
    else:
        out["passed"] = False
    safe = label.replace("=", "-").replace(",", "_")
    save_json(RAW / "parity" / args.model / f"{safe}.json", out)
    s = out.get("summary", {})
    print(
        f"{args.model} {label}: passed={out['passed']} status={out['status']} "
        f"prob_max={s.get('prob_max_abs')} mism={s.get('decision_mismatches')} "
        f"hard={s.get('decision_mismatches_outside_near_tie')} logit_max={s.get('logit_max_abs')} "
        f"act={s.get('action_prob_max_abs')} rep={out.get('repeat_identical')} err={out.get('error')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
