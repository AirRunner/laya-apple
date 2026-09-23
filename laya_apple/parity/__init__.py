"""Parity against the upstream PyTorch CPU FP32 goldens (Phase -1 methodology).

The gate is the one fixed before Phase -1 measured anything
(research/phase-0-feasibility/methodology.md §5) and must not be loosened:

  prompt tokens/markers          exact
  calibrated probability  max|Δ| <= 1e-4 (FP32) / 0.02 (FP16)
  action probability      max|Δ| <= 1e-4 (FP32) / 0.02 (FP16)
  decisions               0 mismatches where the reference top-1/top-2 margin >= 2x tol
                          (flips inside that near-tie band are listed, not failed)
  outputs                 finite; repeated identical calls bitwise identical
"""

from __future__ import annotations

import json
from functools import cache
from importlib import resources

import numpy as np

from ..prompt import Calibration

TOLERANCE = {"float32": 1e-4, "float16": 0.02}


@cache
def load_goldens(model: str) -> dict:
    return json.loads(resources.files("laya_apple.parity.goldens").joinpath(f"{model}.json").read_text())


def _softmax(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def evaluate(model: str, cfg: dict, forward, *, precision: str, max_len: int | None = None, prepare=None) -> dict:
    """Run `forward(items) -> (logits, action_logits)` over the goldens and apply the gate.

    Rows longer than `max_len` are skipped and reported (fixed-shape artifacts cover only
    their bucket). `prepare(state, questions) -> items` checks prompt-token parity if given.
    """
    tol = TOLERANCE[precision]
    calib = Calibration(cfg)
    g = load_goldens(model)
    rows, skipped, token_mismatch = [], 0, 0
    for case in g["cases"]:
        items = case["items"]
        if prepare is not None and prepare(case["state"], case["questions"]) != items:
            token_mismatch += 1
        keep = [i for i, it in enumerate(items) if max_len is None or len(it["ids"]) <= max_len]
        skipped += len(items) - len(keep)
        if not keep:
            continue
        sub = [items[i] for i in keep]
        logits, act = forward(sub)
        for j, i in enumerate(keep):
            it = sub[j]
            k, qt = len(it["markers"]), it["qtype"]
            rl = np.asarray(case["logits"][i], np.float64)
            gl = np.asarray(logits[j][:k], np.float64)
            rp, gp = calib.probabilities(rl, qt, k), calib.probabilities(gl, qt, k)
            srt = np.sort(rp)
            rows.append(
                {
                    "case": case["name"],
                    "row": i,
                    "tokens": len(it["ids"]),
                    "prob_max_abs": float(np.abs(gp - rp).max()) if np.isfinite(gl).all() else float("inf"),
                    "action_prob_max_abs": float(np.abs(_softmax(act[j]) - _softmax(case["action_logits"][i])).max()),
                    "logit_max_abs": float(np.abs(gl - rl).max()),
                    "ref_margin": float(srt[-1] - srt[-2]) if k > 1 else 1.0,
                    "mismatch": int(gp.argmax()) != int(rp.argmax()),
                    "finite": bool(np.isfinite(gl).all() and np.isfinite(act[j]).all()),
                }
            )
    # determinism: the first evaluated case twice, bitwise
    first = next(c for c in g["cases"] if any(max_len is None or len(it["ids"]) <= max_len for it in c["items"]))
    sub = [it for it in first["items"] if max_len is None or len(it["ids"]) <= max_len]
    a1, x1 = forward(sub)
    a2, x2 = forward(sub)
    repeat = bool(np.array_equal(a1, a2) and np.array_equal(x1, x2))
    hard = [r for r in rows if r["mismatch"] and r["ref_margin"] >= 2 * tol]
    flips = [r for r in rows if r["mismatch"] and r["ref_margin"] < 2 * tol]
    summary = {
        "precision": precision,
        "tolerance": tol,
        "rows": len(rows),
        "skipped_rows": skipped,
        "token_mismatch_cases": token_mismatch,
        "prob_max_abs": max((r["prob_max_abs"] for r in rows), default=0.0),
        "action_prob_max_abs": max((r["action_prob_max_abs"] for r in rows), default=0.0),
        "logit_max_abs": max((r["logit_max_abs"] for r in rows), default=0.0),
        "hard_mismatches": len(hard),
        "near_tie_flips": [{k: r[k] for k in ("case", "row", "tokens", "ref_margin")} for r in flips],
        "all_finite": all(r["finite"] for r in rows),
        "repeat_identical": repeat,
    }
    summary["passed"] = bool(
        rows
        and summary["all_finite"]
        and summary["token_mismatch_cases"] == 0
        and summary["hard_mismatches"] == 0
        and summary["prob_max_abs"] <= tol
        and summary["action_prob_max_abs"] <= tol
        and repeat
    )
    return summary
