"""The auto-routing rule, shared by scripts/derive_routing.py (shipped, Phase -1 evidence)
and `laya-apple calibrate` (local capability profiles, v0.3). DEVELOPMENT_PLAN.md §7.2.

Over the offered ANE buckets b1 < b2 < ...: bucket b_i joins auto routing iff
  (1) the BC1S ANE graph passed parity at b_i on CPU_AND_NE, and
  (2) ANE forward P50 at b_i < MLX FP16 forward P50 at b_{i-1} (for b_1: at b_1),
and routing stops at the first bucket that fails. Multi-question requests never auto-route
to the ANE.
"""

from __future__ import annotations


def derive_model(buckets, parity: dict, ane_p50: dict, mlx_p50: dict) -> dict:
    """buckets: offered buckets; parity: {b: (passed, prob_max_abs)}; ane_p50: {b: ms};
    mlx_p50: {length: ms} (must cover every bucket that is compared)."""
    buckets = sorted(buckets)
    decisions, auto, stopped = [], [], False
    for i, b in enumerate(buckets):
        parity_ok, prob = parity.get(b, (False, None))
        ane = ane_p50.get(b)
        ref_len = buckets[i - 1] if i else b
        mlx = mlx_p50.get(ref_len)
        faster = ane is not None and mlx is not None and ane < mlx
        ok = bool(parity_ok) and faster and not stopped
        decisions.append(
            {
                "bucket": b,
                "parity_pass": bool(parity_ok),
                "parity_prob_max_abs": prob,
                "ane_p50_ms": ane,
                "mlx_p50_ms_at": ref_len,
                "mlx_p50_ms": mlx,
                "auto": ok,
            }
        )
        if ok:
            auto.append(b)
        else:
            stopped = True
    return {
        "ane_buckets": [d["bucket"] for d in decisions if d["parity_pass"]],
        "auto_ane_buckets": auto,
        "auto_ane_max_len": max(auto) if auto else 0,
        "auto_ane_max_questions": 1,
        "decisions": decisions,
    }
