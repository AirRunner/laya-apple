"""Heterogeneous serving: short single-question requests go to the ANE, long or
multi-question requests go to the MLX GPU, and both run concurrently.

This is what differentiates laya-apple from a plain MLX or Core ML runtime: one
`Laya` instance serves both devices at once through a worker per device, decides
per request which one to use, and records that decision on the result.

Run:

    uv run --extra ane python examples/heterogeneous_serving.py

Without the `[ane]` extra (no coremltools installed), the ANE path is simply
unavailable: every request still gets an answer, but `routing_reason` for what
would have gone to the ANE reads `ane_runtime_unavailable` and everything runs on
the GPU instead. This script does not hide that — it prints whatever
`routing_reason` the runtime actually recorded.

Expected shape of the output (numbers vary by machine): a table with one row per
submitted request showing its kind, token count, chosen backend/device, routing
reason and timings, followed by per-device totals. On a machine with a validated
ANE artifact for the short buckets, the short single-question requests route to
`ane` and the long/multi-question ones route to `gpu`, overlapping in time.
"""

from __future__ import annotations

import time

from laya_apple import Laya
from laya_apple.workload import make_request

MODEL = "convaiinnovations/laya-typed-decisions"

SHORT_REQUESTS = [
    (
        "The customer was charged twice for the same invoice this month.",
        {"refund": {"type": "noul", "instructions": "Does the customer request a refund?"}},
    ),
    (
        "I cannot log into my account since the last update.",
        {
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this ticket?",
                "criteria": ["low", "medium", "high"],
            }
        },
    ),
    (
        "Please confirm my subscription renewal went through.",
        {
            "sentiment": {
                "type": "choice",
                "instructions": "What is the overall sentiment of the message?",
                "criteria": ["positive", "neutral", "negative"],
            }
        },
    ),
]


def _row(kind, tokens, r):
    rt = r.runtime
    return (
        kind,
        tokens,
        rt.backend,
        rt.device,
        rt.routing_reason,
        f"{rt.queue_wait_ms:.1f}" if rt.queue_wait_ms is not None else "-",
        f"{rt.device_ms:.1f}" if rt.device_ms is not None else "-",
        f"{rt.latency_ms:.1f}",
    )


def main():
    laya = Laya.from_pretrained(MODEL, execution="workers")
    try:
        laya.wait_for_ane(timeout=120)

        # Build every request first, then submit them together. Heavy Python work in the
        # calling thread while requests run competes for the GIL with the ANE dispatcher
        # when the ANE runs on a thread in this process (see docs/guide.md).
        requests = [("short/1q", context, questions) for context, questions in SHORT_REQUESTS]
        requests.append(("long/1q", *make_request(laya.tokenizer, laya.config, length=900, n_questions=1, seed=1)))
        requests.append(
            ("multi-question", *make_request(laya.tokenizer, laya.config, length=200, n_questions=4, seed=2))
        )
        futures = [(kind, laya.submit(context=c, questions=q)) for kind, c, q in requests]

        rows = []
        for kind, fut in futures:
            result = fut.result()
            tokens = result.runtime.sequence_length
            rows.append(_row(kind, tokens, result))

        header = ("kind", "tokens", "backend", "device", "routing_reason", "queue_wait_ms", "device_ms", "latency_ms")
        widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*header))
        print(fmt.format(*("-" * w for w in widths)))
        for row in rows:
            print(fmt.format(*row))

        totals: dict[str, float] = {}
        for row in rows:
            device = row[3]
            totals[device] = totals.get(device, 0.0) + float(row[7])
        print()
        print("totals per device (sum of latency_ms):")
        for device, total in sorted(totals.items()):
            print(f"  {device}: {total:.1f} ms")
    finally:
        laya.close()


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\n(wall time: {time.perf_counter() - t0:.2f} s)")
