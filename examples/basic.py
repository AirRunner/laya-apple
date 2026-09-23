"""Basic prediction: load a checkpoint with device="auto", ask one question, and print
the answer together with where it ran and why.

Run:

    uv run --extra ane python examples/basic.py

Without the `[ane]` extra the request runs on the GPU and `routing_reason` says why.
"""

from __future__ import annotations

import json

from laya_apple import Laya


def main():
    with Laya.from_pretrained("laya-typed-decisions", device="auto") as laya:
        result = laya.predict(
            context="The customer was charged twice for the same invoice this month.",
            questions={"refund": {"type": "noul", "instructions": "Does the customer request a refund?"}},
        )
    rt = result.runtime
    print(json.dumps(result.answers, indent=1, ensure_ascii=False))
    print(f"backend={rt.backend} device={rt.device} routing_reason={rt.routing_reason} latency_ms={rt.latency_ms:.1f}")


if __name__ == "__main__":
    main()
