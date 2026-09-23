"""Auto routing: one `Laya(device="auto")` instance decides per request whether the
short single-question path on the ANE or the MLX GPU serves it, and records why.

Sends a short single-question request (~60 tokens), a long one (~700 tokens) and a
multi-question request, one after another, and prints a table of where each ran.

Run:

    uv run --extra ane python examples/auto_routing.py

On a machine whose profile is not validated, every row routes to the GPU with
`routing_reason == "platform_not_validated"`; that is the expected, recorded result.
"""

from __future__ import annotations

from laya_apple import Laya
from laya_apple.workload import make_request


def main():
    with Laya.from_pretrained("laya-typed-decisions", device="auto") as laya:
        # Build every input before timing anything.
        requests = [
            ("short/1q", *make_request(laya.tokenizer, laya.config, length=60, n_questions=1, seed=1)),
            ("long/1q", *make_request(laya.tokenizer, laya.config, length=700, n_questions=1, seed=2)),
            ("multi-question", *make_request(laya.tokenizer, laya.config, length=120, n_questions=4, seed=3)),
        ]
        rows = []
        for kind, context, questions in requests:
            rt = laya.predict(context=context, questions=questions).runtime
            rows.append((kind, rt.sequence_length, rt.backend, rt.device, rt.routing_reason, f"{rt.latency_ms:.1f}"))

    header = ("kind", "tokens", "backend", "device", "routing_reason", "latency_ms")
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


if __name__ == "__main__":
    main()
