"""Quick functional smoke test: every backend runs one exact-length request.

Not a benchmark (few samples, backends share one process). Used to catch wiring errors.
    uv run python scripts/smoke.py --model laya-typed-decisions --length 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from backends import make_backend  # noqa: E402
from common import agent_config, checkpoint, make_request  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya-typed-decisions")
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    from laya_coreml.tokenizer import Tokenizer

    tok = Tokenizer(checkpoint(args.model) / "tokenizer")
    state, questions, items = make_request(tok, agent_config(args.model), args.length, 1)
    specs = [
        {"backend": "torch", "device": "cpu"},
        {"backend": "torch", "device": "mps"},
        {"backend": "mlx", "dtype": "float16"},
        {"backend": "coreml", "units": "cpu_gpu"},
        {"backend": "coreml", "units": "cpu_ne"},
        {"backend": "coreml", "units": "all"},
        {"backend": "ane", "units": "cpu_ne", "length": args.length},
        {"backend": "ane", "units": "cpu_only", "length": args.length},
        {"backend": "ane", "units": "cpu_gpu", "length": args.length},
        {"backend": "ane", "units": "all", "length": args.length},
    ]
    ref = None
    for spec in specs:
        label = "-".join(str(v) for k, v in spec.items() if k != "length")
        if args.only and label not in args.only:
            continue
        spec["model"] = args.model
        t = time.perf_counter()
        b = make_backend(spec)
        load = time.perf_counter() - t
        got = b.prepare(state, questions)
        assert got == items, f"{label}: prompt tokens differ from the harness"
        lg, ac = b.forward(items)
        ts = []
        for _ in range(10):
            t = time.perf_counter()
            b.forward(items)
            ts.append((time.perf_counter() - t) * 1e3)
        k = len(items[0]["markers"])
        z = lg[0, :k]
        if ref is None:
            ref = z
        print(
            f"{label:28s} load={load:6.1f}s  p50={np.median(ts):8.2f}ms  "
            f"logits={np.round(z, 3)}  max|d|={np.abs(z - ref).max():.4f}",
            flush=True,
        )
        del b


if __name__ == "__main__":
    main()
