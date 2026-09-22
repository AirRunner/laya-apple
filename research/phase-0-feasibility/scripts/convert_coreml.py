"""Ordinary Core ML export via the pinned laya-coreml converter (no graph changes).

    uv run python scripts/convert_coreml.py --model laya-typed-decisions
    uv run python scripts/convert_coreml.py --model laya-typed-decisions --fixed 128 --attention explicit

Default: enumerated sequence shapes 16..max_len, B=1, K=32, FP16, SDPA — exactly the
configuration laya-coreml publishes. `--fixed L` exports one static length (control for
the "does dynamic shape block ANE placement?" question).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ARTIFACTS, MODELS, RAW, append_jsonl, checkpoint, environment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--fixed", type=int, default=None)
    ap.add_argument("--attention", default="sdpa", choices=["sdpa", "explicit"])
    args = ap.parse_args()
    from laya_coreml.convert import convert

    tag = f"fixed{args.fixed}-{args.attention}" if args.fixed else args.attention
    out = ARTIFACTS / "coreml-ordinary" / (args.model if tag == "sdpa" else f"{args.model}-{tag}")
    record = {
        "experiment": "convert_coreml_ordinary",
        "model": args.model,
        "output": str(out),
        "fixed_length": args.fixed,
        "attention": args.attention,
        "environment": environment(),
    }
    t0 = time.perf_counter()
    try:
        if out.exists():
            record["status"] = "exists"
        else:
            convert(
                str(checkpoint(args.model)),
                out,
                max_length=args.fixed,
                flexible=args.fixed is None,
                batch_size=1,
                max_options=32,
                precision="float16",
                revision=MODELS[args.model]["revision"],
                attention=args.attention,
            )
            record["status"] = "ok"
            record["manifest"] = json.loads((out / "coreml_config.json").read_text())
            record["manifest"].pop("files", None)
    except Exception as e:  # keep failures as evidence
        record["status"] = "failed"
        record["error"] = repr(e)
        record["traceback"] = traceback.format_exc()[-4000:]
    record["seconds"] = time.perf_counter() - t0
    append_jsonl(RAW / "conversion.jsonl", record)
    print(json.dumps({k: v for k, v in record.items() if k != "environment"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
