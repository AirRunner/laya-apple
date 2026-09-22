"""Golden references from unmodified upstream Laya (PyTorch, CPU, FP32).

    uv run python scripts/reference.py --model laya-typed-decisions

Writes raw/reference/<model>.json: per case the prompt items, unrounded decision logits,
action logits and the public result. CPU FP32 is the semantic reference because it is
deterministic and uses upstream code unchanged (upstream itself selects FP32 on cpu/mps).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from backends import TorchBackend  # noqa: E402
from common import MODELS, RAW, checkpoint, environment, save_json, sha256_file  # noqa: E402
from fixtures import all_cases  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()
    from laya_coreml.tokenizer import Tokenizer

    b = TorchBackend(args.model, "cpu")
    rust_tok = Tokenizer(checkpoint(args.model) / "tokenizer")
    out = {
        "model": args.model,
        "reference": "upstream NandhaKishorM/laya Agent, torch CPU float32",
        "source_weights_sha256": sha256_file(checkpoint(args.model) / "model.safetensors"),
        "environment": environment(),
        "cases": [],
    }
    from laya_coreml.prompt import PromptMixin

    class _P(PromptMixin):
        pass

    alt = _P()
    alt.tok, alt.cfg = rust_tok, b.agent.cfg
    t0 = time.perf_counter()
    for case in all_cases(args.model):
        items = b.prepare(case["state"], case["questions"])
        # The ports tokenise with the Rust `tokenizers` backend; upstream uses transformers.
        rust_items = alt.prepare(case["state"], case["questions"])[0]
        logits, act = b.forward(items)
        result = b.predict(case["state"], case["questions"])
        out["cases"].append(
            {
                **case,
                "items": items,
                "tokenizer_match": rust_items == items,
                "logits": [row[: len(it["markers"])].tolist() for row, it in zip(logits, items)],
                "action_logits": act.tolist(),
                "result": result,
            }
        )
        print(case["name"], max(len(i["ids"]) for i in items), "tok-match", rust_items == items, flush=True)
    out["seconds"] = time.perf_counter() - t0
    save_json(RAW / "reference" / f"{args.model}.json", out)


if __name__ == "__main__":
    main()
