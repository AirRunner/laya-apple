"""Unmodified upstream Laya reference for the option-order check (PyTorch, CPU, FP32).

Runs the exact same case set and criteria permutations as `scripts/option_order.py run`,
through unmodified upstream Laya (`NandhaKishorM/laya`), so laya-apple's own results can be
compared against the semantic reference with `scripts/option_order.py compare`. CPU FP32 is
the reference precision for the same reason it is in the Phase -1 parity methodology
(research/phase-0-feasibility/methodology.md §2): deterministic, and upstream code unchanged.

Run from the research env (it needs the pinned upstream `laya` package and torch, which only
that venv has):

    cd research/phase-0-feasibility
    uv run python ../option-order/upstream.py --model laya-typed-decisions \\
        --output ../../benchmarks/v1.0/option-order-upstream-laya-typed-decisions.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RESEARCH_SCRIPTS = Path(__file__).resolve().parents[1] / "phase-0-feasibility" / "scripts"
sys.path.insert(0, str(RESEARCH_SCRIPTS))

from common import checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = Path(__file__).resolve().parent / "cases.json"


def load_cases() -> list:
    return json.loads(CASES_PATH.read_text())["cases"]


def criteria_for(case: dict, order: list) -> dict:
    labels, values = case["labels"], case["values"]
    return {labels[i]: values[labels[i]] for i in order}


def _answer(result, qid: str) -> dict:
    """Upstream returns the same {"answers": {...}} shaped result as laya-apple; accept a
    dict or an object exposing `.answers` so either upstream revision works."""
    answers = result["answers"] if isinstance(result, dict) else result.answers
    return answers[qid]


def max_abs_delta_prob(permutations: list) -> float:
    ok = [p for p in permutations if "probabilities" in p]
    if len(ok) < 2:
        return 0.0
    labels = ok[0]["probabilities"].keys()
    return max(
        max(p["probabilities"][label] for p in ok) - min(p["probabilities"][label] for p in ok) for label in labels
    )


def run(model: str) -> list:
    import laya

    agent = laya.load(str(checkpoint(model)), device="cpu")
    out = []
    for case in load_cases():
        perms = []
        for order in case["permutations"]:
            crit = criteria_for(case, order)
            questions = {"q": {"type": "choice", "instructions": case["instructions"], "criteria": crit}}
            result = agent.predict(case["context"], questions)
            ans = _answer(result, "q")
            perms.append({"order": order, "chosen": ans["choice"], "probabilities": ans["probabilities"]})
        chosen = {p["chosen"] for p in perms}
        out.append(
            {
                "name": case["name"],
                "k": len(case["labels"]),
                "permutations": perms,
                "decision_changed": len(chosen) > 1,
                "max_abs_delta_prob": max_abs_delta_prob(perms),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    t0 = time.perf_counter()
    cases = run(args.model)
    changed = sum(1 for c in cases if c["decision_changed"])
    out = {
        "tool": "upstream",
        "reference": "unmodified upstream Laya, torch CPU float32",
        "model": args.model,
        "cases": cases,
        "seconds": time.perf_counter() - t0,
    }
    print(args.model, "decision_changed:", changed, "/", len(cases))
    output = Path(args.output or ROOT / f"benchmarks/v1.0/option-order-upstream-{args.model}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("wrote", output)


if __name__ == "__main__":
    main()
