"""Option-order robustness check for `choice` questions.

A `choice` question's `criteria` is presented to the model as an ordered list of masked
option markers (see `laya_apple.prompt.render_options` / `build_sequence`). Nothing in the
task semantics depends on that order: the chosen label and its probability should not change
when the criteria dict is permuted. This script measures whether that actually holds for
laya-apple, across devices and dtypes, and `compare` holds those results against an unmodified
upstream Laya run (`research/option-order/upstream.py`) so any instability can be attributed
to the runtime or to the upstream model itself.

Regenerate the deterministic case set (stdlib only, no model load):
    uv run python scripts/option_order.py write-cases

Run laya-apple over the cases on one device:
    uv run python scripts/option_order.py run --device gpu --dtype float32 \\
        --model laya-typed-decisions --model laya \\
        --output benchmarks/v1.0/option-order-gpu-float32.json

Compare two result files (laya-apple vs. upstream, or two laya-apple runs):
    uv run python scripts/option_order.py compare \\
        benchmarks/v1.0/option-order-gpu-float32.json \\
        benchmarks/v1.0/option-order-upstream-laya-typed-decisions.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "research/option-order/cases.json"
DEFAULT_MODELS = ("laya-typed-decisions", "laya")
CASE_SEED = 20260923
FIVE_OPTION_SAMPLE = 6

# ----------------------------------------------------------------------------- case templates

# Short, generic, originally written contexts (not drawn from any dataset or model output).
CONTEXTS = {
    "c0": (
        "A customer named Priya wrote in after her package arrived three days late and "
        "slightly damaged. She asked what could be done and mentioned she has been a "
        "member for two years."
    ),
    "c1": (
        "During a routine review, an engineer noticed that a background job had been "
        "silently failing for a week, though no user-facing errors were reported yet."
    ),
    "c2": (
        "A short note arrived from a new subscriber asking how to change the delivery "
        "address on their account before the next shipment goes out."
    ),
    "c3": (
        "The weekly report showed unusually high server load overnight, but response "
        "times stayed within the normal range throughout."
    ),
    "c4": (
        "A student emailed the help desk saying the practice exam timer stopped working "
        "halfway through, and she wants to know if her progress was saved."
    ),
}

# Each template's `values` are original, generic label descriptions; `labels` fixes the
# canonical (unpermuted) order used to report results.
TEMPLATES = {
    "route": {
        "instructions": "Which team should this be routed to?",
        "labels": ["support", "engineering", "sales"],
        "values": {
            "support": "customer service and account issues",
            "engineering": "technical bugs and system behavior",
            "sales": "purchasing, pricing and renewals",
        },
    },
    "sentiment": {
        "instructions": "What is the overall tone of this message?",
        "labels": ["positive", "neutral", "negative"],
        "values": {
            "positive": "satisfied or appreciative",
            "neutral": "matter-of-fact, no strong feeling",
            "negative": "frustrated or unhappy",
        },
    },
    "confidence": {
        "instructions": "How confident does the writer sound?",
        "labels": ["unsure", "moderate", "confident"],
        "values": {
            "unsure": "hesitant or asking for guidance",
            "moderate": "reasonably sure but open to input",
            "confident": "clearly certain about what they want",
        },
    },
    "priority": {
        "instructions": "How urgent is this item?",
        "labels": ["low", "medium", "high", "critical"],
        "values": {
            "low": "can be handled whenever convenient",
            "medium": "should be looked at this week",
            "high": "should be looked at today",
            "critical": "needs immediate attention",
        },
    },
    "channel": {
        "instructions": "Which channel would best fit a reply?",
        "labels": ["email", "chat", "phone", "in_person"],
        "values": {
            "email": "a written, asynchronous reply",
            "chat": "a quick back-and-forth message",
            "phone": "a spoken conversation",
            "in_person": "a face-to-face meeting",
        },
    },
    "timeframe": {
        "instructions": "Over what timeframe does this apply?",
        "labels": ["immediate", "this_week", "this_month", "no_deadline"],
        "values": {
            "immediate": "needs to happen right away",
            "this_week": "should happen within a few days",
            "this_month": "has some weeks of room",
            "no_deadline": "no particular deadline mentioned",
        },
    },
    "action": {
        "instructions": "What is the most fitting next action?",
        "labels": ["acknowledge", "investigate", "escalate", "resolve", "close"],
        "values": {
            "acknowledge": "send a brief confirmation it was received",
            "investigate": "look deeper before deciding anything",
            "escalate": "hand it to someone with more authority",
            "resolve": "take the concrete fix now",
            "close": "no further action is needed",
        },
    },
    "category": {
        "instructions": "Which category best describes this item?",
        "labels": ["billing", "technical", "account", "product", "other"],
        "values": {
            "billing": "charges, invoices or payments",
            "technical": "errors or broken functionality",
            "account": "login, profile or permissions",
            "product": "a question about features or usage",
            "other": "does not fit the other categories",
        },
    },
}


def _permutations_for(k: int, seed: int) -> list:
    """All permutations for k<=4; a fixed-seed sample of 6 (identity first) for k==5."""
    identity = tuple(range(k))
    all_perms = list(itertools.permutations(range(k)))
    if k <= 4:
        return [list(p) for p in all_perms]
    rest = [p for p in all_perms if p != identity]
    sampled = random.Random(seed).sample(rest, FIVE_OPTION_SAMPLE - 1)
    return [list(identity)] + [list(p) for p in sampled]


def generate_cases(seed: int = CASE_SEED) -> dict:
    cases = []
    for tname in sorted(TEMPLATES):
        tmpl = TEMPLATES[tname]
        k = len(tmpl["labels"])
        for cname in sorted(CONTEXTS):
            case_seed = seed ^ hash((tname, cname)) & 0xFFFFFFFF
            cases.append(
                {
                    "name": f"{tname}-{cname}",
                    "context": CONTEXTS[cname],
                    "instructions": tmpl["instructions"],
                    "labels": list(tmpl["labels"]),
                    "values": dict(tmpl["values"]),
                    "permutations": _permutations_for(k, case_seed),
                }
            )
    return {"generated_by": "scripts/option_order.py generate_cases", "seed": seed, "cases": cases}


def write_cases(path: Path = CASES_PATH, seed: int = CASE_SEED) -> dict:
    data = generate_cases(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    return data


def load_cases(path: Path = CASES_PATH) -> list:
    return json.loads(path.read_text())["cases"]


def criteria_for(case: dict, order: list) -> dict:
    """The `criteria` dict for one permutation: same labels/values, reordered."""
    labels, values = case["labels"], case["values"]
    return {labels[i]: values[labels[i]] for i in order}


def max_abs_delta_prob(permutations: list) -> float:
    """Max, over labels, of (max - min) probability across the permutations that succeeded."""
    ok = [p for p in permutations if "probabilities" in p]
    if len(ok) < 2:
        return 0.0
    labels = ok[0]["probabilities"].keys()
    return max(
        max(p["probabilities"][label] for p in ok) - min(p["probabilities"][label] for p in ok) for label in labels
    )


# ----------------------------------------------------------------------------- laya-apple run


def run_model(laya, cases: list) -> list:
    from laya_apple.errors import LayaAppleError

    out = []
    for case in cases:
        perms = []
        for order in case["permutations"]:
            crit = criteria_for(case, order)
            questions = {"q": {"type": "choice", "instructions": case["instructions"], "criteria": crit}}
            try:
                result = laya.predict(context=case["context"], questions=questions)
                ans = result.answers["q"]
                perms.append({"order": order, "chosen": ans["choice"], "probabilities": ans["probabilities"]})
            except LayaAppleError as e:
                perms.append({"order": order, "error": f"{type(e).__name__}: {e}"})
        chosen = {p["chosen"] for p in perms if "chosen" in p}
        out.append(
            {
                "name": case["name"],
                "k": len(case["labels"]),
                "permutations": perms,
                "decision_changed": len(chosen) > 1,
                "max_abs_delta_prob": max_abs_delta_prob(perms),
                "errors": sum(1 for p in perms if "error" in p),
            }
        )
    return out


def cmd_write_cases(args):
    data = write_cases(Path(args.path), args.seed)
    print("wrote", args.path, "(", len(data["cases"]), "cases )")


def cmd_run(args):
    from laya_apple import Laya

    cases = load_cases(Path(args.cases))
    dtype = "float16" if args.device == "ane" else args.dtype
    out = {"tool": "laya-apple", "device": args.device, "dtype": dtype, "models": {}}
    for model in args.model:
        laya = Laya.from_pretrained(model, device=args.device, dtype=dtype)
        try:
            model_cases = run_model(laya, cases)
        finally:
            laya.close()
        changed = sum(1 for c in model_cases if c["decision_changed"])
        out["models"][model] = {
            "cases": model_cases,
            "summary": {"cases": len(model_cases), "decision_changed": changed},
        }
        print(model, "decision_changed:", changed, "/", len(model_cases))
    output = Path(args.output or ROOT / f"benchmarks/v1.0/option-order-{args.device}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("wrote", output)


# ----------------------------------------------------------------------------- compare


def _extract(data: dict) -> dict:
    """{model_name: [case, ...]} for either a laya-apple run or an upstream.py run."""
    if "models" in data:
        return {m: d["cases"] for m, d in data["models"].items()}
    return {data.get("model", "unknown"): data["cases"]}


NEAR_TIE_TOLERANCE = {"float32": 1e-4, "float16": 0.02}


def compare(a_path: Path, b_path: Path, tol: float | None = None) -> dict:
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    a_models, b_models = _extract(a), _extract(b)
    default_tol = tol if tol is not None else NEAR_TIE_TOLERANCE.get(a.get("dtype"), 1e-4)
    margin_gate = 2 * default_tol
    report = {"a": str(a_path), "b": str(b_path), "tolerance": default_tol, "models": {}}
    for model in sorted(set(a_models) & set(b_models)):
        a_cases = {c["name"]: c for c in a_models[model]}
        b_cases = {c["name"]: c for c in b_models[model]}
        common = sorted(set(a_cases) & set(b_cases))
        a_changed = sum(1 for n in common if a_cases[n]["decision_changed"])
        b_changed = sum(1 for n in common if b_cases[n]["decision_changed"])
        coincide = sum(1 for n in common if a_cases[n]["decision_changed"] and b_cases[n]["decision_changed"])
        mismatches, defects = [], []
        for n in common:
            a_perms = {tuple(p["order"]): p for p in a_cases[n]["permutations"] if "chosen" in p}
            b_perms = {tuple(p["order"]): p for p in b_cases[n]["permutations"] if "chosen" in p}
            for order in sorted(set(a_perms) & set(b_perms)):
                ap, bp = a_perms[order], b_perms[order]
                if ap["chosen"] != bp["chosen"]:
                    probs = sorted(bp["probabilities"].values(), reverse=True)
                    margin = probs[0] - probs[1] if len(probs) > 1 else 1.0
                    entry = {"case": n, "order": list(order), "a_chosen": ap["chosen"], "b_chosen": bp["chosen"]}
                    mismatches.append(entry)
                    if margin >= margin_gate:
                        defects.append(entry)
        report["models"][model] = {
            "cases": len(common),
            "decision_changed_fraction_a": round(a_changed / len(common), 4) if common else None,
            "decision_changed_fraction_b": round(b_changed / len(common), 4) if common else None,
            "cases_where_both_change": coincide,
            "per_permutation_mismatches": len(mismatches),
            "defects_outside_near_tie_band": defects,
        }
    return report


def cmd_compare(args):
    report = compare(Path(args.a), Path(args.b), tol=args.tol)
    print(json.dumps(report, indent=1, ensure_ascii=False))


# ----------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    wc = sub.add_parser("write-cases", help="(re)generate research/option-order/cases.json")
    wc.add_argument("--seed", type=int, default=CASE_SEED)
    wc.add_argument("--path", default=str(CASES_PATH))
    wc.set_defaults(func=cmd_write_cases)

    run = sub.add_parser("run", help="run laya-apple over the case set on one device")
    run.add_argument("--device", required=True, choices=("gpu", "ane", "auto"))
    run.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    run.add_argument("--model", action="append", default=None)
    run.add_argument("--cases", default=str(CASES_PATH))
    run.add_argument("--output", default=None)
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser("compare", help="compare two option-order result files")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.add_argument("--tol", type=float, default=None)
    cmp_.set_defaults(func=cmd_compare)

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "run" and args.model is None:
        args.model = list(DEFAULT_MODELS)
    args.func(args)


if __name__ == "__main__":
    main()
