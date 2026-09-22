"""MLComputePlan summaries for full-model packages (placement evidence per length).

    uv run python scripts/plans.py --model laya-typed-decisions --graph ordinary --units cpu_ne
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from backends import ane_package_dir, load_mlmodel  # noqa: E402
from common import ARTIFACTS, RAW, environment, save_json, valid_lengths  # noqa: E402
from device_plan import compute_plan, headline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--graph", choices=["ordinary", "ane", "enumerated"], required=True)
    ap.add_argument("--units", default="cpu_ne")
    ap.add_argument("--lengths", type=int, nargs="*")
    args = ap.parse_args()
    for L in args.lengths or valid_lengths(args.model):
        if args.graph == "enumerated":
            pkg = ARTIFACTS / "coreml-ordinary" / args.model / "model.mlpackage"
        elif args.graph == "ordinary":
            pkg = ARTIFACTS / "coreml-ordinary" / f"{args.model}-fixed{L}-sdpa" / "model.mlpackage"
        else:
            pkg = ane_package_dir(args.model, L) / "model.mlpackage"
        t = time.perf_counter()
        m = load_mlmodel(pkg, args.units)
        load = time.perf_counter() - t
        plan = compute_plan(m, args.units)
        print(args.model, args.graph, args.units, L, f"load {load:.1f}s", headline(plan), flush=True)
        save_json(
            RAW / "profile" / f"plan-{args.model}-{args.graph}-{args.units}-L{L}.json",
            {"model": args.model, "graph": args.graph, "units": args.units, "length": L, "load_s": load, "plan": plan, "environment": environment()},
        )


if __name__ == "__main__":
    main()
