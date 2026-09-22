"""Cold-start: does a pre-compiled .mlmodelc avoid the ~40 s ANE compile in every new process?

    uv run python scripts/coldstart.py --model laya-typed-decisions --length 128 --runs 3

Each measurement is a FRESH Python process (the only way to see per-process cost):
  mlpackage      MLModel(<.mlpackage>)  - coremltools compiles to a temp .mlmodelc each time
  compiled       CompiledMLModel(<stable .mlmodelc path>) - compiled once, reused
For each: load seconds and first-prediction milliseconds, under CPU_AND_NE.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

CHILD = r"""
import sys, time, json, numpy as np
sys.path.insert(0, {scripts!r})
import coremltools as ct
mode, path, L, width = {mode!r}, {path!r}, {L}, {width}
t = time.perf_counter()
if mode == "compiled":
    m = ct.models.CompiledMLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
else:
    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
load = time.perf_counter() - t
f = {{"embeddings": np.zeros((1,width,1,L), np.float16), "full_mask": np.zeros((1,L,1,L), np.float16),
     "local_mask": np.zeros((1,L,1,L), np.float16), "type_vectors": np.zeros((1,width,1,1), np.float16),
     "marker_map": np.zeros((1,L,1,32), np.float16)}}
t = time.perf_counter(); m.predict(f); first = (time.perf_counter() - t) * 1e3
t = time.perf_counter(); m.predict(f); second = (time.perf_counter() - t) * 1e3
print(json.dumps({{"load_s": load, "first_ms": first, "second_ms": second}}))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya-typed-decisions")
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    import coremltools as ct

    from backends import ane_package_dir
    from common import ARTIFACTS, RAW, environment, save_json

    pkg = ane_package_dir(args.model, args.length) / "model.mlpackage"
    width = 1024 if "typed" in args.model or args.model == "laya" else 768
    compiled_dir = ARTIFACTS / "compiled" / args.model / f"L{args.length}"
    if not (compiled_dir / "model.mlmodelc").exists():
        compiled_dir.mkdir(parents=True, exist_ok=True)
        tmp = ct.utils.compile_model(str(pkg))
        shutil.move(str(tmp), str(compiled_dir / "model.mlmodelc"))
    results = []
    for run in range(args.runs):
        for mode, path in (("mlpackage", str(pkg)), ("compiled", str(compiled_dir / "model.mlmodelc"))):
            code = CHILD.format(scripts=str(Path(__file__).parent), mode=mode, path=path, L=args.length, width=width)
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            rec = {"run": run, "mode": mode}
            try:
                rec.update(json.loads(out.stdout.strip().splitlines()[-1]))
            except Exception:
                rec["error"] = out.stderr[-800:]
            results.append(rec)
            print(rec, flush=True)
    save_json(
        RAW / "coldstart" / f"{args.model}-L{args.length}.json",
        {"experiment": "coldstart", "model": args.model, "length": args.length, "results": results, "environment": environment()},
    )


if __name__ == "__main__":
    main()
