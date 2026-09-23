"""Cold start: ane_startup="wait" against "background" (DEVELOPMENT_PLAN.md §12.4, v0.3).

Core ML caches its on-device ANE compile per artifact *location*. To measure a genuinely
cold start, each run copies the model's registered artifacts into a fresh cache root, then
measures in a new process:

- `ready_s`: how long `Laya.from_pretrained(..., execution="workers")` takes to return;
- `first_request_s`: the first short request's latency, and the device it used;
- `ane_ready_s`: time until the ANE serves requests;
- the same process then re-opens the model at the now-warm location (`warm_ready_s`).

    LAYA_APPLE_CACHE=<cache> HF_HUB_OFFLINE=1 \\
        .venv/bin/python scripts/bench_coldstart.py [MODEL ...] [--out benchmarks/v0.3/coldstart.json]

The copies go under <cache>/../laya-apple-coldstart-<pid>/ and are removed afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("laya-typed-decisions", "laya", "laya-multilingual")


def _child(model: str, mode: str) -> dict:
    from laya_apple import Laya
    from laya_apple.workload import make_request

    t0 = time.perf_counter()
    laya = Laya.from_pretrained(model, execution="workers", ane_startup=mode, local_files_only=True)
    ready = time.perf_counter() - t0
    state, qs = make_request(laya.tokenizer, laya.config, 64, 1, seed=3)
    t1 = time.perf_counter()
    r = laya.predict(context=state, questions=qs)
    first = time.perf_counter() - t1
    laya.wait_for_ane()
    ane_ready = time.perf_counter() - t0
    after = laya.predict(context=state, questions=qs)
    laya.close()
    t2 = time.perf_counter()
    with Laya.from_pretrained(model, execution="workers", ane_startup=mode, local_files_only=True):
        warm_ready = time.perf_counter() - t2
    return {
        "model": model,
        "mode": mode,
        "ready_s": round(ready, 3),
        "first_request_s": round(first, 4),
        "first_request_device": r.runtime.device,
        "first_request_reason": r.runtime.routing_reason,
        "ane_ready_s": round(ane_ready, 3),
        "after_ready_device": after.runtime.device,
        "same_answers": r.answers == after.answers,
        "warm_ready_s": round(warm_ready, 3),
    }


def _fresh_cache(src_cache: Path, model: str, dest: Path) -> None:
    from laya_apple.registry import resolve

    spec = resolve(model)
    src = src_cache / "artifacts" / spec.name
    shutil.copytree(src, dest / "artifacts" / spec.name, symlinks=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", nargs="*", default=list(MODELS))
    p.add_argument("--out", default=str(ROOT / "benchmarks" / "v0.3" / "coldstart.json"))
    p.add_argument("--child", nargs=2, metavar=("MODEL", "MODE"), help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.child:
        print(json.dumps(_child(*a.child)))
        return 0

    from laya_apple.hub import cache_root

    src_cache = cache_root()
    work = src_cache.parent / f"laya-apple-coldstart-{os.getpid()}"
    rows = []
    try:
        for model in a.models:
            for mode in ("wait", "background"):
                dest = work / f"{model}-{mode}"
                _fresh_cache(src_cache, model, dest)
                env = dict(os.environ, LAYA_APPLE_CACHE=str(dest), HF_HUB_OFFLINE="1")
                r = subprocess.run(
                    [sys.executable, __file__, "--child", model, mode],
                    env=env,
                    capture_output=True,
                    text=True,
                    cwd=ROOT,
                )
                if r.returncode != 0:
                    raise SystemExit(f"{model} {mode} failed:\n{r.stderr[-3000:]}")
                row = json.loads(r.stdout.strip().splitlines()[-1])
                rows.append(row)
                print(json.dumps(row), flush=True)
                shutil.rmtree(dest)
    finally:
        if work.exists() and work.name.startswith("laya-apple-coldstart-"):
            shutil.rmtree(work)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": rows}, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
