"""laya-apple command line.

laya-apple predict MODEL --context TEXT --questions JSON [--device auto|gpu|ane]
laya-apple info [MODEL]
laya-apple download MODEL...
laya-apple artifacts build MODEL [--length L ...] | list | verify [MODEL]
laya-apple parity MODEL [--device gpu|ane] [--dtype float16|float32]
laya-apple benchmark MODEL [--device ...] [--lengths ...] [--questions N]
"""

from __future__ import annotations

import argparse
import json
import sys


def _json(obj):
    print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def _read_arg(value: str):
    """Inline JSON/text, @file, or - for stdin."""
    if value == "-":
        return sys.stdin.read()
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8") as f:
            return f.read()
    return value


def cmd_predict(a):
    from . import Laya

    laya = Laya.from_pretrained(a.model, device=a.device, dtype=a.dtype, local_files_only=a.offline)
    questions = json.loads(_read_arg(a.questions))
    context = _read_arg(a.context)
    try:
        context = json.loads(context)
    except ValueError:
        pass
    _json(laya.predict(context=context, questions=questions).to_dict())


def cmd_info(a):
    from . import __version__
    from .artifacts import list_artifacts, platform_profile
    from .model import _coremltools_available, platform_validated
    from .registry import models, resolve

    specs = [resolve(a.model)] if a.model else list(models().values())
    present = {
        (
            x["manifest"].get("source", {}).get("model"),
            x["manifest"].get("artifact", {}).get("length"),
            x["manifest"].get("source", {}).get("revision"),
        ): x["manifest"].get("status")
        for x in list_artifacts()
    }
    _json(
        {
            "laya_apple": __version__,
            "platform": platform_profile(),
            "platform_validated_for_auto_ane": platform_validated(),
            "coremltools_available": _coremltools_available(),
            "models": {
                s.name: {
                    "repo": s.repo,
                    "revision": s.revision,
                    "encoder": s.encoder,
                    "max_len": s.max_len,
                    "mlx_dtypes": list(s.mlx_dtypes),
                    "ane_buckets": list(s.ane_buckets),
                    "auto_ane_buckets": list(s.auto_ane_buckets),
                    "artifacts": {str(b): present.get((s.name, b, s.revision), "missing") for b in s.ane_buckets},
                }
                for s in specs
            },
        }
    )


def cmd_download(a):
    from .hub import checkpoint_path, verify_weights
    from .registry import models, resolve

    for spec in [resolve(m) for m in a.models] if a.models else models().values():
        path = checkpoint_path(spec)
        verify_weights(spec, path)
        print(f"{spec.repo}@{spec.revision[:12]} -> {path} (sha256 verified)")


def cmd_artifacts(a):
    from .artifacts import list_artifacts, load_verified
    from .registry import models, resolve

    if a.action == "list":
        _json(
            [
                {
                    "path": x["path"],
                    "status": x["manifest"].get("status"),
                    "source": x["manifest"].get("source"),
                    "artifact": x["manifest"].get("artifact"),
                    "parity_passed": (x["manifest"].get("parity") or {}).get("passed"),
                    "sha256": x["manifest"].get("integrity", {}).get("artifact_sha256"),
                }
                for x in list_artifacts()
            ]
        )
        return 0
    specs = [resolve(a.model)] if a.model else list(models().values())
    if a.action == "build":
        from .conversion.build import build

        for spec in specs:
            for length in a.length or spec.ane_buckets:
                if a.skip_existing and _artifact_ok(spec, length):
                    print(f"{spec.name} L{length}: present and verified, skipped")
                    continue
                print(build(spec, length, local_files_only=a.offline, force=a.force))
        return 0
    # verify
    failed = 0
    for spec in specs:
        for length in a.length or spec.ane_buckets:
            try:
                _, m = load_verified(spec, length, full=True)
                print(f"OK    {spec.name} L{length} {m.artifact_sha256[:12]}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {spec.name} L{length} {type(e).__name__}: {e}")
    return 1 if failed else 0


def _artifact_ok(spec, length) -> bool:
    from .artifacts import load_verified

    try:
        load_verified(spec, length)
        return True
    except Exception:
        return False


def cmd_parity(a):
    from . import Laya
    from .parity import evaluate
    from .registry import resolve

    spec = resolve(a.model)
    laya = Laya.from_pretrained(spec.name, device=a.device, dtype=a.dtype, local_files_only=a.offline)
    backend = laya.ane if a.device == "ane" else laya.mlx
    precision = "float16" if a.device == "ane" else a.dtype
    max_len = max(backend.buckets) if a.device == "ane" else None
    summary = evaluate(
        spec.name,
        laya.config,
        backend.forward,
        precision=precision,
        max_len=max_len,
        prepare=lambda s, q: laya.prepare(s, q).items,
    )
    summary.update(
        model=spec.name,
        device=a.device,
        artifact_revision=None
        if a.device != "ane"
        else {str(b): m.artifact_sha256 for b, m in backend.manifests.items()},
    )
    _json(summary)
    return 0 if summary["passed"] else 1


def cmd_benchmark(a):
    from .benchmark import run

    out = run(
        a.model,
        device=a.device,
        lengths=a.lengths,
        questions=a.questions,
        warmup=a.warmup,
        iters=a.iters,
        dtype=a.dtype,
        local_files_only=a.offline,
    )
    if a.output:
        with open(a.output, "a", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
    _json(out)


def build_parser():
    p = argparse.ArgumentParser(prog="laya-apple", description=__doc__.split("\n")[0])
    p.add_argument("--offline", action="store_true", help="never touch the network (local_files_only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("predict", help="answer questions about a context")
    s.add_argument("model")
    s.add_argument("--context", required=True, help="text, JSON, @file or -")
    s.add_argument("--questions", required=True, help="JSON object, @file or -")
    s.add_argument("--device", default="auto", choices=["auto", "gpu", "ane"])
    s.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    s.set_defaults(fn=cmd_predict)

    s = sub.add_parser("info", help="models, platform, artifacts")
    s.add_argument("model", nargs="?")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("download", help="fetch and verify pinned checkpoints")
    s.add_argument("models", nargs="*")
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("artifacts", help="build, list or verify ANE artifacts")
    s.add_argument("action", choices=["build", "list", "verify"])
    s.add_argument("model", nargs="?")
    s.add_argument("--length", type=int, action="append")
    s.add_argument("--force", action="store_true")
    s.add_argument("--skip-existing", action="store_true")
    s.set_defaults(fn=cmd_artifacts)

    s = sub.add_parser("parity", help="run the parity gate against the shipped goldens")
    s.add_argument("model")
    s.add_argument("--device", default="gpu", choices=["gpu", "ane"])
    s.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    s.set_defaults(fn=cmd_parity)

    s = sub.add_parser("benchmark", help="warm latency on exact-length requests")
    s.add_argument("model")
    s.add_argument("--device", default="auto", choices=["auto", "gpu", "ane"])
    s.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256])
    s.add_argument("--questions", type=int, default=1)
    s.add_argument("--warmup", type=int, default=5)
    s.add_argument("--iters", type=int, default=30)
    s.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    s.add_argument("--output", help="append JSONL records here")
    s.set_defaults(fn=cmd_benchmark)
    return p


def main(argv=None):
    from .errors import LayaAppleError

    a = build_parser().parse_args(argv)
    try:
        return a.fn(a) or 0
    except LayaAppleError as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
