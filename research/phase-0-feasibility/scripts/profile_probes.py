"""Sub-graph probes for Experiment 3 (long-context ANE profiling): attribute ANE time to
embedding/host prep, QKV projection, local attention, global attention, softmax, MLP, norms,
and partition boundaries, instead of assuming attention dominates.

    uv run python scripts/profile_probes.py convert --model laya-typed-decisions --lengths 128 256 512 1024
    uv run python scripts/profile_probes.py run --model laya-typed-decisions --lengths 128 --units cpu_ne \
        --warmup 10 --samples 100 --output raw/profile/probes-laya-typed-decisions.jsonl

`convert` exports one fixed-shape FP16 Core ML package per probe (built from the real
checkpoint weights via `ane_model.ConvBody`/`Probe`); `run` records the MLComputePlan
(anticipated per-operator device placement) and warm `MLModel.predict` timing samples for
already-converted probes. Only `run --samples 5` (wiring only) may be invoked from here; the
real timed sweep is executed later by the orchestrator in a quiet window.

Probe kinds (layer indices per methodology.md: index 0 has identity attn_norm, so global
attention is probed at layer 3, local/sliding-window attention at layer 1):
  layer      one full encoder layer (attn + mlp), global (layer 3) or local (layer 1)
  attention  attention sub-block only (masked, both global/local; windowed for local)
  scores     QK^T + additive mask + softmax + AV only, no qkv/out projections (isolates
             softmax/score work from the linear-in-C projections) -- added here, not in
             ane_model.py, since it is profiling-only and not part of the exported graph.
  mlp        gated MLP block only
  qkv        qkv projection + head split + out projection (no attention math)
  norm       one ChannelNorm application

Each probe runs `repeat` times identically across the full sequence length so a short
sequence still does enough work to time; `repeat` is recorded in the manifest so the run
subcommand and any later analysis can normalise it out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    MODELS,
    RAW,
    append_jsonl,
    checkpoint,
    conditions,
    environment,
    sha256_file,
    stats,
    tree_sha256,
)

PROBES_ROOT_NAME = "probes"
DEFAULT_REPEAT = 4

# (kind, layer_index, mode) -- mode is the ConvBody local_mode used to build the probe.
PROBE_SPECS = [
    ("layer", 3, "masked"),      # global attention layer (index 0 has identity attn_norm)
    ("layer", 1, "masked"),      # local (sliding-window) layer, masked graph
    ("attention", 3, "masked"),  # global attention sub-block
    ("attention", 1, "masked"),  # local attention sub-block, dense-masked
    ("attention", 1, "windowed"),  # local attention sub-block, exact block-windowed
    ("mlp", 1, "masked"),
    ("qkv", 1, "masked"),
    ("norm", 1, "masked"),
    ("scores", 3, "masked"),     # QK^T + mask + softmax + AV only, global layer
]


def probes_dir(model: str, kind: str, layer: int, mode: str, length: int) -> Path:
    from common import ARTIFACTS

    return ARTIFACTS / PROBES_ROOT_NAME / model / f"{kind}-{layer}-{mode}-L{length}"


def build_body(model: str, length: int, mode: str, block: int):
    import torch
    from laya_coreml.torch_model import load_model

    from ane_model import ConvBody

    torch.set_num_threads(8)
    source = load_model(checkpoint(model), length, attention_implementation="explicit")
    body = ConvBody(source, length, local_mode=mode, block=block).eval()
    return source, body


def build_probe(body, kind: str, layer: int, repeat: int):
    """Return an nn.Module(x, mask) -> x for the requested probe kind."""
    import torch
    from torch import nn

    from ane_model import Probe

    if kind != "scores":
        return Probe(body, kind, layer_index=layer, repeat=repeat)

    class ScoresOnly(nn.Module):
        """QK^T + additive mask + softmax + AV only; projections done once, outside the
        repeat loop, so `repeat` isolates the score/softmax/weighted-sum cost from the
        (linear-in-C) qkv and output projections."""

        def __init__(self, body, layer_index, repeat):
            super().__init__()
            self.layer = body.layers[layer_index]
            self.repeat = repeat

        def forward(self, x, mask):
            attn = self.layer.attn
            qn = self.layer.attn_norm(x)
            q, k, v = attn.qkv(qn).chunk(3, dim=1)
            heads = list(
                zip(q.split(attn.dim, dim=1), k.split(attn.dim, dim=1), v.split(attn.dim, dim=1))
            )
            out = x
            for _ in range(self.repeat):
                pieces = []
                for qi, ki, vi in heads:
                    if attn.rope:
                        qi, ki = attn.rotate(qi), attn.rotate(ki)
                    pieces.append(attn._dense(qi, ki, vi, mask))
                out = out + torch.cat(pieces, dim=1)
            return out

    return ScoresOnly(body, layer, repeat)


def convert_one(model, kind, layer, mode, length, body, repeat, record_extra):
    import coremltools as ct
    import torch

    out = probes_dir(model, kind, layer, mode, length)
    if (out / "manifest.json").exists():
        print("exists", out)
        return

    probe = build_probe(body, kind, layer, repeat).eval()
    width = body.embedding_norm.weight.shape[1]
    example = {
        "x": torch.randn(1, width, 1, length),
        "mask": torch.zeros(1, length, 1, length),
    }
    with torch.inference_mode():
        traced = torch.jit.trace(probe, tuple(example.values()), strict=True, check_trace=False)
    mlmodel = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name=k, shape=tuple(v.shape), dtype=np.float16) for k, v in example.items()
        ],
        outputs=[ct.TensorType(name="y")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
    )
    out.mkdir(parents=True, exist_ok=False)
    mlmodel.save(str(out / "model.mlpackage"))
    src = checkpoint(model)
    manifest = {
        "format": "laya-apple-phase0-probe",
        "model": model,
        "repo": MODELS[model]["repo"],
        "revision": MODELS[model]["revision"],
        "source_weights_sha256": sha256_file(src / "model.safetensors"),
        "kind": kind,
        "layer": layer,
        "mode": mode,
        "length": length,
        "repeat": repeat,
        "inputs": {k: list(v.shape) for k, v in example.items()},
        "precision": "float16 compute (ct.precision.FLOAT16), float16 IO",
        "minimum_deployment_target": "macOS15",
        "package_sha256": tree_sha256(out / "model.mlpackage"),
        "graph_source": "scripts/profile_probes.py (Probe from scripts/ane_model.py)",
        "script_sha256": {
            p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("built", out)
    append_jsonl(RAW / "conversion.jsonl", {**record_extra, "output": str(out), "status": "ok"})


def cmd_convert(args):
    for length in args.lengths:
        bodies = {}
        for kind, layer, mode in PROBE_SPECS:
            t0 = time.perf_counter()
            record_extra = {
                "experiment": "convert_probe",
                "model": args.model,
                "kind": kind,
                "layer": layer,
                "mode": mode,
                "length": length,
                "repeat": args.repeat,
            }
            try:
                if mode not in bodies:
                    bodies[mode] = build_body(args.model, length, mode, args.block)
                _, body = bodies[mode]
                convert_one(args.model, kind, layer, mode, length, body, args.repeat, record_extra)
            except Exception as e:
                append_jsonl(
                    RAW / "conversion.jsonl",
                    {
                        **record_extra,
                        "status": "failed",
                        "error": repr(e),
                        "traceback": traceback.format_exc()[-4000:],
                        "seconds": time.perf_counter() - t0,
                    },
                )
                raise


def cmd_run(args):
    from backends import load_mlmodel
    from device_plan import compute_plan, headline

    env = environment()
    rng = np.random.default_rng(0)
    out_path = Path(args.output)
    for length in args.lengths:
        for kind, layer, mode in PROBE_SPECS:
            pdir = probes_dir(args.model, kind, layer, mode, length)
            manifest_path = pdir / "manifest.json"
            if not manifest_path.exists():
                print("missing (run convert first)", pdir)
                continue
            manifest = json.loads(manifest_path.read_text())
            t0 = time.perf_counter()
            model = load_mlmodel(pdir / "model.mlpackage", args.units)
            load_seconds = time.perf_counter() - t0

            plan = compute_plan(model, args.units, keep_ops=True)
            print(kind, layer, mode, length, headline(plan))

            inputs = {
                name: rng.standard_normal(shape).astype(np.float16)
                for name, shape in manifest["inputs"].items()
            }
            for _ in range(args.warmup):
                model.predict(inputs)
            samples_ms = []
            for _ in range(args.samples):
                t = time.perf_counter_ns()
                model.predict(inputs)
                samples_ms.append((time.perf_counter_ns() - t) / 1e6)

            record = {
                "experiment": "profile_probe",
                "model": args.model,
                "kind": kind,
                "layer": layer,
                "mode": mode,
                "length": length,
                "repeat": manifest["repeat"],
                "units": args.units,
                "package_sha256": manifest["package_sha256"],
                "load_seconds": load_seconds,
                "compute_plan": plan,
                "warmup": args.warmup,
                "samples_requested": args.samples,
                "raw_ms": samples_ms,
                "stats": stats(samples_ms),
                "conditions": conditions(),
            }
            append_jsonl(out_path, record)
    append_jsonl(out_path, {"experiment": "profile_probe_environment", "environment": env})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert")
    c.add_argument("--model", required=True, choices=list(MODELS))
    c.add_argument("--lengths", type=int, nargs="+", required=True)
    c.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    c.add_argument("--block", type=int, default=64)
    c.set_defaults(func=cmd_convert)

    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=list(MODELS))
    r.add_argument("--lengths", type=int, nargs="+", required=True)
    r.add_argument("--units", default="cpu_ne", choices=["cpu_only", "cpu_gpu", "cpu_ne", "all"])
    r.add_argument("--warmup", type=int, default=10)
    r.add_argument("--samples", type=int, default=100)
    r.add_argument("--output", required=True)
    r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
