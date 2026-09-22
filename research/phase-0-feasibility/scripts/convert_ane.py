"""Convert the BC1S ConvBody to a fixed-shape Core ML package and record provenance.

    uv run python scripts/convert_ane.py --model laya-typed-decisions --length 128
    uv run python scripts/convert_ane.py --model laya-typed-decisions --length 1024 --variant windowed

Before conversion the ConvBody is checked in FP32 PyTorch against the laya-coreml
export-only DecisionModel on a real exact-length request (layout/masking check only; the
parity experiment compares Core ML against upstream PyTorch goldens).
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
    agent_config,
    append_jsonl,
    checkpoint,
    environment,
    make_request,
    sha256_file,
    tree_sha256,
)


def build(model: str, length: int, variant: str, block: int):
    import torch
    from laya_coreml.torch_model import load_model

    from ane_model import ConvBody

    torch.set_num_threads(8)
    source = load_model(checkpoint(model), length, attention_implementation="explicit")
    body = ConvBody(source, length, local_mode=variant, block=block).eval()
    return source, body


def layout_check(model, source, body, length, batch):
    """FP32 ConvBody vs FP32 export-only DecisionModel on real rows (max |logit diff|)."""
    import torch
    from laya_coreml.tokenizer import Tokenizer
    from safetensors import safe_open

    from backends import ane_features

    src = checkpoint(model)
    tok = Tokenizer(src / "tokenizer")
    _, _, items = make_request(tok, agent_config(model), length, n_questions=batch, seed=1)
    with safe_open(str(src / "model.safetensors"), framework="numpy") as w:
        emb = w.get_tensor("encoder.embeddings.tok_embeddings.weight").astype(np.float32)
        temb = w.get_tensor("type_emb.weight").astype(np.float32)
    pos = np.arange(length)
    window = np.abs(pos[:, None] - pos[None, :]) <= source.encoder.window // 2
    feats = ane_features(items, length, batch, emb, temb, window, tok.pad_token_id)
    order = ["embeddings", "full_mask", "local_mask", "type_vectors", "marker_map"]
    t = [torch.from_numpy(np.asarray(feats[k], np.float32)) for k in order]
    with torch.inference_mode():
        got, _ = body(*t)
        got = got.reshape(batch, -1).numpy()
        ids = torch.full((batch, length), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((batch, length), dtype=torch.long)
        mpos = torch.zeros((batch, 32), dtype=torch.long)
        mmask = torch.zeros((batch, 32), dtype=torch.bool)
        for r, it in enumerate(items):
            ids[r, : len(it["ids"])] = torch.tensor(it["ids"])
            att[r, : len(it["ids"])] = 1
            mpos[r, : len(it["markers"])] = torch.tensor(it["markers"])
            mmask[r, : len(it["markers"])] = True
        qt = torch.tensor([it["qtype"] for it in items])
        ref, _ = source(ids, att, mpos, mmask, qt)
        ref = ref.numpy()
    diffs = [
        float(np.abs(got[r, : len(it["markers"])] - ref[r, : len(it["markers"])]).max())
        for r, it in enumerate(items)
    ]
    return {"rows": len(items), "max_abs_logit_diff_fp32": max(diffs), "per_row": diffs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--variant", default="masked", choices=["masked", "windowed"])
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--skip-layout-check", action="store_true")
    args = ap.parse_args()

    import coremltools as ct
    import torch

    from backends import ane_package_dir

    out = ane_package_dir(args.model, args.length, args.batch, args.variant, args.block)
    record = {
        "experiment": "convert_ane_bc1s",
        "model": args.model,
        "length": args.length,
        "batch": args.batch,
        "variant": args.variant,
        "block": args.block if args.variant == "windowed" else None,
        "output": str(out),
        "environment": environment(),
    }
    t0 = time.perf_counter()
    try:
        if (out / "manifest.json").exists():
            print("exists", out)
            return
        source, body = build(args.model, args.length, args.variant, args.block)
        if not args.skip_layout_check:
            record["layout_check"] = layout_check(args.model, source, body, args.length, args.batch)
            print("layout", record["layout_check"]["max_abs_logit_diff_fp32"], flush=True)
        width = body.embedding_norm.weight.shape[1]
        B, L = args.batch, args.length
        example = {
            "embeddings": torch.randn(B, width, 1, L),
            "full_mask": torch.zeros(B, L, 1, L),
            "local_mask": torch.zeros(B, L, 1, L),
            "type_vectors": torch.zeros(B, width, 1, 1),
            "marker_map": torch.zeros(B, L, 1, 32),
        }
        t1 = time.perf_counter()
        with torch.inference_mode():
            traced = torch.jit.trace(body, tuple(example.values()), strict=True, check_trace=False)
        record["trace_seconds"] = time.perf_counter() - t1
        t1 = time.perf_counter()
        mlmodel = ct.convert(
            traced,
            source="pytorch",
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(name=k, shape=tuple(v.shape), dtype=np.float16) for k, v in example.items()
            ],
            outputs=[ct.TensorType(name="logits"), ct.TensorType(name="cls")],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
            skip_model_load=True,
        )
        record["convert_seconds"] = time.perf_counter() - t1
        out.mkdir(parents=True, exist_ok=False)
        mlmodel.save(str(out / "model.mlpackage"))
        src = checkpoint(args.model)
        manifest = {
            "format": "laya-apple-phase0-ane",
            "model": args.model,
            "repo": MODELS[args.model]["repo"],
            "revision": MODELS[args.model]["revision"],
            "source_weights_sha256": sha256_file(src / "model.safetensors"),
            "shape": {"batch": B, "length": L, "options": 32},
            "variant": args.variant,
            "block": record["block"],
            "precision": "float16 compute (ct.precision.FLOAT16), float16 IO",
            "minimum_deployment_target": "macOS15",
            "package_sha256": tree_sha256(out / "model.mlpackage"),
            "graph_source": "scripts/ane_model.py (adapted from laya-coreml@4619e04 experiments/ane_engineering/model.py)",
            "script_sha256": {
                p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "layout_check": record.get("layout_check"),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        record["status"] = "ok"
        record["package_sha256"] = manifest["package_sha256"]
        record["package_bytes"] = sum(p.stat().st_size for p in (out / "model.mlpackage").rglob("*") if p.is_file())
    except Exception as e:
        record["status"] = "failed"
        record["error"] = repr(e)
        record["traceback"] = traceback.format_exc()[-4000:]
    record["seconds"] = time.perf_counter() - t0
    append_jsonl(RAW / "conversion.jsonl", record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("environment",)}, indent=1)[:2500])
    if record["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
