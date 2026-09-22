"""Probe: convert real ConvEncoderLayer(s) with real weights, fed a real captured
large-magnitude activation, and compare CPU_ONLY vs CPU_AND_NE vs CPU_AND_GPU vs ALL
against an FP32 torch reference. Localizes the divergence within the real graph rather
than a synthetic single-op graph. Does not modify ane_model.py / backends.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from common import RAW, checkpoint, agent_config, make_request  # noqa: E402

OUT_DIR = RAW / "layernorm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROBE_ROOT = Path("<research-artifacts>/probes/layernorm")
PROBE_ROOT.mkdir(parents=True, exist_ok=True)
UNITS = {"cpu_only": "CPU_ONLY", "cpu_ne": "CPU_AND_NE", "cpu_gpu": "CPU_AND_GPU", "all": "ALL"}


def build_source_and_body(model, length):
    from laya_coreml.torch_model import load_model
    from ane_model import ConvBody

    torch.set_num_threads(8)
    source = load_model(checkpoint(model), length, attention_implementation="explicit")
    body = ConvBody(source, length, local_mode="masked").eval()
    return source, body


def real_features(model, length, body):
    from laya_coreml.tokenizer import Tokenizer
    from backends import ane_features
    from safetensors import safe_open

    src = checkpoint(model)
    tok = Tokenizer(src / "tokenizer")
    _, _, items = make_request(tok, agent_config(model), length, n_questions=1, seed=1)
    with safe_open(str(src / "model.safetensors"), framework="numpy") as w:
        emb = w.get_tensor("encoder.embeddings.tok_embeddings.weight").astype(np.float32)
        temb = w.get_tensor("type_emb.weight").astype(np.float32)
    pos = np.arange(length)
    window = np.abs(pos[:, None] - pos[None, :]) <= body.layers[0].attn.length // 2 * 0 + (
        (length,)  # placeholder unused
    )[0] * 0
    return items, emb, temb, tok.pad_token_id


class LayerRange(nn.Module):
    """Chain of real ConvEncoderLayer objects starting at `start`, length `count`."""

    def __init__(self, body, start, count):
        super().__init__()
        self.layers = nn.ModuleList([body.layers[i] for i in range(start, start + count)])

    def forward(self, x, full_mask, local_mask):
        for layer in self.layers:
            x = layer(x, full_mask if layer.kind == "full_attention" else local_mask)
        return x


def convert_probe(name, module, example_inputs, input_names, out_dir_root=PROBE_ROOT):
    import coremltools as ct

    out_dir = out_dir_root / name
    pkg = out_dir / "model.mlpackage"
    if pkg.exists():
        return pkg
    module = module.eval()
    with torch.inference_mode():
        traced = torch.jit.trace(module, example_inputs, strict=True, check_trace=False)
    mlmodel = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name=n, shape=tuple(v.shape), dtype=np.float16)
            for n, v in zip(input_names, example_inputs)
        ],
        outputs=[ct.TensorType(name="y")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(pkg))
    return pkg


def run_units(pkg_path, feed: dict):
    import coremltools as ct

    results = {}
    for key, unit in UNITS.items():
        mlmodel = ct.models.MLModel(str(pkg_path), compute_units=getattr(ct.ComputeUnit, unit))
        out = mlmodel.predict(feed)
        y = list(out.values())[0]
        results[key] = y.astype(np.float64)
    return results


def main():
    model, length = "laya-typed-decisions", 128
    source, body = build_source_and_body(model, length)
    items, emb, temb, pad_id = real_features(model, length, body)
    from backends import ane_features

    pos = np.arange(length)
    window = np.abs(pos[:, None] - pos[None, :]) <= source.encoder.window // 2
    feats = ane_features(items, length, 1, emb, temb, window, pad_id)
    order = ["embeddings", "full_mask", "local_mask", "type_vectors", "marker_map"]
    t = [torch.from_numpy(np.asarray(feats[k], np.float32)) for k in order]
    embeddings, full_mask, local_mask, type_vectors, marker_map = t

    with torch.inference_mode():
        x = body.embedding_norm(embeddings)
        captured = {}
        for i, layer in enumerate(body.layers):
            if i in (0, 5, 14, 19):
                captured[i] = x.clone()
            x = layer(x, full_mask if layer.kind == "full_attention" else local_mask)
        captured["final"] = x.clone()

    results = {}
    for start in (0, 5, 14, 19):
        x_in = captured[start]
        max_abs = float(x_in.abs().max())
        for count in (1, 3):
            if start + count > len(body.layers):
                continue
            probe = LayerRange(body, start, count)
            with torch.inference_mode():
                ref = probe(x_in, full_mask, local_mask).numpy().astype(np.float64)
            name = f"layers_{start}_{count}"
            pkg = convert_probe(
                name, probe, (x_in, full_mask, local_mask), ["x", "full_mask", "local_mask"]
            )
            feed = {
                "x": x_in.numpy().astype(np.float16),
                "full_mask": full_mask.numpy().astype(np.float16),
                "local_mask": local_mask.numpy().astype(np.float16),
            }
            outs = run_units(pkg, feed)
            diffs = {k: float(np.abs(v - ref).max()) for k, v in outs.items()}
            cpu_only_vs_ne = float(np.abs(outs["cpu_only"] - outs["cpu_ne"]).max())
            results[name] = {
                "start_layer": start,
                "count": count,
                "max_abs_input": max_abs,
                "diff_vs_fp32_ref": diffs,
                "cpu_only_vs_cpu_ne": cpu_only_vs_ne,
            }
            print(name, "max|x_in|=", max_abs, diffs)

    (OUT_DIR / "real_layer_probe.json").write_text(json.dumps(results, indent=2, default=str))
    print("wrote", OUT_DIR / "real_layer_probe.json")


if __name__ == "__main__":
    main()
