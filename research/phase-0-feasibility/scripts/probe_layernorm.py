"""Probe: does BC1S ChannelNorm (explicit mean/square-mean LayerNorm) overflow FP16 on the
Core ML CPU_ONLY execution path for large-magnitude ModernBERT residual activations?

Builds tiny single-op Core ML packages (channel-norm only, B,C,1,L layout matching
ane_model.ChannelNorm) fed with large-magnitude synthetic inputs, and compares CPU_ONLY,
CPU_AND_NE, CPU_AND_GPU, ALL against an FP64 numpy reference. Also probes two safer
formulations. Read-only against existing converted packages; does not modify any existing
script. Writes JSON results to raw/layernorm/.

    uv run python scripts/probe_layernorm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from common import RAW  # noqa: E402

OUT_DIR = RAW / "layernorm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROBE_ROOT = Path("<research-artifacts>/probes/layernorm")
PROBE_ROOT.mkdir(parents=True, exist_ok=True)

UNITS = {"cpu_only": "CPU_ONLY", "cpu_ne": "CPU_AND_NE", "cpu_gpu": "CPU_AND_GPU", "all": "ALL"}


# --------------------------------------------------------------------------- formulations


class ChannelNormBaseline(nn.Module):
    """Exact copy of ane_model.ChannelNorm's math (weight=1, bias=0 for the probe)."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        centered = x - x.mean(dim=1, keepdim=True)
        out = centered * (centered.square().mean(dim=1, keepdim=True) + self.eps).rsqrt()
        return out


class ChannelNormFunctionalLN(nn.Module):
    """(a) torch.nn.functional.layer_norm on a permuted view (B,C,1,L) -> (B,L,1,C)."""

    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.num_channels = num_channels

    def forward(self, x):
        # x: B,C,1,L -> B,L,1,C, normalize over last dim (C), permute back
        xp = x.permute(0, 3, 2, 1)
        out = torch.nn.functional.layer_norm(xp, (self.num_channels,), eps=self.eps)
        return out.permute(0, 3, 2, 1)


class ChannelNormPrescaled(nn.Module):
    """(b) pre-scale by a fixed power of two s, compute var of (centered/s), rescale back.

    mean((x/s - mean(x/s))^2) = mean((x-mean(x))^2) / s^2  exactly in real arithmetic, so
    rsqrt(var/s^2 ... ) recombination stays algebraically identical while keeping the
    squared quantity within FP16 range.
    """

    def __init__(self, eps=1e-5, scale=1024.0):
        super().__init__()
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        s = self.scale
        xs = x / s
        centered_s = xs - xs.mean(dim=1, keepdim=True)
        var_s = centered_s.square().mean(dim=1, keepdim=True)  # = var(x)/s^2
        # rsqrt(var(x) + eps) = rsqrt(var_s*s^2 + eps) = (1/s) * rsqrt(var_s + eps/s^2)
        out = centered_s * (var_s + self.eps / (s * s)).rsqrt()
        return out


FORMULATIONS = {
    "baseline": lambda C: ChannelNormBaseline(),
    "functional_ln": lambda C: ChannelNormFunctionalLN(C),
    "prescaled_1024": lambda C: ChannelNormPrescaled(scale=1024.0),
}


# --------------------------------------------------------------------------- helpers


def np_reference(x_f64: np.ndarray, eps=1e-5) -> np.ndarray:
    centered = x_f64 - x_f64.mean(axis=1, keepdims=True)
    var = np.mean(centered**2, axis=1, keepdims=True)
    return centered / np.sqrt(var + eps)


def convert_probe(name: str, module: nn.Module, B, C, L):
    import coremltools as ct

    out_dir = PROBE_ROOT / name
    example = torch.randn(B, C, 1, L)
    module = module.eval()
    with torch.inference_mode():
        traced = torch.jit.trace(module, (example,), strict=True, check_trace=False)
    mlmodel = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="x", shape=(B, C, 1, L), dtype=np.float16)],
        outputs=[ct.TensorType(name="y")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pkg = out_dir / "model.mlpackage"
    if not pkg.exists():
        mlmodel.save(str(pkg))
    return pkg


def run_units(pkg_path, x_f16: np.ndarray):
    import coremltools as ct

    results = {}
    for key, unit in UNITS.items():
        mlmodel = ct.models.MLModel(str(pkg_path), compute_units=getattr(ct.ComputeUnit, unit))
        out = mlmodel.predict({"x": x_f16})
        y = list(out.values())[0]
        results[key] = y.astype(np.float64)
    return results


def compute_plan_summary(pkg_path, units_key):
    sys.path.insert(0, str(Path(__file__).parent))
    from device_plan import compute_plan
    import coremltools as ct
    from backends import COMPUTE_UNITS

    mlmodel = ct.models.MLModel(str(pkg_path), compute_units=getattr(ct.ComputeUnit, COMPUTE_UNITS[units_key]))
    plan = compute_plan(mlmodel, units_key)
    return {k: v for k, v in plan.items() if k in ("preferred", "supported", "transitions")}


def make_input(B, C, L, magnitude, seed=0):
    rng = np.random.default_rng(seed)
    # Simulate ModernBERT-late-layer residual: large-magnitude, roughly normal, one or two
    # outlier channels dominating (matches observed |x|~29700 behavior).
    base = rng.standard_normal((B, C, 1, L)).astype(np.float64) * (magnitude / 6.0)
    # inject a few outlier channels close to `magnitude`
    outliers = rng.choice(C, size=max(1, C // 64), replace=False)
    base[:, outliers, :, :] += magnitude * rng.choice([-1, 1], size=(len(outliers),))[None, :, None, None]
    return base


def main():
    torch.manual_seed(0)
    B, C, L = 1, 1024, 32
    magnitudes = {"realistic_29700": 29700.0, "multilingual_13000": 13000.0, "small_100": 100.0}

    all_results = {}
    for mag_name, mag in magnitudes.items():
        x_f64 = make_input(B, C, L, mag)
        ref = np_reference(x_f64)
        x_f16 = x_f64.astype(np.float16)
        x_f16_finite = np.nan_to_num(x_f16, nan=0.0, posinf=65504.0, neginf=-65504.0)
        overflow_note = {
            "max_abs_x": float(np.abs(x_f64).max()),
            "max_x_squared": float((x_f64**2).max()),
            "fp16_max": 65504.0,
            "x_squared_overflows_fp16": bool((x_f64**2).max() > 65504.0),
            "any_input_clipped_to_f16": bool(not np.array_equal(x_f16.astype(np.float64), x_f16_finite.astype(np.float64)) or np.any(~np.isfinite(x_f16.astype(np.float64)))),
        }

        for form_name, ctor in FORMULATIONS.items():
            module = ctor(C)
            pkg = convert_probe(f"{form_name}_{mag_name}", module, B, C, L)
            unit_outputs = run_units(pkg, x_f16)
            diffs = {}
            for key, y in unit_outputs.items():
                yy = y.reshape(ref.shape) if y.size == ref.size else y
                d = float(np.abs(yy.astype(np.float64) - ref).max())
                diffs[key] = d
            key = f"{form_name}/{mag_name}"
            all_results[key] = {
                "formulation": form_name,
                "magnitude_case": mag_name,
                "overflow_note": overflow_note,
                "max_abs_diff_vs_fp64_ref": diffs,
                "cpu_only_vs_cpu_ne": float(np.abs(unit_outputs["cpu_only"] - unit_outputs["cpu_ne"]).max()),
            }
            print(key, "diffs:", diffs)

    # device placement check for baseline and each candidate fix, at the realistic magnitude
    placement = {}
    for form_name in FORMULATIONS:
        pkg = PROBE_ROOT / f"{form_name}_realistic_29700" / "model.mlpackage"
        try:
            placement[form_name] = compute_plan_summary(pkg, "cpu_ne")
        except Exception as e:  # noqa: BLE001
            placement[form_name] = {"error": repr(e)}
    all_results["_device_placement_cpu_and_ne"] = placement

    (OUT_DIR / "channelnorm_probe.json").write_text(json.dumps(all_results, indent=2, default=str))
    print("wrote", OUT_DIR / "channelnorm_probe.json")


if __name__ == "__main__":
    main()
