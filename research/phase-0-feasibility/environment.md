# Environment

The machine-readable snapshot is [`raw/environment.json`](raw/environment.json). Every raw
record also embeds `environment()` or `conditions()` from the moment it was taken.

## Hardware

| | |
|---|---|
| SoC | Apple M4 Max |
| CPU | 16 cores: 12 performance + 4 efficiency |
| GPU | 40 cores, Metal 4 |
| Neural Engine | 16-core ANE (M4 family) |
| Unified memory | 64 GiB |
| Power | AC, low-power mode off; `pmset -g therm` reported no thermal or performance warning at any recorded point |
| Model artifacts | converted packages on external Samsung T7 USB SSD (`<research-artifacts>`); HF cache on the same volume |

## Software

| | |
|---|---|
| macOS | 26.6.2 (build 25G83) |
| Xcode / Instruments | Xcode 27.0 (27A266a); `xctrace` 27.0 (the Core ML template is now called **Core AI**) |
| Compiler | Apple clang 21.0.0 |
| Python | 3.12.14 (uv-managed venv) |
| uv | 0.12.13 |
| PyTorch | 2.7.0 (pinned: coremltools 9.0's tested version) |
| NumPy | 2.1.3 (coremltools 9.0 breaks on NumPy ≥ 2.5) |
| coremltools | 9.0 |
| transformers | 5.17.0 |
| tokenizers | 0.23.2 |
| safetensors | 0.8.0 |
| huggingface-hub | 1.32.0 |
| MLX / mlx-metal | 0.32.2 |
| psutil | 7.2.2 |

The lockfile is `uv.lock`, with the dependency list in `pyproject.toml`. The research
environment is a separate uv project, so none of this reaches the `laya_apple` runtime
package.

## Reference implementations (exact commits)

| Project | Commit | Used as |
|---|---|---|
| `NandhaKishorM/laya` | `573e5b62696ba441230cd6be71d593331b5d23af` (package 0.3.5) | semantic reference: PyTorch `Agent`, prompt, calibration |
| `mizorewww/laya-mlx` | `0a859518634112655cb97c745dbf04f5191aaf13` (0.2.0) | MLX GPU backend, installed unmodified |
| `mizorewww/laya-coreml` | `4619e0483f07adf39068532e85b42ec2347edb83` (0.1.1) | ordinary Core ML converter/runtime, installed unmodified; ANE graph adapted (see `scripts/ane_model.py`) |

All three are Apache-2.0. Adapted code carries a provenance header naming its source file
and commit.

## Checkpoints (exact Hugging Face revisions)

| Model | Revision | `model.safetensors` SHA-256 | Encoder | max_len | head_max_len |
|---|---|---|---|---:|---:|
| `convaiinnovations/laya` | `c5d78730f3493e4fe16d61507ef4b78eef7318cf` | `891102d3…8d86c` | ModernBERT-large (28 layers, d=1024, 16 heads) | 512 | 192 |
| `convaiinnovations/laya-multilingual` | `052592a15d198d9ad47da779604259b10b47b7aa` | `9d628fd9…8f204` | mmBERT-base (22 layers, d=768, 12 heads, vocab 256k) | 1024 | 256 |
| `convaiinnovations/laya-typed-decisions` | `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2` | `4fa56de7…07a24e` | ModernBERT-large (28 layers, d=1024, 16 heads) | 1024 | 256 |

- `laya` is pinned to the revision both prior ports used. Its current head, `1c5edc17`
  (2026-09-20), adds bundled sub-checkpoints and a README change, but its root
  `model.safetensors` is byte-identical: same LFS SHA-256.
- All encoders alternate one global-attention layer with two sliding-window layers,
  `local_attention = 128`, so the window is |i−j| ≤ 64.
- RoPE θ is 160000 for global layers on all models. For local layers it is 10000 on the
  ModernBERT-large models and 160000 on mmBERT.

## Runtime configuration

- MLX: laya-mlx `Agent(dtype=float16|float32, batch_size=64, compile=False)`, default
  device (GPU).
- PyTorch: upstream `Agent(device="cpu"|"mps")`. It selects FP32 on both, and
  `reference_compile=False` keeps the eager path.
- Core ML:
  - Converted ML Programs: FP16 compute precision, deployment target macOS 15.
  - Loaded through `coremltools.models.MLModel(compute_units=…)`; each compute-unit
    setting is a separate configuration.
- Environment variables set by `scripts/common.py`: `HF_HUB_DISABLE_PROGRESS_BARS=1`,
  `TOKENIZERS_PARALLELISM=false` and `TQDM_DISABLE=1`. None of them affects computation.
