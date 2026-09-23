---
title: "Add an M1/M2 MLX-only benchmark result"
labels: ["good first issue", "help wanted", "benchmark", "hardware"]
---

## Problem

`docs/support-matrix.md` marks M-series SoCs other than M4 Max as "expected to run the
MLX backend correctly (hypothesis, not measured)". An M1 or M2 result — even without
building any ANE artifacts — would turn that hypothesis into a measured fact for the
oldest supported hardware generation.

## Why it matters

MLX is the default, always-available backend (`docs/architecture.md`: "MLX is the general backend"). Confirming it actually runs correctly
and at a known speed on M1/M2 matters more, for most users, than ANE validation, and
this is the lowest-effort way to extend hardware coverage since it needs no ANE artifact
build (`coremltools`/`torch` extras) at all.

## Expected output

A `scripts/hardware_report.py --quick` bundle (MLX-only is fine; the ANE columns will
simply show no artifacts) committed under `hardware-results/<soc>-macos<major>/` per
`docs/community-benchmarks.md`, with the PR description noting the SoC (M1, M1 Pro/Max,
M2, etc.), macOS version, and MLX version.

## How to validate

1. `uv sync` (base install only — no `ane`/`convert` extras needed for an MLX-only result).
2. `laya-apple download laya-typed-decisions` (or the model(s) you want to cover).
3. `scripts/hardware_report.py --quick`.
4. Confirm the bundle lands under `hardware-results/<soc>-macos<major>/`.

## Relevant files

- `docs/support-matrix.md`
- `docs/community-benchmarks.md`
- `scripts/hardware_report.py`

## Difficulty / scope

Good first issue, the lowest-effort hardware contribution in the project: no ANE build,
no `torch`/`coremltools` install, just an M1 or M2 Mac.
