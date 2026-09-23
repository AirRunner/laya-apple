---
title: "Add an M3 Max benchmark result"
labels: ["good first issue", "help wanted", "benchmark", "hardware"]
---

## Problem

Every number in this repository — every forward-pass latency, every parity result,
every heterogeneous-serving multiplier — comes from one machine: an Apple M4 Max on
macOS 26.6.2 (see `docs/support-matrix.md`, "Platform scope"). There is no community
result yet showing how laya-apple performs, or whether the ANE parity gate even passes,
on an M3 Max.

## Why it matters

`docs/support-matrix.md` is explicit that other Apple SoCs are only an "expected"
hypothesis for the MLX backend, and that ANE placement, correctness and routing
thresholds are unknown off the tested machine. Community hardware results are how that
gap gets closed one machine at a time.

## Expected output

A hardware-results bundle produced by `scripts/hardware_report.py` on an M3 Max
machine, committed under `hardware-results/<soc>-macos<major>/` (see
`docs/community-benchmarks.md` for the exact directory naming and bundle contents), plus
a short PR description noting macOS version, MLX version, coremltools version, and
whether the ANE parity gate passed for each model.

## How to validate

1. `uv sync --extra dev --extra ane --extra convert` on the M3 Max machine.
2. Build and parity-validate the ANE artifacts locally: `laya-apple artifacts build laya-typed-decisions` (repeat for `laya` and `laya-multilingual` if time allows).
3. Run `scripts/hardware_report.py` (see `--quick` for a fast pass first) to produce the bundle.
4. Confirm the bundle lands under `hardware-results/<soc>-macos<major>/` per `docs/community-benchmarks.md`, and that it includes raw per-request data, not only summary numbers (`CONTRIBUTING.md`, "Benchmarks record raw data").

## Relevant files

- `docs/support-matrix.md` — platform scope table to update once results land
- `docs/community-benchmarks.md` — bundle format and directory convention
- `scripts/hardware_report.py` — the report generator
- `benchmarks/v1.0.md` — what a full report on the tested machine looks like

## Difficulty / scope

Good first issue. No code changes required, only hardware access (an M3 Max Mac) and
following the existing benchmark tooling. Time is mostly machine time, not engineering
time.
