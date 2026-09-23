---
title: "Add an M4 Pro benchmark result"
labels: ["good first issue", "help wanted", "benchmark", "hardware"]
---

## Problem

The only M-series chip laya-apple has been measured on is the M4 Max (`docs/support-matrix.md`).
An M4 Pro shares the M4 generation's ANE and GPU core design but has fewer GPU cores and
a different memory-bandwidth tier, so both the MLX/ANE crossover point (README, "Auto
routing") and the heterogeneous-serving multiplier are unverified on it.

## Why it matters

The routing thresholds baked into the shipped `laya_apple/data/routing.json` are derived
from M4 Max measurements only. `laya-apple calibrate` exists precisely so other machines
can get a locally-correct profile, but nobody has run it end-to-end on an M4 Pro and
published the result to confirm the pipeline works there.

## Expected output

A `scripts/hardware_report.py` bundle for an M4 Pro under
`hardware-results/<soc>-macos<major>/` (`docs/community-benchmarks.md`), and, if time
allows, the output of `laya-apple calibrate` showing whether the machine gets its own
local routing profile or falls back to `platform_not_validated`.

## How to validate

1. `uv sync --extra dev --extra ane --extra convert` on the M4 Pro machine.
2. `laya-apple artifacts build laya-typed-decisions` to build and parity-validate ANE artifacts locally.
3. `scripts/hardware_report.py --quick` first, then a full run if time allows.
4. Optionally `laya-apple calibrate laya-typed-decisions` and include its output/profile in the PR description.

## Relevant files

- `docs/support-matrix.md`
- `docs/community-benchmarks.md`
- `laya_apple/profiles.py` — what `calibrate` writes and how a local profile is chosen
- `scripts/hardware_report.py`

## Difficulty / scope

Good first issue. Requires an M4 Pro Mac; no code changes needed.
