---
title: "Research: ANE quantization with parity"
labels: ["research", "ane", "correctness", "performance"]
---

## Problem

The README lists quantized artifacts explicitly under "Not measured yet: energy use,
quantized artifacts and cross-SoC validation." `CONTRIBUTING.md`'s artifact release
policy already anticipates this work and its likely outcome: "A variant that fails, such
as a 4-, 6- or 8-bit quantization, or a graph that is faster but numerically off, is
recorded under `research/` with its raw data. It is never registered, offered or routed
to." No quantization experiment exists yet in `research/`.

## Why it matters

Quantization is a natural next lever for ANE latency and memory, but laya-apple's whole
value proposition is that it never trades correctness for speed silently (README,
"Correctness"; the FP16 acceptance criteria: probability error ≤ 0.02, 0 hard
mismatches, bit-identical repeats). Any quantized artifact has to clear that same bar or
it does not ship — that is what makes this a research task rather than an engineering
one.

## Expected output

A research writeup under `research/` (new subdirectory, following the
`phase-0-feasibility/` convention of raw data + scripts + a report) that builds one or
more quantized (4-, 6-, or 8-bit) Core ML exports of at least one model, runs them
through the exact same parity gate as the shipped FP16 artifacts (against the upstream
PyTorch FP32 goldens, same tolerances, same hard-mismatch/near-tie definitions as
`README.md`'s "Correctness" section), and reports the result — pass or fail — with
raw per-row data, not just summary statistics.

## How to validate

- Use the project's existing parity harness (`laya_apple/parity/`,
  `laya_apple/conversion/torch_reference.py` goldens) rather than a new, separately
  written comparison, so the result is directly comparable to the FP16/FP32 rows already
  in `README.md`.
- A passing quantized variant still must clear the full artifact release gate
  (`CONTRIBUTING.md`) before it could ever be proposed for shipping; this issue is
  scoped to the research measurement.
- A failing variant is recorded under `research/` with its raw data per
  `CONTRIBUTING.md`'s "Failed variants stay research" — do not treat "it didn't pass" as
  reason to omit the writeup.

## Relevant files

- `README.md` ("Not measured yet", "Correctness" — tolerances and definitions)
- `CONTRIBUTING.md` ("Artifact release policy")
- `laya_apple/parity/__init__.py`, `laya_apple/parity/ane.py`
- `laya_apple/conversion/`

## Difficulty / scope

Research. Needs ANE hardware, the `ane`/`convert`/`reference` extras, and familiarity
with Core ML quantization APIs. Open-ended; a well-evidenced failure is a valid outcome
per the project's own stated policy.
