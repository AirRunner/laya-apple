---
title: "Add a --json summary output to scripts/hardware_report.py"
labels: ["good first issue", "benchmark"]
---

## Problem

`scripts/hardware_report.py` produces the community hardware-results bundle described
in `docs/community-benchmarks.md`, but there is no single machine-readable summary file
with a documented schema — someone building a dashboard or comparison table across
`hardware-results/*/` has to parse whatever raw/report files the bundle happens to
contain.

## Why it matters

`CONTRIBUTING.md` requires benchmarks to "write raw per-request data, not only summary
statistics" — that rule is about not losing detail, not about making summaries hard to
consume. A stable, documented JSON summary (one small file per bundle, alongside the raw
data) would make it possible to build an aggregate view across contributed hardware
results without every consumer re-deriving its own parsing of the raw files.

## Expected output

A `--json PATH` (or similar) flag on `scripts/hardware_report.py` that writes a compact
summary: SoC/platform identifiers (matching `laya_apple.artifacts.platform_profile()`),
per-model forward P50s at each measured length, parity pass/fail per model, and a schema
version field. Document the schema in `docs/community-benchmarks.md` next to the bundle
format it already describes.

## How to validate

- Run `scripts/hardware_report.py --quick --json summary.json` and confirm the file is
  valid JSON, includes a schema/version field, and the numbers match the full bundle's
  raw data for the same run.
- Add a test that the summary's schema stays stable (or bump the version field
  deliberately) — see `tests/` for the existing test layout.

## Relevant files

- `scripts/hardware_report.py`
- `docs/community-benchmarks.md`
- `laya_apple/artifacts.py` (`platform_profile`) — how the platform is identified elsewhere

## Difficulty / scope

Good first issue. Contained to one script plus a doc update; no ANE hardware required
to test against an already-cached run or a `--quick` MLX-only pass.
