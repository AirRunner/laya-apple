---
title: "Improve Core ML artifact cold-start caching"
labels: ["help wanted", "coreml", "ane", "performance"]
---

## Problem

`benchmarks/v0.3.md` ("Cold start") measures that a fresh artifact location costs
**3-5 minutes** before the first ANE request under `ane_startup="wait"`, almost all of
it Core ML's on-device ANE compile. `ane_startup="background"` works around this by
serving on MLX for that window (the model serves on MLX within 1.4 s, and the ANE joins
once its compile finishes), but the underlying 3-5 minute compile cost itself is
unaddressed. A restart after a location move is cheaper (1.9-2.7 s with `wait`) because
Core ML's own on-device compile cache is reused, but any new location, including a fresh
CI runner or a fresh user machine, pays the full cost again.

## Why it matters

A 3-5 minute wall-clock cost on first use (even hidden behind `background` mode) shapes
real deployment: cold containers, CI matrices building artifacts from scratch, or a
first-time user following the README quickstart with the `[ane]` extra all pay it. The
`background` mode is a mitigation, not a reduction of the underlying cost.

## Expected output

An investigation into whether the artifact build/import path (`laya_apple/lifecycle.py`,
`laya_apple/conversion/build.py`) can pre-warm or ship Core ML's on-device compile
artifacts, or otherwise reduce first-load latency, without weakening the "no silent
fallback" guarantees (the placement probe and compute-plan check still have to run and
still have to fail loudly if the compile placement changes — `docs/no-silent-fallback.md`
rows 9-10). A negative result (confirming this can't be safely reduced further, e.g.
because Core ML's on-device compile cache cannot be exported/imported reproducibly) is
also a valid and useful outcome, written up with evidence.

## How to validate

- Reproduce the `benchmarks/v0.3.md` cold-start measurement methodology (fresh artifact
  location, `ane_startup="wait"`, time to first successful ANE prediction) as a baseline.
- Any proposed change must still pass `laya-apple artifacts warm` and the placement
  probe/compute-plan checks unchanged; do not weaken `PROBE_MAX_RATIO` or the compute
  plan requirement to make cold start look faster.
- Benchmarks follow `docs/benchmarks.md`: raw per-request data, reproducible method.

## Relevant files

- `benchmarks/v0.3.md` ("Cold start" section, exact numbers cited above)
- `laya_apple/lifecycle.py`
- `laya_apple/conversion/build.py`
- `docs/no-silent-fallback.md` (rows 9, 10, 16 — compute plan, placement probe, background start-up failure)

## Difficulty / scope

Needs the `ane`/`convert` extras and willingness to do careful, repeatable timing
measurements. This is a real engineering investigation, not a quick fix; a well-evidenced
"no safe further reduction found" writeup is an acceptable outcome.
