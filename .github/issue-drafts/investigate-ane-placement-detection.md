---
title: "Investigate ANE placement detection beyond the load-time probe"
labels: ["help wanted", "ane", "coreml", "correctness"]
---

## Problem

`docs/no-silent-fallback.md`'s "Runtime placement probe" section states the known
limitation directly: the probe runs at load time only. A placement change after a model
has loaded and passed — for example, if the system evicts the ANE compile cache mid-run
and Core ML silently falls back to `CPU_AND_NE`'s CPU path — is not observed. No such
change was seen during the 600 s soak test on the tested profile, but that is an absence
of evidence on one run, one machine, one duration, not a guarantee.

## Why it matters

"No silent fallback" is a Non-negotiable rule in `CONTRIBUTING.md`. The current
mitigation (a load-time probe comparing the loaded model's speed against a `CPU_ONLY`
instance, threshold `PROBE_MAX_RATIO = 0.8`) is well-evidenced for load time
(ratio 0.32-0.50 for every shipped artifact vs. ~1.0 for a CPU-placed model,
`benchmarks/v1.0/probe.json`) but explicitly does not cover placement changing after
load.

## Expected output

A written investigation (design doc or research writeup under `research/`, following
the project's convention of recording negative/failed results too) into whether a
periodic or per-request lightweight re-probe is feasible without unacceptable latency
overhead, and if not, why not — with evidence, not assumption. If a viable approach is
found, an implementation proposal (not necessarily the implementation itself) that keeps
the explicit-`ane`-raises / `auto`-drops-and-warns behavior from `docs/no-silent-fallback.md`
rows 9-10.

## How to validate

- Reproduce the existing probe's cost measurement (`scripts/bench_probe.py`,
  `benchmarks/v1.0/probe.json`) as a baseline before proposing any additional overhead.
- Any proposed periodic re-probe must not silently continue serving from a
  CPU-placed model; it must raise (explicit `ane`) or drop-and-warn (`auto`), matching
  the existing rows 9-10 behavior, with a test that pins it.
- Try to actually trigger a placement change (e.g. by evicting the ANE compile cache
  mid-run) to get real evidence rather than reasoning from the static compute plan alone.

## Relevant files

- `docs/no-silent-fallback.md` ("Runtime placement probe" section, especially "Limitation")
- `scripts/bench_probe.py`
- `laya_apple/artifacts.py` (compute plan / probe implementation)
- `CONTRIBUTING.md` ("No silent fallback")

## Difficulty / scope

Needs ANE hardware and the `ane` extra. This is research-adjacent: a well-evidenced
"not feasible within acceptable overhead" conclusion is a legitimate and useful result.
