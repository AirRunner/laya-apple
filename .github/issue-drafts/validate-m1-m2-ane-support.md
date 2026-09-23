---
title: "Validate M1/M2 ANE support (build + parity + calibrate)"
labels: ["help wanted", "ane", "hardware", "correctness"]
---

## Problem

The Apple Neural Engine artifacts and the parity gate that validates them
(`laya_apple/conversion/`, `laya_apple/parity/`) have only ever been run on an M4 Max
(`docs/support-matrix.md`). Whether the BC1S channel-first graph rewrite that makes the
M4 Max's ANE pass parity (README, "Correctness first") also passes on the earlier ANE
generation in M1/M2 is unknown.

## Why it matters

`CONTRIBUTING.md`'s artifact release policy is explicit: an ANE artifact configuration
ships only after it passes the same correctness gate (100% ANE compute plan, 0 device
transitions, the runtime placement probe, and the full parity gate) *on the profile it
runs on*. That gate has to be re-run per machine generation — an M4 Max pass says
nothing about M1/M2.

## Expected output

A PR report (does not need to add ANE support to the shipped profiles) showing, for at
least `laya-typed-decisions`: whether `laya-apple artifacts build` produces a 100%-ANE
compute plan on M1 or M2, whether the parity gate passes at FP16 tolerances (prob error
≤ 0.02, 0 hard mismatches — `laya_apple/parity/__init__.py`), and the output of
`laya-apple calibrate` on that machine. If it fails, that is a useful, complete result:
document where and how (compute plan falls back off-ANE, parity fails, or the
`CPU_AND_NE` runtime probe fails).

## How to validate

1. `uv sync --extra dev --extra ane --extra convert` on M1 or M2 hardware.
2. `laya-apple artifacts build laya-typed-decisions` and inspect the compute plan / parity report it produces.
3. `laya-apple parity laya-typed-decisions --device ane` against the shipped goldens.
4. `laya-apple calibrate laya-typed-decisions` to see whether `auto` routing gets a usable local profile.
5. Report results (pass or documented failure) — do not merge a "validated" artifact without it actually clearing every check in `CONTRIBUTING.md`'s artifact release policy.

## Relevant files

- `docs/support-matrix.md`
- `docs/no-silent-fallback.md` (rows 9–10, compute plan + placement probe)
- `laya_apple/conversion/bc1s.py`, `laya_apple/conversion/build.py`
- `laya_apple/parity/__init__.py`, `laya_apple/parity/ane.py`
- `CONTRIBUTING.md` ("Artifact release policy")

## Difficulty / scope

Needs M1 or M2 Apple Silicon hardware and the `ane`/`convert` extras (`coremltools==9.0`,
`torch==2.7.0`). Investigative — the result may be "it doesn't pass here," which is
still valuable and expected to be documented, not silently dropped.
