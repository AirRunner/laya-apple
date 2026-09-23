---
title: "Research: long-context ANE performance"
labels: ["research", "ane", "performance"]
---

## Problem

`research/phase-0-feasibility/ane-long-context.md` already establishes *why* ANE
latency grows superlinearly with sequence length for `laya-typed-decisions`: attention
score work (QKᵀ, mask, softmax, A·V) is quadratic in L and runs at roughly 7× lower
effective throughput on the ANE than the projection/MLP matmuls, and the per-doubling
growth factor rises toward 4 (full-body P50: 8.07ms @ L64 → 169.57ms @ L1024). One
contributing factor is identified and partially fixed in research: 18 of 28 encoder
layers are sliding-window (±64 tokens) but the exported graph computes dense L×L scores
regardless; an exact windowed rewrite cut full-body latency 13% at L512 and 25% at
L1024 in that experiment, but it was never carried through to a shipped, parity-passing
artifact (this repo's shipped ANE buckets top out at L128-L256 per `docs/support-matrix.md`
and the README's supported-models table — longer ANE lengths are simply not offered).

## Why it matters

If the windowed-attention rewrite (or another approach to the quadratic score cost) can
be carried through the full artifact release gate — 100% ANE compute plan, 0 transitions,
the placement probe, and 0 hard mismatches at FP16 tolerance — it could extend validated
ANE coverage to longer sequences than are offered today, where currently `auto` always
routes to MLX (README, "Auto routing": "Longer contexts favour MLX").

## Expected output

A research writeup under `research/` (following the existing `phase-0-feasibility/`
convention: raw data, scripts, a dated report) extending
`research/phase-0-feasibility/ane-long-context.md`'s windowed-attention experiment into
a candidate artifact, run through the full parity gate at one or more longer buckets
(e.g. L256/L512). A negative result — parity fails, or the ANE remains slower than MLX
even after the fix — is a complete and valid outcome and must be recorded, not hidden
(`CONTRIBUTING.md`: "Failed variants stay research").

## How to validate

- Any candidate must pass the exact gate in `CONTRIBUTING.md`'s "Artifact release
  policy" before it could ever be proposed for shipping — this issue is scoped to the
  research and measurement, not to shipping a new bucket.
- Cite real numbers from the new experiment the way `ane-long-context.md` does (P50 in
  ms per length, compute plan percentage, device transitions), not summary claims.

## Relevant files

- `research/phase-0-feasibility/ane-long-context.md` (existing experiment and numbers)
- `research/phase-0-feasibility/methodology.md`
- `laya_apple/conversion/bc1s.py`
- `docs/support-matrix.md` (currently offered ANE lengths per model)
- `CONTRIBUTING.md` ("Artifact release policy")

## Difficulty / scope

Research. Needs ANE hardware, the `ane`/`convert` extras, and comfort profiling Core ML
compute plans. Open-ended; a rigorous negative result is a success criterion, not a
failure to close the issue.
