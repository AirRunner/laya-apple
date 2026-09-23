---
title: "Research: ModernBERT local-attention graph optimization"
labels: ["research", "ane", "performance"]
---

## Problem

`research/phase-0-feasibility/ane-long-context.md` (H2) shows the exported ANE graph for
ModernBERT-large computes dense, full-sequence attention scores for all 28 encoder
layers even though 18 of them are local/sliding-window (±64 tokens) in the original
architecture — the exported graph masks with a −1e4 additive mask instead of restricting
the computation. An exact block-local rewrite (`ConvAttention._windowed` in that
experiment's scripts) measured a 13% full-body latency cut at L512 and 25% at L1024, but
this is recorded as a research probe, not a shipped, parity-validated graph change.

## Why it matters

This is a graph-level optimization independent of sequence-length coverage (unlike the
long-context research issue, which is about extending offered buckets) — it could
reduce latency at buckets already shipped today (L64-L256, per `docs/support-matrix.md`),
if it clears the parity gate.

## Expected output

A research writeup carrying `ConvAttention._windowed` (or an equivalent windowed
rewrite) through the full parity gate at the buckets laya-apple already ships
(`docs/support-matrix.md`'s per-model `ane_buckets`), with before/after latency,
compute-plan percentage, and parity numbers (hard mismatches, near-tie flips, max
probability error) reported the way `README.md`'s correctness tables already do. A
result that fails parity, or that passes but shows no compute-plan or latency
improvement over the current shipped graph, is a complete and useful outcome and must be
recorded under `research/`, not silently dropped (`CONTRIBUTING.md`: "Failed variants
stay research").

## How to validate

- Must pass the same gate as any other artifact change per `CONTRIBUTING.md`'s
  "Artifact release policy" before ever being proposed as a shipping change — this issue
  is the research/measurement step.
- Compare against the existing shipped BC1S graph's numbers in `benchmarks/v1.0.md` and
  `research/phase-0-feasibility/ane-long-context.md`, not against a new, uncontrolled
  baseline.

## Relevant files

- `research/phase-0-feasibility/ane-long-context.md` (H2, the existing windowed probe)
- `laya_apple/conversion/bc1s.py`
- `laya_apple/models/modernbert_mlx.py` (vendored reference architecture — read-only, see `NOTICE`)
- `docs/support-matrix.md`
- `CONTRIBUTING.md` ("Artifact release policy")

## Difficulty / scope

Research. Needs ANE hardware and the `ane`/`convert` extras. Builds directly on an
existing, partially-complete experiment rather than starting from nothing.
