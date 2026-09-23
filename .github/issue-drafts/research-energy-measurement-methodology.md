---
title: "Research: energy measurement methodology"
labels: ["research", "performance"]
---

## Problem

The README lists energy use as explicitly "not yet measured," alongside quantization and
cross-SoC validation. Every existing benchmark (`benchmarks/v1.0.md` and predecessors)
measures latency and throughput only; there is no methodology anywhere in the repo for
measuring GPU/ANE power draw during a `predict()` call, and one of the project's own
headline claims — that heterogeneous GPU+ANE serving gets 2.9-4.6× the throughput of
GPU-only serving (README, "Heterogeneous GPU + ANE serving") — says nothing about
whether that throughput gain costs more, less, or the same energy per request.

## Why it matters

The ANE exists specifically as a low-power inference engine; a throughput comparison
between MLX-GPU-only and GPU+ANE that ignores energy is an incomplete picture of the
actual tradeoff laya-apple is making. This matters most for on-device / battery-powered
deployment scenarios that the README doesn't currently address at all.

## Expected output

A methodology writeup under `research/` proposing how to measure per-request energy on
Apple Silicon (e.g. via `powermetrics` or another instrumentable source), validated
against a known workload, with raw per-request or per-window power samples — not just a
single top-line number — matching the raw-data discipline the rest of the project's
benchmarks follow (`docs/benchmarks.md`). Apply it to at least a GPU-only vs. GPU+ANE
comparison on one model to produce a first real energy figure comparable to the existing
throughput tables.

## How to validate

- The methodology should be reproducible by someone else on the same machine (state
  exact tool invocations, sampling rate, and how idle/background power is subtracted).
- Follow `docs/benchmarks.md`'s existing conventions for what a new benchmark commits:
  raw data plus the exact reproduction steps.
- Cross-check at least one measurement against a sanity baseline (e.g. idle power draw)
  to catch methodology errors before trusting comparative numbers.

## Relevant files

- `README.md` ("Not yet measured")
- `docs/benchmarks.md` (existing benchmark conventions)
- `benchmarks/v1.0.md` (existing throughput tables this would sit alongside)

## Difficulty / scope

Research, methodology-first. Needs Apple Silicon hardware (ANE hardware for the
GPU+ANE side of any comparison) but no ANE artifact building beyond what's already
shipped. The hard part is designing a trustworthy measurement, not running it.
