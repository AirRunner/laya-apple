---
title: "Research: P99 isolation under GPU+ANE concurrency"
labels: ["research", "scheduler", "performance"]
---

## Problem

`benchmarks/v0.2.md`'s Part A (closed-loop exit-gate mix) shows that even in the
heterogeneous (`auto`) configuration, each stream's tail latency is measurably above its
solo value: for example, laya-typed-decisions (thread placement) short-stream P99 rises
+8% and long-stream P99 rises +13% versus running that stream alone, and
laya-multilingual (thread placement) short-stream P99 rises +153%. The README's own
"Limitations" section states this plainly ("Isolation is partial"), and
`docs/guide.md` notes that heavy Python work on the calling thread slows a thread-placed
ANE stream (GIL).

## Why it matters

The project's central heterogeneous-serving claim (2.9-4.6× throughput multiplier) is
about aggregate throughput; the interference cost this issue is about is a tail-latency
side effect that a latency-sensitive caller needs to know about and that isn't yet
explained mechanistically — is it GIL contention (thread placement), OS scheduling,
shared memory bandwidth, or something else, and does it differ between thread- and
process-placed ANE (`benchmarks/v0.2.md`'s two placement rows per model)?

## Expected output

A research writeup investigating the source(s) of P99 interference between the GPU and
ANE streams under `auto`/`execution="workers"`, using `benchmarks/v0.2.md`'s existing
methodology (or scripts/concurrency-style tooling) as the base, and specifically
comparing thread-placement vs. process-placement isolation to explain why they differ so
much for laya-multilingual (+153% thread vs. +99% process) but not for
laya-typed-decisions (+8% thread vs. +91% process — the opposite ordering).

## How to validate

- Use `scripts/bench_concurrency.py` / `scripts/concurrency_report.py` or equivalent so
  results are directly comparable to the existing `benchmarks/v0.2.md` numbers cited
  above.
- Report raw per-window data, not just aggregate deltas, matching
  `docs/benchmarks.md`'s conventions.
- If a mitigation is found, it needs its own throughput/latency tradeoff evidence before
  being proposed as a placement or scheduling change — this issue is scoped to
  understanding the cause.

## Relevant files

- `benchmarks/v0.2.md` (Part A table, the exact deltas cited above)
- `laya_apple/executor.py`
- `laya_apple/scheduling.py`
- `README.md` ("Limitations" — "Isolation is partial")

## Difficulty / scope

Research. Needs ANE hardware with built artifacts and comfort with concurrency
profiling (threads vs. processes, GIL effects). Builds on an existing, well-documented
measurement rather than starting from nothing.
