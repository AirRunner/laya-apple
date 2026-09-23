---
title: "Research: cross-SoC routing calibration"
labels: ["research", "scheduler", "hardware"]
---

## Problem

`laya_apple/profiles.py` and `laya-apple calibrate` exist to let a machine other than
the tested M4 Max get its own locally-derived routing profile (measure MLX/ANE latency
locally, then apply the same rule as `laya_apple/derivation.py` uses for the shipped
table). This mechanism has shipped but, per `docs/support-matrix.md`, has not been
exercised and validated end-to-end on a second SoC — there is no evidence yet that
`calibrate`'s output on, say, an M2 or M3 produces sensible routing thresholds, or that
the "walk stops at the first bucket that fails" rule (README, "Auto routing") behaves
correctly when the underlying MLX/ANE crossover point is different from the M4 Max's.

## Why it matters

`README.md` is explicit that "Measured crossover ≠ production routing threshold" and
that the crossover point itself is unlikely to be portable across SoCs — different core
counts and memory bandwidth almost certainly shift where MLX beats the ANE. Without a
second-SoC calibration run, the whole `calibrate` mechanism is untested outside its
design, not just unshipped for other hardware.

## Expected output

A research writeup running `laya-apple calibrate` on at least one non-M4-Max SoC,
comparing the resulting local profile's routing thresholds and offered `auto` buckets
against the shipped M4 Max table (`laya_apple/data/routing.json`), and sanity-checking
the result against directly-measured MLX/ANE latency at a few lengths on that machine
(i.e. does the derived profile's crossover point match what raw measurement shows).
Include whatever the `derivation.py` rule does when applied to that machine's numbers,
and flag anything that looks wrong (e.g. a bucket included that direct measurement shows
shouldn't be, or vice versa).

## How to validate

- Compare `calibrate`'s derived thresholds against raw, independently-measured MLX and
  ANE latency at the same lengths, not just against the algorithm's own output.
- Reuse `laya_apple/derivation.py`'s existing rule rather than writing a new one — the
  question is whether the rule generalizes, not whether a different rule would do
  better (that would be a separate, follow-on issue).
- Record raw per-length timing data, per `docs/benchmarks.md`'s conventions.

## Relevant files

- `laya_apple/profiles.py`
- `laya_apple/derivation.py`
- `laya_apple/data/routing.json` (shipped M4 Max table, for comparison)
- `README.md` ("Auto routing", "Measured crossover ≠ production routing threshold")
- `docs/support-matrix.md`

## Difficulty / scope

Research. Needs a second Apple SoC and, for full coverage, built ANE artifacts on it
(`ane`/`convert` extras). Builds directly on shipped, working tooling (`calibrate`)
rather than needing new code.
