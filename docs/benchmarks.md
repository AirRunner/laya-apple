# Reproducing the benchmarks

Every number in [`benchmarks/v0.1.md`](../benchmarks/v0.1.md) and
[`benchmarks/v0.2.md`](../benchmarks/v0.2.md) comes from a script under `scripts/` run
against the pinned checkpoints and locally built ANE artifacts, on the tested profile in
[`compatibility.md`](compatibility.md). This page is how to reproduce them, or run the
same methodology on your own machine.

## Setup

```bash
git clone https://github.com/tc3oliver/laya-apple && cd laya-apple
uv sync --extra ane --extra convert --extra dev
```

- `ane` — coremltools, to run the ANE backend at all.
- `convert` — torch, required only to build ANE artifacts (`artifacts build`); the
  conversion script imports `torch` directly.
- `dev` — pytest, needed by `release_gate.py`'s test steps.

Environment, for every command below:

```bash
export LAYA_APPLE_CACHE=/path/to/cache   # or accept the default (~/.cache/laya-apple)
export HF_HUB_OFFLINE=1                  # once checkpoints and artifacts are cached
```

Prerequisites before any benchmark script:

1. Download the checkpoints once (this needs network, so leave `HF_HUB_OFFLINE` unset for
   this step): `laya-apple download laya laya-multilingual laya-typed-decisions`.
2. Build every offered ANE bucket for each model, on the machine you are benchmarking —
   artifacts are never downloaded or committed:
   ```bash
   laya-apple artifacts build laya
   laya-apple artifacts build laya-multilingual
   laya-apple artifacts build laya-typed-decisions
   ```
   Each bucket only becomes usable after it passes the layout, placement and parity
   gates on this machine (README, "The ANE path: building artifacts").

## v0.1 — single-request latency

Driver: [`scripts/release_bench.py`](../scripts/release_bench.py).

```bash
uv run python scripts/release_bench.py benchmarks/v0.1/raw.jsonl
```

- Runs 50 configurations: every (model, device, exact length, question count) combination
  from the script's `LENGTHS` table, including `gpu`/`auto` at every length, `ane` at
  every offered bucket, and a 4-question case at L128.
- Each configuration runs in its own fresh `laya-apple benchmark` process (via
  `.venv/bin/laya-apple --offline benchmark`), so no state or cache warmth carries over
  between configurations.
- Two passes over the full plan, the second in reversed order, so an order effect (e.g.
  thermal drift) shows up as a pass-to-pass difference rather than being silently
  averaged away.
- Requests are **exact-length**: the longest prompt row is exactly L tokens, built by
  `laya_apple.workload` — never padded or truncated to fake a length.
- Per configuration: 10 warm-up iterations, then 50 timed iterations, at two measurement
  boundaries:
  - **forward** — the backend call only, synchronised (excludes prompt build, routing,
    answer formatting);
  - **predict** — end to end: prompt build, routing, forward, answer formatting.
- Raw output: one JSON record per configuration per pass, with every timed sample kept,
  appended as JSON Lines to the path you give it (`benchmarks/v0.1/raw.jsonl`).

Regenerate the markdown tables from the raw data with
[`scripts/bench_report.py`](../scripts/bench_report.py):

```bash
uv run python scripts/bench_report.py benchmarks/v0.1/raw.jsonl
```

It groups records by (model, device, length, question count), averages the two passes'
forward/predict P50s, reports the larger of the two passes' P99s, and reports the
pass-to-pass P50 difference as an order/noise indicator. This is the exact table format
in `benchmarks/v0.1.md`.

## v0.2 — heterogeneous GPU + ANE execution

Driver: [`scripts/bench_concurrency.py`](../scripts/bench_concurrency.py), run once per
model:

```bash
uv run python scripts/bench_concurrency.py --model laya-typed-decisions \
    --short 128 --long 1024 --output benchmarks/v0.2/concurrency-typed.json
```

This exercises `Laya(execution="workers")` through the product API, not the research
harness. Two parts, both measured through real requests to a running `Laya` instance:

- **Part A (closed-loop exit gate):** the Phase -1 short/long stream mix (one short and
  one long stream, one question each, one client thread per stream, all through one
  `Laya` instance), under four conditions — `solo_short`, `solo_long`, `hetero`
  (`device="auto"`) and `gpu_only` (`device="gpu"`, both streams share the one MLX
  queue). 20 s windows, 3 cycles, alternating order; the report takes medians over
  cycles. Run once with the ANE placed on a thread in the caller's process and once as
  its own worker process — the GPU is always a worker process.
- **Part B (open-loop mixed workload):** Poisson arrivals, plus a bursty variant (1 s at
  3× the mean rate, then 2 s idle), over a request mix of 60% short 1-question, 20%
  medium 1-question, 10% long 1-question, 10% short 4-question. The same arrival sequence
  is replayed for `auto` and for `gpu_only`. Latency is measured from arrival to result,
  so queueing time is included.

Every request's answer is checked against the inline-mode result for the same request
and device — an answer mismatch under concurrent load is a hard failure, not noise.

Placement choice (which model uses `ane_placement="thread"` vs `"process"`) comes from
[`scripts/derive_placement.py`](../scripts/derive_placement.py), which picks whichever
placement gave the lower hetero short-stream P99 in Part A and writes
`laya_apple/data/placement.json`.

Regenerate the markdown tables with
[`scripts/concurrency_report.py`](../scripts/concurrency_report.py):

```bash
uv run python scripts/concurrency_report.py
```

It reads `laya_apple/data/placement.json` and the raw Part A/B JSON files under
`benchmarks/v0.2/` (`placement-<model>-{thread,process}.json`,
`concurrency-<model>.json`) and prints the same tables as `benchmarks/v0.2.md`.

## Forthcoming: v0.3 and v1.0

- **v0.3** adds cold-start measurement
  ([`scripts/bench_coldstart.py`](../scripts/bench_coldstart.py)): each run copies a
  model's registered artifacts into a fresh cache root so Core ML's on-device ANE
  compile is genuinely cold, then measures `ready_s`, `first_request_s`, `ane_ready_s`,
  and a subsequent `warm_ready_s` at the now-warm location, in one process:
  `.venv/bin/python scripts/bench_coldstart.py [MODEL ...] --out benchmarks/v0.3/coldstart.json`.
- **v1.0** adds the runtime placement probe evidence
  ([`scripts/bench_probe.py`](../scripts/bench_probe.py)): for every registered ANE
  artifact, it times the fastest of several Core ML predictions on `CPU_AND_NE` (the ANE
  case) against `CPU_ONLY` (the failure the runtime's probe must catch), compared against
  the bucket's measured ANE service time from `routing.json`:
  `.venv/bin/python scripts/bench_probe.py --runs 5 --out benchmarks/v1.0/probe.json`.
- The full release gate — clean-install matrix (Python 3.11–3.13, base and `ane` extras)
  plus the test suite — runs via
  [`scripts/release_gate.py`](../scripts/release_gate.py):
  `uv run python scripts/release_gate.py [--quick] --out benchmarks/release-gate-X.json`.
  Every step runs even after an earlier one fails; the process exits non-zero if any
  required step failed. `--quick` skips the install matrix and slow pytest markers.
- Both scripts take `LAYA_APPLE_CACHE`/`HF_HUB_OFFLINE` as above.
  `benchmarks/v0.3.md` and `benchmarks/v1.0.md` will follow the same structure as
  `v0.1.md`/`v0.2.md`: setup table, driver command, tables generated from raw JSON/JSONL,
  and a comparison against the prior release's expectations.

## What makes results comparable

- **Same profile.** Every table states the exact hardware, macOS build, MLX,
  coremltools and Python versions it was measured on (see `compatibility.md`). Numbers
  from a different profile are not comparable and are not merged into the same table
  (`DEVELOPMENT_PLAN.md` §5, §11: compute-unit and profile configurations are never
  merged).
- **Quiet machine.** Benchmarks assume AC power and an otherwise idle machine. The v0.1
  report records "AC power, no thermal or performance warnings recorded" for its run.
- **Thermal state recorded.** The stress suite
  (`tests/stress/test_sustained_mixed_load.py`) samples `pmset -g therm` during a
  sustained run and stores it alongside the latency data, the same practice the research
  benchmarks used. A run reporting thermal or performance warnings should be treated as
  not comparable to one that reports none.
- **Fresh process per configuration** (v0.1) or **one long-lived instance per run**
  (v0.2), matching what each script actually does — mixing the two within one table
  would conflate process startup cost with steady-state latency.

## Expected variance

`benchmarks/v0.1.md` reports pass-to-pass differences (same machine, same script, second
pass in reversed order) of the forward P50 within ±2.4% for every configuration except
one (`laya-multilingual` ANE L64, −3.7%, which is 0.13 ms on a 3.4 ms baseline — small in
absolute terms). Against the Phase -1 feasibility measurements taken as the original
baseline, the report states: "All nine values are within ±0.5% of Phase -1" for the
model-only P50s it compares (`laya`, `laya-multilingual`, `laya-typed-decisions`, each at
MLX L128, MLX at max length, and ANE L128). Treat a fresh measurement on the same,
undisturbed profile as reproducible to within roughly that range; a larger deviation
means the machine, build, or dependency versions differ from what the report used, not
that the methodology is noisy.
