# Phase -1: feasibility validation

These experiments test the technical assumptions in `DEVELOPMENT_PLAN.md` before any
runtime code is written. Everything here is research code. None of it is imported by the
`laya_apple` package.

**Start with [conclusion.md](conclusion.md).**

## Documents

| Document | Question |
|---|---|
| [environment.md](environment.md) | Hardware, OS, toolchain, pinned packages, repositories, checkpoint revisions |
| [methodology.md](methodology.md) | Inputs, timing boundaries, sampling, parity tolerances (fixed before measurement), device verification |
| [parity.md](parity.md) | Do the three checkpoints match upstream on each backend, compute unit and length? (Experiment 1) |
| [typed-decisions-ane.md](typed-decisions-ane.md) | Is `laya-typed-decisions` viable on the Neural Engine? (Experiment 2) |
| [ane-long-context.md](ane-long-context.md) | Why does ANE latency grow with sequence length? (Experiment 3) |
| [gpu-ane-crossover.md](gpu-ane-crossover.md) | Where does the GPU overtake the ANE, and how do question count and batching change that? (Experiment 4) |
| [concurrency.md](concurrency.md) | Can GPU and ANE serve Laya requests concurrently? (Experiment 5) |
| [conclusion.md](conclusion.md) | Answers, and the architecture decision |

## Headline results (M4 Max, macOS 26.6.2)

- **Correctness.**
  - MLX passes parity on all three checkpoints.
  - On the ANE, only a channel-first (BC1S) rewrite of the graph is correct.
  - laya-coreml's ordinary graph is numerically wrong on the ANE. At L1024 it silently
    falls back to a 2.2 s path.
- **typed-decisions on ANE:** viable up to L1024. It runs at 9.9 ms at L128, fully on
  the ANE, confirmed by compute plan, Instruments and latency controls.
- **Long context:** the quadratic attention block runs ~7× less efficiently on the ANE
  than the matmuls. Local layers also do dense L² work. An exact windowed rewrite gains
  25% at L1024, which still leaves the ANE 1.8× slower than MLX.
- **Crossover:** with one question, the ANE wins up to L128 (ModernBERT-large) or L256
  (mmBERT). With several questions, MLX batching wins almost everywhere.
- **Concurrency:** ANE (short) and GPU (long) at the same time lose nothing, giving
  3.1–3.3× the throughput of the best single device.

## Reproducing

```bash
cd research/phase-0-feasibility
uv sync                                                 # pinned environment (pyproject.toml + uv.lock)
uv run python scripts/reference.py --model laya-typed-decisions      # upstream goldens
uv run python scripts/convert_coreml.py --model laya-typed-decisions --fixed 128
uv run python scripts/convert_ane.py   --model laya-typed-decisions --length 128
./scripts/run_parity.sh                                 # untimed correctness matrix
./scripts/run_timed.sh                                  # every timed experiment, serially
./scripts/run_followup.sh                               # cold start, crossover refinement, windowed parity
uv run python scripts/report.py parity|latency|crossover
```

Converted packages go to `$LAYA_APPLE_ARTIFACTS`, by default
`<research-artifacts>`. They are reproducible and never
committed. Run timed scripts on a quiet machine, one at a time.

## Layout

```text
scripts/    experiment code (provenance headers on adapted code)
raw/        machine-readable results: JSON / JSONL with every sample
  reference/     upstream PyTorch CPU FP32 goldens
  parity/        one file per model × configuration
  bench/         latency.jsonl (all latency cells, raw samples)
  profile/       component probes, compute plans, host-prep timing
  concurrency/   per-request latencies per window
  trace/         allow-listed Instruments summaries
  coldstart/     fresh-process load measurements
  layernorm/     rejected-hypothesis evidence (CPU_ONLY investigation)
  logs/          run logs
reports/    tables generated from raw/ by scripts/report.py
```

## Provenance

| Code | Source | License |
|---|---|---|
| `scripts/ane_model.py` (BC1S graph) | `mizorewww/laya-coreml@4619e04` `experiments/ane_engineering/model.py`; extended for ModernBERT-large, windowed attention and probes | Apache-2.0 |
| `ANEBackend` host runtime in `scripts/backends.py` | `mizorewww/laya-coreml@4619e04` `laya_coreml/ane.py` | Apache-2.0 |
| `scripts/device_plan.py` | `mizorewww/laya-coreml@4619e04` `benchmarks/common.py::compute_plan` | Apache-2.0 |
| MLX backend | `mizorewww/laya-mlx@0a85951`, installed unmodified | Apache-2.0 |
| Ordinary Core ML export | `mizorewww/laya-coreml@4619e04`, installed unmodified | Apache-2.0 |
| Reference model / prompt / calibration | `NandhaKishorM/laya@573e5b6`, installed unmodified | Apache-2.0 |

Fixture and workload text was written for this project, and no third-party fixture text
is committed. The goldens in `raw/reference/` are logits produced here by running the
pinned upstream checkpoints on those fixtures. Instruments traces are reduced to allow-listed counts and
durations before being saved.
