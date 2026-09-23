# laya-apple

[![CI](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml/badge.svg)](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml)

**A correctness-validated heterogeneous Apple silicon runtime for
[Laya](https://github.com/NandhaKishorM/laya).** Laya models take a context and a set of
typed questions (`choice`, `score`, `noul`) and return calibrated decisions in a single
forward pass; there is no text generation.

laya-apple does three things:
- it runs Laya on the **MLX GPU** for every workload;
- it adds the **Apple Neural Engine** only where an exact configuration has been proven
  to give the same decisions as upstream Laya;
- it routes each request to the right engine and serves both engines **at the same
  time**.

## What makes laya-apple different

- **MLX GPU is the general backend.** Every model, length, question count and batch
  runs on MLX.
- **The ANE is used only for validated fixed-shape paths.** A Core ML artifact is
  registered only after it passes a parity gate against upstream PyTorch FP32 on the
  machine that uses it. It must also show a 100% Neural Engine compute plan and pass a
  runtime placement probe.
- **`device="auto"` picks the validated backend per request, from measurements.** Short
  single-question requests go to the ANE. Long contexts and multi-question requests go
  to MLX. Every result records why.
- **GPU and ANE serve requests concurrently.** One `Laya` instance keeps both engines
  busy. On the tested machine that gives 2.9–4.6× the throughput of GPU-only serving on
  a short/long mix, with no answer changes.
- **No silent Core ML fallback is accepted.** An explicit ANE request runs the validated
  artifact or raises. Core ML is never allowed to move work to the CPU unnoticed.

## Quickstart

```python
from laya_apple import Laya

model = Laya.from_pretrained(
    "convaiinnovations/laya-typed-decisions",
    device="auto",
)

result = model.predict(
    context="The customer was charged twice for the same invoice and is frustrated.",
    questions={
        "urgency": {
            "type": "choice",
            "instructions": "How urgent is this?",
            "criteria": ["low", "medium", "high"],
        }
    },
)
print(result.answers["urgency"]["choice"], result.answers["urgency"]["probabilities"])

rt = result.runtime
print(rt.backend, rt.device, rt.routing_reason, f"{rt.latency_ms:.1f} ms")
```

Output on the tested machine, with the `[ane]` extra and validated artifacts:

```text
high {'low': 0.1713, 'medium': 0.3358, 'high': 0.4929}
coreml ane validated_short_single_question_path 11.2 ms
```

- **Without ANE artifacts** (or without the `[ane]` extra), the same request runs on
  MLX, and `routing_reason` says why.
- **With `device="gpu"`,** the answer is the same decision (`high`), with probabilities
  within 0.002. `latency_ms` varies from run to run.
- **`result.runtime` is a `RuntimeInfo`.** It carries:
  - `backend` (`mlx` / `coreml`), `device` (`gpu` / `ane`) and `routing_reason`;
  - `latency_ms`, `sequence_length`, `question_count` and `dtype`;
  - the artifact revision;
  - under `execution="workers"`, `queue_wait_ms` and `device_ms`.

The first call downloads the pinned checkpoint from Hugging Face. After that,
`local_files_only=True` (or `HF_HUB_OFFLINE=1`) runs fully offline. The full user guide
is [`docs/guide.md`](docs/guide.md).

## Install

laya-apple is not published on PyPI. Install from a clone (Python 3.11–3.13, Apple
silicon):

```bash
git clone https://github.com/tc3oliver/laya-apple && cd laya-apple
uv sync                                # MLX backend only
uv sync --extra ane                    # + Neural Engine backend (coremltools 9.0)
uv sync --extra ane --extra convert    # + building ANE artifacts on this machine
laya-apple artifacts build laya-typed-decisions   # build + parity-validate the ANE buckets here
```

## Supported models

| Model | Encoder | max_len | MLX (GPU) | ANE buckets, explicit `device="ane"` | ANE buckets used by `auto` |
|---|---|---:|---|---|---|
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large | 512 | FP16, FP32, any length and batch | 64, 96, 128 | 64, 96, 128 |
| [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | mmBERT-base | 1024 | FP16, FP32, any length and batch | 64, 96, 128, 256 | 64, 96, 128 |
| [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | ModernBERT-large | 1024 | FP16, FP32, any length and batch | 64, 96, 128 | 64, 96, 128 |

**Checkpoints and artifacts:**
- Each checkpoint is pinned to a revision and a weight hash.
- ANE artifacts are fixed-shape and batch 1, with FP16 on `CPU_AND_NE`.
- A request shorter than a bucket is padded to it. A request longer than every validated
  bucket is never truncated and never padded to an unvalidated shape. It raises under
  explicit `device="ane"`, and goes to MLX under `auto`.

Revisions and the per-configuration Core ML status are in
[`docs/support-matrix.md`](docs/support-matrix.md).

## Correctness first

On the tested machine (Apple M4 Max, macOS 26.6.2), the ordinary Core ML export of Laya
is **fast on `CPU_AND_NE`, but it changes decisions**. It shows no error. laya-apple uses a
channel-first (BC1S) rewrite of the graph instead, and registers an ANE artifact only
after it passes the parity gate.

**Definitions** (fixed before measurement,
[methodology](research/phase-0-feasibility/methodology.md)):
- **Probability error:** the largest |Δ| between a backend's calibrated option
  probabilities and those of upstream Laya on PyTorch CPU FP32, over the golden rows.
- **Hard mismatch:** the chosen option differs from upstream, *and* upstream's top-1 /
  top-2 probability margin is at least 2× the tolerance (0.04 for FP16).
- **Near-tie flip:** the chosen option differs inside that band. Such a flip is listed
  row by row, never hidden, and does not fail the gate.
- **FP16 acceptance:** probability error ≤ 0.02, action-probability error ≤ 0.02, **0
  hard mismatches**, all outputs finite, and repeated calls bit-identical. FP32 uses a
  1e-4 tolerance.

v1.0 results against upstream PyTorch FP32 (`benchmarks/v1.0/parity/`):

| Implementation | laya | laya-multilingual | laya-typed-decisions |
|---|---|---|---|
| Core ML ordinary graph · `CPU_AND_NE` | ❌ 12 hard (+5 near-tie), prob err 0.56 | ❌ **85 hard** (+5), prob err 1.0 | ❌ 19 hard (+4), prob err 0.42 |
| Core ML ordinary graph · `CPU_AND_GPU` | ✅ 0 / 0, 0.0066 | ✅ 0 / 0, 0.0059 | ✅ 0 / 0, 0.0028 |
| PyTorch MPS FP32 | ✅ 0 / 0, 3e-6 | ❌ 2 hard (NaN on 2 edge rows) | ✅ 0 / 0, 3e-6 |
| **laya-apple MLX FP16** | ✅ 0 / 0, 0.0037 | ✅ 0 / 0, 0.0045 | ✅ 0 / 0, 0.0017 |
| **laya-apple ANE FP16** (offered buckets) | ✅ 0 hard, **1 near-tie**, 0.012 | ✅ 0 / 0, 0.013 | ✅ 0 / 0, 0.0077 |

Cells show hard mismatches / near-tie flips, then the max probability error. laya-apple
MLX FP32 passes at ≤ 1.1e-5.

- The laya ANE near-tie flip is one golden row whose upstream margin is 0.0035; it shows
  in the L96 and L128 buckets.
- The ordinary Core ML graph on `CPU_AND_NE` also **silently runs on the CPU** at
  laya-typed-decisions L1024: the ANE compiler fails, and the forward pass takes 2.2 s.
  laya-apple checks each artifact's compute plan and times every loaded bucket against
  `CPU_ONLY` ([`docs/no-silent-fallback.md`](docs/no-silent-fallback.md)).

## Single-request performance

**Forward P50 in ms, one question per request.** This is the model call only, not
end-to-end latency. All v1.0 reruns, Apple M4 Max, macOS 26.6.2. ❌ marks configurations
that fail the parity gate above.

**laya-typed-decisions**

| Implementation | L64 | L128 | L256 | L512 | L1024 |
|---|---:|---:|---:|---:|---:|
| PyTorch CPU FP32 | 52.9 | 73.7 | 114.2 | 204.9 | 437.0 |
| PyTorch MPS FP32 | 21.3 | 21.4 | 27.0 | 49.5 | 106.8 |
| Core ML ordinary · `CPU_AND_NE` ❌ | 7.3 | 9.7 | 23.3 | 65.1 | 2201.8 |
| Core ML ordinary · `CPU_AND_GPU` | 8.5 | 12.6 | 20.1 | 36.2 | 71.0 |
| laya-apple MLX FP16 | 9.3 | 12.2 | **19.2** | **35.3** | **71.0** |
| laya-apple ANE | **8.0** | **9.9** | — | — | — |

**laya-multilingual**

| Implementation | L64 | L128 | L256 | L512 | L1024 |
|---|---:|---:|---:|---:|---:|
| PyTorch CPU FP32 | 26.8 | 36.6 | 52.4 | 90.2 | 191.4 |
| PyTorch MPS FP32 ❌ | 16.7 | 17.7 | 17.9 | 23.0 | 48.4 |
| Core ML ordinary · `CPU_AND_NE` ❌ | 3.0 | 4.0 | 10.0 | 30.0 | 90.0 |
| Core ML ordinary · `CPU_AND_GPU` | 4.7 | 6.0 | 9.2 | 15.0 | 29.2 |
| laya-apple MLX FP16 | 5.4 | 6.4 | 8.6 | **14.9** | 29.3 |
| laya-apple ANE | **3.4** | **4.3** | 8.3 | — | — |

**laya** (max_len 512)

| Implementation | L64 | L128 | L256 | L512 |
|---|---:|---:|---:|---:|
| PyTorch CPU FP32 | 52.9 | 73.6 | 114.6 | 204.2 |
| PyTorch MPS FP32 | 21.4 | 21.7 | 27.0 | 49.8 |
| Core ML ordinary · `CPU_AND_NE` ❌ | 7.4 | 9.8 | 23.3 | 65.0 |
| Core ML ordinary · `CPU_AND_GPU` | 8.5 | 12.6 | 20.0 | 36.2 |
| laya-apple MLX FP16 | 9.4 | 12.1 | **19.1** | **35.3** |
| laya-apple ANE | **8.1** | **9.9** | — | — |

**Reading the tables:**
- "—" means the ANE bucket is not offered at that length.
- The ordinary *enumerated-shape* Core ML export runs entirely on the CPU: 217 ms at L128
  on laya.
- **End to end** (`predict`: prompt build, tokenization, routing, inference, calibration,
  formatting) adds 0.08–0.61 ms over the forward pass. For example, laya-typed-decisions
  L128 under `auto` is 9.91 ms forward and 10.06 ms `predict` P50.

All boundaries, P95/P99 figures and the method are in
[`benchmarks/v1.0.md`](benchmarks/v1.0.md).

## Auto routing

What the measurements show:
- **Short single-question requests can favour the ANE.** It wins at L64–L128 for every
  model.
- **Longer contexts favour MLX.** The ANE's attention block runs far less efficiently
  than its matmuls. At L512 on laya-typed-decisions, the BC1S graph measured 54 ms on the
  ANE against 35 ms on MLX (Phase -1).
- **Multi-question requests favour MLX batching.** At L128 with 4 questions, MLX takes
  33.2 ms. The ANE runs questions one after another at batch 1, 39.4 ms on laya in
  Phase -1.

**Measured crossover ≠ production routing threshold.** A bucket joins `auto` only if two
things hold:
- its artifact passed parity;
- its ANE time beats MLX at the *previous*, shorter bucket, because a request just above
  that bucket would otherwise run faster on MLX.

The walk stops at the first bucket that fails. So laya-multilingual L256 stays
explicit-only, even though the ANE is slightly faster at exactly 256 tokens (8.3 ms
against 8.6 ms on MLX): it does not beat MLX at L128 (6.4 ms).

With `execution="workers"`, `auto` also compares queue backlogs. A short request can then
go to MLX when the ANE queue is longer (`ane_backlog_shorter_on_gpu`). Long and
multi-question requests never go to the ANE. How the thresholds were derived is in
[`docs/support-matrix.md`](docs/support-matrix.md). To calibrate another machine, see
[`docs/guide.md`](docs/guide.md).

## Heterogeneous GPU + ANE serving

```python
with Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers") as model:
    futures = [model.submit(context=c, questions=q) for c, q in requests]   # thread-safe
```

- **GPU-only:** every request queues on the GPU.
- **GPU + ANE:**
  - short single-question requests are served by the ANE;
  - long and multi-question requests keep running on the GPU at the same time;
  - short requests no longer wait behind long ones (less head-of-line blocking), and both
    engines do useful work.

**Closed-loop mix, v1.0.** One short stream and one long stream, one client thread each,
through one `Laya` instance:

| Model (short / long tokens) | GPU-only req/s | GPU + ANE req/s | Multiplier | Short-stream P99, ms | Answer mismatches |
|---|---:|---:|---:|---:|---:|
| laya (128 / 512) | 41.9 | 122.5 | **2.92×** | 48.1 → 11.2 | 0 |
| laya-multilingual (96 / 1024) | 55.7 | 241.8 | **4.34×** | 36.5 → 6.9 | 0 |
| laya-typed-decisions (128 / 1024) | 24.0 | 109.6 | **4.57×** | 84.3 → 11.9 | 0 |

**Open-loop arrivals, v1.0.** P99 is measured from arrival, including queueing, for
short single-question requests. The rates are the higher of the two offered rates per
model:

| Model | Poisson, GPU-only → GPU + ANE | Bursty, GPU-only → GPU + ANE |
|---|---:|---:|
| laya (43.1 / 46.2 req/s) | 170.7 → 30.3 ms | 1538.0 → 108.5 ms |
| laya-multilingual (80.5 / 83.8 req/s) | 283.9 → 16.3 ms | 2052.3 → 29.6 ms |
| laya-typed-decisions (33.4 / 35.8 req/s) | 319.4 → 27.3 ms | 1592.9 → 79.5 ms |

- At these rates, long-request P99 also drops, because the GPU no longer serves the
  short traffic.
- All three models had 0 answer mismatches.
- [`examples/heterogeneous_routing.py`](examples/heterogeneous_routing.py) prints backend,
  device and routing reason for each request in a mixed batch.

### Methodology

- **Machine:** Apple M4 Max, macOS 26.6.2 (25G83), MLX 0.32.2, coremltools 9.0, Python
  3.12.14. The machine was quiet, with nothing else using the GPU or ANE.
- **Heterogeneous configuration:** `device="auto"`, `execution="workers"`. The GPU runs in
  a worker process. The ANE runs on a dispatcher thread (laya, laya-typed-decisions) or in
  a worker process (laya-multilingual), chosen per model from measurements.
- **GPU-only baseline:** `device="gpu"`, `execution="workers"`, with both streams on one
  MLX worker queue.
- **Closed loop:**
  - 20 s windows, 3 cycles in alternating order, median over cycles;
  - 5 warm-up predictions per request type per instance before measuring.
- **Open loop:**
  - Poisson arrivals, plus a bursty variant (1 s at 3× the rate, then 2 s idle, with the
    same mean);
  - mix: 60% short 1-question, 20% medium, 10% long, 10% short 4-question;
  - 20 s per rate, with the same arrival sequence for both configurations;
  - latency from arrival to result, queueing included.
- **Correctness:** every answer is compared with the inline answer from the device that
  served it.
- **Single-request latency:**
  - one fresh process per configuration;
  - 10 warm-up and 50 timed iterations;
  - two passes, the second in reverse order.
- **Reproducibility:** the v1.0 re-run reproduced the v0.1 figures within ±1.8% over 50
  configurations.

Full method, tables and raw data: [`benchmarks/v1.0.md`](benchmarks/v1.0.md),
[`docs/benchmarks.md`](docs/benchmarks.md).

## Safety and fallback behaviour

- **Explicit ANE requests never fall back silently.**
  - `device="ane"` runs the exact validated artifact on `CPU_AND_NE` in FP16, or it
    raises.
  - It raises `UnsupportedShapeError` for a request longer than every validated bucket,
    `ArtifactMissingError` for a missing artifact, and `ComputeUnitMismatchError` when the
    compute plan or the runtime probe shows the model is not on the Neural Engine.
- **Unsupported or unvalidated ANE paths fail loudly.** Every load checks the artifact:
  - the manifest schema, revision, weight hash and file hash;
  - the build platform, compute plan and parity record;
  - a timing probe against `CPU_ONLY`.

  A corrupt artifact is quarantined, and a failed build is never registered.
- **Unknown or unvalidated platforms default to MLX.** On a machine whose SoC, macOS major
  version or coremltools version has no validated profile, `auto` uses MLX only
  (`platform_not_validated`). `laya-apple calibrate` can build that machine's profile.
- **A failing device stays failed.** A dead worker fails its requests. Nothing is
  re-run on the other device.

Every path is listed with its test in
[`docs/no-silent-fallback.md`](docs/no-silent-fallback.md).

## Limitations

- **One test machine.** Every benchmark comes from one Apple M4 Max on macOS 26.6.2.
- **Routing thresholds are not portable.** Do not assume the same thresholds on another
  Apple SoC. There, `auto` stays on MLX until the ANE artifacts are built and calibrated
  on that machine.
- **Long contexts stay on MLX,** which is faster than the ANE there. The ANE path is
  batch 1 only.
- **Isolation between the two engines is partial.**
  - Each stream's P99 under concurrency is above its solo value
    ([`benchmarks/v0.2.md`](benchmarks/v0.2.md)).
  - With the ANE on a thread, heavy Python work in the calling thread slows it (GIL).
- **Cold start on a fresh artifact location** costs 3–5 minutes of Core ML compile per
  model. Build and import pre-warm it, and `ane_startup="background"` serves on MLX in
  the meantime ([`benchmarks/v0.3.md`](benchmarks/v0.3.md)).
- **`choice` decisions can depend on option order.** This is a property of upstream Laya:
  - upstream changes its decision under permutation in 35% of the test cases for
    laya-typed-decisions, and 22.5% for laya;
  - laya-apple MLX FP32 reproduces upstream exactly, permutation by permutation
    ([`research/option-order/`](research/option-order/)).
- **Not yet measured:** energy use, quantized artifacts and cross-SoC validation.

## Reproduction

- [`research/phase-0-feasibility/`](research/phase-0-feasibility/) holds the feasibility
  study:
  - conversion scripts (`scripts/convert_ane.py`, `scripts/convert_coreml.py`);
  - the parity methodology;
  - the raw data behind the routing table.
- [`benchmarks/`](benchmarks/) holds the per-release reports. `benchmarks/v1.0/` is the raw
  v1.0 data, and [`scripts/v1_report.py`](scripts/v1_report.py) renders every table
  above from it.
- [`docs/benchmarks.md`](docs/benchmarks.md) lists the commands to reproduce every report.
- [`laya_apple/conversion/`](laya_apple/conversion/) is the production artifact build,
  with its parity gate.

## Documentation

- [`docs/guide.md`](docs/guide.md) is the user guide:
  - the question schema and devices;
  - workers mode;
  - artifacts: build, verify, prune, export/import and provenance;
  - calibration, the CLI, the failure policy and diagnostics.
- [`docs/api.md`](docs/api.md) covers the stable public API and the deprecation policy.
- [`docs/compatibility.md`](docs/compatibility.md) says what is tested, expected and
  unknown.
- [`docs/support-matrix.md`](docs/support-matrix.md) lists models, revisions and Core ML
  configuration status.
- [`CHANGELOG.md`](CHANGELOG.md).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for:
- development setup and the test tiers;
- the artifact release policy;
- the rules every change keeps: parity tolerances are never loosened, no silent fallback,
  and no committed model artifacts.

## Security

See [`SECURITY.md`](SECURITY.md) for supported versions and how to report a
vulnerability privately.

## License and attribution

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for adapted components and
their upstream revisions.
- Model weights are not included or redistributed. They are downloaded from the pinned
  Hugging Face revisions.
- This is an independent project, not an official release of Convai Innovations, Apple
  or MLX.
