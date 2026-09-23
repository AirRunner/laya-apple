# laya-apple

Apple-native inference runtime for the Laya family of non-autoregressive
typed-decision models. Laya takes a context and a set of typed questions
(`choice`, `score`, `noul`) and returns calibrated decisions in a single
forward pass — there is no text generation. `laya-apple` runs Laya on Apple
Silicon with MLX on the GPU as the default backend, and the Apple Neural
Engine (via Core ML) as an optional accelerator for a narrow, validated set
of short, single-question, fixed-shape requests.

## Status

v0.3. Tested only on **Apple M4 Max, macOS 26.6.2**, with MLX 0.32.2 and
coremltools 9.0. Other Apple Silicon and macOS versions are expected to run
the MLX backend correctly but are not validated; the ANE path is validated
only on the tested profile (see `docs/support-matrix.md`).

## Install

`laya-apple` is not published on PyPI. Install from a clone:

```bash
git clone https://github.com/tc3oliver/laya-apple && cd laya-apple
uv sync                          # base: MLX backend only
uv sync --extra ane              # + Neural Engine backend (coremltools==9.0, pins numpy<2.2)
uv sync --extra ane --extra convert   # + building ANE artifacts (torch==2.7.0)
```

or with pip into an existing environment: `pip install ".[ane]"`.

Requires Python 3.11–3.13 and Apple Silicon (`arm64`) for MLX.

## Quickstart

```python
from laya_apple import Laya

model = Laya.from_pretrained("convaiinnovations/laya-typed-decisions")  # device="auto"
result = model.predict(
    context="The customer was charged twice for the same invoice and is frustrated.",
    questions={
        "refund": {"type": "noul", "instructions": "Does the customer request a refund?"},
        "urgency": {
            "type": "choice",
            "instructions": "How urgent is this?",
            "criteria": ["low", "medium", "high"],
        },
        "satisfaction": {
            "type": "score",
            "instructions": "Rate the customer's satisfaction.",
            "criteria": ["very unhappy", "unhappy", "neutral", "happy", "very happy"],
        },
    },
)
print(result.answers)
print(result.runtime)
```

Output of this exact script (MLX FP16, the `[ane]` extra installed, Apple M4 Max). The
answers are deterministic for a given device and precision; `latency_ms` is not.

```text
{'refund': {'type': 'noul', 'confidence': 0.5002, 'action': {'act_probability': 1.0}, 'noul': 0.4998},
 'urgency': {'type': 'choice', 'confidence': 0.0744, 'action': {'act_probability': 1.0}, 'choice': 'high',
             'probabilities': {'low': 0.1702, 'medium': 0.3375, 'high': 0.4923}},
 'satisfaction': {'type': 'score', 'confidence': 0.623, 'action': {'act_probability': 1.0}, 'score': 0.2479,
                  'legend': {'0': 'very unhappy', '1': 'unhappy', '2': 'neutral', '3': 'happy', '4': 'very happy'},
                  'probabilities': {'0': 0.7986, '1': 0.1756, '2': 0.0118, '3': 0.0071, '4': 0.0068}}}
backend=mlx device=gpu reason=multiple_questions model=laya-typed-decisions L=53 q=3 latency_ms=26.15
```

Three questions go to MLX (`multiple_questions`). Without the `[ane]` extra, the reason
is `ane_runtime_unavailable` instead.

The first call downloads and caches the pinned checkpoint from Hugging Face
unless it is already cached; set `local_files_only=True` to run fully
offline once cached (see [Offline use](#offline-use)).

### Question schema

`questions` is a dict of `{name: {"type": ..., "instructions": ..., "criteria": ...}}`,
matching upstream Laya's schema:

| `type` | `criteria` | Meaning |
|---|---|---|
| `choice` | non-empty dict of `{label: description}`, or a list of unique label strings | pick one label |
| `score` | non-empty list of level descriptions | pick a level, 0-indexed |
| `noul` | optional dict `{"false": ..., "true": ...}` | yes/no |

A list of plain strings is also accepted as a convenience; each string
becomes a `noul` question named after itself:

```python
model.predict(context="...", questions=["Does the customer request a refund?"])
```

## Devices

`Laya.from_pretrained(model_id, device="auto" | "gpu" | "ane", ...)`.

- `device="gpu"`: MLX only. Every request runs on the GPU.
- `device="ane"`: Core ML / ANE only. A request that does not fit a validated
  artifact fails loudly (`UnsupportedShapeError`, `ArtifactMissingError`, etc.)
  — it never runs on MLX instead.
- `device="auto"` (default): MLX for everything, except a request routes to
  the ANE when **all** of the following hold:

  1. the request has exactly one question;
  2. the longest prompt row fits one of the model's auto-ANE buckets;
  3. a validated artifact for that bucket is present in the cache and passes
     its placement/parity checks at load;
  4. the platform profile matches a validated profile (currently only Apple
     M4 Max / macOS 26.6.2 / coremltools 9.0).

  Otherwise the request runs on MLX. Every result records exactly one
  routing reason in `result.runtime.routing_reason`:

  | Reason | Meaning |
  |---|---|
  | `gpu_requested` | `device="gpu"` was requested |
  | `ane_requested` | `device="ane"` was requested |
  | `validated_short_single_question_path` | auto chose the ANE |
  | `multiple_questions` | auto chose MLX: more than one question |
  | `sequence_exceeds_ane_auto_range` | auto chose MLX: the longest row is above the auto buckets |
  | `ane_artifact_unavailable` | auto chose MLX: no validated artifact for the bucket |
  | `ane_runtime_unavailable` | auto chose MLX: coremltools is not installed |
  | `platform_not_validated` | auto chose MLX: unknown hardware/OS profile |

Auto-ANE buckets, generated from measured evidence
(`laya_apple/data/routing.json`):

| Model | Buckets for explicit `device="ane"` | Buckets used by `device="auto"` |
|---|---|---|
| `laya` | 64, 96, 128 | 64, 96, 128 |
| `laya-multilingual` | 64, 96, 128, 256 | 64, 96, 128 |
| `laya-typed-decisions` | 64, 96, 128 | 64, 96, 128 |

Longer validated lengths lose to MLX on latency and are not offered by
either explicit `device="ane"` or `auto`. Multi-question requests are never
auto-routed to the ANE: MLX batching wins at every measured length for 4 and
8 questions, and 2–3 questions were not measured, so they route to MLX
conservatively.

## Concurrent GPU + ANE execution

By default (`execution="inline"`) a `Laya` instance runs one request at a time in the
calling process. For serving many requests, use `execution="workers"`:

```python
from laya_apple import Laya

with Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers") as laya:
    future = laya.submit(context="...", questions={...})        # concurrent.futures.Future
    result = laya.predict(context="...", questions={...})       # thread-safe, blocking
    # in asyncio code: result = await laya.apredict(context="...", questions={...})
```

- MLX (GPU) runs in its own worker process.
- Core ML (ANE) runs either on a dedicated thread in your process or in its own worker
  process. The choice is made per model from measurements (`ane_placement="auto"`), and
  you can override it with `"thread"` or `"process"`.
- Your process builds prompts, routes, and formats answers.
- The two devices serve requests at the same time. Each device runs its own queue in
  arrival order.
- `close()` (or leaving the `with` block) finishes queued work and stops the worker.

**Why this placement, and its limits.** Every alternative was measured (see
[`research/v0.2-concurrency/`](research/v0.2-concurrency/)):
- Both backends in one interpreter cost the GPU 9–11% of its throughput.
- With both devices busy, a device whose requests arrive from another process runs its
  host-side work 4–6× slower.
- Core ML's Python `predict` holds the GIL for much of an ANE call, so running the ANE in
  your process costs more at high short-request rates.
- Results on the Phase -1 mix with the chosen placement (`benchmarks/v0.2.md`):

  | Model | Short-stream P99 vs solo | Long stream throughput vs solo | Aggregate vs GPU-only |
  |---|---:|---:|---:|
  | laya-typed-decisions | +8% | −11% | 4.6× |
  | laya | +5% | −12% | 2.9× |
  | laya-multilingual | +99% | −13% | 3.6× |

- Under open-loop load, short requests see 2.5–66× lower P99 than with GPU-only serving.
- Complete isolation (each stream within 10% of its solo P99) was not reached on this
  platform.

**Routing with queues.** On an idle machine, `auto` behaves exactly like `inline`. When a
device is busy, the router compares expected completion times: the device's backlog plus
the request's measured service time. Two additional reasons can then appear:

| Reason | Meaning |
|---|---|
| `ane_backlog_shorter_on_gpu` | a short single-question request went to MLX because the ANE queue was longer |
| `gpu_backlog_shorter_on_ane` | a request in the tie band (an explicit-only bucket, e.g. multilingual L256) went to the ANE because the GPU queue was longer |

Long requests and multi-question requests never go to the ANE under `auto`, however busy
the GPU is. Every `RuntimeInfo` records `execution`, `queue_wait_ms` and the backlog
estimates the router saw (`gpu_backlog_ms`, `ane_backlog_ms`).

If the ANE worker process dies:
- the request running on it raises `BackendUnavailableError`;
- later `auto` requests run on MLX with reason `ane_runtime_unavailable`, and a warning is
  issued once;
- explicit `device="ane"` requests keep raising.

## The ANE path: building artifacts

`device="ane"` and the auto-ANE path need a Core ML artifact for the exact
(model, revision, bucket) tuple. Artifacts are never downloaded or
committed — build them locally:

```bash
laya-apple artifacts build laya-typed-decisions
```

This builds every offered bucket for the model (or pass `--length 64` to
build one). Each bucket goes through:

1. **layout** — the channel-first BC1S PyTorch body is checked against the
   upstream FP32 `DecisionModel` on an exact-length row;
2. **conversion** — traced and converted to Core ML with FP16 compute and
   FP16 I/O, fixed shape (`B=1 × L=<bucket>`), macOS 15 deployment target;
3. **compile** — to a `model.mlmodelc` (the intermediate `.mlpackage` is not
   kept, since loading it recompiles for the ANE on every process — 37.8s
   vs. 0.21s for a compiled artifact);
4. **placement** — the Core ML compute plan on `CPU_AND_NE` must show 100%
   of operations on the Neural Engine and 0 device transitions;
5. **parity** — every shipped golden row that fits the bucket, checked
   against the upstream PyTorch FP32 reference, with the unchanged Phase -1
   gate (see [Correctness](#correctness));
6. **atomic registration** — the artifact is built in a temporary directory
   and renamed into place only after every step above passes. A partial or
   failed build never appears as usable; a failure leaves the manifest under
   `artifacts/rejected/` as evidence.

Building one bucket takes 1.5–3.5 minutes on the tested hardware, including the pre-warm
load at the registered path.

Artifacts are cached under `$LAYA_APPLE_CACHE` if set, else
`$XDG_CACHE_HOME/laya-apple`, else `~/.cache/laya-apple`. They live outside
the source tree and are never committed to Git or redistributed with the
package.

Re-check everything already registered:

```bash
laya-apple artifacts verify
```

This reloads every cached artifact and re-checks its manifest schema,
revision and weight hash, graph variant, bucket, compute units, parity status,
build platform profile, file hashes and Core ML placement.

At runtime the manifest, revision, profile and compute-unit checks run on every
load. The file hash and the placement check (about 1.1 s per bucket together)
run on first load and again whenever the artifact's files, the macOS build or
the coremltools version change; a passing result is stamped under
`<cache>/verified/`. `artifacts verify` ignores the stamp.

**Cold start.** Loading a compiled artifact triggers Core ML's on-device ANE
compile the first time a given artifact location is loaded (25–86 s per bucket
on the tested machine); the system caches the result. `artifacts build` pays
this cost once at the registered path, so later loads take 0.14–0.31 s per
bucket. If the system evicts its cache, the next load pays it again.

## Artifact lifecycle

- **Concurrent builds are safe.** One build per (model, revision, bucket) runs at a time
  across processes, under a file lock in `<cache>/artifacts/.locks/`. A second builder
  waits, then finds the artifact already registered.
- **Corruption is quarantined.** An artifact whose files no longer match their manifest
  hash, or whose manifest cannot be read, is moved to `<cache>/artifacts/quarantine/` when
  it is detected. The error names the rebuild command. Nothing corrupt stays where the
  runtime looks.
- **Cleanup is explicit.** `laya-apple artifacts prune` lists what it would delete and why:
  - other revisions or weights, unregistered models, buckets no longer offered;
  - builds from another platform profile, unvalidated or rejected builds;
  - quarantined entries, abandoned staging directories, orphaned verification stamps.
  Add `--yes` to delete. It only ever deletes inside the cache.
- **Warm after eviction.** `laya-apple artifacts warm MODEL` pays Core ML's on-device ANE
  compile ahead of the first request.

**Moving artifacts between machines.** Artifacts are never downloaded. You can build them
once and carry them to another machine yourself:

```bash
laya-apple artifacts export laya-typed-decisions --out exports/   # one .tar.gz per offered bucket
# on the other machine:
laya-apple artifacts import exports/laya-typed-decisions-L64.tar.gz
```

An import is registered only after the receiving machine has checked, itself:
- the manifest against the pinned checkpoint;
- the build platform profile;
- the file hash;
- the compute plan (100% ANE, 0 transitions);
- the full parity gate against the shipped goldens (the checkpoint must be downloaded).

The local results are recorded in the manifest under `imported`.

**Cold start.** With `execution="workers"`, `ane_startup="background"` makes
`from_pretrained` return as soon as MLX is ready. The ANE finishes loading behind it,
including any on-device compile. Until it is ready, `auto` routes to MLX with reason
`ane_starting`. `laya.wait_for_ane()` blocks until it is ready, and
`info()["ane_ready"]` reports it. Measured start times are in
[`benchmarks/v0.3.md`](benchmarks/v0.3.md).

## Calibrating another machine

The shipped routing table applies only to the profile it was measured on (SoC, macOS
major version, coremltools version). On any other profile, `auto` uses MLX only
(`platform_not_validated`). To enable the ANE on your machine:

```bash
laya-apple artifacts build laya-typed-decisions   # builds and parity-validates here
laya-apple calibrate laya-typed-decisions          # measures MLX and ANE latency here
```

`calibrate` applies the same rule that produced the shipped table to local measurements.
It writes `<cache>/profiles/<profile>.json`. Such a local profile is used only when no
shipped profile matches the machine. `Laya.info()["routing_profile"]` reports which table
is in effect: `shipped`, `local:<path>`, or `None` (MLX only).

## No silent fallback

- An explicit `device="ane"` request runs on the exact validated artifact
  with the exact validated compute units, or it raises. It never silently
  runs on MLX, CPU, or a different Core ML placement.
- Under `device="auto"`, a decision to use MLX instead of the ANE is made
  **before** the request runs, and the reason is recorded in
  `result.runtime.routing_reason`. It is never a reaction to Core ML
  changing placement mid-flight.
- The Core ML compute plan is checked against the declared placement (100%
  ANE, 0 transitions) at build time, at first load, and again after any change
  to the artifact files, the macOS build or coremltools. A mismatch raises
  `ComputeUnitMismatchError` instead of silently running on CPU.

### Failure policy

Every failure the runtime can detect is a specific exception from
`laya_apple.errors`:

| Condition | Error |
|---|---|
| model not registered | `UnsupportedModelError` |
| malformed questions | `InvalidRequestError` |
| ANE requested, request exceeds the largest offered bucket, or the model/shape pair is not validated | `UnsupportedShapeError` |
| ANE requested, no artifact in the cache | `ArtifactMissingError` |
| artifact built from another revision or other weights | `ArtifactRevisionError` |
| artifact files do not match their manifest hash | `ArtifactIntegrityError` |
| artifact has no passing parity record | `ArtifactParityError` |
| requested compute units differ from the validated ones, or the loaded compute plan is not 100% ANE / 0 transitions | `ComputeUnitMismatchError` |
| MLX or coremltools unavailable for the requested device | `BackendUnavailableError` |
| offline and not cached | `BackendUnavailableError` (with download instructions) |

## CLI reference

The `--offline` flag is **global** and must come before the subcommand:

```bash
laya-apple --offline predict laya-typed-decisions --context "..." --questions '{...}'
```

Commands:

```text
laya-apple predict MODEL --context TEXT --questions JSON [--device auto|gpu|ane] [--dtype float16|float32]
laya-apple info [MODEL]
laya-apple download MODEL...
laya-apple artifacts build MODEL [--length L ...] [--force] [--skip-existing]
laya-apple artifacts list
laya-apple artifacts verify [MODEL] [--length L ...]
laya-apple artifacts warm [MODEL] [--length L ...]
laya-apple artifacts prune [--yes]
laya-apple artifacts export MODEL [--length L ...] [--out DIR]
laya-apple artifacts import ARCHIVE.tar.gz [--force]
laya-apple calibrate [MODEL ...] [--warmup N] [--iters N]
laya-apple parity MODEL [--device gpu|ane] [--dtype float16|float32]
laya-apple benchmark MODEL [--device auto|gpu|ane] [--lengths L ...] [--questions N] [--warmup N] [--iters N] [--output FILE]
```

`--context` and `--questions` each accept inline text/JSON, `@file` to read
from a file, or `-` to read from stdin.

Example:

```bash
laya-apple --offline predict laya-typed-decisions \
  --context "The customer was charged twice." \
  --questions '{"refund": {"type": "noul", "instructions": "Does the customer request a refund?"}}'
```

```bash
laya-apple --offline info laya-typed-decisions
```

## Offline use

Once checkpoints (and, if used, ANE artifacts) are cached, no network access
is needed. Three equivalent ways to force this:

- `Laya.from_pretrained(model_id, local_files_only=True)`
- `HF_HUB_OFFLINE=1` in the environment
- `laya-apple --offline <subcommand> ...` on the CLI

Running offline against an uncached checkpoint raises
`BackendUnavailableError` with download instructions instead of hanging or
silently going online.

## Runtime diagnostics

Every `Result.runtime` (a `RuntimeInfo`) records:

- `backend` (`mlx` / `coreml`) and `device` (`gpu` / `ane`);
- `model` and `model_revision`;
- `sequence_length` (longest prompt row, in tokens) and `question_count`;
- `routing_reason`;
- `artifact_revision` (the ANE artifact hash, or `mlx:<weights sha256 prefix>`);
- `compute_units` and `buckets` (Core ML only);
- `dtype`;
- `latency_ms`, from the call to the result (queueing included);
- `execution` (`inline` / `workers`); with workers, `queue_wait_ms` and the
  `gpu_backlog_ms` / `ane_backlog_ms` estimates the router used.

## Correctness

The semantic reference is unmodified upstream Laya
(`NandhaKishorM/laya@573e5b6`) running PyTorch on CPU in FP32. MLX and Core
ML are validated backends, not references. The parity gate requires
calibrated-probability max error ≤ 0.02 (FP16) and 0 hard decision
mismatches.

MLX FP16 parity measured for this release:

| Model | Rows | Max probability error | Hard mismatches |
|---|---:|---:|---:|
| `laya` | 163 | 0.0037 | 0 |
| `laya-multilingual` | 187 | 0.0045 | 0 |
| `laya-typed-decisions` | 187 | 0.0017 | 0 |

Run the gate yourself: `laya-apple parity laya-typed-decisions --device gpu`.

## Performance

Measured for this release (Apple M4 Max, macOS 26.6.2). Model-only forward P50 in ms, one
question, exact-length requests. Full tables with P99, end-to-end and load times:
[`benchmarks/v0.1.md`](benchmarks/v0.1.md).

| Model | MLX L128 | ANE L128 | MLX at max_len | `auto` at L128 |
|---|---:|---:|---:|---|
| `laya` | 12.27 | 9.89 | 35.26 (L512) | ANE, 9.91 |
| `laya-multilingual` | 6.46 | 4.31 | 29.27 (L1024) | ANE, 4.35 |
| `laya-typed-decisions` | 12.26 | 9.88 | 71.02 (L1024) | ANE, 9.87 |

- The ANE's advantage is narrow: 1.15–1.63× over MLX, only for short single-question
  requests. The GPU wins by 1.5–2.8× at long lengths, and MLX batching wins for
  multi-question requests.
- End-to-end overhead (prompt, routing, formatting) is 0.02–0.76 ms.
- Loading a model takes 0.15–0.39 s on MLX alone, and 1.5–1.7 s with ANE artifacts. That
  includes a one-time ~1.1 s coremltools import; each bucket then loads in 0.14–0.31 s.

## Limitations

- Validated only on Apple M4 Max, macOS 26.6.2, coremltools 9.0. Other
  Apple Silicon/macOS combinations are expected to run MLX correctly but are
  unvalidated; the ANE path requires validation on the specific machine.
- The ANE path is `B=1` only. Multi-question requests never auto-route to
  the ANE; with explicit `device="ane"` their questions run one after
  another. Batched (`B>1`) artifacts are not offered: their parity is
  unmeasured.
- `device="auto"` never uses the ANE above L128 for `laya` and
  `laya-typed-decisions`, or above L128 for `laya-multilingual` (its longer
  L256 bucket is explicit-only).
- With the default `execution="inline"`, a `Laya` instance runs one request at a time;
  use `execution="workers"` for concurrent GPU + ANE serving.
- Core ML `ALL` and `CPU_ONLY` compute-unit configurations are never used in
  production; both were shown incorrect or unstable on the ANE/CPU paths in
  Phase -1.
- No energy or power measurements are made or claimed.

## License and attribution

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for adapted
components and their upstream revisions. Model weights are not included or
redistributed; they are downloaded from the pinned Hugging Face revisions
listed in `docs/support-matrix.md`. This is an independent project, not an
official Convai Innovations, Apple, or MLX release.

## More

- [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md) — architecture,
  routing derivation, correctness policy, and roadmap.
- [`docs/support-matrix.md`](docs/support-matrix.md) — platform and model
  support matrix.
- [`docs/compatibility.md`](docs/compatibility.md) — what is tested, expected
  and unknown, and what happens on an untested machine.
- [`docs/api.md`](docs/api.md) — the stable public API and the deprecation
  policy.
- [`docs/no-silent-fallback.md`](docs/no-silent-fallback.md) — every path that
  could run a request somewhere other than recorded, and the test that pins it.
- [`docs/benchmarks.md`](docs/benchmarks.md) — how to reproduce every benchmark.
- [`research/phase-0-feasibility/`](research/phase-0-feasibility/) — the
  measurements this project is built on.
