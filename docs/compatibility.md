# Compatibility statement (v1.0)

This page states, for each dimension of the runtime, what is **tested** (every measured
number in this repository comes from this exact profile), what is **expected** (a
hypothesis, not measured), and what is **unknown**. Where it disagrees with
[`support-matrix.md`](support-matrix.md) or [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md),
those win; this page restates their conclusions for someone deciding whether to run
`laya-apple` on a given machine.

## Tested profile

Every benchmark, parity record and routing threshold in this repository was measured on
exactly one profile:

| | |
|---|---|
| SoC | Apple M4 Max |
| macOS | 26.6.2 (25G83) |
| MLX | 0.32.2 |
| coremltools | 9.0 |
| NumPy | 2.1.3 (the `[ane]` extra pins `numpy>=1.26,<2.2`) |
| Python | 3.12.14, but 3.11–3.13 are all in the install matrix ([`release_gate.py`](../scripts/release_gate.py) builds and smoke-tests base and `ane` extras on 3.11, 3.12, 3.13) |

## Per-dimension status

| Dimension | Tested | Expected | Unknown |
|---|---|---|---|
| SoC | Apple M4 Max | Other Apple M-series SoCs run MLX correctly (hypothesis: MLX itself is validated across Apple Silicon upstream) | ANE placement, correctness and routing thresholds on any other SoC |
| macOS | 26.6.2 | macOS 15–26 run MLX correctly | ANE behaviour on any macOS other than 26.6.2; macOS 27.x specifically — prior third-party work saw different Core ML placement there (enumerated shapes moved to the GPU) |
| Python | 3.12.14 (benchmarks); 3.11–3.13 (install matrix) | — | Any Python outside 3.11–3.13 (unsupported, not merely untested) |
| MLX | 0.32.2 | Other MLX versions within the package's declared constraint are expected to work for the GPU backend | Numerical or performance drift on a materially different MLX version |
| coremltools | 9.0 (pinned exactly by the `ane`/`convert` extras) | — | Any other coremltools version — the placement and parity gates have not been run against one, and coremltools 9.0 constrains NumPy to `<2.2` for a reason: it breaks on NumPy ≥ 2.5 |
| NumPy | 2.1.3, and `>=1.26,<2.2` when the `ane` extra is installed | — | Behaviour outside that pinned range with the ANE backend |
| `laya` × MLX | validated | — | — |
| `laya` × Core ML/ANE | validated (buckets 64/96/128, `CPU_AND_NE`) | — | Batched (`B>1`), other compute units, other profiles |
| `laya-multilingual` × MLX | validated | — | — |
| `laya-multilingual` × Core ML/ANE | validated (buckets 64/96/128/256, `CPU_AND_NE`) | — | Batched (`B>1`), other compute units, other profiles |
| `laya-typed-decisions` × MLX | validated | — | — |
| `laya-typed-decisions` × Core ML/ANE | validated (buckets 64/96/128, `CPU_AND_NE`) | — | Batched (`B>1`), other compute units, other profiles |
| Compute units | `CPU_AND_NE` (ANE, shipped), Metal GPU (MLX, shipped) | `CPU_AND_GPU` is correct but unused (slower than MLX) | — |
| Compute units | — | — | `CPU_ONLY` is invalid (FP16 precision on Core ML's CPU path); `ALL` is never used because placement is not stable across runs |
| Execution: `inline` | validated (single request at a time, thread-safe) | — | — |
| Execution: `workers` | validated: GPU always in its own worker process; ANE placement (`thread`/`process`) chosen per model from measurement on the tested profile | The same GPU-process / ANE-thread-or-process split should hold on other Apple Silicon, since the mechanism (GIL contention, IPC cost) is not M4-Max-specific | Whether the per-model `thread` vs `process` choice in `laya_apple/data/placement.json` is the right one on a different SoC — it was derived from measurements on this machine only |
| `ane_placement="thread"` | validated for `laya`, `laya-typed-decisions` | — | — |
| `ane_placement="process"` | validated for `laya-multilingual` | — | — |
| Offline operation | validated (`local_files_only=True`, `HF_HUB_OFFLINE=1`, `--offline`) — no network access once checkpoints/artifacts are cached | — | — |

## What happens on an untested profile

`laya-apple` never guesses. On a platform profile (SoC, macOS major version,
coremltools version) that does not match a shipped, validated profile:

- `device="auto"` uses MLX only. Every result records
  `routing_reason == "platform_not_validated"`.
- `device="ane"` still works, but only after you validate it yourself on that machine:

  ```bash
  laya-apple artifacts build laya-typed-decisions   # builds locally; each bucket must
                                                     # pass its own placement and parity
                                                     # gate on this machine before it is
                                                     # registered
  laya-apple calibrate laya-typed-decisions          # measures MLX and ANE latency here
  ```

  `calibrate` applies the same rule that produced the shipped routing table
  (`docs/support-matrix.md`, "How the auto-ANE buckets were derived") to local
  measurements, and writes `<cache>/profiles/<profile>.json`. A local profile is used
  only when no shipped profile matches the running machine — it never overrides a
  validated shipped profile. `Laya.info()["routing_profile"]` reports which table is in
  effect: `"shipped"`, `"local:<path>"`, or `None` (MLX only, no profile matched).

- Artifacts are tied to the profile that built them. An artifact built on one SoC,
  macOS major version, or coremltools version is refused when loaded or imported on a
  different one (`ArtifactRevisionError`) — it is never silently accepted and re-checked
  loosely.
- `artifacts export` / `artifacts import` do not trust the exporting machine's checks.
  The importing machine re-validates everything itself: the manifest against the pinned
  checkpoint, the build platform profile, the file hash, the compute plan (100% ANE, 0
  device transitions), and the full parity gate against the shipped goldens (which
  requires the checkpoint to be downloaded or already cached there).

This is the same "no silent fallback" policy the runtime applies everywhere else (see
[`no-silent-fallback.md`](no-silent-fallback.md)): an unvalidated configuration is never
quietly used as if it were validated.

## Where to look for more detail

- Per-model, per-compute-unit validation status, exact pinned revisions and weight
  hashes: [`support-matrix.md`](support-matrix.md).
- How the auto-ANE bucket list and routing thresholds were derived from evidence:
  `support-matrix.md`, "How the auto-ANE buckets were derived", and
  `DEVELOPMENT_PLAN.md` §7.
- Platform scope as an architectural decision, including the roadmap for
  per-(SoC, macOS major, coremltools) capability profiles: `DEVELOPMENT_PLAN.md` §4 and
  §12.4.
- The stable public API and what compatibility promises apply to it (routing reasons,
  CLI, file formats): [`api.md`](api.md).
