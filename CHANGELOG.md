# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.2.0] - 2026-09-23

### Added

- **`execution="workers"`: heterogeneous GPU + ANE serving.**
  - MLX runs in a worker process. Core ML runs either on a dispatcher thread in the
    caller's process or in a worker process, chosen per model from measurements
    (`ane_placement="auto"`, `laya_apple/data/placement.json`,
    `scripts/derive_placement.py`).
  - Each device has its own FIFO queue.
  - `Laya.submit()` returns a `concurrent.futures.Future`. `Laya.apredict()` is
    awaitable. `predict()` is thread-safe. `Laya.close()` and the context-manager form
    stop the workers.
  - Workers start as `python -m laya_apple.executor` over an authenticated AF_UNIX
    connection, so the caller's `__main__` is never re-imported.
  - Workers load in parallel and warm every compiled shape before serving.
- **Queue-aware, isolation-first routing** (`laya_apple/scheduling.py`).
  - Expected completion is the device backlog plus a measured service time (Phase -1
    forward P50s, now in `routing.json` as `service_ms`).
  - It is identical to the v0.1 rule on an idle machine, which is tested exhaustively.
  - New reasons: `ane_backlog_shorter_on_gpu` and `gpu_backlog_shorter_on_ane` (the
    tie-band bucket, loaded in workers mode).
  - Long and multi-question requests never go to the ANE.
- **`RuntimeInfo` fields:** `execution`, `queue_wait_ms`, `device_ms`, and the
  `gpu_backlog_ms` / `ane_backlog_ms` values the router saw.
- **Failure handling:**
  - A dead GPU worker fails its requests loudly. Nothing is moved to the other device.
  - A dead ANE worker process is reported as `ane_runtime_unavailable` under `auto`, with
    one warning.
  - `close()` fails any still-queued request instead of leaving it pending.
- **Benchmarks and research:**
  - `scripts/bench_concurrency.py` covers the closed-loop exit-gate mix plus open-loop
    Poisson and bursty mixed workloads. Every answer is checked against the inline result
    for the same device.
  - `benchmarks/v0.2.md` is the release report.
  - `research/v0.2-concurrency/` holds the step-1 gate and the request-driven findings.

### Changed

- Vendored model code is excluded from `ruff format` so it stays byte-identical to its
  upstream revision.

### Known limitations

- **Isolation is partial for request-driven serving** on the tested platform:
  - The step-1 criterion "each stream's P99 within 10% of solo" was not met by any of the
    tested designs; the measured values are in `benchmarks/v0.2.md`.
  - With both devices busy, a device fed over IPC runs its host-side work several times
    slower.
  - Core ML's Python `predict` holds the GIL for much of an ANE call.

## [0.1.0] - 2026-09-23

### Added

- **Runtime:** `Laya.from_pretrained()` and `Laya.predict()` with a common
  request/result schema and `RuntimeInfo` diagnostics.
- All three pinned checkpoints (`laya`, `laya-multilingual`,
  `laya-typed-decisions`) at fixed revisions, with weight-hash verification
  on load.
- MLX backend (default), FP16 and FP32.
- Core ML / ANE backend for the offered fixed-shape buckets, `bc1s-masked`
  graph, `CPU_AND_NE` compute units, `B=1`; multi-question requests can run
  on the ANE sequentially only when explicitly requested (`device="ane"`),
  and never auto-route there.
- Model and capability registry, including recorded invalid configurations.
- `device="gpu" | "ane" | "auto"` routing, with deterministic, per-model
  auto-routing thresholds generated from measured Phase -1 evidence
  (`laya_apple/data/routing.json`, `scripts/derive_routing.py`).
- Strict validation for every unsupported request or artifact state, each
  raising a specific `laya_apple.errors` exception; no silent fallback
  between devices or Core ML compute-unit placements.
- Artifact pipeline: build → compile → placement check → parity gate →
  atomic registration, with a full-provenance manifest per artifact.
- Parity infrastructure: shipped PyTorch FP32 goldens, `laya-apple parity`,
  and a `tests/parity` suite.
- CLI (`laya-apple`): `predict`, `info`, `download`, `artifacts
  build|list|verify`, `parity`, `benchmark`, with a global `--offline` flag.
- Offline inference via `local_files_only=True`, `HF_HUB_OFFLINE=1`, or
  `--offline`; no telemetry.
- Documentation: README, this changelog, `docs/support-matrix.md`,
  `LICENSE`/`NOTICE`.

### Release gate (DEVELOPMENT_PLAN.md §12.2), Apple M4 Max / macOS 26.6.2

1. **Clean install:** base and `[ane]` in fresh venvs on Python 3.11, 3.12 and 3.13.
2. **README quickstart:** runs unchanged, offline, in all six environments.
3. **MLX:** passes parity for all three checkpoints in FP16 (max probability error ≤ 0.0045)
   and FP32 (≤ 1e-4), 0 hard mismatches.
4. **ANE:** all 10 offered buckets build from the release commit and pass placement (100%
   ANE, 0 transitions) and parity (≤ 0.0128, 0 hard mismatches). Each loads in 0.14–0.31 s
   from the cache.
5. **Unsupported paths:** every row of the failure table raises its own error, with tests.
6. **Auto routing:** matches the derived rule in every benchmarked configuration, and is
   tested for determinism.
7. **CLI:** every command works.
8. **Offline:** `HF_HUB_OFFLINE=1` and `local_files_only=True` work end to end.
9. **Provenance:** recorded in every manifest, including a clean release-commit revision.
10. **Benchmark:** `benchmarks/v0.1.md`, 100 of 100 configurations ok.
11. **Documentation:** checked against code and data by an independent review pass.
12. **Tests:** the full suite passes, 257 tests including integration and parity.

### Not included in v0.1

- Concurrency or request queues (planned for v0.2).
- Core ML on the GPU (`CPU_AND_GPU`) as a product path — validated for
  correctness in Phase -1 but not exposed.
- Batched (`B>1`) ANE artifacts — parity at `B>1` is unmeasured.
- Long-context ANE optimization, windowed attention, quantization, and
  energy/power measurement — all tracked as non-blocking research.
- Validation on any hardware/OS profile other than Apple M4 Max / macOS
  26.6.2 / coremltools 9.0.
