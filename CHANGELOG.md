# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

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
