# Contributing

## Dev setup

```bash
git clone https://github.com/tc3oliver/laya-apple && cd laya-apple
uv sync --extra dev                          # tests, ruff, psutil
uv sync --extra dev --extra ane              # + Neural Engine backend (coremltools==9.0)
uv sync --extra dev --extra ane --extra convert     # + building ANE artifacts (torch==2.7.0)
uv sync --extra dev --extra reference        # + regenerating PyTorch FP32 goldens
```

`ane`, `convert` and `reference` pin `numpy<2.2` and specific `torch`/`transformers`/`laya`
versions; do not relax those pins in a PR unless the PR is specifically about updating them.

## Running tests

Tests are marked with `integration`, `parity`, `ane` and `stress` (see
`[tool.pytest.ini_options]` in `pyproject.toml`). The fast suite is everything else and
needs no downloaded checkpoints or built artifacts:

```bash
uv run pytest -q -m "not integration and not parity and not ane and not stress"
```

`integration`, `parity` and `ane` tests need a checkpoint already in the Hugging Face
cache (run once online, or `laya-apple download <model>`, first):

```bash
HF_HUB_OFFLINE=1 uv run pytest -q -m "integration or parity or ane"
```

`LAYA_APPLE_CACHE` points laya-apple's own cache (build artifacts, verification stamps)
at a directory of your choice; it defaults to `~/.cache/laya-apple`. Set `HF_HUB_OFFLINE=1`
once checkpoints are cached to keep tests from touching the network.

Stress tests are opt-in and long-running:

```bash
LAYA_APPLE_STRESS=1 uv run pytest -q -m stress                    # ~60s
LAYA_APPLE_STRESS=1 LAYA_APPLE_STRESS_SECONDS=600 uv run pytest -q -m stress   # release soak
```

## Lint and format

```bash
uv run ruff check .
uv run ruff format laya_apple scripts tests
```

`research/` is excluded from ruff entirely. `laya_apple/models/modernbert_mlx.py` and
`laya_apple/conversion/torch_reference.py` are vendored upstream code kept byte-identical
to their source revision (see `NOTICE`) and are excluded from `ruff format`; leave their
formatting alone.

## Release gate

`scripts/release_gate.py` runs the full pre-release check: the install matrix, the slow
pytest markers, and (optionally) a soak test. See `--help` for the current flags,
including `--quick` (skip the install matrix and slow markers) and `--soak SECONDS`. Run
it before proposing a release, not as a substitute for the fast suite during normal
development.

## Non-negotiable rules

These are enforced by tests and reviewed as blocking, not stylistic:

- **Parity tolerances are never loosened.** The shipped tolerances
  (`laya_apple/parity/__init__.py`) are FP32 `1e-4`, FP16 `0.02` max absolute probability
  difference, and zero hard mismatches. A PR that needs a looser tolerance to pass is a
  regression, not a passing test; fix the regression instead.
- **No silent fallback.** A request never runs on a different device, bucket, precision,
  or the network (when offline was requested) without that being visible in the result or
  the exception. See [`docs/no-silent-fallback.md`](docs/no-silent-fallback.md). If your
  change adds or touches a fallback-relevant code path (a device or bucket decision, an
  `except` clause around backend selection, a routing decision), add a row to that page
  and a test that pins the behavior.
- **Generated Core ML artifacts and model weights are never committed.** `.mlpackage`,
  `.mlmodelc`, `.safetensors` and similar are gitignored; build or download them locally.
- **Benchmarks record raw data and are reproducible.** A new or changed benchmark follows
  [`docs/benchmarks.md`](docs/benchmarks.md): it writes raw per-request data, not only
  summary statistics, and states exactly how to reproduce it.
- **Public API changes follow `docs/api.md`.** The public API, the CLI, the stable
  `info()` keys, the routing-reason strings, and the artifact manifest format are covered
  by Semantic Versioning as of 1.0. Read the deprecation policy in
  [`docs/api.md`](docs/api.md) before changing or removing any of them.

## Artifact release policy

An ANE artifact configuration can ship only if it passes the same correctness gate as
the shipped ones. That applies to a new graph variant, a new bucket, a new compute-unit
setting, and any quantized or otherwise optimized export.

**The gate**, all on the tested profile:
- a 100% ANE compute plan with 0 device transitions;
- the runtime placement probe;
- the full parity gate against the upstream PyTorch FP32 goldens, with FP16
  probability error ≤ 0.02 and 0 hard mismatches.

**Other rules:**
- **Failed variants stay research.** A variant that fails, such as a 4-, 6- or 8-bit
  quantization, or a graph that is faster but numerically off, is recorded under
  `research/` with its raw data. It is never registered, offered or routed to.
- **Fixed shapes only.** The ANE runs fixed-shape buckets that each passed parity. A
  request that fits no validated bucket raises `UnsupportedShapeError`. Dynamic or
  enumerated shapes are not offered unless they are re-validated on the current runtime.
  On the tested profile they ran entirely on the CPU.
- **Routing changes need evidence.** A change to routing thresholds or auto buckets
  needs new measurements, derived with `laya_apple/derivation.py` and recorded in
  `CHANGELOG.md`.

## Commits and pull requests

- Write commit subjects in the imperative mood ("Add X", not "Added X" or "Adds X"),
  under about 70 characters.
- Explain *why* a change was made in the body, not just what changed; the diff already
  shows what changed.
- Keep a pull request to one logical change. Update `CHANGELOG.md` under `[Unreleased]`
  for any user-visible change.
- Before opening a PR: run the fast test suite, `ruff check .`, and `ruff format --check`.
  If your change touches parity, routing, placement or the artifact manifest, also run
  the relevant `scripts/derive_*.py --check` or `scripts/make_goldens.py --check`.
