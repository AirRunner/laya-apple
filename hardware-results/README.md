# Hardware results

One directory per submitted run of `scripts/hardware_report.py`. The matrix built from
them, and how to add your Mac, is in
[`docs/community-benchmarks.md`](../docs/community-benchmarks.md).

## Directory name

`<soc>-macos<major>`, e.g. `apple-m4-max-macos26`: the SoC brand string, lower-cased
with every run of non-alphanumerics replaced by `-`, then the macOS major version. If
that directory already exists, the script adds `-<YYYYmmdd-HHMMSS>` rather than
overwrite it.

## Files

- `summary.md`: the human-readable report. It starts with the matrix row (MLX, ANE,
  Auto uses ANE, Heterogeneous), then the environment, then one section per model.
  It is generated from `bundle.json`; `scripts/hardware_report.py --render
  <dir>/bundle.json` reproduces it.
- `bundle.json`: everything, with raw samples. Top-level keys:

  | Key | Contents |
  |---|---|
  | `format`, `format_version` | `laya-apple-hardware-report`, `1` |
  | `started_at`, `wall_time_s` | UTC start time, total run time |
  | `environment` | SoC brand string, `hw.model`, memory, macOS version and build, Python, MLX, coremltools and NumPy versions, laya-apple version with git revision and dirty flag |
  | `platform` | the platform profile routing uses, whether a shipped routing profile matches it, and the local calibration profile, if any |
  | `configuration` | models, quick or full, warm-up and iteration counts, the latency and routing shapes, the heterogeneous window length, and that latency was measured in-process |
  | `models.<name>` | pinned revision, pinned and verified weight SHA-256, then `mlx_parity`, `ane_parity`, `mlx_latency`, `ane_latency`, `routing`, `heterogeneous` |
  | `matrix`, `matrix_row` | the four cells and the table row |

  Each step has a `status`: `ok`, `error` (with the exception and traceback),
  `unavailable` (ANE artifacts missing or rejected, with the reason and the build
  command) or `skipped` (with a reason). Latency records are those of
  `laya-apple benchmark`, with `samples` holding every timed iteration. The
  heterogeneous record keeps the full closed-loop windows of
  `scripts/bench_concurrency.py` part A, per-request latencies included.

Paths are rewritten before anything is written: the checkout becomes `.` and the home
directory `~`.
