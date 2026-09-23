# Community benchmarks: the Apple Silicon matrix

Every number laya-apple ships was measured on one machine, an Apple M4 Max
([`compatibility.md`](compatibility.md)). This page collects results from other Macs,
each one produced by the same script and submitted as a pull request, so the matrix
fills in from measurements rather than expectations.

## Matrix

| SoC | MLX | ANE | Auto uses ANE | Heterogeneous | Evidence |
|---|---|---|---|---|---|
| M1 | ? | ? | ? | ? | |
| M1 Pro | ? | ? | ? | ? | |
| M1 Max | ? | ? | ? | ? | |
| M1 Ultra | ? | ? | ? | ? | |
| M2 | ? | ? | ? | ? | |
| M2 Pro | ? | ? | ? | ? | |
| M2 Max | ? | ? | ? | ? | |
| M2 Ultra | ? | ? | ? | ? | |
| M3 | ? | ? | ? | ? | |
| M3 Pro | ? | ? | ? | ? | |
| M3 Max | ? | ? | ? | ? | |
| M3 Ultra | ? | ? | ? | ? | |
| M4 | ? | ? | ? | ? | |
| M4 Pro | ? | ? | ? | ? | |
| M4 Max (macOS 26.6.2) | ✓ | ✓ | yes | ✓ | [`benchmarks/v1.0.md`](../benchmarks/v1.0.md) |

What each column means:

- **MLX**: MLX FP16 passes the parity gate against the shipped PyTorch FP32 goldens for
  every model measured.
- **ANE**: the ANE artifacts built on that machine pass the same gate. `untested` means
  no artifacts were built there.
- **Auto uses ANE**: `device="auto"` sends at least one request to the ANE. On a profile
  that is not shipped and not calibrated, the answer is `no` with `routing_reason`
  `platform_not_validated`. That is the runtime working as designed, not a failure.
- **Heterogeneous**: a short closed-loop mix of short and long requests has zero answer
  mismatches and more aggregate throughput with GPU + ANE than GPU alone.

`?` means nobody has submitted a result yet. It says nothing about whether that Mac
works.

## Add your Mac

You need a Mac with Apple Silicon, [uv](https://docs.astral.sh/uv/), disk space for the
checkpoints and, for the optional ANE step, 5–10 minutes.

1. Clone the repository and install it with the ANE and conversion extras:

   ```bash
   git clone https://github.com/tc3oliver/laya-apple
   cd laya-apple
   uv sync --extra ane --extra convert
   ```

2. Download the pinned checkpoint and verify its hash:

   ```bash
   uv run laya-apple download laya-typed-decisions
   ```

3. Optional: build the ANE artifacts on your machine.

   ```bash
   uv run laya-apple artifacts build laya-typed-decisions
   ```

   This converts the model for each ANE bucket and registers an artifact only if it
   passes the placement and parity gates on your Mac. It takes about 5–10 minutes. You
   can skip it: results for MLX alone are useful too, and the report then records the
   ANE column as `untested` together with this command.

4. Run the report:

   ```bash
   uv run python scripts/hardware_report.py
   ```

   The script records your SoC, memory, macOS version and build, the package versions
   and the laya-apple revision on its own. Then it runs, for each model, MLX parity,
   ANE parity when artifacts exist, warm latency, the routing decisions of
   `device="auto"`, and a short heterogeneous mix when the ANE is in use. Without
   `--models` it measures all three models, so download them all first
   (`uv run laya-apple download`) or pass `--models laya-typed-decisions`. `--quick`
   measures only `laya-typed-decisions` with fewer iterations.

5. Open a pull request that adds the directory the script prints,
   `hardware-results/<soc>-macos<major>/`, unchanged. Include the matrix row from
   `summary.md` in the description.

The script records no host name or serial number, and it replaces your home directory
with `~` and the checkout path with `.`. Read `bundle.json` before you submit it anyway.

## What reviewers check

- **The bundle is complete.** It has `bundle.json` and `summary.md`, every requested
  model has a section, and any step that did not run says why (`unavailable` with the
  build command, or `skipped` with a reason). Errors stay in the bundle; they are
  results too.
- **Parity.** The MLX and ANE parity summaries are the verdicts of the same gate
  `laya-apple parity` applies, with the per-row maxima recorded. A ✓ needs `passed:
  true` for every model measured.
- **No manual edits.** `summary.md` must render from `bundle.json`, the latency records
  must keep their raw samples, and the environment must match the directory name.
  Reviewers regenerate it with
  `uv run python scripts/hardware_report.py --render hardware-results/<dir>/bundle.json`
  and compare. Do not edit either file. If something looks wrong, say so in the pull
  request instead.
- **Provenance.** The laya-apple revision is a published commit, and the dirty flag is
  false or explained.

## How a result changes the matrix

A merged bundle fills in that SoC's row from its `summary.md`. If two bundles for the
same SoC disagree, the row shows both until the reason is understood.

A result does **not** change what `device="auto"` does on anyone's machine. The shipped
routing table (`laya_apple/data/routing.json`) applies only to the profiles it was
derived on, and a validated ANE result on another profile does not add that profile to
it. On your own machine, `laya-apple calibrate` writes a local profile that lets `auto`
use the ANE there ([`compatibility.md`](compatibility.md), "What happens on an untested
profile"). Adding a new shipped profile is a separate change that re-derives the table
from committed measurements. It is not done through this matrix.

The bundle format is described in [`hardware-results/README.md`](../hardware-results/README.md).
