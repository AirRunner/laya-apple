# Methodology

This page defines how every Phase -1 number was produced. The **parity tolerances**
(§5) were committed before any parity comparison ran. See the git history of this file.

## 1. Inputs

### Context-length generation

`scripts/common.py::make_request(tok, cfg, L, n_questions, seed)` builds requests whose
**longest prompt row is exactly `L` tokens**. It uses the same upstream-compatible
`build_sequence` the runtimes use: `[CLS] <type> instructions [SEP] [MASK] opt … [SEP] state [SEP]`.
The state grows one word at a time from a seeded stream of generated support-ticket
sentences until the longest row reaches `L`. If `L` cannot be hit exactly, the function
raises. It never pads or truncates to fake a length.

- Lengths: 64, 96, 128, 256, 512 and 1024, capped at each checkpoint's `max_len`
  (`laya` 512, `laya-multilingual` 1024, `laya-typed-decisions` 1024).
- With several questions, the other rows are a few tokens shorter. That is what real
  multi-question requests look like, because the question prefixes differ in length.
  Padded backends pad to `L`, and MLX pads to the batch maximum, which is also `L`.

### Question generation

There is a fixed pool of 8 generated questions: 4 `noul`, 2 `choice` (4 and 3 options)
and 2 `score` (3 and 4 levels). A request with *n* questions takes pool entries
`seed … seed+n-1` cyclically. No third-party fixture text is used.

### Parity fixtures (`scripts/fixtures.py`)

- **Synthetic, length-controlled:** for each valid `L` × seeds {11, 12, 13}, one request
  with all 8 pool questions, giving 24 rows per length.
- **Edge cases (hand-written):** nine languages (en, de, fr, es, zh, ja, hi, ru, ar), an
  empty state, a conversation list, literal mask tokens, structured (dict/JSON)
  instructions and criteria, a 5-level score, a custom-criteria noul, a 20-option choice,
  and a state that overflows `max_len` and is truncated by upstream's own rule.

## 2. Reference

The semantic reference is **unmodified upstream Laya** (`NandhaKishorM/laya@573e5b6`,
`laya.load(path, device="cpu")`): PyTorch CPU, FP32, Hugging Face ModernBERT with SDPA.
`scripts/reference.py` stores prompt items, unrounded decision logits, action logits and
the public result for every fixture in `raw/reference/<model>.json`. It also checks that
the Rust `tokenizers` path used by the MLX and Core ML ports produces identical token
ids to upstream's `transformers` tokenizer.

## 3. Backends

| Label | Implementation | Precision | Notes |
|---|---|---|---|
| `torch cpu` | upstream Agent | FP32 | reference |
| `torch mps` | upstream Agent | FP32 | upstream's own Apple GPU path |
| `mlx float32` / `mlx float16` | laya-mlx@0a85951 Agent | FP32 / FP16 | Metal GPU, eager, `batch_size=64` |
| `coreml … enumerated` | laya-coreml@4619e04 `convert()` defaults | FP16 | BxLxC graph, SDPA, EnumeratedShapes 16…max_len, B=1 |
| `coreml … fixed` | same converter, `flexible=False` | FP16 | one static-shape package per length |
| `ane …` | `scripts/ane_model.py` ConvBody | FP16 | B,C,1,L layout, 1×1 convs, per-head attention; host embedding lookup and action head |

Every Core ML configuration is measured separately under each `MLComputeUnits` setting:
`CPU_ONLY`, `CPU_AND_GPU`, `CPU_AND_NE` and `ALL`. They are never merged. `ALL` is not
treated as GPU, and `CPU_AND_NE` is not treated as ANE (see §6).

Fixed-shape Core ML packages serve variable-length requests through `Bucketed`. Each row
is padded to the smallest exported length that holds it, and the padding is masked. A row
longer than the largest export raises an error and is never truncated.

## 4. Timing

### Boundaries

- **model-only (`forward`)**: prepared token rows in, numpy float32 decision logits and
  action logits out, synchronised.
  - MLX: `mx.array` construction + forward + `mx.eval`.
  - torch: host→device copy + forward + `.cpu()`.
  - Core ML ordinary: `MLModel.predict` on int32 inputs.
  - ANE graph: host embedding gather + mask construction + `MLModel.predict` + host
    action head.
- **end-to-end (`predict`)**: public API call. Covers prompt construction, tokenisation,
  collation, inference, calibration and result formatting.
- Model loading, Core ML compilation and downloads are **never** inside warm timings.
  They are reported separately as cold-init (`load_s`, `first_call_ms`).

Timer: `time.perf_counter_ns`. One sample is one request.

### Warmup, samples, isolation

- Each (model × backend configuration) runs in a **fresh process**, and jobs run
  sequentially with no other model job running.
- Per workload cell there are 10 warmup calls, discarded, then a time-budgeted sample
  count: at least 50 and at most 300 samples, targeting about 6 s per cell. The actual *n*
  is stored with every cell.
- The benchmark matrix runs in **two passes with reversed backend order**, so slow drift
  (thermal, background) shows up as a pass-to-pass difference instead of as a backend
  difference. Both passes are kept.
- All raw per-sample timings are stored in `raw/**.jsonl`.
- Conditions are recorded per cell: load average, `pmset -g therm`, and power source.

### Statistics

The reported statistics are mean, standard deviation, P50, P95, P99, min and max.
Percentiles use numpy linear interpolation. P99 over fewer than 100 samples is close to
the maximum; it is shown, but read it that way. Throughput is `requests / Σ latency`,
not `1 / P50`.

### Memory

- MLX: `mx.get_peak_memory()` (allocator peak) and active/cache memory.
- torch MPS: current and driver-allocated bytes.
- All backends: process RSS after load and after the run, plus `ru_maxrss` peak.

These metrics are not interchangeable across frameworks and are not presented as if
they were.

## 5. Parity tolerances (fixed before measurement)

The comparison is against the reference logits and action logits for the **same
reference tokens**. Calibrated probabilities use the checkpoint temperatures clamped to
[0.5, 5.0], as upstream does.

| Quantity | FP32 backends | FP16 backends | Gated? |
|---|---|---|---|
| prompt tokens / markers | exact | exact | yes |
| calibrated probability, max \|Δ\| over options | ≤ 1e-4 | ≤ 0.02 | yes |
| action probability, max \|Δ\| | ≤ 1e-4 | ≤ 0.02 | yes |
| selected decision (argmax) | 0 mismatches* | 0 mismatches* | yes |
| all outputs finite | required | required | yes |
| repeated identical call → bitwise identical output | required | required | yes |
| logit max / mean \|Δ\|, relative error | reported | reported | no |

\* A mismatch counts against the gate when the reference top-1/top-2 calibrated margin
is **≥ 2 × the probability tolerance** (0.04 for FP16, 2e-4 for FP32). Inside that band,
a perturbation within tolerance can legitimately flip the argmax. Such "near-tie flips"
are listed row by row and never hidden.

Justification:

- 0.02 is the FP16 gate both prior ports fixed before measuring. It is about 3× the FP16
  drift they observed, and it is small compared with decision-relevant differences.
- Probability space is used because the temperatures differ per question bucket, which
  makes raw-logit error incomparable across rows.
- Logit error is reported rather than gated for the same reason.
- The action head is gated in probability space because its logits saturate: large logit
  errors can be invisible in probability. The raw action-logit error is reported so this
  stays visible.

## 6. Device verification

Requesting `CPU_AND_NE` only *permits* the ANE. ANE execution is claimed only with
evidence from at least two of:

1. **MLComputePlan** (`scripts/device_plan.py`): the preferred device and estimated cost
   of every MIL operation, plus the number of device transitions in program order. This
   is Core ML's anticipated placement, not a trace.
2. **Latency controls:** the same package under `CPU_ONLY`, `CPU_AND_GPU` and
   `CPU_AND_NE`. If `CPU_AND_NE` is much faster than `CPU_ONLY` while the GPU is
   excluded, work ran on the ANE.
3. **Instruments** (`xctrace`, *Core AI* template in Xcode 27) recording Neural Engine
   intervals while the workload runs (`scripts/trace_ane.py`).

CPU/GPU fallback is reported as the counts and cost share of non-ANE operations and as
transitions. It is never folded into an "ANE" label.

## 7. Known limitations

- One machine (M4 Max 16C/40G, 64 GB), one OS build (macOS 26.6.2 / 25G83), one session
  per experiment. No cross-SoC claim is made.
- Desktop session with normal background activity, on AC power. No clock pinning is
  possible on macOS.
- No energy measurement: `powermetrics` needs root, which was unavailable. Energy claims
  from prior work are not reproduced here.
- The parity fixtures test fidelity to upstream, not task accuracy (no labelled dataset
  was evaluated).
- Converted packages live on an external USB SSD (`<data>`). This affects
  cold-load time only; Core ML compiles into an internal-disk cache before warm inference.
