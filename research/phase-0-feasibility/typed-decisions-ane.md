# Experiment 2 — `laya-typed-decisions` on the Apple Neural Engine

**Question:** Can `convaiinnovations/laya-typed-decisions` run through Core ML on the ANE
with acceptable correctness? At which lengths, and what blocks it?

**Answer: yes, with a specific graph and strict runtime constraints.**

- The **channel-first BC1S graph** (`scripts/ane_model.py`) converts, compiles for the
  ANE, runs entirely on the ANE and passes decision parity at **every length from L64 to
  L1024**.
- The **simplest path**, laya-coreml's ordinary export, fails:
  - with flexible (enumerated) shapes it never reaches the ANE on this OS;
  - with fixed shapes it reaches the ANE up to L512 but produces **wrong decisions**
    there;
  - at L1024 its ANE compilation fails and Core ML silently falls back to a 2.2 s CPU path.

**Minimum feasibility gate:**

| Requirement | Result |
|---|---|
| Core ML compilation | **PASS** |
| L128 inference | **PASS** |
| Decision parity | **PASS** |
| ANE execution of meaningful graph regions | **CONFIRMED** (the whole transformer body) |

## The simplest Core ML path first

The task asks for the simplest correct path before any optimisation. That path is the
laya-coreml converter, unmodified: export-only PyTorch DecisionModel → TorchScript →
ML Program, FP16, B=1, K=32, SDPA attention.

### A. Enumerated shapes (laya-coreml's published default)

| Check | Result | Evidence |
|---|---|---|
| Conversion | PASS: 24 s, one package for lengths 16…1024 | `raw/conversion.jsonl` |
| Compilation / load | PASS: 4–9 s | smoke test and `raw/bench/latency.jsonl` |
| Inference | PASS | |
| Decision parity | **PASS**: max probability error 0.0016 | `raw/parity/laya-typed-decisions/coreml_units-cpu_gpu_enumerated-True.json` |
| ANE execution | **NO**: MLComputePlan puts 1648/1648 ops on **CPU** under every setting, including `CPU_AND_NE` and `ALL` | `raw/profile/plan-laya-typed-decisions-enumerated-<units>-L128.json` |
| CPU fallback | **Total**. E5RT logs `tensor_buffer has known strides while the model has FlexibleShapeInfo` | `raw/logs/*` |
| Latency | 218 ms at L128 and 917 ms at L512 (`CPU_AND_GPU` requested) | `raw/bench/latency.jsonl` |

**Blocker: flexible input shapes.** On macOS 26.6.2 with coremltools 9.0, the
enumerated-shape program is not partitioned onto the GPU or the ANE at all.

- laya-coreml reported the same package running on the GPU on macOS 27.2. **This
  behaviour depends on the OS version.**
- Whether a Swift/`MLMultiArray` caller avoids it was **not tested**, so the root cause
  (runtime vs Python binding) is inconclusive.

### B. Fixed shapes (same converter, `flexible=False`, one package per length)

MLComputePlan under `CPU_AND_NE` (`raw/profile/plan-laya-typed-decisions-ordinary-cpu_ne-L*.json`):

| L | ANE ops | CPU ops | GPU ops | ANE cost share | device transitions | parity | forward P50 |
|---:|---:|---:|---:|---:|---:|---|---:|
| 64 | 1084 | 42 | 0 | 73.5% | 6 | ❌ 0.19 | 7.39 ms |
| 96 | 1084 | 42 | 0 | 80.9% | 6 | ❌ 0.11 | 7.99 |
| 128 | 1084 | 42 | 0 | 84.4% | 6 | ❌ 0.42 | 9.73 |
| 256 | 1084 | 42 | 0 | 91.5% | 6 | ❌ 0.17 | 23.43 |
| 512 | 1084 | 42 | 0 | 93.7% | 6 | ❌ 0.31 | 65.03 |
| 1024 | **0** | 1093 | 0 | 0% | 0 | ✅ 0.0007 | **2182.64** |

Parity is the max calibrated-probability error against a gate of 0.02
(`raw/parity/laya-typed-decisions/coreml_units-cpu_ne.json`). 19 decisions changed in
total.

- **L64–L512:**
  - Conversion, compilation, loading and inference all succeed.
  - The graph runs **partitioned**: 6 CPU↔ANE transitions. Instruments shows about 3
    separate ANE intervals per call at L128 (1,946 intervals, 0.58 ms median).
  - The results are **wrong**.
  - The same packages are correct on `CPU_AND_GPU` (max 0.0028) and on `CPU_ONLY`.
- **L1024:**
  - `ANECCompile() FAILED`.
  - Core ML does not raise. It places the whole program on CPU: the plan shows 0 ANE
    ops, and Instruments shows **0 ANE intervals** (`raw/trace/typed-L1024-ordinary-cpu_ne`).
  - A call takes 2.18 s, 2.5× slower than the same package on explicit `CPU_ONLY`
    (878 ms).
  - It is correct only because it is not on the ANE.

**Blocker: ANE numerics of the BxLxC/SDPA graph.**

- *Which* operator introduces the error was not bisected. The error is large from L64,
  so it is not a length-dependent overflow.
- Silent fallback at L1024 is a second, independent blocker for any design that trusts
  `CPU_AND_NE` to mean "ANE".

## The ANE-specific graph (BC1S)

This graph was adapted from laya-coreml's multilingual-only ANE prototype. It uses
B,C,1,L activations, 1×1 convolutions for every projection, per-head attention with
einsums, explicit RoPE halves, and host-side embedding lookup and action head. It is
the original checkpoint weights, re-laid-out, with no retraining.

| L | Conversion | Compile/load (`CPU_AND_NE`) | Inference | Parity (max prob err, hard mismatches) | Plan: ANE ops / CPU / GPU / transitions | ANE runtime evidence | forward P50 | e2e P50 |
|---:|---|---:|---|---|---|---|---:|---:|
| 64 | PASS | 40.5 s | PASS | ✅ 0.0046, 0 | 10594 / 0 / 0 / 0 | plan + latency control | 8.07 | 8.16 |
| 96 | PASS | 39.1 s | PASS | ✅ 0.0046, 0 | 10594 / 0 / 0 / 0 | plan + latency control | 9.02 | 9.14 |
| **128** | PASS | 38.3 s | PASS | ✅ **0.0077, 0** | 10594 / 0 / 0 / 0 | **plan + Instruments + latency control** | **9.93** | **10.05** |
| 256 | PASS | 42.8 s | PASS | ✅ 0.0023, 0 | 10594 / 0 / 0 / 0 | plan + latency control | 20.07 | 20.28 |
| 512 | PASS | 44.7 s | PASS | ✅ 0.0036, 0 (1 near-tie flip) | 10594 / 0 / 0 / 0 | plan + latency control | 54.43 | 54.84 |
| 1024 | PASS | 42.8 s | PASS | ✅ 0.0024, 0 | see `raw/profile/plan-laya-typed-decisions-ane-cpu_ne-L1024.json` | **plan + Instruments + latency control** | 169.57 | 170.53 |

Sources:

- **Parity:** `raw/parity/laya-typed-decisions/ane_units-cpu_ne.json`. Rows are bucketed
  to the listed lengths, including the edge cases.
- **Plans:** `raw/profile/plan-laya-typed-decisions-L*-masked.json` (L64–512),
  `…-ane-cpu_ne-L1024.json` and `…-ane-all-L*.json`.
- **Timings:** `raw/bench/latency.jsonl` (pass A; pass B within 1%).
- **Conversion:** 30–35 s per length. The FP32 layout check against the export-only
  model is ≤ 8e-6 logit difference (`raw/conversion.jsonl`).

### Device-placement evidence (ANE is not assumed from `CPU_AND_NE`)

1. **MLComputePlan:** 100% of estimated cost on `MLNeuralEngineComputeDevice`, zero
   CPU/GPU operations, zero device transitions, at every measured length.
2. **Instruments (Xcode 27 *Core AI* template, attached to the workload PID):**
   - L128, recorded in the quiet timed queue under `ALL`, whose plan is 100% ANE at this
     length (`raw/trace/typed-L128-ane-all`): 639 `Neural Engine Prediction` intervals,
     median 9.32 ms, in a 6 s recording. At 9.9 ms per call about 606 calls fit in that
     window, so this is **≈ one ANE interval per prediction**.
   - L1024 under `CPU_AND_NE` (`raw/trace/typed-L1024-ane-cpu_ne`): 38 intervals, median
     161.6 ms, against a 169.6 ms call.
   - About 94–95% of each call is ANE execution.
   - An earlier L128 `CPU_AND_NE` trace (`raw/trace/typed-L128-ane-cpu_ne`: 772
     intervals, median 9.23 ms) was recorded while another process was running ANE
     probes. The hardware table is global, so its interval *count* is not attributable
     to this workload. Only its median duration is consistent with the result above.
3. **Latency control:** the same package under `CPU_ONLY` takes 27.1 ms at L128 and
   275 ms at L1024, 2.7× and 1.6× slower. It is also numerically different, as expected
   for a different engine. Under `CPU_AND_GPU` it takes 26.0 / 99.3 ms.

**CPU fallback:** none inside the graph. The CPU only does what is designed to run
there:

- tokenisation;
- embedding gather and mask construction, 0.14 ms at L128 and 4.1 ms at L1024;
- the 1024→256→2 action head and calibration.

End-to-end `predict` minus `forward` is ≤ 1 ms.

**GPU fallback:** none under `CPU_AND_NE`.

Under `ALL`, the plan (`raw/profile/plan-laya-typed-decisions-ane-all-L*.json`) keeps
L64–L256 100% on the ANE. At L512 and L1024 **Core ML moves the graph to the GPU**: 96%
and 97% of estimated cost, with 1 and 4 transitions. The L512 timing could not reveal
this, because ANE 54.4, `ALL` 54.2 and GPU 53.2 ms are nearly equal.

Parity still passes under `ALL`, but `ALL` does not mean "ANE" for this package.

### Batched exports

B=4 and B=8 packages at L64/128/256 convert and pass the FP32 layout check (≤ 1.8e-5).
They were benchmarked but **not parity-tested** against the goldens. Latency:
gpu-ane-crossover.md.

### Windowed variant (exact local attention)

Converts at L128–L1024 and passes parity (max 0.0077, 0 hard mismatches). It is 25%
faster at L1024. Its ANE compile/load grows to 536 s at L1024 (ane-long-context.md).

## Constraints under which typed-decisions ANE is viable

1. **Graph:** the BC1S layout. The ordinary graph must never be scheduled on the ANE.
2. **Shapes:** fixed, one package per length bucket, with requests padded up to the
   bucket and never truncated. Flexible shapes did not reach any accelerator on this OS.
3. **Compute units:** `CPU_AND_NE`. `ALL` is correct but moves L ≥ 512 to the GPU, so
   it is not an ANE setting. **Never `CPU_ONLY`**: the BC1S graph is
   numerically wrong there (max probability error 0.10).
4. **Compilation:**
   - Compile once to `.mlmodelc`.
   - Load from a stable path: 0.21 s in a fresh process, against about 38 s when loading
     the `.mlpackage`, which recompiles into a new temp path every time
     (`raw/coldstart/laya-typed-decisions-L128.json`).
   - The first compile of each bucket costs 35–45 s and must happen at install or
     first-use time, not per request.
5. **Worthwhile workload:** one question and L ≤ 128. Beyond that, MLX is equal or
   faster (gpu-ane-crossover.md). Long requests must not be sent to the ANE, because the
   ANE serialises and blocks short requests for 170 ms (concurrency.md).
6. **Verification:** every shipped package needs a recorded plan (0 transitions, 100%
   ANE) and a parity pass. A build that silently loses ANE placement still "works".

## What remains unknown

- Why the ordinary graph is wrong on the ANE (no operator bisection).
- Whether flexible shapes fail because of the runtime or the Python binding. Not tested
  from Swift.
- Behaviour on other SoCs or OS builds. The prior work on macOS 27.2 saw different
  flexible-shape placement.
- Energy per decision. Not measurable without root.
