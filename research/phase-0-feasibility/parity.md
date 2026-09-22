# Experiment 1 — Reference reproduction: parity

**Question:** Do the three Laya checkpoints produce upstream-equivalent decisions on every
Apple-native backend available here, at every valid context length?

**Answer:**

- **MLX (FP16 and FP32)** matches upstream on all three checkpoints at every length.
- **The channel-first ANE graph** matches upstream on all three checkpoints at every
  length on `CPU_AND_NE`, `ALL` and `CPU_AND_GPU`.
- **The ordinary laya-coreml graph is numerically wrong wherever the ANE executes it.**
  On multilingual the probability error reaches 1.0.
- **Both Core ML graphs are wrong on at least one compute-unit setting.** A runtime that
  lets Core ML "fall back" freely can silently change decisions.

Raw data: `raw/parity/<model>/<config>.json`, one per configuration with every row.
Generated table: [`reports/parity-matrix.md`](reports/parity-matrix.md). Goldens:
`raw/reference/<model>.json`. Method and pre-registered tolerances: methodology.md §5.

## Fixture

| Model | rows | synthetic (8 questions × 3 seeds × lengths) | edge cases |
|---|---:|---|---|
| laya | 163 | L64, 96, 128, 256, 512 | 9 languages, empty, conversation, mask literals, structured/5-level/custom/20-option, overflow |
| laya-multilingual | 187 | L64 … L1024 | same |
| laya-typed-decisions | 187 | L64 … L1024 | same |

- **Tokens:** every backend reproduced upstream's prompt ids and markers exactly
  (0 token-mismatch cases in 36 configurations). The Rust `tokenizers` path used by the
  ports equals upstream's `transformers` tokenizer on every fixture.
- **Determinism:** every configuration returned bitwise-identical logits and action
  logits on repeated identical calls.

## Support matrix (model × backend × length × correctness)

✅ passes the gate at that length. ❌ fails. ⚠ passes with a listed near-tie flip. Value =
max calibrated-probability |Δ| vs PyTorch CPU FP32. The FP16 gate is 0.02; FP32 is 1e-4.

### laya-typed-decisions (ModernBERT-large, 1024)

| Backend | L64 | L96 | L128 | L256 | L512 | L1024 | edge | Verdict |
|---|---|---|---|---|---|---|---|---|
| PyTorch MPS FP32 | ✅ 1.6e-6 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **PASS** (max 2.8e-6) |
| MLX GPU FP32 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **PASS** (max 3.4e-6) |
| MLX GPU FP16 | ✅ .0014 | ✅ | ✅ | ✅ | ✅ | ✅ .0004 | ✅ | **PASS** (max .0017) |
| Core ML ordinary, enumerated shapes (runs on CPU, see below) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **PASS** (max .0016) |
| Core ML ordinary fixed · CPU_AND_GPU | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | **PASS** (max .0028) |
| Core ML ordinary fixed · CPU_ONLY | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | **PASS** (max .0102) |
| Core ML ordinary fixed · CPU_AND_NE | ❌ .19 | ❌ .11 | ❌ .42 | ❌ .17 | ❌ .31 | ✅ .0007† | — | **FAIL**, 19 hard mismatches |
| Core ML ordinary fixed · ALL | ❌ .19 | ❌ .11 | ❌ .42 | ✅ | ✅ | ✅ | — | **FAIL**, 3 hard |
| **Core ML ANE graph · CPU_AND_NE** | ✅ .0046 | ✅ .0046 | ✅ .0077 | ✅ .0023 | ⚠ .0036 | ✅ .0024 | — | **PASS** (max .0077, 1 near-tie flip) |
| Core ML ANE graph · ALL | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | **PASS** (max .0077) |
| Core ML ANE graph · CPU_AND_GPU | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | **PASS** (max .0020) |
| Core ML ANE graph · CPU_ONLY | ❌ .10 | ❌ .09 | ❌ .07 | ❌ .06 | ❌ .07 | ❌ .06 | — | **FAIL** (0 hard mismatches, gate exceeded) |

† This L1024 package failed ANE compilation (`ANECCompile() FAILED` in the log, 134 s
load) and silently ran elsewhere. It is correct because it was *not* on the ANE. See
typed-decisions-ane.md.

### laya-multilingual (mmBERT-base, 1024)

| Backend | L64 | L96 | L128 | L256 | L512 | L1024 | edge | Verdict |
|---|---|---|---|---|---|---|---|---|
| PyTorch MPS FP32 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ NaN | **FAIL**: NaN output on a padded batch |
| MLX GPU FP32 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **PASS** (max 6.7e-6) |
| MLX GPU FP16 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **PASS** (max .0045) |
| Core ML ordinary, enumerated | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **PASS** (max .0017) |
| Core ML ordinary fixed · CPU_AND_GPU | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | **PASS** (max .0058) |
| Core ML ordinary fixed · CPU_ONLY | ✅ | ❌ .037 | ❌ .092 | ✅ | ✅ | ✅ | — | **FAIL** (0 mismatches, gate exceeded) |
| Core ML ordinary fixed · CPU_AND_NE | ❌ 1.0 | ❌ 1.0 | ❌ .999 | ❌ .986 | ❌ .998 | ❌ .994 | — | **FAIL**, 85 hard mismatches of 187 |
| Core ML ordinary fixed · ALL | ❌ 1.0 | ❌ 1.0 | ✅ | ✅ | ✅ | ✅ | — | **FAIL**, 42 hard |
| **Core ML ANE graph · CPU_AND_NE** | ✅ .0079 | ✅ .0128 | ✅ .0105 | ✅ .0066 | ✅ .0047 | ✅ .0029 | — | **PASS** (max .0128) |
| Core ML ANE graph · ALL | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | **PASS** (max .0128) |
| Core ML ANE graph · CPU_AND_GPU | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | **PASS** (max .0032) |
| Core ML ANE graph · CPU_ONLY | ❌ | ❌ | ❌ .20 | ❌ | ❌ | ❌ | — | **FAIL** |

### laya (ModernBERT-large, 512)

| Backend | L64 | L96 | L128 | L256 | L512 | edge | Verdict |
|---|---|---|---|---|---|---|---|
| PyTorch MPS FP32 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **PASS** (max 3.1e-6) |
| MLX GPU FP32 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **PASS** (max 1.0e-5) |
| MLX GPU FP16 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **PASS** (max .0037) |
| Core ML ordinary, enumerated | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **PASS** (max .0048) |
| Core ML ordinary fixed · CPU_AND_GPU | ✅ | ✅ | ✅ | ✅ | ✅ | — | **PASS** (max .0066) |
| Core ML ordinary fixed · CPU_ONLY | ✅ | ❌ .0202 | ✅ | ✅ | ✅ | — | **FAIL** (by 0.0002, 1 near-tie flip) |
| Core ML ordinary fixed · CPU_AND_NE | ❌ .26 | ❌ .27 | ❌ .32 | ❌ .41 | ❌ .56 | — | **FAIL**, 12 hard |
| Core ML ordinary fixed · ALL | ❌ .26 | ❌ .27 | ❌ .32 | ✅ | ✅ | — | **FAIL**, 6 hard |
| **Core ML ANE graph · CPU_AND_NE** | ✅ .0093 | ⚠ .0092 | ✅ .0122 | ✅ .0083 | ✅ .0056 | — | **PASS** (max .0122, 1 near-tie flip) |
| Core ML ANE graph · ALL | ✅ | ⚠ | ✅ | ✅ | ✅ | — | **PASS** (max .0122) |
| Core ML ANE graph · CPU_AND_GPU | ✅ | ✅ | ✅ | ✅ | ✅ | — | **PASS** (max .0043) |
| Core ML ANE graph · CPU_ONLY | ❌ | ❌ | ❌ | ❌ | ❌ .09 | — | **FAIL** |

**Edge cases vs fixed-shape Core ML.** Those configurations serve every row through
`Bucketed`, which pads each row to the next exported length. Their edge-case rows are
folded into the length columns, so "—" in the edge column does not mean untested. The
mixed-length `edge-questions` case (57–115 tokens) exercises padding to 64, 96 and 128
within one request.

## Error magnitudes (all rows)

| Configuration (typed-decisions) | logit max \|Δ\| | logit mean \|Δ\| | prob P99 \|Δ\| | action-logit max \|Δ\| |
|---|---:|---:|---:|---:|
| MLX FP16 | 0.022 | 0.0014 | 0.0014 | 22.8 |
| ANE graph · CPU_AND_NE | 0.053 | 0.0052 | 0.0046 | 22.1 |
| ordinary fixed · CPU_AND_GPU | 0.028 | 0.0016 | 0.0017 | 27.8 |
| ordinary fixed · CPU_AND_NE | 2.44 | 0.247 | 0.272 | 1924.6 |
| ANE graph · CPU_ONLY | 0.75 | 0.223 | 0.095 | 2239.2 |

The action head's logits are large and saturated. Every FP16 configuration shows
**action-logit errors of 20–60 while action probabilities match exactly** (max |Δ| = 0
everywhere a graph is otherwise correct). The gate is in probability space, as
pre-registered, so this does not fail anything. It does mean the `act_probability`
output would hide any real change in the action head. A future dataset-level check
should include cases where the action head is not saturated.

## Findings

### F1. The ordinary Core ML graph is numerically broken on the ANE

**Evidence:**

- Fixed-shape ordinary packages fail on `CPU_AND_NE` for all three checkpoints: 12, 19
  and 85 hard decision mismatches.
- Under `ALL`, the short lengths fail with the same magnitudes while L256+ pass. The
  per-length MLComputePlans (`raw/profile/plan-laya-typed-decisions-ordinary-*.json`)
  confirm this is placement:
  - Under `ALL`, L64–L128 place 1076 ops on ANE, 37 on GPU and 13 on CPU.
  - Under `ALL`, L256–L1024 place 0 ops on ANE and about 1060 on GPU. ANE compilation
    failed at those lengths (`ANECCompile() FAILED` in `raw/logs/plans.log`).
  - Under `CPU_AND_NE`, L64–L512 place 1084 ops on ANE and 42 on CPU, with 6 device
    transitions. L1024 places 0 on ANE and 1093 on CPU.

  **The graph is correct exactly where the ANE does not execute it.**
- The identical packages are correct under `CPU_AND_GPU`.
- Which operator introduces the error was **not** bisected for this graph.

**Result:** confirmed. The BxLxC/SDPA graph cannot be served on the ANE. laya-coreml
could never have observed this: on its OS its SDPA export was never placed on the ANE
at all.

### F2. The BC1S ANE graph is correct on the ANE, but wrong on CPU_ONLY

**Evidence:**

- It passes on `CPU_AND_NE`, `ALL` and `CPU_AND_GPU` for every checkpoint and length.
- It fails on `CPU_ONLY` (probability error 0.06–0.20) for every checkpoint.
- A separate root-cause investigation rejected the initial hypothesis (FP16 overflow in
  LayerNorm, where residual |x| reaches ~29,700 in ModernBERT-large):
  - The norm compiles to a native MIL `layer_norm` that behaves identically on all four
    unit settings, even at |x| = 29,700.
  - The error is already present at layer 0 (|x| ≈ 13).
  - It is isolated to FP16 1×1 conv/matmul on the Core ML CPU (BNNS) path.
  - The same conv converted with FP32 compute precision agrees on every unit setting
    (`raw/layernorm/`).

**Result:** confirmed. A runtime must never let this package fall back to `CPU_ONLY`.
If it must run without ANE/GPU, it needs an FP32-compute variant, which is untested at
full-model scale.

### F3. Enumerated-shape Core ML runs entirely on CPU on this OS, correct but slow

**Evidence:** the MLComputePlan for the published laya-coreml configuration
(EnumeratedShapes 16…max_len, B=1) is 1648/1648 operations on CPU under every
compute-unit setting. E5RT logs `tensor_buffer has known strides while the model has
FlexibleShapeInfo`. It is ~200 ms at L128 on typed-decisions in the smoke test; formal
latency is in gpu-ane-crossover.md.

**Result:**

- Correctness passes because it is the CPU FP16 path of a graph whose CPU numerics are
  acceptable.
- Whether this is macOS 26.6.2 + coremltools 9.0 behaviour or inherent to Core ML
  flexible shapes is **inconclusive**. It was not tested from Swift/`MLMultiArray`.
- laya-coreml measured this configuration on GPU on macOS 27.2. **The same artifact
  behaves differently across OS versions.**

### F4. Upstream PyTorch MPS emits NaN for a padded multilingual batch

**Evidence:**

- The `edge-questions` case batches three rows of 64, 71 and 57 tokens, so padding is
  present.
- On MPS FP32, laya-multilingual returns NaN logits for all three rows. CPU FP32 on
  identical tensors is finite.
- The ModernBERT-large checkpoints are unaffected on the same case.

**Result:** confirmed as an upstream PyTorch-MPS / mmBERT padding issue on torch 2.7.0.
The root cause was not investigated: it is out of scope for laya-apple, which does not
ship PyTorch.

### F5. MLX is the only backend with no failing configuration

**Evidence:** MLX FP32 is within 1e-5 and MLX FP16 within 0.0045 on every row of every
model, with 0 mismatches.

**Result:** confirmed. MLX FP16 is suitable as the Apple-native correctness baseline for
routine regression. PyTorch CPU FP32 remains the semantic reference.

## Answer to Experiment 1's parity questions

| Model | PyTorch (MPS) | MLX GPU | Core ML GPU (`CPU_AND_GPU`) | Core ML ANE (`CPU_AND_NE`) |
|---|---|---|---|---|
| laya | ✅ | ✅ FP16/FP32 | ✅ both graphs | ✅ **BC1S graph only** (ordinary ❌) |
| laya-multilingual | ❌ NaN on padded batch | ✅ FP16/FP32 | ✅ both graphs | ✅ **BC1S graph only** (ordinary ❌) |
| laya-typed-decisions | ✅ | ✅ FP16/FP32 | ✅ both graphs | ✅ **BC1S graph only** (ordinary ❌) |

## Limitations

- This is fidelity to upstream on 163–187 rows per model, not task accuracy. No labelled
  dataset was evaluated.
- Synthetic states are generated support-ticket text, so decisions are sometimes
  near-ties. Every near-tie flip is listed per configuration in the raw files. Among
  passing configurations only three occur, marked ⚠: reference margins 0.0022 and
  0.0035.
- The margin to the 0.02 FP16 gate is thinner for the ANE graph (max 0.0128) than for
  MLX FP16 (max 0.0045). A larger fixture could find a failing row. Any production ANE
  path needs continuous parity checking, not a one-time pass.
