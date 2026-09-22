# Experiment 3 — Why ANE latency grows with sequence length

**Model profiled:** `laya-typed-decisions`, ModernBERT-large. It is the plan's focus and
shares its architecture with `laya`. The BC1S graph runs cleanly on the ANE for this
model: 100% of planned cost on ANE, 0 device transitions, ≈ one ANE interval per call, and
parity passes at every length.

**Answer:** ANE latency grows superlinearly because **attention score work — QKᵀ, the
additive mask, softmax and A·V — is quadratic in L and runs at roughly 7× lower
effective throughput on the ANE than the projection and MLP matmuls.** Two properties of
the graph make it worse:

1. **Local-attention layers do full L×L work.** 18 of the 28 encoder layers are sliding
   window (±64 tokens), but the exported graph computes dense scores and masks them.
   Restricting them to their window, exactly, cuts full-body latency by 13% at L512 and
   25% at L1024.
2. **The score work is split into 16 per-head 64-channel contractions.** This is the
   ANE-friendly layout that makes the graph compile to ANE at all, but it is a poor fit
   for large score matrices.

It is **not** caused by CPU/GPU fallback, graph partitioning or host-side data movement
for this graph. Those were measured and are small or absent.

Scripts: `scripts/profile_probes.py`, `scripts/ane_model.py` (`Probe`, `_windowed`),
`scripts/device_plan.py`, `scripts/trace_ane.py`.
Raw data:

- `raw/profile/probes-laya-typed-decisions.jsonl`: 36 component probes with every sample
  and the compute plan for each.
- `raw/profile/plan-*.json`: full-body per-operator MLComputePlans.
- `raw/profile/host-prep-*.json`
- `raw/bench/latency.jsonl`: tags `pass-a`, `pass-b` and `windowed`.
- `raw/trace/*/summary.json`

## Baseline: full-body scaling

Forward P50 in ms, one question, B=1, from `raw/bench/latency.jsonl` pass A. Pass B
reproduced every ANE value to within 1%.

| L | ANE graph · CPU_AND_NE | × vs previous L | MLX FP16 (GPU) | × vs previous L | ANE graph · CPU_ONLY | ANE speed-up over its own CPU path |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 8.07 | | 9.39 | | 16.38 | 2.03× |
| 128 | 9.93 | 1.23 | 12.23 | 1.30 | 27.13 | 2.73× |
| 256 | 20.07 | 2.02 | 19.23 | 1.57 | 53.90 | 2.69× |
| 512 | 54.43 | 2.71 | 35.21 | 1.83 | 109.78 | 2.02× |
| 1024 | 169.57 | 3.12 | 71.00 | 2.02 | 275.18 | 1.62× |

The ANE's per-doubling growth factor rises towards 4, the signature of a quadratic term
taking over. MLX grows by at most 2× per doubling, i.e. linearly. The ANE's advantage
over the CPU running the same graph also shrinks with length.

## Component attribution

Each probe is a Core ML package built from the **real checkpoint weights** that runs one
component 4 times (`repeat=4`), fixed-shape, under `CPU_AND_NE`. Every probe was planned
100% on the ANE with 0 transitions, except `norm` at L128, which is planned on CPU (see
limitations). Values are **per single application (P50 ÷ 4)** in ms, 100 samples each.
Layers 1 and 3 are a local and a global layer.

| Component (per layer) | L128 | L256 | L512 | L1024 | ×(128→1024) | Scaling |
|---|---:|---:|---:|---:|---:|---|
| whole layer, global (idx 3) | 0.378 | 0.700 | 1.753 | 5.540 | 14.6 | superlinear |
| whole layer, local (idx 1), masked as exported | 0.370 | 0.695 | 1.740 | 5.558 | 15.0 | superlinear |
| attention incl. projections, global | 0.230 | 0.408 | 1.023 | 3.938 | 17.1 | ≈ quadratic tail |
| attention incl. projections, local masked | 0.218 | 0.398 | 1.013 | 3.928 | 18.1 | ≈ quadratic tail |
| **scores only: QKᵀ + mask + softmax + A·V** | 0.170 | 0.288 | 0.750 | 3.385 | **19.9** | quadratic |
| QKV + output projections | 0.133 | 0.258 | 0.468 | 1.030 | 7.8 | linear |
| MLP (GeGLU, 1024→5248→1024) | 0.208 | 0.380 | 0.870 | 1.870 | 9.0 | linear |
| LayerNorm | 0.055 | 0.118 | 0.243 | 0.458 | 8.2 | linear |
| **attention, local, windowed (exact)** | 0.230 | 0.400 | 0.743 | 1.610 | **7.0** | linear |

### Where the added time goes

Per layer, from the isolated probes. The components overlap in a fused layer, so shares
are of the component sum, not of the layer.

| Transition | Δ layer (global) | Δ scores | Δ projections | Δ MLP | Δ 2 norms | score share of Δ |
|---|---:|---:|---:|---:|---:|---:|
| L128 → L256 | +0.32 | +0.12 | +0.13 | +0.17 | +0.13 | 22% |
| L256 → L512 | +1.05 | +0.46 | +0.21 | +0.49 | +0.25 | 33% |
| L512 → L1024 | +3.79 | **+2.64** | +0.56 | +1.00 | +0.43 | **57%** |

Summing the component probes gives 0.62, 1.16, 2.57 and 7.2 ms per layer at L128 to
L1024. The measured whole-layer probe is 0.37, 0.70, 1.74 and 5.55 ms. The component
sums overstate because each isolated probe pays its own input/output overhead.

**Full-body closure.** 28 × 5.55 ms per layer = 155 ms, plus the two decision-head
layers at about 11 ms (estimated, see below), gives ≈166 ms. The measured forward P50 at
L1024 is 169.6 ms. At L128: 28 × 0.37 = 10.4 ms against 9.9 ms measured. The per-layer
probes therefore account for the full-body time within about 5%.

### Effective throughput: why the quadratic part is so expensive

FLOPs are counted from shapes: d = 1024, 16 heads × 64 channels, MLP intermediate 2624
with GeGLU.

| Operation at L1024, per layer | FLOPs | measured ms | effective TFLOP/s |
|---|---:|---:|---:|
| MLP: 2·L·(1024·5248 + 2624·1024) | 16.5 G | 1.87 | **8.8** |
| QKV + out projections: 2·L·(1024·3072 + 1024·1024) | 8.6 G | 1.03 | **8.3** |
| Scores: QKᵀ + A·V, 2 · 2·L²·1024 | 4.3 G | 3.39 | **1.3** |

The score contractions do a quarter of the MLP's arithmetic but take 1.8× its time.
They are 16 independent [64 × L] × [L × L] einsums, plus an L×L softmax over the key
axis. That shape uses the ANE far less efficiently than the large 1×1 convolutions. On
the GPU the same attention is a single fused `mx.fast.scaled_dot_product_attention`
call, and MLX's layer time stays linear to L1024.

## Hypotheses and evidence

### H1. Attention is responsible for most of the long-context scaling

**Evidence:**

- The score-only probe grows 19.9× from L128 to L1024, against 8× for the linear
  components.
- Score work goes from 22% of the per-layer increment (L128→L256) to 57% (L512→L1024).
- In the MLComputePlan cost estimate (`raw/profile/plan-*-masked.json`), conv's share
  falls from 94.5% at L64 to 42% at L512. Over the same range the attention elementwise
  and contraction ops grow: `add` (mask) 1.1%→17.5%, `mul` (RoPE/scale) 1.7%→14.5%,
  `einsum` 0.4%→7.2%, `softmax` <1%→5.7%.

**Result:** **confirmed for L ≥ 512.** Below L256, attention is not the bottleneck: at
L128 the MLP is the largest component (0.21 vs 0.17 ms for scores), and the linear
parts together are ~70% of the layer. **"Attention is the bottleneck" is true only for
long contexts.** For the short requests where the ANE is actually used, projections and
MLP dominate.

### H2. Local-attention layers compute unnecessary full-sequence work

**Evidence:**

- The exported graph builds the dense L×L score matrix for local layers and adds a
  −1e4 mask outside |i−j| ≤ 64. The masked local-attention probe costs the same as the
  global one at every length: 3.928 vs 3.938 ms at L1024.
- An exact block-local rewrite (`ConvAttention._windowed`) was built to test this. It
  uses 64-query blocks that only visit keys within ±64 of the block, with the same RoPE,
  mask values and padded-query rule.
  - It matches the dense graph in FP32 to ≤ 4.6e-6 logit difference at L128–L1024
    (`raw/conversion.jsonl`).
  - Local-attention probe: 3.93 → 1.61 ms at L1024 (−59%).
  - Full body, forward P50, windowed vs masked: L128 10.14 vs 9.93 (+2%), L256 19.76 vs
    20.07 (−2%), L512 47.52 vs 54.43 (−13%), L1024 127.56 vs 169.57 (−25%).
  - Parity of the windowed full body against the upstream goldens, with rows bucketed to
    128/256/512/1024: **PASS**. Max calibrated-probability error 0.0077, 0 hard
    mismatches, 1 near-tie flip at reference margin 0.0022, bitwise repeatable
    (`raw/parity/laya-typed-decisions/ane_units-cpu_ne_variant-windowed_*.json`).

**Result:** **confirmed.** The ModernBERT local-attention structure is not exploited by
the exported graph.

**Cost of the rewrite:**

- There is no gain below L256, because the window already covers most of the sequence.
- ANE compile/load time explodes, because the graph grows from 10.6k to 50.7k MIL ops at
  L1024: load takes 55 s at L128 and **536 s at L1024**, against ~43 s for the masked
  graph.
- Even windowed, L1024 on the ANE (127.6 ms) is 1.8× slower than MLX (71.0 ms). The
  global layers (10 of 28), plus the decision head's two full-attention layers, still do
  full L² work.

### H3. CPU or GPU fallback / graph partitioning causes the slowdown

**Evidence:**

- MLComputePlan under `CPU_AND_NE` for the masked BC1S body at L64, 96, 128, 256 and 512:
  10,594 operations, **100% estimated cost on ANE, 0 CPU/GPU operations, 0 device
  transitions**. The windowed body at L128–L1024 is the same (12.9k–50.7k ops).
- Instruments (`raw/trace/typed-L1024-ane-cpu_ne`): one "Neural Engine Prediction"
  interval per call, median **161.6 ms**, against a 169.6 ms wall-clock forward P50.
  **About 95% of the call is the ANE executing.** At L128
  (`raw/trace/typed-L128-ane-all`): 9.32 ms on the ANE against a 9.9 ms call.
- Under `ALL`, the same package runs L1024 at 114 ms, and `CPU_AND_GPU` at 99 ms. At long
  lengths Core ML moves work to the GPU when allowed. That is a deliberate choice, not a
  fallback, and it shows the ANE is not the fastest place for this graph at L1024.

**Result:** **rejected** for the BC1S graph. There is no fallback or partition boundary
at any length.

For contrast, the ordinary laya-coreml graph at L1024 failed ANE compilation
(`ANECCompile() FAILED`). Instruments recorded **0 ANE intervals**, and it fell back to a
2,183 ms path. At L128 it runs as about 3 ANE segments per call (1,946 intervals of
0.58 ms median in 6 s) with 6 device transitions. Fallback and partitioning are real
risks, but they belong to the other graph (see typed-decisions-ane.md).

### H4. Host-side input preparation and memory movement

**Evidence:**

- Host feature construction per call — embedding gather, two L×L FP16 additive masks,
  marker map — measured on CPU (`raw/profile/host-prep-laya-typed-decisions.json`):
  0.05 ms at L64, 0.14 at L128, 0.38 at L256, 1.21 at L512, 4.09 at L1024. That is 0.6%,
  1.4%, 1.9%, 2.2% and 2.4% of the call.
- The masks are 2 × L² × 2 bytes = 4 MiB per call at L1024 crossing into Core ML.
- The gap between wall-clock time and the traced ANE interval is about 8 ms at L1024 and
  about 0.6 ms at L128.
- End-to-end `predict` minus `forward` (tokenisation, calibration, formatting) is ≤ 1 ms
  at every length.

**Result:** **rejected as a main cause.** It is a small, avoidable quadratic term: masks
could be cached per length and patched for padding.

### H5. Projections, MLP and normalisation scale linearly

**Evidence:** growth from L128 to L1024 is ×7.8 (projections), ×9.0 (MLP) and ×8.2
(norm) for an 8× token increase. The effective matmul throughput of 8.3–8.8 TFLOP/s is
flat across lengths.

**Result:** **confirmed.** They are the dominant cost at short L and are not what
breaks long-context scaling.

### H6. Transpose, reshape, concat and split overhead

**Evidence:**

- The masked BC1S graph contains **no `transpose`**: the layout uses einsum and avoids
  them. `split` (per-head channel split) holds 6–8% of the estimated cost at every
  length. `concat` is not among the top operators.
- The windowed graph introduces `slice_by_index` (10.5%) and `transpose` (2.5%) of the
  estimated cost at L1024: its tiling overhead.

**Result:** **inconclusive for runtime share.** Core ML exposes no per-operator
*runtime* timing on the ANE, only plan estimates. The plan puts layout ops at under ~10%
and does not show them growing faster than L. Nothing indicates they drive the scaling.

### H7. Softmax specifically is the bottleneck

**Evidence:** softmax is included in the score-only probe, which grows 19.9×. The plan
attributes 5.7% (L512) to `softmax`, compared with 17.5% `add` and 7.2% `einsum`.
Softmax was not probed separately.

**Result:** **inconclusive.** The quadratic block as a whole is the bottleneck. Its
internal split between contraction, mask-add and softmax is not measured, only
estimated by the plan.

### H8. Embedding and decision head

**Evidence:**

- The embedding lookup runs on the host and is included in H4's 0.05–4 ms.
- The embedding norm is one LayerNorm (0.06–0.46 ms).
- The decision head (2 layers, full attention, ReLU FFN 4096) was not probed separately.
  The estimate of about 2 × 5.5 ms ≈ 11 ms at L1024 assumes a global-layer cost. This
  estimate is what makes the full-body closure add up.
- The scorer runs on at most 32 marker positions and is negligible.

**Result:** **partly measured.** The head's share is estimated, not measured. It is
full attention, so it scales like a global layer.

## What is known and what is not

**Known (measured):**

- The ANE body time is almost entirely ANE execution, with no fallback or partitions.
- The per-layer cost reproduces the full-body time within about 5%.
- The quadratic score block is 57% of the growth from L512 to L1024 and runs at about
  1.3 TFLOP/s, against about 8.5 for matmuls.
- Local layers waste most of their score work, and an exact windowed rewrite recovers
  25% at L1024.
- Linear parts dominate below L256.

**Not known:**

- the runtime split inside the score block (einsum vs mask-add vs softmax);
- per-operator ANE runtime (no tool exposes it);
- the decision head's measured share;
- whether other exact formulations beat the per-head einsum layout at long L, e.g.
  fewer, larger head groups, or chunked keys with online softmax;
- whether a newer deployment target (macOS 26 ops) changes any of this. Everything here
  targets macOS 15.

## Implications

- **ANE long-context "optimization" has a ceiling.** Even removing all local-attention
  waste leaves L1024 at 1.8× MLX. Matching the GPU at L512+ would need the global
  attention layers and the head to become much cheaper on the ANE. No exact technique
  measured here does that.
- The **exact windowed rewrite is worth keeping for L512–L1024 ANE packages only if
  those lengths are ever routed to the ANE.** gpu-ane-crossover.md and concurrency.md
  argue they should not be. Its 9-minute compile is a real cost.
- The actionable work for ANE value is at **short L**, where projections and MLP
  dominate. The candidates are:
  - weight compression (W8), which lowers weight traffic. The prior work saw ~2% speed
    on multilingual, not measured here.
  - avoiding the per-call L×L host masks.
  - batching at L64 (see gpu-ane-crossover.md).

## Limitations

- **Isolated probes vs in-graph placement.** Probes carry their own I/O overhead. One
  probe, `norm` at L128, was planned on CPU in isolation, although the same op is on the
  ANE inside the full body. That timing cell is CPU, not ANE.
- **Probe repetition.** Probe timings apply one layer's weights 4 times. A real body
  touches 28 different weight sets, so weight-fetch behaviour may differ.
- **Model coverage.** Profiling covered typed-decisions only. laya shares the
  architecture, and its full-body timings match typed-decisions within 1%.
  Multilingual (mmBERT-base, d = 768) was profiled only at full-body level. Its crossover
  and scaling shape are similar: ×2.7 for L256→512 and ×3.6 for L512→1024.
