# Phase -1 conclusion

All numbers come from one Apple M4 Max (16 CPU / 40 GPU cores, 64 GB) on macOS 26.6.2,
with the pinned stack in environment.md. Each claim links to the experiment document
holding its evidence. Anything not measured is labelled as such.

## Model support

"Works" means it runs **and** passes the pre-registered parity gate (methodology.md §5)
at every valid length.

| Model | PyTorch (upstream) | MLX GPU | Core ML GPU | Core ML ANE |
|---|---|---|---|---|
| `laya` (ModernBERT-large, 512) | ✅ CPU and MPS FP32 | ✅ FP16 and FP32, L64–512 | ✅ fixed-shape ordinary graph on `CPU_AND_GPU`, L64–512 | ✅ **BC1S graph only**, `CPU_AND_NE`, L64–512 |
| `laya-multilingual` (mmBERT-base, 1024) | ✅ CPU; ❌ **MPS returns NaN** on a padded multi-row batch | ✅ FP16 and FP32, L64–1024 | ✅ fixed-shape ordinary graph, L64–1024 | ✅ **BC1S graph only**, L64–1024 |
| `laya-typed-decisions` (ModernBERT-large, 1024) | ✅ CPU and MPS FP32 | ✅ FP16 and FP32, L64–1024 | ✅ fixed-shape ordinary graph, L64–1024 | ✅ **BC1S graph only**, L64–1024 |

The fixed-shape Core ML GPU graph matches MLX speed (within 0–8% at L≥256, faster at
L≤96).

Configurations that do **not** work:

- **laya-coreml's published enumerated-shape export**:
  - Correct, but runs entirely on CPU on this OS under every compute-unit setting:
    1648/1648 ops planned on CPU.
  - Latency is 83–921 ms, 14–25× slower than the same graph with a fixed shape (parity.md
    F3, gpu-ane-crossover.md).
- **The ordinary (BxLxC/SDPA) graph on the ANE** (`CPU_AND_NE`, and `ALL` at short
  lengths) is **numerically wrong** for all three models:
  - Up to 85 of 187 decisions changed on multilingual.
  - At L1024 it failed ANE compilation and **silently fell back** to a 2.2 s path, with 0
    ANE intervals in Instruments.
- **The BC1S ANE graph under `CPU_ONLY`** is wrong (probability error 0.06–0.20). The
  cause is FP16 conv/matmul precision on the Core ML CPU path. The first hypothesis,
  FP16 LayerNorm overflow, was tested and rejected (parity.md F2).

## Correctness

| Passes parity (all lengths) | Fails | Why it fails |
|---|---|---|
| PyTorch MPS FP32: laya, typed-decisions | PyTorch MPS: multilingual | NaN on a padded 3-row batch; torch 2.7.0 MPS issue, not investigated further |
| MLX FP32 and FP16: all three | — | — |
| Core ML ordinary fixed, `CPU_AND_GPU`: all three | Core ML ordinary fixed, `CPU_AND_NE` / `ALL`: all three | ANE numerics of this graph; not bisected to an operator |
| Core ML ordinary enumerated: all three (on CPU) | Core ML ordinary fixed, `CPU_ONLY`: laya (by 0.0002), multilingual | CPU FP16 precision |
| Core ML BC1S, `CPU_AND_NE` / `ALL` / `CPU_AND_GPU`: all three | Core ML BC1S, `CPU_ONLY`: all three | CPU FP16 conv precision |
| Core ML BC1S windowed, `CPU_AND_NE`: typed-decisions | — | — |

**Margins.**

- The ANE path passes with less room than MLX: max probability error 0.0077 on
  typed-decisions and 0.0128 on multilingual, against 0.0017–0.0045 for MLX FP16 and a
  gate of 0.02.
- FP16 action-head *logits* differ by 20–60 while action *probabilities* agree exactly,
  because the head is saturated. A probability-level gate cannot see changes there.

**Consequence for the runtime.** Core ML must never be allowed to "fall back" between
compute units on its own. Each Laya Core ML graph is correct only on specific units, and
the failure is silent.

## typed-decisions on the ANE

**Technically viable: yes.** The minimum feasibility gate passes, and the result extends
to L1024 (typed-decisions-ane.md).

| Gate item | Result |
|---|---|
| Core ML conversion | PASS: BC1S graph at L64–1024, B ∈ {1, 4, 8} |
| Core ML compilation (ANE) | PASS: ~38–45 s per length for the masked graph, 55–536 s for the windowed graph |
| L128 inference | PASS: 9.93 ms model-only, 10.05 ms end-to-end |
| Decision parity | PASS: every length, max probability error 0.0077, 0 hard mismatches |
| ANE executes the graph | CONFIRMED on three independent sources (below) |

The three sources of ANE evidence:

1. MLComputePlan: 100% of estimated cost on ANE, 0 CPU/GPU ops, 0 transitions.
2. Instruments *Core AI*: ≈ one "Neural Engine Prediction" interval per
   call, covering ~94–95% of the call.
3. Latency control: 2.7× faster than the same package on `CPU_ONLY`.

**Constraints:**

1. Only the channel-first BC1S rewrite is correct on the ANE. The ordinary export is not.
2. Fixed shapes only, one package per length bucket. Flexible shapes run on CPU on this OS.
3. The runtime must pin compute units to `CPU_AND_NE` and refuse `CPU_ONLY` for this
   package. `ALL` is correct, but it plans L ≥ 512 onto the GPU, so it is not an ANE
   setting.
4. The package must be compiled once and loaded as a `.mlmodelc` from a stable path:
   0.21 s load, against ~38 s per process when loading the `.mlpackage`.
5. Latency is competitive only for one question at L ≤ 128.

## Long context

**The measured ANE bottleneck** (ane-long-context.md):

- **Confirmed:**
  - ANE time grows superlinearly: ×3.1 per doubling at L512→1024, against ×2.0 for MLX.
  - The growth comes from the quadratic attention-score block (QKᵀ, mask, softmax,
    A·V). It accounts for 57% of the per-layer increment from L512 to L1024, and it runs
    at an effective **1.3 TFLOP/s on the ANE, against ~8.5 TFLOP/s for projections and
    MLP**.
  - The exported graph also does full L×L score work in the 18 sliding-window layers.
    An exact windowed rewrite passes parity and saves 13% at L512 and 25% at L1024. At
    L1024 it is still 1.8× slower than MLX.
  - The per-layer probes reproduce the full-body time within ~5%.
- **Rejected:** CPU/GPU fallback and graph partitioning (0 transitions, ≈ one ANE interval
  per call), and host-side preparation (≤ 2.4%, ≤ 4.1 ms at L1024).
- **Not the bottleneck at short L:** at L128 projections and MLP are ~70% of a layer.
  "Attention is the ANE bottleneck" holds only for L ≥ 512.
- **Hypotheses left open:**
  - how the score block splits between einsum, mask-add and softmax;
  - per-operator ANE runtime, which no tool exposed;
  - the decision head's share, which is estimated, not probed;
  - whether a different exact attention layout, or a newer deployment target, improves
    long L.

## GPU / ANE crossover

Measured on this machine (gpu-ane-crossover.md); single idle request, model-only ≈
end-to-end:

| Model | 1 question | 4 questions | 8 questions |
|---|---|---|---|
| laya, laya-typed-decisions | ANE ≤ **L128** (1.16–1.29× over MLX). Tie band L160–256. GPU above (1.5× at L512, 2.4× at L1024) | GPU, except a tie at L64 with a B4 ANE export | GPU everywhere |
| laya-multilingual | ANE ≤ **L256** (1.03–1.65×). GPU from L320 (2.8× at L1024) | GPU, except L64 with a B4 export (1.16×) | GPU everywhere |

The ANE curve for ModernBERT-large jumps +58% between L128 and L160. The crossover is a
step, not a smooth convergence.

## Concurrency

**Yes.** MLX GPU and Core ML ANE execute Laya requests concurrently, with no measurable
interference (concurrency.md). Short requests went to the ANE (L128 / L96) and long
requests (L1024) to the GPU, as separate processes:

- **Aggregate request rate: +232% on typed-decisions (3.32×) and +210% on multilingual
  (3.10×)** over the best single device serving the same mix.
- Per-device throughput change: **0.0% to +1.1%**. P99 change: **within ±0.5%**.
- Same-device controls:
  - GPU + GPU costs short requests 60–73% of their throughput, and P99 rises from 12.5 to
    50 ms.
  - ANE + ANE serialises: short requests wait behind long ones, losing 94%, with P99 at
    171 ms.

Not measured: a single-process threaded runtime, open-loop or bursty arrivals, windows
longer than 20 s, and energy.

## Architecture decision

**Continue with MLX GPU + Core ML ANE + `device="auto"` + request-level heterogeneous
scheduling, but narrow the role of each part.** The evidence changes the design in six
places.

1. **The ANE is a short-request accelerator and an isolation device, not a faster
   general backend.**
   - Its latency lead is 1.2–1.6× and exists only for one question at L ≤ 128 (L ≤ 256
     on mmBERT).
   - Its lasting value is as a second device that keeps short requests away from the
     GPU (3.1–3.3× aggregate).
   - **"Long-context ANE optimisation" is demoted from a milestone to a research
     track**: the best exact rewrite measured still leaves L1024 at 1.8× MLX.
2. **The Core ML backend ships two graphs, one per device:**
   - the BC1S graph for the ANE, which is required for correctness on the ANE;
   - the fixed-shape ordinary graph for Core ML GPU, which is faster there than BC1S:
     71 vs 99 ms at L1024.

   Both use fixed-length buckets only, compiled once to `.mlmodelc`. Enumerated/flexible
   shapes are not used.
3. **Compute units are pinned per graph, and fallback is an error.**
   - BC1S may run on `CPU_AND_NE` only (`ALL` is correct but leaves the ANE at L ≥ 512),
     never `CPU_ONLY`.
   - The ordinary graph may run on `CPU_AND_GPU`, never `CPU_AND_NE`.
   - An ANE compile failure must raise, not silently degrade to a 2 s CPU path. This
     makes `strict_device` a correctness requirement, not a benchmarking nicety.
4. **`device="auto"` v0 is a small measured table.** It is keyed on model family,
   question count and longest row: ANE only for q = 1 and L ≤ 128 (ModernBERT) or
   L ≤ 256 (mmBERT); GPU otherwise. Load-awareness (next point) matters more than the
   crossover itself.
5. **The scheduler's first job is isolation, not prediction.**
   - It keeps long requests off the ANE entirely, because the ANE serialises.
   - It sends short requests to the ANE even in the tie band when the GPU is busy.
   - A cost model is not needed for v0.3.
6. **MLX stays the GPU backend** for its batching (1.1–2.4× faster than sequential Core ML
   on multi-question requests) and for having no per-length packages. The Core ML GPU
   graph is a measured, correct alternative. It is not dropped from the plan, but it is
   not required for v0.1.

## What this study does not establish

- **Energy.** No measurement was possible without root. Every "efficiency" or "battery"
  claim in the plan remains unsupported.
- **Other hardware and OS builds.** Crossovers, the L128 ANE step and the enumerated-shape
  CPU fallback are M4 Max / macOS 26.6.2 facts. The prior M3 Max / macOS 27.2 work saw
  enumerated shapes run on the GPU.
- **Task accuracy.** Parity is fidelity to upstream on 163–187 rows per model, not
  accuracy on a labelled dataset.
- **In-process concurrency.** Only separate processes were measured.
