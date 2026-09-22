# Experiment 4 — GPU / ANE crossover

**Question:** At what workload size does the GPU (MLX) become faster than the ANE on this
machine? Does the answer change with the number of questions or with batch size?

**Answer:**

- **One question per request:** the ANE is faster up to **L128** for the ModernBERT-large
  checkpoints (`laya`, `laya-typed-decisions`) and up to **L256** for `laya-multilingual`.
  - Between L128 and L256, the ModernBERT-large pair is within ±8%, with the GPU ahead at
    most points.
  - Above L256–L320 the GPU wins by an increasing margin: 1.5× at L512 and 2.4–2.8× at
    L1024.
- **Several questions per request:** the GPU wins almost everywhere, because MLX batches
  all questions into one forward pass. The ANE only keeps up at **L64 with a B=4 ANE
  export**: 1.02× on typed-decisions, 1.16× on multilingual.
- **The ANE's latency lead is small** where it exists: 1.16–1.29× over MLX on
  ModernBERT-large and 1.5–1.65× on multilingual. **Its value is as a second,
  independent device** (concurrency.md), not as a faster one.

Data:

- `raw/bench/latency.jsonl`: tags `pass-a` (matrix), `refine` (extra lengths), `batch`
  (B>1 ANE exports), `pass-b` (order control).
- Generated tables: [`reports/crossover.md`](reports/crossover.md) and
  `reports/crossover.json`, the machine-readable input for a future routing table.

All numbers are model-only `forward` P50 in ms, M4 Max, single process, warm. End-to-end
`predict` adds ≤ 1 ms on every backend (tokenisation, calibration, formatting), so the
crossovers are the same end-to-end. Pass B, run in reversed order, reproduced every MLX
and ANE value to within 1%.

Two "GPU" paths are shown because they turned out to be equivalent:

- MLX FP16;
- the Core ML ordinary graph with a **fixed** shape on `CPU_AND_GPU`, which passes
  parity. The enumerated-shape variant runs on CPU at 83–921 ms and is excluded.

"ANE" is the BC1S graph on `CPU_AND_NE`, B=1, the only ANE configuration that passes
parity.

## One question per request

### laya-typed-decisions (ModernBERT-large)

| L | MLX FP16 | Core ML GPU (fixed) | ANE | MLX / ANE | best |
|---:|---:|---:|---:|---:|---|
| 64 | 9.39 | 8.52 | **8.07** | 1.16 | ANE (1.06× over Core ML GPU) |
| 96 | 11.57 | 10.69 | **9.02** | 1.28 | ANE |
| 128 | 12.23 | 12.65 | **9.93** | 1.23 | ANE |
| 160 | **14.69** | — | 15.73 | 0.93 | GPU |
| 192 | **15.48** | — | 16.76 | 0.92 | GPU |
| 224 | 18.58 | — | **18.39** | 1.01 | tie |
| 256 | **19.23** | 20.05 | 20.07 | 0.96 | GPU |
| 512 | **35.21** | 36.19 | 54.43 | 0.65 | GPU |
| 1024 | **71.00** | 71.06 | 169.57 | 0.42 | GPU |

The ANE curve has a **step between L128 and L160**: 9.93 → 15.73 ms, +58% for +25%
tokens. After that it rises smoothly. The MLX curve is smooth. The crossover is set by
that step, not by a gradual convergence. Log-log interpolation over the coarse grid
alone gave "≈ L227", which is why the extra lengths were measured.

### laya (ModernBERT-large, max 512)

| L | MLX FP16 | Core ML GPU (fixed) | ANE | MLX / ANE | best |
|---:|---:|---:|---:|---:|---|
| 64 | 9.40 | 8.51 | **8.10** | 1.16 | ANE |
| 96 | 11.59 | 10.73 | **9.00** | 1.29 | ANE |
| 128 | 12.27 | 12.60 | **9.86** | 1.24 | ANE |
| 256 | **19.13** | 20.08 | 20.03 | 0.96 | GPU |
| 512 | **35.22** | 36.19 | 54.51 | 0.65 | GPU |

This is the same architecture as typed-decisions, and every cell agrees within 1%. The
refined typed-decisions boundary applies.

### laya-multilingual (mmBERT-base)

| L | MLX FP16 | Core ML GPU (fixed) | ANE | MLX / ANE | best |
|---:|---:|---:|---:|---:|---|
| 64 | 5.42 | 4.84 | **3.43** | 1.58 | ANE (1.41× over Core ML GPU) |
| 96 | 6.36 | 5.32 | **3.85** | 1.65 | ANE |
| 128 | 6.47 | 5.98 | **4.31** | 1.50 | ANE |
| 256 | 8.56 | 9.23 | **8.35** | 1.03 | ANE (marginal) |
| 320 | **10.06** | — | 11.86 | 0.85 | GPU |
| 384 | **11.69** | — | 13.92 | 0.84 | GPU |
| 448 | **13.40** | — | 18.14 | 0.74 | GPU |
| 512 | **14.97** | 14.96 | 22.75 | 0.66 | GPU |
| 1024 | **29.11** | 29.15 | 82.74 | 0.35 | GPU |

Crossover: **between L256 and L320**, about L264 by log-log interpolation. The smaller
model (d = 768, 12 heads) keeps the ANE ahead roughly twice as long as ModernBERT-large.

## Question count and batch size

Each question is its own encoder row, so a request with *q* questions is a batch of *q*
sequences.

- MLX batches them into one forward pass (`batch_size=64`).
- The ANE packages are fixed B=1 and run questions sequentially, unless a B>1 package is
  exported.

### B=1 ANE (sequential) vs MLX (batched)

| model | q | L64 | L96 | L128 | L256 | L512 | L1024 |
|---|---:|---:|---:|---:|---:|---:|---:|
| typed-decisions MLX / ANE | 4 | 0.58 | 0.73 | 0.84 | 0.79 | 0.57 | 0.39 |
| typed-decisions MLX / ANE | 8 | 0.51 | 0.65 | 0.79 | 0.74 | 0.55 | 0.38 |
| multilingual MLX / ANE | 4 | 0.60 | 0.71 | 0.81 | 0.74 | 0.53 | 0.32 |
| multilingual MLX / ANE | 8 | 0.50 | 0.62 | 0.68 | 0.68 | 0.51 | 0.31 |

A ratio below 1 means the GPU is faster. **With B=1 exports the GPU wins every
multi-question cell.**

### Batched ANE exports

These put B questions in one Core ML call. They were converted for L64/128/256 only.
Batched ANE parity was not run separately; only the FP32 layout check passed, at
≤ 1.8e-5.

| model | B = q | L | ANE batched | ANE sequential | MLX FP16 | Core ML GPU (B=1, sequential) | MLX / ANE-batched |
|---|---:|---:|---:|---:|---:|---:|---:|
| typed-decisions | 4 | 64 | **18.40** | 32.29 | 18.77 | 34.11 | **1.02** |
| typed-decisions | 4 | 128 | 41.59 | 39.67 | **33.16** | 50.56 | 0.80 |
| typed-decisions | 4 | 256 | 92.49 | 80.18 | **63.46** | 80.19 | 0.69 |
| typed-decisions | 8 | 64 | 40.82 | 64.60 | **32.73** | — | 0.80 |
| typed-decisions | 8 | 128 | 87.49 | 78.93 | **62.07** | — | 0.71 |
| typed-decisions | 8 | 256 | 207.19 | 159.90 | **118.66** | — | 0.57 |
| multilingual | 4 | 64 | **7.15** | 13.73 | 8.30 | 20.02 | **1.16** |
| multilingual | 4 | 128 | 15.23 | 17.01 | **13.86** | 23.93 | 0.91 |
| multilingual | 4 | 256 | 37.76 | 33.38 | **24.56** | 36.91 | 0.65 |
| multilingual | 8 | 64 | 14.03 | 27.36 | **13.65** | — | 0.97 |
| multilingual | 8 | 128 | 32.48 | 34.40 | **23.56** | — | 0.73 |
| multilingual | 8 | 256 | 92.55 | 66.70 | **45.11** | — | 0.49 |

- **Batching on the ANE only helps at L64.** At L128 it is neutral; at L256 it is
  *worse* than running the questions one at a time. The ANE is already saturated by a
  single ~128-token sequence, so extra rows add their full cost. The GPU still has
  headroom and amortises weight reads across the batch.
- **The crossover moves down as questions are added.** With q ≥ 4 the ANE's region
  shrinks from "L ≤ 128/256" to "L64, B=4 export" and disappears at q = 8.

## Crossover summary

| Model | q = 1 | q = 4 | q = 8 |
|---|---|---|---|
| laya-typed-decisions | ANE for L ≤ 128; tie band 160–256; GPU above | ANE only at L64 with a B4 export (1.02×); GPU otherwise | GPU everywhere |
| laya | same as typed-decisions (L ≤ 128) | not measured with a B4 export; B=1 sequential loses everywhere | GPU everywhere |
| laya-multilingual | ANE for L ≤ 256; GPU from L320 | ANE only at L64 with a B4 export (1.16×); GPU otherwise | GPU everywhere (0.97× at L64, B8) |

## Workload ranges → currently best backend (input for a deterministic `device="auto"`)

The criterion is latency of a single request on an otherwise idle machine, among
configurations that **pass parity**.

| Model | Questions per request | Longest row L | Best backend | Margin |
|---|---|---|---|---|
| laya, laya-typed-decisions | 1 | ≤ 128 | **ANE** (BC1S, `CPU_AND_NE`) | 1.16–1.29× over MLX, 1.06–1.27× over Core ML GPU |
| laya, laya-typed-decisions | 1 | 129 – 256 | GPU (MLX) | ≤ 8% either way; treat as GPU |
| laya, laya-typed-decisions | 1 | > 256 | **GPU** (MLX) | 1.5–2.4× |
| laya, laya-typed-decisions | 2–4 | ≤ 64 | tie (ANE B4 ≈ MLX) | 2% |
| laya, laya-typed-decisions | ≥ 2 | otherwise | **GPU** (MLX) | 1.2–2.6× |
| laya-multilingual | 1 | ≤ 256 | **ANE** | 1.03–1.65× |
| laya-multilingual | 1 | > 256 | **GPU** (MLX) | 1.2–2.8× |
| laya-multilingual | 2–4 | ≤ 64 | ANE with a B4 export | 1.16× |
| laya-multilingual | ≥ 2 | otherwise | **GPU** (MLX) | 1.1–3.1× |

This table is the evidence for a scheduler, not routing code. Three caveats a real
policy has to add:

1. **Load, not only latency.** concurrency.md shows the dominant gain comes from keeping
   short requests off a device that is busy with long ones, and from never putting long
   requests on the serial ANE. Under concurrent traffic, a request in the tie band should
   go to whichever device is idle.
2. **Energy was not measured.** A policy that trades latency for energy ("efficiency")
   has no data from this study.
3. **Bucketing.** Fixed-shape ANE packages pad each request up to the next bucket, and
   ANE cost is stepwise, with a jump after L128. A real ANE deployment for
   ModernBERT-large only needs buckets up to 128 (e.g. 64/96/128). For mmBERT, a 256
   bucket is also worth shipping.

## Other observations

- **Core ML GPU (ordinary fixed graph) is as fast as MLX** at L256 and above, within 0–8%
  from L256 to L1024 on both architectures. It is 7–16% *faster* than MLX at L64–L96. It
  is slower for multi-question requests only because it is a B=1 export run
  sequentially. A single framework (Core ML) could therefore serve both GPU and ANE at
  q = 1. MLX's measured advantages are batching and no per-length packages.
- **`ALL` is neither "GPU" nor "ANE".** With the BC1S package, `ALL` matched ANE timings
  up to L512. At L1024 it ran at 114 ms (typed) and 49.6 ms (multilingual), between the
  ANE (170 / 83) and GPU (99 / 49) figures for that graph. Core ML repartitions by length
  on its own. With the ordinary graph, `ALL` at short lengths behaves like the ANE path,
  and so is numerically wrong (parity.md).
- **Upstream PyTorch MPS is 1.4–2.3× slower than MLX** (typed L128: 21.8 vs 12.2 ms;
  L1024: 107.8 vs 71.0 ms). It is not a candidate backend.

## Limitations

- Idle-machine single-request latency only. The crossover under load is a scheduling
  question (concurrency.md).
- Refinement lengths were measured for q = 1 only. The batched-ANE comparison covers
  L64/128/256 and B ∈ {4, 8} only.
- These are one SoC and one OS build. The L128→L160 ANE step and the absolute
  crossovers are M4 Max / macOS 26.6.2 facts, not general truths. The ratios reported
  by the prior M3 Max / macOS 27.2 work (MLX 6.94 vs ANE 4.98 ms on multilingual L96)
  are in the same direction as the 6.36 vs 3.85 ms measured here.
