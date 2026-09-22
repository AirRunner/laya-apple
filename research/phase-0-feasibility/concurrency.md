# Experiment 5 — GPU + ANE concurrency

**Question:** Can MLX on the GPU and Core ML on the ANE execute Laya requests at the same
time, with useful aggregate throughput and without unacceptable tail-latency damage?

**Answer:** Yes, and cleanly. Short requests went to the ANE and long requests to the
GPU, concurrently:

- **Neither stream lost measurable throughput or tail latency** (differences within ±1.1%).
- The heterogeneous split served the mixed workload at **3.3× (typed-decisions) / 3.1×
  (multilingual) the request rate of the best single-device configuration**.
- Putting both streams on one device causes **severe interference**:
  - On the GPU, short requests lose 60–73% of their throughput.
  - On the ANE, work is serialised: short requests wait behind long ones and lose 94%.

Script: `scripts/concurrency.py`. Raw data (every request's latency and start offset, per
window): `raw/concurrency/laya-typed-decisions-128-1024.json` and
`raw/concurrency/laya-multilingual-96-1024.json`.

## Setup

- **Streams.** Each stream is a **separate persistent process** that owns one backend,
  with the model loaded and warmed (10 calls) before any window. Each request is a
  closed loop: the next request is issued as soon as the previous one returns. Each
  request is one complete end-to-end `predict` with one question.
- **Timing windows.** Every window lasts 20 s. All participants in a window start at the
  same absolute monotonic time. There is a 2 s pause between windows.
- **Conditions per model.** Each stream runs alone; `ane_short + gpu_long` is the
  proposed heterogeneous split; `gpu_short + gpu_long` and `ane_short + ane_long` are
  same-device controls.
- **Cycles.** 3 cycles, with the order reversed on each alternate cycle. All windows are
  kept. Across cycles, request rates varied by less than 0.6%.
- **Workloads:**
  - typed-decisions: short = L128 (ANE BC1S graph, `CPU_AND_NE`), long = L1024 (MLX FP16).
  - multilingual: short = L96, long = L1024. There is no `ane_long` control for this model.
- **Why separate processes.** This is the best case for a runtime: no GIL and independent
  submission queues. A single-process threaded runtime was not measured and could be
  worse.

## Results: laya-typed-decisions (short L128, long L1024)

Request rate is the mean over 3 windows. Latency percentiles are pooled over all requests
of that condition. CPU is cores consumed by the process.

| Condition | Stream | req/s | P50 ms | P95 ms | P99 ms | max ms | CPU cores | RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ANE short alone | ane_short | 99.7 | 10.03 | 10.13 | 10.23 | 51.6 | 0.07 | 728 |
| GPU long alone | gpu_long | 13.9 | 71.70 | 71.86 | 72.25 | 84.9 | 0.06 | 947 |
| GPU short alone | gpu_short | 81.2 | 12.34 | 12.43 | 12.52 | 27.1 | 0.20 | 945 |
| ANE long alone | ane_long | 5.9 | 170.59 | 171.08 | 173.68 | 214.2 | 0.06 | 760 |
| **ANE short + GPU long** | ane_short | **100.4** | **9.95** | 10.04 | 10.21 | 41.4 | 0.07 | 728 |
| | gpu_long | **13.9** | **71.77** | 71.94 | 72.12 | 91.3 | 0.06 | 947 |
| GPU short + GPU long | gpu_short | 22.0 | 44.99 | 49.48 | 50.17 | 60.2 | 0.06 | 945 |
| | gpu_long | 12.4 | 80.75 | 81.45 | 81.68 | 93.4 | 0.05 | 947 |
| ANE short + ANE long | ane_short | 6.0 | 170.32 | 171.33 | 171.57 | 171.7 | 0.01 | 728 |
| | ane_long | 5.8 | 170.32 | 171.36 | 180.88 | 218.0 | 0.06 | 762 |

## Results: laya-multilingual (short L96, long L1024)

| Condition | Stream | req/s | P50 ms | P95 ms | P99 ms | max ms | CPU cores | RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ANE short alone | ane_short | 249.6 | 4.01 | 4.06 | 4.12 | 31.4 | 0.13 | 1277 |
| GPU long alone | gpu_long | 33.7 | 29.63 | 29.76 | 29.94 | 48.3 | 0.08 | 1171 |
| GPU short alone | gpu_short | 157.3 | 6.37 | 6.45 | 6.54 | 22.0 | 0.30 | 1169 |
| **ANE short + GPU long** | ane_short | **252.3** | **3.96** | 4.04 | 4.14 | 25.7 | 0.13 | 1247 |
| | gpu_long | **33.7** | **29.62** | 29.76 | 29.90 | 45.4 | 0.08 | 1171 |
| GPU short + GPU long | gpu_short | 63.4 | 16.07 | 17.16 | 17.58 | 31.3 | 0.13 | 1169 |
| | gpu_long | 28.9 | 34.65 | 34.97 | 35.11 | 49.5 | 0.07 | 1171 |

## Aggregate throughput

The workload is the same in every row: a short stream and a long stream, both issuing
requests as fast as they are served. One question per request, so requests/s equals
decisions/s.

| | typed-decisions req/s | multilingual req/s |
|---|---:|---:|
| **GPU + ANE (heterogeneous)** | **114.3** (100.4 + 13.9) | **286.0** (252.3 + 33.7) |
| GPU only (both streams on MLX) | 34.4 (22.0 + 12.4) | 92.3 (63.4 + 28.9) |
| ANE only (both streams on ANE) | 11.8 (6.0 + 5.8) | not measured |
| **max(GPU only, ANE only)** | 34.4 | 92.3 |
| **Heterogeneous / best single device** | **3.32×** | **3.10×** |

Weighted speedup is a mix-independent view: the sum over streams of each stream's rate
divided by its own solo rate.

| | typed-decisions | multilingual |
|---|---:|---:|
| GPU + ANE | **2.01** (1.007 + 1.000) | **2.01** (1.011 + 1.000) |
| GPU + GPU | 1.16 (0.27 + 0.89) | 1.26 (0.40 + 0.86) |
| ANE + ANE | 1.04 (0.06 + 0.98) | — |

A weighted speedup of 2.0 means both devices deliver their full solo throughput at once.

## Per-device degradation and tail latency

| Stream | Solo → with the other device busy | Throughput Δ | P99 Δ |
|---|---|---:|---:|
| typed ane_short (with GPU long) | 99.7 → 100.4 req/s | +0.7% | −0.2% |
| typed gpu_long (with ANE short) | 13.9 → 13.9 req/s | 0.0% | −0.2% |
| multilingual ane_short (with GPU long) | 249.6 → 252.3 req/s | +1.1% | +0.5% |
| multilingual gpu_long (with ANE short) | 33.7 → 33.7 req/s | 0.0% | −0.1% |
| typed gpu_short (with GPU long) | 81.2 → 22.0 req/s | **−72.9%** | **+301%** (12.5 → 50.2 ms) |
| typed ane_short (with ANE long) | 99.7 → 6.0 req/s | **−94.0%** | **+1577%** (10.2 → 171.6 ms) |

The heterogeneous rows lie inside cycle-to-cycle noise. No tail-latency degradation was
measured: P99 stays within 0.5% and the maxima do not grow. The few outliers of 25–90 ms
appear in solo windows as well.

## Findings

### H1. GPU and ANE are independent resources for Laya: confirmed

**Evidence:** in both models, both streams keep their solo throughput and P99 when run
together, across 3 alternating cycles.

**Result:** confirmed on this machine for the tested pair. The two engines do not
visibly contend for memory bandwidth at these request sizes. That is plausible: the
L1024 GPU stream moves about 0.8 GiB of weights per request, 14 times a second, which is
far below M4 Max bandwidth.

**Not tested:**

- saturating GPU workloads, such as large batches or the other Laya model at the same
  time;
- sustained load for more than 20 s per window, and thermal steady state;
- energy.

### H2. The ANE is a single serial queue: confirmed

**Evidence:** with short and long streams on the ANE at the same time, the short stream's
latency becomes the long stream's latency: 170 ms P50 for an L128 request whose solo
latency is 10 ms. Each stream gets roughly alternating service. This is consistent with
the ANE executing one prediction at a time. Instruments shows ≈ one "Neural Engine
Prediction" interval per call (`raw/trace/`).

**Consequence:** routing a long request to the ANE is harmful even when the ANE is idle
at that moment. It blocks every short request that arrives during its 170 ms. The
scheduler must bound ANE work per request, not only choose the faster device.

### H3. Head-of-line blocking is the main cost of single-device serving

**Evidence:** on the GPU the long stream loses only 11–14%, but the short stream loses
60–73% and its P99 rises from 12.5 to 50 ms. The likely mechanism is that GPU work from
the two processes is time-sliced, so short requests queue behind long kernels. This
mechanism is inferred, not traced.

**Result:** confirmed. Keeping short requests off the device that runs long requests is
worth more than any single-device speedup measured in this study.

### H4. Host CPU is not a bottleneck

**Evidence:** every stream consumed at most 0.3 CPU cores (measured with
`process_time`, all threads). The host-side ANE feature construction is 0.14 ms at L128
(`raw/profile/host-prep-*.json`).

**Result:** confirmed for single-question requests.

## Answer

| Criterion | typed-decisions | multilingual |
|---|---|---|
| Aggregate throughput vs best single device | **+232% (3.32×)** | **+210% (3.10×)** |
| Throughput degradation, ANE stream | none (+0.7%) | none (+1.1%) |
| Throughput degradation, GPU stream | none (0.0%) | none (0.0%) |
| Tail-latency degradation | none measurable | none measurable |
| Memory | two resident copies of the weights: about 0.7 GiB ANE package + 0.95 GiB MLX per model | same (1.2 + 1.2 GiB RSS) |

**Request-level heterogeneous execution is technically worthwhile.** It is the strongest
positive result of Phase -1. The gain comes from two things:

- device independence: the two engines don't slow each other;
- isolation: short requests are never stuck behind long ones.

Raw ANE speed contributes little. On typed-decisions, the ANE's solo advantage at L128
is only 1.23×.

## Limitations

- These are separate processes. A threaded single-process runtime, with GIL, Core ML and
  MLX threads sharing one process, was not measured and must be measured before v0.3
  commits to an in-process design.
- Each stream is a closed loop with one request in flight. Open-loop arrivals with
  queueing, bursts and multi-question requests were not tested.
- 20 s windows × 3 cycles. Thermal behaviour over minutes was not characterised.
  `pmset -g therm` reported no warning at any point.
- Energy was not measured (`powermetrics` needs root). A throughput gain says nothing
  about joules per decision.
