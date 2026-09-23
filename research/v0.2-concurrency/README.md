# v0.2 step 1 — single-process GPU + ANE concurrency gate

**Question.** Phase -1 measured GPU + ANE concurrency only across separate processes
(3.10–3.32× the best single device). Does it hold when one application process drives both
devices, and if not, what is the smallest execution model that keeps it?
(DEVELOPMENT_PLAN.md §12.3: "Do not assume threads work, and do not assume processes are
required.")

**Answer [Measured].**

- **Step 1.** Threads in one process do **not** meet the exit gate: the GPU stream loses
  9–11% of its throughput, and P99 grows 11–23% on one of the streams. Giving **either**
  device its own process removed the interference completely (all changes within ±2.4%).
- **Qualification from the product implementation (section "Request-driven execution"
  below).** Those process results hold only when each worker drives its own closed loop.
  When requests reach a device from another process, as they do in any real server,
  running both devices at once slows the IPC-driven device's host-side work 4–6×. None of
  the seven request-driven designs tested meets "P99 within 10% of solo".
- **Decision.** The GPU always runs in a worker process. The ANE runs either on a thread in
  the caller's process or in its own process, chosen per model from measurement:
  - thread for the ModernBERT-large models;
  - process for mmBERT-base, whose roughly 250 short requests/s make Core ML's GIL-holding
    `predict` costly in-process.

  See `benchmarks/v0.2.md`.

## Setup

- Apple M4 Max, macOS 26.6.2 (25G83), Python 3.12, MLX 0.32.2, coremltools 9.0,
  laya-apple 0.1.0 (the product runtime, not the research harness).
- Streams, closed-loop (next request as soon as the previous returns), 1 question each:
  - `ane_short`: `Laya(device="ane")`, BC1S on `CPU_AND_NE`, exact-length L128 (typed) /
    L96 (multilingual);
  - `gpu_long`: `Laya(device="gpu")`, MLX FP16, exact-length L1024;
  - `gpu_short`: MLX FP16 at the short length (same-device control).
- Each stream issues full `Laya.predict` calls (prompt build, routing, forward, answer
  formatting) unless stated. 20 s windows, 3 cycles, order reversed on odd cycles, 2 s
  settle. Medians over cycles are reported. Answers are compared with each stream's
  first answer and with its solo reference; any change is flagged.
- Scripts: `scripts/inproc.py` (gate), `scripts/followup.py` (cause),
  `scripts/summarize.py` (tables). Raw windows with every latency sample: `raw/`.

## Gate result

| Model / mix | Execution | ane_short req/s (Δ solo) | ANE P99 Δ | gpu_long req/s (Δ solo) | GPU P99 Δ | Aggregate vs GPU-only | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| typed-decisions L128 + L1024 | threads, one process | 98.2 (−1.2%) | **+12.9%** | 12.4 (**−10.8%**) | **+12.5%** | 3.48× | **fail** (P99) |
| | two processes | 100.2 (+0.8%) | +1.2% | 13.9 (0.0%) | −0.6% | **3.59×** | **pass** |
| multilingual L96 + L1024 | threads, one process | 243.8 (−2.2%) | **+23.0%** | 30.6 (**−9.0%**) | **+11.0%** | 3.05× | **fail** (P99) |
| | two processes | 252.1 (+1.1%) | +2.4% | 33.7 (+0.2%) | −0.3% | **3.17×** | **pass** |

Exit criteria (DEVELOPMENT_PLAN.md §12.3): aggregate ≥ 2.5× the best single-device
throughput, each stream's P99 within 10% of its solo value, no correctness change.
"GPU-only" is the same two streams both on MLX (threads, one process): 31.8 req/s (typed),
90.1 req/s (multilingual). No answer changed under load in any window.

Same-device control (both streams on MLX): the short stream loses 61–76% of its
throughput and its P99 grows 173–419%, as in Phase -1. Heterogeneous execution is what
protects short requests.

## Where the thread penalty comes from (typed-decisions)

| Condition | ane_short req/s | ANE P99 | gpu_long req/s | GPU P99 |
|---|---:|---:|---:|---:|
| solo, full predict | 99.4 | 10.24 | 13.9 | 72.35 |
| solo, forward only | 100.7 | 10.06 | 14.1 | 71.32 |
| threads, full predict | 98.2 | 11.50 | 12.4 | 81.65 |
| threads, **forward only** | 100.8 | 10.51 | 12.6 | 80.01 |
| ANE in-process thread, **GPU in a worker process** | 100.3 | 10.28 | 13.9 | 72.19 |
| GPU in-process thread, **ANE in a worker process** | 100.3 | 10.21 | 13.9 | 72.26 |

- Removing prompt building and answer formatting from the loop does not remove the GPU
  penalty (−10.6% against solo forward-only). The contention is between the two backends'
  own host-side work in one interpreter: MLX builds its lazy graph in Python for every
  forward, and the ANE path does NumPy feature construction and the FP32 action head per
  request. The GIL is the most likely shared resource. It was not isolated further,
  because the fix below does not depend on it. [Hypothesis]
- Moving **either** backend to its own process restores solo rates and tails for both
  streams. One interpreter per device is sufficient; the device drivers themselves do not
  interfere.

## Decision [Decision]

This section records the step-1 decision. It was revised after the request-driven
measurements below; see that section for the final placement.

- **Initial (step 1):** each device's backend in a dedicated worker process.
- **Final:** the GPU (MLX) in a worker process, and the ANE (Core ML) on a dispatcher
  thread in the caller's process. The caller's process builds prompts, routes, and formats
  answers.
- Jobs on one device run FIFO, one at a time. The ANE serialises work anyway, and
  concurrent MLX streams in one process lose throughput (control rows above).
- Queue-aware routing (`laya_apple/scheduling.py`) uses the per-device backlog as its
  first cost input, as planned.
- Threads-only execution, with both backends in one interpreter, is not offered.

## Request-driven execution (measured while implementing v0.2)

The product executor first used one worker process per device, as decided above. The
v0.2 release benchmark then ran the same mix through `Laya(execution="workers")`. The
difference from step 1 is that each request travels from the caller's threads to the
device over IPC, instead of each worker looping on its own. Typed-decisions, L128 short +
L1024 long, 6 s windows, repeated across several fresh processes over about two hours.
Probes are in `scripts/request_driven/`.

| Design (request path) | short req/s | short P99 ms | long req/s | long P99 ms | Stability |
|---|---:|---:|---:|---:|---|
| solo (either stream alone, product IPC path) | 99.5 | 10.5 | 14.0 | 72.7 | stable |
| step-1 "processes": each worker loops on its own, no per-request IPC | 100.3 | 10.4 | 13.9 | 73.5 | stable, every run |
| **both devices in worker processes** (first v0.2 design) | 72–73 | 20 | 12.5 | 86–88 | bimodal; degraded in most runs (full rate in some) |
| step-1 research workers, but one parent message per request | 75 | 14.7 | 12.7 | 85 | degraded in every run |
| two separate parent processes, one per stream, per-request IPC | 75 | 14.5 | 12.7 | 85 | degraded in 8 of 10 windows |
| GPU inline in the caller + ANE worker process | 94–96 | 14–22 | 13.7–13.8 | 85 | variable |
| **ANE on a thread in the caller + GPU worker process** (chosen) | 97–98 | 12.2–14.4 | 12.3–12.5 | 87–94 | stable over 18 windows |

**Mechanism [Measured].** The per-stage timings inside the ANE worker, in the degraded
state:
- Feature building rose from 0.16 to 0.78 ms and the action-head tail from 0.05 to 0.34 ms.
- Core ML `predict` rose by 1.2 ms.
- A cache-resident integer loop on the same thread took 0.25–0.30 ms with the ANE alone,
  and 1.16–1.72 ms while the GPU stream ran (4–6×).
- A separate CPU probe process kept its normal speed throughout.

So the IPC-driven device thread runs at a far lower CPU performance level (efficiency
cores or a much lower clock) whenever the other device is also busy. The SoC as a whole
is not throttled. The kernel's performance controller is not observable without root, so
the exact policy is [Hypothesis].

**Mitigations that did not help [Measured].**
- USER_INTERACTIVE thread QoS on the worker and dispatcher threads. The QoS change was
  verified to take effect (0x15 → 0x21).
- A 1 ms GIL switch interval.
- Spin-polling instead of blocking, for 2, 5 and 50 ms, in the workers.
- A relay: an IPC thread plus a separate compute thread.
- Starting the workers in a new session.

**Consequences.**
- The step-1 exit criterion "each stream's P99 within 10% of its solo value" is not met
  by any request-driven design on this platform. The Phase -1 and step-1 isolation
  numbers describe self-driven loops, not serving.
- The v0.2 release benchmark reports the achieved isolation against its alternative,
  GPU-only serving, as well as against solo. See `DEVELOPMENT_PLAN.md` §12.3 for the
  revised gate.
- Placement is ANE on a thread, GPU in a process. The GPU stays out of the caller's
  interpreter because two backends in one interpreter interfere (step 1). The ANE stays in
  it because, of all tested designs, only an in-process ANE kept the short stream near its
  solo rate.

## Not measured here

- Open-loop and bursty arrivals, mixed lengths and multi-question requests. These are
  measured against the product executor in the v0.2 release benchmark.
- Energy, and other SoCs or OS versions.
