---
title: "Heterogeneous scheduler instrumentation (queue depth / backlog traces)"
labels: ["help wanted", "scheduler", "performance"]
---

## Problem

`laya_apple/scheduling.py` (`decide_queued`, v0.2) routes short requests to the ANE or
GPU based on a live backlog snapshot (`ANE_GPU_BACKLOG` /
`GPU_ANE_BACKLOG` = `gpu_backlog_shorter_on_ane` / `ane_backlog_shorter_on_gpu`), and
`laya_apple/executor.py` runs the actual worker queues behind it. `RuntimeInfo` records
`queue_wait_ms` and `device_ms` per request (README, "`result.runtime` is a
`RuntimeInfo`"), but there is no way to see the backlog decision *process* over time —
only its effect on one request's `routing_reason`. Diagnosing "why did short-stream P99
spike" today means re-deriving it from individual request logs.

## Why it matters

`README.md` ("Limitations": "Isolation is partial") and `benchmarks/v0.2.md` already document
that each stream's P99 under concurrency is above its solo value. Understanding *why* —
queue depth spikes, backlog snapshot staleness, scheduling latency — needs
instrumentation that doesn't exist yet.

## Expected output

An opt-in tracing/instrumentation hook (e.g. an optional callback or a debug log line)
in `scheduling.py`/`executor.py` that records, per routing decision, the backlog
snapshot it saw and the decision it made, cheap enough to leave disabled by default with
zero measurable overhead. Not a new default-on feature — the "no silent fallback" and
performance-sensitive nature of this code path means any always-on instrumentation needs
its cost measured and justified.

## How to validate

- With instrumentation off (default), reproduce a `benchmarks/v0.2.md`-style
  closed-loop or open-loop run and confirm no measurable latency regression.
- With instrumentation on, show a trace that explains a real backlog-driven routing
  decision (`ane_backlog_shorter_on_gpu` / `gpu_backlog_shorter_on_ane`) end to end.
- Add a test pinning that the instrumentation hook never changes routing decisions
  (observability only, per `docs/no-silent-fallback.md`'s framing of routing
  correctness).

## Relevant files

- `laya_apple/scheduling.py` (`decide_queued`, `ServiceModel`, `ANE_GPU_BACKLOG`/`GPU_ANE_BACKLOG`)
- `laya_apple/executor.py`
- `README.md` ("Limitations": "Isolation is partial") and `benchmarks/v0.2.md`
- `docs/no-silent-fallback.md` (row 2, queue-based routing)

## Difficulty / scope

No ANE hardware strictly required to prototype against the GPU-only path, but full
validation needs a machine with built ANE artifacts to exercise real heterogeneous
routing. Touches performance-sensitive scheduling code — keep the change minimal and
opt-in.
