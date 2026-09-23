"""Queue-aware, isolation-first routing for heterogeneous execution (v0.2).

`decide_queued` adds one input to the v0.1 rule: a snapshot of each device's backlog. For
the same snapshot it is deterministic, and with both queues empty it returns exactly what
routing.decide returns, so an idle machine behaves like v0.1.
"""

from __future__ import annotations

from .registry import ModelSpec
from .routing import EXCEEDS_AUTO_RANGE, MULTIPLE_QUESTIONS, AneState, Decision, decide

ANE_GPU_BACKLOG = "gpu_backlog_shorter_on_ane"  # auto chose the ANE for a tie-band length: GPU busy
GPU_ANE_BACKLOG = "ane_backlog_shorter_on_gpu"  # auto chose MLX for an ANE-range request: ANE busy
QUEUE_REASONS = (ANE_GPU_BACKLOG, GPU_ANE_BACKLOG)


class ServiceModel:
    """Per-request service-time estimates (ms) from measured forward P50s (routing.json).

    MLX: linear interpolation over the measured lengths, then over question counts (1, 4, 8),
    extrapolated linearly beyond 8. ANE: the bucket's B=1 P50 times the question count (rows
    run one after another). Estimates only rank devices; they never gate correctness.
    """

    def __init__(self, table: dict):
        self.gpu = {int(q): {int(L): v for L, v in row.items()} for q, row in table["gpu"].items()}
        self.ane = {int(L): v for L, v in table["ane"]["1"].items()}

    @staticmethod
    def _interp(points: dict, x: float) -> float:
        xs = sorted(points)
        if x <= xs[0]:  # below the shortest measurement: no cheaper than it (as in the v0.1 derivation)
            return points[xs[0]]
        for a, b in zip(xs, xs[1:]):
            if x <= b:
                return points[a] + (points[b] - points[a]) * (x - a) / (b - a)
        a, b = xs[-2], xs[-1]
        return points[b] + (points[b] - points[a]) * (x - b) / (b - a)

    def gpu_ms(self, length: int, questions: int = 1) -> float:
        by_q = {q: self._interp(row, length) for q, row in self.gpu.items()}
        return self._interp(by_q, questions)

    def ane_ms(self, bucket: int, questions: int = 1) -> float:
        return self.ane[bucket] * questions


def decide_queued(
    device: str,
    spec: ModelSpec,
    question_count: int,
    sequence_length: int,
    ane: AneState,
    *,
    service: ServiceModel,
    gpu_backlog_ms: float,
    ane_backlog_ms: float,
    tie_buckets: tuple = (),
) -> Decision:
    """v0.2 rule: the v0.1 rule on an idle machine; queue-aware when a device is busy.

    Compared quantity: expected completion = backlog on the device + this request's
    service estimate. Deterministic for a given backlog snapshot.
      * explicit gpu/ane, unavailable ANE, multiple questions, rows longer than every
        offered ANE bucket, or a missing artifact: exactly as v0.1 (never ANE for long work);
      * an auto bucket (<= auto_ane_max_len): ANE, unless the ANE backlog makes MLX finish
        sooner -> MLX (`ane_backlog_shorter_on_gpu`);
      * an explicit-only bucket whose artifact is loaded (`tie_buckets`; the tie band, e.g.
        multilingual 256): MLX, unless the GPU backlog
        makes the ANE finish sooner, using the conservative MLX estimate at the previous
        bucket -> ANE (`gpu_backlog_shorter_on_ane`).
    """
    base = decide(device, spec, question_count, sequence_length, ane)
    if device != "auto" or base.reason in (ane.unavailable, MULTIPLE_QUESTIONS):
        return base
    offered = set(ane.buckets) | set(tie_buckets)
    bucket = next((b for b in spec.ane_buckets if sequence_length <= b), None)
    if base.target == "ane":
        ane_done = ane_backlog_ms + service.ane_ms(base.bucket)
        gpu_done = gpu_backlog_ms + service.gpu_ms(sequence_length)
        if gpu_done < ane_done:
            return Decision("gpu", GPU_ANE_BACKLOG)
        return base
    # base went to MLX because the row is above the auto range: consider the tie band
    if base.reason != EXCEEDS_AUTO_RANGE or bucket is None or bucket not in offered:
        return base
    prev = max([b for b in spec.ane_buckets if b < bucket], default=bucket)
    ane_done = ane_backlog_ms + service.ane_ms(bucket)
    gpu_done = gpu_backlog_ms + service.gpu_ms(min(prev, sequence_length))
    if ane_done < gpu_done:
        return Decision("ane", ANE_GPU_BACKLOG, bucket)
    return base
