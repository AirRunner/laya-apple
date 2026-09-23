"""Deterministic device selection (DEVELOPMENT_PLAN.md §7.2, §7.3).

`decide` is a pure function of the request shape and what is available. It uses no
timing, no queue state and no randomness, so the same inputs always give the same target
and the same reason. The queue-aware v0.2 rule builds on it in laya_apple.scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass

from .registry import ModelSpec

DEVICES = ("auto", "gpu", "ane")

GPU_REQUESTED = "gpu_requested"
ANE_REQUESTED = "ane_requested"
ANE_AUTO = "validated_short_single_question_path"
MULTIPLE_QUESTIONS = "multiple_questions"
EXCEEDS_AUTO_RANGE = "sequence_exceeds_ane_auto_range"
ARTIFACT_UNAVAILABLE = "ane_artifact_unavailable"
RUNTIME_UNAVAILABLE = "ane_runtime_unavailable"
PLATFORM_NOT_VALIDATED = "platform_not_validated"

REASONS = (
    GPU_REQUESTED,
    ANE_REQUESTED,
    ANE_AUTO,
    MULTIPLE_QUESTIONS,
    EXCEEDS_AUTO_RANGE,
    ARTIFACT_UNAVAILABLE,
    RUNTIME_UNAVAILABLE,
    PLATFORM_NOT_VALIDATED,
)


@dataclass(frozen=True)
class AneState:
    """What the ANE path can offer to `auto` in this process.

    `unavailable` is None when usable, else RUNTIME_UNAVAILABLE / PLATFORM_NOT_VALIDATED.
    `buckets` are the auto buckets whose artifacts loaded and verified.
    """

    unavailable: str | None = None
    buckets: tuple = ()


@dataclass(frozen=True)
class Decision:
    target: str  # "gpu" | "ane"
    reason: str
    bucket: int | None = None  # the ANE bucket auto would use


def decide(device: str, spec: ModelSpec, question_count: int, sequence_length: int, ane: AneState) -> Decision:
    if device == "gpu":
        return Decision("gpu", GPU_REQUESTED)
    if device == "ane":
        return Decision("ane", ANE_REQUESTED)
    if device != "auto":
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
    # Order matters only for which reason is reported; each check alone routes to MLX.
    if ane.unavailable:
        return Decision("gpu", ane.unavailable)
    if question_count > spec.auto_ane_max_questions:
        return Decision("gpu", MULTIPLE_QUESTIONS)
    bucket = next((b for b in spec.auto_ane_buckets if sequence_length <= b), None)
    if bucket is None or sequence_length > spec.auto_ane_max_len:
        return Decision("gpu", EXCEEDS_AUTO_RANGE)
    if bucket not in ane.buckets:
        return Decision("gpu", ARTIFACT_UNAVAILABLE)
    return Decision("ane", ANE_AUTO, bucket)
