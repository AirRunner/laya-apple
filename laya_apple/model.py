"""`Laya`: the public entry point (DEVELOPMENT_PLAN.md §10).

from laya_apple import Laya
laya = Laya.from_pretrained("convaiinnovations/laya-typed-decisions")
result = laya.predict(context="...", questions={...})
result.answers, result.runtime
"""

from __future__ import annotations

import json
import threading
import time
import warnings
from pathlib import Path

from . import routing
from .artifacts import platform_profile, profile_matches
from .errors import ArtifactMissingError, BackendUnavailableError, InvalidRequestError, LayaAppleError
from .hub import checkpoint_path, verify_weights
from .prompt import Calibration, Tokenizer, format_answers, prepare
from .registry import ANE_COMPUTE_UNITS, ANE_PRECISION, ModelSpec, resolve, routing_table
from .result import Result, RuntimeInfo


def _coremltools_available() -> bool:
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return False
    return True


def platform_validated(profile: dict | None = None) -> bool:
    profile = profile or platform_profile()
    return any(profile_matches(profile, p) for p in routing_table().get("validated_profiles", []))


class Laya:
    """A loaded checkpoint plus the backends its `device` setting allows."""

    def __init__(self, spec: ModelSpec, checkpoint: Path, device: str, *, dtype: str, batch_size: int):
        if device not in routing.DEVICES:
            raise ValueError(f"device must be one of {routing.DEVICES}, got {device!r}")
        self.spec, self.checkpoint, self.device, self.dtype = spec, checkpoint, device, dtype
        self.config = json.loads((checkpoint / "rl_agent_config.json").read_text())
        encoder_cfg = json.loads((checkpoint / "encoder/config.json").read_text())
        self.tokenizer = Tokenizer(checkpoint / "tokenizer")
        self.calibration = Calibration(self.config)
        self.mlx = self.ane = None
        self.ane_state = routing.AneState(routing.RUNTIME_UNAVAILABLE)
        # One request at a time per instance: Core ML predictions and MLX graph evaluation
        # on a shared model are not documented as re-entrant.
        self._lock = threading.Lock()

        if device in ("auto", "gpu"):
            from .backends.mlx import MLXBackend

            self.mlx = MLXBackend(spec, checkpoint, self.tokenizer.pad_token_id, dtype=dtype, batch_size=batch_size)
        if device == "ane":
            self.ane = self._load_ane(spec.ane_buckets, encoder_cfg, strict=True)
            self.ane_state = routing.AneState(None, self.ane.buckets)
        elif device == "auto":
            self.ane_state = self._auto_ane(encoder_cfg)

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str = "auto",
        *,
        dtype: str = "float16",
        local_files_only: bool = False,
        batch_size: int = 16,
    ) -> "Laya":
        """Load a pinned checkpoint.

        device="gpu": MLX only. device="ane": validated Core ML artifacts only; requests
        they cannot serve raise. device="auto": MLX, plus the ANE for the validated short
        single-question path when its artifacts are present and verified (§7.2).
        """
        if device not in routing.DEVICES:
            raise ValueError(f"device must be one of {routing.DEVICES}, got {device!r}")
        spec = resolve(model_id)
        path = checkpoint_path(spec, local_files_only=local_files_only)
        verify_weights(spec, path)
        return cls(spec, path, device, dtype=dtype, batch_size=batch_size)

    def _load_ane(self, buckets, encoder_cfg, *, strict):
        if not _coremltools_available():
            raise BackendUnavailableError("device='ane' needs coremltools: install the [ane] extra (uv sync --extra ane)")
        from .backends.coreml_ane import ANEBackend

        return ANEBackend(
            self.spec,
            self.checkpoint,
            self.tokenizer.pad_token_id,
            int(encoder_cfg["local_attention"]),
            buckets,
            strict=strict,
        )

    def _auto_ane(self, encoder_cfg) -> routing.AneState:
        if not self.spec.auto_ane_buckets:
            return routing.AneState(None, ())
        if not _coremltools_available():
            return routing.AneState(routing.RUNTIME_UNAVAILABLE)
        if not platform_validated():
            return routing.AneState(routing.PLATFORM_NOT_VALIDATED)
        self.ane = self._load_ane(self.spec.auto_ane_buckets, encoder_cfg, strict=False)
        rejected = {b: e for b, e in self.ane.load_errors.items() if not isinstance(e, ArtifactMissingError)}
        for b, e in rejected.items():
            warnings.warn(
                f"laya-apple: ANE artifact {self.spec.name} L{b} rejected ({type(e).__name__}: {e}); "
                "auto routes these requests to MLX",
                RuntimeWarning,
                stacklevel=4,
            )
        if not self.ane.models:
            self.ane = None
            return routing.AneState(None, ())
        return routing.AneState(None, self.ane.buckets)

    # ------------------------------------------------------------------ inference

    def route(self, prepared) -> routing.Decision:
        return routing.decide(self.device, self.spec, prepared.question_count, prepared.sequence_length, self.ane_state)

    def prepare(self, context, questions):
        return prepare(self.tokenizer, self.config, context, questions)

    def predict(self, context=None, questions=None, *, state=None) -> Result:
        """Answer `questions` about `context` (upstream calls it `state`; either works)."""
        if state is not None:
            if context is not None:
                raise InvalidRequestError("pass context or state, not both")
            context = state
        if context is None:
            context = ""
        if questions is None:
            raise InvalidRequestError("questions are required")
        t0 = time.perf_counter()
        prep = self.prepare(context, questions)
        decision = self.route(prep)
        backend = self.ane if decision.target == "ane" else self.mlx
        if backend is None:  # unreachable by construction; never substitute another device
            raise LayaAppleError(f"internal routing error: {decision}")
        with self._lock:
            if decision.target == "ane":
                buckets = tuple(backend.check(prep.items))  # raises before any work runs
            logits, act = backend.forward(prep.items)
        answers = format_answers(prep, logits, act, self.calibration)
        latency = (time.perf_counter() - t0) * 1000
        runtime = RuntimeInfo(
            backend=backend.name,
            device=backend.device,
            model=self.spec.name,
            model_revision=self.spec.revision,
            sequence_length=prep.sequence_length,
            question_count=prep.question_count,
            routing_reason=decision.reason,
            artifact_revision=backend.artifact_revision(prep.items),
            latency_ms=latency,
            compute_units=ANE_COMPUTE_UNITS if decision.target == "ane" else None,
            buckets=buckets if decision.target == "ane" else (),
            dtype=ANE_PRECISION if decision.target == "ane" else self.dtype,
        )
        return Result(answers=answers, usage={"input_tokens": prep.input_tokens, "output_tokens": 0}, runtime=runtime)

    def info(self) -> dict:
        return {
            "model": self.spec.name,
            "repo": self.spec.repo,
            "revision": self.spec.revision,
            "device": self.device,
            "dtype": self.dtype,
            "mlx": self.mlx is not None,
            "ane_buckets": list(self.ane.buckets) if self.ane else [],
            "ane_load_errors": {
                str(b): f"{type(e).__name__}: {e}" for b, e in (self.ane.load_errors.items() if self.ane else [])
            },
            "auto_ane": {"unavailable": self.ane_state.unavailable, "buckets": list(self.ane_state.buckets)},
        }
