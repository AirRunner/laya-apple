"""Internal backend contract.

A backend turns prepared prompt rows into raw decision logits and action logits. Prompt
construction, calibration and answer formatting are shared (laya_apple.prompt), so every
backend returns the same public schema.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Backend(Protocol):
    name: str  # "mlx" | "coreml"
    device: str  # "gpu" | "ane"

    def forward(self, items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """(logits[B, K], action_logits[B, A]) as float32, synchronised."""
        ...

    def artifact_revision(self, items: list[dict]) -> str: ...
