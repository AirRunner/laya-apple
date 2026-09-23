from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_routing_json_is_fresh():
    """scripts/derive_routing.py --check must pass: committed routing.json matches the evidence."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "derive_routing.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
