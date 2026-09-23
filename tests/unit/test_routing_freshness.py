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


def test_placement_json_is_fresh():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    r = subprocess.run(
        [sys.executable, str(root / "scripts/derive_placement.py"), "--check"], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr
