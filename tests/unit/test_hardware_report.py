"""Unit tests for the pure parts of scripts/hardware_report.py (no model loading)."""

from __future__ import annotations

import importlib.util
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def hr():
    spec = importlib.util.spec_from_file_location("hardware_report", ROOT / "scripts" / "hardware_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------- slug and output dir


@pytest.mark.parametrize(
    "soc, macos, expected",
    [
        ("Apple M4 Max", "26.6.2", "apple-m4-max-macos26"),
        ("Apple M1", "15.5", "apple-m1-macos15"),
        ("Apple M2 Ultra", "14", "apple-m2-ultra-macos14"),
        ("", "", "unknown-soc-macosunknown"),
        (None, None, "unknown-soc-macosunknown"),
    ],
)
def test_slug(hr, soc, macos, expected):
    assert hr.slug(soc, macos) == expected


def test_output_dir_never_overwrites(tmp_path, hr):
    now = datetime(2026, 9, 23, 21, 5, 7)
    first = hr.output_dir(tmp_path, "apple-m4-max-macos26", now)
    assert first == tmp_path / "apple-m4-max-macos26"
    first.mkdir()
    second = hr.output_dir(tmp_path, "apple-m4-max-macos26", now)
    assert second == tmp_path / "apple-m4-max-macos26-20260923-210507"


# --------------------------------------------------------------------- sanitization


def test_sanitize_replaces_home_everywhere(hr):
    home = "/Users/someone"
    data = {
        "path": f"{home}/.cache/laya-apple",
        "nested": {"list": [f"error in {home}/x.py", 3, None], f"{home}/key": (f"{home}", 1.5)},
        "other": "/opt/data/cache",
    }
    out = hr.sanitize(data, home)
    assert out == {
        "path": "~/.cache/laya-apple",
        "nested": {"list": ["error in ~/x.py", 3, None], "~/key": ["~", 1.5]},
        "other": "/opt/data/cache",
    }
    assert home not in repr(out)


def test_sanitize_replaces_checkout_root_before_home(hr):
    home = "/Users/someone"
    root = f"{home}/src/laya-apple"
    out = hr.sanitize({"source": f"{root}/laya_apple", "cache": f"{home}/.cache"}, home, root)
    assert out == {"source": "./laya_apple", "cache": "~/.cache"}


def test_sanitize_defaults_to_real_home(hr):
    assert hr.sanitize(str(Path.home()) + "/a") == "~/a"


# --------------------------------------------------------------------- environment


def test_collect_environment_shape(monkeypatch, hr):
    answers = {
        ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M9 Test",
        ("sysctl", "-n", "hw.model"): "Mac99,1",
        ("sysctl", "-n", "hw.memsize"): str(64 * 2**30),
        ("sw_vers", "-productVersion"): "27.1",
        ("sw_vers", "-buildVersion"): "27A100",
    }

    def fake_check_output(cmd, **kwargs):
        cmd = tuple(cmd)
        if cmd[0] == "git":
            if "rev-parse" in cmd:
                return "0123456789abcdef0123456789abcdef01234567\n"
            return " M laya_apple/model.py\n"
        if cmd in answers:
            return answers[cmd] + "\n"
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(hr.subprocess, "check_output", fake_check_output)
    env = hr.collect_environment()
    assert env["soc"] == "Apple M9 Test"
    assert env["hw_model"] == "Mac99,1"
    assert env["memory_bytes"] == 64 * 2**30
    assert env["memory_gb"] == 64
    assert (env["macos"], env["macos_build"]) == ("27.1", "27A100")
    assert set(env["packages"]) == {"mlx", "coremltools", "numpy"}
    assert isinstance(env["python"], str)
    assert set(env["laya_apple"]) == {"version", "source", "git"}
    git = env["laya_apple"]["git"]
    if git is not None:  # only when laya_apple is imported from this checkout
        assert git == {"revision": "0123456789abcdef0123456789abcdef01234567", "dirty": True}
    assert hr.slug(env["soc"], env["macos"]) == "apple-m9-test-macos27"


def test_collect_environment_tolerates_missing_tools(monkeypatch, hr):
    def failing(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(hr.subprocess, "check_output", failing)
    env = hr.collect_environment()
    assert env["soc"] is None and env["memory_bytes"] is None and env["macos"] is None
    assert env["laya_apple"]["git"] is None


def test_git_info_outside_checkout(tmp_path, hr):
    assert hr.git_info(tmp_path) is None


# --------------------------------------------------------------------- matrix


def _model(mlx=True, ane=True, uses_ane=True, het=True):
    def section(v):
        if v is None:
            return {"status": "unavailable", "reason": "ArtifactMissingError"}
        return {"status": "ok", "passed": v}

    return {
        "mlx_parity": section(mlx),
        "ane_parity": section(ane),
        "routing": {"status": "ok", "uses_ane": uses_ane},
        "heterogeneous": section(het) if het is not None else {"status": "skipped", "reason": "no ANE"},
    }


def test_matrix_all_pass(hr):
    cells = hr.matrix_cells({"a": _model(), "b": _model()})
    assert cells == {"mlx": "✓", "ane": "✓", "auto_uses_ane": "yes", "heterogeneous": "✓"}


def test_matrix_mlx_only_machine(hr):
    cells = hr.matrix_cells({"a": _model(ane=None, uses_ane=False, het=None)})
    assert cells == {"mlx": "✓", "ane": "untested", "auto_uses_ane": "no", "heterogeneous": "untested"}


def test_matrix_any_failure_is_a_failure(hr):
    cells = hr.matrix_cells({"a": _model(), "b": _model(mlx=False, ane=False, het=False)})
    assert cells["mlx"] == "✗" and cells["ane"] == "✗" and cells["heterogeneous"] == "✗"


def test_matrix_errors_are_failures(hr):
    model = _model()
    model["mlx_parity"] = {"status": "error", "error": "RuntimeError: boom"}
    assert hr.matrix_cells({"a": model})["mlx"] == "✗"


def test_matrix_row_format(hr):
    env = {"soc": "Apple M4 Max", "memory_gb": 64, "macos": "26.6.2"}
    row = hr.matrix_row(env, {"a": _model(ane=None, uses_ane=False, het=None)})
    assert row == "| Apple M4 Max (64 GB, macOS 26.6.2) | ✓ | untested | no | untested |"


def test_sanitize_replaces_the_cache_directory(hr):
    out = hr.sanitize(
        {"p": "/data/cache/laya-apple/profiles/x.json"}, home="/Users/someone", cache="/data/cache/laya-apple"
    )
    assert out == {"p": "<cache>/profiles/x.json"}
