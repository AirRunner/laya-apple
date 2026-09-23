"""Unit tests for profiles.calibrate's internals: parity extraction and the locked,
concurrency-safe profile write (code review finding 12)."""

from __future__ import annotations

import json
import threading

import pytest

from laya_apple.errors import ArtifactParityError
from laya_apple.profiles import PROFILE_FORMAT, PROFILE_VERSION, _parity_records, _save_profile


class _Manifest:
    def __init__(self, data):
        self.data = data


def test_parity_records_extracts_passed_and_prob():
    manifests = {64: _Manifest({"parity": {"passed": True, "prob_max_abs": 0.01}})}
    assert _parity_records("m", manifests) == {64: (True, 0.01)}


def test_parity_records_missing_raises_parity_error():
    manifests = {64: _Manifest({"parity": None})}
    with pytest.raises(ArtifactParityError):
        _parity_records("m", manifests)


def test_parity_records_incomplete_raises_parity_error():
    manifests = {64: _Manifest({"parity": {"passed": True}})}  # no prob_max_abs
    with pytest.raises(ArtifactParityError):
        _parity_records("m", manifests)


def test_save_profile_concurrent_writers_do_not_corrupt_the_file(tmp_path):
    path = tmp_path / "profile.json"
    profile = {"soc": "test", "macos": "1.0", "coremltools": "9.0"}
    errors = []

    def write(model, n):
        try:
            for i in range(20):
                _save_profile(path, profile, model, {"i": i, "n": n})
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=write, args=(f"model-{n}", n)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    data = json.loads(path.read_text())
    assert data["format"] == PROFILE_FORMAT and data["format_version"] == PROFILE_VERSION
    assert set(data["models"]) == {f"model-{n}" for n in range(6)}  # every writer's entry survived
