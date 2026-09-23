from __future__ import annotations

import pytest

from laya_apple.conversion.build import build
from laya_apple.errors import UnsupportedShapeError
from laya_apple.registry import models


def test_build_rejects_non_offered_bucket_before_doing_any_work(monkeypatch):
    spec = models()["laya"]
    bad_length = max(spec.ane_buckets) + 1
    assert bad_length not in spec.ane_buckets

    def _boom(*a, **k):
        raise AssertionError("build() should not touch the network/checkpoint for an unsupported bucket")

    monkeypatch.setattr("laya_apple.hub.checkpoint_path", _boom)
    with pytest.raises(UnsupportedShapeError):
        build(spec, bad_length)
