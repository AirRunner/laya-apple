from __future__ import annotations

import warnings

import pytest

from laya_apple import Laya, routing
from laya_apple.errors import ArtifactMissingError, ArtifactRevisionError, UnsupportedShapeError
from laya_apple.registry import models
from laya_apple.workload import make_request

pytestmark = pytest.mark.integration


def _has_any_real_artifact():
    from laya_apple.artifacts import list_artifacts

    return any(m["manifest"].get("status") == "validated" for m in list_artifacts())


def test_ane_device_row_longer_than_largest_bucket_raises_and_never_touches_mlx(model_name):
    spec = models()[model_name]
    if not spec.ane_buckets:
        pytest.skip(f"{model_name} has no ANE buckets")
    largest = max(spec.ane_buckets)
    too_long = largest + 1
    if too_long > spec.max_len:
        pytest.skip(f"{model_name}: no room above the largest bucket within max_len")
    try:
        laya = Laya.from_pretrained(model_name, device="ane", local_files_only=True)
    except ArtifactMissingError:
        pytest.skip(f"{model_name}: no ANE artifacts present to build this Laya instance")
    assert laya.mlx is None
    state, qs = make_request(laya.tokenizer, laya.config, too_long, n_questions=1)
    with pytest.raises(UnsupportedShapeError):
        laya.predict(context=state, questions=qs)
    assert laya.mlx is None


def test_ane_device_empty_artifact_cache_raises_at_load(model_name, tmp_path, monkeypatch):
    spec = models()[model_name]
    if not spec.ane_buckets:
        pytest.skip(f"{model_name} has no ANE buckets")
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path))
    with pytest.raises(ArtifactMissingError):
        Laya.from_pretrained(model_name, device="ane", local_files_only=True)


def test_auto_with_empty_artifact_cache_routes_to_gpu_with_artifact_unavailable(model_name, tmp_path, monkeypatch):
    spec = models()[model_name]
    if not spec.auto_ane_buckets:
        pytest.skip(f"{model_name} has no auto ANE buckets")
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path))
    laya = Laya.from_pretrained(model_name, device="auto", local_files_only=True)
    state, qs = make_request(laya.tokenizer, laya.config, spec.auto_ane_buckets[0], n_questions=1)
    result = laya.predict(context=state, questions=qs)
    assert result.runtime.device == "gpu"
    assert result.runtime.routing_reason == routing.ARTIFACT_UNAVAILABLE


def test_auto_short_single_question_routes_to_ane_when_artifacts_exist(cached_laya, model_name):
    spec = models()[model_name]
    if not spec.auto_ane_buckets:
        pytest.skip(f"{model_name} has no auto ANE buckets")
    laya = cached_laya(model_name, device="auto")
    if not laya.ane_state.buckets:
        pytest.skip(f"{model_name}: no validated ANE artifacts present in this cache")
    length = min(spec.auto_ane_buckets)
    state, qs = make_request(laya.tokenizer, laya.config, length, n_questions=1)
    result = laya.predict(context=state, questions=qs)
    assert result.runtime.device == "ane"
    assert result.runtime.routing_reason == routing.ANE_AUTO


def test_artifact_revision_mismatch_under_explicit_ane_raises(model_name, tmp_path, monkeypatch):
    """Copy a real, currently-present artifact into a tmp cache and corrupt its manifest revision."""
    spec = models()[model_name]
    if not spec.ane_buckets:
        pytest.skip(f"{model_name} has no ANE buckets")
    import json
    import shutil

    from laya_apple.artifacts import artifact_dir, artifacts_root

    bucket = spec.ane_buckets[0]
    real_root = artifacts_root()  # uses current LAYA_APPLE_CACHE env, set by the harness
    src = artifact_dir(spec, bucket)
    if not src.exists():
        pytest.skip(f"{model_name} L{bucket}: no real artifact present in {real_root} to copy")
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path))
    dst = artifact_dir(spec, bucket)
    # The manifest is rejected before any model file is read, so the compiled model itself
    # (~0.7 GB) need not be copied; an empty model.mlmodelc keeps read_manifest satisfied.
    (dst / "model.mlmodelc").mkdir(parents=True)
    shutil.copy2(src / "manifest.json", dst / "manifest.json")
    manifest_path = dst / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["source"]["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(data))

    with pytest.raises(ArtifactRevisionError):
        Laya.from_pretrained(model_name, device="ane", local_files_only=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        laya = Laya.from_pretrained(model_name, device="auto", local_files_only=True)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    state, qs = make_request(laya.tokenizer, laya.config, bucket, n_questions=1)
    result = laya.predict(context=state, questions=qs)
    assert result.runtime.device == "gpu"
