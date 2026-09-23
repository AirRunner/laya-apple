from __future__ import annotations

import json

import pytest

from laya_apple.artifacts import (
    ArtifactIntegrityError,
    ArtifactMissingError,
    ArtifactParityError,
    ArtifactRevisionError,
    ComputeUnitMismatchError,
    artifact_dir,
    list_artifacts,
    read_manifest,
    verify_files,
    verify_manifest,
    verify_profile,
)
from laya_apple.registry import models

SPEC = models()["laya"]
BUCKET = SPEC.ane_buckets[0]


def test_valid_manifest_passes(manifest_factory):
    verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET))


def test_bad_format_raises_integrity_error(manifest_factory, delete_key):
    with pytest.raises(ArtifactIntegrityError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET, format="something-else"))


def test_bad_format_version_raises_integrity_error(manifest_factory):
    with pytest.raises(ArtifactIntegrityError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET, format_version=99))


def test_wrong_revision_raises_revision_error(manifest_factory):
    with pytest.raises(ArtifactRevisionError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET, **{"source.revision": "0" * 40}))


def test_wrong_weights_sha_raises_revision_error(manifest_factory):
    with pytest.raises(ArtifactRevisionError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET, **{"source.weights_sha256": "0" * 64}))


def test_wrong_repo_raises_revision_error(manifest_factory):
    with pytest.raises(ArtifactRevisionError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET, **{"source.repo": "someone/else"}))


def test_wrong_graph_raises_revision_error(manifest_factory):
    with pytest.raises(ArtifactRevisionError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET, **{"artifact.graph": "other-graph"}))


def test_wrong_length_raises_revision_error(manifest_factory):
    with pytest.raises(ArtifactRevisionError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET, **{"artifact.length": BUCKET + 1}))


def test_wrong_batch_raises_revision_error(manifest_factory):
    with pytest.raises(ArtifactRevisionError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET, **{"artifact.batch": 2}))


def test_wrong_precision_raises_revision_error(manifest_factory):
    with pytest.raises(ArtifactRevisionError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET, **{"artifact.precision": "float32"}))


def test_wrong_max_options_raises_revision_error(manifest_factory):
    with pytest.raises(ArtifactRevisionError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET, **{"artifact.max_options": 8}))


def test_wrong_compute_units_argument_raises_compute_unit_mismatch(manifest_factory):
    with pytest.raises(ComputeUnitMismatchError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET), compute_units="CPU_ONLY")


def test_manifest_declares_wrong_compute_units_raises_compute_unit_mismatch(manifest_factory):
    with pytest.raises(ComputeUnitMismatchError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET, **{"placement.compute_units": "CPU_ONLY"}))


def test_parity_missing_raises_parity_error(manifest_factory, delete_key):
    with pytest.raises(ArtifactParityError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET, parity=delete_key))


def test_parity_failed_raises_parity_error(manifest_factory):
    with pytest.raises(ArtifactParityError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET, parity={"passed": False}))


def test_status_not_validated_raises_parity_error(manifest_factory):
    with pytest.raises(ArtifactParityError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET, status="building"))


def test_no_hash_raises_integrity_error(manifest_factory):
    with pytest.raises(ArtifactIntegrityError):
        verify_manifest(SPEC, BUCKET, manifest_factory(SPEC, BUCKET, **{"integrity.artifact_sha256": ""}))


def test_verify_profile_mismatch_raises_revision_error():
    built = {"soc": "Apple M1", "macos": "13.0", "coremltools": "9.0"}
    current = {"soc": "Apple M4 Max", "macos": "15.0", "coremltools": "9.0"}
    with pytest.raises(ArtifactRevisionError):
        verify_profile({"platform": built}, current)


def test_verify_profile_match_passes():
    profile = {"soc": "Apple M4 Max", "macos": "15.1", "coremltools": "9.0"}
    verify_profile({"platform": profile}, profile)


def test_read_manifest_on_empty_cache_raises_missing_with_build_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path))
    with pytest.raises(ArtifactMissingError) as exc:
        read_manifest(SPEC, BUCKET)
    assert "artifacts build" in str(exc.value)


def test_verify_files_detects_tampered_file(tmp_path, manifest_factory):
    from laya_apple.artifacts import COMPILED, Manifest, tree_sha256

    d = tmp_path / "artifact"
    (d / COMPILED).mkdir(parents=True)
    f = d / COMPILED / "weights.bin"
    f.write_bytes(b"original content")
    good_hash = tree_sha256(d / COMPILED)
    manifest = Manifest(manifest_factory(SPEC, BUCKET, **{"integrity.artifact_sha256": good_hash}))
    verify_files(manifest, d)  # passes before tampering

    f.write_bytes(b"tampered content!!")
    from laya_apple.artifacts import ArtifactIntegrityError as AIE

    with pytest.raises(AIE):
        verify_files(manifest, d)


def test_list_artifacts_on_empty_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path))
    assert list_artifacts() == []


def test_list_artifacts_finds_written_manifests(tmp_path, monkeypatch, manifest_factory):
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path))
    d = artifact_dir(SPEC, BUCKET)
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(manifest_factory(SPEC, BUCKET)))
    found = list_artifacts()
    assert len(found) == 1
    assert found[0]["manifest"]["source"]["model"] == SPEC.name
