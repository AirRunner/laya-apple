"""Unit tests for the pure helpers in scripts/release_gate.py (no model loading)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_release_gate():
    spec = importlib.util.spec_from_file_location("release_gate", ROOT / "scripts" / "release_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def release_gate():
    return _load_release_gate()


# --------------------------------------------------------------------- doc references


def test_parse_doc_test_refs_extracts_file_and_name(release_gate):
    md = "See `test_no_silent_fallback.py::test_thing` and `other.py::test_prefix_*`."
    refs = release_gate.parse_doc_test_refs(md)
    assert refs == [("test_no_silent_fallback.py", "test_thing"), ("other.py", "test_prefix_*")]


def test_resolve_doc_test_refs_finds_exact_and_prefix_matches(tmp_path, release_gate):
    test_file = tmp_path / "test_foo.py"
    test_file.write_text("def test_exact_match():\n    pass\n\n\ndef test_prefixed_thing():\n    pass\n")
    refs = [("test_foo.py", "test_exact_match"), ("test_foo.py", "test_prefixed_*")]
    unresolved = release_gate.resolve_doc_test_refs(refs, tmp_path)
    assert unresolved == []


def test_resolve_doc_test_refs_reports_missing_function(tmp_path, release_gate):
    test_file = tmp_path / "test_foo.py"
    test_file.write_text("def test_real():\n    pass\n")
    refs = [("test_foo.py", "test_real"), ("test_foo.py", "test_does_not_exist")]
    unresolved = release_gate.resolve_doc_test_refs(refs, tmp_path)
    assert unresolved == ["test_foo.py::test_does_not_exist"]


def test_resolve_doc_test_refs_reports_missing_file(tmp_path, release_gate):
    refs = [("nope.py", "test_something")]
    unresolved = release_gate.resolve_doc_test_refs(refs, tmp_path)
    assert unresolved == ["nope.py::test_something"]


# --------------------------------------------------------------------- manifest schema


def test_check_manifests_passes_a_good_manifest(tmp_path, release_gate, manifest_factory):
    from laya_apple.registry import models

    spec = next(iter(models().values()))
    manifest = manifest_factory(spec, 64, **{"integrity.artifact_sha256": "a" * 64})

    good_dir = tmp_path / spec.name / spec.revision[:12] / "graph-L64-B1"
    good_dir.mkdir(parents=True)
    (good_dir / "manifest.json").write_text(json.dumps(manifest))

    ok, message = release_gate.check_manifests(tmp_path)
    assert ok, message
    assert "1 manifest" in message


def test_check_manifests_fails_a_bad_manifest(tmp_path, release_gate, manifest_factory, delete_key):
    from laya_apple.registry import models

    spec = next(iter(models().values()))
    manifest = manifest_factory(spec, 64, **{"integrity.artifact_sha256": "a" * 64, "status": delete_key})

    bad_dir = tmp_path / spec.name / spec.revision[:12] / "graph-L64-B1"
    bad_dir.mkdir(parents=True)
    (bad_dir / "manifest.json").write_text(json.dumps(manifest))

    ok, message = release_gate.check_manifests(tmp_path)
    assert not ok
    assert "manifest.json" in message


def test_check_manifests_ignores_excluded_subtrees(tmp_path, release_gate, manifest_factory, delete_key):
    from laya_apple.registry import models

    spec = next(iter(models().values()))
    bad_manifest = manifest_factory(spec, 64, **{"integrity.artifact_sha256": "a" * 64, "status": delete_key})

    quarantine_dir = tmp_path / "quarantine" / spec.name / "graph-L64-B1"
    quarantine_dir.mkdir(parents=True)
    (quarantine_dir / "manifest.json").write_text(json.dumps(bad_manifest))

    ok, message = release_gate.check_manifests(tmp_path)
    assert ok, message
    assert "0 manifest" in message


def test_find_manifest_paths_missing_root_is_empty(tmp_path, release_gate):
    assert release_gate.find_manifest_paths(tmp_path / "does-not-exist") == []
