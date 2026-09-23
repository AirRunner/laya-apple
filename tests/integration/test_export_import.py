"""v0.3 artifact distribution: export here, import elsewhere only after local re-validation."""

from __future__ import annotations

import io
import json
import tarfile

import pytest

from laya_apple import lifecycle
from laya_apple.artifacts import COMPILED, artifact_dir, artifacts_root, load_verified
from laya_apple.errors import ArtifactIntegrityError, ArtifactMissingError
from laya_apple.registry import models

pytestmark = [pytest.mark.integration, pytest.mark.ane]
SPEC = models()["laya-multilingual"]
BUCKET = 64


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    try:
        load_verified(SPEC, BUCKET)
    except ArtifactMissingError:
        pytest.skip("no multilingual L64 artifact in the cache to export")
    return lifecycle.export_artifact(SPEC, BUCKET, tmp_path_factory.mktemp("export") / "ml-L64")


def test_import_revalidates_and_registers(exported, tmp_path, monkeypatch):
    src_manifest = json.loads((artifact_dir(SPEC, BUCKET) / "manifest.json").read_text())
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path))
    final = lifecycle.import_artifact(exported, local_files_only=True, log=lambda m: None)
    assert final == artifact_dir(SPEC, BUCKET) and str(final).startswith(str(tmp_path))
    data = json.loads((final / "manifest.json").read_text())
    assert data["integrity"] == src_manifest["integrity"]  # original provenance kept
    imp = data["imported"]
    assert imp["parity"]["passed"] is True and imp["parity"]["hard_mismatches"] == 0
    assert imp["placement"]["transitions"] == 0 and imp["placement"]["ops"]["cpu"] == 0
    model, manifest = load_verified(SPEC, BUCKET)
    assert manifest.artifact_sha256 == src_manifest["integrity"]["artifact_sha256"]


def test_tampered_export_is_refused(exported, tmp_path, monkeypatch):
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(exported) as src, tarfile.open(bad, "w:gz") as dst:
        tampered = False
        for m in src.getmembers():
            f = src.extractfile(m) if m.isfile() else None
            if f is not None and not tampered and m.name.startswith(COMPILED) and m.size > 0:
                payload = bytearray(f.read())
                payload[0] ^= 0xFF
                m.size = len(payload)
                dst.addfile(m, io.BytesIO(bytes(payload)))
                tampered = True
            else:
                dst.addfile(m, f)
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path / "cache"))
    with pytest.raises(ArtifactIntegrityError):
        lifecycle.import_artifact(bad, local_files_only=True, log=lambda m: None)
    assert not artifact_dir(SPEC, BUCKET).exists()
    assert not any((artifacts_root() / ".staging").glob("*"))  # nothing left behind


def test_non_artifact_archive_is_refused(tmp_path, monkeypatch):
    bogus = tmp_path / "x.tar.gz"
    with tarfile.open(bogus, "w:gz") as t:
        info = tarfile.TarInfo("readme.txt")
        info.size = 2
        t.addfile(info, io.BytesIO(b"hi"))
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path / "cache"))
    from laya_apple.errors import ArtifactError

    with pytest.raises(ArtifactError):
        lifecycle.import_artifact(bogus, local_files_only=True, log=lambda m: None)
