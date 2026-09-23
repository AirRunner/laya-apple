"""The artifact manifest format (format_version 1) is a shipped JSON Schema, enforced on load."""

from __future__ import annotations

import pytest

from laya_apple.artifacts import verify_manifest
from laya_apple.errors import ArtifactIntegrityError, ArtifactParityError, ArtifactRevisionError
from laya_apple.registry import models
from laya_apple.schema import errors, load_schema, manifest_errors

SPEC = models()["laya"]


def test_synthetic_manifest_matches_schema(manifest_factory):
    assert manifest_errors(manifest_factory(SPEC, 64)) == []


def test_unknown_fields_are_allowed(manifest_factory):
    data = manifest_factory(SPEC, 64, **{"future_field": {"x": 1}, "artifact.new_thing": 3})
    assert manifest_errors(data) == []
    verify_manifest(SPEC, 64, data)


@pytest.mark.parametrize(
    "override, fragment, error",
    [
        ({"integrity.artifact_sha256": "abc"}, "$.integrity.artifact_sha256", ArtifactIntegrityError),
        ({"source.revision": "short"}, "$.source.revision", ArtifactRevisionError),
        ({"artifact.batch": True}, "$.artifact.batch", ArtifactIntegrityError),
        ({"artifact.length": "sixty-four"}, "$.artifact.length", ArtifactIntegrityError),
        ({"status": "maybe"}, "$.status", ArtifactParityError),
    ],
)
def test_malformed_manifest_is_refused_with_the_field_named(manifest_factory, override, fragment, error):
    """The specific check fires first where one exists; otherwise the schema names the field."""
    data = manifest_factory(SPEC, 64, **override)
    assert any(e.startswith(fragment) for e in manifest_errors(data))
    with pytest.raises(error):
        verify_manifest(SPEC, 64, data)


def test_missing_platform_is_a_schema_error(manifest_factory, delete_key):
    data = manifest_factory(SPEC, 64, platform=delete_key)
    with pytest.raises(ArtifactIntegrityError, match="missing required field 'platform'"):
        verify_manifest(SPEC, 64, data)


def test_validator_subset():
    s = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer", "minimum": 1}}}
    assert errors(s, {"a": 1}) == []
    assert errors(s, {}) == ["$: missing required field 'a'"]
    assert errors(s, {"a": 0}) == ["$.a: 0 < 1"]
    assert errors(s, []) == ["$: expected object, got list"]
    assert load_schema("manifest.schema.json")["properties"]["format_version"] == {"const": 1}
