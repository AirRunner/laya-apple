"""Validation of the shipped JSON Schemas without a jsonschema dependency.

Supports the subset the shipped schemas use: type, const, enum, required, properties,
items, minimum, minLength, pattern. Unknown keys in a document are allowed (additive
format changes keep the version).
"""

from __future__ import annotations

import json
import re
from functools import cache
from importlib import resources

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "number": (int, float),
    "integer": int,
}


@cache
def load_schema(name: str) -> dict:
    return json.loads(resources.files("laya_apple.data").joinpath(name).read_text())


def errors(schema: dict, value, path: str = "$") -> list[str]:
    """Every violation of `schema` in `value`, as 'path: problem' strings."""
    out: list[str] = []
    t = schema.get("type")
    if t is not None:
        py = _TYPES[t]
        if isinstance(value, bool) and t in ("integer", "number"):
            return [f"{path}: expected {t}, got boolean"]
        if not isinstance(value, py):
            return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "const" in schema and value != schema["const"]:
        out.append(f"{path}: expected {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and "minimum" in schema:
        if value < schema["minimum"]:
            out.append(f"{path}: {value} < {schema['minimum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            out.append(f"{path}: shorter than {schema['minLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            out.append(f"{path}: does not match {schema['pattern']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                out.append(f"{path}: missing required field {key!r}")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                out.extend(errors(sub, value[key], f"{path}.{key}"))
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            out.extend(errors(schema["items"], item, f"{path}[{i}]"))
    return out


def manifest_errors(data) -> list[str]:
    return errors(load_schema("manifest.schema.json"), data)
