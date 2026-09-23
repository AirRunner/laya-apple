---
title: "Add an environment diagnostics section to laya-apple info"
labels: ["good first issue", "coreml", "documentation"]
---

## Problem

`laya-apple info` (`laya_apple/cli.py:cmd_info`) already reports `laya_apple` version,
`platform_profile()`, `platform_validated_for_auto_ane`, `coremltools_available`, and
per-model artifact status. It does not report a few environment facts that
`docs/no-silent-fallback.md` and `docs/compatibility.md` treat as relevant to whether the
machine can run laya-apple correctly: MLX version/availability, and whether
`HF_HUB_OFFLINE`/`local_files_only` is in effect.

## Why it matters

When something doesn't work, `laya-apple info` is the first thing a user or issue report
should paste. Right now it's missing a couple of pieces of context (MLX version, offline
mode) that repeatedly matter for diagnosing "why doesn't this work on my machine"
reports.

## Expected output

Extend the JSON `laya-apple info` prints (`laya_apple/cli.py`, `cmd_info`) with an
`environment` (or similarly named) block: MLX version if importable, whether
`HF_HUB_OFFLINE`/`local_files_only` would apply, and Python version. Read `cmd_info`
first — do not duplicate fields it already reports (`platform`,
`coremltools_available`, artifact status are already there).

## How to validate

- `laya-apple info` still returns valid JSON with all existing keys unchanged (this is a
  stable CLI output per `docs/api.md`'s Semantic Versioning coverage — additive only,
  no renamed or removed keys).
- Add/extend a test asserting the new keys are present, e.g. under `tests/` near any
  existing CLI/info test.

## Relevant files

- `laya_apple/cli.py` (`cmd_info`)
- `docs/api.md` — CLI output is covered by SemVer as of 1.0; this must be additive
- `docs/compatibility.md`

## Difficulty / scope

Good first issue. One function, additive JSON fields, no ANE hardware required.
