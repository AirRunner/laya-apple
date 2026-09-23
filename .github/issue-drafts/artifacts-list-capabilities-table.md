---
title: "Add a human-readable table for laya-apple artifacts list --capabilities"
labels: ["good first issue", "coreml"]
---

## Problem

`laya-apple artifacts list --capabilities` (`laya_apple/cli.py:cmd_artifacts`) prints
`artifact_capabilities()` as raw JSON only (`_json(artifact_capabilities())`). The plain
`laya-apple artifacts list` (no flag) is also JSON. For a human checking what's built
and parity-validated on their machine, that means eyeballing a JSON blob instead of a
scannable table.

## Expected output

A `--format table` (or a plain default when stdout is a TTY, keeping `--json` for
machine consumption — pick whichever matches how other laya-apple CLI commands already
handle this, check `cli.py`'s `_json` helper and existing flags first) that renders
`artifact_capabilities()` as columns: model, bucket/length, status, parity passed,
compute plan. Keep the existing JSON output as the default or via an explicit flag so
scripts that already parse it (`docs/no-silent-fallback.md`'s tooling, CI) don't break.

## How to validate

- `laya-apple artifacts list --capabilities` (existing default behavior) is unchanged
  unless a new flag is explicitly passed.
- New table output is legible with 0, 1, and several artifacts built.
- Any existing test around `cmd_artifacts` / `artifact_capabilities` still passes.

## Relevant files

- `laya_apple/cli.py` (`cmd_artifacts`, the `list` action and `--capabilities` flag)
- `laya_apple/artifacts.py` (`artifact_capabilities`, `list_artifacts`)

## Difficulty / scope

Good first issue. Presentation-only change to one CLI subcommand; no ANE hardware
required to test against artifacts you already have (or an empty artifact directory).
