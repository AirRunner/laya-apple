---
title: "Add a score/noul question walkthrough to docs/guide.md"
labels: ["good first issue", "documentation"]
---

## Problem

`docs/guide.md`'s question schema table documents `choice`, `score`, and `noul` question
types, but the README's runnable example (and the guide's own worked examples, per its
current structure) leans on `choice`. There's no equivalent step-by-step walkthrough for
a `score` question (a non-empty list of level descriptions, 0-indexed) or a `noul`
question (`{"false": ..., "true": ...}`, yes/no).

## Why it matters

Someone who only reads the README's `choice` example doesn't have a copy-pasteable
pattern for the other two question types laya-apple actually supports and validates.

## Expected output

A short new section (or subsection) in `docs/guide.md` with a runnable `predict()` call
for a `score` question and one for a `noul` question, each showing the resulting
`result.answers[...]` shape, next to the existing schema table.

## How to validate

- The example code actually runs against a downloaded checkpoint
  (`laya-apple download <model>`) and the printed shape in the doc matches real output.
- Doesn't restate the schema table verbatim — link to it instead of duplicating it.

## Relevant files

- `docs/guide.md` (question schema table, existing examples)
- `laya_apple/schema.py` (validation for `choice`/`score`/`noul`)

## Difficulty / scope

Good first issue. Documentation only; needs a downloaded checkpoint to verify the
example output but no ANE artifacts.
