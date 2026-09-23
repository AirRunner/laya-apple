---
title: "Validate another macOS / coremltools version"
labels: ["help wanted", "coreml", "hardware", "correctness"]
---

## Problem

Every measurement and every validated artifact ships from one profile: macOS 26.6.2 with
coremltools 9.0 (`docs/support-matrix.md`). `docs/support-matrix.md` also notes that
"macOS 27.x" is unknown, and cites prior third-party observation of different Core ML
placement behaviour there (enumerated shapes running on the GPU instead of the ANE). No
one has run laya-apple's own build-and-parity pipeline against a different macOS major
version or coremltools release to see whether the same BC1S graph still places 100% on
the ANE.

## Why it matters

`profiles.py`'s docstring says the shipped routing table is valid only for the profile
it was measured on, and a mismatched profile falls back to MLX-only
(`platform_not_validated`, `docs/no-silent-fallback.md` row 18). Knowing what actually
happens on the next macOS/coremltools combination — rather than just falling back safely
— tells maintainers whether the BC1S rewrite is robust to a Core ML compiler version
change or whether it needs its own follow-up fix.

## Expected output

A report of what `laya-apple artifacts build` and `laya-apple parity --device ane`
produce on a different macOS major version and/or coremltools release than 26 / 9.0:
compute plan (ANE % and transition count), parity pass/fail, and the placement probe
ratio. Include exact `sw_vers` and `pip show coremltools` output. If it fails, describe
the failure mode (build error, wrong device placement, parity regression) — that is a
complete and useful result on its own.

## How to validate

1. `uv sync --extra dev --extra ane --extra convert` on the target macOS/coremltools combination.
2. `laya-apple artifacts build laya-typed-decisions`.
3. `laya-apple parity laya-typed-decisions --device ane`.
4. `laya-apple info` to capture `platform_profile()` and `platform_validated_for_auto_ane`.
5. Write up pass/fail with the raw compute-plan and parity output attached.

## Relevant files

- `docs/support-matrix.md` ("Platform scope")
- `laya_apple/artifacts.py` (`platform_profile`, `profile_matches`)
- `laya_apple/profiles.py`
- `docs/no-silent-fallback.md` (row 18, unvalidated platform)

## Difficulty / scope

Needs access to a Mac running a macOS version or coremltools release other than the
tested combination, plus the `ane`/`convert` extras. Investigative; a documented failure
is as valuable as a pass.
