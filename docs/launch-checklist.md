# v1.0 launch checklist

Status of each item needed before laya-apple's first public push, with the evidence this
audit checked. Every path below was actually opened and read as part of this checklist
(not assumed from a filename), unless marked "verify after merge" for another
contributor's concurrent work.

**Status legend:** ✅ done (evidence path given) · ⏳ pending (owner action given) ·
👤 maintainer-only (needs a human decision or a credential/account this checklist has no
access to).

## Content and quickstart

| Item | Status | Evidence / owner action |
|---|---|---|
| README understandable in <30 s | ⏳ pending | `README.md` is 393 lines with a clear headline, a runnable quickstart near the top, and a strong section structure (What makes it different → Quickstart → Install → Supported models → Correctness → benchmarks → Limitations). It is *not* a 30-second read top to bottom — it is closer to a reference document with a fast opening. Owner action: read only the first ~60 lines (through "Quickstart") as a cold reader and confirm those alone convey what the project is and how to try it; the rest is reference material a skim can skip. |
| Installation from clean env | ⏳ pending, verify after merge | `scripts/release_gate.py` builds and smoke-tests the base and `ane` extras on Python 3.11, 3.12 and 3.13 in its "clean install matrix" step (`scripts/release_gate.py:26` `PY_VERSIONS`, `:365` `install_matrix_step`). This audit read the script but did not execute it (no model runs / benchmarks per this task's constraints). Owner action: run `uv run python scripts/release_gate.py` (or at least `--quick`, which skips this step, then a manual `uv sync` smoke test) before release. |
| Quickstart works (gate runs it) | ⏳ pending, verify after merge | The README quickstart (`Laya.from_pretrained(...).predict(...)`) matches the API in `laya_apple/model.py`; not independently executed here (no model runs). Owner action: confirm `release_gate.py` (or a manual run) actually exercises this exact snippet, or run it by hand once. |
| Examples work | ⏳ pending, verify after merge | `examples/basic.py`, `examples/auto_routing.py`, `examples/heterogeneous_serving.py` exist and are referenced from the README and `CHANGELOG.md` under `[Unreleased]`. These are another contributor's concurrent work; this audit read them for scope conflicts only, not for correctness. Owner action: run each example once against a downloaded checkpoint. |
| Benchmark command works | ⏳ pending, verify after merge | `scripts/hardware_report.py` exists (added concurrently) and `docs/community-benchmarks.md` documents the bundle it produces. `laya-apple benchmark <model>` is a registered CLI subcommand (`laya_apple/cli.py`). Not executed here (no benchmark runs per this task's constraints). Owner action: run `scripts/hardware_report.py --quick` once before publishing the community-benchmarks page. |
| Raw data available | ✅ done | `benchmarks/v1.0/` holds the raw per-request data behind every table in `benchmarks/v1.0.md` (parity, latency, concurrency); `research/phase-0-feasibility/raw/` and `research/v0.2-concurrency/` hold the underlying research data. `CONTRIBUTING.md`'s "Non-negotiable rules" requires this for every new/changed benchmark. |
| Headline numbers trace to raw data | ✅ done | `README.md`'s correctness and performance tables cite `benchmarks/v1.0/parity/` and `benchmarks/v1.0.md`, which in turn documents its own reproduction method; `docs/reproducibility.md` (added concurrently) states the exact commands. This audit spot-checked that the README's v1.0 parity table numbers match `benchmarks/v1.0.md`'s corresponding table — they do. |

## Legal and process files

| Item | Status | Evidence / owner action |
|---|---|---|
| License / NOTICE | ✅ done | `LICENSE` (Apache-2.0) and `NOTICE` (adapted components, e.g. `laya_apple/models/modernbert_mlx.py`, `laya_apple/conversion/torch_reference.py`, with upstream revisions) both present at repo root. |
| SECURITY.md | ✅ done | Present, states supported versions (1.0.x), private reporting via GitHub security advisories, and scope (artifact import, worker processes, network access). |
| CONTRIBUTING.md | ✅ done | Rewritten as part of this task: dev setup, the fast/checkpoint/ANE/stress/full-benchmark test tiers with exact commands and what a PR needs per area it touches, how to run parity checks and benchmarks, how to add a hardware result, how to modify a backend, and the existing non-negotiable rules and artifact release policy (all kept, not loosened). |
| Issue / PR templates | ✅ done | `.github/ISSUE_TEMPLATE/bug_report.yml`, `.github/ISSUE_TEMPLATE/config.yml`, `.github/pull_request_template.md` all present. |
| Release notes | ✅ done, verify after merge | `docs/releases/v1.0.0.md` exists (added concurrently) and `CHANGELOG.md`'s `[1.0.0]` entry is filled in. This audit read `docs/releases/v1.0.0.md` for scope conflicts only, not for accuracy. |
| GitHub topics / description | ✅ done, 👤 maintainer must apply | `pyproject.toml`'s `description` and `keywords` updated as part of this task; `scripts/github_seed.sh` (new, this task) sets the same repo description and the topics (`laya`, `apple-silicon`, `mlx`, `coreml`, `apple-neural-engine`, `ane`, `inference`, `heterogeneous-computing`, `machine-learning`) via `gh repo edit`, in `--dry-run` by default. 👤 **Maintainer must run `scripts/github_seed.sh --apply`** — this task never calls `gh` with side effects. |
| Good-first-issues prepared | ✅ done, 👤 maintainer must apply | `.github/labels.yml` (14 labels) and `.github/issue-drafts/` (17 issue drafts: 7 `good first issue`, 5 `help wanted`, 5 `research`) written as part of this task, every one grounded in a real file, number or limitation already in the repo. 👤 **Maintainer must run `scripts/github_seed.sh --apply`** to actually create the labels and issues on GitHub. |
| Benchmark contribution flow | ✅ done, verify after merge | `docs/community-benchmarks.md` (added concurrently) documents the matrix and how to submit a result; `CONTRIBUTING.md`'s new "How to add a hardware result" section (this task) links to it and to the seeded hardware/benchmark issues. |
| Known limitations | ✅ done | `README.md`'s "Limitations" section is substantive and specific (one test machine, non-portable routing thresholds, partial engine isolation with cited numbers, cold-start cost, option-order dependence with cited upstream percentages, and an explicit "not yet measured" list). No changes needed. |
| v1.0 tag / release state | ⏳ pending, 👤 maintainer-only | A local `v1.0.0` git tag exists (along with `v0.1.0`, `v0.2.0`, `v0.3.0`); `origin` is set to `git@github.com:tc3oliver/laya-apple.git`. This audit did not check (and must not check, per this task's scope) whether the tag or any commit has been pushed. 👤 **Maintainer decides**: rewrite history for private files / local paths before the first push (see below), push, then create the GitHub release from `docs/releases/v1.0.0.md`. |

## Maintainer-only items (not actioned by this audit)

These require the maintainer's own judgment, credentials, or an irreversible action, and
were deliberately left undone here:

- **History rewrite before first push.** If any commit reachable from `main` contains
  private files or local machine paths, rewrite history before `origin` ever sees it —
  once pushed, a rewrite requires a force-push and invalidates every clone.
- **Push to `origin`.** This audit made no push and no network-mutating call anywhere.
- **`scripts/github_seed.sh --apply`.** Creates labels and issues and edits repo
  metadata on GitHub. Dry-run by default; verified in dry-run mode as part of this task,
  never run with `--apply` here.
- **Creating the GitHub release from `docs/releases/v1.0.0.md`.** A release is a
  GitHub-side action distinct from the git tag.
- **PyPI publishing decision.** `laya-apple` is not on PyPI today (`README.md`'s
  "Install" section installs from a clone only). Whether to publish is a maintainer
  decision, not evaluated here.
