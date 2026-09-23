#!/usr/bin/env bash
#
# github_seed.sh — seed the GitHub repository's labels, starter issues, description
# and topics from files tracked in this repo.
#
# What it does, in order:
#   1. Reads .github/labels.yml and creates/updates each label (`gh label create --force`).
#   2. Reads each .github/issue-drafts/*.md (YAML front matter: title, labels; the rest
#      of the file is the issue body) and creates an issue for it, unless an issue with
#      the same title already exists (open or closed) — idempotent, safe to re-run.
#   3. Sets the repo description and topics via `gh repo edit`.
#
# It never pushes code and never touches branches, PRs or commits — only labels, issues,
# and repo metadata.
#
# Usage:
#   scripts/github_seed.sh                 # dry run (default): prints what it would do
#   scripts/github_seed.sh --dry-run        # same, explicit
#   scripts/github_seed.sh --apply          # actually calls `gh` and makes the changes
#
# Requires: `gh` authenticated against the target repo, `python3` (used to parse the
# YAML-ish label/front-matter files without a yq dependency), and to be run from the
# repository root (or anywhere inside it — paths below are relative to the repo root,
# resolved via `git rev-parse`).

set -euo pipefail

DRY_RUN=1
for arg in "$@"; do
  case "$arg" in
    --apply)
      DRY_RUN=0
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      sed -n '1,25p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
LABELS_FILE="$REPO_ROOT/.github/labels.yml"
DRAFTS_DIR="$REPO_ROOT/.github/issue-drafts"

REPO_DESCRIPTION="Correctness-validated heterogeneous Laya runtime for Apple Silicon (MLX GPU + Apple Neural Engine)"
REPO_TOPICS=(laya apple-silicon mlx coreml apple-neural-engine ane inference heterogeneous-computing machine-learning)

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

require_gh() {
  if [[ "$DRY_RUN" -eq 0 ]] && ! command -v gh >/dev/null 2>&1; then
    echo "gh CLI not found; install it or run with --dry-run" >&2
    exit 1
  fi
}

# --- 1. Labels ---------------------------------------------------------------

seed_labels() {
  if [[ ! -f "$LABELS_FILE" ]]; then
    echo "No $LABELS_FILE found, skipping labels." >&2
    return
  fi

  echo "== Labels =="
  # Parse labels.yml into "name<TAB>color<TAB>description" lines with python3, so we
  # don't need a yq/PyYAML dependency for this simple, flat structure.
  while IFS=$'\t' read -r name color description; do
    [[ -z "$name" ]] && continue
    run gh label create "$name" --color "$color" --description "$description" --force
  done < <(python3 - "$LABELS_FILE" <<'PY'
import re
import sys

path = sys.argv[1]
name = color = desc = None
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        m = re.match(r'^-\s+name:\s*"(.*)"\s*$', line)
        if m:
            if name is not None:
                print(f"{name}\t{color}\t{desc}")
            name, color, desc = m.group(1), "", ""
            continue
        m = re.match(r'^\s+color:\s*"(.*)"\s*$', line)
        if m and name is not None:
            color = m.group(1)
            continue
        m = re.match(r'^\s+description:\s*"(.*)"\s*$', line)
        if m and name is not None:
            desc = m.group(1)
            continue
    if name is not None:
        print(f"{name}\t{color}\t{desc}")
PY
)
}

# --- 2. Issues -----------------------------------------------------------------

existing_issue_titles() {
  gh issue list --state all --limit 500 --json title --jq '.[].title'
}

seed_issues() {
  if [[ ! -d "$DRAFTS_DIR" ]]; then
    echo "No $DRAFTS_DIR found, skipping issues." >&2
    return
  fi

  echo "== Issues =="

  local existing=""
  if [[ "$DRY_RUN" -eq 0 ]]; then
    existing="$(existing_issue_titles || true)"
  fi

  shopt -s nullglob
  for draft in "$DRAFTS_DIR"/*.md; do
    # Front matter is delimited by leading and second "---" lines.
    local title labels body_file
    title="$(python3 - "$draft" <<'PY'
import re
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    text = f.read()
m = re.match(r'^---\n(.*?)\n---\n', text, re.DOTALL)
front = m.group(1) if m else ""
for line in front.splitlines():
    tm = re.match(r'^title:\s*"(.*)"\s*$', line)
    if tm:
        print(tm.group(1))
        break
PY
)"
    labels="$(python3 - "$draft" <<'PY'
import re
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    text = f.read()
m = re.match(r'^---\n(.*?)\n---\n', text, re.DOTALL)
front = m.group(1) if m else ""
lm = re.search(r'^labels:\s*\[(.*)\]\s*$', front, re.MULTILINE)
if lm:
    items = [x.strip().strip('"') for x in lm.group(1).split(",") if x.strip()]
    print(",".join(items))
PY
)"

    if [[ -z "$title" ]]; then
      echo "Skipping $draft: no title in front matter" >&2
      continue
    fi

    if [[ "$DRY_RUN" -eq 0 ]] && grep -Fxq "$title" <<<"$existing"; then
      echo "Issue already exists, skipping: $title"
      continue
    fi

    body_file="$(mktemp)"
    python3 - "$draft" > "$body_file" <<'PY'
import re
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    text = f.read()
m = re.match(r'^---\n(.*?)\n---\n(.*)$', text, re.DOTALL)
body = m.group(2) if m else text
sys.stdout.write(body)
PY

    local -a label_args=()
    if [[ -n "$labels" ]]; then
      IFS=',' read -ra label_list <<<"$labels"
      for l in "${label_list[@]}"; do
        label_args+=(--label "$l")
      done
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
      printf '[dry-run] gh issue create --title %q --body-file %s --label %s\n' \
        "$title" "$draft (rendered body)" "$labels"
    else
      gh issue create --title "$title" --body-file "$body_file" "${label_args[@]}"
    fi
    rm -f "$body_file"
  done
}

# --- 3. Repo metadata -----------------------------------------------------------

seed_repo_metadata() {
  echo "== Repo description and topics =="
  run gh repo edit --description "$REPO_DESCRIPTION"
  run gh repo edit --add-topic "$(
    IFS=,
    echo "${REPO_TOPICS[*]}"
  )"
}

main() {
  require_gh
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Running in dry-run mode (default). Pass --apply to make changes."
  fi
  seed_labels
  seed_issues
  seed_repo_metadata
}

main "$@"
