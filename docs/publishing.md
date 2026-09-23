# Publishing to PyPI

laya-apple is published to [PyPI](https://pypi.org/project/laya-apple/) by
[`.github/workflows/release.yml`](../.github/workflows/release.yml) using
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/). No PyPI API token or
password exists for this project, in GitHub secrets or anywhere else. Do not create one.

## Build locally

```bash
uv build                         # dist/laya_apple-<version>.tar.gz and -py3-none-any.whl
uvx twine check --strict dist/*
unzip -Z1 dist/*.whl             # only laya_apple/ and laya_apple-<version>.dist-info/
```

The distributions contain the `laya_apple` package only: its code, the bundled routing,
placement and manifest-schema data, and the parity goldens that `laya-apple parity`
uses. Benchmarks, research, tests, scripts, examples and generated Core ML models never
ship. The release workflow fails if any of them appear, or if a local path appears in any
shipped file. It also checks that `LICENSE` and `NOTICE` are in the wheel metadata.

To check that the wheel imports on its own, outside the source tree:

```bash
uv venv /tmp/laya-wheel && uv pip install --python /tmp/laya-wheel/bin/python dist/*.whl
cd /tmp && /tmp/laya-wheel/bin/python -c "import laya_apple; print(laya_apple.__version__, laya_apple.__file__)"
```

## How Trusted Publishing works

1. The `publish` job runs in the GitHub environment `pypi` with `permissions: id-token: write`.
2. `pypa/gh-action-pypi-publish` asks GitHub for a short-lived OIDC token. The token names
   the repository, the workflow file and the environment.
3. PyPI accepts the upload only if those match the trusted publisher registered for the
   `laya-apple` project: owner `tc3oliver`, repository `laya-apple`, workflow
   `release.yml`, environment `pypi`. The token expires after the upload.

A fork, a copy of the workflow in another repository, or another workflow in this
repository cannot publish. The `publish` job also refuses to run outside
`tc3oliver/laya-apple`.

## Configure PyPI (once)

A maintainer does this once, logged in to PyPI.

1. Go to <https://pypi.org/manage/account/publishing/> and, under *Add a new pending
   publisher* → *GitHub*, enter:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `laya-apple` |
   | Owner | `tc3oliver` |
   | Repository name | `laya-apple` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   A *pending* publisher reserves the name. The first successful upload creates the
   project, and the publisher then appears under the project's *Publishing* settings.
2. On GitHub, the `pypi` environment (*Settings → Environments*) should require a
   reviewer. The publish job then waits for an approval before it can request a token.

## Make a release

1. Update `version` in `pyproject.toml` and `__version__` in `laya_apple/__init__.py`, add
   the `CHANGELOG.md` section, and run the release gate
   (`uv run python scripts/release_gate.py`, see [`CONTRIBUTING.md`](../CONTRIBUTING.md)).
2. Commit, then create and push an annotated tag that matches the version exactly:

   ```bash
   git tag -a v1.2.3 -m "laya-apple 1.2.3"
   git push origin main v1.2.3
   ```

3. The push runs `release.yml`. The `build` job checks that the tag equals the package
   version, builds the sdist and wheel, checks their contents and metadata, installs the
   wheel into a clean environment and imports it. The `publish` job then uploads the same
   files to PyPI, after the environment approval if one is configured.
4. Create the GitHub release from the tag, with the notes from `CHANGELOG.md`.

PyPI never accepts the same version twice. If a build is wrong after upload, fix it and
release a new patch version.

### Tags created before the workflow existed

Tags `v1.0.1` and older point at commits that do not contain `release.yml`, so pushing them
cannot trigger it. To publish one of them without moving the tag, run the workflow by hand
from `main`:

```bash
gh workflow run release.yml --ref main -f tag=v1.0.1
```

The workflow then builds that tag's source, with the same checks as a tag push. It refuses
a ref other than `main` and a tag that is not `vX.Y.Z`. It never creates or changes a
tag.
