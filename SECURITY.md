# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | Yes |
| < 1.0 | No |

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Report it privately
through [GitHub's private vulnerability reporting](https://github.com/tc3oliver/laya-apple/security/advisories/new)
on this repository. If that is not available, use GitHub's security advisories for
`tc3oliver/laya-apple` directly.

Include a description of the issue, the affected version, and steps to reproduce if you
have them. You should get an initial response within a few days.

## Scope

`laya-apple` runs models locally on Apple Silicon; it is not a network service. Areas
relevant to a security report:

- **Artifact import.** Importing an exported artifact archive (`laya-apple import` /
  `laya_apple.lifecycle`) extracts a tar archive with Python's `tarfile` `data` extraction
  filter, which rejects absolute paths, `..` traversal and links that would land outside
  the destination. After extraction, the imported artifact is fully re-validated locally
  against its manifest (hash, source revision, weights) before it is trusted; a tampered
  or mismatched archive is quarantined rather than used.
- **Worker processes.** The Neural Engine can run in a separate worker process
  (`laya_apple.executor`). Workers connect back to the parent over a Unix domain socket
  (`AF_UNIX`) authenticated with a per-run key; there is no network listener.
- **Network access.** `laya-apple` makes no network calls of its own beyond downloading
  pinned Hugging Face checkpoint revisions, and each downloaded weights file is verified
  against a pinned SHA-256 before use. Setting `local_files_only=True` (or
  `HF_HUB_OFFLINE=1`) disables all network access; a checkpoint not already cached then
  fails explicitly instead of falling back to the network.

If you find a way for an artifact import, a worker connection, or the offline path to
behave differently from the above, that is exactly what this policy wants reported.
