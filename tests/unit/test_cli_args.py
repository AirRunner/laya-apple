"""cli.py: prune's reported count, `artifacts import` argument validation, and that main()
does not swallow unrelated ValueErrors as clean CLI errors (code review finding 11)."""

from __future__ import annotations

import pytest

from laya_apple import cli


def test_cmd_artifacts_prune_prints_the_actual_removed_count(monkeypatch, capsys):
    plan = [{"path": "/x/a", "reason": "r", "bytes": 10}, {"path": "/x/b", "reason": "r", "bytes": 20}]
    monkeypatch.setattr("laya_apple.lifecycle.plan_prune", lambda: plan)
    # prune() removes fewer than planned (e.g. another process raced it); the printed count
    # must reflect what prune() actually reports, not len(plan).
    monkeypatch.setattr("laya_apple.lifecycle.prune", lambda p: p[:1])
    a = cli.build_parser().parse_args(["artifacts", "prune", "--yes"])
    assert cli.cmd_artifacts(a) == 0
    out = capsys.readouterr().out
    assert "removed 1 entries" in out


def test_cmd_artifacts_import_without_path_raises_clear_system_exit():
    a = cli.build_parser().parse_args(["artifacts", "import"])
    with pytest.raises(SystemExit, match="archive path"):
        cli.cmd_artifacts(a)


def test_main_wraps_argument_value_error_from_from_pretrained(monkeypatch):
    def boom(*a, **k):
        raise ValueError("bad dtype")

    monkeypatch.setattr("laya_apple.Laya.from_pretrained", boom)
    rc = cli.main(["predict", "laya", "--context", "hi", "--questions", "{}"])
    assert rc == 2


def test_main_does_not_catch_unrelated_value_errors(monkeypatch):
    """Only ValueErrors from the from_pretrained call sites are converted to clean CLI
    errors; a bug elsewhere must keep its traceback (main only catches LayaAppleError)."""

    def boom(a):
        raise ValueError("some internal bug, not user input")

    monkeypatch.setattr(cli, "cmd_info", boom)
    with pytest.raises(ValueError, match="internal bug"):
        cli.main(["info"])
