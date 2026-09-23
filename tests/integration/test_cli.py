from __future__ import annotations

import pytest

from laya_apple.cli import main

pytestmark = pytest.mark.integration


def test_cli_info_returns_zero(capsys):
    assert main(["--offline", "info"]) == 0
    out = capsys.readouterr().out
    assert "laya_apple" in out


def test_cli_info_for_one_model_returns_zero(model_name, capsys):
    assert main(["--offline", "info", model_name]) == 0


def test_cli_predict_returns_zero(model_name, capsys):
    rc = main(
        [
            "--offline",
            "predict",
            model_name,
            "--context",
            "The customer wants a refund.",
            "--questions",
            '{"q": {"type": "noul", "instructions": "refund requested?"}}',
            "--device",
            "gpu",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert '"answers"' in out


def test_cli_artifacts_list_returns_zero(capsys):
    assert main(["artifacts", "list"]) == 0


def test_cli_unsupported_model_returns_2(capsys):
    rc = main(["--offline", "info", "not-a-model"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "UnsupportedModelError" in err
