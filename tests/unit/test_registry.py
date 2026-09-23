from __future__ import annotations

import pytest

from laya_apple.errors import UnsupportedModelError
from laya_apple.registry import models, resolve


@pytest.mark.parametrize("name", list(models().keys()))
def test_resolve_by_short_name(name):
    assert resolve(name).name == name


@pytest.mark.parametrize("name,spec_name", [(s.repo, s.name) for s in models().values()])
def test_resolve_by_repo(name, spec_name):
    assert resolve(name).name == spec_name


def test_resolve_unknown_model_raises():
    with pytest.raises(UnsupportedModelError):
        resolve("definitely-not-a-real-model")
