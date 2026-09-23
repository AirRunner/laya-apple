"""v0.3 local capability profiles: which routing table applies, and that a local one is honoured."""

from __future__ import annotations

import json

import pytest

from laya_apple import Laya, routing
from laya_apple import model as model_mod
from laya_apple.artifacts import platform_profile
from laya_apple.derivation import derive_model
from laya_apple.profiles import PROFILE_FORMAT, PROFILE_VERSION, load_local, local_profile_path, profile_key
from laya_apple.registry import models, routing_table
from laya_apple.workload import make_request

pytestmark = [pytest.mark.integration, pytest.mark.ane]
MODEL = "laya-typed-decisions"


def _write_local(entry: dict, model=MODEL, platform=None):
    path = local_profile_path(platform)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"revision": models()[model].revision, **entry}  # tied to the pinned revision by default
    path.write_text(
        json.dumps(
            {
                "format": PROFILE_FORMAT,
                "format_version": PROFILE_VERSION,
                "platform": platform or platform_profile(),
                "models": {model: entry},
            }
        )
    )
    return path


@pytest.fixture
def local_cache(tmp_path, monkeypatch):
    """Profiles in a temp cache; artifacts stay where they are (LAYA_APPLE_CACHE unchanged)."""
    from laya_apple import profiles

    monkeypatch.setattr(profiles, "cache_root", lambda: tmp_path)
    return tmp_path


def test_derivation_reproduces_the_shipped_table():
    table = routing_table()["models"]
    for name, spec in models().items():
        row = table[name]
        par = {d["bucket"]: (d["parity_pass"], d["parity_prob_max_abs"]) for d in row["decisions"]}
        ane = {d["bucket"]: d["ane_p50_ms"] for d in row["decisions"]}
        mlx = {d["mlx_p50_ms_at"]: d["mlx_p50_ms"] for d in row["decisions"]}
        again = derive_model(spec.ane_buckets, par, ane, mlx)
        assert again["auto_ane_buckets"] == row["auto_ane_buckets"]


def test_profile_key_is_filesystem_safe():
    key = profile_key({"soc": "Apple M4 Max", "macos": "26.6.2", "coremltools": "9.0"})
    assert key == "Apple_M4_Max-macos26-coremltools9.0"


def test_local_profile_ignored_on_a_shipped_profile(local_cache):
    _write_local({**routing_table()["models"][MODEL], "auto_ane_buckets": [64], "auto_ane_max_len": 64})
    laya = Laya.from_pretrained(MODEL, local_files_only=True)
    assert laya.routing_profile == "shipped"
    assert laya.ane_state.buckets == tuple(models()[MODEL].auto_ane_buckets)


def test_unvalidated_platform_without_profile_is_mlx_only(local_cache, monkeypatch):
    monkeypatch.setattr(model_mod, "platform_validated", lambda profile=None: False)
    laya = Laya.from_pretrained(MODEL, local_files_only=True)
    assert laya.routing_profile is None
    state, qs = make_request(laya.tokenizer, laya.config, 64, 1)
    r = laya.predict(context=state, questions=qs)
    assert r.runtime.device == "gpu" and r.runtime.routing_reason == routing.PLATFORM_NOT_VALIDATED


def test_unvalidated_platform_uses_a_matching_local_profile(local_cache, monkeypatch):
    monkeypatch.setattr(model_mod, "platform_validated", lambda profile=None: False)
    base = routing_table()["models"][MODEL]
    path = _write_local({**base, "auto_ane_buckets": [64], "auto_ane_max_len": 64})
    assert load_local(MODEL)["source"] == str(path)
    laya = Laya.from_pretrained(MODEL, local_files_only=True)
    assert laya.routing_profile == f"local:{path}"
    assert laya.ane_state.buckets == (64,)
    for L, device, reason in ((64, "ane", routing.ANE_AUTO), (96, "gpu", routing.EXCEEDS_AUTO_RANGE)):
        state, qs = make_request(laya.tokenizer, laya.config, L, 1)
        r = laya.predict(context=state, questions=qs)
        assert (r.runtime.device, r.runtime.routing_reason) == (device, reason)


def test_stale_local_profile_wrong_revision_is_ignored_with_warning(local_cache):
    base = routing_table()["models"][MODEL]
    path = _write_local({**base, "revision": "0" * 40})
    with pytest.warns(RuntimeWarning, match="stale local profile"):
        assert load_local(MODEL) is None
    assert path.exists()  # ignored, not deleted


def test_stale_local_profile_wrong_artifact_hash_is_ignored_with_warning(local_cache):
    spec = models()[MODEL]
    base = routing_table()["models"][MODEL]
    bucket = spec.ane_buckets[0]
    path = _write_local({**base, "artifacts": {str(bucket): "0" * 64}})
    with pytest.warns(RuntimeWarning, match="stale local profile"):
        assert load_local(MODEL) is None
    assert path.exists()


def test_local_profile_with_matching_revision_and_artifacts_is_used(local_cache):
    from laya_apple.artifacts import artifact_dir

    spec = models()[MODEL]
    base = routing_table()["models"][MODEL]
    bucket = spec.ane_buckets[0]
    manifest_path = artifact_dir(spec, bucket) / "manifest.json"
    if not manifest_path.exists():
        pytest.skip(f"no built ANE artifact for {MODEL} L{bucket} in this cache")
    actual_sha = json.loads(manifest_path.read_text())["integrity"]["artifact_sha256"]
    path = _write_local({**base, "artifacts": {str(bucket): actual_sha}})
    assert load_local(MODEL)["source"] == str(path)


def test_local_profile_for_another_machine_is_ignored(local_cache, monkeypatch):
    monkeypatch.setattr(model_mod, "platform_validated", lambda profile=None: False)
    other = dict(platform_profile(), coremltools="0.0")
    path = _write_local({**routing_table()["models"][MODEL]}, platform=other)
    # written under the other profile's key and with the other profile inside: not used here
    assert path != local_profile_path()
    laya = Laya.from_pretrained(MODEL, local_files_only=True)
    assert laya.routing_profile is None
