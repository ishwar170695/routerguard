"""Guards that stop a broken Colab environment from producing silent garbage.

The specific hazard: `pip install datasets` upgrades huggingface-hub past 1.0, after
which transformers 4.48.3 cannot be imported at all. It bit us once already.
"""

import pytest

from routerguard import env

GOOD = {
    "python": "3.10.12",
    "torch": "2.14.1",
    "transformers": "4.48.3",
    "huggingface-hub": "0.36.2",
    "datasets": "4.8.5",
    "accelerate": "1.15.0",
    "bitsandbytes": "0.50.2",
    "numpy": "2.2.6",
    "pyyaml": "5.4.1",
}


def _patch(monkeypatch, **overrides):
    versions = dict(GOOD, **overrides)
    monkeypatch.setattr(env, "versions", lambda: versions)
    monkeypatch.setattr(env, "gpu_info", lambda: {"cuda_available": False})


def test_accepts_supported_versions(monkeypatch):
    _patch(monkeypatch)
    report = env.assert_env(strict=False)
    assert report["transformers_is_primary"] is True
    assert not report.get("warnings")


def test_rejects_hub_newer_than_one(monkeypatch):
    _patch(monkeypatch, **{"huggingface-hub": "2.4.0"})
    with pytest.raises(RuntimeError, match="huggingface-hub==2.4.0 is incompatible"):
        env.assert_env(strict=False)


def test_rejects_hub_too_old(monkeypatch):
    _patch(monkeypatch, **{"huggingface-hub": "0.20.3"})
    with pytest.raises(RuntimeError, match="huggingface-hub==0.20.3 is incompatible"):
        env.assert_env(strict=False)


def test_rejects_datasets_five(monkeypatch):
    _patch(monkeypatch, datasets="5.1.0")
    with pytest.raises(RuntimeError, match="datasets==5.1.0 is outside the range"):
        env.assert_env(strict=False)


def test_rejects_wrong_transformers(monkeypatch):
    _patch(monkeypatch, transformers="5.0.0")
    with pytest.raises(RuntimeError, match="transformers==5.0.0 is not supported"):
        env.assert_env(strict=True)


def test_fallback_transformers_is_allowed_but_flagged(monkeypatch):
    _patch(monkeypatch, transformers="4.49.0")
    report = env.assert_env(strict=False)
    assert report["transformers_is_primary"] is False
    assert any("fallback transformers 4.49.0" in w for w in report["warnings"])


def test_install_hint_carries_both_bounds():
    hint = env.INSTALL_HINT
    assert "huggingface-hub>=0.24.0,<1.0" in hint
    assert "datasets>=2.19.0,<5.0.0" in hint
    assert "transformers==4.48.3" in hint
