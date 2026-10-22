import pytest

from scripts.download_schwab_tmf import resolve_token_path


def test_explicit_token_path(tmp_path, monkeypatch):
    path = tmp_path / "schwab_token.json"
    monkeypatch.setenv("SCHWAB_TOKEN_PATH", str(path))

    assert resolve_token_path() == path


def test_linux_requires_explicit_token_path(monkeypatch):
    monkeypatch.delenv("SCHWAB_TOKEN_PATH", raising=False)

    with pytest.raises(RuntimeError, match="must be configured"):
        resolve_token_path(platform_name="posix")


def test_windows_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("SCHWAB_TOKEN_PATH", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    expected = (
        tmp_path
        / "TMF-Intraday-Predictive-Analytics"
        / "schwab_token.json"
    )

    assert resolve_token_path(platform_name="nt") == expected
