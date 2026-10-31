"""Offline safeguards for scheduled rolling TMF paired research."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import automate_daily_research as engine


def window(status="READY", session="2026-10-30"):
    return SimpleNamespace(
        status=status, session=session,
        finalized_at_utc=pd.Timestamp("2026-10-30T20:30:00Z"),
        next_open_utc=pd.Timestamp("2026-11-02T14:30:00Z"),
    )


def test_dry_run_has_no_data_access_or_writes(monkeypatch, capsys):
    monkeypatch.setattr(engine, "resolve_market_session", lambda: window())
    monkeypatch.setattr(engine, "save_event", lambda *a, **kw:
                        (_ for _ in ()).throw(AssertionError("write forbidden")))
    monkeypatch.setattr(engine, "run_child", lambda *a, **kw:
                        (_ for _ in ()).throw(AssertionError("subprocess forbidden")))
    assert engine.run_cycle("unused", "unused", dry_run=True) == "DRY_RUN_ONLY"
    assert '"order_generated": false' in capsys.readouterr().out


def test_nonready_session_never_issues(monkeypatch):
    monkeypatch.setattr(engine, "resolve_market_session",
                        lambda: window("NOT_READY"))
    monkeypatch.setattr(engine, "run_child", lambda *a, **kw:
                        (_ for _ in ()).throw(AssertionError("subprocess forbidden")))
    assert (engine.run_cycle("unused", "unused", dry_run=True)
            == "SKIPPED_NO_PROSPECTIVE_WINDOW")


def test_subprocess_failure_does_not_expose_oauth_url(monkeypatch):
    fake = SimpleNamespace(
        returncode=1,
        stdout="https://api.schwabapi.com/v1/oauth/authorize?secret=example",
        stderr="TOKEN=private",
    )
    monkeypatch.setattr(engine.subprocess, "run", lambda *a, **kw: fake)
    with pytest.raises(engine.ResearchCycleError) as error:
        engine.run_child("Schwab download", "python", "scripts.download_schwab_tmf")
    assert "example" not in str(error.value)
    assert "private" not in str(error.value)
    assert "exited 1" in str(error.value)


def test_future_utc_v3_score_is_deferred(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "V3_DIR", tmp_path)
    forecast = tmp_path / "TMF_V3_SCHWAB_2099-01-05_V3_FINAL_5d.json"
    forecast.write_text('{"forecast_origin": "2099-01-05"}', encoding="utf-8")
    monkeypatch.setattr(engine, "matching_maturities",
                        lambda session: ([("2099-01-05", forecast)], []))
    monkeypatch.setattr(engine, "audit_v3", lambda origin: forecast)
    monkeypatch.setattr(engine, "run_child", lambda *a, **kw:
                        (_ for _ in ()).throw(AssertionError("do not score early")))
    actions = []
    engine.score_maturities("2099-01-12", tmp_path / "daily.csv",
                            "python", "python", actions)
    assert actions == ["V3_SCORE_DEFERRED_UTC_CUTOFF:2099-01-05"]


def test_existing_forecast_source_requires_original_hash(tmp_path, monkeypatch):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    monkeypatch.setattr(engine, "SOURCE", data_dir)
    source = data_dir / "TMF_1d_20261030T212451Z.csv"
    source.write_text("example", encoding="utf-8")
    forecast = tmp_path / "f.json"
    forecast.write_text(
        '{"source_snapshot":"TMF_1d_20261030T212451Z.csv",'
        '"source_sha256":"bad"}', encoding="utf-8"
    )
    with pytest.raises(engine.ResearchCycleError, match="source integrity"):
        engine.original_source(forecast)


def test_preflight_rejects_missing_timesfm_interpreter(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "resolve_market_session", lambda: window())
    monkeypatch.setattr(engine, "save_event", lambda *a, **kw: tmp_path / "receipt")
    with pytest.raises(engine.ResearchCycleError, match="preflight"):
        engine.run_cycle("analytics", tmp_path / "missing.exe")
