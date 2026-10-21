"""Tests for completed and interrupted TMF forecast runs."""

import json

import pytest

from scripts import run_daily_forecast as engine


SESSION = "2026-10-20"


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    output = tmp_path / "forward_test"
    sources = tmp_path / "schwab"
    runs = output / "runs"

    output.mkdir()
    sources.mkdir()
    runs.mkdir()

    monkeypatch.setattr(engine, "OUTPUT_DIR", output)
    monkeypatch.setattr(engine, "DATA_DIR", sources)
    monkeypatch.setattr(engine, "RUN_DIR", runs)

    daily = sources / "TMF_1d_test.csv"
    intraday = sources / "TMF_5m_test.csv"

    daily.write_text("frozen daily data")
    intraday.write_text("frozen intraday data")

    forecast = output / (
        f"TMF_V3_SCHWAB_{SESSION}_"
        f"{engine.FINAL_V3_NAME}_{engine.HORIZON}d.json"
    )

    forecast.write_text(json.dumps({
        "model": engine.FINAL_V3_NAME,
        "forecast_origin": SESSION,
        "forecast_target_date":
            engine.forecast_target(SESSION),
        "horizon_trading_days": engine.HORIZON,
        "source_snapshot": daily.name,
        "source_sha256": engine.digest(daily),
        "order_generated": False,
    }))

    manifest = runs / (
        f"TMF_V3_{SESSION}_"
        f"{engine.FINAL_V3_NAME}_{engine.HORIZON}d.json"
    )

    manifest.write_text(json.dumps({
        "status": "GENERATED",
        "session": SESSION,
        "model": engine.FINAL_V3_NAME,
        "horizon_trading_days": engine.HORIZON,
        "forecast": forecast.name,
        "forecast_sha256": engine.digest(forecast),
        "daily_snapshot": daily.name,
        "daily_sha256": engine.digest(daily),
        "intraday_sha256": engine.digest(intraday),
        "order_generated": False,
    }))

    return {
        "forecast": forecast,
        "manifest": manifest,
        "daily": daily,
        "intraday": intraday,
    }


def test_completed_run_is_recognized(completed_run):
    result = engine.audit_existing_run(SESSION)
    assert result == completed_run["forecast"]


def test_missing_manifest_rejected(completed_run):
    completed_run["manifest"].unlink()

    with pytest.raises(RuntimeError, match="INCOMPLETE_RUN"):
        engine.audit_existing_run(SESSION)


def test_orphaned_manifest_rejected(completed_run):
    completed_run["forecast"].unlink()

    with pytest.raises(RuntimeError, match="ORPHANED_MANIFEST"):
        engine.audit_existing_run(SESSION)


def test_modified_forecast_rejected(completed_run):
    completed_run["forecast"].write_text(
        completed_run["forecast"].read_text() + "\n"
    )

    with pytest.raises(RuntimeError, match="forecast hash mismatch"):
        engine.audit_existing_run(SESSION)


def test_modified_market_data_rejected(completed_run):
    completed_run["intraday"].write_text("modified data")

    with pytest.raises(RuntimeError, match="source mismatch"):
        engine.audit_existing_run(SESSION)


def test_invalid_manifest_metadata_rejected(completed_run):
    path = completed_run["manifest"]
    record = json.loads(path.read_text())
    record["status"] = "UNVERIFIED"
    path.write_text(json.dumps(record))

    with pytest.raises(RuntimeError, match="metadata mismatch"):
        engine.audit_existing_run(SESSION)
