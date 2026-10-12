"""Contract tests for read-only V3/V4 reporting joins."""

import csv
import json
from datetime import date
from pathlib import Path

import pytest

from src.analysis.integrate_v3_v4_assessments import (
    build_assessment,
    load_report,
)


def forecast_file(tmp_path: Path) -> Path:
    path = tmp_path / "forecast.json"
    path.write_text(json.dumps({
        "model": "V3_FINAL",
        "forecast_origin": "2026-10-09",
        "forecast_target_date": "2026-10-16",
        "issued_at_utc": "2026-10-12T01:10:10+00:00",
        "horizon_trading_days": 5,
        "probability_positive_return": 0.6046,
        "tmf_close": 25.99,
        "training_base_rate": 0.4336,
    }), encoding="utf-8")
    return path


def report_file(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["market_date", "state"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_matching_reports_preserve_independent_assessments(tmp_path):
    forecast = forecast_file(tmp_path)
    opp = report_file(tmp_path / "opportunity.csv", [
        {"market_date": "2026-10-09", "state": "WATCH"}
    ])
    risk = report_file(tmp_path / "risk.csv", [
        {"market_date": "2026-10-09", "state": "RED"}
    ])
    result = build_assessment(forecast, opp, risk)
    assert result["forecast"]["probability_positive_return"] == 0.6046
    assert result["v4_opportunity"]["data"]["state"] == "WATCH"
    assert result["v4_risk"]["data"]["state"] == "RED"
    assert result["all_assessments_date_aligned"] is True
    assert result["trading"]["orders_generated"] is False
    assert result["trading"]["status"] == "NOT_AUTHORIZED_READ_ONLY"
    assert result["forecast_source"]["sha256"]


def test_stale_data_fails_closed(tmp_path):
    forecast = forecast_file(tmp_path)
    opp = report_file(tmp_path / "opportunity.csv", [
        {"market_date": "2026-10-08", "state": "ENTRY_CANDIDATE"}
    ])
    result = build_assessment(forecast, opp, tmp_path / "missing.csv")
    assert result["v4_opportunity"]["status"] == "date_mismatch"
    assert result["v4_opportunity"]["latest_available_as_of"] == "2026-10-08"
    assert result["v4_risk"]["status"] == "unavailable"
    assert result["all_assessments_date_aligned"] is False
    assert result["trading"]["orders_generated"] is False


def test_duplicate_session_rejected(tmp_path):
    path = report_file(tmp_path / "opportunity.csv", [
        {"market_date": "2026-10-09", "state": "WATCH"},
        {"market_date": "2026-10-09", "state": "ENTRY_CANDIDATE"},
    ])
    result = load_report(path, date(2026, 10, 9))
    assert result["status"] == "ambiguous_duplicate_date"
    assert result["data"] is None


def test_invalid_forecast_probability_rejected(tmp_path):
    forecast = forecast_file(tmp_path)
    payload = json.loads(forecast.read_text(encoding="utf-8"))
    payload["probability_positive_return"] = 1.25
    forecast.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid V3"):
        build_assessment(forecast, tmp_path / "o.csv", tmp_path / "r.csv")


def test_future_dated_report_never_substituted(tmp_path):
    path = report_file(tmp_path / "risk.csv", [
        {"market_date": "2026-10-12", "state": "GREEN"}
    ])
    result = load_report(path, date(2026, 10, 9))
    assert result["status"] == "date_mismatch"
    assert result["data"] is None
