"""Synthetic, no-Schwab-network tests for TimesFM outcome evaluation."""

import json

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest

from scripts import score_timesfm25_forecast as scorer


ORIGIN = "2026-10-21"
DATES = ["2026-10-22", "2026-10-23", "2026-10-26",
         "2026-10-27", "2026-10-28"]
CAPTURE = "20261028T210000Z"


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    research = tmp_path / "timesfm25"
    research.mkdir()
    (research / "runs").mkdir()
    sources = tmp_path / "schwab"
    sources.mkdir()
    history = tmp_path / "historical.csv"
    history.write_text("historical input", encoding="utf-8")

    forecast = research / "TMF_TIMESFM25_2026-10-21_5d.json"
    original = {
        "model": "TIMESFM_2P5_200M",
        "forecast_origin": ORIGIN,
        "forecast_target_date": DATES[-1],
        "forecast_dates": DATES,
        "horizon_trading_days": 5,
        "checkpoint_revision": "a" * 40,
        "issued_at_utc": "2026-10-22T13:13:06+00:00",
        "tmf_close": 28.22,
        "median_forecast": [28.1, 28.3, 28.27, 28.37, 28.38],
        "q10_forecast": [27, 27, 27, 27, 27],
        "q90_forecast": [30, 30, 30, 30, 30],
    }
    forecast.write_text(json.dumps(original), encoding="utf-8")
    manifest = research / "runs/TMF_TIMESFM25_2026-10-21_5d_manifest.json"
    manifest.write_text('{"status":"AUDITED_LEGACY"}', encoding="utf-8")

    daily = sources / f"TMF_1d_{CAPTURE}.csv"
    trading_days = [ORIGIN, *DATES]
    close_values = [28.22, 28.15, 28.27, 28.39, 28.49, 28.90]
    pd.DataFrame({
        "market_date": trading_days,
        "open": close_values,
        "high": np.array(close_values) + 0.2,
        "low": np.array(close_values) - 0.2,
        "close": close_values,
        "volume": [10000] * len(close_values),
    }).to_csv(daily, index=False)

    schedule = mcal.get_calendar("NYSE").schedule(
        start_date=DATES[-1], end_date=DATES[-1],
    ).iloc[0]
    bars = pd.date_range(
        schedule["market_open"], schedule["market_close"],
        freq="5min", inclusive="left",
    )
    intraday = sources / f"TMF_5m_{CAPTURE}.csv"
    pd.DataFrame({
        "timestamp_utc": [x.isoformat() for x in bars],
        "market_date": DATES[-1],
        "open": 28.89, "high": 29.0, "low": 28.8,
        "close": 28.9, "volume": 100,
    }).to_csv(intraday, index=False)

    monkeypatch.setattr(scorer, "FORECAST_DIR", research)
    monkeypatch.setattr(scorer, "SOURCE_DIR", sources)
    monkeypatch.setattr(scorer, "HISTORY", history)
    monkeypatch.setattr(scorer, "OUTPUT_DIR", research / "evaluations")
    monkeypatch.setattr(
        scorer, "audit_existing_run",
        lambda session, **kwargs: forecast if session == ORIGIN else None,
    )
    return {
        "daily": daily, "intraday": intraday, "forecast": forecast,
        "research": research, "original_forecast": forecast.read_bytes(),
    }


def evaluate(scenario, *, now="2026-10-28T21:30:00Z", write=False):
    return scorer.evaluate(
        ORIGIN, scenario["daily"], now=now, write=write,
    )


def test_scoring_and_persistence_baseline(scenario):
    path, scored = evaluate(scenario)
    assert path is None
    assert scored["status"] == "EVALUATED"
    assert scored["target_close"] == pytest.approx(28.90)
    assert scored["realized_return_pct"] > 0
    assert scored["direction_correct"] is True
    assert scored["median_mae"] < scored["persistence_mae"]
    assert scored["q10_q90_coverage"] == pytest.approx(1.0)
    assert scored["target_within_q10_q90"] is True
    assert scored["order_generated"] is False
    assert not (scenario["research"] / "evaluations").exists()


def test_immutable_evaluation_and_no_forecast_changes(scenario):
    path, scored = evaluate(scenario, write=True)
    assert path.exists()
    assert json.loads(path.read_text())["forecast_origin"] == ORIGIN
    with pytest.raises(FileExistsError):
        evaluate(scenario, write=True)
    assert scenario["forecast"].read_bytes() == scenario["original_forecast"]


def test_rejects_future_snapshot_capture(scenario):
    with pytest.raises(ValueError, match="future"):
        evaluate(scenario, now="2026-10-28T20:59:59Z")


def test_rejects_missing_target_bar(scenario):
    df = pd.read_csv(scenario["intraday"])
    df = df.iloc[1:]
    df.to_csv(scenario["intraday"], index=False)
    with pytest.raises(ValueError, match="coverage failure"):
        evaluate(scenario)


def test_rejects_missing_intermediate_close(scenario):
    df = pd.read_csv(scenario["daily"])
    df = df.loc[df["market_date"] != "2026-10-26"]
    df.to_csv(scenario["daily"], index=False)
    with pytest.raises(ValueError, match="required trading session"):
        evaluate(scenario)


def test_rejects_modified_origin_close(scenario):
    df = pd.read_csv(scenario["daily"])
    df.loc[0, "close"] = 27.0
    df.to_csv(scenario["daily"], index=False)
    with pytest.raises(ValueError, match="origin close"):
        evaluate(scenario)


def test_unknown_forecast_rejected(scenario):
    with pytest.raises(FileNotFoundError, match="No audited"):
        scorer.evaluate(
            "2026-10-20", scenario["daily"],
            now="2026-10-28T21:30:00Z",
        )


def test_rejects_target_snapshot_outside_source_directory(scenario, tmp_path):
    outside = tmp_path / "TMF_1d_20261028T210000Z.csv"
    outside.write_bytes(scenario["daily"].read_bytes())
    with pytest.raises(ValueError, match="outside Schwab"):
        scorer.evaluate(
            ORIGIN, outside, now="2026-10-28T21:30:00Z",
        )
