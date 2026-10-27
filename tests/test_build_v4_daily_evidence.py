"""Fail-closed contract tests for V4 daily point-in-time evidence generator."""
import json

import pandas as pd
import pytest

from scripts import build_v4_daily_evidence as subject


def sources(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "INPUT_DIR", tmp_path)
    market = tmp_path / "market.csv"
    treasury = tmp_path / "treasury.csv"
    dates = pd.bdate_range("2026-06-01", periods=90)
    market_rows = [
        {
            "market_date": d.date().isoformat(),
            "open": 20.0, "high": 21.0, "low": 19.0,
            "close": 20.0, "volume": 100,
            "available_at_utc": (d + pd.Timedelta(hours=22)).isoformat() + "Z",
        } for d in dates
    ]
    pd.DataFrame(market_rows).to_csv(market, index=False)
    treasury_rows = [
        {
            "reference_date": d.date().isoformat(),
            "yield_30y": 4.0, "real_yield_10y": 2.0,
            "available_at_utc": (d + pd.Timedelta(hours=21)).isoformat() + "Z",
        } for d in dates
    ]
    pd.DataFrame(treasury_rows).to_csv(treasury, index=False)
    return market, treasury


def test_duplicated_market_date_rejected(tmp_path, monkeypatch):
    market, _ = sources(tmp_path, monkeypatch)
    data = pd.read_csv(market)
    data.loc[1, "market_date"] = data.loc[0, "market_date"]
    data.to_csv(market, index=False)
    with pytest.raises(ValueError, match="Duplicated"):
        subject._read_history(market, subject.MARKET_COLUMNS, "market_date")


def test_naive_availability_rejected(tmp_path, monkeypatch):
    market, _ = sources(tmp_path, monkeypatch)
    data = pd.read_csv(market)
    data.loc[0, "available_at_utc"] = "2026-06-01T22:00:00"
    data.to_csv(market, index=False)
    with pytest.raises(ValueError, match="timezone-aware"):
        subject._read_history(market, subject.MARKET_COLUMNS, "market_date")


def test_future_reference_date_rejected(tmp_path, monkeypatch):
    market, _ = sources(tmp_path, monkeypatch)
    _, data = subject._read_history(
        market, subject.MARKET_COLUMNS, "market_date",
    )
    with pytest.raises(ValueError, match="future-reference-date"):
        subject._reject_unavailable(
            data, "market_date", pd.Timestamp("2026-10-01T22:00Z"),
            "2026-06-01",
        )


def test_observation_unavailable_asof_rejected(tmp_path, monkeypatch):
    market, _ = sources(tmp_path, monkeypatch)
    _, data = subject._read_history(
        market, subject.MARKET_COLUMNS, "market_date",
    )
    with pytest.raises(ValueError, match="unavailable at issuance"):
        subject._reject_unavailable(
            data, "market_date", pd.Timestamp("2026-06-01T21:00Z"),
            data["market_date"].iloc[-1],
        )


def test_asof_rejects_later_treasury_release():
    market = pd.DataFrame({
        "market_date": ["2026-10-26"],
        "_available": [pd.Timestamp("2026-10-26T21:30:00Z")],
    })
    treasury = pd.DataFrame({
        "reference_date": ["2026-10-23", "2026-10-26"],
        "_available": [
            pd.Timestamp("2026-10-23T19:00:00Z"),
            pd.Timestamp("2026-10-26T22:00:00Z"),
        ],
        "change_30y_1d_bp": [1, 2],
        "change_30y_5d_bp": [1, 2],
        "real_yield_10y_change_5d_bp": [1, 2],
    })
    joined = subject._align_asof(market, treasury)
    assert joined.loc[0, "reference_date"] == "2026-10-23"
    assert joined.loc[0, "treasury_age_days"] == 3


def test_asof_uses_same_day_only_if_available():
    market = pd.DataFrame({
        "market_date": ["2026-10-26"],
        "_available": [pd.Timestamp("2026-10-26T21:30:00Z")],
    })
    treasury = pd.DataFrame({
        "reference_date": ["2026-10-23", "2026-10-26"],
        "_available": [
            pd.Timestamp("2026-10-23T19:00:00Z"),
            pd.Timestamp("2026-10-26T21:00:00Z"),
        ],
        "change_30y_1d_bp": [1, 2],
        "change_30y_5d_bp": [1, 2],
        "real_yield_10y_change_5d_bp": [1, 2],
    })
    joined = subject._align_asof(market, treasury)
    assert joined.loc[0, "reference_date"] == "2026-10-26"


def test_no_write_check_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(subject.ledger, "COMPONENT_DIR", tmp_path / "out")
    monkeypatch.setattr(
        subject, "_compute",
        lambda *args, **kwargs: {
            "snapshot": "TMF_1d_20261026T210550Z.csv",
            "output_components": {
                role: {"role": role, "status": "SYNTHETIC"}
                for role in subject.ROLE_FILENAMES
            },
        },
    )
    result = subject.build(
        "2026-10-26", "daily", "market", "treasury", write=False,
    )
    assert len(result["component_files"]) == 3
    assert not (tmp_path / "out").exists()


def test_write_once(tmp_path, monkeypatch):
    monkeypatch.setattr(subject.ledger, "COMPONENT_DIR", tmp_path / "out")
    monkeypatch.setattr(
        subject, "_compute",
        lambda *args, **kwargs: {
            "snapshot": "TMF_1d_20261026T210550Z.csv",
            "output_components": {
                role: {"role": role, "status": "SYNTHETIC"}
                for role in subject.ROLE_FILENAMES
            },
        },
    )
    subject.build("2026-10-26", "daily", "market", "treasury", write=True)
    assert len(list((tmp_path / "out").glob("*.json"))) == 3
    with pytest.raises(FileExistsError, match="no overwrite"):
        subject.build("2026-10-26", "daily", "market", "treasury", write=True)
