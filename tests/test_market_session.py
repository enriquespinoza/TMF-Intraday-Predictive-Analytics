"""Boundary tests for the restart-safe NYSE forecast issuance window."""

import pandas as pd
import pytest

from scripts.market_session import (
    require_issuance_window,
    resolve_market_session,
)
from scripts.validate_tmf_session import validate_session
from scripts.validate_tmf_snapshot_pair import validate_snapshot_pair


@pytest.mark.parametrize(
    "now,status,origin",
    [
        # Oct 21 / Oct 22: normal Eastern Daylight Time session.
        ("2026-10-21T20:15:00Z", "NOT_READY", "2026-10-21"),
        ("2026-10-21T20:29:59Z", "NOT_READY", "2026-10-21"),
        ("2026-10-21T20:30:00Z", "READY", "2026-10-21"),
        ("2026-10-21T22:55:00Z", "READY", "2026-10-21"),
        ("2026-10-22T05:00:00Z", "READY", "2026-10-21"),
        ("2026-10-22T13:29:59Z", "READY", "2026-10-21"),
        ("2026-10-22T13:30:00Z", "MISSED_WINDOW", "2026-10-21"),
        ("2026-10-22T14:00:00Z", "MISSED_WINDOW", "2026-10-21"),
        # Weekend: Friday remains the most recent trading session.
        ("2026-10-24T15:00:00Z", "READY", "2026-10-23"),
        ("2026-10-26T13:30:00Z", "MISSED_WINDOW", "2026-10-23"),
        # Early close on the Friday after Thanksgiving (EST).
        ("2026-11-27T18:29:59Z", "NOT_READY", "2026-11-27"),
        ("2026-11-27T18:30:00Z", "READY", "2026-11-27"),
        ("2026-11-29T12:00:00Z", "READY", "2026-11-27"),
        ("2026-11-30T14:30:00Z", "MISSED_WINDOW", "2026-11-27"),
        # NYSE is closed for Christmas Day.
        ("2026-12-25T15:00:00Z", "READY", "2026-12-24"),
    ],
)
def test_session_boundary(now, status, origin):
    result = resolve_market_session(now)
    assert result.status == status
    assert str(result.session) == origin


def test_timezone_naive_time_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_market_session("2026-10-22 01:00:00")


def test_wrong_origin_rejected():
    with pytest.raises(ValueError, match="prospective post-close"):
        require_issuance_window(
            "2026-10-20", "2026-10-22T05:00:00Z"
        )


def test_missed_issuance_window_rejected():
    with pytest.raises(ValueError, match="MISSED_WINDOW"):
        require_issuance_window(
            "2026-10-21", "2026-10-22T13:30:00Z"
        )


def test_overnight_snapshot_session_validation(tmp_path):
    daily = tmp_path / "TMF_daily.csv"
    pd.DataFrame([{
        "market_date": "2026-10-21",
        "open": 28.0,
        "high": 28.5,
        "low": 27.9,
        "close": 28.22,
        "volume": 100000,
    }]).to_csv(daily, index=False)

    result = validate_session(
        daily,
        now="2026-10-22T05:00:00Z",
        expected_session="2026-10-21",
    )
    assert result["market_date"] == "2026-10-21"

    with pytest.raises(ValueError, match="MISSED_WINDOW"):
        validate_session(
            daily,
            now="2026-10-22T13:30:00Z",
            expected_session="2026-10-21",
        )


def test_overnight_snapshot_pair(tmp_path):
    # Construct a synthetic complete 78-bar regular NYSE session.
    import pandas_market_calendars as mcal

    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(
        start_date="2026-10-21", end_date="2026-10-21"
    )
    opens = schedule.iloc[0]
    bars = pd.date_range(
        opens["market_open"], opens["market_close"],
        freq="5min", inclusive="left",
    )
    assert len(bars) == 78

    daily = tmp_path / "TMF_1d_20261021T225435Z.csv"
    intraday = tmp_path / "TMF_5m_20261021T225435Z.csv"

    pd.DataFrame([{
        "market_date": "2026-10-21", "open": 28.0,
        "high": 28.5, "low": 27.9, "close": 28.22,
        "volume": 100000,
    }]).to_csv(daily, index=False)

    pd.DataFrame([{
        "timestamp_utc": stamp.isoformat(),
        "market_date": "2026-10-21",
        "open": 28.0, "high": 28.5, "low": 27.9,
        "close": 28.22, "volume": 1000,
    } for stamp in bars]).to_csv(intraday, index=False)

    result = validate_snapshot_pair(
        daily,
        now="2026-10-22T05:00:00Z",
        expected_session="2026-10-21",
    )
    assert result["observed_bars"] == 78
    assert result["market_date"] == "2026-10-21"

    with pytest.raises(ValueError, match="MISSED_WINDOW"):
        validate_snapshot_pair(
            daily,
            now="2026-10-22T13:30:00Z",
            expected_session="2026-10-21",
        )
