"""Offline regression tests for Schwab daily overlap retention.

These tests never call Schwab, issue forecasts, or place orders.
"""

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from scripts import download_schwab_tmf as downloader


TAIL = (
    date(2026, 9, 29),
    date(2026, 9, 30),
    date(2026, 10, 1),
    date(2026, 10, 2),
    date(2026, 10, 5),
)


def write_frozen_history(tmp_path, dates=None):
    if dates is None:
        dates = (date(2026, 9, 28),) + TAIL
    path = tmp_path / "TMF_1d.csv"
    rows = [
        datetime.combine(day, datetime.min.time(),
                         tzinfo=ZoneInfo("America/New_York")).isoformat()
        for day in dates
    ]
    pd.DataFrame({"timestamp": rows}).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("captured_at", [
    "2026-11-03T22:30:25+00:00",
    "2026-11-04T22:30:18+00:00",
    "2026-11-05T23:30:00+00:00",
    "2027-02-03T22:30:00+00:00",
])
def test_fixed_daily_anchor_survives_rolling_window(tmp_path, captured_at):
    sessions = downloader.required_history_sessions(
        write_frozen_history(tmp_path)
    )
    assert sessions == TAIL
    end = datetime.fromisoformat(captured_at)
    daily, intraday = downloader.request_start_times(end, sessions)

    assert daily == datetime(
        2026, 9, 29, tzinfo=ZoneInfo("America/New_York")
    )
    assert daily.utcoffset() == timedelta(hours=-4)
    assert intraday == end - timedelta(days=35)
    assert intraday.tzinfo is timezone.utc


def test_missing_reference_session_fails_closed():
    observed = pd.DataFrame({
        "market_date": [TAIL[0], TAIL[1], TAIL[3], TAIL[4]]
    })
    with pytest.raises(ValueError, match="2026-10-01"):
        downloader.require_daily_overlap(observed, TAIL)


def test_all_required_sessions_accepted():
    observed = pd.DataFrame({"market_date": list(TAIL)})
    downloader.require_daily_overlap(observed, TAIL)


@pytest.mark.parametrize("dates", [
    (date(2026, 10, 2), date(2026, 10, 5)),
    TAIL + (TAIL[-1],),
    tuple(reversed(TAIL)),
])
def test_invalid_historical_sessions_fail_closed(tmp_path, dates):
    path = write_frozen_history(tmp_path, dates)
    with pytest.raises(ValueError, match="insufficient or invalid"):
        downloader.required_history_sessions(path)


def test_downloader_uses_fixed_daily_and_rolling_intraday(tmp_path, monkeypatch):
    history = write_frozen_history(tmp_path)
    output = tmp_path / "schwab"
    monkeypatch.setattr(downloader, "HISTORY", history)
    monkeypatch.setattr(downloader, "OUTPUT_DIR", output)
    calls = []

    def candle(day):
        stamp = datetime.combine(
            day, datetime.min.time(), tzinfo=timezone.utc
        ) + timedelta(hours=14, minutes=30)
        return {
            "datetime": int(stamp.timestamp() * 1000),
            "open": 30.0, "high": 30.5, "low": 29.5,
            "close": 30.1, "volume": 1000,
        }

    class Response:
        def __init__(self, candles):
            self.candles = candles

        def raise_for_status(self):
            pass

        def json(self):
            return {"candles": self.candles}

    class Client:
        def get_price_history_every_day(self, ticker, **kwargs):
            calls.append(("1d", ticker, kwargs))
            return Response([candle(day) for day in TAIL] + [
                candle(date(2026, 11, 4))
            ])

        def get_price_history_every_five_minutes(self, ticker, **kwargs):
            calls.append(("5m", ticker, kwargs))
            return Response([candle(date(2026, 11, 4))])

    monkeypatch.setattr(downloader, "create_client", lambda: Client())
    downloader.main()

    assert len(calls) == 2
    daily_kind, daily_ticker, daily_args = calls[0]
    intra_kind, intra_ticker, intra_args = calls[1]
    assert (daily_kind, daily_ticker) == ("1d", "TMF")
    assert (intra_kind, intra_ticker) == ("5m", "TMF")
    assert daily_args["start_datetime"] == datetime(
        2026, 9, 29, tzinfo=ZoneInfo("America/New_York")
    )
    assert (
        intra_args["end_datetime"] - intra_args["start_datetime"]
        == timedelta(days=35)
    )
    assert daily_args["end_datetime"] == intra_args["end_datetime"]
    assert len(list(output.glob("TMF_1d_*.csv"))) == 1
    assert len(list(output.glob("TMF_5m_*.csv"))) == 1
