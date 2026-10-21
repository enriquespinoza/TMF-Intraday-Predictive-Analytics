import pandas as pd
import pandas_market_calendars as mcal
import pytest

from scripts.validate_tmf_snapshot_pair import (
    validate_snapshot_pair,
)


def make_snapshot(tmp_path, market_date="2026-10-20"):
    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(
        start_date=market_date,
        end_date=market_date,
    )

    market_open = schedule.iloc[0]["market_open"]
    market_close = schedule.iloc[0]["market_close"]

    timestamps = pd.date_range(
        start=market_open,
        end=market_close,
        freq="5min",
        inclusive="left",
    )

    snapshot_time = market_close + pd.Timedelta(minutes=60)
    now = market_close + pd.Timedelta(minutes=90)
    snapshot_id = snapshot_time.strftime("%Y%m%dT%H%M%SZ")

    daily = tmp_path / f"TMF_1d_{snapshot_id}.csv"
    intraday = tmp_path / f"TMF_5m_{snapshot_id}.csv"

    pd.DataFrame([{
        "market_date": market_date,
        "open": 28.00,
        "high": 28.30,
        "low": 27.80,
        "close": 28.09,
        "volume": 1000000,
    }]).to_csv(daily, index=False)

    pd.DataFrame([{
        "timestamp_utc": timestamp.isoformat(),
        "market_date": market_date,
        "open": 28.00,
        "high": 28.30,
        "low": 27.80,
        "close": 28.10,
        "volume": 10000,
    } for timestamp in timestamps]).to_csv(
        intraday, index=False
    )

    return daily, intraday, now, len(timestamps)


def test_complete_session_passes(tmp_path):
    daily, _, now, expected = make_snapshot(tmp_path)

    result = validate_snapshot_pair(daily, now=now)

    assert result["status"] == "POST_CLOSE_SNAPSHOT_VALIDATED"
    assert result["expected_bars"] == expected
    assert result["observed_bars"] == expected
    assert result["order_generated"] is False


def test_before_close_rejected(tmp_path):
    daily, _, _, _ = make_snapshot(tmp_path)

    early = pd.Timestamp("2026-10-20T19:00:00Z")

    with pytest.raises(ValueError, match="post-close"):
        validate_snapshot_pair(daily, now=early)


def test_missing_five_minute_bar_rejected(tmp_path):
    daily, intraday, now, _ = make_snapshot(tmp_path)

    data = pd.read_csv(intraday)
    data = data.drop(index=10)
    data.to_csv(intraday, index=False)

    with pytest.raises(ValueError, match="coverage failure"):
        validate_snapshot_pair(daily, now=now)


def test_duplicate_five_minute_bar_rejected(tmp_path):
    daily, intraday, now, _ = make_snapshot(tmp_path)

    data = pd.read_csv(intraday)
    data = pd.concat([data, data.iloc[[0]]])
    data.to_csv(intraday, index=False)

    with pytest.raises(ValueError, match="Duplicate"):
        validate_snapshot_pair(daily, now=now)


def test_daily_close_mismatch_rejected(tmp_path):
    daily, _, now, _ = make_snapshot(tmp_path)

    data = pd.read_csv(daily)
    data.loc[0, "close"] = 27.50
    data.to_csv(daily, index=False)

    with pytest.raises(ValueError, match="close mismatch"):
        validate_snapshot_pair(daily, now=now)


def test_missing_intraday_snapshot_rejected(tmp_path):
    daily, intraday, now, _ = make_snapshot(tmp_path)
    intraday.unlink()

    with pytest.raises(FileNotFoundError):
        validate_snapshot_pair(daily, now=now)


def test_early_close_session(tmp_path):
    # Friday after Thanksgiving: NYSE shortened session.
    daily, _, now, expected = make_snapshot(
        tmp_path, market_date="2026-11-27"
    )

    result = validate_snapshot_pair(daily, now=now)

    assert expected == 42
    assert result["expected_bars"] == 42
    assert result["observed_bars"] == 42
