"""Validate a post-close Schwab daily / 5-minute snapshot pair."""

import argparse
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from scripts.validate_tmf_session import (
    FINALIZATION_DELAY,
    validate_session,
)


def validate_snapshot_pair(daily_path, now=None):
    daily_path = Path(daily_path)

    prefix = "TMF_1d_"
    if not daily_path.stem.startswith(prefix):
        raise ValueError("Invalid daily snapshot filename")

    snapshot_id = daily_path.stem[len(prefix):]

    intraday_path = daily_path.with_name(
        f"TMF_5m_{snapshot_id}.csv"
    )

    if not intraday_path.is_file():
        raise FileNotFoundError(intraday_path)

    now = (
        pd.Timestamp.now(tz="UTC")
        if now is None
        else pd.Timestamp(now)
    )

    if now.tzinfo is None:
        raise ValueError("Validation time must be timezone-aware")

    now = now.tz_convert("UTC")

    # Existing check: today must be an NYSE session
    # whose post-close waiting period has elapsed.
    daily_result = validate_session(daily_path, now=now)
    market_date = pd.Timestamp(
        daily_result["market_date"]
    ).date()

    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(
        start_date=market_date,
        end_date=market_date,
    )

    market_open = schedule.iloc[0]["market_open"]
    market_close = schedule.iloc[0]["market_close"]

    # Verify when Schwab snapshot was created.
    snapshot_time = pd.to_datetime(
        snapshot_id,
        format="%Y%m%dT%H%M%SZ",
        utc=True,
        errors="raise",
    )

    earliest = market_close + FINALIZATION_DELAY

    if snapshot_time < earliest:
        raise ValueError("Snapshot was captured too early")

    if snapshot_time > now:
        raise ValueError("Snapshot timestamp is in the future")

    intraday = pd.read_csv(intraday_path)

    required = {
        "timestamp_utc", "market_date",
        "open", "high", "low", "close", "volume",
    }

    if not required.issubset(intraday.columns):
        raise ValueError("Missing five-minute data columns")

    if intraday.empty:
        raise ValueError("Empty five-minute snapshot")

    dates = pd.to_datetime(
        intraday["market_date"]
    ).dt.date

    if max(dates) != market_date:
        raise ValueError("Five-minute snapshot is stale")

    bars = intraday.loc[dates == market_date].copy()

    bars["timestamp_utc"] = pd.to_datetime(
        bars["timestamp_utc"],
        utc=True,
        errors="raise",
    )

    if bars["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate five-minute timestamps")

    # Exact exchange-calendar timestamps, including
    # early-close sessions.
    expected = pd.date_range(
        start=market_open,
        end=market_close,
        freq="5min",
        inclusive="left",
    )

    observed = pd.DatetimeIndex(
        bars["timestamp_utc"]
    ).sort_values()

    missing = expected.difference(observed)
    extra = observed.difference(expected)

    if len(missing) or len(extra):
        raise ValueError(
            f"Five-minute coverage failure: "
            f"{len(missing)} missing, {len(extra)} extra"
        )

    for column in [
        "open", "high", "low", "close", "volume"
    ]:
        values = pd.to_numeric(
            bars[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(f"Invalid {column} values")
        if (values < 0).any():
            raise ValueError(f"Negative {column} values")
        if column != "volume" and (values == 0).any():
            raise ValueError(f"Zero {column} price")

    last_bar = bars.sort_values(
        "timestamp_utc"
    ).iloc[-1]

    daily_close = daily_result["close"]
    intraday_close = float(last_bar["close"])

    difference = abs(daily_close - intraday_close)

    # Temporary research tolerance; investigate recurring
    # disagreement before production authorization.
    if difference > 0.020001:
        raise ValueError(
            f"Daily / five-minute close mismatch: "
            f"${difference:.4f}"
        )

    return {
        "status": "POST_CLOSE_SNAPSHOT_VALIDATED",
        "market_date": str(market_date),
        "snapshot_id": snapshot_id,
        "daily_close": daily_close,
        "intraday_close": intraday_close,
        "close_difference": round(difference, 6),
        "expected_bars": len(expected),
        "observed_bars": len(observed),
        "daily_snapshot": daily_path.name,
        "intraday_snapshot": intraday_path.name,
        "order_generated": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--daily",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    result = validate_snapshot_pair(args.daily)

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
