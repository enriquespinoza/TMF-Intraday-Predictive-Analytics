"""Fail-closed validation of a completed TMF trading session."""

import argparse
from datetime import timezone
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from scripts.market_session import resolve_market_session


NY = "America/New_York"
CALENDAR = mcal.get_calendar("NYSE")
FINALIZATION_DELAY = pd.Timedelta(minutes=30)


def validate_session(snapshot, now=None, expected_session=None):
    if now is None:
        now = pd.Timestamp.now(tz="UTC")
    else:
        now = pd.Timestamp(now)
        if now.tzinfo is None:
            raise ValueError("Validation time must be timezone-aware")
        now = now.tz_convert("UTC")

    window = resolve_market_session(now)
    if window.status != "READY":
        raise ValueError(
            f"{window.status}: outside prospective post-close "
            "validation window"
        )

    today = window.session
    if (
        expected_session is not None
        and str(expected_session) != str(today)
    ):
        raise ValueError("Validated session mismatch")

    market_close = window.finalized_at_utc - FINALIZATION_DELAY

    data = pd.read_csv(snapshot)

    required = {
        "market_date", "open", "high",
        "low", "close", "volume"
    }

    if not required.issubset(data.columns):
        raise ValueError("Missing required Schwab columns")

    if data.empty:
        raise ValueError("Empty Schwab snapshot")

    dates = pd.to_datetime(
        data["market_date"],
        errors="raise",
    ).dt.date

    if dates.duplicated().any():
        raise ValueError("Duplicate market dates")

    if data[list(required - {"market_date"})].isna().any().any():
        raise ValueError("Missing OHLCV values")

    latest = max(dates)

    if latest != today:
        raise ValueError(
            f"Stale or mismatched session: "
            f"latest={latest}, expected={today}"
        )

    latest_row = data.loc[dates == latest].iloc[0]

    if float(latest_row["close"]) <= 0:
        raise ValueError("Invalid closing price")

    return {
        "market_date": str(latest),
        "close": float(latest_row["close"]),
        "market_close_utc": market_close.isoformat(),
        "validated_at_utc": now.isoformat(),
        "snapshot": Path(snapshot).name,
        "status": "POST_CLOSE_TIME_AND_DATE_VALIDATED",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()

    result = validate_session(args.snapshot)

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
