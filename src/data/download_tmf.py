"""Download and append new TMF daily market data."""

from __future__ import annotations

from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


RAW_DATA_DIR = Path(
    "data/raw"
)

DEFAULT_OUTPUT = (
    RAW_DATA_DIR
    / "TMF_1d.csv"
)


def normalize_yahoo_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize Yahoo TMF data to project OHLCV schema."""

    if df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    df = df.reset_index()

    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(
            " ",
            "_",
        )
        for column in df.columns
    ]

    if "date" in df.columns:
        df = df.rename(
            columns={
                "date":
                    "timestamp",
            }
        )

    if "datetime" in df.columns:
        df = df.rename(
            columns={
                "datetime":
                    "timestamp",
            }
        )

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = (
        set(required)
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Yahoo data missing columns: "
            f"{sorted(missing)}"
        )

    df = (
        df[
            required
        ]
        .copy()
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return df


def latest_completed_date() -> date:
    """
    Return the latest date eligible for daily-bar ingestion.

    Current-session data is not accepted before 4:15 PM ET.
    """

    now = datetime.now(
        ZoneInfo(
            "America/New_York"
        )
    )

    if now.time() >= time(
        16,
        15,
    ):
        return now.date()

    return (
        now.date()
        - timedelta(
            days=1
        )
    )


def update_tmf_daily(
    output_path: Path = DEFAULT_OUTPUT,
    through_date: date | None = None,
) -> pd.DataFrame:
    """
    Append only unseen TMF daily bars.

    Existing historical rows are never replaced.
    """

    output_path = Path(
        output_path
    )

    if not output_path.exists():
        raise FileNotFoundError(
            "Existing TMF history is required "
            "for append-only updates: "
            f"{output_path}"
        )

    existing = pd.read_csv(
        output_path
    )

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    if list(
        existing.columns
    ) != required_columns:
        raise ValueError(
            "Existing TMF CSV schema does not "
            "match expected OHLCV schema."
        )

    existing_dates = pd.to_datetime(
        existing[
            "timestamp"
        ],
        utc=True,
    )

    last_date = (
        existing_dates
        .dt
        .tz_convert(
            "America/New_York"
        )
        .dt
        .date
        .max()
    )

    if through_date is None:
        through_date = (
            latest_completed_date()
        )

    if through_date <= last_date:

        print(
            "No update required."
        )

        print(
            "Latest stored date:",
            last_date,
        )

        return pd.DataFrame(
            columns=required_columns
        )

    start_date = (
        last_date
        + timedelta(
            days=1
        )
    )

    # Yahoo's end date is exclusive.
    end_date = (
        through_date
        + timedelta(
            days=1
        )
    )

    ticker = yf.Ticker(
        "TMF"
    )

    downloaded = ticker.history(
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="1d",
        auto_adjust=True,
    )

    new_rows = normalize_yahoo_data(
        downloaded
    )

    if new_rows.empty:

        print(
            "Yahoo returned no new completed "
            "TMF sessions."
        )

        return new_rows

    new_dates = (
        pd.to_datetime(
            new_rows[
                "timestamp"
            ],
            utc=True,
        )
        .dt
        .tz_convert(
            "America/New_York"
        )
        .dt
        .date
    )

    new_rows = (
        new_rows.loc[
            new_dates > last_date
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if new_rows.empty:

        print(
            "No unseen TMF sessions "
            "were returned."
        )

        return new_rows

    # Append bytes rather than rewrite historical rows.
    new_rows.to_csv(
        output_path,
        mode="a",
        header=False,
        index=False,
    )

    validation = pd.read_csv(
        output_path
    )

    validation_dates = pd.to_datetime(
        validation[
            "timestamp"
        ],
        utc=True,
    )

    if (
        validation_dates
        .dt
        .date
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate trading date detected "
            "after append."
        )

    if not (
        validation_dates
        .is_monotonic_increasing
    ):
        raise RuntimeError(
            "TMF timestamps are no longer "
            "chronological."
        )

    print(
        f"Appended {len(new_rows):,} "
        "new TMF daily bar(s)."
    )

    print(
        "Previous last date:",
        last_date,
    )

    print(
        "New last date:",
        pd.to_datetime(
            new_rows[
                "timestamp"
            ].iloc[-1],
            utc=True,
        )
        .tz_convert(
            "America/New_York"
        )
        .date(),
    )

    return new_rows


if __name__ == "__main__":

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    appended = update_tmf_daily()

    if not appended.empty:

        print()
        print(
            appended.to_string(
                index=False
            )
        )