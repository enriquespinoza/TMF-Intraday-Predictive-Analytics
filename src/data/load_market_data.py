"""Market-data loading and validation utilities."""

from pathlib import Path

import pandas as pd


REQUIRED_OHLCV_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load an OHLCV CSV and standardize it for the TMF pipeline."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Market data file not found: {path}")

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    # Common timestamp names
    timestamp_candidates = [
        "datetime",
        "timestamp",
        "date",
        "time",
    ]

    timestamp_column = next(
        (
            column
            for column in timestamp_candidates
            if column in df.columns
        ),
        None,
    )

    if timestamp_column is None:
        raise ValueError(
            "Dataset must contain a date, datetime, "
            "timestamp, or time column."
        )

    missing = REQUIRED_OHLCV_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required OHLCV columns: {sorted(missing)}"
        )

    # Standardize timestamp
    df["timestamp"] = pd.to_datetime(
        df[timestamp_column],
        errors="raise",
        utc=True,
    )

    if timestamp_column != "timestamp":
        df = df.drop(columns=[timestamp_column])

    # Ensure chronological ordering
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )

    # Numeric conversion
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    return df