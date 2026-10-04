"""Download TMF market data for research and development."""

from pathlib import Path

import pandas as pd
import yfinance as yf


RAW_DATA_DIR = Path("data/raw")


def download_tmf(
    interval: str = "1d",
    period: str = "5y",
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Download TMF OHLCV data and optionally save it."""

    ticker = yf.Ticker("TMF")

    df = ticker.history(
        period=period,
        interval=interval,
        auto_adjust=True,
    )

    if df.empty:
        raise RuntimeError(
            f"No TMF data returned for interval={interval}, period={period}"
        )

    df = df.reset_index()

    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]

    if "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})

    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})

    keep_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    df = df[keep_columns].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    if output_path is not None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            output_path,
            index=False,
        )

    return df


if __name__ == "__main__":
    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily = download_tmf(
        interval="1d",
        period="5y",
        output_path=RAW_DATA_DIR / "TMF_1d.csv",
    )

    print(f"Downloaded {len(daily):,} daily TMF bars.")
    print()
    print(daily.tail())