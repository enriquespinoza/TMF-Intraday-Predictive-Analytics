"""Download U.S. Treasury constant-maturity yields from FRED."""

from pathlib import Path

import pandas as pd
from pandas_datareader import data as web


RAW_DATA_DIR = Path("data/raw")

OUTPUT_FILE = RAW_DATA_DIR / "treasury_yields_daily.csv"

SERIES = {
    "DGS3MO": "yield_3m",
    "DGS6MO": "yield_6m",
    "DGS1": "yield_1y",
    "DGS2": "yield_2y",
    "DGS3": "yield_3y",
    "DGS5": "yield_5y",
    "DGS7": "yield_7y",
    "DGS10": "yield_10y",
    "DGS20": "yield_20y",
    "DGS30": "yield_30y",
}

START_DATE = "2021-09-01"


def download_treasury_yields(
    start: str = START_DATE,
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily Treasury constant-maturity yields."""

    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    data = web.DataReader(
        list(SERIES.keys()),
        "fred",
        start,
        end,
    )

    data = data.rename(columns=SERIES)

    data.index.name = "timestamp"

    data = data.reset_index()

    return data


def main() -> None:
    """Download and save Treasury yield data."""

    print("Downloading Treasury yields from FRED...")

    data = download_treasury_yields()

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Downloaded {len(data):,} Treasury observations."
    )

    print()
    print(data.tail())

    print()
    print("Missing values:")
    print(data.isna().sum())

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()