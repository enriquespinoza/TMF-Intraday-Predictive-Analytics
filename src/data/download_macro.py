"""Download raw macroeconomic data for the TMF V3 pipeline.

The macro series themselves are defined in config/macro_series.py.

This module downloads raw FRED observations and preserves them before any
feature engineering or market-date alignment is performed.

Important:
    FRED observations are not automatically point-in-time observations.
    Monthly and quarterly series may subsequently be revised. Release-date
    and vintage handling will be implemented separately before these series
    are permitted into predictive backtests.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas_datareader import data as web

from config.macro_series import MACRO_SERIES


START_DATE = "2021-09-01"

RAW_MACRO_DIR = Path("data/raw/macro")


def download_fred_series(
    series_id: str,
    start_date: str = START_DATE,
) -> pd.DataFrame:
    """Download one raw FRED series."""

    data = web.DataReader(
        series_id,
        "fred",
        start=start_date,
    )

    data = data.reset_index()

    data.columns = [
        "reference_date",
        "value",
    ]

    data["reference_date"] = pd.to_datetime(
        data["reference_date"],
        errors="raise",
    )

    data["value"] = pd.to_numeric(
        data["value"],
        errors="coerce",
    )

    return data


def save_raw_series(
    name: str,
    config: dict,
    data: pd.DataFrame,
) -> Path:
    """Save one raw macro series with identifying metadata."""

    RAW_MACRO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = data.copy()

    output["series_name"] = name
    output["series_id"] = config["series_id"]
    output["category"] = config["category"]
    output["source"] = config["source"]
    output["frequency"] = config["frequency"]
    output["units"] = config["units"]

    column_order = [
        "reference_date",
        "value",
        "series_name",
        "series_id",
        "category",
        "source",
        "frequency",
        "units",
    ]

    output = output[column_order]

    output_path = (
        RAW_MACRO_DIR
        / f"{name}.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    return output_path


def download_macro_registry() -> None:
    """Download all supported raw FRED macro series."""

    fred_series = {
        name: config
        for name, config in MACRO_SERIES.items()
        if config["source"] == "FRED"
    }

    print(
        f"Downloading {len(fred_series)} "
        "FRED macro series..."
    )

    failures: list[tuple[str, str, str]] = []

    for name, config in fred_series.items():
        series_id = config["series_id"]

        print(
            f"Downloading {name} "
            f"({series_id})..."
        )

        try:
            data = download_fred_series(
                series_id=series_id,
            )

            output_path = save_raw_series(
                name=name,
                config=config,
                data=data,
            )

            missing = int(
                data["value"].isna().sum()
            )

            print(
                f"  rows={len(data):,}, "
                f"missing={missing:,}, "
                f"saved={output_path}"
            )

        except Exception as exc:
            failures.append(
                (
                    name,
                    series_id,
                    str(exc),
                )
            )

            print(
                f"  FAILED: {exc}"
            )

    print()

    if failures:
        print(
            f"Completed with "
            f"{len(failures)} failure(s):"
        )

        for name, series_id, error in failures:
            print(
                f"  {name} "
                f"({series_id}): "
                f"{error}"
            )

        raise RuntimeError(
            "One or more macro series "
            "failed to download."
        )

    print(
        "All macro series downloaded successfully."
    )


if __name__ == "__main__":
    download_macro_registry()