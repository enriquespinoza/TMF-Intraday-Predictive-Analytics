"""Download point-in-time macroeconomic vintages from FRED/ALFRED.

The initial implementation downloads PAYEMS only so that the vintage
schema and revision behavior can be validated before scaling to all
release-sensitive macroeconomic series.

FRED API output_type=3 returns observations by vintage date with only
new and revised observations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


FRED_OBSERVATIONS_URL = (
    "https://api.stlouisfed.org/"
    "fred/series/observations"
)

RAW_VINTAGE_DIR = Path(
    "data/raw/macro_vintages"
)

DEFAULT_REALTIME_START = "2021-09-01"
DEFAULT_REALTIME_END = "9999-12-31"

# Start with one series. Do not expand this list until PAYEMS
# vintage behavior has been validated.
VINTAGE_SERIES = {
    "nonfarm_payrolls": {
        "series_id": "PAYEMS",
        "category": "labor",
    },
}


def get_fred_api_key() -> str:
    """Load and validate the FRED API key."""

    load_dotenv()

    api_key = os.getenv(
        "FRED_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY is not configured. "
            "Add it to the project's .env file."
        )

    api_key = api_key.strip()

    if len(api_key) != 32:
        raise RuntimeError(
            "FRED_API_KEY must be a "
            "32-character API key."
        )

    if not api_key.isalnum():
        raise RuntimeError(
            "FRED_API_KEY must contain only "
            "letters and numbers."
        )

    return api_key


def request_vintage_observations(
    series_id: str,
    api_key: str,
    realtime_start: str = DEFAULT_REALTIME_START,
    realtime_end: str = DEFAULT_REALTIME_END,
) -> dict[str, Any]:
    """Request new and revised observations from FRED/ALFRED."""

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": realtime_start,
        "realtime_end": realtime_end,
        "observation_start": realtime_start,
        "output_type": 3,
        "sort_order": "asc",
        "limit": 100000,
    }

    response = requests.get(
        FRED_OBSERVATIONS_URL,
        params=params,
        timeout=30,
    )

    if not response.ok:
        message = response.text[:500]

        raise RuntimeError(
            f"FRED request failed for {series_id}: "
            f"HTTP {response.status_code} "
            f"{message}"
    )

    payload = response.json()

    if "observations" not in payload:
        raise RuntimeError(
            f"FRED response for {series_id} "
            "did not contain observations."
        )

    return payload


def parse_vintage_observations(
    payload: dict,
    series_name: str,
    series_id: str,
) -> pd.DataFrame:
    """
    Convert FRED output_type=3 observations from wide vintage format
    into canonical long-form vintage events.

    FRED output_type=3 returns rows similar to:

        {
            "date": "2021-09-01",
            "PAYEMS_20211008": "147553",
            "PAYEMS_20211105": "147788",
            "PAYEMS_20211203": "147855"
        }

    Each vintage column represents the value for the reference period
    as known on that vintage date.

    Output columns:
        series_name
        series_id
        reference_date
        vintage_date
        value_as_known
    """

    observations = payload.get("observations", [])

    if not observations:
        raise ValueError(
            f"No vintage observations returned for {series_name}"
        )

    records = []

    prefix = f"{series_id}_"

    for observation in observations:
        reference_date_raw = observation.get("date")

        if reference_date_raw is None:
            raise ValueError(
                f"Vintage observation missing date for {series_name}"
            )

        reference_date = pd.to_datetime(
            reference_date_raw,
            errors="raise",
        ).normalize()

        for column_name, raw_value in observation.items():
            if column_name == "date":
                continue

            if not column_name.startswith(prefix):
                continue

            vintage_date_text = column_name[len(prefix):]

            try:
                vintage_date = pd.to_datetime(
                    vintage_date_text,
                    format="%Y%m%d",
                    errors="raise",
                ).normalize()
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid vintage column for {series_name}: "
                    f"{column_name}"
                ) from exc

            value_as_known = pd.to_numeric(
                raw_value,
                errors="coerce",
            )

            records.append(
                {
                    "series_name": series_name,
                    "series_id": series_id,
                    "reference_date": reference_date,
                    "vintage_date": vintage_date,
                    "value_as_known": value_as_known,
                }
            )

    if not records:
        raise ValueError(
            f"No vintage columns found for {series_name}"
        )

    data = pd.DataFrame(records)

    data = data.sort_values(
        ["reference_date", "vintage_date"]
    ).reset_index(drop=True)

    return data


def validate_vintage_data(
    data: pd.DataFrame,
    series_name: str,
) -> None:
    """
    Validate canonical long-form macro vintage observations.
    """

    required_columns = {
        "series_name",
        "series_id",
        "reference_date",
        "vintage_date",
        "value_as_known",
    }

    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            f"Vintage data for {series_name} missing columns: "
            f"{sorted(missing)}"
        )

    if data.empty:
        raise ValueError(
            f"Vintage data for {series_name} is empty"
        )

    if data["reference_date"].isna().any():
        raise ValueError(
            f"Vintage data for {series_name} contains missing "
            "reference dates"
        )

    if data["vintage_date"].isna().any():
        raise ValueError(
            f"Vintage data for {series_name} contains missing "
            "vintage dates"
        )

    invalid_timing = (
        data["vintage_date"] < data["reference_date"]
    )

    if invalid_timing.any():
        bad_rows = data.loc[
            invalid_timing,
            [
                "reference_date",
                "vintage_date",
                "value_as_known",
            ],
        ]

        raise ValueError(
            f"Vintage data for {series_name} contains vintage dates "
            f"before reference dates:\n{bad_rows}"
        )

    duplicates = data.duplicated(
        subset=[
            "series_name",
            "reference_date",
            "vintage_date",
        ],
        keep=False,
    )

    if duplicates.any():
        bad_rows = data.loc[
            duplicates,
            [
                "series_name",
                "reference_date",
                "vintage_date",
                "value_as_known",
            ],
        ]

        raise ValueError(
            f"Vintage data for {series_name} contains duplicate "
            f"vintage events:\n{bad_rows}"
        )


def save_vintage_data(
    data: pd.DataFrame,
    series_name: str,
    output_dir: Path = RAW_VINTAGE_DIR,
) -> Path:
    """Save one raw vintage history."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_dir
        / f"{series_name}.csv"
    )

    data.to_csv(
        path,
        index=False,
    )

    return path


def download_vintage_series(
    series_name: str,
    config: dict[str, str],
    api_key: str,
    realtime_start: str = DEFAULT_REALTIME_START,
    realtime_end: str = DEFAULT_REALTIME_END,
) -> tuple[pd.DataFrame, Path]:
    """Download, validate, and save one vintage series."""

    series_id = config[
        "series_id"
    ]

    payload = (
        request_vintage_observations(
            series_id=series_id,
            api_key=api_key,
            realtime_start=realtime_start,
            realtime_end=realtime_end,
        )
    )

    data = (
        parse_vintage_observations(
            payload=payload,
            series_name=series_name,
            series_id=series_id,
        )
    )

    validate_vintage_data(
        data=data,
        series_name=series_name,
    )

    path = save_vintage_data(
        data=data,
        series_name=series_name,
    )

    return data, path


def main() -> None:
    """Download the initial ALFRED vintage dataset."""

    api_key = get_fred_api_key()

    print(
        "\nDownloading macro vintages"
    )
    print("=" * 72)

    for (
        series_name,
        config,
    ) in VINTAGE_SERIES.items():

        data, path = (
            download_vintage_series(
                series_name=series_name,
                config=config,
                api_key=api_key,
            )
        )

        revisions = (
            data.groupby(
                "reference_date"
            )
            .size()
        )

        revised_periods = int(
            (revisions > 1).sum()
        )

        print(
            f"{series_name}"
        )
        print(
            f"  series_id: "
            f"{config['series_id']}"
        )
        print(
            f"  vintage rows: "
            f"{len(data)}"
        )
        print(
            f"  reference periods: "
            f"{data['reference_date'].nunique()}"
        )
        print(
            f"  revised periods: "
            f"{revised_periods}"
        )
        print(
            f"  vintage range: "
            f"{data['vintage_date'].min().date()} "
            f"-> "
            f"{data['vintage_date'].max().date()}"
        )
        print(
            f"  saved: {path}"
        )


if __name__ == "__main__":
    main()