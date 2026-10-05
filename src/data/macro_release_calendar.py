"""Point-in-time release calendar utilities for V3 macro data."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from config.macro_series import MACRO_SERIES


RELEASE_CALENDAR_DIR = Path(
    "data/raw/macro_release_calendar"
)

REQUIRED_RELEASE_COLUMNS = {
    "series_name",
    "reference_date",
    "release_date",
    "release_time",
    "release_timezone",
}


def requires_release_alignment(
    series_name: str,
) -> bool:
    """Return whether a series requires release-date alignment."""

    if series_name not in MACRO_SERIES:
        raise KeyError(
            f"Unknown macro series: {series_name}"
        )

    return bool(
        MACRO_SERIES[series_name][
            "requires_release_alignment"
        ]
    )


def _validate_release_time(
    value: str,
) -> str:
    """Validate HH:MM release-time format."""

    try:
        parsed = datetime.strptime(
            str(value),
            "%H:%M",
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid release_time: {value}. "
            "Expected HH:MM."
        ) from exc

    return parsed.strftime("%H:%M")


def _validate_timezone(
    value: str,
) -> str:
    """Validate an IANA timezone name."""

    timezone = str(value)

    try:
        ZoneInfo(timezone)
    except Exception as exc:
        raise ValueError(
            "Invalid release_timezone: "
            f"{timezone}"
        ) from exc

    return timezone


def validate_release_calendar(
    calendar: pd.DataFrame,
    series_name: str,
) -> pd.DataFrame:
    """Validate and normalize one macro release calendar."""

    if series_name not in MACRO_SERIES:
        raise KeyError(
            f"Unknown macro series: {series_name}"
        )

    missing_columns = (
        REQUIRED_RELEASE_COLUMNS
        - set(calendar.columns)
    )

    if missing_columns:
        raise ValueError(
            "Release calendar missing required "
            f"columns: {sorted(missing_columns)}"
        )

    data = calendar.copy()

    data["reference_date"] = pd.to_datetime(
        data["reference_date"],
        errors="raise",
    ).dt.normalize()

    data["release_date"] = pd.to_datetime(
        data["release_date"],
        errors="raise",
    ).dt.normalize()

    if data["series_name"].isna().any():
        raise ValueError(
            "Release calendar contains missing "
            "series_name values."
        )

    if data["reference_date"].isna().any():
        raise ValueError(
            "Release calendar contains missing "
            "reference_date values."
        )

    if data["release_date"].isna().any():
        raise ValueError(
            "Release calendar contains missing "
            "release_date values."
        )

    if data["release_time"].isna().any():
        raise ValueError(
            "Release calendar contains missing "
            "release_time values."
        )

    if data["release_timezone"].isna().any():
        raise ValueError(
            "Release calendar contains missing "
            "release_timezone values."
        )

    unexpected_names = (
        set(
            data[
                "series_name"
            ].unique()
        )
        - {series_name}
    )

    if unexpected_names:
        raise ValueError(
            f"Release calendar for {series_name} "
            "contains unexpected series names: "
            f"{sorted(unexpected_names)}"
        )

    data["release_time"] = (
        data["release_time"]
        .astype(str)
        .map(_validate_release_time)
    )

    data["release_timezone"] = (
        data["release_timezone"]
        .astype(str)
        .map(_validate_timezone)
    )

    if (
        data["release_date"]
        < data["reference_date"]
    ).any():
        raise ValueError(
            "Release date cannot precede "
            "reference date."
        )

    duplicates = data.duplicated(
        subset=[
            "series_name",
            "reference_date",
        ],
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            "Release calendar contains duplicate "
            "reference periods."
        )

    return (
        data.sort_values(
            "reference_date"
        )
        .reset_index(drop=True)
    )


def load_release_calendar(
    series_name: str,
    calendar_dir: Path = RELEASE_CALENDAR_DIR,
) -> pd.DataFrame:
    """Load and validate an explicit release calendar."""

    if not requires_release_alignment(
        series_name
    ):
        raise ValueError(
            f"{series_name} does not require "
            "release-date alignment."
        )

    path = (
        calendar_dir
        / f"{series_name}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            "No explicit release calendar found "
            f"for {series_name}: {path}"
        )

    calendar = pd.read_csv(path)

    return validate_release_calendar(
        calendar=calendar,
        series_name=series_name,
    )


def attach_release_dates(
    observations: pd.DataFrame,
    calendar: pd.DataFrame,
    series_name: str,
) -> pd.DataFrame:
    """Attach release metadata to raw macro observations."""

    if not requires_release_alignment(
        series_name
    ):
        raise ValueError(
            f"{series_name} does not require "
            "release-date alignment."
        )

    required_observation_columns = {
        "reference_date",
        "value",
    }

    missing = (
        required_observation_columns
        - set(observations.columns)
    )

    if missing:
        raise ValueError(
            "Macro observations missing required "
            f"columns: {sorted(missing)}"
        )

    obs = observations.copy()

    obs["reference_date"] = pd.to_datetime(
        obs["reference_date"],
        errors="raise",
    ).dt.normalize()

    release_calendar = (
        validate_release_calendar(
            calendar=calendar,
            series_name=series_name,
        )
    )

    release_calendar = release_calendar[
        [
            "reference_date",
            "release_date",
            "release_time",
            "release_timezone",
        ]
    ]

    result = obs.merge(
        release_calendar,
        on="reference_date",
        how="left",
        validate="many_to_one",
    )

    missing_release = (
        result["release_date"].isna()
        & result["value"].notna()
    )

    if missing_release.any():
        dates = (
            result.loc[
                missing_release,
                "reference_date",
            ]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

        raise ValueError(
            f"{series_name} has observations "
            "without known release metadata: "
            f"{dates[:10]}"
        )

    return result


def prepare_market_observed_series(
    observations: pd.DataFrame,
    series_name: str,
) -> pd.DataFrame:
    """Prepare daily market-observed data for alignment."""

    if requires_release_alignment(
        series_name
    ):
        raise ValueError(
            f"{series_name} requires an explicit "
            "release calendar."
        )

    required = {
        "reference_date",
        "value",
    }

    missing = (
        required
        - set(observations.columns)
    )

    if missing:
        raise ValueError(
            "Macro observations missing required "
            f"columns: {sorted(missing)}"
        )

    data = observations.copy()

    data["reference_date"] = pd.to_datetime(
        data["reference_date"],
        errors="raise",
    ).dt.normalize()

    data["market_date_available"] = (
        data["reference_date"]
    )

    return data


def prepare_release_series(
    observations: pd.DataFrame,
    calendar: pd.DataFrame,
    series_name: str,
) -> pd.DataFrame:
    """Prepare release-sensitive data for daily market alignment."""

    data = attach_release_dates(
        observations=observations,
        calendar=calendar,
        series_name=series_name,
    )

    # Current model is daily, so availability is represented
    # at date resolution. release_time and release_timezone are
    # retained for future intraday alignment.
    data["market_date_available"] = (
        data["release_date"]
    )

    return data