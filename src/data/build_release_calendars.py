"""Build series-specific macro release calendars.

This module converts verified economic-release event calendars into the
series-specific calendars used by the V3 point-in-time macro pipeline.

It deliberately does not infer or approximate release dates. Historical
release events must be supplied explicitly from an authoritative source.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.macro_release_calendar import (
    RELEASE_CALENDAR_DIR,
    validate_release_calendar,
)


EVENT_CALENDAR_DIR = Path(
    "data/raw/macro_release_events"
)


EMPLOYMENT_SITUATION_SERIES = [
    "nonfarm_payrolls",
    "unemployment_rate",
    "u6_unemployment_rate",
    "labor_force_participation",
    "part_time_economic_reasons",
    "average_weekly_hours",
    "average_hourly_earnings",
]

EMPLOYMENT_SITUATION_CES_SERIES = [
    "nonfarm_payrolls",
    "average_weekly_hours",
    "average_hourly_earnings",
]

REQUIRED_EVENT_COLUMNS = {
    "event_name",
    "reference_date",
    "release_date",
    "release_time",
    "release_timezone",
}


def validate_event_calendar(
    events: pd.DataFrame,
    expected_event_name: str,
) -> pd.DataFrame:
    """Validate and normalize an economic release-event calendar."""

    missing = (
        REQUIRED_EVENT_COLUMNS
        - set(events.columns)
    )

    if missing:
        raise ValueError(
            "Event calendar missing required "
            f"columns: {sorted(missing)}"
        )

    data = events.copy()

    data["reference_date"] = pd.to_datetime(
        data["reference_date"],
        errors="raise",
    ).dt.normalize()

    data["release_date"] = pd.to_datetime(
        data["release_date"],
        errors="raise",
    ).dt.normalize()

    required_values = [
        "event_name",
        "reference_date",
        "release_date",
        "release_time",
        "release_timezone",
    ]

    for column in required_values:
        if data[column].isna().any():
            raise ValueError(
                "Event calendar contains missing "
                f"{column} values."
            )

    unexpected = (
        set(data["event_name"].unique())
        - {expected_event_name}
    )

    if unexpected:
        raise ValueError(
            f"Expected event_name "
            f"{expected_event_name!r}, found: "
            f"{sorted(unexpected)}"
        )

    duplicates = data.duplicated(
        subset=[
            "event_name",
            "reference_date",
        ],
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            "Event calendar contains duplicate "
            "reference periods."
        )

    if (
        data["release_date"]
        < data["reference_date"]
    ).any():
        raise ValueError(
            "Release date cannot precede "
            "reference date."
        )

    return (
        data.sort_values(
            "reference_date"
        )
        .reset_index(drop=True)
    )


def expand_event_calendar(
    events: pd.DataFrame,
    event_name: str,
    series_names: list[str],
) -> dict[str, pd.DataFrame]:
    """Expand one release event into series-specific calendars."""

    validated = validate_event_calendar(
        events=events,
        expected_event_name=event_name,
    )

    calendars: dict[str, pd.DataFrame] = {}

    for series_name in series_names:
        calendar = validated[
            [
                "reference_date",
                "release_date",
                "release_time",
                "release_timezone",
            ]
        ].copy()

        calendar.insert(
            0,
            "series_name",
            series_name,
        )

        calendar = validate_release_calendar(
            calendar=calendar,
            series_name=series_name,
        )

        calendars[series_name] = calendar

    return calendars


def save_series_calendars(
    calendars: dict[str, pd.DataFrame],
    output_dir: Path = RELEASE_CALENDAR_DIR,
) -> list[Path]:
    """Save validated series-specific release calendars."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths: list[Path] = []

    for series_name, calendar in calendars.items():
        path = (
            output_dir
            / f"{series_name}.csv"
        )

        calendar.to_csv(
            path,
            index=False,
        )

        paths.append(path)

    return paths
def add_employment_situation_exceptions(
    calendars: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Add documented series-specific Employment Situation exceptions.

    October 2025 CES establishment data were released with the
    November 2025 Employment Situation on December 16, 2025.

    October 2025 CPS household data were not collected, so this
    exception applies only to the CES series.
    """

    exception = {
        "reference_date": pd.Timestamp(
            "2025-10-01"
        ),
        "release_date": pd.Timestamp(
            "2025-12-16"
        ),
        "release_time": "08:30",
        "release_timezone": (
            "America/New_York"
        ),
    }

    updated = {
        series_name: calendar.copy()
        for series_name, calendar
        in calendars.items()
    }

    for series_name in (
        EMPLOYMENT_SITUATION_CES_SERIES
    ):
        if series_name not in updated:
            continue

        calendar = updated[
            series_name
        ].copy()

        if (
            calendar["reference_date"]
            == exception["reference_date"]
        ).any():
            raise ValueError(
                "October 2025 CES exception "
                f"already exists for {series_name}."
            )

        exception_row = pd.DataFrame(
            [
                {
                    "series_name": series_name,
                    **exception,
                }
            ]
        )

        calendar = pd.concat(
            [
                calendar,
                exception_row,
            ],
            ignore_index=True,
        )

        updated[series_name] = (
            validate_release_calendar(
                calendar=calendar,
                series_name=series_name,
            )
        )

    return updated

def build_employment_situation_calendars(
    event_path: Path | None = None,
    output_dir: Path = RELEASE_CALENDAR_DIR,
) -> list[Path]:
    """Build calendars for Employment Situation series."""

    if event_path is None:
        event_path = (
            EVENT_CALENDAR_DIR
            / "employment_situation.csv"
        )

    if not event_path.exists():
        raise FileNotFoundError(
            "Employment Situation event calendar "
            f"not found: {event_path}"
        )

    events = pd.read_csv(
        event_path
    )

    calendars = expand_event_calendar(
        events=events,
        event_name="employment_situation",
        series_names=EMPLOYMENT_SITUATION_SERIES,
    )

    calendars = (
        add_employment_situation_exceptions(
            calendars
        )
    )

    return save_series_calendars(
        calendars=calendars,
        output_dir=output_dir,
    )


def main() -> None:
    """Build currently supported release calendars."""

    paths = (
        build_employment_situation_calendars()
    )

    print(
        "Built Employment Situation "
        f"calendars: {len(paths)}"
    )

    for path in paths:
        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()