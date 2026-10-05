"""
Point-in-time reconstruction utilities for macroeconomic vintage data.

This module converts raw FRED/ALFRED vintage events into the economic
information set that was available as of a specified date.

Important distinction
---------------------
reference_date:
    Economic period being measured.

vintage_date:
    Date on which FRED/ALFRED records a new or revised value.

value_as_known:
    Value for a reference period as known on that vintage date.

This module does not determine intraday release timing. Official release
calendars remain authoritative for known release dates/times.
"""

from __future__ import annotations

import pandas as pd


REQUIRED_VINTAGE_COLUMNS = {
    "series_name",
    "series_id",
    "reference_date",
    "vintage_date",
    "value_as_known",
}


def validate_vintage_events(
    data: pd.DataFrame,
) -> None:
    """Validate a long-form vintage event table."""

    missing = REQUIRED_VINTAGE_COLUMNS.difference(
        data.columns
    )

    if missing:
        raise ValueError(
            "Vintage event data missing required columns: "
            f"{sorted(missing)}"
        )

    if data.empty:
        raise ValueError(
            "Vintage event data is empty."
        )

    if data["reference_date"].isna().any():
        raise ValueError(
            "Vintage event data contains missing reference dates."
        )

    if data["vintage_date"].isna().any():
        raise ValueError(
            "Vintage event data contains missing vintage dates."
        )

    invalid = (
        data["vintage_date"]
        < data["reference_date"]
    )

    if invalid.any():
        raise ValueError(
            "Vintage event data contains vintage dates "
            "before reference dates."
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
        raise ValueError(
            "Vintage event data contains duplicate "
            "reference-date/vintage-date events."
        )


def prepare_vintage_events(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize and validate a vintage event table.
    """

    result = data.copy()

    result["reference_date"] = pd.to_datetime(
        result["reference_date"],
        errors="raise",
    ).dt.normalize()

    result["vintage_date"] = pd.to_datetime(
        result["vintage_date"],
        errors="raise",
    ).dt.normalize()

    result["value_as_known"] = pd.to_numeric(
        result["value_as_known"],
        errors="coerce",
    )

    validate_vintage_events(result)

    result = result.sort_values(
        [
            "series_name",
            "vintage_date",
            "reference_date",
        ]
    ).reset_index(drop=True)

    return result


def reconstruct_as_of(
    data: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    Reconstruct the complete macro history known as of a date.

    For every reference period, select the most recent vintage whose
    vintage_date is less than or equal to as_of_date.

    This prevents later revisions from leaking backward into historical
    model observations.
    """

    events = prepare_vintage_events(data)

    as_of = pd.Timestamp(
        as_of_date
    ).normalize()

    available = events.loc[
        events["vintage_date"] <= as_of
    ].copy()

    if available.empty:
        return events.iloc[0:0].copy()

    available = available.sort_values(
        [
            "series_name",
            "reference_date",
            "vintage_date",
        ]
    )

    result = (
        available
        .groupby(
            [
                "series_name",
                "series_id",
                "reference_date",
            ],
            as_index=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    result["as_of_date"] = as_of

    return result.sort_values(
        [
            "series_name",
            "reference_date",
        ]
    ).reset_index(drop=True)


def latest_reference_period_as_of(
    data: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    Return the latest reference period available for each series
    as of a specified date.

    This is useful for constructing model-date macro state variables.
    """

    history = reconstruct_as_of(
        data=data,
        as_of_date=as_of_date,
    )

    if history.empty:
        return history

    history = history.sort_values(
        [
            "series_name",
            "reference_date",
        ]
    )

    result = (
        history
        .groupby(
            "series_name",
            as_index=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    return result
def attach_release_calendar(
    vintage_data: pd.DataFrame,
    release_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach official release metadata to macro vintage events.

    Initial observations are tied to the official release calendar
    for their reference period.

    Later revisions retain their ALFRED vintage date as the date
    on which the revised value became available.

    For a daily close-based model:
        market_date_available = vintage_date

    Official release_time and release_timezone are preserved when
    the vintage event occurs on the scheduled release date for that
    reference period.

    Revision events occurring on later dates do not inherit the
    original release time.
    """

    vintages = prepare_vintage_events(vintage_data)

    calendar = release_calendar.copy()

    required_calendar_columns = {
        "series_name",
        "reference_date",
        "release_date",
        "release_time",
        "release_timezone",
    }

    missing = required_calendar_columns.difference(
        calendar.columns
    )

    if missing:
        raise ValueError(
            "Release calendar missing required columns: "
            f"{sorted(missing)}"
        )

    calendar["reference_date"] = pd.to_datetime(
        calendar["reference_date"],
        errors="raise",
    ).dt.normalize()

    calendar["release_date"] = pd.to_datetime(
        calendar["release_date"],
        errors="raise",
    ).dt.normalize()

    duplicates = calendar.duplicated(
        subset=[
            "series_name",
            "reference_date",
        ],
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            "Release calendar contains duplicate "
            "series/reference-date rows."
        )

    result = vintages.merge(
        calendar[
            [
                "series_name",
                "reference_date",
                "release_date",
                "release_time",
                "release_timezone",
            ]
        ],
        on=[
            "series_name",
            "reference_date",
        ],
        how="left",
        validate="many_to_one",
    )

    result = result.rename(
        columns={
            "release_date": "original_release_date",
        }
    )
    #
    # Every initial observation should have release metadata.
    #
    first_vintage = (
        result.groupby(
            [
                "series_name",
                "reference_date",
            ]
        )["vintage_date"]
        .transform("min")
    )

    result["is_initial_release"] = (
        result["vintage_date"] == first_vintage
    )

    missing_initial_calendar = (
        result["is_initial_release"]
        & result["original_release_date"].isna()
    )

    if missing_initial_calendar.any():
        bad = result.loc[
            missing_initial_calendar,
            [
                "series_name",
                "reference_date",
                "vintage_date",
            ],
        ]

        raise ValueError(
            "Initial vintage observations are missing "
            "release-calendar metadata:\n"
            f"{bad.to_string(index=False)}"
        )

    #
    # An initial observation should not appear before its
    # authoritative release date.
    #
    invalid_initial_timing = (
        result["is_initial_release"]
        & (
            result["vintage_date"]
            < result["original_release_date"]
        )
    )

    if invalid_initial_timing.any():
        bad = result.loc[
            invalid_initial_timing,
            [
                "series_name",
                "reference_date",
                "original_release_date",
                "vintage_date",
            ]
        ]

        raise ValueError(
            "Initial vintage predates official release date:\n"
            f"{bad.to_string(index=False)}"
        )

    #
    # Preserve release clock metadata only when the vintage event
    # occurs on that reference period's official release date.
    #
    same_day_as_release = (
        result["vintage_date"]
        == result["original_release_date"]
    )

    result.loc[
        ~same_day_as_release,
        "release_time",
    ] = pd.NA

    result.loc[
        ~same_day_as_release,
        "release_timezone",
    ] = pd.NA

    #
    # Daily-model availability.
    #
    result["market_date_available"] = (
        result["vintage_date"]
    )

    columns = [
        "series_name",
        "series_id",
        "reference_date",
        "original_release_date",
        "release_time",
        "release_timezone",
        "vintage_date",
        "value_as_known",
        "market_date_available",
        "is_initial_release",
    ]

    return (
        result[columns]
        .sort_values(
            [
                "market_date_available",
                "reference_date",
            ]
        )
        .reset_index(drop=True)
    )

def attach_event_release_timing(
    canonical_data: pd.DataFrame,
    event_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach event-level release timing to canonical vintage events.

    Vintage events are matched to macro release events using:

        vintage_date == event_calendar.release_date

    This allows an initial observation and revisions published during
    the same release event to receive the same event timing metadata.
    """

    result = canonical_data.copy()
    events = event_calendar.copy()

    required_canonical_columns = {
        "series_name",
        "series_id",
        "reference_date",
        "original_release_date",
        "vintage_date",
        "value_as_known",
        "market_date_available",
        "is_initial_release",
    }

    missing = required_canonical_columns.difference(
        result.columns
    )

    if missing:
        raise ValueError(
            "Canonical vintage data missing required columns: "
            f"{sorted(missing)}"
        )

    required_event_columns = {
        "event_name",
        "reference_date",
        "release_date",
        "release_time",
        "release_timezone",
    }

    missing = required_event_columns.difference(
        events.columns
    )

    if missing:
        raise ValueError(
            "Event calendar missing required columns: "
            f"{sorted(missing)}"
        )

    result["vintage_date"] = pd.to_datetime(
        result["vintage_date"],
        errors="raise",
    ).dt.normalize()

    events["reference_date"] = pd.to_datetime(
        events["reference_date"],
        errors="raise",
    ).dt.normalize()

    events["release_date"] = pd.to_datetime(
        events["release_date"],
        errors="raise",
    ).dt.normalize()

    duplicate_event_dates = events.duplicated(
        subset=["release_date"],
        keep=False,
    )

    if duplicate_event_dates.any():
        bad = events.loc[
            duplicate_event_dates,
            [
                "event_name",
                "reference_date",
                "release_date",
            ],
        ]

        raise ValueError(
            "Event calendar contains duplicate release dates:\n"
            f"{bad.to_string(index=False)}"
        )

    events = events.rename(
        columns={
            "reference_date": "event_reference_date",
            "release_date": "event_release_date",
            "release_time": "event_release_time",
            "release_timezone": "event_release_timezone",
        }
    )

    result = result.merge(
        events[
            [
                "event_name",
                "event_reference_date",
                "event_release_date",
                "event_release_time",
                "event_release_timezone",
            ]
        ],
        left_on="vintage_date",
        right_on="event_release_date",
        how="left",
        validate="many_to_one",
    )

    return result.sort_values(
        [
            "market_date_available",
            "reference_date",
        ]
    ).reset_index(drop=True)