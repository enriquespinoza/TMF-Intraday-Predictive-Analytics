"""
Point-in-time labor-market features.

V1 supports PAYEMS (nonfarm payrolls) only.

Features are calculated from historical ALFRED vintages rather than
today's revised payroll history.
"""

from __future__ import annotations

import pandas as pd

from src.data.macro_vintage import reconstruct_as_of


PAYEMS_SERIES_NAME = "nonfarm_payrolls"


def _prepare_payems_vintages(
    vintage_data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize and isolate PAYEMS vintage observations."""

    required = {
        "series_name",
        "series_id",
        "reference_date",
        "vintage_date",
        "value_as_known",
    }

    missing = required.difference(vintage_data.columns)

    if missing:
        raise ValueError(
            "PAYEMS vintage data missing required columns: "
            f"{sorted(missing)}"
        )

    result = vintage_data.copy()

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

    result = result.loc[
        result["series_name"] == PAYEMS_SERIES_NAME
    ].copy()

    if result.empty:
        raise ValueError(
            "No nonfarm_payrolls observations found."
        )

    return result


def payroll_state_as_of(
    vintage_data: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    Reconstruct PAYEMS history as known on a date.

    payroll_change_1m is calculated using the point-in-time values
    known on that date.
    """

    payems = _prepare_payems_vintages(
        vintage_data
    )

    history = reconstruct_as_of(
        data=payems,
        as_of_date=as_of_date,
    )

    if history.empty:
        return history

    history = history.sort_values(
        "reference_date"
    ).reset_index(drop=True)

    history["payroll_level"] = (
        history["value_as_known"]
    )

    history["payroll_change_1m"] = (
        history["payroll_level"].diff()
    )

    return history


def payroll_revision_features(
    vintage_data: pd.DataFrame,
    event_date: str | pd.Timestamp,
) -> dict[str, float | pd.Timestamp]:
    """
    Calculate PAYEMS revisions published on an event date.

    The newest reference period is an initial observation and is not
    counted as a revision.

    Older reference periods updated on the same vintage date are
    compared with their immediately preceding vintage.
    """

    payems = _prepare_payems_vintages(
        vintage_data
    )

    event_date = pd.Timestamp(
        event_date
    ).normalize()

    event_rows = payems.loc[
        payems["vintage_date"] == event_date
    ].copy()

    if event_rows.empty:
        return {
            "event_date": event_date,
            "payroll_revision_1m": 0.0,
            "payroll_revision_2m": 0.0,
            "payroll_revision_total": 0.0,
        }

    # Determine the first vintage ever observed for each reference period.
    # This is more robust than assuming only the newest reference period
    # on an event date is an initial release.
    first_vintages = (
        payems.groupby(
            "reference_date",
            as_index=False,
        )["vintage_date"]
        .min()
        .rename(
            columns={
                "vintage_date": "first_vintage_date",
            }
        )
    )

    event_rows = event_rows.merge(
        first_vintages,
        on="reference_date",
        how="left",
        validate="many_to_one",
    )

    # A value is a revision only if this event occurs after that
    # reference period's first-ever vintage.
    revision_rows = event_rows.loc[
        event_rows["vintage_date"]
        > event_rows["first_vintage_date"]
    ].copy()

    revisions: list[tuple[pd.Timestamp, float]] = []

    for row in revision_rows.itertuples():
        prior = payems.loc[
            (
                payems["reference_date"]
                == row.reference_date
            )
            & (
                payems["vintage_date"]
                < event_date
            )
        ].sort_values(
            "vintage_date"
        )

        if prior.empty:
            continue

        previous_value = float(
            prior.iloc[-1]["value_as_known"]
        )

        revision_value = (
            float(row.value_as_known)
            - previous_value
        )

        revisions.append(
            (
                row.reference_date,
                revision_value,
            )
        )

    revisions.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    revision_1m = (
        revisions[0][1]
        if len(revisions) >= 1
        else 0.0
    )

    revision_2m = (
        revisions[1][1]
        if len(revisions) >= 2
        else 0.0
    )

    return {
        "event_date": event_date,
        "payroll_revision_1m": revision_1m,
        "payroll_revision_2m": revision_2m,
        "payroll_revision_total": (
            revision_1m
            + revision_2m
        ),
    }


def build_payroll_event_features(
    vintage_data: pd.DataFrame,
    event_date: str | pd.Timestamp,
) -> dict[str, float | pd.Timestamp]:
    """
    Build PAYEMS features immediately after a release event.
    """

    event_date = pd.Timestamp(
        event_date
    ).normalize()

    history = payroll_state_as_of(
        vintage_data=vintage_data,
        as_of_date=event_date,
    )

    if history.empty:
        raise ValueError(
            "No PAYEMS history available as of "
            f"{event_date.date()}."
        )

    latest = history.iloc[-1]

    revisions = payroll_revision_features(
        vintage_data=vintage_data,
        event_date=event_date,
    )

    breadth = payroll_revision_breadth(
        vintage_data=vintage_data,
        event_date=event_date,
    )
    payroll_change = (
        float(latest["payroll_change_1m"])
        if pd.notna(
            latest["payroll_change_1m"]
        )
        else float("nan")
    )

    return {
        "event_date": event_date,
        "reference_date": latest[
            "reference_date"
        ],
        "payroll_level": float(
            latest["payroll_level"]
        ),
        "payroll_change_1m": payroll_change,
        "payroll_revision_1m": revisions[
            "payroll_revision_1m"
        ],
        "payroll_revision_2m": revisions[
            "payroll_revision_2m"
        ],
        "payroll_revision_total": revisions[
            "payroll_revision_total"
        ],
        "payroll_initial_periods": breadth[
        "payroll_initial_periods"
        ],
        "payroll_revised_periods": breadth[
        "payroll_revised_periods"
        ],
        "payroll_broad_revision_event": breadth[
        "payroll_broad_revision_event"
        ],
    }
def add_payroll_standardized_features(
    event_features: pd.DataFrame,
    min_history: int = 12,
) -> pd.DataFrame:
    """
    Add point-in-time expanding z-scores to PAYEMS event features.

    payroll_change_z:
        Standardizes payroll_change_1m using PRIOR release events only.

    payroll_revision_z:
        Standardizes payroll_revision_total using PRIOR REGULAR
        revision events only.

        Broad revision events:
        - retain their raw revision magnitude
        - retain payroll_broad_revision_event = 1
        - receive payroll_revision_z = NaN
        - are excluded from future regular-revision baselines

    The current observation never contributes to its own mean or
    standard deviation.
    """

    required = {
        "event_release_date",
        "payroll_change_1m",
        "payroll_revision_total",
        "payroll_broad_revision_event",
    }

    missing = required.difference(
        event_features.columns
    )

    if missing:
        raise ValueError(
            "PAYEMS event features missing required columns: "
            f"{sorted(missing)}"
        )

    if min_history < 2:
        raise ValueError(
            "min_history must be at least 2."
        )

    result = event_features.copy()

    result["event_release_date"] = pd.to_datetime(
        result["event_release_date"],
        errors="raise",
    ).dt.normalize()

    result = result.sort_values(
        "event_release_date"
    ).reset_index(drop=True)

    #
    # ------------------------------------------------------------
    # Payroll change z-score
    # ------------------------------------------------------------
    #
    # All releases can contribute to the payroll-change distribution.
    # The current release is shifted out so it cannot influence its
    # own standardization.
    #
    prior_payroll_changes = (
        result["payroll_change_1m"]
        .shift(1)
    )

    payroll_change_mean = (
        prior_payroll_changes
        .expanding(
            min_periods=min_history
        )
        .mean()
    )

    payroll_change_std = (
        prior_payroll_changes
        .expanding(
            min_periods=min_history
        )
        .std(ddof=1)
    )

    result["payroll_change_z"] = (
        result["payroll_change_1m"]
        - payroll_change_mean
    ) / payroll_change_std

    result.loc[
        (
            payroll_change_std.isna()
            | (payroll_change_std == 0)
        ),
        "payroll_change_z",
    ] = pd.NA

    #
    # ------------------------------------------------------------
    # Payroll revision z-score
    # ------------------------------------------------------------
    #
    # Broad benchmark revisions are retained in the dataset but
    # excluded from the distribution used to define a "normal"
    # monthly revision.
    #
    regular_revisions = (
        result["payroll_revision_total"]
        .where(
            result[
                "payroll_broad_revision_event"
            ] == 0
        )
    )

    #
    # Shift BEFORE expanding statistics:
    # only information from previous releases can be used.
    #
    prior_regular_revisions = (
        regular_revisions.shift(1)
    )

    revision_mean = (
        prior_regular_revisions
        .expanding(
            min_periods=min_history
        )
        .mean()
    )

    revision_std = (
        prior_regular_revisions
        .expanding(
            min_periods=min_history
        )
        .std(ddof=1)
    )

    result["payroll_revision_z"] = (
        result["payroll_revision_total"]
        - revision_mean
    ) / revision_std

    #
    # A benchmark revision does NOT receive a normal-revision z-score.
    #
    result.loc[
        result[
            "payroll_broad_revision_event"
        ] == 1,
        "payroll_revision_z",
    ] = pd.NA

    result.loc[
        (
            revision_std.isna()
            | (revision_std == 0)
        ),
        "payroll_revision_z",
    ] = pd.NA

    return result


def build_payroll_event_history(
    vintage_data: pd.DataFrame,
    event_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one point-in-time PAYEMS feature row per Employment
    Situation release event.

    Multiple reference periods released on the same event date
    still produce one feature row for that event.
    """

    required_event_columns = {
        "event_name",
        "reference_date",
        "release_date",
        "release_time",
        "release_timezone",
    }

    missing = required_event_columns.difference(
        event_calendar.columns
    )

    if missing:
        raise ValueError(
            "Event calendar missing required columns: "
            f"{sorted(missing)}"
        )

    events = event_calendar.copy()

    events["reference_date"] = pd.to_datetime(
        events["reference_date"],
        errors="raise",
    ).dt.normalize()

    events["release_date"] = pd.to_datetime(
        events["release_date"],
        errors="raise",
    ).dt.normalize()

    duplicate_release_dates = events.duplicated(
        subset=["release_date"],
        keep=False,
    )

    if duplicate_release_dates.any():
        raise ValueError(
            "Employment Situation calendar contains "
            "duplicate release dates."
        )

    events = events.sort_values(
        "release_date"
    ).reset_index(drop=True)

    rows = []

    for event in events.itertuples():

        features = build_payroll_event_features(
            vintage_data=vintage_data,
            event_date=event.release_date,
        )

        rows.append(
            {
                "event_name": event.event_name,
                "event_reference_date": event.reference_date,
                "event_release_date": event.release_date,
                "event_release_time": event.release_time,
                "event_release_timezone": (
                    event.release_timezone
                ),
                "reference_date": features[
                    "reference_date"
                ],
                "payroll_level": features[
                    "payroll_level"
                ],
                "payroll_change_1m": features[
                    "payroll_change_1m"
                ],
                "payroll_revision_1m": features[
                    "payroll_revision_1m"
                ],
                "payroll_revision_2m": features[
                    "payroll_revision_2m"
                ],
                "payroll_revision_total": features[
                    "payroll_revision_total"
                ],
                "payroll_initial_periods": features[
                    "payroll_initial_periods"
                ],
                "payroll_revised_periods": features[
                    "payroll_revised_periods"
                ],
                "payroll_broad_revision_event": features[
                    "payroll_broad_revision_event"
                ],
            }
        )

        history = pd.DataFrame(rows)

    history = add_payroll_standardized_features(
        event_features=history,
        min_history=12,
    )

    return history

def merge_payroll_features_to_market(
    market_data: pd.DataFrame,
    payroll_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Backward-merge PAYEMS event features onto daily market data.

    For daily TMF bars, an Employment Situation release at 08:30 ET
    is considered available to that same trading day's close.

    No market row may receive payroll information from a future
    release date.
    """

    market = market_data.copy()
    payroll = payroll_features.copy()

    if "timestamp" not in market.columns:
        raise ValueError(
            "Market data must contain a timestamp column."
        )

    required_payroll_columns = {
        "event_release_date",
        "event_release_time",
        "event_release_timezone",
        "reference_date",
        "payroll_level",
        "payroll_change_1m",
        "payroll_revision_1m",
        "payroll_revision_2m",
        "payroll_revision_total",
    }

    missing = required_payroll_columns.difference(
        payroll.columns
    )

    if missing:
        raise ValueError(
            "Payroll feature data missing required columns: "
            f"{sorted(missing)}"
        )

    market["market_date"] = pd.to_datetime(
        market["timestamp"],
        utc=True,
        errors="raise",
    ).dt.tz_localize(None).dt.normalize()

    payroll["event_release_date"] = pd.to_datetime(
        payroll["event_release_date"],
        errors="raise",
    ).dt.normalize()

    payroll["reference_date"] = pd.to_datetime(
        payroll["reference_date"],
        errors="raise",
    ).dt.normalize()

    if payroll["event_release_date"].duplicated().any():
        raise ValueError(
            "Payroll feature data contains duplicate "
            "event release dates."
        )

    market = market.sort_values(
        "market_date"
    ).reset_index(drop=True)

    payroll = payroll.sort_values(
        "event_release_date"
    ).reset_index(drop=True)

    result = pd.merge_asof(
        market,
        payroll,
        left_on="market_date",
        right_on="event_release_date",
        direction="backward",
        allow_exact_matches=True,
    )

    #
    # Point-in-time invariant:
    # payroll information must never come from a future release.
    #
    invalid = (
        result["event_release_date"].notna()
        & (
            result["event_release_date"]
            > result["market_date"]
        )
    )

    if invalid.any():
        raise ValueError(
            "Payroll merge introduced future information."
        )

    result["days_since_payroll_release"] = (
        result["market_date"]
        - result["event_release_date"]
    ).dt.days

    result["payroll_release_day"] = (
        result["market_date"]
        == result["event_release_date"]
    ).astype("int8")

    return result
def payroll_revision_breadth(
    vintage_data: pd.DataFrame,
    event_date: str | pd.Timestamp,
) -> dict[str, int]:
    """
    Describe how many PAYEMS reference periods changed on an event date.

    Broad revision events are defined structurally as events that revise
    more than two previously published reference periods.
    """

    payems = _prepare_payems_vintages(
        vintage_data
    )

    event_date = pd.Timestamp(
        event_date
    ).normalize()

    first_vintages = (
        payems
        .groupby(
            "reference_date",
            as_index=False,
        )["vintage_date"]
        .min()
        .rename(
            columns={
                "vintage_date": "first_vintage_date",
            }
        )
    )

    event_rows = payems.loc[
        payems["vintage_date"] == event_date
    ].copy()

    if event_rows.empty:
        return {
            "payroll_initial_periods": 0,
            "payroll_revised_periods": 0,
            "payroll_broad_revision_event": 0,
        }

    event_rows = event_rows.merge(
        first_vintages,
        on="reference_date",
        how="left",
        validate="many_to_one",
    )

    initial_periods = int(
        (
            event_rows["vintage_date"]
            == event_rows["first_vintage_date"]
        ).sum()
    )

    revised_periods = int(
        (
            event_rows["vintage_date"]
            > event_rows["first_vintage_date"]
        ).sum()
    )

    return {
        "payroll_initial_periods": initial_periods,
        "payroll_revised_periods": revised_periods,
        "payroll_broad_revision_event": int(
            revised_periods > 2
        ),
    }