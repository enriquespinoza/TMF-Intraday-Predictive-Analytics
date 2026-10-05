"""Tests for macro release-calendar construction."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.build_release_calendars import (
    EMPLOYMENT_SITUATION_CES_SERIES,
    EMPLOYMENT_SITUATION_SERIES,
    add_employment_situation_exceptions,
    build_employment_situation_calendars,
    expand_event_calendar,
    save_series_calendars,
    validate_event_calendar,
)


def make_employment_events() -> pd.DataFrame:
    """Create synthetic Employment Situation events."""

    return pd.DataFrame(
        {
            "event_name": [
                "employment_situation",
                "employment_situation",
            ],
            "reference_date": [
                "2025-01-01",
                "2025-02-01",
            ],
            "release_date": [
                "2025-02-07",
                "2025-03-07",
            ],
            "release_time": [
                "08:30",
                "08:30",
            ],
            "release_timezone": [
                "America/New_York",
                "America/New_York",
            ],
        }
    )


def test_event_calendar_validates():
    """Valid event calendars should normalize dates."""

    result = validate_event_calendar(
        events=make_employment_events(),
        expected_event_name=(
            "employment_situation"
        ),
    )

    assert len(result) == 2

    assert pd.api.types.is_datetime64_any_dtype(
        result["reference_date"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result["release_date"]
    )


def test_missing_event_column_rejected():
    """Incomplete event calendars should fail."""

    events = (
        make_employment_events()
        .drop(columns=["release_time"])
    )

    with pytest.raises(
        ValueError,
        match="missing required",
    ):
        validate_event_calendar(
            events=events,
            expected_event_name=(
                "employment_situation"
            ),
        )


def test_wrong_event_name_rejected():
    """Unexpected event types should fail."""

    events = make_employment_events()

    events.loc[
        0,
        "event_name",
    ] = "cpi"

    with pytest.raises(
        ValueError,
        match="Expected event_name",
    ):
        validate_event_calendar(
            events=events,
            expected_event_name=(
                "employment_situation"
            ),
        )


def test_duplicate_event_period_rejected():
    """A release event cannot duplicate a reference period."""

    events = make_employment_events()

    duplicate = events.iloc[
        [0]
    ].copy()

    events = pd.concat(
        [
            events,
            duplicate,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_event_calendar(
            events=events,
            expected_event_name=(
                "employment_situation"
            ),
        )


def test_event_expands_to_all_employment_series():
    """One event calendar should support all Employment Situation series."""

    calendars = expand_event_calendar(
        events=make_employment_events(),
        event_name="employment_situation",
        series_names=(
            EMPLOYMENT_SITUATION_SERIES
        ),
    )

    assert set(calendars) == set(
        EMPLOYMENT_SITUATION_SERIES
    )

    assert len(calendars) == 7


def test_expanded_calendar_has_correct_series_name():
    """Expanded calendars should identify their target series."""

    calendars = expand_event_calendar(
        events=make_employment_events(),
        event_name="employment_situation",
        series_names=[
            "nonfarm_payrolls",
        ],
    )

    result = calendars[
        "nonfarm_payrolls"
    ]

    assert (
        result["series_name"]
        == "nonfarm_payrolls"
    ).all()


def test_expansion_preserves_release_metadata():
    """Expansion should preserve exact event timing."""

    calendars = expand_event_calendar(
        events=make_employment_events(),
        event_name="employment_situation",
        series_names=[
            "unemployment_rate",
        ],
    )

    result = calendars[
        "unemployment_rate"
    ]

    assert (
        result.loc[
            0,
            "release_date",
        ]
        == pd.Timestamp(
            "2025-02-07"
        )
    )

    assert (
        result.loc[
            0,
            "release_time",
        ]
        == "08:30"
    )

    assert (
        result.loc[
            0,
            "release_timezone",
        ]
        == "America/New_York"
    )


def test_save_series_calendars(
    tmp_path,
):
    """Expanded calendars should save independently."""

    calendars = expand_event_calendar(
        events=make_employment_events(),
        event_name="employment_situation",
        series_names=[
            "nonfarm_payrolls",
            "unemployment_rate",
        ],
    )

    paths = save_series_calendars(
        calendars=calendars,
        output_dir=tmp_path,
    )

    assert len(paths) == 2

    assert (
        tmp_path
        / "nonfarm_payrolls.csv"
    ).exists()

    assert (
        tmp_path
        / "unemployment_rate.csv"
    ).exists()


def test_build_employment_calendars_from_file(
    tmp_path,
):
    """Builder should load one event file and produce seven calendars."""

    event_path = (
        tmp_path
        / "employment_situation.csv"
    )

    make_employment_events().to_csv(
        event_path,
        index=False,
    )

    output_dir = (
        tmp_path
        / "output"
    )

    paths = (
        build_employment_situation_calendars(
            event_path=event_path,
            output_dir=output_dir,
        )
    )

    assert len(paths) == 7

    for series_name in (
        EMPLOYMENT_SITUATION_SERIES
    ):
        assert (
            output_dir
            / f"{series_name}.csv"
        ).exists()


def test_missing_event_file_rejected(
    tmp_path,
):
    """Builder should fail closed when authoritative events are unavailable."""

    missing = (
        tmp_path
        / "missing.csv"
    )

    with pytest.raises(
        FileNotFoundError,
        match="not found",
    ):
        build_employment_situation_calendars(
            event_path=missing,
            output_dir=tmp_path,
        )

def test_october_2025_ces_exception_added():
    """October 2025 CES data should receive the delayed release."""

    calendars = expand_event_calendar(
        events=make_employment_events(),
        event_name="employment_situation",
        series_names=EMPLOYMENT_SITUATION_SERIES,
    )

    calendars = add_employment_situation_exceptions(
        calendars
    )

    for series_name in EMPLOYMENT_SITUATION_CES_SERIES:
        calendar = calendars[series_name]

        row = calendar.loc[
            calendar["reference_date"]
            == pd.Timestamp("2025-10-01")
        ]

        assert len(row) == 1

        assert (
            row.iloc[0]["release_date"]
            == pd.Timestamp("2025-12-16")
        )

        assert (
            row.iloc[0]["release_time"]
            == "08:30"
        )

        assert (
            row.iloc[0]["release_timezone"]
            == "America/New_York"
        )


def test_october_2025_exception_not_added_to_cps():
    """October 2025 CPS household data should not get a release."""

    calendars = expand_event_calendar(
        events=make_employment_events(),
        event_name="employment_situation",
        series_names=EMPLOYMENT_SITUATION_SERIES,
    )

    calendars = add_employment_situation_exceptions(
        calendars
    )

    cps_series = [
        "unemployment_rate",
        "u6_unemployment_rate",
        "labor_force_participation",
        "part_time_economic_reasons",
    ]

    for series_name in cps_series:
        calendar = calendars[series_name]

        assert not (
            calendar["reference_date"]
            == pd.Timestamp("2025-10-01")
        ).any()


def test_october_2025_ces_exception_duplicate_rejected():
    """The October 2025 CES exception cannot be inserted twice."""

    calendars = expand_event_calendar(
        events=make_employment_events(),
        event_name="employment_situation",
        series_names=EMPLOYMENT_SITUATION_SERIES,
    )

    calendars = add_employment_situation_exceptions(
        calendars
    )

    with pytest.raises(
        ValueError,
        match="already exists",
    ):
        add_employment_situation_exceptions(
            calendars
        )