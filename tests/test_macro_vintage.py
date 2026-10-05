"""Tests for point-in-time macro vintage reconstruction."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.macro_vintage import (
    attach_event_release_timing,
    attach_release_calendar,
    latest_reference_period_as_of,
    prepare_vintage_events,
    reconstruct_as_of,
    validate_vintage_events,
)


def make_payems_vintages() -> pd.DataFrame:
    """
    Create a small PAYEMS revision history modeled on the real
    output_type=3 structure returned by FRED.

    September 2021:
        2021-10-08 -> 147553
        2021-11-05 -> 147788
        2021-12-03 -> 147855

    October 2021:
        2021-11-05 -> 148319
        2021-12-03 -> 148401

    November 2021:
        2021-12-03 -> 148611
    """

    return pd.DataFrame(
        {
            "series_name": [
                "nonfarm_payrolls",
                "nonfarm_payrolls",
                "nonfarm_payrolls",
                "nonfarm_payrolls",
                "nonfarm_payrolls",
                "nonfarm_payrolls",
            ],
            "series_id": [
                "PAYEMS",
                "PAYEMS",
                "PAYEMS",
                "PAYEMS",
                "PAYEMS",
                "PAYEMS",
            ],
            "reference_date": [
                "2021-09-01",
                "2021-09-01",
                "2021-09-01",
                "2021-10-01",
                "2021-10-01",
                "2021-11-01",
            ],
            "vintage_date": [
                "2021-10-08",
                "2021-11-05",
                "2021-12-03",
                "2021-11-05",
                "2021-12-03",
                "2021-12-03",
            ],
            "value_as_known": [
                147553,
                147788,
                147855,
                148319,
                148401,
                148611,
            ],
        }
    )


def test_prepare_vintage_events_normalizes_dates():
    data = make_payems_vintages()

    result = prepare_vintage_events(data)

    assert pd.api.types.is_datetime64_any_dtype(
        result["reference_date"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result["vintage_date"]
    )


def test_reconstruct_as_of_first_release():
    """
    On 2021-10-08, only the first September payroll value
    should be known.
    """

    data = make_payems_vintages()

    result = reconstruct_as_of(
        data=data,
        as_of_date="2021-10-08",
    )

    assert len(result) == 1

    row = result.iloc[0]

    assert (
        row["reference_date"]
        == pd.Timestamp("2021-09-01")
    )

    assert (
        row["vintage_date"]
        == pd.Timestamp("2021-10-08")
    )

    assert row["value_as_known"] == 147553


def test_future_revision_cannot_leak_backward():
    """
    Critical anti-lookahead test.

    The September payroll value was revised to 147788 on
    2021-11-05 and 147855 on 2021-12-03.

    When reconstructing the information set as of 2021-10-08,
    neither future revision may appear.
    """

    data = make_payems_vintages()

    result = reconstruct_as_of(
        data=data,
        as_of_date="2021-10-08",
    )

    september = result.loc[
        result["reference_date"]
        == pd.Timestamp("2021-09-01")
    ]

    assert len(september) == 1

    row = september.iloc[0]

    assert row["value_as_known"] == 147553

    assert (
        row["vintage_date"]
        == pd.Timestamp("2021-10-08")
    )

    assert 147788 not in result["value_as_known"].values
    assert 147855 not in result["value_as_known"].values

    assert (
        result["vintage_date"]
        <= pd.Timestamp("2021-10-08")
    ).all()


def test_reconstruct_uses_latest_revision_available():
    """
    By 2021-11-05, the November revision of September payroll
    should replace the original October release.
    """

    data = make_payems_vintages()

    result = reconstruct_as_of(
        data=data,
        as_of_date="2021-11-05",
    )

    september = result.loc[
        result["reference_date"]
        == pd.Timestamp("2021-09-01")
    ].iloc[0]

    october = result.loc[
        result["reference_date"]
        == pd.Timestamp("2021-10-01")
    ].iloc[0]

    assert september["value_as_known"] == 147788
    assert september["vintage_date"] == pd.Timestamp(
        "2021-11-05"
    )

    assert october["value_as_known"] == 148319
    assert october["vintage_date"] == pd.Timestamp(
        "2021-11-05"
    )


def test_reconstruct_updates_multiple_periods():
    """
    A single vintage date can update several reference periods.

    On 2021-12-03:
        September -> 147855
        October   -> 148401
        November  -> 148611
    """

    data = make_payems_vintages()

    result = reconstruct_as_of(
        data=data,
        as_of_date="2021-12-03",
    )

    expected = {
        pd.Timestamp("2021-09-01"): 147855,
        pd.Timestamp("2021-10-01"): 148401,
        pd.Timestamp("2021-11-01"): 148611,
    }

    actual = dict(
        zip(
            result["reference_date"],
            result["value_as_known"],
        )
    )

    assert actual == expected


def test_as_of_before_first_vintage_returns_empty():
    """
    No PAYEMS observation should be available before the
    first vintage in the test dataset.
    """

    data = make_payems_vintages()

    result = reconstruct_as_of(
        data=data,
        as_of_date="2021-10-07",
    )

    assert result.empty


def test_latest_reference_period_as_of():
    """
    On 2021-12-03, November 2021 is the latest reference
    period available.
    """

    data = make_payems_vintages()

    result = latest_reference_period_as_of(
        data=data,
        as_of_date="2021-12-03",
    )

    assert len(result) == 1

    row = result.iloc[0]

    assert (
        row["reference_date"]
        == pd.Timestamp("2021-11-01")
    )

    assert (
        row["vintage_date"]
        == pd.Timestamp("2021-12-03")
    )

    assert row["value_as_known"] == 148611


def test_duplicate_vintage_event_rejected():
    data = make_payems_vintages()

    duplicate = data.iloc[[0]].copy()

    data = pd.concat(
        [data, duplicate],
        ignore_index=True,
    )

    prepared = data.copy()

    prepared["reference_date"] = pd.to_datetime(
        prepared["reference_date"]
    )

    prepared["vintage_date"] = pd.to_datetime(
        prepared["vintage_date"]
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_vintage_events(prepared)


def test_vintage_before_reference_rejected():
    data = pd.DataFrame(
        {
            "series_name": [
                "nonfarm_payrolls"
            ],
            "series_id": [
                "PAYEMS"
            ],
            "reference_date": [
                pd.Timestamp("2021-09-01")
            ],
            "vintage_date": [
                pd.Timestamp("2021-08-31")
            ],
            "value_as_known": [
                147553
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="before reference dates",
    ):
        validate_vintage_events(data)

def make_payems_release_calendar() -> pd.DataFrame:
    """Create a small Employment Situation release calendar."""

    return pd.DataFrame(
        {
            "series_name": [
                "nonfarm_payrolls",
                "nonfarm_payrolls",
                "nonfarm_payrolls",
            ],
            "reference_date": [
                "2021-09-01",
                "2021-10-01",
                "2021-11-01",
            ],
            "release_date": [
                "2021-10-08",
                "2021-11-05",
                "2021-12-03",
            ],
            "release_time": [
                "08:30",
                "08:30",
                "08:30",
            ],
            "release_timezone": [
                "America/New_York",
                "America/New_York",
                "America/New_York",
            ],
        }
    )


def test_attach_release_calendar_uses_original_release_date():
    vintages = make_payems_vintages()
    calendar = make_payems_release_calendar()

    result = attach_release_calendar(
        vintage_data=vintages,
        release_calendar=calendar,
    )

    assert "original_release_date" in result.columns
    assert "release_date" not in result.columns


def test_initial_release_has_correct_original_release_date():
    vintages = make_payems_vintages()
    calendar = make_payems_release_calendar()

    result = attach_release_calendar(
        vintage_data=vintages,
        release_calendar=calendar,
    )

    october = result.loc[
        (
            result["reference_date"]
            == pd.Timestamp("2021-10-01")
        )
        & result["is_initial_release"]
    ].iloc[0]

    assert (
        october["original_release_date"]
        == pd.Timestamp("2021-11-05")
    )

    assert (
        october["vintage_date"]
        == pd.Timestamp("2021-11-05")
    )

    assert october["release_time"] == "08:30"

    assert (
        october["release_timezone"]
        == "America/New_York"
    )


def test_revision_preserves_original_release_date():
    """
    September's November revision must retain September's
    original release date of October 8.
    """

    vintages = make_payems_vintages()
    calendar = make_payems_release_calendar()

    result = attach_release_calendar(
        vintage_data=vintages,
        release_calendar=calendar,
    )

    revision = result.loc[
        (
            result["reference_date"]
            == pd.Timestamp("2021-09-01")
        )
        & (
            result["vintage_date"]
            == pd.Timestamp("2021-11-05")
        )
    ].iloc[0]

    assert (
        revision["original_release_date"]
        == pd.Timestamp("2021-10-08")
    )

    assert revision["value_as_known"] == 147788
    assert not revision["is_initial_release"]


def test_revision_does_not_inherit_original_release_clock():
    """
    A later revision should not incorrectly inherit the clock
    time of the reference period's original release.
    """

    vintages = make_payems_vintages()
    calendar = make_payems_release_calendar()

    result = attach_release_calendar(
        vintage_data=vintages,
        release_calendar=calendar,
    )

    revision = result.loc[
        (
            result["reference_date"]
            == pd.Timestamp("2021-09-01")
        )
        & (
            result["vintage_date"]
            == pd.Timestamp("2021-11-05")
        )
    ].iloc[0]

    assert pd.isna(
        revision["release_time"]
    )

    assert pd.isna(
        revision["release_timezone"]
    )


def test_market_date_available_still_uses_vintage_date():
    vintages = make_payems_vintages()
    calendar = make_payems_release_calendar()

    result = attach_release_calendar(
        vintage_data=vintages,
        release_calendar=calendar,
    )

    assert (
        result["market_date_available"]
        == result["vintage_date"]
    ).all()

def make_employment_situation_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_name": [
                "employment_situation",
                "employment_situation",
                "employment_situation",
            ],
            "reference_date": [
                "2021-09-01",
                "2021-10-01",
                "2021-11-01",
            ],
            "release_date": [
                "2021-10-08",
                "2021-11-05",
                "2021-12-03",
            ],
            "release_time": [
                "08:30",
                "08:30",
                "08:30",
            ],
            "release_timezone": [
                "America/New_York",
                "America/New_York",
                "America/New_York",
            ],
        }
    )


def test_revision_inherits_event_release_timing():
    vintages = make_payems_vintages()
    calendar = make_payems_release_calendar()
    events = make_employment_situation_events()

    canonical = attach_release_calendar(
        vintage_data=vintages,
        release_calendar=calendar,
    )

    result = attach_event_release_timing(
        canonical_data=canonical,
        event_calendar=events,
    )

    revision = result.loc[
        (
            result["reference_date"]
            == pd.Timestamp("2021-09-01")
        )
        & (
            result["vintage_date"]
            == pd.Timestamp("2021-11-05")
        )
    ].iloc[0]

    assert (
        revision["original_release_date"]
        == pd.Timestamp("2021-10-08")
    )

    assert (
        revision["event_release_date"]
        == pd.Timestamp("2021-11-05")
    )

    assert revision["event_release_time"] == "08:30"

    assert (
        revision["event_release_timezone"]
        == "America/New_York"
    )


def test_revision_event_reference_period_is_later_report():
    vintages = make_payems_vintages()
    calendar = make_payems_release_calendar()
    events = make_employment_situation_events()

    canonical = attach_release_calendar(
        vintage_data=vintages,
        release_calendar=calendar,
    )

    result = attach_event_release_timing(
        canonical_data=canonical,
        event_calendar=events,
    )

    revision = result.loc[
        (
            result["reference_date"]
            == pd.Timestamp("2021-09-01")
        )
        & (
            result["vintage_date"]
            == pd.Timestamp("2021-11-05")
        )
    ].iloc[0]

    assert (
        revision["event_reference_date"]
        == pd.Timestamp("2021-10-01")
    )


def test_initial_and_revision_share_same_release_event():
    vintages = make_payems_vintages()
    calendar = make_payems_release_calendar()
    events = make_employment_situation_events()

    canonical = attach_release_calendar(
        vintage_data=vintages,
        release_calendar=calendar,
    )

    result = attach_event_release_timing(
        canonical_data=canonical,
        event_calendar=events,
    )

    nov_5 = result.loc[
        result["vintage_date"]
        == pd.Timestamp("2021-11-05")
    ]

    assert len(nov_5) == 2

    assert (
        nov_5["event_release_date"]
        == pd.Timestamp("2021-11-05")
    ).all()

    assert (
        nov_5["event_release_time"]
        == "08:30"
    ).all()

    assert (
        nov_5["event_reference_date"]
        == pd.Timestamp("2021-10-01")
    ).all()


def test_unmatched_vintage_does_not_invent_event_timing():
    vintages = make_payems_vintages()
    calendar = make_payems_release_calendar()
    events = make_employment_situation_events()

    canonical = attach_release_calendar(
        vintage_data=vintages,
        release_calendar=calendar,
    )

    result = attach_event_release_timing(
        canonical_data=canonical,
        event_calendar=events,
    )

    unmatched = result.loc[
        result["vintage_date"]
        == pd.Timestamp("2022-02-04")
    ]

    if not unmatched.empty:
        assert unmatched["event_release_date"].isna().all()
        assert unmatched["event_release_time"].isna().all()
        assert unmatched["event_release_timezone"].isna().all()