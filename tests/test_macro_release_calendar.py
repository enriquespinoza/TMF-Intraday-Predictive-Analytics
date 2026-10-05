"""Tests for point-in-time macro release alignment."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.macro_release_calendar import (
    attach_release_dates,
    prepare_market_observed_series,
    prepare_release_series,
    requires_release_alignment,
    validate_release_calendar,
)


def make_calendar() -> pd.DataFrame:
    """Create a small synthetic CPI release calendar."""

    return pd.DataFrame(
        {
            "series_name": [
                "cpi",
                "cpi",
            ],
            "reference_date": [
                "2025-01-01",
                "2025-02-01",
            ],
            "release_date": [
                "2025-02-12",
                "2025-03-12",
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


def make_observations() -> pd.DataFrame:
    """Create synthetic CPI observations."""

    return pd.DataFrame(
        {
            "reference_date": [
                "2025-01-01",
                "2025-02-01",
            ],
            "value": [
                319.1,
                319.8,
            ],
        }
    )


def test_release_alignment_requirement():
    """Monthly releases and daily market series should differ."""

    assert (
        requires_release_alignment(
            "cpi"
        )
        is True
    )

    assert (
        requires_release_alignment(
            "breakeven_10y"
        )
        is False
    )


def test_unknown_series_rejected():
    """Unknown series should fail explicitly."""

    with pytest.raises(KeyError):
        requires_release_alignment(
            "not_a_real_series"
        )


def test_release_calendar_validation():
    """Valid release calendar should parse dates."""

    result = validate_release_calendar(
        calendar=make_calendar(),
        series_name="cpi",
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result["reference_date"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result["release_date"]
    )


def test_release_before_reference_rejected():
    """Release dates cannot precede their reference periods."""

    calendar = make_calendar()

    calendar.loc[
        0,
        "release_date",
    ] = "2024-12-31"

    with pytest.raises(ValueError):
        validate_release_calendar(
            calendar=calendar,
            series_name="cpi",
        )


def test_duplicate_reference_period_rejected():
    """One series cannot have duplicate release records per period."""

    calendar = make_calendar()

    duplicate = calendar.iloc[
        [0]
    ].copy()

    calendar = pd.concat(
        [
            calendar,
            duplicate,
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        validate_release_calendar(
            calendar=calendar,
            series_name="cpi",
        )


def test_release_dates_attached():
    """Known release dates should attach to observations."""

    result = attach_release_dates(
        observations=make_observations(),
        calendar=make_calendar(),
        series_name="cpi",
    )

    assert result.loc[
        0,
        "release_date",
    ] == pd.Timestamp(
        "2025-02-12"
    )

    assert result.loc[
        1,
        "release_date",
    ] == pd.Timestamp(
        "2025-03-12"
    )


def test_missing_release_metadata_fails_closed():
    """Observed values without known release dates must be rejected."""

    observations = make_observations()

    calendar = make_calendar().iloc[
        [0]
    ]

    with pytest.raises(
        ValueError,
        match="without known release metadata",
    ):
        attach_release_dates(
            observations=observations,
            calendar=calendar,
            series_name="cpi",
        )


def test_release_series_uses_release_date():
    """Economic releases should become available on release date."""

    result = prepare_release_series(
        observations=make_observations(),
        calendar=make_calendar(),
        series_name="cpi",
    )

    assert result.loc[
        0,
        "market_date_available",
    ] == pd.Timestamp(
        "2025-02-12"
    )

    assert (
        result.loc[
            0,
            "market_date_available",
        ]
        != result.loc[
            0,
            "reference_date",
        ]
    )


def test_market_series_uses_observation_date():
    """Daily market data should use its observation date."""

    observations = pd.DataFrame(
        {
            "reference_date": [
                "2025-01-02",
                "2025-01-03",
            ],
            "value": [
                2.31,
                2.34,
            ],
        }
    )

    result = prepare_market_observed_series(
        observations=observations,
        series_name="breakeven_10y",
    )

    assert (
        result["market_date_available"]
        == result["reference_date"]
    ).all()


def test_release_series_cannot_use_market_path():
    """Release-sensitive series must not bypass release alignment."""

    with pytest.raises(ValueError):
        prepare_market_observed_series(
            observations=make_observations(),
            series_name="cpi",
        )


def test_market_series_cannot_use_release_path():
    """Market-observed series should not require a release calendar."""

    with pytest.raises(ValueError):
        prepare_release_series(
            observations=pd.DataFrame(
                {
                    "reference_date": [
                        "2025-01-02",
                    ],
                    "value": [
                        2.31,
                    ],
                }
            ),
            calendar=make_calendar(),
            series_name="breakeven_10y",
        )

def test_release_metadata_preserved():
    """Release time and timezone should survive alignment."""

    result = prepare_release_series(
        observations=make_observations(),
        calendar=make_calendar(),
        series_name="cpi",
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


def test_missing_release_time_rejected():
    """Release-sensitive data must include release time."""

    calendar = (
        make_calendar()
        .drop(columns=["release_time"])
    )

    with pytest.raises(
        ValueError,
        match="missing required",
    ):
        validate_release_calendar(
            calendar=calendar,
            series_name="cpi",
        )


def test_invalid_release_time_rejected():
    """Malformed release times should fail validation."""

    calendar = make_calendar()

    calendar.loc[
        0,
        "release_time",
    ] = "25:90"

    with pytest.raises(
        ValueError,
        match="Invalid release_time",
    ):
        validate_release_calendar(
            calendar=calendar,
            series_name="cpi",
        )


def test_invalid_release_timezone_rejected():
    """Invalid timezone names should fail validation."""

    calendar = make_calendar()

    calendar.loc[
        0,
        "release_timezone",
    ] = "Eastern"

    with pytest.raises(
        ValueError,
        match="Invalid release_timezone",
    ):
        validate_release_calendar(
            calendar=calendar,
            series_name="cpi",
        )