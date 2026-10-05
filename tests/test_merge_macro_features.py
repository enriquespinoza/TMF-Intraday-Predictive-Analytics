"""Tests for point-in-time macro-to-market alignment."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.merge_macro_features import (
    merge_macro_family,
    merge_macro_series,
    validate_macro_data,
)


def make_market() -> pd.DataFrame:
    """Create a small market-date fixture."""

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-11-19",
                    "2025-11-20",
                    "2025-11-21",
                    "2025-12-15",
                    "2025-12-16",
                    "2025-12-17",
                ]
            ),
            "close": [
                30.0,
                30.1,
                30.2,
                31.0,
                31.1,
                31.2,
            ],
        }
    )


def make_payrolls() -> pd.DataFrame:
    """Create shutdown-era payroll observations."""

    return pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                [
                    "2025-09-01",
                    "2025-10-01",
                    "2025-11-01",
                ]
            ),
            "value": [
                158548.0,
                158408.0,
                158449.0,
            ],
            "release_date": pd.to_datetime(
                [
                    "2025-11-20",
                    "2025-12-16",
                    "2025-12-16",
                ]
            ),
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
            "market_date_available": pd.to_datetime(
                [
                    "2025-11-20",
                    "2025-12-16",
                    "2025-12-16",
                ]
            ),
        }
    )


def test_future_release_not_visible():
    """A market row cannot see a future release."""

    result = merge_macro_series(
        market=make_market(),
        macro=make_payrolls(),
        series_name="nonfarm_payrolls",
    )

    row = result.loc[
        result["timestamp"]
        == pd.Timestamp("2025-11-19")
    ].iloc[0]

    assert pd.isna(
        row["nonfarm_payrolls"]
    )


def test_release_visible_same_day():
    """An 08:30 release is available to the daily close."""

    result = merge_macro_series(
        market=make_market(),
        macro=make_payrolls(),
        series_name="nonfarm_payrolls",
    )

    row = result.loc[
        result["timestamp"]
        == pd.Timestamp("2025-11-20")
    ].iloc[0]

    assert (
        row["nonfarm_payrolls"]
        == 158548.0
    )

    assert (
        row[
            "nonfarm_payrolls_reference_date"
        ]
        == pd.Timestamp("2025-09-01")
    )


def test_previous_release_carried_forward():
    """Latest known release should persist until replaced."""

    result = merge_macro_series(
        market=make_market(),
        macro=make_payrolls(),
        series_name="nonfarm_payrolls",
    )

    row = result.loc[
        result["timestamp"]
        == pd.Timestamp("2025-12-15")
    ].iloc[0]

    assert (
        row["nonfarm_payrolls"]
        == 158548.0
    )

    assert (
        row[
            "nonfarm_payrolls_reference_date"
        ]
        == pd.Timestamp("2025-09-01")
    )


def test_same_day_multiple_releases_use_latest_reference_period():
    """Same-day releases should expose the latest reference period."""

    result = merge_macro_series(
        market=make_market(),
        macro=make_payrolls(),
        series_name="nonfarm_payrolls",
    )

    row = result.loc[
        result["timestamp"]
        == pd.Timestamp("2025-12-16")
    ].iloc[0]

    assert (
        row["nonfarm_payrolls"]
        == 158449.0
    )

    assert (
        row[
            "nonfarm_payrolls_reference_date"
        ]
        == pd.Timestamp("2025-11-01")
    )

    assert (
        row[
            "nonfarm_payrolls_market_date_available"
        ]
        == pd.Timestamp("2025-12-16")
    )


def test_market_date_never_precedes_availability():
    """Every attached value must already be public."""

    result = merge_macro_series(
        market=make_market(),
        macro=make_payrolls(),
        series_name="nonfarm_payrolls",
    )

    known = result.loc[
        result[
            "nonfarm_payrolls_market_date_available"
        ].notna()
    ]

    assert (
        known[
            "nonfarm_payrolls_market_date_available"
        ]
        <= known["timestamp"]
    ).all()


def test_invalid_macro_timing_rejected():
    """Macro data cannot be available before its reference period."""

    macro = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                ["2025-09-01"]
            ),
            "value": [100.0],
            "market_date_available": pd.to_datetime(
                ["2025-08-31"]
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="available before",
    ):
        validate_macro_data(
            macro=macro,
            series_name="test_series",
        )


def test_null_macro_observation_not_merged():
    """A missing economic observation must not become information."""

    macro = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                [
                    "2025-09-01",
                    "2025-10-01",
                ]
            ),
            "value": [
                4.4,
                None,
            ],
            "market_date_available": pd.to_datetime(
                [
                    "2025-11-20",
                    None,
                ]
            ),
        }
    )

    result = merge_macro_series(
        market=make_market(),
        macro=macro,
        series_name="unemployment_rate",
    )

    row = result.loc[
        result["timestamp"]
        == pd.Timestamp("2025-12-17")
    ].iloc[0]

    assert (
        row["unemployment_rate"]
        == 4.4
    )

    assert (
        row[
            "unemployment_rate_reference_date"
        ]
        == pd.Timestamp("2025-09-01")
    )


def test_macro_family_merges_multiple_series():
    """Multiple macro series should share the same market timeline."""

    payrolls = make_payrolls()

    wages = make_payrolls().copy()

    wages["value"] = [
        36.70,
        36.85,
        37.00,
    ]

    result = merge_macro_family(
        market=make_market(),
        macro_series={
            "nonfarm_payrolls": payrolls,
            "average_hourly_earnings": wages,
        },
    )

    assert len(result) == len(
        make_market()
    )

    assert "nonfarm_payrolls" in result.columns

    assert (
        "average_hourly_earnings"
        in result.columns
    )


def test_data_age_calculated_from_release_date():
    """Data age should measure days since market availability."""

    result = merge_macro_series(
        market=make_market(),
        macro=make_payrolls(),
        series_name="nonfarm_payrolls",
    )

    row = result.loc[
        result["timestamp"]
        == pd.Timestamp("2025-12-15")
    ].iloc[0]

    assert (
        row[
            "nonfarm_payrolls_data_age_days"
        ]
        == 25
    )