"""Tests for point-in-time macro availability validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.validate_macro_point_in_time import (
    validate_point_in_time_series,
)


def test_valid_release_availability():
    """A release after its reference period is valid."""

    prepared = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                ["2025-09-01"]
            ),
            "value": [159000.0],
            "market_date_available": pd.to_datetime(
                ["2025-11-20"]
            ),
        }
    )

    result = validate_point_in_time_series(
        series_name="nonfarm_payrolls",
        prepared=prepared,
    )

    assert result["observations"] == 1
    assert result["valid_availability"] == 1
    assert result["invalid_availability"] == 0


def test_missing_availability_rejected():
    """A known observation must have an availability date."""

    prepared = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                ["2025-09-01"]
            ),
            "value": [159000.0],
            "market_date_available": [
                pd.NaT
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="without market availability",
    ):
        validate_point_in_time_series(
            series_name="nonfarm_payrolls",
            prepared=prepared,
        )


def test_lookahead_availability_rejected():
    """Availability cannot precede the reference period."""

    prepared = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                ["2025-09-01"]
            ),
            "value": [159000.0],
            "market_date_available": pd.to_datetime(
                ["2025-08-29"]
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="lookahead",
    ):
        validate_point_in_time_series(
            series_name="nonfarm_payrolls",
            prepared=prepared,
        )


def test_null_observation_does_not_require_availability():
    """A missing economic observation does not create information."""

    prepared = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                ["2025-10-01"]
            ),
            "value": [None],
            "market_date_available": [
                pd.NaT
            ],
        }
    )

    result = validate_point_in_time_series(
        series_name="unemployment_rate",
        prepared=prepared,
    )

    assert result["observations"] == 0
    assert result["invalid_availability"] == 0