import numpy as np
import pandas as pd

from src.features.treasury_features import (
    build_treasury_features,
)


def create_treasury_sample(
    rows: int = 30,
) -> pd.DataFrame:
    """Create deterministic Treasury data."""

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="D",
            ),
            "yield_2y": np.linspace(
                4.00,
                4.29,
                rows,
            ),
            "yield_5y": np.linspace(
                4.20,
                4.49,
                rows,
            ),
            "yield_10y": np.linspace(
                4.40,
                4.69,
                rows,
            ),
            "yield_30y": np.linspace(
                4.60,
                4.89,
                rows,
            ),
        }
    )


def test_treasury_feature_columns_created():

    df = create_treasury_sample()

    result = build_treasury_features(df)

    expected = {
        "change_2y_1d_bp",
        "change_10y_5d_bp",
        "change_30y_20d_bp",
        "curve_10y_2y_bp",
        "curve_30y_2y_bp",
        "curve_30y_10y_bp",
        "curve_30y_10y_bp_change_5d",
        "yield_30y_volatility_20d",
    }

    assert expected.issubset(
        result.columns
    )


def test_curve_calculation():

    df = create_treasury_sample()

    result = build_treasury_features(df)

    expected = (
        result["yield_30y"].iloc[-1]
        - result["yield_10y"].iloc[-1]
    ) * 100

    actual = (
        result["curve_30y_10y_bp"].iloc[-1]
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_yield_change_in_basis_points():

    df = create_treasury_sample()

    result = build_treasury_features(df)

    # Sample increases approximately 1 bp per day.
    actual = result[
        "change_30y_1d_bp"
    ].iloc[-1]

    assert np.isclose(
        actual,
        1.0,
    )


def test_row_count_preserved():

    df = create_treasury_sample()

    result = build_treasury_features(df)

    assert len(result) == len(df)