import numpy as np
import pandas as pd

from src.features.treasury_features import (
    build_treasury_features,
)


def create_treasury_sample(
    rows: int = 30,
) -> pd.DataFrame:
    """Create deterministic full-curve Treasury data."""

    daily_change = np.arange(rows) * 0.01

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="D",
            ),
            "yield_3m": 4.00 + daily_change,
            "yield_6m": 4.10 + daily_change,
            "yield_1y": 4.20 + daily_change,
            "yield_2y": 4.30 + daily_change,
            "yield_3y": 4.40 + daily_change,
            "yield_5y": 4.50 + daily_change,
            "yield_7y": 4.60 + daily_change,
            "yield_10y": 4.70 + daily_change,
            "yield_20y": 4.90 + daily_change,
            "yield_30y": 5.00 + daily_change,
        }
    )


def test_treasury_feature_columns_created():

    df = create_treasury_sample()

    result = build_treasury_features(df)

    expected = {
        # Existing V1 features
        "change_2y_1d_bp",
        "change_10y_5d_bp",
        "change_30y_20d_bp",
        "curve_10y_2y_bp",
        "curve_30y_2y_bp",
        "curve_30y_10y_bp",
        "curve_30y_10y_bp_change_5d",
        "yield_30y_volatility_20d",

        # Expanded maturity features
        "change_3m_1d_bp",
        "change_6m_5d_bp",
        "change_1y_20d_bp",
        "change_3y_1d_bp",
        "change_7y_5d_bp",
        "change_20y_20d_bp",

        # V2 structural factors
        "curve_level",
        "curve_slope_30y_3m_bp",
        "curve_curvature_5y_bp",
        "curve_level_change_1d_bp",
        "curve_level_change_5d_bp",
        "curve_level_change_20d_bp",
        "curve_slope_change_1d_bp",
        "curve_slope_change_5d_bp",
        "curve_slope_change_20d_bp",
        "curve_curvature_change_1d_bp",
        "curve_curvature_change_5d_bp",
        "curve_curvature_change_20d_bp",
        "curve_level_volatility_20d",
        "curve_slope_volatility_20d",
        "curve_curvature_volatility_20d",
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

    # Every maturity increases exactly 1 bp per observation.
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


def test_v2_curve_level():

    df = create_treasury_sample()

    result = build_treasury_features(df)

    expected = df[
        [
            "yield_3m",
            "yield_6m",
            "yield_1y",
            "yield_2y",
            "yield_3y",
            "yield_5y",
            "yield_7y",
            "yield_10y",
            "yield_20y",
            "yield_30y",
        ]
    ].iloc[0].mean()

    actual = result[
        "curve_level"
    ].iloc[0]

    assert np.isclose(
        actual,
        expected,
    )


def test_v2_curve_slope():

    df = create_treasury_sample()

    result = build_treasury_features(df)

    # Initial curve:
    # 3M = 4.00%
    # 30Y = 5.00%
    #
    # Slope = 100 bp.
    expected = 100.0

    actual = result[
        "curve_slope_30y_3m_bp"
    ].iloc[0]

    assert np.isclose(
        actual,
        expected,
    )


def test_v2_curve_curvature():

    df = create_treasury_sample()

    result = build_treasury_features(df)

    # Initial curve:
    # 3M = 4.00%
    # 5Y = 4.50%
    # 30Y = 5.00%
    #
    # Curvature:
    # (2 * 4.50 - 4.00 - 5.00) * 100 = 0 bp.
    expected = 0.0

    actual = result[
        "curve_curvature_5y_bp"
    ].iloc[0]

    assert np.isclose(
        actual,
        expected,
    )


def test_v2_factor_changes():

    df = create_treasury_sample()

    result = build_treasury_features(df)

    # Every maturity rises exactly 1 bp per observation.
    #
    # Therefore the equal-weighted curve level also rises
    # exactly 1 bp per observation.
    assert np.isclose(
        result[
            "curve_level_change_1d_bp"
        ].iloc[-1],
        1.0,
    )

    assert np.isclose(
        result[
            "curve_level_change_5d_bp"
        ].iloc[-1],
        5.0,
    )

    # Because every maturity moves upward by the same amount,
    # the shape of the curve does not change.
    assert np.isclose(
        result[
            "curve_slope_change_5d_bp"
        ].iloc[-1],
        0.0,
    )

    assert np.isclose(
        result[
            "curve_curvature_change_5d_bp"
        ].iloc[-1],
        0.0,
    )


def test_incomplete_treasury_rows_removed():

    df = create_treasury_sample()

    df.loc[
        10,
        "yield_7y",
    ] = np.nan

    result = build_treasury_features(df)

    assert len(result) == len(df) - 1

    assert not result[
        "yield_7y"
    ].isna().any()