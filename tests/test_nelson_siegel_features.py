import numpy as np
import pandas as pd
import pytest

from src.features.nelson_siegel_features import (
    MATURITIES,
    NELSON_SIEGEL_LAMBDA,
    YIELD_COLUMNS,
    build_nelson_siegel_features,
    fit_nelson_siegel_curve,
    nelson_siegel_loadings,
)


def create_exact_nelson_siegel_curve(
    beta0=5.0,
    beta1=-2.0,
    beta2=1.0,
):
    """Generate yields exactly from known NS factors."""

    loadings = nelson_siegel_loadings(
        MATURITIES,
        decay=NELSON_SIEGEL_LAMBDA,
    )

    betas = np.array(
        [
            beta0,
            beta1,
            beta2,
        ]
    )

    return loadings @ betas


def create_treasury_sample(
    rows=30,
):
    """Create deterministic full-grid Treasury data."""

    base_curve = (
        create_exact_nelson_siegel_curve()
    )

    records = []

    for i in range(rows):

        # Parallel +1 bp shift per observation.
        shifted_curve = (
            base_curve
            + i * 0.01
        )

        record = {
            "timestamp":
                pd.Timestamp("2026-01-01")
                + pd.Timedelta(days=i)
        }

        for column, value in zip(
            YIELD_COLUMNS,
            shifted_curve,
        ):
            record[column] = value

        records.append(record)

    return pd.DataFrame(records)


def test_loadings_have_correct_shape():

    loadings = nelson_siegel_loadings(
        MATURITIES
    )

    assert loadings.shape == (
        len(MATURITIES),
        3,
    )


def test_level_loading_is_one():

    loadings = nelson_siegel_loadings(
        MATURITIES
    )

    assert np.allclose(
        loadings[:, 0],
        1.0,
    )


def test_invalid_maturity_rejected():

    with pytest.raises(ValueError):

        nelson_siegel_loadings(
            np.array(
                [
                    0.0,
                    1.0,
                    5.0,
                ]
            )
        )


def test_exact_curve_recovers_known_betas():

    expected_beta0 = 5.0
    expected_beta1 = -2.0
    expected_beta2 = 1.0

    yields = (
        create_exact_nelson_siegel_curve(
            beta0=expected_beta0,
            beta1=expected_beta1,
            beta2=expected_beta2,
        )
    )

    result = fit_nelson_siegel_curve(
        yields
    )

    assert np.isclose(
        result["ns_beta0_level"],
        expected_beta0,
        atol=1e-10,
    )

    assert np.isclose(
        result["ns_beta1_slope"],
        expected_beta1,
        atol=1e-10,
    )

    assert np.isclose(
        result["ns_beta2_curvature"],
        expected_beta2,
        atol=1e-10,
    )


def test_exact_curve_has_near_zero_rmse():

    yields = (
        create_exact_nelson_siegel_curve()
    )

    result = fit_nelson_siegel_curve(
        yields
    )

    assert result["ns_fit_rmse"] < 1e-10


def test_feature_columns_created():

    df = create_treasury_sample()

    result = (
        build_nelson_siegel_features(
            df
        )
    )

    expected = {
        "ns_beta0_level",
        "ns_beta1_slope",
        "ns_beta2_curvature",
        "ns_fit_rmse",
        "ns_beta0_level_change_1d_bp",
        "ns_beta0_level_change_5d_bp",
        "ns_beta0_level_change_20d_bp",
        "ns_beta1_slope_change_5d_bp",
        "ns_beta2_curvature_change_5d_bp",
        "ns_beta0_level_volatility_20d",
        "ns_beta1_slope_volatility_20d",
        "ns_beta2_curvature_volatility_20d",
    }

    assert expected.issubset(
        result.columns
    )


def test_parallel_shift_changes_only_level():

    df = create_treasury_sample()

    result = (
        build_nelson_siegel_features(
            df
        )
    )

    # Every maturity shifts upward exactly 1 bp
    # per observation.
    assert np.isclose(
        result[
            "ns_beta0_level_change_1d_bp"
        ].iloc[-1],
        1.0,
        atol=1e-8,
    )

    # Parallel shifts should leave slope and
    # curvature essentially unchanged.
    assert np.isclose(
        result[
            "ns_beta1_slope_change_1d_bp"
        ].iloc[-1],
        0.0,
        atol=1e-8,
    )

    assert np.isclose(
        result[
            "ns_beta2_curvature_change_1d_bp"
        ].iloc[-1],
        0.0,
        atol=1e-8,
    )


def test_five_day_parallel_shift():

    df = create_treasury_sample()

    result = (
        build_nelson_siegel_features(
            df
        )
    )

    assert np.isclose(
        result[
            "ns_beta0_level_change_5d_bp"
        ].iloc[-1],
        5.0,
        atol=1e-8,
    )


def test_incomplete_curve_removed():

    df = create_treasury_sample()

    df.loc[
        10,
        "yield_7y",
    ] = np.nan

    result = (
        build_nelson_siegel_features(
            df
        )
    )

    assert len(result) == len(df) - 1


def test_missing_required_column_rejected():

    df = create_treasury_sample()

    df = df.drop(
        columns=[
            "yield_20y"
        ]
    )

    with pytest.raises(
        ValueError
    ):
        build_nelson_siegel_features(
            df
        )