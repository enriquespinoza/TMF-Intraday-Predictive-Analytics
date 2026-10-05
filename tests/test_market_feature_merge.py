import numpy as np
import pandas as pd

from src.features.merge_market_features import (
    align_treasury_to_tmf,
)


def create_tmf_sample():
    """Create deterministic TMF market observations."""

    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
            ],
            "close": [
                30.0,
                31.0,
                32.0,
            ],
        }
    )


def create_treasury_sample():
    """Create deterministic full-curve Treasury observations."""

    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-02",
                "2026-01-06",
            ],
            "yield_3m": [
                4.00,
                4.10,
            ],
            "yield_6m": [
                4.05,
                4.15,
            ],
            "yield_1y": [
                4.10,
                4.20,
            ],
            "yield_2y": [
                4.20,
                4.30,
            ],
            "yield_3y": [
                4.30,
                4.40,
            ],
            "yield_5y": [
                4.40,
                4.50,
            ],
            "yield_7y": [
                4.50,
                4.60,
            ],
            "yield_10y": [
                4.60,
                4.70,
            ],
            "yield_20y": [
                4.80,
                4.90,
            ],
            "yield_30y": [
                5.00,
                5.10,
            ],
        }
    )


def test_merge_preserves_tmf_rows():

    tmf = create_tmf_sample()
    treasury = create_treasury_sample()

    result = align_treasury_to_tmf(
        tmf,
        treasury,
    )

    assert len(result) == len(tmf)


def test_exact_date_match():

    tmf = create_tmf_sample()
    treasury = create_treasury_sample()

    result = align_treasury_to_tmf(
        tmf,
        treasury,
    )

    first = result.iloc[0]

    assert first["yield_30y"] == 5.00

    assert (
        first["treasury_observation_date"]
        == pd.Timestamp("2026-01-02")
    )

    assert first["treasury_data_age_days"] == 0


def test_previous_treasury_observation_used():

    tmf = create_tmf_sample()
    treasury = create_treasury_sample()

    result = align_treasury_to_tmf(
        tmf,
        treasury,
    )

    jan_5 = result[
        result["market_date"]
        == pd.Timestamp("2026-01-05")
    ].iloc[0]

    # Jan. 5 has no Treasury observation in the fixture,
    # so the most recent prior observation must be Jan. 2.
    assert jan_5["yield_30y"] == 5.00

    assert (
        jan_5["treasury_observation_date"]
        == pd.Timestamp("2026-01-02")
    )

    assert jan_5["treasury_data_age_days"] == 3


def test_future_treasury_data_not_used():

    tmf = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-05",
            ],
            "close": [
                31.0,
            ],
        }
    )

    treasury = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-06",
            ],
            "yield_3m": [
                4.10,
            ],
            "yield_6m": [
                4.15,
            ],
            "yield_1y": [
                4.20,
            ],
            "yield_2y": [
                4.30,
            ],
            "yield_3y": [
                4.40,
            ],
            "yield_5y": [
                4.50,
            ],
            "yield_7y": [
                4.60,
            ],
            "yield_10y": [
                4.70,
            ],
            "yield_20y": [
                4.90,
            ],
            "yield_30y": [
                5.10,
            ],
        }
    )

    result = align_treasury_to_tmf(
        tmf,
        treasury,
    )

    # The only Treasury observation occurs after the TMF
    # observation. A backward as-of merge must not use it.
    assert pd.isna(
        result["yield_30y"].iloc[0]
    )

    assert pd.isna(
        result[
            "treasury_observation_date"
        ].iloc[0]
    )


def test_v2_curve_features_are_aligned():

    tmf = create_tmf_sample()
    treasury = create_treasury_sample()

    result = align_treasury_to_tmf(
        tmf,
        treasury,
    )

    first = result.iloc[0]

    # Jan. 2:
    # 3M = 4.00%
    # 30Y = 5.00%
    #
    # Slope = 100 bp.
    assert np.isclose(
        first["curve_slope_30y_3m_bp"],
        100.0,
    )

    # Jan. 2:
    # 3M = 4.00%
    # 5Y = 4.40%
    # 30Y = 5.00%
    #
    # Curvature =
    # (2 * 4.40 - 4.00 - 5.00) * 100
    # = -20 bp.
    assert np.isclose(
        first["curve_curvature_5y_bp"],
        -20.0,
    )


def test_previous_v2_curve_features_are_used():

    tmf = create_tmf_sample()
    treasury = create_treasury_sample()

    result = align_treasury_to_tmf(
        tmf,
        treasury,
    )

    jan_5 = result[
        result["market_date"]
        == pd.Timestamp("2026-01-05")
    ].iloc[0]

    # Jan. 5 must inherit the Jan. 2 curve features,
    # not the future Jan. 6 Treasury observation.
    assert np.isclose(
        jan_5["curve_slope_30y_3m_bp"],
        100.0,
    )

    assert np.isclose(
        jan_5["curve_curvature_5y_bp"],
        -20.0,
    )

    assert (
        jan_5["treasury_observation_date"]
        == pd.Timestamp("2026-01-02")
    )
def test_future_v2_curve_features_not_used():

    tmf = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-05",
            ],
            "close": [
                31.0,
            ],
        }
    )

    treasury = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-06",
            ],
            "yield_3m": [
                4.10,
            ],
            "yield_6m": [
                4.15,
            ],
            "yield_1y": [
                4.20,
            ],
            "yield_2y": [
                4.30,
            ],
            "yield_3y": [
                4.40,
            ],
            "yield_5y": [
                4.50,
            ],
            "yield_7y": [
                4.60,
            ],
            "yield_10y": [
                4.70,
            ],
            "yield_20y": [
                4.90,
            ],
            "yield_30y": [
                5.10,
            ],
        }
    )

    result = align_treasury_to_tmf(
        tmf,
        treasury,
    )

    # The only Treasury observation occurs after
    # the TMF observation. V2A features must not
    # leak backward from Jan. 6 into Jan. 5.
    assert pd.isna(
        result[
            "curve_level"
        ].iloc[0]
    )

    assert pd.isna(
        result[
            "curve_slope_30y_3m_bp"
        ].iloc[0]
    )

    assert pd.isna(
        result[
            "curve_curvature_5y_bp"
        ].iloc[0]
    )

def test_v2b_nelson_siegel_features_are_aligned():

    tmf = create_tmf_sample()
    treasury = create_treasury_sample()

    result = align_treasury_to_tmf(
        tmf,
        treasury,
    )

    expected_columns = {
        "ns_beta0_level",
        "ns_beta1_slope",
        "ns_beta2_curvature",
        "ns_fit_rmse",
        "ns_beta0_level_change_5d_bp",
        "ns_beta1_slope_change_5d_bp",
        "ns_beta2_curvature_change_5d_bp",
        "ns_beta0_level_volatility_20d",
        "ns_beta1_slope_volatility_20d",
        "ns_beta2_curvature_volatility_20d",
    }

    assert expected_columns.issubset(
        result.columns
    )

    first = result.iloc[0]

    # The fitted Nelson-Siegel factors should exist
    # for the Jan. 2 Treasury curve.
    assert pd.notna(
        first["ns_beta0_level"]
    )

    assert pd.notna(
        first["ns_beta1_slope"]
    )

    assert pd.notna(
        first["ns_beta2_curvature"]
    )

    assert pd.notna(
        first["ns_fit_rmse"]
    )


def test_previous_v2b_nelson_siegel_features_are_used():

    tmf = create_tmf_sample()
    treasury = create_treasury_sample()

    result = align_treasury_to_tmf(
        tmf,
        treasury,
    )

    jan_2 = result[
        result["market_date"]
        == pd.Timestamp("2026-01-02")
    ].iloc[0]

    jan_5 = result[
        result["market_date"]
        == pd.Timestamp("2026-01-05")
    ].iloc[0]

    # Jan. 5 has no Treasury observation.
    # It must inherit the Nelson-Siegel factors
    # calculated from the Jan. 2 Treasury curve.
    assert np.isclose(
        jan_5["ns_beta0_level"],
        jan_2["ns_beta0_level"],
    )

    assert np.isclose(
        jan_5["ns_beta1_slope"],
        jan_2["ns_beta1_slope"],
    )

    assert np.isclose(
        jan_5["ns_beta2_curvature"],
        jan_2["ns_beta2_curvature"],
    )

    assert np.isclose(
        jan_5["ns_fit_rmse"],
        jan_2["ns_fit_rmse"],
    )

    assert (
        jan_5["treasury_observation_date"]
        == pd.Timestamp("2026-01-02")
    )

    assert (
        jan_5["treasury_data_age_days"]
        == 3
    )


def test_future_v2b_nelson_siegel_features_not_used():

    tmf = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-05",
            ],
            "close": [
                31.0,
            ],
        }
    )

    treasury = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-06",
            ],
            "yield_3m": [
                4.10,
            ],
            "yield_6m": [
                4.15,
            ],
            "yield_1y": [
                4.20,
            ],
            "yield_2y": [
                4.30,
            ],
            "yield_3y": [
                4.40,
            ],
            "yield_5y": [
                4.50,
            ],
            "yield_7y": [
                4.60,
            ],
            "yield_10y": [
                4.70,
            ],
            "yield_20y": [
                4.90,
            ],
            "yield_30y": [
                5.10,
            ],
        }
    )

    result = align_treasury_to_tmf(
        tmf,
        treasury,
    )

    # The only Treasury observation is Jan. 6.
    # The TMF observation is Jan. 5.
    #
    # A backward point-in-time merge must not
    # use the future Jan. 6 Nelson-Siegel factors.
    assert pd.isna(
        result[
            "ns_beta0_level"
        ].iloc[0]
    )

    assert pd.isna(
        result[
            "ns_beta1_slope"
        ].iloc[0]
    )

    assert pd.isna(
        result[
            "ns_beta2_curvature"
        ].iloc[0]
    )

    assert pd.isna(
        result[
            "ns_fit_rmse"
        ].iloc[0]
    )

    assert pd.isna(
        result[
            "treasury_observation_date"
        ].iloc[0]
    )