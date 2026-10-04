import pandas as pd

from src.analysis.treasury_regimes import (
    add_treasury_regime,
    summarize_regimes,
)


def test_bull_steepening():

    df = pd.DataFrame(
        {
            "change_2y_5d_bp": [-20.0],
            "change_30y_5d_bp": [-10.0],
        }
    )

    result = add_treasury_regime(df)

    assert (
        result["treasury_regime"].iloc[0]
        == "bull_steepening"
    )


def test_bull_flattening():

    df = pd.DataFrame(
        {
            "change_2y_5d_bp": [-10.0],
            "change_30y_5d_bp": [-20.0],
        }
    )

    result = add_treasury_regime(df)

    assert (
        result["treasury_regime"].iloc[0]
        == "bull_flattening"
    )


def test_bear_steepening():

    df = pd.DataFrame(
        {
            "change_2y_5d_bp": [10.0],
            "change_30y_5d_bp": [20.0],
        }
    )

    result = add_treasury_regime(df)

    assert (
        result["treasury_regime"].iloc[0]
        == "bear_steepening"
    )


def test_bear_flattening():

    df = pd.DataFrame(
        {
            "change_2y_5d_bp": [20.0],
            "change_30y_5d_bp": [10.0],
        }
    )

    result = add_treasury_regime(df)

    assert (
        result["treasury_regime"].iloc[0]
        == "bear_flattening"
    )


def test_twist_regimes():

    df = pd.DataFrame(
        {
            "change_2y_5d_bp": [
                -10.0,
                10.0,
            ],
            "change_30y_5d_bp": [
                10.0,
                -10.0,
            ],
        }
    )

    result = add_treasury_regime(df)

    assert (
        result["treasury_regime"].iloc[0]
        == "twist_steepening"
    )

    assert (
        result["treasury_regime"].iloc[1]
        == "twist_flattening"
    )


def test_regime_summary():

    df = pd.DataFrame(
        {
            "treasury_regime": [
                "bull_steepening",
                "bull_steepening",
            ],
            "forward_return_1": [
                0.01,
                -0.01,
            ],
            "forward_return_3": [
                0.02,
                0.01,
            ],
            "forward_return_5": [
                0.05,
                -0.01,
            ],
            "forward_return_10": [
                0.06,
                0.02,
            ],
            "forward_return_20": [
                0.10,
                0.04,
            ],
        }
    )

    result = summarize_regimes(df)

    assert len(result) == 1

    assert (
        result["mean_return_5d"].iloc[0]
        == 0.02
    )

    assert (
        result["positive_rate_5d"].iloc[0]
        == 0.5
    )
    