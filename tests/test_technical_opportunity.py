import numpy as np
import pandas as pd
import pytest

from src.signals.technical_opportunity import (
    add_technical_opportunity_features,
    ema_pullback_score,
    trailing_percentile,
)


def test_trailing_percentile_uses_only_prior_values():
    values = pd.Series(
        [
            10.0,
            20.0,
            30.0,
            40.0,
        ]
    )

    result = trailing_percentile(
        values,
        lookback=3,
        min_history=2,
    )

    assert np.isnan(
        result.iloc[0]
    )

    assert np.isnan(
        result.iloc[1]
    )

    # 30 is above both prior values.
    assert result.iloc[2] == pytest.approx(
        1.0
    )

    # 40 compares only with 10,20,30.
    assert result.iloc[3] == pytest.approx(
        1.0
    )


def test_current_value_is_not_in_own_history():
    values = pd.Series(
        [
            1.0,
            1.0,
            100.0,
        ]
    )

    result = trailing_percentile(
        values,
        lookback=10,
        min_history=2,
    )

    # If current value were incorrectly included,
    # percentile would be below 1.0.
    assert result.iloc[2] == pytest.approx(
        1.0
    )


def test_rolling_window_drops_old_observations():
    values = pd.Series(
        [
            100.0,
            1.0,
            2.0,
            3.0,
        ]
    )

    result = trailing_percentile(
        values,
        lookback=2,
        min_history=2,
    )

    # At row 3, history must be [1, 2].
    # The old 100 observation must not remain.
    assert result.iloc[3] == pytest.approx(
        1.0
    )


def test_ema_pullback_scoring():
    streak = pd.Series(
        [0, 1, 2, 3, 4, 10]
    )

    result = ema_pullback_score(
        streak
    )

    assert result.tolist() == [
        0.5,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
    ]


def test_deeper_pullback_creates_higher_score():
    dates = pd.date_range(
        "2025-01-01",
        periods=6,
        freq="D",
    )

    data = pd.DataFrame(
        {
            "market_date": dates,

            "return_1d": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.01,
            ],

            "return_5d": [
                -0.01,
                -0.02,
                -0.03,
                -0.04,
                -0.05,
                -0.10,
            ],

            "rsi_14": [
                50,
                45,
                40,
                35,
                30,
                20,
            ],

            "distance_from_20d_high": [
                -0.01,
                -0.02,
                -0.03,
                -0.04,
                -0.05,
                -0.10,
            ],

            "below_ema21_streak": [
                0,
                0,
                0,
                0,
                0,
                1,
            ],
        }
    )

    result = (
        add_technical_opportunity_features(
            data,
            lookback=5,
            min_history=5,
        )
    )

    last = result.iloc[-1]

    assert last[
        "tech_pullback_5d_score"
    ] == pytest.approx(
        1.0
    )

    assert last[
        "tech_rsi_score"
    ] == pytest.approx(
        1.0
    )

    assert last[
        "tech_distance_high_score"
    ] == pytest.approx(
        1.0
    )

    assert last[
        "tech_ema_pullback_score"
    ] == pytest.approx(
        1.0
    )


def test_scores_stay_between_zero_and_one():
    n = 100

    data = pd.DataFrame(
        {
            "market_date":
                pd.date_range(
                    "2025-01-01",
                    periods=n,
                    freq="D",
                ),

            "return_1d":
                np.linspace(
                    -0.05,
                    0.05,
                    n,
                ),

            "return_5d":
                np.linspace(
                    -0.10,
                    0.10,
                    n,
                ),

            "rsi_14":
                np.linspace(
                    20,
                    80,
                    n,
                ),

            "distance_from_20d_high":
                np.linspace(
                    -0.20,
                    0.0,
                    n,
                ),

            "below_ema21_streak":
                [0] * n,
        }
    )

    result = (
        add_technical_opportunity_features(
            data,
            lookback=63,
            min_history=20,
        )
    )

    score_columns = [
        "tech_pullback_5d_score",
        "tech_rsi_score",
        "tech_distance_high_score",
        "tech_stabilization_score",
        "tech_ema_pullback_score",
    ]

    valid = result[
        score_columns
    ].dropna()

    assert (
        valid.min().min()
        >= 0.0
    )

    assert (
        valid.max().max()
        <= 1.0
    )


def test_final_score_stays_between_zero_and_100():
    n = 100

    data = pd.DataFrame(
        {
            "market_date":
                pd.date_range(
                    "2025-01-01",
                    periods=n,
                    freq="D",
                ),

            "return_1d":
                np.random.default_rng(
                    42
                ).normal(
                    0,
                    0.02,
                    n,
                ),

            "return_5d":
                np.random.default_rng(
                    43
                ).normal(
                    0,
                    0.05,
                    n,
                ),

            "rsi_14":
                np.linspace(
                    25,
                    75,
                    n,
                ),

            "distance_from_20d_high":
                np.linspace(
                    -0.25,
                    0.0,
                    n,
                ),

            "below_ema21_streak":
                [0] * n,
        }
    )

    result = (
        add_technical_opportunity_features(
            data,
            lookback=63,
            min_history=20,
        )
    )

    valid = result[
        "technical_opportunity_score"
    ].dropna()

    assert (
        valid.min()
        >= 0.0
    )

    assert (
        valid.max()
        <= 100.0
    )