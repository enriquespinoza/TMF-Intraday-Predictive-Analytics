import numpy as np
import pandas as pd
import pytest

from src.signals.market_confirmation import (
    add_market_confirmation_features,
    calculate_signed_relative_volume,
)


def test_signed_volume_direction():
    returns = pd.Series(
        [
            0.02,
            -0.02,
            0.0,
        ]
    )

    volume = pd.Series(
        [
            2.0,
            2.0,
            2.0,
        ]
    )

    result = (
        calculate_signed_relative_volume(
            returns,
            volume,
        )
    )

    assert result.tolist() == [
        2.0,
        -2.0,
        0.0,
    ]


def test_high_volume_up_day_is_positive_confirmation():
    n = 6

    data = pd.DataFrame(
        {
            "market_date":
                pd.date_range(
                    "2025-01-01",
                    periods=n,
                    freq="D",
                ),

            "return_1d": [
                -0.01,
                -0.01,
                -0.01,
                0.01,
                0.01,
                0.03,
            ],

            "return_3d": [
                -0.05,
                -0.04,
                -0.03,
                -0.02,
                -0.01,
                0.05,
            ],

            "close_vs_ema9": [
                -0.05,
                -0.04,
                -0.03,
                -0.02,
                -0.01,
                0.05,
            ],

            "ema_spread_pct": [
                -0.05,
                -0.04,
                -0.03,
                -0.02,
                -0.01,
                0.05,
            ],

            "relative_volume": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                3.0,
            ],
        }
    )

    result = (
        add_market_confirmation_features(
            data,
            lookback=5,
            min_history=5,
        )
    )

    last = result.iloc[-1]

    assert last[
        "market_price_response_score"
    ] == pytest.approx(
        1.0
    )

    assert last[
        "market_ema9_confirmation_score"
    ] == pytest.approx(
        1.0
    )

    assert last[
        "market_ema_structure_score"
    ] == pytest.approx(
        1.0
    )

    assert last[
        "market_volume_participation_score"
    ] == pytest.approx(
        1.0
    )

    assert last[
        "market_confirmation_score"
    ] == pytest.approx(
        100.0
    )


def test_high_volume_down_day_has_weak_volume_confirmation():
    data = pd.DataFrame(
        {
            "market_date":
                pd.date_range(
                    "2025-01-01",
                    periods=6,
                    freq="D",
                ),

            "return_1d": [
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                -0.05,
            ],

            "return_3d": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.05,
            ],

            "close_vs_ema9": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.05,
            ],

            "ema_spread_pct": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.05,
            ],

            "relative_volume": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                3.0,
            ],
        }
    )

    result = (
        add_market_confirmation_features(
            data,
            lookback=5,
            min_history=5,
        )
    )

    last = result.iloc[-1]

    assert last[
        "signed_relative_volume"
    ] == pytest.approx(
        -3.0
    )

    assert last[
        "market_volume_participation_score"
    ] == pytest.approx(
        0.0
    )


def test_current_row_is_excluded_from_history():
    data = pd.DataFrame(
        {
            "market_date":
                pd.date_range(
                    "2025-01-01",
                    periods=6,
                    freq="D",
                ),

            "return_1d":
                [0.01] * 6,

            "return_3d": [
                1,
                2,
                3,
                4,
                5,
                100,
            ],

            "close_vs_ema9": [
                1,
                2,
                3,
                4,
                5,
                100,
            ],

            "ema_spread_pct": [
                1,
                2,
                3,
                4,
                5,
                100,
            ],

            "relative_volume": [
                1,
                1,
                1,
                1,
                1,
                10,
            ],
        }
    )

    result = (
        add_market_confirmation_features(
            data,
            lookback=5,
            min_history=5,
        )
    )

    last = result.iloc[-1]

    assert last[
        "market_price_response_score"
    ] == pytest.approx(
        1.0
    )

    assert last[
        "market_ema9_confirmation_score"
    ] == pytest.approx(
        1.0
    )


def test_market_scores_are_bounded():
    rng = np.random.default_rng(
        42
    )

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
                rng.normal(
                    0,
                    0.02,
                    n,
                ),

            "return_3d":
                rng.normal(
                    0,
                    0.04,
                    n,
                ),

            "close_vs_ema9":
                rng.normal(
                    0,
                    0.03,
                    n,
                ),

            "ema_spread_pct":
                rng.normal(
                    0,
                    0.02,
                    n,
                ),

            "relative_volume":
                rng.uniform(
                    0.5,
                    2.0,
                    n,
                ),
        }
    )

    result = (
        add_market_confirmation_features(
            data,
            lookback=63,
            min_history=20,
        )
    )

    components = [
        "market_price_response_score",
        "market_ema9_confirmation_score",
        "market_ema_structure_score",
        "market_volume_participation_score",
    ]

    valid = result[
        components
    ].dropna()

    assert (
        valid.min().min()
        >= 0.0
    )

    assert (
        valid.max().max()
        <= 1.0
    )

    final = result[
        "market_confirmation_score"
    ].dropna()

    assert final.min() >= 0.0
    assert final.max() <= 100.0


def test_missing_column_rejected():
    data = pd.DataFrame(
        {
            "market_date":
                pd.date_range(
                    "2025-01-01",
                    periods=5,
                    freq="D",
                ),

            "return_1d":
                [0.0] * 5,

            "return_3d":
                [0.0] * 5,

            "close_vs_ema9":
                [0.0] * 5,

            "ema_spread_pct":
                [0.0] * 5,
        }
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        add_market_confirmation_features(
            data
        )