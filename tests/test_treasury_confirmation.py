import numpy as np
import pandas as pd
import pytest

from src.signals.treasury_confirmation import (
    add_treasury_confirmation_features,
)


def make_data():
    return pd.DataFrame(
        {
            "market_date":
                pd.date_range(
                    "2025-01-01",
                    periods=6,
                    freq="D",
                ),

            "change_30y_1d_bp": [
                5,
                4,
                3,
                2,
                1,
                -10,
            ],

            "change_30y_5d_bp": [
                10,
                8,
                6,
                4,
                2,
                -20,
            ],

            "real_yield_10y_change_5d_bp": [
                8,
                6,
                4,
                2,
                1,
                -15,
            ],
        }
    )


def test_large_yield_decline_gets_high_score():
    result = (
        add_treasury_confirmation_features(
            make_data(),
            lookback=5,
            min_history=5,
        )
    )

    last = result.iloc[-1]

    assert last[
        "treasury_30y_1d_score"
    ] == pytest.approx(
        1.0
    )

    assert last[
        "treasury_30y_5d_score"
    ] == pytest.approx(
        1.0
    )

    assert last[
        "treasury_real_10y_5d_score"
    ] == pytest.approx(
        1.0
    )

    assert last[
        "treasury_confirmation_score"
    ] == pytest.approx(
        100.0
    )


def test_scores_are_bounded():
    n = 100

    data = pd.DataFrame(
        {
            "market_date":
                pd.date_range(
                    "2025-01-01",
                    periods=n,
                    freq="D",
                ),

            "change_30y_1d_bp":
                np.linspace(
                    -20,
                    20,
                    n,
                ),

            "change_30y_5d_bp":
                np.linspace(
                    -40,
                    40,
                    n,
                ),

            "real_yield_10y_change_5d_bp":
                np.linspace(
                    -30,
                    30,
                    n,
                ),
        }
    )

    result = (
        add_treasury_confirmation_features(
            data,
            lookback=63,
            min_history=20,
        )
    )

    components = [
        "treasury_30y_1d_score",
        "treasury_30y_5d_score",
        "treasury_real_10y_5d_score",
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
        "treasury_confirmation_score"
    ].dropna()

    assert final.min() >= 0.0
    assert final.max() <= 100.0


def test_current_observation_does_not_affect_history():
    data = make_data()

    original = (
        add_treasury_confirmation_features(
            data,
            lookback=5,
            min_history=3,
        )
    )

    modified = data.copy()

    modified.loc[
        5,
        "change_30y_1d_bp",
    ] = -1000.0

    changed = (
        add_treasury_confirmation_features(
            modified,
            lookback=5,
            min_history=3,
        )
    )

    # Earlier rows cannot change when a future/current
    # observation is modified.
    pd.testing.assert_series_equal(
        original.loc[
            :4,
            "treasury_30y_1d_score",
        ],
        changed.loc[
            :4,
            "treasury_30y_1d_score",
        ],
    )


def test_missing_column_rejected():
    data = make_data().drop(
        columns=[
            "change_30y_5d_bp"
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        add_treasury_confirmation_features(
            data,
            lookback=5,
            min_history=3,
        )