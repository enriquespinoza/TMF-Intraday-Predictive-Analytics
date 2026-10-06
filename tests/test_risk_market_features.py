import pandas as pd
import pytest

from src.features.risk_market_features import (
    add_risk_inputs,
    prepare_real_yield_10y,
)


def test_real_yield_change_uses_five_valid_observations():
    data = pd.DataFrame(
        {
            "reference_date":
                pd.to_datetime(
                    [
                        "2025-01-02",
                        "2025-01-03",
                        "2025-01-06",
                        "2025-01-07",
                        "2025-01-08",
                        "2025-01-09",
                    ]
                ),
            "value": [
                2.00,
                2.01,
                2.02,
                2.03,
                2.04,
                2.10,
            ],
            "series_name":
                ["real_yield_10y"] * 6,
        }
    )

    result = prepare_real_yield_10y(
        data
    )

    assert result.loc[
        5,
        "real_yield_10y_change_5d_bp",
    ] == pytest.approx(
        10.0
    )


def test_missing_observation_does_not_count():
    data = pd.DataFrame(
        {
            "reference_date":
                pd.to_datetime(
                    [
                        "2025-01-02",
                        "2025-01-03",
                        "2025-01-06",
                        "2025-01-07",
                        "2025-01-08",
                        "2025-01-09",
                        "2025-01-10",
                    ]
                ),
            "value": [
                2.00,
                2.01,
                None,
                2.02,
                2.03,
                2.04,
                2.10,
            ],
            "series_name":
                ["real_yield_10y"] * 7,
        }
    )

    result = prepare_real_yield_10y(
        data
    )

    # Valid observations:
    # 2.00, 2.01, 2.02, 2.03, 2.04, 2.10
    assert result.iloc[
        -1
    ][
        "real_yield_10y_change_5d_bp"
    ] == pytest.approx(
        10.0
    )


def test_risk_input_transformations():
    data = pd.DataFrame(
        {
            "market_date":
                pd.to_datetime(
                    ["2025-01-10"]
                ),
            "close":
                [40.0],
            "ema_21":
                [41.0],
            "return_1d":
                [-0.025],
            "change_30y_1d_bp":
                [8.0],
            "change_30y_5d_bp":
                [15.0],
            "volatility_5d":
                [0.03],
            "volatility_20d":
                [0.02],
            "real_yield_10y_state_date":
                pd.to_datetime(
                    ["2025-01-10"]
                ),
            "real_yield_10y_change_5d_bp":
                [12.0],
        }
    )

    result = add_risk_inputs(
        data
    )

    row = result.iloc[0]

    assert row[
        "tmf_return_1d_pct"
    ] == pytest.approx(
        -2.5
    )

    assert row[
        "volatility_ratio_5d_20d"
    ] == pytest.approx(
        1.5
    )

    assert bool(
        row[
            "below_ema21"
        ]
    ) is True

    assert bool(
        row[
            "critical_data_stale"
        ]
    ) is False


def test_old_real_yield_is_stale():
    data = pd.DataFrame(
        {
            "market_date":
                pd.to_datetime(
                    ["2025-01-10"]
                ),
            "close":
                [40.0],
            "ema_21":
                [41.0],
            "return_1d":
                [0.01],
            "change_30y_1d_bp":
                [1.0],
            "change_30y_5d_bp":
                [2.0],
            "volatility_5d":
                [0.02],
            "volatility_20d":
                [0.02],
            "real_yield_10y_state_date":
                pd.to_datetime(
                    ["2025-01-03"]
                ),
            "real_yield_10y_change_5d_bp":
                [2.0],
        }
    )

    result = add_risk_inputs(
        data
    )

    assert bool(
        result.loc[
            0,
            "critical_data_stale",
        ]
    ) is True