import pandas as pd
import pytest

from src.analysis.backtest_risk_overlay import (
    STATE_EXPOSURE,
    build_overlay,
)


def test_state_exposures_are_frozen():
    assert STATE_EXPOSURE == {
        "GREEN": 1.0,
        "YELLOW": 0.5,
        "RED": 0.0,
        "EMERGENCY_EXIT": 0.0,
    }


def test_state_is_lagged_one_day():
    data = pd.DataFrame(
        {
            "market_date":
                pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-02",
                        "2026-01-05",
                    ]
                ),
            "risk_state": [
                "GREEN",
                "RED",
                "YELLOW",
            ],
            "return_1d": [
                0.01,
                0.02,
                -0.03,
            ],
        }
    )

    result = build_overlay(
        data
    )

    assert result[
        "strategy_exposure"
    ].tolist() == [
        0.0,
        1.0,
        0.0,
    ]


def test_red_state_does_not_remove_same_day_loss():
    data = pd.DataFrame(
        {
            "market_date":
                pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-02",
                    ]
                ),
            "risk_state": [
                "GREEN",
                "RED",
            ],
            "return_1d": [
                0.01,
                -0.08,
            ],
        }
    )

    result = build_overlay(
        data
    )

    # GREEN from prior day means the strategy is still exposed
    # during the day that produces the RED signal.
    assert result.loc[
        1,
        "strategy_exposure",
    ] == pytest.approx(
        1.0
    )

    assert result.loc[
        1,
        "overlay_return_gross",
    ] == pytest.approx(
        -0.08
    )


def test_yellow_means_half_exposure_next_day():
    data = pd.DataFrame(
        {
            "market_date":
                pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-02",
                    ]
                ),
            "risk_state": [
                "YELLOW",
                "GREEN",
            ],
            "return_1d": [
                0.0,
                0.04,
            ],
        }
    )

    result = build_overlay(
        data
    )

    assert result.loc[
        1,
        "strategy_exposure",
    ] == pytest.approx(
        0.5
    )

    assert result.loc[
        1,
        "overlay_return_gross",
    ] == pytest.approx(
        0.02
    )