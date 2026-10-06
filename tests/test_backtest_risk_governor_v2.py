import pandas as pd
import pytest

from src.analysis.backtest_risk_governor_v2 import (
    add_overlay_returns,
    evaluate_chronologically,
)


def make_row(
    date,
    yield_5d=0.0,
    below_ema=False,
):
    return {
        "market_date":
            pd.Timestamp(date),

        "tmf_return_1d_pct":
            0.5,

        "change_30y_1d_bp":
            0.0,

        "change_30y_5d_bp":
            yield_5d,

        "real_yield_10y_change_5d_bp":
            0.0,

        "volatility_ratio_5d_20d":
            1.0,

        "below_ema21":
            below_ema,

        "critical_data_stale":
            False,

        "return_1d":
            0.01,
    }


def test_chronological_hysteresis_is_preserved():
    data = pd.DataFrame(
        [
            make_row(
                "2026-01-02"
            ),

            make_row(
                "2026-01-05",
                yield_5d=30.0,
            ),

            make_row(
                "2026-01-06"
            ),

            make_row(
                "2026-01-07"
            ),

            make_row(
                "2026-01-08"
            ),

            make_row(
                "2026-01-09"
            ),
        ]
    )

    result = evaluate_chronologically(
        data
    )

    assert result[
        "exposure_state"
    ].tolist() == [
        "GREEN",
        "RED",
        "RED",
        "YELLOW",
        "YELLOW",
        "GREEN",
    ]


def test_ema_blocks_entry_but_not_exposure():
    data = pd.DataFrame(
        [
            make_row(
                "2026-01-02",
                below_ema=True,
            )
        ]
    )

    result = evaluate_chronologically(
        data
    )

    assert result.loc[
        0,
        "exposure_state",
    ] == "GREEN"

    assert bool(
        result.loc[
            0,
            "entry_blocked",
        ]
    ) is True

    assert result.loc[
        0,
        "max_position_multiplier",
    ] == pytest.approx(
        1.0
    )


def test_exposure_applies_next_day():
    data = pd.DataFrame(
        {
            "market_date":
                pd.to_datetime(
                    [
                        "2026-01-02",
                        "2026-01-05",
                    ]
                ),

            "max_position_multiplier":
                [
                    1.0,
                    0.0,
                ],

            "return_1d":
                [
                    0.01,
                    -0.05,
                ],
        }
    )

    result = add_overlay_returns(
        data
    )

    # Day 1 has no prior decision.
    assert result.loc[
        0,
        "strategy_exposure",
    ] == pytest.approx(
        0.0
    )

    # Day 2 still receives the prior day's GREEN exposure.
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
        -0.05
    )