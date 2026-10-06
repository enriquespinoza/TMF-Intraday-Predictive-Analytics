import numpy as np
import pandas as pd
import pytest

from src.signals.macro_regime import (
    add_macro_regime_features,
    broad_revision_score,
    build_payroll_state_table,
    score_payroll_states,
)


def test_repeated_daily_rows_count_as_one_macro_state():
    data = pd.DataFrame(
        {
            "market_date":
                pd.date_range(
                    "2025-01-01",
                    periods=9,
                    freq="D",
                ),

            "payroll_change_1m":
                [
                    200, 200, 200,
                    150, 150, 150,
                    100, 100, 100,
                ],

            "payroll_revision_total":
                [
                    0, 0, 0,
                    -10, -10, -10,
                    -20, -20, -20,
                ],

            "payroll_broad_revision_event":
                [
                    0, 0, 0,
                    1, 1, 1,
                    1, 1, 1,
                ],
        }
    )

    states = build_payroll_state_table(
        data
    )

    assert len(
        states
    ) == 3


def test_lower_payroll_growth_gets_higher_score():
    states = pd.DataFrame(
        {
            "payroll_state_market_date":
                pd.to_datetime(
                    [
                        "2025-01-01",
                        "2025-02-01",
                        "2025-03-01",
                    ]
                ),

            "payroll_state_id":
                [1, 2, 3],

            "payroll_change_1m":
                [200, 150, 50],

            "payroll_revision_total":
                [0, -5, -20],

            "payroll_broad_revision_event":
                [0, 0, 1],
        }
    )

    result = score_payroll_states(
        states,
        lookback=2,
        min_history=2,
    )

    last = result.iloc[-1]

    assert last[
        "macro_payroll_growth_score"
    ] == pytest.approx(
        1.0
    )

    assert last[
        "macro_payroll_revision_score"
    ] == pytest.approx(
        1.0
    )


def test_broad_downward_revision_scores_high():
    event = pd.Series(
        [1, 1, 0, 1]
    )

    revision = pd.Series(
        [-20.0, 20.0, -20.0, 0.0]
    )

    result = broad_revision_score(
        event,
        revision,
    )

    assert result.tolist() == [
        1.0,
        0.0,
        0.5,
        0.5,
    ]


def test_macro_score_is_carried_forward():
    data = pd.DataFrame(
        {
            "market_date":
                pd.date_range(
                    "2025-01-01",
                    periods=8,
                    freq="D",
                ),

            "payroll_change_1m":
                [
                    200,
                    200,
                    150,
                    150,
                    100,
                    100,
                    100,
                    100,
                ],

            "payroll_revision_total":
                [
                    0,
                    0,
                    -5,
                    -5,
                    -20,
                    -20,
                    -20,
                    -20,
                ],

            "payroll_broad_revision_event":
                [
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                ],
        }
    )

    result = add_macro_regime_features(
        data,
        lookback=2,
        min_history=2,
    )

    # Third state starts at row 4 and should retain the same
    # score until another PAYEMS state arrives.
    scores = result.loc[
        4:7,
        "macro_regime_score",
    ]

    assert scores.notna().all()

    assert scores.nunique() == 1


def test_current_state_is_excluded_from_own_history():
    states = pd.DataFrame(
        {
            "payroll_state_market_date":
                pd.to_datetime(
                    [
                        "2025-01-01",
                        "2025-02-01",
                        "2025-03-01",
                    ]
                ),

            "payroll_state_id":
                [1, 2, 3],

            "payroll_change_1m":
                [100, 200, 1000],

            "payroll_revision_total":
                [0, 0, 0],

            "payroll_broad_revision_event":
                [0, 0, 0],
        }
    )

    result = score_payroll_states(
        states,
        lookback=2,
        min_history=2,
    )

    # 1000 is above both PRIOR values.
    # Its historical percentile must therefore be 1.0.
    assert result.loc[
        2,
        "payroll_change_percentile",
    ] == pytest.approx(
        1.0
    )


def test_macro_scores_are_bounded():
    n = 20

    states = pd.DataFrame(
        {
            "payroll_state_market_date":
                pd.date_range(
                    "2024-01-01",
                    periods=n,
                    freq="MS",
                ),

            "payroll_state_id":
                np.arange(
                    1,
                    n + 1,
                ),

            "payroll_change_1m":
                np.linspace(
                    300,
                    50,
                    n,
                ),

            "payroll_revision_total":
                np.linspace(
                    20,
                    -30,
                    n,
                ),

            "payroll_broad_revision_event":
                [
                    i % 2
                    for i in range(n)
                ],
        }
    )

    result = score_payroll_states(
        states,
        lookback=12,
        min_history=6,
    )

    components = [
        "macro_payroll_growth_score",
        "macro_payroll_revision_score",
        "macro_broad_revision_score",
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
        "macro_regime_score"
    ].dropna()

    assert final.min() >= 0.0
    assert final.max() <= 100.0