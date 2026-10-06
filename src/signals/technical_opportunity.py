"""
Technical Opportunity V1 for the TMF trading engine.

The model is deterministic and target-independent.

It uses point-in-time trailing percentiles rather than fixed
predictive thresholds.

The current observation is excluded from its own historical
reference window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_LOOKBACK = 252
DEFAULT_MIN_HISTORY = 63


REQUIRED_COLUMNS = (
    "market_date",
    "return_1d",
    "return_5d",
    "rsi_14",
    "distance_from_20d_high",
    "below_ema21_streak",
)


@dataclass(frozen=True)
class TechnicalOpportunityResult:
    """Technical Opportunity V1 result for one observation."""

    pullback_5d_score: float
    rsi_score: float
    distance_high_score: float
    stabilization_score: float
    ema_pullback_score: float

    technical_opportunity_score: float


def trailing_percentile(
    values: pd.Series,
    lookback: int = DEFAULT_LOOKBACK,
    min_history: int = DEFAULT_MIN_HISTORY,
) -> pd.Series:
    """
    Calculate percentile of current value versus PRIOR observations.

    The current row is excluded from its reference sample.

    Result:
        0.0 -> unusually low versus prior observations
        1.0 -> unusually high versus prior observations
    """

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    result = pd.Series(
        np.nan,
        index=numeric.index,
        dtype=float,
    )

    array = numeric.to_numpy(
        dtype=float
    )

    for i in range(
        len(array)
    ):
        current = array[i]

        if np.isnan(current):
            continue

        start = max(
            0,
            i - lookback,
        )

        history = array[
            start:i
        ]

        history = history[
            ~np.isnan(history)
        ]

        if (
            len(history)
            < min_history
        ):
            continue

        # Mid-rank percentile handles ties more smoothly.
        less = np.sum(
            history < current
        )

        equal = np.sum(
            history == current
        )

        percentile = (
            less
            + 0.5 * equal
        ) / len(history)

        result.iloc[i] = float(
            percentile
        )

    return result


def ema_pullback_score(
    streak: pd.Series,
) -> pd.Series:
    """
    Score EMA21 pullback context.

    0 sessions below:
        0.50

    1-3 sessions below:
        1.00

    4+ sessions below:
        0.00

    Risk Governor V3 separately controls entry permission.
    """

    streak = pd.to_numeric(
        streak,
        errors="coerce",
    )

    result = pd.Series(
        np.nan,
        index=streak.index,
        dtype=float,
    )

    valid = streak.notna()

    result.loc[
        valid
        & (streak == 0)
    ] = 0.50

    result.loc[
        valid
        & streak.between(
            1,
            3,
        )
    ] = 1.00

    result.loc[
        valid
        & (streak >= 4)
    ] = 0.00

    return result


def add_technical_opportunity_features(
    data: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    min_history: int = DEFAULT_MIN_HISTORY,
) -> pd.DataFrame:
    """
    Calculate Technical Opportunity V1 scores.

    All percentile-based features are strictly backward looking.
    """

    missing = (
        set(REQUIRED_COLUMNS)
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            "Technical opportunity dataset "
            "missing columns: "
            f"{sorted(missing)}"
        )

    if lookback < 1:
        raise ValueError(
            "lookback must be at least 1."
        )

    if min_history < 1:
        raise ValueError(
            "min_history must be at least 1."
        )

    if min_history > lookback:
        raise ValueError(
            "min_history cannot exceed lookback."
        )

    output = (
        data.copy()
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )

    # Historical percentiles.
    output[
        "return_5d_percentile"
    ] = trailing_percentile(
        output[
            "return_5d"
        ],
        lookback=lookback,
        min_history=min_history,
    )

    output[
        "rsi_14_percentile"
    ] = trailing_percentile(
        output[
            "rsi_14"
        ],
        lookback=lookback,
        min_history=min_history,
    )

    output[
        "distance_20d_high_percentile"
    ] = trailing_percentile(
        output[
            "distance_from_20d_high"
        ],
        lookback=lookback,
        min_history=min_history,
    )

    output[
        "return_1d_percentile"
    ] = trailing_percentile(
        output[
            "return_1d"
        ],
        lookback=lookback,
        min_history=min_history,
    )

    # Lower percentile = deeper pullback.
    output[
        "tech_pullback_5d_score"
    ] = (
        1.0
        - output[
            "return_5d_percentile"
        ]
    )

    output[
        "tech_rsi_score"
    ] = (
        1.0
        - output[
            "rsi_14_percentile"
        ]
    )

    output[
        "tech_distance_high_score"
    ] = (
        1.0
        - output[
            "distance_20d_high_percentile"
        ]
    )

    # Stronger current-day performance = more stabilization.
    output[
        "tech_stabilization_score"
    ] = output[
        "return_1d_percentile"
    ]

    output[
        "tech_ema_pullback_score"
    ] = ema_pullback_score(
        output[
            "below_ema21_streak"
        ]
    )

    component_columns = [
        "tech_pullback_5d_score",
        "tech_rsi_score",
        "tech_distance_high_score",
        "tech_stabilization_score",
        "tech_ema_pullback_score",
    ]

    complete = (
        output[
            component_columns
        ]
        .notna()
        .all(axis=1)
    )

    output[
        "technical_opportunity_score"
    ] = np.nan

    output.loc[
        complete,
        "technical_opportunity_score",
    ] = (
        output.loc[
            complete,
            component_columns,
        ]
        .mean(axis=1)
        * 100.0
    )

    return output


def score_current_row(
    row: pd.Series,
) -> TechnicalOpportunityResult:
    """
    Convert an already-scored dataframe row to a structured result.
    """

    required = [
        "tech_pullback_5d_score",
        "tech_rsi_score",
        "tech_distance_high_score",
        "tech_stabilization_score",
        "tech_ema_pullback_score",
        "technical_opportunity_score",
    ]

    missing = [
        column
        for column in required
        if pd.isna(
            row.get(
                column,
                np.nan,
            )
        )
    ]

    if missing:
        raise ValueError(
            "Current row does not have complete "
            "technical opportunity scores: "
            f"{missing}"
        )

    return TechnicalOpportunityResult(
        pullback_5d_score=float(
            row[
                "tech_pullback_5d_score"
            ]
        ),

        rsi_score=float(
            row[
                "tech_rsi_score"
            ]
        ),

        distance_high_score=float(
            row[
                "tech_distance_high_score"
            ]
        ),

        stabilization_score=float(
            row[
                "tech_stabilization_score"
            ]
        ),

        ema_pullback_score=float(
            row[
                "tech_ema_pullback_score"
            ]
        ),

        technical_opportunity_score=float(
            row[
                "technical_opportunity_score"
            ]
        ),
    )