"""
Treasury Confirmation V1.

Measures whether long-duration and real Treasury yields are moving
in a direction supportive of TMF.

All percentile calculations use prior observations only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.signals.technical_opportunity import (
    DEFAULT_LOOKBACK,
    DEFAULT_MIN_HISTORY,
    trailing_percentile,
)


REQUIRED_COLUMNS = (
    "market_date",
    "change_30y_1d_bp",
    "change_30y_5d_bp",
    "real_yield_10y_change_5d_bp",
)


@dataclass(frozen=True)
class TreasuryConfirmationResult:
    """Treasury Confirmation V1 result."""

    yield_30y_1d_score: float
    yield_30y_5d_score: float
    real_yield_10y_5d_score: float

    treasury_confirmation_score: float


def add_treasury_confirmation_features(
    data: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    min_history: int = DEFAULT_MIN_HISTORY,
) -> pd.DataFrame:
    """
    Add Treasury Confirmation V1 features.

    Lower yield changes receive higher confirmation scores.
    """

    missing = (
        set(REQUIRED_COLUMNS)
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            "Treasury confirmation dataset "
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
        .sort_values("market_date")
        .reset_index(drop=True)
    )

    output[
        "treasury_30y_1d_percentile"
    ] = trailing_percentile(
        output["change_30y_1d_bp"],
        lookback=lookback,
        min_history=min_history,
    )

    output[
        "treasury_30y_5d_percentile"
    ] = trailing_percentile(
        output["change_30y_5d_bp"],
        lookback=lookback,
        min_history=min_history,
    )

    output[
        "treasury_real_10y_5d_percentile"
    ] = trailing_percentile(
        output[
            "real_yield_10y_change_5d_bp"
        ],
        lookback=lookback,
        min_history=min_history,
    )

    # Falling yields are supportive of TMF.
    output[
        "treasury_30y_1d_score"
    ] = (
        1.0
        - output[
            "treasury_30y_1d_percentile"
        ]
    )

    output[
        "treasury_30y_5d_score"
    ] = (
        1.0
        - output[
            "treasury_30y_5d_percentile"
        ]
    )

    output[
        "treasury_real_10y_5d_score"
    ] = (
        1.0
        - output[
            "treasury_real_10y_5d_percentile"
        ]
    )

    components = [
        "treasury_30y_1d_score",
        "treasury_30y_5d_score",
        "treasury_real_10y_5d_score",
    ]

    complete = (
        output[
            components
        ]
        .notna()
        .all(axis=1)
    )

    output[
        "treasury_confirmation_score"
    ] = np.nan

    output.loc[
        complete,
        "treasury_confirmation_score",
    ] = (
        output.loc[
            complete,
            components,
        ]
        .mean(axis=1)
        * 100.0
    )

    return output


def score_current_row(
    row: pd.Series,
) -> TreasuryConfirmationResult:
    """Convert an already-scored row into a result object."""

    required = [
        "treasury_30y_1d_score",
        "treasury_30y_5d_score",
        "treasury_real_10y_5d_score",
        "treasury_confirmation_score",
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
            "Incomplete Treasury confirmation scores: "
            f"{missing}"
        )

    return TreasuryConfirmationResult(
        yield_30y_1d_score=float(
            row[
                "treasury_30y_1d_score"
            ]
        ),

        yield_30y_5d_score=float(
            row[
                "treasury_30y_5d_score"
            ]
        ),

        real_yield_10y_5d_score=float(
            row[
                "treasury_real_10y_5d_score"
            ]
        ),

        treasury_confirmation_score=float(
            row[
                "treasury_confirmation_score"
            ]
        ),
    )