"""
Market Confirmation V1.

Measures whether TMF price action is beginning to respond positively
to an otherwise favorable opportunity environment.

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
    "return_1d",
    "return_3d",
    "close_vs_ema9",
    "ema_spread_pct",
    "relative_volume",
)


@dataclass(frozen=True)
class MarketConfirmationResult:
    """Market Confirmation V1 result."""

    price_response_score: float
    ema9_confirmation_score: float
    ema_structure_score: float
    volume_participation_score: float

    market_confirmation_score: float


def calculate_signed_relative_volume(
    return_1d: pd.Series,
    relative_volume: pd.Series,
) -> pd.Series:
    """
    Combine current-day direction with relative volume.

    Positive return:
        positive relative volume

    Negative return:
        negative relative volume

    Zero return:
        zero
    """

    daily_return = pd.to_numeric(
        return_1d,
        errors="coerce",
    )

    volume = pd.to_numeric(
        relative_volume,
        errors="coerce",
    )

    return (
        np.sign(daily_return)
        * volume
    )


def add_market_confirmation_features(
    data: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    min_history: int = DEFAULT_MIN_HISTORY,
) -> pd.DataFrame:
    """
    Add Market Confirmation V1 features.

    All percentile reference samples contain prior observations only.
    """

    missing = (
        set(REQUIRED_COLUMNS)
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            "Market confirmation dataset "
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

    output[
        "signed_relative_volume"
    ] = calculate_signed_relative_volume(
        output[
            "return_1d"
        ],
        output[
            "relative_volume"
        ],
    )

    output[
        "market_return_3d_percentile"
    ] = trailing_percentile(
        output[
            "return_3d"
        ],
        lookback=lookback,
        min_history=min_history,
    )

    output[
        "market_close_vs_ema9_percentile"
    ] = trailing_percentile(
        output[
            "close_vs_ema9"
        ],
        lookback=lookback,
        min_history=min_history,
    )

    output[
        "market_ema_spread_percentile"
    ] = trailing_percentile(
        output[
            "ema_spread_pct"
        ],
        lookback=lookback,
        min_history=min_history,
    )

    output[
        "market_signed_volume_percentile"
    ] = trailing_percentile(
        output[
            "signed_relative_volume"
        ],
        lookback=lookback,
        min_history=min_history,
    )

    # Higher percentile = stronger confirmation.
    output[
        "market_price_response_score"
    ] = output[
        "market_return_3d_percentile"
    ]

    output[
        "market_ema9_confirmation_score"
    ] = output[
        "market_close_vs_ema9_percentile"
    ]

    output[
        "market_ema_structure_score"
    ] = output[
        "market_ema_spread_percentile"
    ]

    output[
        "market_volume_participation_score"
    ] = output[
        "market_signed_volume_percentile"
    ]

    component_columns = [
        "market_price_response_score",
        "market_ema9_confirmation_score",
        "market_ema_structure_score",
        "market_volume_participation_score",
    ]

    complete = (
        output[
            component_columns
        ]
        .notna()
        .all(axis=1)
    )

    output[
        "market_confirmation_score"
    ] = np.nan

    output.loc[
        complete,
        "market_confirmation_score",
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
) -> MarketConfirmationResult:
    """
    Convert an already-scored row into a structured result.
    """

    required = [
        "market_price_response_score",
        "market_ema9_confirmation_score",
        "market_ema_structure_score",
        "market_volume_participation_score",
        "market_confirmation_score",
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
            "Market Confirmation V1 scores: "
            f"{missing}"
        )

    return MarketConfirmationResult(
        price_response_score=float(
            row[
                "market_price_response_score"
            ]
        ),

        ema9_confirmation_score=float(
            row[
                "market_ema9_confirmation_score"
            ]
        ),

        ema_structure_score=float(
            row[
                "market_ema_structure_score"
            ]
        ),

        volume_participation_score=float(
            row[
                "market_volume_participation_score"
            ]
        ),

        market_confirmation_score=float(
            row[
                "market_confirmation_score"
            ]
        ),
    )