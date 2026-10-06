"""
Macro Regime V1 for the TMF Opportunity Engine.

Uses point-in-time PAYEMS-A features as slow-moving economic context.

Important:
Percentiles are calculated across genuine PAYEMS state changes, not
across repeated daily rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.signals.technical_opportunity import (
    trailing_percentile,
)


DEFAULT_MACRO_STATE_LOOKBACK = 36
DEFAULT_MACRO_MIN_HISTORY = 12


PAYEMS_FEATURES = (
    "payroll_change_1m",
    "payroll_revision_total",
    "payroll_broad_revision_event",
)


@dataclass(frozen=True)
class MacroRegimeResult:
    """Macro Regime V1 result."""

    payroll_growth_score: float
    payroll_revision_score: float
    broad_revision_score: float

    macro_regime_score: float


def normalize_revision_event(
    values: pd.Series,
) -> pd.Series:
    """
    Convert payroll_broad_revision_event to numeric 0/1 values.

    Supports numeric, boolean, and common string representations.
    """

    mapping = {
        "true": 1.0,
        "false": 0.0,
        "yes": 1.0,
        "no": 0.0,
        "1": 1.0,
        "0": 0.0,
    }

    result = values.copy()

    if pd.api.types.is_bool_dtype(
        result
    ):
        return result.astype(float)

    numeric = pd.to_numeric(
        result,
        errors="coerce",
    )

    unresolved = numeric.isna()

    if unresolved.any():
        text_values = (
            result.loc[
                unresolved
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
        )

        numeric.loc[
            unresolved
        ] = text_values

    return numeric.astype(float)


def build_payroll_state_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reduce daily PAYEMS features to genuine state changes.

    A new state begins whenever any frozen PAYEMS-A feature changes.
    """

    required = {
        "market_date",
        *PAYEMS_FEATURES,
    }

    missing = (
        required
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            "Macro regime dataset missing columns: "
            f"{sorted(missing)}"
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
        "market_date"
    ] = pd.to_datetime(
        output[
            "market_date"
        ],
        errors="raise",
    )

    output[
        "payroll_change_1m"
    ] = pd.to_numeric(
        output[
            "payroll_change_1m"
        ],
        errors="coerce",
    )

    output[
        "payroll_revision_total"
    ] = pd.to_numeric(
        output[
            "payroll_revision_total"
        ],
        errors="coerce",
    )

    output[
        "payroll_broad_revision_event"
    ] = normalize_revision_event(
        output[
            "payroll_broad_revision_event"
        ]
    )

    complete = (
        output[
            list(
                PAYEMS_FEATURES
            )
        ]
        .notna()
        .all(axis=1)
    )

    valid = (
        output.loc[
            complete,
            [
                "market_date",
                *PAYEMS_FEATURES,
            ],
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if valid.empty:
        raise ValueError(
            "No complete PAYEMS states available."
        )

    changed = (
        valid[
            list(
                PAYEMS_FEATURES
            )
        ]
        .ne(
            valid[
                list(
                    PAYEMS_FEATURES
                )
            ].shift()
        )
        .any(axis=1)
    )

    states = (
        valid.loc[
            changed
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    states[
        "payroll_state_id"
    ] = np.arange(
        1,
        len(states) + 1,
    )

    states = states.rename(
        columns={
            "market_date":
                "payroll_state_market_date",
        }
    )

    return states


def broad_revision_score(
    event: pd.Series,
    revision_total: pd.Series,
) -> pd.Series:
    """
    Directional score for broad payroll revisions.

    Broad downward revision:
        1.0

    No broad event:
        0.5

    Broad upward revision:
        0.0

    Zero revision:
        0.5
    """

    event_numeric = (
        normalize_revision_event(
            event
        )
    )

    revision_numeric = (
        pd.to_numeric(
            revision_total,
            errors="coerce",
        )
    )

    result = pd.Series(
        np.nan,
        index=event.index,
        dtype=float,
    )

    complete = (
        event_numeric.notna()
        & revision_numeric.notna()
    )

    result.loc[
        complete
    ] = 0.50

    broad = (
        complete
        & (event_numeric > 0)
    )

    result.loc[
        broad
        & (revision_numeric < 0)
    ] = 1.00

    result.loc[
        broad
        & (revision_numeric > 0)
    ] = 0.00

    return result


def score_payroll_states(
    states: pd.DataFrame,
    lookback: int = DEFAULT_MACRO_STATE_LOOKBACK,
    min_history: int = DEFAULT_MACRO_MIN_HISTORY,
) -> pd.DataFrame:
    """
    Score genuine PAYEMS states using prior macro states only.
    """

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
        states.copy()
        .sort_values(
            "payroll_state_market_date"
        )
        .reset_index(
            drop=True
        )
    )

    output[
        "payroll_change_percentile"
    ] = trailing_percentile(
        output[
            "payroll_change_1m"
        ],
        lookback=lookback,
        min_history=min_history,
    )

    output[
        "payroll_revision_percentile"
    ] = trailing_percentile(
        output[
            "payroll_revision_total"
        ],
        lookback=lookback,
        min_history=min_history,
    )

    output[
        "macro_payroll_growth_score"
    ] = (
        1.0
        - output[
            "payroll_change_percentile"
        ]
    )

    output[
        "macro_payroll_revision_score"
    ] = (
        1.0
        - output[
            "payroll_revision_percentile"
        ]
    )

    output[
        "macro_broad_revision_score"
    ] = broad_revision_score(
        output[
            "payroll_broad_revision_event"
        ],
        output[
            "payroll_revision_total"
        ],
    )

    components = [
        "macro_payroll_growth_score",
        "macro_payroll_revision_score",
        "macro_broad_revision_score",
    ]

    complete = (
        output[
            components
        ]
        .notna()
        .all(axis=1)
    )

    output[
        "macro_regime_score"
    ] = np.nan

    output.loc[
        complete,
        "macro_regime_score",
    ] = (
        output.loc[
            complete,
            components,
        ]
        .mean(axis=1)
        * 100.0
    )

    return output


def add_macro_regime_features(
    data: pd.DataFrame,
    lookback: int = DEFAULT_MACRO_STATE_LOOKBACK,
    min_history: int = DEFAULT_MACRO_MIN_HISTORY,
) -> pd.DataFrame:
    """
    Add Macro Regime V1 state and scores to daily market observations.

    Scores are calculated once per genuine PAYEMS state and then
    carried forward until the next state change.
    """

    daily = (
        data.copy()
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )

    daily[
        "market_date"
    ] = pd.to_datetime(
        daily[
            "market_date"
        ],
        errors="raise",
    )

    states = (
        build_payroll_state_table(
            daily
        )
    )

    scored_states = (
        score_payroll_states(
            states,
            lookback=lookback,
            min_history=min_history,
        )
    )

    score_columns = [
        "payroll_state_market_date",
        "payroll_state_id",
        "payroll_change_1m",
        "payroll_revision_total",
        "payroll_broad_revision_event",
        "payroll_change_percentile",
        "payroll_revision_percentile",
        "macro_payroll_growth_score",
        "macro_payroll_revision_score",
        "macro_broad_revision_score",
        "macro_regime_score",
    ]

    state_scores = (
        scored_states[
            score_columns
        ]
        .copy()
        .sort_values(
            "payroll_state_market_date"
        )
    )

    # Remove raw PAYEMS fields from the daily frame before merging
    # them back from the state table to avoid duplicate columns.
    daily_base = daily.drop(
        columns=[
            column
            for column in PAYEMS_FEATURES
            if column in daily.columns
        ]
    )

    merged = pd.merge_asof(
        daily_base.sort_values(
            "market_date"
        ),
        state_scores,
        left_on="market_date",
        right_on="payroll_state_market_date",
        direction="backward",
        allow_exact_matches=True,
    )

    future_state = (
        merged[
            "payroll_state_market_date"
        ].notna()
        & (
            merged[
                "payroll_state_market_date"
            ]
            > merged[
                "market_date"
            ]
        )
    )

    if future_state.any():
        raise AssertionError(
            "Future PAYEMS state entered "
            "Macro Regime V1 dataset."
        )

    return merged


def score_current_row(
    row: pd.Series,
) -> MacroRegimeResult:
    """Convert a scored row into a structured result."""

    required = [
        "macro_payroll_growth_score",
        "macro_payroll_revision_score",
        "macro_broad_revision_score",
        "macro_regime_score",
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
            "Macro Regime V1 scores: "
            f"{missing}"
        )

    return MacroRegimeResult(
        payroll_growth_score=float(
            row[
                "macro_payroll_growth_score"
            ]
        ),

        payroll_revision_score=float(
            row[
                "macro_payroll_revision_score"
            ]
        ),

        broad_revision_score=float(
            row[
                "macro_broad_revision_score"
            ]
        ),

        macro_regime_score=float(
            row[
                "macro_regime_score"
            ]
        ),
    )