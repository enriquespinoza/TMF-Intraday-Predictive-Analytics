"""
Point-in-time realized inflation features.

This module constructs the frozen INFLATION-B feature set:

    core_cpi_yoy_pct
    core_cpi_3m_annualized_pct

Core CPI values are reconstructed from ALFRED vintages as they were
known at each point in time.

Important methodology
---------------------
- No present-day revised CPI history is used retrospectively.
- Reference-period comparisons use exact calendar months.
- Missing reference periods are never interpolated or forward-filled.
- October 2025 remains missing because CPI for that month was not
  published.
- Annual seasonal-factor revisions are incorporated beginning on their
  actual ALFRED vintage date.
- 2026 may exist in raw/generated data but remains locked downstream
  for model selection and evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.macro_vintage import (
    reconstruct_as_of,
)


CORE_CPI_SERIES_NAME = "core_cpi"

RAW_VINTAGE_FILE = Path(
    "data/raw/macro_vintages/core_cpi.csv"
)

FEATURE_OUTPUT_FILE = Path(
    "data/features/core_cpi_point_in_time_features.csv"
)


def prepare_core_cpi_vintages(
    vintage_data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize Core CPI vintage observations."""

    required_columns = {
        "series_name",
        "series_id",
        "reference_date",
        "vintage_date",
        "value_as_known",
    }

    missing = (
        required_columns
        - set(vintage_data.columns)
    )

    if missing:
        raise ValueError(
            "Core CPI vintage data missing "
            f"required columns: {sorted(missing)}"
        )

    data = vintage_data.loc[
        vintage_data["series_name"]
        == CORE_CPI_SERIES_NAME
    ].copy()

    if data.empty:
        raise ValueError(
            "No core_cpi observations found."
        )

    data["reference_date"] = pd.to_datetime(
        data["reference_date"],
        errors="raise",
    ).dt.normalize()

    data["vintage_date"] = pd.to_datetime(
        data["vintage_date"],
        errors="raise",
    ).dt.normalize()

    data["value_as_known"] = pd.to_numeric(
        data["value_as_known"],
        errors="coerce",
    )

    return (
        data.sort_values(
            [
                "reference_date",
                "vintage_date",
            ]
        )
        .reset_index(drop=True)
    )


def exact_reference_value(
    state: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> float:
    """
    Return the value for one exact reference month.

    Missing months remain missing. The function never substitutes
    the nearest available month.
    """

    reference_date = pd.Timestamp(
        reference_date
    ).normalize()

    match = state.loc[
        state["reference_date"]
        == reference_date,
        "value_as_known",
    ]

    if match.empty:
        return np.nan

    value = match.iloc[-1]

    if pd.isna(value):
        return np.nan

    return float(value)


def calculate_yoy_pct(
    current_value: float,
    year_ago_value: float,
) -> float:
    """Calculate 12-month percentage inflation."""

    if (
        pd.isna(current_value)
        or pd.isna(year_ago_value)
        or current_value <= 0
        or year_ago_value <= 0
    ):
        return np.nan

    return (
        (
            current_value
            / year_ago_value
        )
        - 1.0
    ) * 100.0


def calculate_3m_annualized_pct(
    current_value: float,
    three_month_ago_value: float,
) -> float:
    """Calculate annualized three-month inflation."""

    if (
        pd.isna(current_value)
        or pd.isna(three_month_ago_value)
        or current_value <= 0
        or three_month_ago_value <= 0
    ):
        return np.nan

    return (
        (
            (
                current_value
                / three_month_ago_value
            )
            ** 4
        )
        - 1.0
    ) * 100.0


def core_cpi_features_as_of(
    vintage_data: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
) -> dict:
    """
    Reconstruct Core CPI state as known on one date and calculate
    the two frozen realized-inflation features.
    """

    data = prepare_core_cpi_vintages(
        vintage_data
    )

    as_of_date = pd.Timestamp(
        as_of_date
    ).normalize()

    state = reconstruct_as_of(
        data,
        as_of_date,
    )

    if state.empty:
        return {
            "inflation_state_date":
                as_of_date,
            "core_cpi_reference_date":
                pd.NaT,
            "core_cpi_yoy_pct":
                np.nan,
            "core_cpi_3m_annualized_pct":
                np.nan,
        }

    state = state.copy()

    state["reference_date"] = (
        pd.to_datetime(
            state["reference_date"],
            errors="raise",
        ).dt.normalize()
    )

    state["value_as_known"] = (
        pd.to_numeric(
            state["value_as_known"],
            errors="coerce",
        )
    )

    # Only genuinely published values are eligible
    # to define the current CPI state.
    valid_state = state.dropna(
        subset=["value_as_known"]
    )

    if valid_state.empty:
        return {
            "inflation_state_date":
                as_of_date,
            "core_cpi_reference_date":
                pd.NaT,
            "core_cpi_yoy_pct":
                np.nan,
            "core_cpi_3m_annualized_pct":
                np.nan,
        }

    latest_reference_date = (
        valid_state[
            "reference_date"
        ].max()
    )

    current_value = exact_reference_value(
        state,
        latest_reference_date,
    )

    year_ago_reference = (
        latest_reference_date
        - pd.DateOffset(months=12)
    )

    three_month_reference = (
        latest_reference_date
        - pd.DateOffset(months=3)
    )

    year_ago_value = exact_reference_value(
        state,
        year_ago_reference,
    )

    three_month_ago_value = (
        exact_reference_value(
            state,
            three_month_reference,
        )
    )

    return {
        "inflation_state_date":
            as_of_date,
        "core_cpi_reference_date":
            latest_reference_date,
        "core_cpi_yoy_pct":
            calculate_yoy_pct(
                current_value,
                year_ago_value,
            ),
        "core_cpi_3m_annualized_pct":
            calculate_3m_annualized_pct(
                current_value,
                three_month_ago_value,
            ),
    }


def build_core_cpi_feature_history(
    vintage_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build point-in-time Core CPI features for every vintage date.

    This captures both:
    - normal monthly CPI releases
    - later seasonal-factor revisions
    """

    data = prepare_core_cpi_vintages(
        vintage_data
    )

    state_dates = (
        data["vintage_date"]
        .drop_duplicates()
        .sort_values()
    )

    rows = []

    for state_date in state_dates:

        rows.append(
            core_cpi_features_as_of(
                vintage_data=data,
                as_of_date=state_date,
            )
        )

    history = pd.DataFrame(
        rows
    )

    return (
        history.sort_values(
            "inflation_state_date"
        )
        .reset_index(drop=True)
    )


def merge_core_cpi_features_to_market(
    market_data: pd.DataFrame,
    inflation_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Backward-merge point-in-time Core CPI features onto market dates.

    A market observation may only see the most recent inflation state
    that existed on or before that trading date.
    """

    market = market_data.copy()
    features = inflation_features.copy()

    if "market_date" not in market.columns:
        raise ValueError(
            "Market data must contain "
            "'market_date'."
        )

    market["market_date"] = pd.to_datetime(
        market["market_date"],
        errors="raise",
    ).dt.normalize()

    features[
        "inflation_state_date"
    ] = pd.to_datetime(
        features[
            "inflation_state_date"
        ],
        errors="raise",
    ).dt.normalize()

    market = market.sort_values(
        "market_date"
    )

    features = features.sort_values(
        "inflation_state_date"
    )

    merged = pd.merge_asof(
        market,
        features,
        left_on="market_date",
        right_on="inflation_state_date",
        direction="backward",
        allow_exact_matches=True,
    )

    violation = (
        merged[
            "inflation_state_date"
        ].notna()
        & (
            merged[
                "inflation_state_date"
            ]
            > merged["market_date"]
        )
    )

    if violation.any():
        raise AssertionError(
            "Future Core CPI state entered "
            "the market dataset."
        )

    return merged


def main() -> None:
    """Generate point-in-time Core CPI feature history."""

    vintage_data = pd.read_csv(
        RAW_VINTAGE_FILE,
        parse_dates=[
            "reference_date",
            "vintage_date",
        ],
    )

    features = (
        build_core_cpi_feature_history(
            vintage_data
        )
    )

    FEATURE_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        FEATURE_OUTPUT_FILE,
        index=False,
    )

    print(
        "Saved:",
        FEATURE_OUTPUT_FILE,
    )

    print(
        "Rows:",
        len(features),
    )

    print(
        "State range:",
        features[
            "inflation_state_date"
        ].min().date(),
        "to",
        features[
            "inflation_state_date"
        ].max().date(),
    )

    print(
        "\nLatest states:"
    )

    print(
        features.tail(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
    