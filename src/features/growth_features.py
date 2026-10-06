"""
Point-in-time growth/activity features.

Frozen V3 candidates
--------------------
GROWTH-A
    retail_sales_mom_pct
    retail_sales_3m_annualized_pct

GROWTH-B
    industrial_production_mom_pct
    industrial_production_3m_annualized_pct

Methodology
-----------
- Uses genuine ALFRED vintage histories.
- Reconstructs each series exactly as it was known on each vintage date.
- Uses exact calendar-month comparisons rather than positional shift().
- Missing reference months are never substituted.
- Revisions become visible only from their actual vintage date forward.
- When merged to market data, same-day vintage states are NOT allowed.
  This conservative rule avoids assumptions about intraday publication
  timing for normal and benchmark revisions.
- 2026 may exist in generated data but remains locked downstream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.macro_vintage import (
    reconstruct_as_of,
)


RAW_VINTAGE_DIR = Path(
    "data/raw/macro_vintages"
)

FEATURE_DIR = Path(
    "data/features"
)


SERIES_CONFIG = {
    "retail_sales": {
        "input_file":
            RAW_VINTAGE_DIR
            / "retail_sales.csv",
        "output_file":
            FEATURE_DIR
            / "retail_sales_point_in_time_features.csv",
        "state_date_column":
            "retail_sales_state_date",
        "reference_date_column":
            "retail_sales_reference_date",
        "level_column":
            "retail_sales_level",
        "mom_column":
            "retail_sales_mom_pct",
        "annualized_3m_column":
            "retail_sales_3m_annualized_pct",
    },

    "industrial_production": {
        "input_file":
            RAW_VINTAGE_DIR
            / "industrial_production.csv",
        "output_file":
            FEATURE_DIR
            / "industrial_production_point_in_time_features.csv",
        "state_date_column":
            "industrial_production_state_date",
        "reference_date_column":
            "industrial_production_reference_date",
        "level_column":
            "industrial_production_level",
        "mom_column":
            "industrial_production_mom_pct",
        "annualized_3m_column":
            "industrial_production_3m_annualized_pct",
    },
}


def prepare_growth_vintages(
    data: pd.DataFrame,
    series_name: str,
) -> pd.DataFrame:
    """Validate and prepare one ALFRED growth vintage history."""

    required = {
        "series_name",
        "series_id",
        "reference_date",
        "vintage_date",
        "value_as_known",
    }

    missing = (
        required
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"{series_name} vintage data missing "
            f"columns: {sorted(missing)}"
        )

    output = data.copy()

    output = output.loc[
        output["series_name"]
        == series_name
    ].copy()

    if output.empty:
        raise ValueError(
            f"No vintage rows found for {series_name}."
        )

    output["reference_date"] = (
        pd.to_datetime(
            output["reference_date"],
            errors="raise",
        )
    )

    output["vintage_date"] = (
        pd.to_datetime(
            output["vintage_date"],
            errors="raise",
        )
    )

    output["value_as_known"] = (
        pd.to_numeric(
            output["value_as_known"],
            errors="coerce",
        )
    )

    duplicate_mask = (
        output.duplicated(
            subset=[
                "reference_date",
                "vintage_date",
            ]
        )
    )

    if duplicate_mask.any():
        raise ValueError(
            f"Duplicate vintage observations "
            f"found for {series_name}."
        )

    return (
        output.sort_values(
            [
                "vintage_date",
                "reference_date",
            ]
        )
        .reset_index(drop=True)
    )


def exact_reference_value(
    state: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> float:
    """
    Retrieve one exact economic reference month.

    No nearest-month substitution is permitted.
    """

    reference_date = pd.Timestamp(
        reference_date
    )

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


def calculate_mom_pct(
    current_value: float,
    previous_month_value: float,
) -> float:
    """Calculate exact-month month-over-month percent change."""

    if (
        pd.isna(current_value)
        or pd.isna(previous_month_value)
        or previous_month_value == 0
    ):
        return np.nan

    return (
        (
            current_value
            / previous_month_value
        )
        - 1.0
    ) * 100.0


def calculate_3m_annualized_pct(
    current_value: float,
    three_month_ago_value: float,
) -> float:
    """Calculate exact 3-month annualized growth rate."""

    if (
        pd.isna(current_value)
        or pd.isna(three_month_ago_value)
        or three_month_ago_value <= 0
        or current_value <= 0
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


def growth_features_as_of(
    vintage_data: pd.DataFrame,
    as_of_date: pd.Timestamp,
    series_name: str,
) -> dict:
    """Reconstruct frozen growth features as known on one date."""

    as_of_date = pd.Timestamp(
        as_of_date
    )

    state = reconstruct_as_of(
        vintage_data,
        as_of_date,
    )

    if state.empty:
        return {
            "state_date":
                as_of_date,
            "reference_date":
                pd.NaT,
            "level":
                np.nan,
            "mom_pct":
                np.nan,
            "annualized_3m_pct":
                np.nan,
        }

    valid = state.loc[
        state["value_as_known"].notna()
    ].copy()

    if valid.empty:
        return {
            "state_date":
                as_of_date,
            "reference_date":
                pd.NaT,
            "level":
                np.nan,
            "mom_pct":
                np.nan,
            "annualized_3m_pct":
                np.nan,
        }

    latest_reference_date = (
        valid["reference_date"].max()
    )

    current_value = (
        exact_reference_value(
            state,
            latest_reference_date,
        )
    )

    previous_month = (
        latest_reference_date
        - pd.DateOffset(months=1)
    )

    three_months_ago = (
        latest_reference_date
        - pd.DateOffset(months=3)
    )

    previous_value = (
        exact_reference_value(
            state,
            previous_month,
        )
    )

    three_month_value = (
        exact_reference_value(
            state,
            three_months_ago,
        )
    )

    return {
        "state_date":
            as_of_date,
        "reference_date":
            latest_reference_date,
        "level":
            current_value,
        "mom_pct":
            calculate_mom_pct(
                current_value,
                previous_value,
            ),
        "annualized_3m_pct":
            calculate_3m_annualized_pct(
                current_value,
                three_month_value,
            ),
    }


def build_growth_feature_history(
    vintage_data: pd.DataFrame,
    series_name: str,
) -> pd.DataFrame:
    """
    Build point-in-time feature states for every genuine
    ALFRED vintage date.
    """

    config = SERIES_CONFIG[
        series_name
    ]

    data = prepare_growth_vintages(
        vintage_data,
        series_name,
    )

    state_dates = (
        data["vintage_date"]
        .drop_duplicates()
        .sort_values()
    )

    rows = []

    for state_date in state_dates:

        row = growth_features_as_of(
            vintage_data=data,
            as_of_date=state_date,
            series_name=series_name,
        )

        rows.append(
            {
                config[
                    "state_date_column"
                ]:
                    row["state_date"],

                config[
                    "reference_date_column"
                ]:
                    row["reference_date"],

                config[
                    "level_column"
                ]:
                    row["level"],

                config[
                    "mom_column"
                ]:
                    row["mom_pct"],

                config[
                    "annualized_3m_column"
                ]:
                    row[
                        "annualized_3m_pct"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


def merge_growth_features_to_market(
    market_data: pd.DataFrame,
    growth_features: pd.DataFrame,
    series_name: str,
) -> pd.DataFrame:
    """
    Merge one growth state history backward to market observations.

    Exact same-day matches are deliberately prohibited because we do
    not have verified intraday release times for every vintage event.
    """

    config = SERIES_CONFIG[
        series_name
    ]

    state_column = config[
        "state_date_column"
    ]

    market = market_data.copy()

    market["market_date"] = (
        pd.to_datetime(
            market["market_date"],
            errors="raise",
        )
    )

    features = (
        growth_features.copy()
    )

    features[state_column] = (
        pd.to_datetime(
            features[state_column],
            errors="raise",
        )
    )

    market = market.sort_values(
        "market_date"
    )

    features = features.sort_values(
        state_column
    )

    merged = pd.merge_asof(
        market,
        features,
        left_on="market_date",
        right_on=state_column,
        direction="backward",
        allow_exact_matches=False,
    )

    future_mask = (
        merged[state_column].notna()
        & (
            merged[state_column]
            >= merged["market_date"]
        )
    )

    if future_mask.any():
        raise AssertionError(
            f"Non-prior {series_name} "
            "state entered market data."
        )

    return merged


def build_and_save_series(
    series_name: str,
) -> pd.DataFrame:
    """Build and save one frozen growth feature history."""

    config = SERIES_CONFIG[
        series_name
    ]

    raw = pd.read_csv(
        config["input_file"],
        parse_dates=[
            "reference_date",
            "vintage_date",
        ],
    )

    features = (
        build_growth_feature_history(
            vintage_data=raw,
            series_name=series_name,
        )
    )

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        config["output_file"],
        index=False,
    )

    return features


def main() -> None:
    """Build both frozen growth candidate histories."""

    print("=" * 72)
    print("POINT-IN-TIME GROWTH FEATURES")
    print("=" * 72)

    for series_name in [
        "retail_sales",
        "industrial_production",
    ]:

        features = (
            build_and_save_series(
                series_name
            )
        )

        config = SERIES_CONFIG[
            series_name
        ]

        state_column = config[
            "state_date_column"
        ]

        feature_columns = [
            config[
                "mom_column"
            ],
            config[
                "annualized_3m_column"
            ],
        ]

        print(
            f"\n{series_name}"
        )

        print(
            "Saved:",
            config["output_file"],
        )

        print(
            "Rows:",
            len(features),
        )

        print(
            "State range:",
            features[
                state_column
            ].min().date(),
            "to",
            features[
                state_column
            ].max().date(),
        )

        print(
            "Missing feature values:"
        )

        print(
            features[
                feature_columns
            ].isna().sum()
        )

        print(
            "\nLatest states:"
        )

        print(
            features.tail(
                10
            ).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
    