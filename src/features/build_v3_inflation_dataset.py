"""
Build the V3 point-in-time dataset for frozen inflation evaluation.

Starting dataset:
    Technical features + PAYEMS features

Adds:
    INFLATION-A
        breakeven_10y
        breakeven_10y_change_5d_bp

    INFLATION-B
        core_cpi_yoy_pct
        core_cpi_3m_annualized_pct

Methodology
-----------
- All merges are backward-looking.
- Exact same-day matches are allowed.
- No future macro state may enter a market row.
- Missing CPI reference periods are preserved.
- 2026 may remain in the generated dataset, but remains locked
  downstream for feature selection and model validation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.inflation_features import (
    merge_core_cpi_features_to_market,
)

from src.features.inflation_market_features import (
    merge_breakeven_features_to_market,
)


BASE_DATA_FILE = Path(
    "data/features/"
    "TMF_1d_technical_PAYEMS_features.csv"
)

CORE_CPI_FEATURE_FILE = Path(
    "data/features/"
    "core_cpi_point_in_time_features.csv"
)

BREAKEVEN_FEATURE_FILE = Path(
    "data/features/"
    "breakeven_10y_features.csv"
)

OUTPUT_FILE = Path(
    "data/features/"
    "TMF_1d_technical_PAYEMS_inflation_features.csv"
)


def load_base_data() -> pd.DataFrame:
    """Load Technical + PAYEMS feature dataset."""

    data = pd.read_csv(
        BASE_DATA_FILE,
        parse_dates=["market_date"],
    )

    if data.empty:
        raise ValueError(
            "Base Technical + PAYEMS dataset is empty."
        )

    if data["market_date"].duplicated().any():
        raise ValueError(
            "Duplicate market dates found "
            "in base dataset."
        )

    return data.sort_values(
        "market_date"
    ).reset_index(drop=True)


def load_core_cpi_features() -> pd.DataFrame:
    """Load point-in-time Core CPI features."""

    return pd.read_csv(
        CORE_CPI_FEATURE_FILE,
        parse_dates=[
            "inflation_state_date",
            "core_cpi_reference_date",
        ],
    )


def load_breakeven_features() -> pd.DataFrame:
    """Load market-implied inflation features."""

    return pd.read_csv(
        BREAKEVEN_FEATURE_FILE,
        parse_dates=[
            "breakeven_state_date",
        ],
    )


def validate_combined_data(
    data: pd.DataFrame,
) -> None:
    """Validate point-in-time integrity of combined dataset."""

    required_columns = {
        "market_date",
        "payroll_change_1m",
        "payroll_revision_total",
        "payroll_revision_2m",
        "payroll_broad_revision_event",
        "breakeven_state_date",
        "breakeven_10y",
        "breakeven_10y_change_5d_bp",
        "inflation_state_date",
        "core_cpi_reference_date",
        "core_cpi_yoy_pct",
        "core_cpi_3m_annualized_pct",
    }

    missing = (
        required_columns
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            "Combined V3 dataset missing "
            f"columns: {sorted(missing)}"
        )

    if data["market_date"].duplicated().any():
        raise ValueError(
            "Combined V3 dataset contains "
            "duplicate market dates."
        )

    breakeven_future = (
        data["breakeven_state_date"].notna()
        & (
            data["breakeven_state_date"]
            > data["market_date"]
        )
    )

    if breakeven_future.any():
        raise AssertionError(
            "Future breakeven state detected."
        )

    cpi_future = (
        data["inflation_state_date"].notna()
        & (
            data["inflation_state_date"]
            > data["market_date"]
        )
    )

    if cpi_future.any():
        raise AssertionError(
            "Future Core CPI state detected."
        )

    # The economic reference month must not be
    # later than the market date.
    cpi_reference_future = (
        data[
            "core_cpi_reference_date"
        ].notna()
        & (
            data[
                "core_cpi_reference_date"
            ]
            > data["market_date"]
        )
    )

    if cpi_reference_future.any():
        raise AssertionError(
            "Future CPI reference period detected."
        )


def build_v3_inflation_dataset(
    base_data: pd.DataFrame,
    core_cpi_features: pd.DataFrame,
    breakeven_features: pd.DataFrame,
) -> pd.DataFrame:
    """Combine frozen inflation features with V3 base data."""

    combined = (
        merge_breakeven_features_to_market(
            market_data=base_data,
            breakeven_features=(
                breakeven_features
            ),
        )
    )

    combined = (
        merge_core_cpi_features_to_market(
            market_data=combined,
            inflation_features=(
                core_cpi_features
            ),
        )
    )

    validate_combined_data(
        combined
    )

    return (
        combined.sort_values(
            "market_date"
        )
        .reset_index(drop=True)
    )


def main() -> None:
    """Build and save combined V3 inflation dataset."""

    base = load_base_data()

    core_cpi = (
        load_core_cpi_features()
    )

    breakeven = (
        load_breakeven_features()
    )

    combined = (
        build_v3_inflation_dataset(
            base_data=base,
            core_cpi_features=core_cpi,
            breakeven_features=breakeven,
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    inflation_columns = [
        "breakeven_10y",
        "breakeven_10y_change_5d_bp",
        "core_cpi_yoy_pct",
        "core_cpi_3m_annualized_pct",
    ]

    print(
        "Saved:",
        OUTPUT_FILE,
    )

    print(
        "Shape:",
        combined.shape,
    )

    print(
        "Market range:",
        combined[
            "market_date"
        ].min().date(),
        "to",
        combined[
            "market_date"
        ].max().date(),
    )

    print(
        "\nMissing inflation values:"
    )

    print(
        combined[
            inflation_columns
        ].isna().sum()
    )

    print(
        "\nFuture-state violations:"
    )

    print(
        "Breakeven:",
        (
            combined[
                "breakeven_state_date"
            ]
            > combined[
                "market_date"
            ]
        ).sum(),
    )

    print(
        "Core CPI:",
        (
            combined[
                "inflation_state_date"
            ]
            > combined[
                "market_date"
            ]
        ).sum(),
    )


if __name__ == "__main__":
    main()