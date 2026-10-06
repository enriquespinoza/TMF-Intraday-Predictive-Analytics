"""
Market features required by V4 Risk Governor V1.

This module creates the missing 10-year real-yield risk input and
merges it onto the existing TMF technical + Treasury dataset.

Risk Governor V1 requires:

- TMF one-day return
- 30Y Treasury one-day change
- 30Y Treasury five-day change
- 10Y real-yield five-day change
- 5D / 20D realized-volatility ratio
- TMF position relative to EMA21

The 10-year real-yield five-day change uses five valid published
observations. Holidays and missing FRED observations do not count as
observations.

This is an end-of-day research dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TREASURY_FEATURE_FILE = Path(
    "data/features/"
    "TMF_1d_technical_treasury_features.csv"
)

REAL_YIELD_FILE = Path(
    "data/raw/macro/"
    "real_yield_10y.csv"
)

OUTPUT_FILE = Path(
    "data/features/"
    "TMF_1d_risk_governor_features.csv"
)


def prepare_real_yield_10y(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare DFII10 and calculate a five-valid-observation change.
    """

    required = {
        "reference_date",
        "value",
    }

    missing = (
        required
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            "10-year real-yield data missing "
            f"columns: {sorted(missing)}"
        )

    output = data.copy()

    output["reference_date"] = (
        pd.to_datetime(
            output["reference_date"],
            errors="raise",
        )
    )

    output["value"] = (
        pd.to_numeric(
            output["value"],
            errors="coerce",
        )
    )

    if "series_name" in output.columns:
        output = output.loc[
            output["series_name"]
            == "real_yield_10y"
        ].copy()

    if output.empty:
        raise ValueError(
            "No real_yield_10y observations found."
        )

    if output[
        "reference_date"
    ].duplicated().any():
        raise ValueError(
            "Duplicate real-yield dates detected."
        )

    # A holiday/missing observation is not one of the
    # five published observations.
    output = (
        output.dropna(
            subset=["value"]
        )
        .sort_values(
            "reference_date"
        )
        .reset_index(drop=True)
    )

    output = output.rename(
        columns={
            "reference_date":
                "real_yield_10y_state_date",

            "value":
                "real_yield_10y",
        }
    )

    output[
        "real_yield_10y_change_5d_bp"
    ] = (
        (
            output[
                "real_yield_10y"
            ]
            - output[
                "real_yield_10y"
            ].shift(5)
        )
        * 100.0
    )

    return output[
        [
            "real_yield_10y_state_date",
            "real_yield_10y",
            "real_yield_10y_change_5d_bp",
        ]
    ]


def merge_real_yield_to_market(
    market_data: pd.DataFrame,
    real_yield_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge latest known real-yield observation to each market day.

    Exact same-day matches are allowed because this research layer
    represents an end-of-day risk decision.
    """

    market = market_data.copy()

    market["market_date"] = (
        pd.to_datetime(
            market["market_date"],
            errors="raise",
        )
    )

    real_yield = (
        real_yield_features.copy()
    )

    real_yield[
        "real_yield_10y_state_date"
    ] = pd.to_datetime(
        real_yield[
            "real_yield_10y_state_date"
        ],
        errors="raise",
    )

    market = market.sort_values(
        "market_date"
    )

    real_yield = real_yield.sort_values(
        "real_yield_10y_state_date"
    )

    merged = pd.merge_asof(
        market,
        real_yield,
        left_on="market_date",
        right_on=(
            "real_yield_10y_state_date"
        ),
        direction="backward",
        allow_exact_matches=True,
    )

    future = (
        merged[
            "real_yield_10y_state_date"
        ].notna()
        & (
            merged[
                "real_yield_10y_state_date"
            ]
            > merged[
                "market_date"
            ]
        )
    )

    if future.any():
        raise AssertionError(
            "Future real-yield observation "
            "entered risk dataset."
        )

    return merged


def add_risk_inputs(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create Risk Governor V1 market inputs."""

    output = data.copy()

    required = {
        "market_date",
        "close",
        "ema_21",
        "return_1d",
        "change_30y_1d_bp",
        "change_30y_5d_bp",
        "volatility_5d",
        "volatility_20d",
        "real_yield_10y_state_date",
        "real_yield_10y_change_5d_bp",
    }

    missing = (
        required
        - set(output.columns)
    )

    if missing:
        raise ValueError(
            "Risk dataset missing columns: "
            f"{sorted(missing)}"
        )

    output[
        "tmf_return_1d_pct"
    ] = (
        output[
            "return_1d"
        ]
        * 100.0
    )

    output[
        "volatility_ratio_5d_20d"
    ] = (
        output[
            "volatility_5d"
        ]
        / output[
            "volatility_20d"
        ]
    )

    output[
        "below_ema21"
    ] = (
        output["close"]
        < output["ema_21"]
    )

    output[
        "real_yield_age_days"
    ] = (
        output[
            "market_date"
        ]
        - output[
            "real_yield_10y_state_date"
        ]
    ).dt.days

    required_numeric = [
        "tmf_return_1d_pct",
        "change_30y_1d_bp",
        "change_30y_5d_bp",
        "real_yield_10y_change_5d_bp",
        "volatility_ratio_5d_20d",
    ]

    missing_required = (
        output[
            required_numeric
        ]
        .isna()
        .any(axis=1)
    )

    # Friday -> Monday = 3 calendar days.
    # Friday -> Tuesday after a Monday holiday = 4.
    # Anything older is treated as stale.
    stale_real_yield = (
        output[
            "real_yield_age_days"
        ].isna()
        | (
            output[
                "real_yield_age_days"
            ]
            > 4
        )
    )

    output[
        "critical_data_stale"
    ] = (
        missing_required
        | stale_real_yield
    )

    return output


def build_risk_market_dataset() -> pd.DataFrame:
    """Build complete Risk Governor V1 market-state dataset."""

    market = pd.read_csv(
        TREASURY_FEATURE_FILE,
        parse_dates=[
            "market_date",
        ],
    )

    real_yield_raw = pd.read_csv(
        REAL_YIELD_FILE,
    )

    real_yield = (
        prepare_real_yield_10y(
            real_yield_raw
        )
    )

    combined = (
        merge_real_yield_to_market(
            market_data=market,
            real_yield_features=real_yield,
        )
    )

    combined = (
        add_risk_inputs(
            combined
        )
    )

    return combined


def main() -> None:
    """Generate and save Risk Governor market features."""

    data = (
        build_risk_market_dataset()
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    governor_inputs = [
        "tmf_return_1d_pct",
        "change_30y_1d_bp",
        "change_30y_5d_bp",
        "real_yield_10y_change_5d_bp",
        "volatility_ratio_5d_20d",
    ]

    print("=" * 72)
    print("RISK GOVERNOR MARKET FEATURES")
    print("=" * 72)

    print(
        "\nSaved:",
        OUTPUT_FILE,
    )

    print(
        "Shape:",
        data.shape,
    )

    print(
        "Market range:",
        data[
            "market_date"
        ].min().date(),
        "to",
        data[
            "market_date"
        ].max().date(),
    )

    print(
        "\nMissing governor inputs:"
    )

    print(
        data[
            governor_inputs
        ].isna().sum()
    )

    print(
        "\nStale-data rows:",
        int(
            data[
                "critical_data_stale"
            ].sum()
        ),
    )

    print(
        "\nReal-yield state age:"
    )

    print(
        data[
            "real_yield_age_days"
        ].describe()
    )


if __name__ == "__main__":
    main()