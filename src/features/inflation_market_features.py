"""
Market-implied inflation features for frozen INFLATION-A.

Frozen features:

    breakeven_10y
    breakeven_10y_change_5d_bp

Methodology
-----------
- T10YIE is a daily market-derived FRED series.
- Missing FRED observations, such as holidays, are not treated as
  economic observations.
- The 5-day change compares the current valid observation with the
  fifth previous valid observation.
- Percentage-point changes are converted to basis points by
  multiplying by 100.
- Market dates receive only values known on or before that date
  through a backward as-of merge.
- 2026 may remain in generated data but stays locked downstream
  during model selection and validation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BREAKEVEN_SERIES_NAME = "breakeven_10y"

RAW_BREAKEVEN_FILE = Path(
    "data/raw/macro/breakeven_10y.csv"
)

FEATURE_OUTPUT_FILE = Path(
    "data/features/breakeven_10y_features.csv"
)


def prepare_breakeven_10y(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize the 10-year breakeven series."""

    required_columns = {
        "reference_date",
        "value",
        "series_name",
        "series_id",
        "category",
        "source",
        "frequency",
        "units",
    }

    missing = (
        required_columns
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            "Breakeven data missing required "
            f"columns: {sorted(missing)}"
        )

    result = data.loc[
        data["series_name"]
        == BREAKEVEN_SERIES_NAME
    ].copy()

    if result.empty:
        raise ValueError(
            "No breakeven_10y observations found."
        )

    result["reference_date"] = pd.to_datetime(
        result["reference_date"],
        errors="raise",
    ).dt.normalize()

    result["value"] = pd.to_numeric(
        result["value"],
        errors="coerce",
    )

    if result[
        "reference_date"
    ].duplicated().any():
        raise ValueError(
            "Duplicate breakeven_10y "
            "reference dates found."
        )

    return (
        result.sort_values(
            "reference_date"
        )
        .reset_index(drop=True)
    )


def build_breakeven_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct frozen INFLATION-A features.

    Five-day change is calculated across five valid published
    observations, so holiday NaNs do not distort the horizon.
    """

    prepared = prepare_breakeven_10y(
        data
    )

    valid = (
        prepared.dropna(
            subset=["value"]
        )
        .loc[
            :,
            [
                "reference_date",
                "value",
            ],
        ]
        .copy()
    )

    valid = valid.rename(
        columns={
            "reference_date":
                "breakeven_state_date",
            "value":
                "breakeven_10y",
        }
    )

    valid[
        "breakeven_10y_change_5d_bp"
    ] = (
        valid["breakeven_10y"]
        - valid["breakeven_10y"].shift(5)
    ) * 100.0

    return (
        valid.sort_values(
            "breakeven_state_date"
        )
        .reset_index(drop=True)
    )


def merge_breakeven_features_to_market(
    market_data: pd.DataFrame,
    breakeven_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Backward-merge breakeven features onto market observations.

    No market row may receive a breakeven state from the future.
    """

    market = market_data.copy()
    features = breakeven_features.copy()

    if "market_date" not in market.columns:
        raise ValueError(
            "Market data must contain "
            "'market_date'."
        )

    required_features = {
        "breakeven_state_date",
        "breakeven_10y",
        "breakeven_10y_change_5d_bp",
    }

    missing = (
        required_features
        - set(features.columns)
    )

    if missing:
        raise ValueError(
            "Breakeven feature data missing "
            f"columns: {sorted(missing)}"
        )

    market["market_date"] = pd.to_datetime(
        market["market_date"],
        errors="raise",
    ).dt.normalize()

    features[
        "breakeven_state_date"
    ] = pd.to_datetime(
        features[
            "breakeven_state_date"
        ],
        errors="raise",
    ).dt.normalize()

    market = market.sort_values(
        "market_date"
    )

    features = features.sort_values(
        "breakeven_state_date"
    )

    merged = pd.merge_asof(
        market,
        features,
        left_on="market_date",
        right_on="breakeven_state_date",
        direction="backward",
        allow_exact_matches=True,
    )

    future_violation = (
        merged[
            "breakeven_state_date"
        ].notna()
        & (
            merged[
                "breakeven_state_date"
            ]
            > merged["market_date"]
        )
    )

    if future_violation.any():
        raise AssertionError(
            "Future breakeven state entered "
            "the market dataset."
        )

    return merged


def main() -> None:
    """Generate frozen INFLATION-A feature history."""

    raw = pd.read_csv(
        RAW_BREAKEVEN_FILE
    )

    features = build_breakeven_features(
        raw
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
            "breakeven_state_date"
        ].min().date(),
        "to",
        features[
            "breakeven_state_date"
        ].max().date(),
    )

    print(
        "\nMissing feature values:"
    )

    print(
        features[
            [
                "breakeven_10y",
                "breakeven_10y_change_5d_bp",
            ]
        ].isna().sum()
    )

    print(
        "\nLatest observations:"
    )

    print(
        features.tail(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
    