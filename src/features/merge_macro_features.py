"""Point-in-time alignment of macroeconomic data to market observations.

This module provides the common leakage-control layer used to align
release-sensitive macroeconomic observations with market data.

The central rule is:

    macro information may only be attached to a market observation
    when market_date_available <= market_date

The merge is backward-looking only. Future macro observations must
never be attached to earlier market observations.
"""

from __future__ import annotations

import pandas as pd


REQUIRED_MARKET_COLUMNS = {
    "timestamp",
}

REQUIRED_MACRO_COLUMNS = {
    "reference_date",
    "value",
    "market_date_available",
}


def _normalize_date(
    values: pd.Series,
) -> pd.Series:
    """Convert date-like values to timezone-naive normalized dates."""

    parsed = pd.to_datetime(
        values,
        errors="raise",
        utc=True,
    )

    return (
        parsed
        .dt.tz_localize(None)
        .dt.normalize()
    )


def validate_market_data(
    market: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize market observations."""

    missing = (
        REQUIRED_MARKET_COLUMNS
        - set(market.columns)
    )

    if missing:
        raise ValueError(
            "Market data missing required "
            f"columns: {sorted(missing)}"
        )

    result = market.copy()

    result["timestamp"] = _normalize_date(
        result["timestamp"]
    )

    if result["timestamp"].isna().any():
        raise ValueError(
            "Market data contains missing timestamps."
        )

    if result["timestamp"].duplicated().any():
        raise ValueError(
            "Market data contains duplicate timestamps."
        )

    return result.sort_values(
        "timestamp"
    ).reset_index(drop=True)


def validate_macro_data(
    macro: pd.DataFrame,
    series_name: str,
) -> pd.DataFrame:
    """Validate a prepared point-in-time macro series."""

    missing = (
        REQUIRED_MACRO_COLUMNS
        - set(macro.columns)
    )

    if missing:
        raise ValueError(
            f"{series_name} missing required "
            f"columns: {sorted(missing)}"
        )

    result = macro.copy()

    result["reference_date"] = _normalize_date(
        result["reference_date"]
    )

    result["market_date_available"] = (
        _normalize_date(
            result["market_date_available"]
        )
    )

    # Null economic observations contain no information.
    result = result.loc[
        result["value"].notna()
    ].copy()

    if result.empty:
        return result

    if (
        result["market_date_available"]
        .isna()
        .any()
    ):
        raise ValueError(
            f"{series_name} contains non-null "
            "observations without market availability."
        )

    invalid = (
        result["market_date_available"]
        < result["reference_date"]
    )

    if invalid.any():
        raise ValueError(
            f"{series_name} contains observations "
            "available before their reference date."
        )

    if (
        result["reference_date"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            f"{series_name} contains duplicate "
            "reference dates."
        )

    return result.sort_values(
        [
            "market_date_available",
            "reference_date",
        ]
    ).reset_index(drop=True)


def prepare_macro_for_merge(
    macro: pd.DataFrame,
    series_name: str,
) -> pd.DataFrame:
    """Rename one macro series into merge-safe feature columns."""

    result = validate_macro_data(
        macro=macro,
        series_name=series_name,
    )

    if result.empty:
        return pd.DataFrame(
            columns=[
                "market_date_available",
                series_name,
                f"{series_name}_reference_date",
                f"{series_name}_release_date",
                f"{series_name}_release_time",
                f"{series_name}_release_timezone",
            ]
        )

    rename_map = {
        "value": series_name,
        "reference_date": (
            f"{series_name}_reference_date"
        ),
    }

    optional_columns = {
        "release_date": (
            f"{series_name}_release_date"
        ),
        "release_time": (
            f"{series_name}_release_time"
        ),
        "release_timezone": (
            f"{series_name}_release_timezone"
        ),
    }

    for source, target in optional_columns.items():
        if source in result.columns:
            rename_map[source] = target

    result = result.rename(
        columns=rename_map
    )

    keep_columns = [
        "market_date_available",
        series_name,
        f"{series_name}_reference_date",
    ]

    for target in optional_columns.values():
        if target in result.columns:
            keep_columns.append(target)

    return result[
        keep_columns
    ].copy()


def merge_macro_series(
    market: pd.DataFrame,
    macro: pd.DataFrame,
    series_name: str,
) -> pd.DataFrame:
    """Backward-align one macro series to market observations."""

    market_data = validate_market_data(
        market
    )

    macro_data = prepare_macro_for_merge(
        macro=macro,
        series_name=series_name,
    )

    availability_column = (
        f"{series_name}_market_date_available"
    )

    if macro_data.empty:
        result = market_data.copy()

        result[series_name] = pd.NA
        result[
            f"{series_name}_reference_date"
        ] = pd.NaT
        result[availability_column] = pd.NaT

        return result

    macro_data = macro_data.rename(
        columns={
            "market_date_available":
                availability_column
        }
    )

    result = pd.merge_asof(
        market_data.sort_values(
            "timestamp"
        ),
        macro_data.sort_values(
            availability_column
        ),
        left_on="timestamp",
        right_on=availability_column,
        direction="backward",
        allow_exact_matches=True,
    )

    known = result[
        availability_column
    ].notna()

    leakage = (
        known
        & (
            result[availability_column]
            > result["timestamp"]
        )
    )

    if leakage.any():
        raise RuntimeError(
            f"Point-in-time violation detected "
            f"while merging {series_name}."
        )

    result[
        f"{series_name}_data_age_days"
    ] = (
        result["timestamp"]
        - result[availability_column]
    ).dt.days

    return result


def merge_macro_family(
    market: pd.DataFrame,
    macro_series: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge multiple point-in-time macro series onto market data."""

    result = validate_market_data(
        market
    )

    for series_name, macro in macro_series.items():
        result = merge_macro_series(
            market=result,
            macro=macro,
            series_name=series_name,
        )

    return result
