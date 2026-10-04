"""Point-in-time alignment of TMF and Treasury features."""

from pathlib import Path

import pandas as pd

from src.features.treasury_features import (
    build_treasury_features,
)


TMF_FILE = Path(
    "data/features/TMF_1d_technical_features.csv"
)

TREASURY_FILE = Path(
    "data/raw/treasury_yields_daily.csv"
)

OUTPUT_DIR = Path("data/features")

OUTPUT_FILE = (
    OUTPUT_DIR / "TMF_1d_technical_treasury_features.csv"
)


def normalize_date(
    series: pd.Series,
) -> pd.Series:
    """Convert timestamps to timezone-neutral calendar dates."""

    timestamps = pd.to_datetime(
        series,
        utc=True,
    )

    return timestamps.dt.tz_convert(None).dt.normalize()


def align_treasury_to_tmf(
    tmf: pd.DataFrame,
    treasury: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align Treasury information to TMF trading dates.

    Treasury observations are forward-filled only after being
    ordered chronologically. No backward filling is permitted.

    treasury_observation_date records the actual date from which
    the Treasury information originated.
    """

    tmf_data = tmf.copy()
    treasury_data = treasury.copy()

    # ---------------------------------------------------------
    # Normalize timestamps
    # ---------------------------------------------------------

    tmf_data["market_date"] = normalize_date(
        tmf_data["timestamp"]
    )

    treasury_data["market_date"] = normalize_date(
        treasury_data["timestamp"]
    )

    # Preserve the actual Treasury observation date.
    treasury_data["treasury_observation_date"] = (
        treasury_data["market_date"]
    )

    # ---------------------------------------------------------
    # Treasury features
    # ---------------------------------------------------------

    treasury_data = build_treasury_features(
        treasury_data
    )

    # ---------------------------------------------------------
    # Remove rows where no Treasury yield observation exists.
    #
    # This prevents missing FRED observations from replacing
    # valid prior Treasury information during the as-of merge.
    # ---------------------------------------------------------

    treasury_data = treasury_data.dropna(
        subset=[
            "yield_2y",
            "yield_5y",
            "yield_10y",
            "yield_30y",
        ]
    )

    # ---------------------------------------------------------
    # Sort before as-of merge
    # ---------------------------------------------------------

    tmf_data = tmf_data.sort_values(
        "market_date"
    ).reset_index(drop=True)

    treasury_data = treasury_data.sort_values(
        "market_date"
    ).reset_index(drop=True)

    # Treasury timestamp is no longer needed because
    # treasury_observation_date preserves its provenance.
    treasury_data = treasury_data.drop(
        columns=["timestamp"],
        errors="ignore",
    )

    # ---------------------------------------------------------
    # Point-in-time merge
    #
    # direction="backward" means:
    #
    # use Treasury observation at or BEFORE the TMF date.
    #
    # Never use a future Treasury observation.
    # ---------------------------------------------------------

    merged = pd.merge_asof(
        tmf_data,
        treasury_data,
        on="market_date",
        direction="backward",
        allow_exact_matches=True,
    )

    # ---------------------------------------------------------
    # Measure Treasury data age
    # ---------------------------------------------------------

    merged["treasury_data_age_days"] = (
        merged["market_date"]
        - merged["treasury_observation_date"]
    ).dt.days

    return merged


def main() -> None:
    """Build combined TMF + Treasury feature dataset."""

    print("Loading TMF technical features...")

    tmf = pd.read_csv(TMF_FILE)

    print(
        f"TMF observations: {len(tmf):,}"
    )

    print("Loading Treasury data...")

    treasury = pd.read_csv(TREASURY_FILE)

    print(
        f"Treasury observations: {len(treasury):,}"
    )

    print("Aligning Treasury data to TMF trading dates...")

    merged = align_treasury_to_tmf(
        tmf,
        treasury,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Market feature merge complete.")
    print()

    print(
        f"Rows:    {len(merged):,}"
    )

    print(
        f"Columns: {len(merged.columns):,}"
    )

    print()

    print("Treasury data age:")
    print(
        merged[
            "treasury_data_age_days"
        ].value_counts(
            dropna=False
        ).sort_index()
    )

    print()

    print("Missing Treasury yields:")
    print(
        merged[
            [
                "yield_2y",
                "yield_5y",
                "yield_10y",
                "yield_30y",
            ]
        ].isna().sum()
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()