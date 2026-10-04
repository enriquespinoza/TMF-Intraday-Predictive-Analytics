"""Treasury yield-curve regime analysis for TMF."""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/features/TMF_1d_technical_treasury_features.csv"
)

OUTPUT_DIR = Path("reports")

OUTPUT_FILE = (
    OUTPUT_DIR / "TMF_treasury_regime_analysis.csv"
)

DEVELOPMENT_END = pd.Timestamp("2025-12-31")

HORIZONS = [1, 3, 5, 10, 20]


def classify_treasury_regime(
    row: pd.Series,
) -> str:
    """Classify Treasury curve regime using 5-day yield changes."""

    change_2y = row["change_2y_5d_bp"]
    change_30y = row["change_30y_5d_bp"]

    if pd.isna(change_2y) or pd.isna(change_30y):
        return "unclassified"

    # Both yields falling
    if change_2y < 0 and change_30y < 0:

        if change_2y < change_30y:
            return "bull_steepening"

        return "bull_flattening"

    # Both yields rising
    if change_2y > 0 and change_30y > 0:

        if change_30y > change_2y:
            return "bear_steepening"

        return "bear_flattening"

    # Short rates falling while long rates rise
    if change_2y < 0 and change_30y > 0:
        return "twist_steepening"

    # Short rates rising while long rates fall
    if change_2y > 0 and change_30y < 0:
        return "twist_flattening"

    return "mixed_or_flat"


def add_treasury_regime(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add Treasury regime classification."""

    data = df.copy()

    data["treasury_regime"] = data.apply(
        classify_treasury_regime,
        axis=1,
    )

    return data


def summarize_regimes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize TMF forward returns by Treasury regime."""

    rows = []

    for regime, group in df.groupby(
        "treasury_regime"
    ):

        row = {
            "treasury_regime": regime,
            "observations": len(group),
        }

        for horizon in HORIZONS:

            target = f"forward_return_{horizon}"

            sample = group[target].dropna()

            row[f"n_{horizon}d"] = len(sample)

            row[f"mean_return_{horizon}d"] = (
                sample.mean()
            )

            row[f"median_return_{horizon}d"] = (
                sample.median()
            )

            row[f"positive_rate_{horizon}d"] = (
                (sample > 0).mean()
                if len(sample) > 0
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_baseline(
    df: pd.DataFrame,
) -> dict:
    """Calculate unconditional TMF forward-return baseline."""

    baseline = {}

    for horizon in HORIZONS:

        target = f"forward_return_{horizon}"

        sample = df[target].dropna()

        baseline[horizon] = {
            "mean": sample.mean(),
            "positive_rate": (
                (sample > 0).mean()
                if len(sample) > 0
                else np.nan
            ),
        }

    return baseline


def add_excess_performance(
    summary: pd.DataFrame,
    baseline: dict,
) -> pd.DataFrame:
    """Add performance relative to the unconditional TMF baseline."""

    result = summary.copy()

    for horizon in HORIZONS:

        result[
            f"excess_return_{horizon}d"
        ] = (
            result[
                f"mean_return_{horizon}d"
            ]
            - baseline[horizon]["mean"]
        )

        result[
            f"excess_positive_rate_{horizon}d"
        ] = (
            result[
                f"positive_rate_{horizon}d"
            ]
            - baseline[horizon]["positive_rate"]
        )

    return result


def main() -> None:
    """Run Treasury regime research."""

    print("Loading TMF + Treasury dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["market_date"],
    )

    print(
        f"Full dataset: {len(df):,} observations"
    )

    # Development data only.
    # 2026 remains untouched.
    development = df[
        df["market_date"] <= DEVELOPMENT_END
    ].copy()

    print(
        f"Development dataset: "
        f"{len(development):,} observations"
    )

    print(
        "Development period:",
        development["market_date"].min(),
        "→",
        development["market_date"].max(),
    )

    # Classify regimes.
    development = add_treasury_regime(
        development
    )

    # Calculate unconditional baseline.
    baseline = calculate_baseline(
        development
    )

    # Calculate regime statistics.
    summary = summarize_regimes(
        development
    )

    # Calculate excess performance.
    summary = add_excess_performance(
        summary,
        baseline,
    )

    summary = summary.sort_values(
        "excess_return_5d",
        ascending=False,
    )

    # Save results.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # Display unconditional baseline.
    print()
    print("UNCONDITIONAL TMF BASELINE")
    print()

    for horizon in HORIZONS:

        print(
            f"{horizon:>2}D: "
            f"mean = "
            f"{baseline[horizon]['mean']:.4%}, "
            f"positive = "
            f"{baseline[horizon]['positive_rate']:.2%}"
        )

    # Display regime results.
    print()
    print("TREASURY REGIME RESULTS")
    print("Development sample only")
    print()

    display_columns = [
        "treasury_regime",
        "observations",
        "mean_return_5d",
        "excess_return_5d",
        "positive_rate_5d",
        "mean_return_10d",
        "excess_return_10d",
        "mean_return_20d",
        "excess_return_20d",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
        )
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()