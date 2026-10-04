"""Research relationships between TMF features and forward returns."""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/features/TMF_1d_technical_features.csv"
)

OUTPUT_DIR = Path("reports")
OUTPUT_FILE = OUTPUT_DIR / "TMF_feature_research.csv"


FEATURES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "ema_spread_pct",
    "close_vs_ema9",
    "close_vs_ema21",
    "rsi_14",
    "rsi_centered",
    "momentum_5d_pct",
    "momentum_10d_pct",
    "volatility_5d",
    "volatility_20d",
    "range_pct",
    "relative_volume",
    "log_volume",
    "distance_from_20d_high",
    "distance_from_20d_low",
]


TARGETS = [
    "forward_return_1",
    "forward_return_3",
    "forward_return_5",
    "forward_return_10",
    "forward_return_20",
]


def calculate_feature_relationships(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate feature/target relationships."""

    rows = []

    for feature in FEATURES:
        for target in TARGETS:

            if (
                feature not in df.columns
                or target not in df.columns
            ):
                continue
            sample = df[
                [feature, target]
            ].dropna()

            if len(sample) < 3:
                continue

            pearson = sample[
                feature
            ].corr(
                sample[target],
                method="pearson",
            )

            spearman = sample[
                feature
            ].corr(
                sample[target],
                method="spearman",
            )

            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "observations": len(sample),
                    "pearson_corr": pearson,
                    "spearman_corr": spearman,
                    "abs_pearson_corr": abs(pearson),
                    "abs_spearman_corr": abs(spearman),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    """Run feature research."""

    print("Loading technical feature dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"],
    )

    print(
        f"Loaded {len(df):,} observations "
        f"and {len(df.columns)} columns."
    )

    print("Calculating feature relationships...")

    results = calculate_feature_relationships(df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Feature research complete.")
    print()

    five_day = (
        results[
            results["target"]
            == "forward_return_5"
        ]
        .sort_values(
            "abs_spearman_corr",
            ascending=False,
        )
    )

    print("5-DAY FORWARD RETURN")
    print(
        five_day[
            [
                "feature",
                "observations",
                "pearson_corr",
                "spearman_corr",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Full report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
