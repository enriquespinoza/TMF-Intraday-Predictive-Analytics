"""Research Treasury-rate relationships with TMF forward returns."""

from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/features/TMF_1d_technical_treasury_features.csv"
)

OUTPUT_FILE = Path(
    "reports/TMF_treasury_feature_research.csv"
)


TREASURY_FEATURES = [
    # Yield levels
    "yield_2y",
    "yield_5y",
    "yield_10y",
    "yield_30y",

    # 1-day changes
    "change_2y_1d_bp",
    "change_5y_1d_bp",
    "change_10y_1d_bp",
    "change_30y_1d_bp",

    # 5-day changes
    "change_2y_5d_bp",
    "change_5y_5d_bp",
    "change_10y_5d_bp",
    "change_30y_5d_bp",

    # 20-day changes
    "change_2y_20d_bp",
    "change_5y_20d_bp",
    "change_10y_20d_bp",
    "change_30y_20d_bp",

    # Curve slopes
    "curve_10y_2y_bp",
    "curve_30y_2y_bp",
    "curve_30y_5y_bp",
    "curve_30y_10y_bp",

    # Curve changes
    "curve_10y_2y_bp_change_5d",
    "curve_30y_2y_bp_change_5d",
    "curve_30y_5y_bp_change_5d",
    "curve_30y_10y_bp_change_5d",

    # Rate volatility
    "yield_10y_volatility_20d",
    "yield_30y_volatility_20d",
]


TARGETS = [
    "forward_return_1",
    "forward_return_3",
    "forward_return_5",
    "forward_return_10",
    "forward_return_20",
]


def calculate_treasury_relationships(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate Treasury feature relationships with TMF returns."""

    rows = []

    for feature in TREASURY_FEATURES:

        if feature not in df.columns:
            continue

        for target in TARGETS:

            if target not in df.columns:
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
    """Run Treasury feature research."""

    print("Loading combined TMF + Treasury dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["market_date"],
    )

    print(
        f"Loaded {len(df):,} observations "
        f"and {len(df.columns)} columns."
    )

    print("Calculating Treasury relationships...")

    results = calculate_treasury_relationships(df)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

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

    print()
    print("5-DAY TMF FORWARD RETURN")
    print()

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
    print(
        f"Full report saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()