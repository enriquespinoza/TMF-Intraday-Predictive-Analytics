"""
Exploratory research for point-in-time PAYEMS features.

Important:
- Uses development data only through 2025-12-31.
- Purges rows whose 5-trading-day target extends into 2026.
- 2026 remains locked for model selection.
- Reports both daily-state relationships and release-day relationships.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/features/"
    "TMF_1d_technical_PAYEMS_features.csv"
)

DEVELOPMENT_END = pd.Timestamp("2025-12-31")
TARGET_HORIZON = 5

PAYROLL_FEATURES = [
    "payroll_change_1m",
    "payroll_change_z",
    "payroll_revision_1m",
    "payroll_revision_2m",
    "payroll_revision_total",
    "payroll_revision_z",
    "days_since_payroll_release",
]

TARGET = "forward_return_5"


def correlation_table(
    data: pd.DataFrame,
    features: list[str],
    target: str,
) -> pd.DataFrame:
    """Calculate Pearson and Spearman relationships."""

    rows = []

    for feature in features:
        sample = data[
            [
                feature,
                target,
            ]
        ].dropna()

        if sample.empty:
            continue

        rows.append(
            {
                "feature": feature,
                "observations": len(sample),
                "unique_feature_values": (
                    sample[feature].nunique()
                ),
                "pearson": (
                    sample[feature]
                    .corr(
                        sample[target],
                        method="pearson",
                    )
                ),
                "spearman": (
                    sample[feature]
                    .corr(
                        sample[target],
                        method="spearman",
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    df["market_date"] = pd.to_datetime(
        df["market_date"],
        errors="raise",
    )

    df["event_release_date"] = pd.to_datetime(
        df["event_release_date"],
        errors="coerce",
    )

    df = df.sort_values(
        "market_date"
    ).reset_index(drop=True)

    #
    # Match the 5-trading-day forward-return target.
    #
    # If forward_return_5 for row i uses the close five market
    # observations later, this is its target end date.
    #
    df["target_end_date_5d"] = (
        df["market_date"].shift(
            -TARGET_HORIZON
        )
    )

    #
    # Development data only.
    #
    # A row is eligible only if BOTH:
    #   1. the feature date is in development, and
    #   2. the target itself ends by 2025-12-31.
    #
    development = df.loc[
        (
            df["market_date"]
            <= DEVELOPMENT_END
        )
        & (
            df["target_end_date_5d"]
            <= DEVELOPMENT_END
        )
    ].copy()

    print("=" * 90)
    print("PAYEMS FEATURE RESEARCH — DEVELOPMENT SAMPLE ONLY")
    print("=" * 90)

    print()
    print("Full dataset rows:")
    print(len(df))

    print()
    print("Development rows with purged 5-day target:")
    print(len(development))

    print()
    print("Development date range:")
    print(
        development["market_date"].min(),
        "->",
        development["market_date"].max(),
    )

    print()
    print("Latest 5-day target end:")
    print(
        development["target_end_date_5d"].max()
    )

    print()
    print("2026 feature rows used:")
    print(
        (
            development["market_date"].dt.year
            == 2026
        ).sum()
    )

    print()
    print("2026 target-end rows used:")
    print(
        (
            development[
                "target_end_date_5d"
            ].dt.year
            == 2026
        ).sum()
    )

    #
    # Feature distributions.
    #
    print()
    print("=" * 90)
    print("PAYEMS FEATURE DISTRIBUTIONS")
    print("=" * 90)

    print(
        development[
            PAYROLL_FEATURES
        ].describe().T.to_string()
    )

    #
    # Daily state relationships.
    #
    # These features persist between releases, so this should not
    # be interpreted as 1,000+ independent macro observations.
    #
    print()
    print("=" * 90)
    print("DAILY STATE VS 5-DAY TMF FORWARD RETURN")
    print("=" * 90)

    daily_correlations = correlation_table(
        data=development,
        features=PAYROLL_FEATURES,
        target=TARGET,
    )

    print(
        daily_correlations.to_string(
            index=False
        )
    )

    #
    # Release-day research.
    #
    # This avoids counting the same payroll information repeatedly
    # on every trading day between monthly reports.
    #
    release_days = development.loc[
        development["payroll_release_day"]
        == 1
    ].copy()

    regular_release_days = release_days.loc[
        release_days["payroll_broad_revision_event"]
        == 0
    ].copy()

    broad_release_days = release_days.loc[
        release_days["payroll_broad_revision_event"]
        == 1
    ].copy()

    print()
    print("=" * 90)
    print("EMPLOYMENT SITUATION RELEASE DAYS")
    print("=" * 90)

    print(
        "Release-day observations:",
        len(release_days),
    )

    print()
    print("Release-day date range:")
    print(
        release_days["market_date"].min(),
        "->",
        release_days["market_date"].max(),
    )

    release_correlations = correlation_table(
        data=release_days,
        features=[
            "payroll_change_1m",
            "payroll_change_z",
            "payroll_revision_1m",
            "payroll_revision_2m",
            "payroll_revision_total",
            "payroll_revision_z",
        ],
        target=TARGET,
    )

    print()
    print("Release feature correlations:")
    print(
        release_correlations.to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("REGULAR PAYROLL RELEASES — BROAD REVISIONS EXCLUDED")
    print("=" * 90)

    print(
        "Regular release observations:",
        len(regular_release_days),
    )

    print(
        "Broad revision observations:",
        len(broad_release_days),
    )

    regular_correlations = correlation_table(
        data=regular_release_days,
        features=[
             "payroll_change_1m",
      "payroll_change_z",
        "payroll_revision_1m",
        "payroll_revision_2m",
        "payroll_revision_total",
        "payroll_revision_z",
        ],
        target=TARGET,
    )

    print()
    print("Regular-release correlations:")
    print(
        regular_correlations.to_string(
            index=False
        )
    )

    print()
    print("Broad revision events:")
    print(
        broad_release_days[
            [
                "market_date",
                "payroll_change_1m",
                "payroll_revised_periods",
                "payroll_revision_1m",
                "payroll_revision_2m",
                "payroll_revision_total",
                TARGET,
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Release correlations:")
    print(
        release_correlations.to_string(
            index=False
        )
    )

#
# Positive vs negative revision events.
#
    usable_revisions = regular_release_days.loc[
        regular_release_days[
            "payroll_revision_total"
        ].notna()
    ].copy()

    positive_revision = usable_revisions.loc[
        usable_revisions[
            "payroll_revision_total"
        ] > 0
    ]

    negative_revision = usable_revisions.loc[
        usable_revisions[
            "payroll_revision_total"
        ] < 0
    ]

    print()
    print("=" * 90)
    print("REVISION SIGN ANALYSIS")
    print("=" * 90)

    print(
        "Positive revision events:",
        len(positive_revision),
    )

    print(
        "Mean 5d TMF return after positive revisions:",
        positive_revision[TARGET].mean(),
    )

    print(
        "Negative revision events:",
        len(negative_revision),
    )

    if not negative_revision.empty:
        print(
            "Mean 5d TMF return after negative revisions:",
            negative_revision[TARGET].mean(),
        )

#
# Show the largest revisions for audit/research.
#
    print()
    print("=" * 90)
    print("LARGEST ABSOLUTE PAYROLL REVISIONS")
    print("=" * 90)

    largest = (
        release_days
        .assign(
            abs_revision=lambda x:
            x["payroll_revision_total"].abs()
        )
        .sort_values(
            "abs_revision",
            ascending=False,
        )
        [
            [
                "market_date",
                "payroll_change_1m",
                "payroll_revision_1m",
                "payroll_revision_2m",
                "payroll_revision_total",
                TARGET,
            ]
        ]
        .head(10)
    )

    print(
        largest.to_string(
            index=False
        )
    )
    print()
    print("=" * 90)
    print("STANDARDIZED PAYEMS FEATURES — REGULAR RELEASES")
    print("=" * 90)

    standardized = regular_release_days[
        [
            "market_date",
            "payroll_change_1m",
            "payroll_change_z",
            "payroll_revision_total",
            "payroll_revision_z",
            "forward_return_5",
        ]
    ].dropna(
        subset=[
            "payroll_change_z",
            "payroll_revision_z",
            "forward_return_5",
        ]
    )

    print(
        "Usable standardized release observations:",
        len(standardized),
    )

    print()
    print("Pearson:")

    print(
        standardized[
            [
                "payroll_change_z",
                "payroll_revision_z",
                "forward_return_5",
            ]
        ]
        .corr(method="pearson")
        .to_string()
    )

    print()
    print("Spearman:")

    print(
        standardized[
            [
                "payroll_change_z",
                "payroll_revision_z",
                "forward_return_5",
            ]
        ]
        .corr(method="spearman")
        .to_string()
    )


if __name__ == "__main__":
    main()