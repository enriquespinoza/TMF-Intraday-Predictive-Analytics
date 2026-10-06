"""
Evaluate Market Confirmation V1 independently.

This analysis does not optimize thresholds.

It examines:

- Market Confirmation score coverage
- score distribution
- V3-entry-open observations
- forward 1-day and 5-day returns
- 5-day positive-return rate
- maximum adverse excursion
- maximum favorable excursion
- score quintiles

Market Confirmation V1 should already be frozen before this
target-based evaluation is run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.signals.market_confirmation import (
    add_market_confirmation_features,
)


INPUT_FILE = Path(
    "reports/"
    "V4_risk_governor_v3_daily.csv"
)

DAILY_OUTPUT_FILE = Path(
    "reports/"
    "V4_market_confirmation_v1_daily.csv"
)

SUMMARY_OUTPUT_FILE = Path(
    "reports/"
    "V4_market_confirmation_v1_summary.csv"
)

QUINTILE_OUTPUT_FILE = Path(
    "reports/"
    "V4_market_confirmation_v1_quintiles.csv"
)


def add_forward_path_metrics(
    data: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Add close-to-close forward MAE and MFE.

    Today's close is the reference point.

    The path includes only future closes t+1 through t+horizon.
    """

    output = data.copy()

    paths = []

    for step in range(
        1,
        horizon + 1,
    ):
        future_close = (
            output[
                "close"
            ]
            .shift(-step)
        )

        path_return = (
            future_close
            / output[
                "close"
            ]
            - 1.0
        )

        paths.append(
            path_return
        )

    path_matrix = pd.concat(
        paths,
        axis=1,
    )

    output[
        "forward_5d_mae"
    ] = path_matrix.min(
        axis=1
    )

    output[
        "forward_5d_mfe"
    ] = path_matrix.max(
        axis=1
    )

    return output


def summarize_sample(
    data: pd.DataFrame,
    name: str,
) -> dict:
    """
    Summarize one Market Confirmation sample.
    """

    return {
        "sample":
            name,

        "n":
            len(data),

        "mean_score":
            data[
                "market_confirmation_score"
            ].mean(),

        "median_score":
            data[
                "market_confirmation_score"
            ].median(),

        "mean_forward_return_5":
            data[
                "forward_return_5"
            ].mean(),

        "positive_5d_rate":
            (
                data[
                    "forward_return_5"
                ]
                > 0
            ).mean(),

        "mean_forward_5d_mae":
            data[
                "forward_5d_mae"
            ].mean(),

        "mean_forward_5d_mfe":
            data[
                "forward_5d_mfe"
            ].mean(),
    }


def main() -> None:
    data = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "market_date",
        ],
    )

    data = (
        data
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )

    scored = (
        add_market_confirmation_features(
            data
        )
    )

    scored = (
        add_forward_path_metrics(
            scored
        )
    )

    eligible = (
        scored[
            "market_confirmation_score"
        ].notna()
        & scored[
            "forward_return_5"
        ].notna()
    )

    evaluation = (
        scored.loc[
            eligible
        ]
        .copy()
    )

    entry_open_mask = (
        evaluation[
            "v3_new_entries_allowed"
        ].astype(bool)
    )

    open_sample = (
        evaluation.loc[
            entry_open_mask
        ]
        .copy()
    )

    summary = pd.DataFrame(
        [
            summarize_sample(
                evaluation,
                "ALL_COMPLETE",
            ),

            summarize_sample(
                open_sample,
                "V3_ENTRY_OPEN",
            ),
        ]
    )

    if len(open_sample) >= 5:

        open_sample[
            "market_score_quintile"
        ] = pd.qcut(
            open_sample[
                "market_confirmation_score"
            ],
            q=5,
            labels=[
                "Q1_LOW",
                "Q2",
                "Q3",
                "Q4",
                "Q5_HIGH",
            ],
            duplicates="drop",
        )

        quintiles = (
            open_sample.groupby(
                "market_score_quintile",
                observed=True,
                as_index=False,
            )
            .agg(
                n=(
                    "market_date",
                    "size",
                ),

                mean_score=(
                    "market_confirmation_score",
                    "mean",
                ),

                mean_forward_return_1=(
                    "forward_return_1",
                    "mean",
                ),

                mean_forward_return_5=(
                    "forward_return_5",
                    "mean",
                ),

                median_forward_return_5=(
                    "forward_return_5",
                    "median",
                ),

                positive_5d_rate=(
                    "forward_return_5",
                    lambda x:
                        (
                            x > 0
                        ).mean(),
                ),

                mean_forward_5d_mae=(
                    "forward_5d_mae",
                    "mean",
                ),

                median_forward_5d_mae=(
                    "forward_5d_mae",
                    "median",
                ),

                mean_forward_5d_mfe=(
                    "forward_5d_mfe",
                    "mean",
                ),

                median_forward_5d_mfe=(
                    "forward_5d_mfe",
                    "median",
                ),
            )
        )

    else:
        quintiles = pd.DataFrame()

    DAILY_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored.to_csv(
        DAILY_OUTPUT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
    )

    quintiles.to_csv(
        QUINTILE_OUTPUT_FILE,
        index=False,
    )

    print("=" * 78)
    print(
        "V4 OPPORTUNITY ENGINE V1 — "
        "MARKET CONFIRMATION COMPONENT"
    )
    print("=" * 78)

    print(
        "\nSCORE COVERAGE"
    )

    print(
        "Total rows:",
        len(scored),
    )

    print(
        "Complete Market Confirmation scores:",
        int(
            scored[
                "market_confirmation_score"
            ].notna().sum()
        ),
    )

    scored_dates = scored.loc[
        scored[
            "market_confirmation_score"
        ].notna(),
        "market_date",
    ]

    if not scored_dates.empty:
        print(
            "First scored date:",
            scored_dates.min().date(),
        )

    print(
        "\nSUMMARY"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nV3 ENTRY-OPEN SCORE DISTRIBUTION"
    )

    print(
        open_sample[
            "market_confirmation_score"
        ].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
            ]
        )
    )

    print(
        "\nENTRY-OPEN SCORE QUINTILES"
    )

    if quintiles.empty:
        print(
            "Insufficient observations."
        )

    else:
        print(
            quintiles.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.6f}",
            )
        )

    print(
        "\nCOMPONENT CORRELATIONS"
    )

    component_columns = [
        "market_price_response_score",
        "market_ema9_confirmation_score",
        "market_ema_structure_score",
        "market_volume_participation_score",
        "market_confirmation_score",
    ]

    correlations = (
        open_sample[
            component_columns
        ]
        .corr()
    )

    print(
        correlations.to_string(
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print(
        "\nImportant:"
    )

    print(
        "These forward-return results are "
        "descriptive historical research."
    )

    print(
        "Do not choose a Market Confirmation "
        "threshold from this table yet."
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {DAILY_OUTPUT_FILE}"
    )

    print(
        f"  {SUMMARY_OUTPUT_FILE}"
    )

    print(
        f"  {QUINTILE_OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()