"""
Final Opportunity Engine V1 architecture diagnostic.

Primary question:

Among Technical Opportunity Q4 observations, does a pooled
non-extreme Treasury environment (Q1-Q3) behave better than
extreme Treasury Confirmation Q4?

This is descriptive architecture research, not a predictive model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "reports/"
    "V4_opportunity_engine_v1_interaction_daily.csv"
)

DAILY_OUTPUT_FILE = Path(
    "reports/"
    "V4_opportunity_engine_v1_final_architecture_daily.csv"
)

OVERALL_OUTPUT_FILE = Path(
    "reports/"
    "V4_opportunity_engine_v1_final_architecture_overall.csv"
)

PERIOD_OUTPUT_FILE = Path(
    "reports/"
    "V4_opportunity_engine_v1_final_architecture_periods.csv"
)

BREAKDOWN_OUTPUT_FILE = Path(
    "reports/"
    "V4_opportunity_engine_v1_final_architecture_treasury_breakdown.csv"
)

ROBUSTNESS_OUTPUT_FILE = Path(
    "reports/"
    "V4_opportunity_engine_v1_final_architecture_robustness.csv"
)


MIN_CONTRAST_N = 5
SMALL_SAMPLE_N = 10


REQUIRED_COLUMNS = {
    "market_date",
    "technical_quartile",
    "treasury_quartile",
    "forward_return_1",
    "forward_return_5",
    "forward_5d_mae",
    "forward_5d_mfe",
}


def summarize(
    data: pd.DataFrame,
) -> dict:
    """Common outcome metrics."""

    if data.empty:
        return {
            "n": 0,
            "mean_forward_return_1": np.nan,
            "mean_forward_return_5": np.nan,
            "median_forward_return_5": np.nan,
            "positive_5d_rate": np.nan,
            "mean_forward_5d_mae": np.nan,
            "median_forward_5d_mae": np.nan,
            "mean_forward_5d_mfe": np.nan,
            "median_forward_5d_mfe": np.nan,
        }

    return {
        "n":
            len(data),

        "mean_forward_return_1":
            data[
                "forward_return_1"
            ].mean(),

        "mean_forward_return_5":
            data[
                "forward_return_5"
            ].mean(),

        "median_forward_return_5":
            data[
                "forward_return_5"
            ].median(),

        "positive_5d_rate":
            (
                data[
                    "forward_return_5"
                ] > 0
            ).mean(),

        "mean_forward_5d_mae":
            data[
                "forward_5d_mae"
            ].mean(),

        "median_forward_5d_mae":
            data[
                "forward_5d_mae"
            ].median(),

        "mean_forward_5d_mfe":
            data[
                "forward_5d_mfe"
            ].mean(),

        "median_forward_5d_mfe":
            data[
                "forward_5d_mfe"
            ].median(),
    }


def classify_treasury(
    quartile: str,
) -> str:
    """
    Pool Treasury Q1-Q3 as non-extreme.

    Q4 remains the extreme-confirmation state.
    """

    if quartile == "Q4_HIGH":
        return "EXTREME"

    if quartile in {
        "Q1_LOW",
        "Q2",
        "Q3",
    }:
        return "NON_EXTREME"

    raise ValueError(
        f"Unexpected Treasury quartile: {quartile}"
    )


def calendar_period(
    timestamp: pd.Timestamp,
) -> str:
    """Fixed calendar research periods."""

    year = timestamp.year

    if year in {
        2022,
        2023,
    }:
        return "2022_2023"

    if year == 2026:
        return "2026_YTD"

    return str(
        year
    )


def add_temporal_partitions(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recreate temporal partitions using the COMPLETE common sample.

    Blocks are assigned before filtering to Technical Q4.
    """

    output = (
        data.copy()
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )

    output[
        "calendar_period"
    ] = output[
        "market_date"
    ].map(
        calendar_period
    )

    order = pd.Series(
        np.arange(
            len(output)
        ),
        index=output.index,
    )

    output[
        "chronological_block"
    ] = pd.qcut(
        order,
        q=4,
        labels=[
            "BLOCK_1",
            "BLOCK_2",
            "BLOCK_3",
            "BLOCK_4",
        ],
    )

    return output


def build_comparison(
    data: pd.DataFrame,
    period_scheme: str,
    period: str,
) -> dict:
    """
    Compare non-extreme versus extreme Treasury conditions.
    """

    non_extreme = data.loc[
        data[
            "treasury_architecture_state"
        ] == "NON_EXTREME"
    ]

    extreme = data.loc[
        data[
            "treasury_architecture_state"
        ] == "EXTREME"
    ]

    non_metrics = summarize(
        non_extreme
    )

    extreme_metrics = summarize(
        extreme
    )

    eligible = (
        non_metrics[
            "n"
        ] >= MIN_CONTRAST_N
        and extreme_metrics[
            "n"
        ] >= MIN_CONTRAST_N
    )

    return_delta = (
        non_metrics[
            "mean_forward_return_5"
        ]
        - extreme_metrics[
            "mean_forward_return_5"
        ]
    )

    positive_rate_delta = (
        non_metrics[
            "positive_5d_rate"
        ]
        - extreme_metrics[
            "positive_5d_rate"
        ]
    )

    mae_delta = (
        non_metrics[
            "mean_forward_5d_mae"
        ]
        - extreme_metrics[
            "mean_forward_5d_mae"
        ]
    )

    mfe_delta = (
        non_metrics[
            "mean_forward_5d_mfe"
        ]
        - extreme_metrics[
            "mean_forward_5d_mfe"
        ]
    )

    supported = bool(
        eligible
        and return_delta > 0
        and mae_delta > 0
    )

    return {
        "period_scheme":
            period_scheme,

        "period":
            period,

        "non_extreme_n":
            non_metrics[
                "n"
            ],

        "extreme_n":
            extreme_metrics[
                "n"
            ],

        "non_extreme_mean_return_5":
            non_metrics[
                "mean_forward_return_5"
            ],

        "extreme_mean_return_5":
            extreme_metrics[
                "mean_forward_return_5"
            ],

        "return_5d_delta":
            return_delta,

        "non_extreme_positive_rate":
            non_metrics[
                "positive_5d_rate"
            ],

        "extreme_positive_rate":
            extreme_metrics[
                "positive_5d_rate"
            ],

        "positive_rate_delta":
            positive_rate_delta,

        "non_extreme_mean_mae":
            non_metrics[
                "mean_forward_5d_mae"
            ],

        "extreme_mean_mae":
            extreme_metrics[
                "mean_forward_5d_mae"
            ],

        "mae_delta":
            mae_delta,

        "non_extreme_mean_mfe":
            non_metrics[
                "mean_forward_5d_mfe"
            ],

        "extreme_mean_mfe":
            extreme_metrics[
                "mean_forward_5d_mfe"
            ],

        "mfe_delta":
            mfe_delta,

        "non_extreme_small_sample":
            non_metrics[
                "n"
            ] < SMALL_SAMPLE_N,

        "extreme_small_sample":
            extreme_metrics[
                "n"
            ] < SMALL_SAMPLE_N,

        "eligible":
            eligible,

        "directionally_supported":
            supported,
    }


def build_period_comparisons(
    technical_q4: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build calendar and chronological comparisons.
    """

    records = []

    schemes = {
        "calendar":
            "calendar_period",

        "chronological":
            "chronological_block",
    }

    for scheme, column in schemes.items():

        for period, group in technical_q4.groupby(
            column,
            observed=True,
            sort=False,
        ):
            records.append(
                build_comparison(
                    group,
                    period_scheme=scheme,
                    period=str(
                        period
                    ),
                )
            )

    return pd.DataFrame(
        records
    )


def build_treasury_breakdown(
    technical_q4: pd.DataFrame,
) -> pd.DataFrame:
    """
    Descriptive Q1/Q2/Q3/Q4 breakdown inside Technical Q4.

    This is not used to choose an individual Treasury threshold.
    """

    records = []

    schemes = {
        "full_sample":
            None,

        "calendar":
            "calendar_period",

        "chronological":
            "chronological_block",
    }

    for scheme, column in schemes.items():

        if column is None:
            groups = [
                (
                    "ALL",
                    technical_q4,
                )
            ]

        else:
            groups = list(
                technical_q4.groupby(
                    column,
                    observed=True,
                    sort=False,
                )
            )

        for period, group in groups:

            for quartile, subset in group.groupby(
                "treasury_quartile",
                observed=True,
                sort=False,
            ):

                row = {
                    "period_scheme":
                        scheme,

                    "period":
                        str(
                            period
                        ),

                    "treasury_quartile":
                        quartile,
                }

                row.update(
                    summarize(
                        subset
                    )
                )

                row[
                    "small_sample_flag"
                ] = (
                    len(
                        subset
                    )
                    < SMALL_SAMPLE_N
                )

                records.append(
                    row
                )

    return pd.DataFrame(
        records
    )


def build_robustness_summary(
    periods: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize support rates by temporal partition scheme.
    """

    records = []

    for scheme, group in periods.groupby(
        "period_scheme",
        observed=True,
        sort=False,
    ):

        eligible = group.loc[
            group[
                "eligible"
            ]
        ]

        supported = eligible.loc[
            eligible[
                "directionally_supported"
            ]
        ]

        records.append(
            {
                "period_scheme":
                    scheme,

                "eligible_periods":
                    len(
                        eligible
                    ),

                "supporting_periods":
                    len(
                        supported
                    ),

                "support_rate":
                    (
                        len(
                            supported
                        )
                        / len(
                            eligible
                        )
                        if len(
                            eligible
                        )
                        else np.nan
                    ),

                "supporting_period_names":
                    "|".join(
                        supported[
                            "period"
                        ].astype(str)
                    ),

                "failing_period_names":
                    "|".join(
                        eligible.loc[
                            ~eligible[
                                "directionally_supported"
                            ],
                            "period",
                        ].astype(str)
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


def main() -> None:
    data = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "market_date",
        ],
    )

    missing = (
        REQUIRED_COLUMNS
        - set(
            data.columns
        )
    )

    if missing:
        raise ValueError(
            "Interaction dataset missing columns: "
            f"{sorted(missing)}"
        )

    data = (
        add_temporal_partitions(
            data
        )
    )

    data[
        "treasury_architecture_state"
    ] = data[
        "treasury_quartile"
    ].map(
        classify_treasury
    )

    technical_q4 = (
        data.loc[
            data[
                "technical_quartile"
            ] == "Q4_HIGH"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if technical_q4.empty:
        raise ValueError(
            "No Technical Q4 observations found."
        )

    overall = pd.DataFrame(
        [
            build_comparison(
                technical_q4,
                period_scheme=(
                    "full_sample"
                ),
                period="ALL",
            )
        ]
    )

    periods = (
        build_period_comparisons(
            technical_q4
        )
    )

    breakdown = (
        build_treasury_breakdown(
            technical_q4
        )
    )

    robustness = (
        build_robustness_summary(
            periods
        )
    )

    DAILY_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    technical_q4.to_csv(
        DAILY_OUTPUT_FILE,
        index=False,
    )

    overall.to_csv(
        OVERALL_OUTPUT_FILE,
        index=False,
    )

    periods.to_csv(
        PERIOD_OUTPUT_FILE,
        index=False,
    )

    breakdown.to_csv(
        BREAKDOWN_OUTPUT_FILE,
        index=False,
    )

    robustness.to_csv(
        ROBUSTNESS_OUTPUT_FILE,
        index=False,
    )

    print("=" * 78)
    print(
        "V4 OPPORTUNITY ENGINE V1 — "
        "FINAL ARCHITECTURE DIAGNOSTIC"
    )
    print("=" * 78)

    print(
        "\nTECHNICAL Q4 SAMPLE"
    )

    print(
        "Observations:",
        len(
            technical_q4
        ),
    )

    print(
        "Date range:",
        technical_q4[
            "market_date"
        ].min().date(),
        "to",
        technical_q4[
            "market_date"
        ].max().date(),
    )

    print(
        "\nTREASURY ARCHITECTURE STATE COUNTS"
    )

    print(
        technical_q4[
            "treasury_architecture_state"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nFULL-SAMPLE PRIMARY CONTRAST"
    )

    print(
        overall.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nTEMPORAL PRIMARY CONTRASTS"
    )

    display_columns = [
        "period_scheme",
        "period",
        "non_extreme_n",
        "extreme_n",
        "non_extreme_mean_return_5",
        "extreme_mean_return_5",
        "return_5d_delta",
        "non_extreme_positive_rate",
        "extreme_positive_rate",
        "non_extreme_mean_mae",
        "extreme_mean_mae",
        "mae_delta",
        "eligible",
        "directionally_supported",
    ]

    print(
        periods[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nFULL-SAMPLE TREASURY QUARTILE BREAKDOWN "
        "WITHIN TECHNICAL Q4"
    )

    full_breakdown = breakdown.loc[
        breakdown[
            "period_scheme"
        ] == "full_sample"
    ]

    breakdown_columns = [
        "treasury_quartile",
        "n",
        "mean_forward_return_5",
        "positive_5d_rate",
        "mean_forward_5d_mae",
        "mean_forward_5d_mfe",
        "small_sample_flag",
    ]

    print(
        full_breakdown[
            breakdown_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nROBUSTNESS SUMMARY"
    )

    print(
        robustness.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print(
        "\nARCHITECTURE INTERPRETATION RULE"
    )

    print(
        "NON_EXTREME is directionally supported "
        "only when it has BOTH higher mean 5d "
        "return and less-negative mean MAE."
    )

    print(
        "This result does not establish a "
        "profitable trading rule."
    )

    print(
        "It determines whether the proposed "
        "ENTRY CANDIDATE versus DO NOT CHASE "
        "architecture is reasonable for forward "
        "testing."
    )

    print(
        "\nSaved:"
    )

    for path in [
        DAILY_OUTPUT_FILE,
        OVERALL_OUTPUT_FILE,
        PERIOD_OUTPUT_FILE,
        BREAKDOWN_OUTPUT_FILE,
        ROBUSTNESS_OUTPUT_FILE,
    ]:
        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()