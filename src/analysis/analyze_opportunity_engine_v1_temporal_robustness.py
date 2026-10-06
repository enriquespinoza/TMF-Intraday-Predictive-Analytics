"""
Opportunity Engine V1 temporal robustness diagnostic.

Tests whether relationships found in the frozen interaction diagnostic
persist across calendar periods and equal-count chronological blocks.

This is descriptive stability research, not a predictive model and not
independent out-of-sample validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "reports/"
    "V4_opportunity_engine_v1_interaction_daily.csv"
)

BASELINE_OUTPUT_FILE = Path(
    "reports/"
    "V4_opportunity_engine_v1_temporal_baselines.csv"
)

CELL_OUTPUT_FILE = Path(
    "reports/"
    "V4_opportunity_engine_v1_temporal_role_cells.csv"
)

CONTRAST_OUTPUT_FILE = Path(
    "reports/"
    "V4_opportunity_engine_v1_temporal_contrasts.csv"
)

SUMMARY_OUTPUT_FILE = Path(
    "reports/"
    "V4_opportunity_engine_v1_temporal_robustness_summary.csv"
)


MIN_SINGLE_CELL_N = 10
MIN_CONTRAST_CELL_N = 5


REQUIRED_COLUMNS = {
    "market_date",
    "forward_return_1",
    "forward_return_5",
    "forward_5d_mae",
    "forward_5d_mfe",
    "technical_quartile",
    "treasury_quartile",
    "macro_quartile",
    "market_quartile",
}


CELL_DEFINITIONS = [
    {
        "role": "technical_q4",
        "cell": "TECH_Q4",
        "filters": {
            "technical_quartile": "Q4_HIGH",
        },
    },

    {
        "role": "treasury_q2_vs_q4",
        "cell": "TREASURY_Q2",
        "filters": {
            "treasury_quartile": "Q2",
        },
    },

    {
        "role": "treasury_q2_vs_q4",
        "cell": "TREASURY_Q4",
        "filters": {
            "treasury_quartile": "Q4_HIGH",
        },
    },

    {
        "role": "technical_market",
        "cell": "TECH_Q4_MARKET_Q1",
        "filters": {
            "technical_quartile": "Q4_HIGH",
            "market_quartile": "Q1_LOW",
        },
    },

    {
        "role": "technical_market",
        "cell": "TECH_Q1_MARKET_Q4",
        "filters": {
            "technical_quartile": "Q1_LOW",
            "market_quartile": "Q4_HIGH",
        },
    },

    {
        "role": "technical_macro",
        "cell": "TECH_Q4_MACRO_Q1",
        "filters": {
            "technical_quartile": "Q4_HIGH",
            "macro_quartile": "Q1_LOW",
        },
    },

    {
        "role": "technical_macro",
        "cell": "TECH_Q4_MACRO_Q4",
        "filters": {
            "technical_quartile": "Q4_HIGH",
            "macro_quartile": "Q4_HIGH",
        },
    },

    {
        "role": "technical_macro",
        "cell": "TECH_Q1_MACRO_Q4",
        "filters": {
            "technical_quartile": "Q1_LOW",
            "macro_quartile": "Q4_HIGH",
        },
    },

    {
        "role": "technical_macro",
        "cell": "TECH_Q1_MACRO_Q1",
        "filters": {
            "technical_quartile": "Q1_LOW",
            "macro_quartile": "Q1_LOW",
        },
    },

    {
        "role": "macro_market",
        "cell": "MACRO_Q4_MARKET_Q4",
        "filters": {
            "macro_quartile": "Q4_HIGH",
            "market_quartile": "Q4_HIGH",
        },
    },

    {
        "role": "macro_market",
        "cell": "MACRO_Q1_MARKET_Q4",
        "filters": {
            "macro_quartile": "Q1_LOW",
            "market_quartile": "Q4_HIGH",
        },
    },
]


CONTRAST_DEFINITIONS = [
    {
        "hypothesis": "TREASURY_Q2_BEATS_Q4",
        "preferred_cell": "TREASURY_Q2",
        "comparison_cell": "TREASURY_Q4",
    },

    {
        "hypothesis": "TECH_Q4_MARKET_Q1_BEATS_TECH_Q1_MARKET_Q4",
        "preferred_cell": "TECH_Q4_MARKET_Q1",
        "comparison_cell": "TECH_Q1_MARKET_Q4",
    },

    {
        "hypothesis": "TECH_Q4_MACRO_Q1_BEATS_TECH_Q4_MACRO_Q4",
        "preferred_cell": "TECH_Q4_MACRO_Q1",
        "comparison_cell": "TECH_Q4_MACRO_Q4",
    },

    {
        "hypothesis": "TECH_Q1_MACRO_Q4_BEATS_TECH_Q1_MACRO_Q1",
        "preferred_cell": "TECH_Q1_MACRO_Q4",
        "comparison_cell": "TECH_Q1_MACRO_Q1",
    },

    {
        "hypothesis": "MACRO_Q4_MARKET_Q4_BEATS_MACRO_Q1_MARKET_Q4",
        "preferred_cell": "MACRO_Q4_MARKET_Q4",
        "comparison_cell": "MACRO_Q1_MARKET_Q4",
    },
]


def summarize(
    data: pd.DataFrame,
) -> dict:
    """Common outcome statistics."""

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


def calendar_period(
    timestamp: pd.Timestamp,
) -> str:
    """Fixed calendar partitions for the current research sample."""

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
    Add calendar periods and four approximately equal chronological
    blocks.

    No target information is used.
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


def filter_cell(
    data: pd.DataFrame,
    filters: dict[str, str],
) -> pd.DataFrame:
    """Apply exact quartile filters."""

    mask = pd.Series(
        True,
        index=data.index,
    )

    for column, value in filters.items():
        mask &= (
            data[
                column
            ] == value
        )

    return data.loc[
        mask
    ].copy()


def build_period_baselines(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Build period-specific baselines."""

    records = []

    schemes = {
        "calendar":
            "calendar_period",

        "chronological":
            "chronological_block",
    }

    for scheme_name, period_column in schemes.items():

        for period, group in data.groupby(
            period_column,
            observed=True,
            sort=False,
        ):
            row = {
                "period_scheme":
                    scheme_name,

                "period":
                    str(period),
            }

            row.update(
                summarize(
                    group
                )
            )

            records.append(
                row
            )

    return pd.DataFrame(
        records
    )


def build_role_cells(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate every pre-specified role cell inside each time period.
    """

    records = []

    schemes = {
        "calendar":
            "calendar_period",

        "chronological":
            "chronological_block",
    }

    for scheme_name, period_column in schemes.items():

        for period, group in data.groupby(
            period_column,
            observed=True,
            sort=False,
        ):
            baseline = summarize(
                group
            )

            for definition in CELL_DEFINITIONS:

                subset = filter_cell(
                    group,
                    definition[
                        "filters"
                    ],
                )

                metrics = summarize(
                    subset
                )

                row = {
                    "period_scheme":
                        scheme_name,

                    "period":
                        str(period),

                    "role":
                        definition[
                            "role"
                        ],

                    "cell":
                        definition[
                            "cell"
                        ],
                }

                row.update(
                    metrics
                )

                row[
                    "return_5d_lift_vs_period"
                ] = (
                    metrics[
                        "mean_forward_return_5"
                    ]
                    - baseline[
                        "mean_forward_return_5"
                    ]
                )

                row[
                    "positive_rate_lift_vs_period"
                ] = (
                    metrics[
                        "positive_5d_rate"
                    ]
                    - baseline[
                        "positive_5d_rate"
                    ]
                )

                row[
                    "mae_lift_vs_period"
                ] = (
                    metrics[
                        "mean_forward_5d_mae"
                    ]
                    - baseline[
                        "mean_forward_5d_mae"
                    ]
                )

                row[
                    "mfe_lift_vs_period"
                ] = (
                    metrics[
                        "mean_forward_5d_mfe"
                    ]
                    - baseline[
                        "mean_forward_5d_mfe"
                    ]
                )

                row[
                    "small_sample_flag"
                ] = (
                    metrics[
                        "n"
                    ]
                    < 10
                )

                row[
                    "critical_sample_flag"
                ] = (
                    metrics[
                        "n"
                    ]
                    < 5
                )

                records.append(
                    row
                )

    return pd.DataFrame(
        records
    )


def build_contrasts(
    cells: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare the pre-specified preferred cell with its comparison cell.

    Positive return_delta means the preferred cell had higher return.

    Positive mae_delta means the preferred cell had less-negative MAE.
    """

    records = []

    period_keys = (
        cells[
            [
                "period_scheme",
                "period",
            ]
        ]
        .drop_duplicates()
    )

    for _, period_key in period_keys.iterrows():

        scheme = period_key[
            "period_scheme"
        ]

        period = period_key[
            "period"
        ]

        period_cells = cells.loc[
            (
                cells[
                    "period_scheme"
                ] == scheme
            )
            & (
                cells[
                    "period"
                ] == period
            )
        ]

        for definition in CONTRAST_DEFINITIONS:

            preferred_rows = period_cells.loc[
                period_cells[
                    "cell"
                ]
                == definition[
                    "preferred_cell"
                ]
            ]

            comparison_rows = period_cells.loc[
                period_cells[
                    "cell"
                ]
                == definition[
                    "comparison_cell"
                ]
            ]

            if (
                preferred_rows.empty
                or comparison_rows.empty
            ):
                continue

            preferred = (
                preferred_rows.iloc[0]
            )

            comparison = (
                comparison_rows.iloc[0]
            )

            eligible = (
                preferred[
                    "n"
                ]
                >= MIN_CONTRAST_CELL_N
                and comparison[
                    "n"
                ]
                >= MIN_CONTRAST_CELL_N
            )

            return_delta = (
                preferred[
                    "mean_forward_return_5"
                ]
                - comparison[
                    "mean_forward_return_5"
                ]
            )

            positive_rate_delta = (
                preferred[
                    "positive_5d_rate"
                ]
                - comparison[
                    "positive_5d_rate"
                ]
            )

            mae_delta = (
                preferred[
                    "mean_forward_5d_mae"
                ]
                - comparison[
                    "mean_forward_5d_mae"
                ]
            )

            mfe_delta = (
                preferred[
                    "mean_forward_5d_mfe"
                ]
                - comparison[
                    "mean_forward_5d_mfe"
                ]
            )

            supported = (
                bool(
                    eligible
                    and return_delta > 0
                    and mae_delta > 0
                )
            )

            records.append(
                {
                    "period_scheme":
                        scheme,

                    "period":
                        period,

                    "hypothesis":
                        definition[
                            "hypothesis"
                        ],

                    "preferred_cell":
                        definition[
                            "preferred_cell"
                        ],

                    "comparison_cell":
                        definition[
                            "comparison_cell"
                        ],

                    "preferred_n":
                        int(
                            preferred[
                                "n"
                            ]
                        ),

                    "comparison_n":
                        int(
                            comparison[
                                "n"
                            ]
                        ),

                    "return_5d_delta":
                        return_delta,

                    "positive_rate_delta":
                        positive_rate_delta,

                    "mae_delta":
                        mae_delta,

                    "mfe_delta":
                        mfe_delta,

                    "eligible":
                        eligible,

                    "directionally_supported":
                        supported,
                }
            )

    return pd.DataFrame(
        records
    )


def build_robustness_summary(
    cells: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize temporal consistency of the pre-specified roles."""

    records = []

    schemes = [
        "calendar",
        "chronological",
    ]

    for scheme in schemes:

        # Hypothesis 1:
        # Technical Q4 relative to its own period baseline.
        tech = cells.loc[
            (
                cells[
                    "period_scheme"
                ] == scheme
            )
            & (
                cells[
                    "cell"
                ] == "TECH_Q4"
            )
        ].copy()

        tech[
            "eligible"
        ] = (
            tech[
                "n"
            ] >= MIN_SINGLE_CELL_N
        )

        tech[
            "supported"
        ] = (
            tech[
                "eligible"
            ]
            & (
                tech[
                    "return_5d_lift_vs_period"
                ] > 0
            )
            & (
                tech[
                    "mae_lift_vs_period"
                ] > 0
            )
        )

        eligible = tech.loc[
            tech[
                "eligible"
            ]
        ]

        supported = eligible.loc[
            eligible[
                "supported"
            ]
        ]

        records.append(
            {
                "period_scheme":
                    scheme,

                "hypothesis":
                    "TECH_Q4_BEATS_PERIOD_BASELINE",

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
            }
        )

        contrast_subset = contrasts.loc[
            contrasts[
                "period_scheme"
            ] == scheme
        ]

        for hypothesis, group in contrast_subset.groupby(
            "hypothesis",
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

                    "hypothesis":
                        hypothesis,

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
                }
            )

    return pd.DataFrame(
        records
    )


def print_cells(
    cells: pd.DataFrame,
    cell_names: list[str],
    title: str,
) -> None:
    """Print selected role cells."""

    print(
        f"\n{title}"
    )

    subset = cells.loc[
        cells[
            "cell"
        ].isin(
            cell_names
        )
    ]

    columns = [
        "period_scheme",
        "period",
        "cell",
        "n",
        "mean_forward_return_5",
        "positive_5d_rate",
        "mean_forward_5d_mae",
        "mean_forward_5d_mfe",
        "return_5d_lift_vs_period",
        "mae_lift_vs_period",
        "small_sample_flag",
        "critical_sample_flag",
    ]

    print(
        subset[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
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
        data
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )

    data = (
        add_temporal_partitions(
            data
        )
    )

    baselines = (
        build_period_baselines(
            data
        )
    )

    cells = (
        build_role_cells(
            data
        )
    )

    contrasts = (
        build_contrasts(
            cells
        )
    )

    robustness = (
        build_robustness_summary(
            cells,
            contrasts,
        )
    )

    BASELINE_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    baselines.to_csv(
        BASELINE_OUTPUT_FILE,
        index=False,
    )

    cells.to_csv(
        CELL_OUTPUT_FILE,
        index=False,
    )

    contrasts.to_csv(
        CONTRAST_OUTPUT_FILE,
        index=False,
    )

    robustness.to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
    )

    print("=" * 78)
    print(
        "V4 OPPORTUNITY ENGINE V1 — "
        "TEMPORAL ROBUSTNESS DIAGNOSTIC"
    )
    print("=" * 78)

    print(
        "\nCOMMON SAMPLE"
    )

    print(
        "Rows:",
        len(
            data
        ),
    )

    print(
        "Date range:",
        data[
            "market_date"
        ].min().date(),
        "to",
        data[
            "market_date"
        ].max().date(),
    )

    print(
        "\nPERIOD COUNTS"
    )

    print(
        "\nCalendar:"
    )

    print(
        data[
            "calendar_period"
        ]
        .value_counts(
            sort=False
        )
        .to_string()
    )

    print(
        "\nChronological blocks:"
    )

    print(
        data[
            "chronological_block"
        ]
        .value_counts(
            sort=False
        )
        .to_string()
    )

    print(
        "\nPERIOD BASELINES"
    )

    print(
        baselines.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print_cells(
        cells,
        [
            "TECH_Q4",
        ],
        "TECHNICAL Q4 VS PERIOD BASELINE",
    )

    print_cells(
        cells,
        [
            "TREASURY_Q2",
            "TREASURY_Q4",
        ],
        "TREASURY Q2 VS Q4",
    )

    print_cells(
        cells,
        [
            "TECH_Q4_MARKET_Q1",
            "TECH_Q1_MARKET_Q4",
        ],
        "TECHNICAL × MARKET",
    )

    print_cells(
        cells,
        [
            "TECH_Q4_MACRO_Q1",
            "TECH_Q4_MACRO_Q4",
            "TECH_Q1_MACRO_Q4",
            "TECH_Q1_MACRO_Q1",
        ],
        "TECHNICAL × MACRO",
    )

    print_cells(
        cells,
        [
            "MACRO_Q4_MARKET_Q4",
            "MACRO_Q1_MARKET_Q4",
        ],
        "MACRO × MARKET",
    )

    print(
        "\nPAIRWISE CONTRASTS"
    )

    contrast_columns = [
        "period_scheme",
        "period",
        "hypothesis",
        "preferred_n",
        "comparison_n",
        "return_5d_delta",
        "positive_rate_delta",
        "mae_delta",
        "mfe_delta",
        "eligible",
        "directionally_supported",
    ]

    print(
        contrasts[
            contrast_columns
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
        "\nImportant:"
    )

    print(
        "This tests temporal stability of "
        "previously discovered relationships."
    )

    print(
        "It is not independent out-of-sample "
        "validation."
    )

    print(
        "A relationship is counted as supported "
        "only when return AND MAE move in the "
        "pre-specified favorable direction."
    )

    print(
        "Periods with insufficient cell counts "
        "are excluded from robustness support rates."
    )

    print(
        "\nSaved:"
    )

    for path in [
        BASELINE_OUTPUT_FILE,
        CELL_OUTPUT_FILE,
        CONTRAST_OUTPUT_FILE,
        SUMMARY_OUTPUT_FILE,
    ]:
        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()