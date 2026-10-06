"""
Evaluate Macro Regime V1 independently.

The macro score is generated from the existing point-in-time PAYEMS
dataset and then matched to Risk Governor V3 market dates.

Evaluation includes both daily observations and unique macro-state
observations because one PAYEMS state can persist across many trading
days.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.signals.macro_regime import (
    add_macro_regime_features,
)


PAYEMS_FILE = Path(
    "data/features/"
    "TMF_1d_technical_PAYEMS_features.csv"
)

RISK_FILE = Path(
    "reports/"
    "V4_risk_governor_v3_daily.csv"
)

DAILY_OUTPUT_FILE = Path(
    "reports/"
    "V4_macro_regime_v1_daily.csv"
)

SUMMARY_OUTPUT_FILE = Path(
    "reports/"
    "V4_macro_regime_v1_summary.csv"
)

DAILY_QUINTILE_FILE = Path(
    "reports/"
    "V4_macro_regime_v1_daily_quintiles.csv"
)

STATE_QUINTILE_FILE = Path(
    "reports/"
    "V4_macro_regime_v1_state_quintiles.csv"
)


def add_forward_path_metrics(
    data: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """Add close-to-close 5-day MAE and MFE."""

    output = data.copy()

    paths = []

    for step in range(
        1,
        horizon + 1,
    ):
        future_close = (
            output[
                "close"
            ].shift(
                -step
            )
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

    matrix = pd.concat(
        paths,
        axis=1,
    )

    output[
        "forward_5d_mae"
    ] = matrix.min(
        axis=1
    )

    output[
        "forward_5d_mfe"
    ] = matrix.max(
        axis=1
    )

    return output


def add_quintiles(
    data: pd.DataFrame,
    score_column: str,
    output_column: str,
) -> pd.DataFrame:
    """
    Add dynamic quintile labels.

    labels=False avoids failures if repeated macro scores cause qcut
    to collapse duplicate boundaries.
    """

    output = data.copy()

    codes = pd.qcut(
        output[
            score_column
        ],
        q=5,
        labels=False,
        duplicates="drop",
    )

    if codes.notna().any():
        highest = int(
            codes.max()
        )

        def label(code):
            if pd.isna(code):
                return None

            code = int(
                code
            )

            if code == 0:
                return "Q1_LOW"

            if code == highest:
                return (
                    f"Q{code + 1}_HIGH"
                )

            return f"Q{code + 1}"

        output[
            output_column
        ] = codes.map(
            label
        )

    else:
        output[
            output_column
        ] = None

    return output


def summarize_sample(
    data: pd.DataFrame,
    name: str,
) -> dict:
    """Summarize one daily macro sample."""

    return {
        "sample":
            name,

        "n":
            len(data),

        "macro_states":
            int(
                data[
                    "payroll_state_id"
                ].nunique()
            ),

        "mean_score":
            data[
                "macro_regime_score"
            ].mean(),

        "median_score":
            data[
                "macro_regime_score"
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
    payems = pd.read_csv(
        PAYEMS_FILE,
        parse_dates=[
            "market_date",
        ],
    )

    risk = pd.read_csv(
        RISK_FILE,
        parse_dates=[
            "market_date",
        ],
    )

    macro = (
        add_macro_regime_features(
            payems
        )
    )

    macro_columns = [
        "market_date",
        "payroll_state_market_date",
        "payroll_state_id",
        "payroll_change_1m",
        "payroll_revision_total",
        "payroll_broad_revision_event",
        "macro_payroll_growth_score",
        "macro_payroll_revision_score",
        "macro_broad_revision_score",
        "macro_regime_score",
    ]

    macro = macro[
        macro_columns
    ].copy()

    data = risk.merge(
        macro,
        on="market_date",
        how="left",
        validate="one_to_one",
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
        add_forward_path_metrics(
            data
        )
    )

    eligible = (
        data[
            "macro_regime_score"
        ].notna()
        & data[
            "forward_return_5"
        ].notna()
    )

    evaluation = (
        data.loc[
            eligible
        ]
        .copy()
    )

    entry_open = (
        evaluation[
            "v3_new_entries_allowed"
        ].astype(bool)
    )

    open_sample = (
        evaluation.loc[
            entry_open
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

    daily_quintiles = (
        add_quintiles(
            open_sample,
            score_column=(
                "macro_regime_score"
            ),
            output_column=(
                "macro_score_quintile"
            ),
        )
    )

    daily_quintile_summary = (
        daily_quintiles.groupby(
            "macro_score_quintile",
            observed=True,
            as_index=False,
        )
        .agg(
            n=(
                "market_date",
                "size",
            ),

            macro_states=(
                "payroll_state_id",
                "nunique",
            ),

            mean_score=(
                "macro_regime_score",
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

            mean_forward_5d_mfe=(
                "forward_5d_mfe",
                "mean",
            ),
        )
    )

    # Each PAYEMS state receives equal weight here regardless of how
    # many trading days that state remained active.
    state_sample = (
        open_sample.groupby(
            "payroll_state_id",
            as_index=False,
        )
        .agg(
            state_start=(
                "payroll_state_market_date",
                "first",
            ),

            macro_regime_score=(
                "macro_regime_score",
                "first",
            ),

            open_trading_days=(
                "market_date",
                "size",
            ),

            mean_forward_return_5=(
                "forward_return_5",
                "mean",
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

            mean_forward_5d_mfe=(
                "forward_5d_mfe",
                "mean",
            ),
        )
    )

    state_sample = (
        add_quintiles(
            state_sample,
            score_column=(
                "macro_regime_score"
            ),
            output_column=(
                "macro_score_quintile"
            ),
        )
    )

    state_quintile_summary = (
        state_sample.groupby(
            "macro_score_quintile",
            observed=True,
            as_index=False,
        )
        .agg(
            macro_states=(
                "payroll_state_id",
                "size",
            ),

            mean_score=(
                "macro_regime_score",
                "mean",
            ),

            mean_state_forward_return_5=(
                "mean_forward_return_5",
                "mean",
            ),

            mean_state_positive_5d_rate=(
                "positive_5d_rate",
                "mean",
            ),

            mean_state_forward_5d_mae=(
                "mean_forward_5d_mae",
                "mean",
            ),

            mean_state_forward_5d_mfe=(
                "mean_forward_5d_mfe",
                "mean",
            ),
        )
    )

    DAILY_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        DAILY_OUTPUT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
    )

    daily_quintile_summary.to_csv(
        DAILY_QUINTILE_FILE,
        index=False,
    )

    state_quintile_summary.to_csv(
        STATE_QUINTILE_FILE,
        index=False,
    )

    print("=" * 78)
    print(
        "V4 OPPORTUNITY ENGINE V1 — MACRO REGIME COMPONENT"
    )
    print("=" * 78)

    print(
        "\nSCORE COVERAGE"
    )

    print(
        "Total market rows:",
        len(data),
    )

    print(
        "Complete macro scores:",
        int(
            data[
                "macro_regime_score"
            ].notna().sum()
        ),
    )

    print(
        "Scored PAYEMS states:",
        int(
            data.loc[
                data[
                    "macro_regime_score"
                ].notna(),
                "payroll_state_id",
            ].nunique()
        ),
    )

    scored_dates = data.loc[
        data[
            "macro_regime_score"
        ].notna(),
        "market_date",
    ]

    if not scored_dates.empty:
        print(
            "First scored market date:",
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
            "macro_regime_score"
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
        "\nENTRY-OPEN DAILY QUINTILES"
    )

    print(
        daily_quintile_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nENTRY-OPEN MACRO-STATE QUINTILES"
    )

    print(
        state_quintile_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nImportant:"
    )

    print(
        "Daily rows are relevant to trading exposure "
        "but are not independent macro observations."
    )

    print(
        "The macro-state table gives each PAYEMS "
        "state equal weight."
    )

    print(
        "Do not choose a macro threshold from these "
        "historical results."
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
        f"  {DAILY_QUINTILE_FILE}"
    )

    print(
        f"  {STATE_QUINTILE_FILE}"
    )


if __name__ == "__main__":
    main()