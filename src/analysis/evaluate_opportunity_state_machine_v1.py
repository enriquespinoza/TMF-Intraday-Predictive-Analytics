"""
Evaluate the frozen Opportunity Engine V1 state machine.

This script:

1. Rebuilds the daily component dataset.
2. Applies the frozen Opportunity Engine V1 policy chronologically.
3. Measures state frequency.
4. Measures forward return / MAE / MFE by state.
5. Measures state transitions.
6. Reports annual and research-period behavior.

It does NOT:

- optimize thresholds
- modify the state machine
- simulate position sizing
- calculate trading P&L
- create new entry rules

The state machine must remain frozen after this evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.signals.opportunity_state_machine import (
    OpportunityInputsV1,
    evaluate_opportunity_v1,
    load_policy_v1,
)


RISK_FILE = Path(
    "reports/"
    "V4_risk_governor_v3_daily.csv"
)

TECHNICAL_FILE = Path(
    "reports/"
    "V4_technical_opportunity_v1_daily.csv"
)

TREASURY_FILE = Path(
    "reports/"
    "V4_treasury_confirmation_v1_daily.csv"
)

MACRO_FILE = Path(
    "reports/"
    "V4_macro_regime_v1_daily.csv"
)

MARKET_FILE = Path(
    "reports/"
    "V4_market_confirmation_v1_daily.csv"
)

POLICY_FILE = Path(
    "config/"
    "opportunity_engine_v1_policy.json"
)


DAILY_OUTPUT_FILE = Path(
    "reports/"
    "V4_opportunity_state_machine_v1_daily.csv"
)

SUMMARY_OUTPUT_FILE = Path(
    "reports/"
    "V4_opportunity_state_machine_v1_summary.csv"
)

TRANSITION_OUTPUT_FILE = Path(
    "reports/"
    "V4_opportunity_state_machine_v1_transitions.csv"
)

ANNUAL_OUTPUT_FILE = Path(
    "reports/"
    "V4_opportunity_state_machine_v1_annual.csv"
)

RESEARCH_PERIOD_OUTPUT_FILE = Path(
    "reports/"
    "V4_opportunity_state_machine_v1_research_periods.csv"
)


def parse_bool_value(
    value,
) -> bool:
    """
    Parse booleans safely.

    Important:
    bool("False") is True in Python, so CSV strings must not
    be converted with plain bool().
    """

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(
            value
        )

    if pd.isna(
        value
    ):
        raise ValueError(
            "Boolean value is missing."
        )

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        if value in {
            0,
            0.0,
        }:
            return False

        if value in {
            1,
            1.0,
        }:
            return True

    text = (
        str(
            value
        )
        .strip()
        .lower()
    )

    if text in {
        "true",
        "1",
        "yes",
        "y",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        "n",
    }:
        return False

    raise ValueError(
        f"Unable to parse boolean value: {value!r}"
    )


def optional_float(
    value,
):
    """Convert NaN to None for optional component scores."""

    if pd.isna(
        value
    ):
        return None

    return float(
        value
    )


def load_score(
    path: Path,
    score_column: str,
) -> pd.DataFrame:
    """Load one component score by market date."""

    data = pd.read_csv(
        path,
        parse_dates=[
            "market_date",
        ],
    )

    required = {
        "market_date",
        score_column,
    }

    missing = (
        required
        - set(
            data.columns
        )
    )

    if missing:
        raise ValueError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    output = (
        data[
            [
                "market_date",
                score_column,
            ]
        ]
        .copy()
        .sort_values(
            "market_date"
        )
    )

    if output[
        "market_date"
    ].duplicated().any():
        raise ValueError(
            f"Duplicate market dates in {path}."
        )

    return output


def build_state_dataset() -> pd.DataFrame:
    """Build one point-in-time daily dataset."""

    risk = pd.read_csv(
        RISK_FILE,
        parse_dates=[
            "market_date",
        ],
    )

    required = {
        "market_date",
        "close",
        "forward_return_1",
        "forward_return_5",
        "v3_new_entries_allowed",
    }

    missing = (
        required
        - set(
            risk.columns
        )
    )

    if missing:
        raise ValueError(
            "Risk dataset missing columns: "
            f"{sorted(missing)}"
        )

    if risk[
        "market_date"
    ].duplicated().any():
        raise ValueError(
            "Risk dataset contains duplicate dates."
        )

    data = (
        risk.copy()
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )

    sources = [
        (
            TECHNICAL_FILE,
            "technical_opportunity_score",
        ),
        (
            TREASURY_FILE,
            "treasury_confirmation_score",
        ),
        (
            MACRO_FILE,
            "macro_regime_score",
        ),
        (
            MARKET_FILE,
            "market_confirmation_score",
        ),
    ]

    for path, score_column in sources:

        component = load_score(
            path,
            score_column,
        )

        data = data.merge(
            component,
            on="market_date",
            how="left",
            validate="one_to_one",
        )

    return (
        data
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )


def add_forward_path_metrics(
    data: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """Add future close-to-close MAE and MFE."""

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
            .shift(
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


def load_policy_dates() -> tuple[
    pd.Timestamp,
    pd.Timestamp,
]:
    """Load calibration dates recorded in the frozen policy."""

    with open(
        POLICY_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        policy = json.load(
            file
        )

    return (
        pd.Timestamp(
            policy[
                "calibration_start"
            ]
        ),
        pd.Timestamp(
            policy[
                "calibration_end"
            ]
        ),
    )


def classify_research_period(
    market_date: pd.Timestamp,
    calibration_start: pd.Timestamp,
    calibration_end: pd.Timestamp,
) -> str:
    """
    Label observations relative to the threshold calibration window.

    POST_CALIBRATION is descriptive only.
    It is NOT claimed to be a fully independent OOS sample.
    """

    if market_date < calibration_start:
        return "PRE_CALIBRATION"

    if market_date <= calibration_end:
        return "CALIBRATION_WINDOW"

    return "POST_CALIBRATION_DESCRIPTIVE"


def apply_state_machine(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Apply frozen Opportunity Engine V1 chronologically."""

    output = data.copy()

    policy = load_policy_v1()

    states = []
    entry_candidates = []
    risk_blocked_values = []
    technical_strong_values = []
    treasury_extreme_values = []
    reason_values = []

    for _, row in output.iterrows():

        entries_allowed = parse_bool_value(
            row[
                "v3_new_entries_allowed"
            ]
        )

        inputs = OpportunityInputsV1(
            risk_entry_blocked=(
                not entries_allowed
            ),

            technical_opportunity_score=(
                optional_float(
                    row[
                        "technical_opportunity_score"
                    ]
                )
            ),

            treasury_confirmation_score=(
                optional_float(
                    row[
                        "treasury_confirmation_score"
                    ]
                )
            ),

            macro_regime_score=(
                optional_float(
                    row[
                        "macro_regime_score"
                    ]
                )
            ),

            market_confirmation_score=(
                optional_float(
                    row[
                        "market_confirmation_score"
                    ]
                )
            ),
        )

        decision = evaluate_opportunity_v1(
            inputs,
            policy,
        )

        states.append(
            decision.state.value
        )

        entry_candidates.append(
            decision.entry_candidate
        )

        risk_blocked_values.append(
            decision.risk_entry_blocked
        )

        technical_strong_values.append(
            decision.technical_strong_setup
        )

        treasury_extreme_values.append(
            decision.treasury_extreme_confirmation
        )

        reason_values.append(
            " | ".join(
                decision.reasons
            )
        )

    output[
        "opportunity_state"
    ] = states

    output[
        "entry_candidate"
    ] = entry_candidates

    output[
        "risk_entry_blocked"
    ] = risk_blocked_values

    output[
        "technical_strong_setup"
    ] = technical_strong_values

    output[
        "treasury_extreme_confirmation"
    ] = treasury_extreme_values

    output[
        "opportunity_reasons"
    ] = reason_values

    output[
        "required_scores_complete"
    ] = (
        output[
            "technical_opportunity_score"
        ].notna()
        & output[
            "treasury_confirmation_score"
        ].notna()
    )

    calibration_start, calibration_end = (
        load_policy_dates()
    )

    output[
        "research_period"
    ] = output[
        "market_date"
    ].map(
        lambda date:
            classify_research_period(
                date,
                calibration_start,
                calibration_end,
            )
    )

    output[
        "calendar_year"
    ] = output[
        "market_date"
    ].dt.year

    return output


def summarize_group(
    group: pd.DataFrame,
) -> dict:
    """Summarize state outcomes."""

    return {
        "n":
            len(
                group
            ),

        "mean_forward_return_1":
            group[
                "forward_return_1"
            ].mean(),

        "mean_forward_return_5":
            group[
                "forward_return_5"
            ].mean(),

        "median_forward_return_5":
            group[
                "forward_return_5"
            ].median(),

        "positive_5d_rate":
            (
                group[
                    "forward_return_5"
                ] > 0
            ).mean(),

        "mean_forward_5d_mae":
            group[
                "forward_5d_mae"
            ].mean(),

        "median_forward_5d_mae":
            group[
                "forward_5d_mae"
            ].median(),

        "mean_forward_5d_mfe":
            group[
                "forward_5d_mfe"
            ].mean(),

        "median_forward_5d_mfe":
            group[
                "forward_5d_mfe"
            ].median(),
    }


def build_state_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize outcomes by state."""

    valid = data.loc[
        data[
            "required_scores_complete"
        ]
        & data[
            "forward_return_5"
        ].notna()
    ].copy()

    records = []

    total = len(
        valid
    )

    for state, group in valid.groupby(
        "opportunity_state",
        observed=True,
        sort=False,
    ):

        row = {
            "opportunity_state":
                state,

            "share":
                (
                    len(
                        group
                    )
                    / total
                    if total
                    else np.nan
                ),
        }

        row.update(
            summarize_group(
                group
            )
        )

        records.append(
            row
        )

    return pd.DataFrame(
        records
    )


def build_transition_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Build chronological state-transition counts and probabilities."""

    valid = (
        data.loc[
            data[
                "required_scores_complete"
            ]
        ]
        .copy()
        .sort_values(
            "market_date"
        )
    )

    valid[
        "previous_state"
    ] = valid[
        "opportunity_state"
    ].shift(
        1
    )

    transitions = valid.loc[
        valid[
            "previous_state"
        ].notna()
    ]

    result = (
        transitions.groupby(
            [
                "previous_state",
                "opportunity_state",
            ],
            observed=True,
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "transition_count",
            }
        )
    )

    totals = (
        result.groupby(
            "previous_state"
        )[
            "transition_count"
        ]
        .transform(
            "sum"
        )
    )

    result[
        "transition_probability"
    ] = (
        result[
            "transition_count"
        ]
        / totals
    )

    return result


def build_annual_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """State performance by calendar year."""

    valid = data.loc[
        data[
            "required_scores_complete"
        ]
        & data[
            "forward_return_5"
        ].notna()
    ].copy()

    records = []

    for (
        year,
        state,
    ), group in valid.groupby(
        [
            "calendar_year",
            "opportunity_state",
        ],
        observed=True,
        sort=False,
    ):

        row = {
            "calendar_year":
                int(
                    year
                ),

            "opportunity_state":
                state,
        }

        row.update(
            summarize_group(
                group
            )
        )

        records.append(
            row
        )

    return pd.DataFrame(
        records
    )


def build_research_period_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """State performance relative to calibration period."""

    valid = data.loc[
        data[
            "required_scores_complete"
        ]
        & data[
            "forward_return_5"
        ].notna()
    ].copy()

    records = []

    for (
        period,
        state,
    ), group in valid.groupby(
        [
            "research_period",
            "opportunity_state",
        ],
        observed=True,
        sort=False,
    ):

        row = {
            "research_period":
                period,

            "opportunity_state":
                state,
        }

        row.update(
            summarize_group(
                group
            )
        )

        records.append(
            row
        )

    return pd.DataFrame(
        records
    )


def main() -> None:

    data = build_state_dataset()

    data = add_forward_path_metrics(
        data
    )

    data = apply_state_machine(
        data
    )

    summary = build_state_summary(
        data
    )

    transitions = build_transition_table(
        data
    )

    annual = build_annual_summary(
        data
    )

    research_periods = (
        build_research_period_summary(
            data
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

    transitions.to_csv(
        TRANSITION_OUTPUT_FILE,
        index=False,
    )

    annual.to_csv(
        ANNUAL_OUTPUT_FILE,
        index=False,
    )

    research_periods.to_csv(
        RESEARCH_PERIOD_OUTPUT_FILE,
        index=False,
    )

    complete = data.loc[
        data[
            "required_scores_complete"
        ]
    ]

    valid_target = complete.loc[
        complete[
            "forward_return_5"
        ].notna()
    ]

    print("=" * 78)
    print(
        "V4 OPPORTUNITY ENGINE V1 — "
        "FROZEN STATE-MACHINE EVALUATION"
    )
    print("=" * 78)

    print(
        "\nCOVERAGE"
    )

    print(
        "Total market rows:",
        len(
            data
        ),
    )

    print(
        "Required-score complete rows:",
        len(
            complete
        ),
    )

    print(
        "Valid 5d evaluation rows:",
        len(
            valid_target
        ),
    )

    if not complete.empty:

        print(
            "First complete date:",
            complete[
                "market_date"
            ].min().date(),
        )

        print(
            "Last complete date:",
            complete[
                "market_date"
            ].max().date(),
        )

    print(
        "\nSTATE COUNTS — COMPLETE SCORE ROWS"
    )

    state_counts = (
        complete[
            "opportunity_state"
        ]
        .value_counts()
        .rename_axis(
            "opportunity_state"
        )
        .reset_index(
            name="n"
        )
    )

    state_counts[
        "share"
    ] = (
        state_counts[
            "n"
        ]
        / len(
            complete
        )
    )

    print(
        state_counts.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nSTATE OUTCOME SUMMARY"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nSTATE TRANSITIONS"
    )

    print(
        transitions.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nANNUAL STATE SUMMARY"
    )

    print(
        annual.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nRESEARCH-PERIOD SUMMARY"
    )

    print(
        research_periods.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nImportant:"
    )

    print(
        "This evaluates the frozen state machine "
        "descriptively."
    )

    print(
        "No P&L or execution simulation is being "
        "performed yet."
    )

    print(
        "POST_CALIBRATION_DESCRIPTIVE is not "
        "claimed to be fully independent OOS evidence."
    )

    print(
        "The true forward test begins only after "
        "the V1 architecture is frozen."
    )

    print(
        "\nSaved:"
    )

    for path in [
        DAILY_OUTPUT_FILE,
        SUMMARY_OUTPUT_FILE,
        TRANSITION_OUTPUT_FILE,
        ANNUAL_OUTPUT_FILE,
        RESEARCH_PERIOD_OUTPUT_FILE,
    ]:

        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()