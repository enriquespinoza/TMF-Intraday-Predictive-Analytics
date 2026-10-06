"""
Freeze Opportunity Engine V1 score boundaries.

The research interaction dataset already contains frozen Q1-Q4 labels.

This script converts the Q4 boundaries into fixed numeric score
thresholds for future state-machine and forward-test use.

Future observations must NOT recalculate quartiles from an expanding
or full-history sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "reports/"
    "V4_opportunity_engine_v1_interaction_daily.csv"
)

OUTPUT_FILE = Path(
    "config/"
    "opportunity_engine_v1_policy.json"
)


def derive_q4_boundary(
    data: pd.DataFrame,
    score_column: str,
    quartile_column: str,
) -> dict:
    """
    Derive a fixed numeric boundary reproducing the frozen Q4 labels.

    Requires a clean gap between Q1-Q3 scores and Q4 scores.
    """

    q4 = data.loc[
        data[quartile_column] == "Q4_HIGH",
        score_column,
    ].dropna()

    non_q4 = data.loc[
        data[quartile_column] != "Q4_HIGH",
        score_column,
    ].dropna()

    if q4.empty:
        raise ValueError(
            f"No Q4 observations for {score_column}."
        )

    if non_q4.empty:
        raise ValueError(
            f"No non-Q4 observations for {score_column}."
        )

    q4_min = float(
        q4.min()
    )

    non_q4_max = float(
        non_q4.max()
    )

    if q4_min <= non_q4_max:
        raise ValueError(
            f"Q4 boundary for {score_column} "
            "cannot be represented by one fixed threshold. "
            f"q4_min={q4_min}, "
            f"non_q4_max={non_q4_max}"
        )

    # Midpoint reproduces the historical Q4 split without choosing
    # one observed score arbitrarily.
    threshold = (
        q4_min
        + non_q4_max
    ) / 2.0

    historical_classification = (
        data[score_column]
        >= threshold
    )

    frozen_classification = (
        data[quartile_column]
        == "Q4_HIGH"
    )

    matches = (
        historical_classification
        == frozen_classification
    )

    if not matches.all():
        raise AssertionError(
            f"Frozen threshold does not reproduce "
            f"{quartile_column} exactly."
        )

    return {
        "threshold":
            threshold,

        "q4_min_observed":
            q4_min,

        "non_q4_max_observed":
            non_q4_max,

        "q4_observations":
            int(
                len(q4)
            ),

        "non_q4_observations":
            int(
                len(non_q4)
            ),
    }


def main() -> None:
    data = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "market_date",
        ],
    )

    required = {
        "market_date",
        "technical_opportunity_score",
        "technical_quartile",
        "treasury_confirmation_score",
        "treasury_quartile",
    }

    missing = (
        required
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            "Interaction dataset missing columns: "
            f"{sorted(missing)}"
        )

    technical = derive_q4_boundary(
        data=data,
        score_column=(
            "technical_opportunity_score"
        ),
        quartile_column=(
            "technical_quartile"
        ),
    )

    treasury = derive_q4_boundary(
        data=data,
        score_column=(
            "treasury_confirmation_score"
        ),
        quartile_column=(
            "treasury_quartile"
        ),
    )

    policy = {
        "policy_name":
            "Opportunity Engine V1",

        "policy_status":
            "research-derived candidate for forward testing",

        "calibration_source":
            str(
                INPUT_FILE
            ),

        "calibration_rows":
            int(
                len(data)
            ),

        "calibration_start":
            str(
                data[
                    "market_date"
                ].min().date()
            ),

        "calibration_end":
            str(
                data[
                    "market_date"
                ].max().date()
            ),

        "technical": {
            "strong_setup_threshold":
                technical[
                    "threshold"
                ],

            **{
                key: value
                for key, value
                in technical.items()
                if key != "threshold"
            },
        },

        "treasury": {
            "extreme_confirmation_threshold":
                treasury[
                    "threshold"
                ],

            **{
                key: value
                for key, value
                in treasury.items()
                if key != "threshold"
            },
        },

        "decision_authority": {
            "risk_governor":
                True,

            "technical_opportunity":
                True,

            "treasury_extension_check":
                True,

            "macro_regime":
                False,

            "market_confirmation_v1":
                False,
        },
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            policy,
            file,
            indent=2,
        )

        file.write(
            "\n"
        )

    print("=" * 78)
    print(
        "OPPORTUNITY ENGINE V1 — "
        "FROZEN POLICY BOUNDARIES"
    )
    print("=" * 78)

    print(
        "\nCalibration rows:",
        len(data),
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
        "\nTECHNICAL"
    )

    print(
        "Max non-Q4 score:",
        f"{technical['non_q4_max_observed']:.6f}",
    )

    print(
        "Min Q4 score:",
        f"{technical['q4_min_observed']:.6f}",
    )

    print(
        "Frozen strong-setup threshold:",
        f"{technical['threshold']:.6f}",
    )

    print(
        "\nTREASURY"
    )

    print(
        "Max non-Q4 score:",
        f"{treasury['non_q4_max_observed']:.6f}",
    )

    print(
        "Min Q4 score:",
        f"{treasury['q4_min_observed']:.6f}",
    )

    print(
        "Frozen extreme threshold:",
        f"{treasury['threshold']:.6f}",
    )

    print(
        "\nPASS — fixed numeric boundaries reproduce "
        "the frozen historical Q4 classifications."
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()