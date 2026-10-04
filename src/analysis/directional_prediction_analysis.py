"""Analyze out-of-sample TMF directional predictions."""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "reports/TMF_directional_predictions.csv"
)

BUCKET_OUTPUT_FILE = Path(
    "reports/TMF_directional_probability_buckets.csv"
)

CALIBRATION_OUTPUT_FILE = Path(
    "reports/TMF_directional_calibration.csv"
)

FOLD_BUCKET_OUTPUT_FILE = Path(
    "reports/TMF_directional_probability_buckets_by_fold.csv"
)

PROBABILITY_COLUMNS = [
    "technical_probability",
    "treasury_probability",
    "combined_probability",
]


def assign_probability_quintiles(
    df: pd.DataFrame,
    probability_column: str,
) -> pd.Series:
    """Assign predictions to probability quintiles."""

    ranked = (
        df[probability_column]
        .rank(
            method="first",
            pct=True,
        )
    )

    return pd.cut(
        ranked,
        bins=[
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        labels=[
            "Q1_lowest",
            "Q2",
            "Q3",
            "Q4",
            "Q5_highest",
        ],
        include_lowest=True,
    )


def summarize_probability_buckets(
    df: pd.DataFrame,
    probability_column: str,
) -> pd.DataFrame:
    """Summarize outcomes by probability quintile."""

    data = df.copy()

    data["probability_bucket"] = (
        assign_probability_quintiles(
            data,
            probability_column,
        )
    )

    summary = (
        data.groupby(
            "probability_bucket",
            observed=True,
        )
        .agg(
            observations=(
                "actual_up",
                "size",
            ),
            mean_predicted_probability=(
                probability_column,
                "mean",
            ),
            actual_positive_rate=(
                "actual_up",
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
        )
        .reset_index()
    )

    summary.insert(
        0,
        "model",
        probability_column,
    )

    return summary

def summarize_fold_probability_buckets(
    df: pd.DataFrame,
    probability_column: str,
) -> pd.DataFrame:
    """Summarize probability quintiles independently by fold."""

    results = []

    for fold_name, fold_data in df.groupby(
        "fold"
    ):
        data = fold_data.copy()

        data["probability_bucket"] = (
            assign_probability_quintiles(
                data,
                probability_column,
            )
        )

        summary = (
            data.groupby(
                "probability_bucket",
                observed=True,
            )
            .agg(
                observations=(
                    "actual_up",
                    "size",
                ),
                mean_predicted_probability=(
                    probability_column,
                    "mean",
                ),
                actual_positive_rate=(
                    "actual_up",
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
            )
            .reset_index()
        )

        summary.insert(
            0,
            "fold",
            fold_name,
        )

        summary.insert(
            1,
            "model",
            probability_column,
        )

        results.append(summary)

    return pd.concat(
        results,
        ignore_index=True,
    )

def summarize_calibration(
    df: pd.DataFrame,
    probability_column: str,
) -> pd.DataFrame:
    """Compare predicted and realized probabilities."""

    data = df.copy()

    data["calibration_bucket"] = pd.cut(
        data[probability_column],
        bins=np.linspace(
            0.0,
            1.0,
            11,
        ),
        include_lowest=True,
    )

    summary = (
        data.groupby(
            "calibration_bucket",
            observed=True,
        )
        .agg(
            observations=(
                "actual_up",
                "size",
            ),
            mean_predicted_probability=(
                probability_column,
                "mean",
            ),
            actual_positive_rate=(
                "actual_up",
                "mean",
            ),
            mean_forward_return_5=(
                "forward_return_5",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.insert(
        0,
        "model",
        probability_column,
    )

    return summary


def main() -> None:
    """Run prediction-level analysis."""

    print(
        "Loading out-of-sample predictions..."
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["market_date"],
    )

    print()
    print(
        "Validation observations:",
        len(df),
    )

    print(
        "Validation period:",
        df["market_date"].min().date(),
        "to",
        df["market_date"].max().date(),
    )

    print()

    bucket_results = []
    calibration_results = []

    for probability_column in (
        PROBABILITY_COLUMNS
    ):

        bucket_summary = (
            summarize_probability_buckets(
                df,
                probability_column,
            )
        )

        calibration_summary = (
            summarize_calibration(
                df,
                probability_column,
            )
        )

        bucket_results.append(
            bucket_summary
        )

        calibration_results.append(
            calibration_summary
        )

    bucket_results = pd.concat(
        bucket_results,
        ignore_index=True,
    )

    calibration_results = pd.concat(
        calibration_results,
        ignore_index=True,
    )

    BUCKET_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bucket_results.to_csv(
        BUCKET_OUTPUT_FILE,
        index=False,
    )

    calibration_results.to_csv(
        CALIBRATION_OUTPUT_FILE,
        index=False,
    )

    print(
        "TECHNICAL MODEL "
        "PROBABILITY QUINTILES"
    )

    print()

    technical = bucket_results[
        bucket_results["model"]
        == "technical_probability"
    ]

    print(
        technical.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved probability buckets to:",
        BUCKET_OUTPUT_FILE,
    )

    print(
        "Saved calibration analysis to:",
        CALIBRATION_OUTPUT_FILE,
    )

    # ---------------------------------
    # Fold-specific technical quintiles
    # ---------------------------------

    fold_bucket_results = (
        summarize_fold_probability_buckets(
            df,
            "technical_probability",
        )
    )

    fold_bucket_results.to_csv(
        FOLD_BUCKET_OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "TECHNICAL MODEL QUINTILES BY FOLD"
    )
    print()

    print(
        fold_bucket_results.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved fold probability buckets to:",
        FOLD_BUCKET_OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()