"""
Frozen PAYEMS candidate evaluation.

Compares:

1. Technical V1 matched-sample control
2. Technical V1 + PAYEMS-A
3. Technical V1 + PAYEMS-B

Evaluation rules
----------------
- Candidate definitions are imported from config/payems_candidates.py.
- Technical V1 features are imported from the existing frozen classifier.
- Existing LogisticRegression pipeline is reused unchanged.
- Existing purged expanding walk-forward folds are reused unchanged.
- All three models use identical observations within each fold.
- 2026 is excluded completely from this development evaluation.
- No feature selection or model tuning is performed here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.payems_candidates import (
    LOCKED_OOS_YEAR,
    PAYEMS_A_FEATURES,
    PAYEMS_A_NAME,
    PAYEMS_B_FEATURES,
    PAYEMS_B_NAME,
)

from src.analysis.purged_split import (
    expanding_walk_forward_splits,
)

from src.models.directional_classifier import (
    HORIZON,
    SELECTED_TECHNICAL_FEATURES,
    TARGET_CLASS,
    add_directional_target,
    evaluate_base_rate,
    evaluate_logistic_model,
)


DATA_FILE = Path(
    "data/features/"
    "TMF_1d_technical_PAYEMS_features.csv"
)

RESULTS_FILE = Path(
    "reports/"
    "PAYEMS_frozen_candidate_evaluation.csv"
)

SUMMARY_FILE = Path(
    "reports/"
    "PAYEMS_frozen_candidate_summary.csv"
)


MODEL_SPECS = {
    "technical_v1_matched":
        tuple(SELECTED_TECHNICAL_FEATURES),

    PAYEMS_A_NAME:
        tuple(SELECTED_TECHNICAL_FEATURES)
        + tuple(PAYEMS_A_FEATURES),

    PAYEMS_B_NAME:
        tuple(SELECTED_TECHNICAL_FEATURES)
        + tuple(PAYEMS_B_FEATURES),
}


# Require all frozen PAYEMS candidate inputs to exist so
# every model is evaluated on exactly the same rows.
MATCHED_SAMPLE_FEATURES = tuple(
    dict.fromkeys(
        tuple(PAYEMS_A_FEATURES)
        + tuple(PAYEMS_B_FEATURES)
    )
)


def load_evaluation_data() -> pd.DataFrame:
    """Load development data and exclude locked 2026 rows."""

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["market_date"],
    )

    df = add_directional_target(df)

    # Hard development cutoff.
    df = df.loc[
        df["market_date"]
        < pd.Timestamp(
            f"{LOCKED_OOS_YEAR}-01-01"
        )
    ].copy()

    if df.empty:
        raise ValueError(
            "No development observations remain "
            "after applying the locked-OOS cutoff."
        )

    if (
        df["market_date"].dt.year
        >= LOCKED_OOS_YEAR
    ).any():
        raise AssertionError(
            "Locked 2026 observations entered "
            "the development dataset."
        )

    required_columns = {
        "market_date",
        TARGET_CLASS,
        *SELECTED_TECHNICAL_FEATURES,
        *MATCHED_SAMPLE_FEATURES,
    }

    missing = sorted(
        required_columns.difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Evaluation dataset is missing "
            f"required columns: {missing}"
        )

    return df


def prepare_matched_fold(
    fold: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restrict train and validation to the same
    candidate-eligible observations.

    Technical feature NaNs are not removed because
    the frozen V1 model handles them using its
    existing median imputer.
    """

    train = (
        fold["train"]
        .dropna(
            subset=[
                TARGET_CLASS,
                *MATCHED_SAMPLE_FEATURES,
            ]
        )
        .copy()
    )

    validation = (
        fold["validation"]
        .dropna(
            subset=[
                TARGET_CLASS,
                *MATCHED_SAMPLE_FEATURES,
            ]
        )
        .copy()
    )

    if train.empty:
        raise ValueError(
            f"{fold['name']} has no eligible "
            "training observations."
        )

    if validation.empty:
        raise ValueError(
            f"{fold['name']} has no eligible "
            "validation observations."
        )

    return train, validation


def evaluate_fold(
    fold: dict,
) -> list[dict]:
    """Evaluate every frozen model on one matched fold."""

    train, validation = prepare_matched_fold(
        fold
    )

    validation_year = int(
        validation["market_date"]
        .dt.year
        .iloc[0]
    )

    if validation_year >= LOCKED_OOS_YEAR:
        raise AssertionError(
            "Locked OOS year entered evaluation."
        )

    base_metrics = evaluate_base_rate(
        train=train,
        validation=validation,
    )

    rows = []

    for model_name, features in MODEL_SPECS.items():

        metrics = evaluate_logistic_model(
            train=train,
            validation=validation,
            features=list(features),
        )

        row = {
            "fold": fold["name"],
            "validation_year":
                validation_year,
            "model": model_name,
            "train_n": len(train),
            "validation_n": len(validation),
            "train_base_rate":
                train[TARGET_CLASS].mean(),
            "validation_base_rate":
                validation[TARGET_CLASS].mean(),
            **metrics,
            "base_rate_brier":
                base_metrics["brier_score"],
            "base_rate_log_loss":
                base_metrics["log_loss"],
            "base_rate_accuracy":
                base_metrics["accuracy"],
        }

        rows.append(row)

    return rows


def add_fold_deltas(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add candidate improvement versus matched
    Technical V1.

    Positive delta values mean improvement:
    - AUC: candidate - V1
    - Brier: V1 - candidate
    - Log loss: V1 - candidate
    """

    output = results.copy()

    output["delta_auc_vs_v1"] = 0.0
    output["delta_brier_vs_v1"] = 0.0
    output["delta_log_loss_vs_v1"] = 0.0

    for fold_name in output["fold"].unique():

        fold_mask = (
            output["fold"] == fold_name
        )

        baseline = output.loc[
            fold_mask
            & (
                output["model"]
                == "technical_v1_matched"
            )
        ].iloc[0]

        output.loc[
            fold_mask,
            "delta_auc_vs_v1",
        ] = (
            output.loc[
                fold_mask,
                "roc_auc",
            ]
            - baseline["roc_auc"]
        )

        output.loc[
            fold_mask,
            "delta_brier_vs_v1",
        ] = (
            baseline["brier_score"]
            - output.loc[
                fold_mask,
                "brier_score",
            ]
        )

        output.loc[
            fold_mask,
            "delta_log_loss_vs_v1",
        ] = (
            baseline["log_loss"]
            - output.loc[
                fold_mask,
                "log_loss",
            ]
        )

    return output


def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Create across-fold summary statistics."""

    summary = (
        results
        .groupby(
            "model",
            as_index=False,
        )
        .agg(
            folds=("fold", "count"),
            mean_auc=("roc_auc", "mean"),
            min_auc=("roc_auc", "min"),
            max_auc=("roc_auc", "max"),
            mean_brier=(
                "brier_score",
                "mean",
            ),
            mean_log_loss=(
                "log_loss",
                "mean",
            ),
            mean_accuracy=(
                "accuracy",
                "mean",
            ),
            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),
            mean_delta_auc_vs_v1=(
                "delta_auc_vs_v1",
                "mean",
            ),
            mean_delta_brier_vs_v1=(
                "delta_brier_vs_v1",
                "mean",
            ),
            mean_delta_log_loss_vs_v1=(
                "delta_log_loss_vs_v1",
                "mean",
            ),
        )
    )

    return summary


def main() -> None:
    """Run frozen PAYEMS candidate evaluation."""

    print("=" * 72)
    print("FROZEN PAYEMS CANDIDATE EVALUATION")
    print("=" * 72)

    print(
        "\nLocked OOS year:",
        LOCKED_OOS_YEAR,
    )

    print(
        "\nTechnical V1 features:"
    )

    for feature in SELECTED_TECHNICAL_FEATURES:
        print(f"  - {feature}")

    print(
        f"\n{PAYEMS_A_NAME}:"
    )

    for feature in PAYEMS_A_FEATURES:
        print(f"  - {feature}")

    print(
        f"\n{PAYEMS_B_NAME}:"
    )

    for feature in PAYEMS_B_FEATURES:
        print(f"  - {feature}")

    df = load_evaluation_data()

    print(
        "\nDevelopment range:",
        df["market_date"].min().date(),
        "to",
        df["market_date"].max().date(),
    )

    print(
        "Development rows:",
        len(df),
    )

    folds = expanding_walk_forward_splits(
        df=df,
        horizon=HORIZON,
    )

    result_rows = []

    for fold in folds:

        train, validation = (
            prepare_matched_fold(fold)
        )

        print("\n" + "-" * 72)
        print(
            fold["name"],
            "| train:",
            len(train),
            "| validation:",
            len(validation),
        )

        print(
            "Train:",
            train["market_date"]
            .min()
            .date(),
            "to",
            train["market_date"]
            .max()
            .date(),
        )

        print(
            "Validation:",
            validation["market_date"]
            .min()
            .date(),
            "to",
            validation["market_date"]
            .max()
            .date(),
        )

        result_rows.extend(
            evaluate_fold(fold)
        )

    results = pd.DataFrame(
        result_rows
    )

    results = add_fold_deltas(
        results
    )

    summary = build_summary(
        results
    )

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    display_columns = [
        "validation_year",
        "model",
        "train_n",
        "validation_n",
        "roc_auc",
        "brier_score",
        "log_loss",
        "accuracy",
        "balanced_accuracy",
        "delta_auc_vs_v1",
        "delta_brier_vs_v1",
        "delta_log_loss_vs_v1",
    ]

    print("\n")
    print("=" * 72)
    print("FOLD RESULTS")
    print("=" * 72)

    print(
        results[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n")
    print("=" * 72)
    print("ACROSS-FOLD SUMMARY")
    print("=" * 72)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        "\nSaved fold results to:",
        RESULTS_FILE,
    )

    print(
        "Saved summary to:",
        SUMMARY_FILE,
    )


if __name__ == "__main__":
    main()