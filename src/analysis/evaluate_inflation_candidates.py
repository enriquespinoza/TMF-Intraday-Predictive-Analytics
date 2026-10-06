"""
Frozen inflation candidate evaluation.

Compares:

1. V3 base:
       Technical V1 + PAYEMS-A

2. V3 base + INFLATION-A:
       breakeven_10y
       breakeven_10y_change_5d_bp

3. V3 base + INFLATION-B:
       core_cpi_yoy_pct
       core_cpi_3m_annualized_pct

Rules
-----
- Inflation candidate definitions are frozen before evaluation.
- PAYEMS-A remains unchanged.
- Existing Technical V1 logistic-regression pipeline is reused.
- Existing purged 2023/2024/2025 folds are reused.
- All models use identical observations within each fold.
- 2026 is excluded from development evaluation.
- No hyperparameter tuning is performed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.inflation_candidates import (
    INFLATION_A_FEATURES,
    INFLATION_A_NAME,
    INFLATION_B_FEATURES,
    INFLATION_B_NAME,
    LOCKED_OOS_YEAR,
)

from config.payems_candidates import (
    PAYEMS_A_FEATURES,
)

from src.analysis.purged_split import (
    expanding_walk_forward_splits,
)

from src.models.directional_classifier import (
    HORIZON,
    SELECTED_TECHNICAL_FEATURES,
    TARGET_CLASS,
    add_directional_target,
    evaluate_logistic_model,
)


DATA_FILE = Path(
    "data/features/"
    "TMF_1d_technical_PAYEMS_inflation_features.csv"
)

RESULTS_FILE = Path(
    "reports/"
    "inflation_frozen_candidate_evaluation.csv"
)

SUMMARY_FILE = Path(
    "reports/"
    "inflation_frozen_candidate_summary.csv"
)


V3_BASE_NAME = "technical_v1_plus_payems_a"

V3_BASE_FEATURES = (
    tuple(SELECTED_TECHNICAL_FEATURES)
    + tuple(PAYEMS_A_FEATURES)
)


MODEL_SPECS = {
    V3_BASE_NAME:
        V3_BASE_FEATURES,

    INFLATION_A_NAME:
        V3_BASE_FEATURES
        + tuple(INFLATION_A_FEATURES),

    INFLATION_B_NAME:
        V3_BASE_FEATURES
        + tuple(INFLATION_B_FEATURES),
}


MATCHED_SAMPLE_FEATURES = tuple(
    dict.fromkeys(
        tuple(PAYEMS_A_FEATURES)
        + tuple(INFLATION_A_FEATURES)
        + tuple(INFLATION_B_FEATURES)
    )
)


def load_evaluation_data() -> pd.DataFrame:
    """Load development dataset and exclude locked 2026."""

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["market_date"],
    )

    df = add_directional_target(df)

    df = df.loc[
        df["market_date"]
        < pd.Timestamp(
            f"{LOCKED_OOS_YEAR}-01-01"
        )
    ].copy()

    if df.empty:
        raise ValueError(
            "No development observations remain."
        )

    if (
        df["market_date"].dt.year
        >= LOCKED_OOS_YEAR
    ).any():
        raise AssertionError(
            "Locked 2026 observations entered "
            "inflation evaluation."
        )

    required = {
        "market_date",
        TARGET_CLASS,
        *SELECTED_TECHNICAL_FEATURES,
        *MATCHED_SAMPLE_FEATURES,
    }

    missing = sorted(
        required.difference(df.columns)
    )

    if missing:
        raise ValueError(
            "Evaluation dataset missing "
            f"columns: {missing}"
        )

    return df


def prepare_matched_fold(
    fold: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use identical eligible rows for all three models."""

    required = [
        TARGET_CLASS,
        *MATCHED_SAMPLE_FEATURES,
    ]

    train = (
        fold["train"]
        .dropna(subset=required)
        .copy()
    )

    validation = (
        fold["validation"]
        .dropna(subset=required)
        .copy()
    )

    if train.empty:
        raise ValueError(
            f"{fold['name']} has no training rows."
        )

    if validation.empty:
        raise ValueError(
            f"{fold['name']} has no validation rows."
        )

    return train, validation


def evaluate_fold(
    fold: dict,
) -> list[dict]:
    """Evaluate frozen inflation models on one fold."""

    train, validation = prepare_matched_fold(
        fold
    )

    validation_year = int(
        validation[
            "market_date"
        ].dt.year.iloc[0]
    )

    if validation_year >= LOCKED_OOS_YEAR:
        raise AssertionError(
            "Locked year entered evaluation."
        )

    rows = []

    for model_name, features in MODEL_SPECS.items():

        metrics = evaluate_logistic_model(
            train=train,
            validation=validation,
            features=list(features),
        )

        rows.append(
            {
                "fold": fold["name"],
                "validation_year":
                    validation_year,
                "model": model_name,
                "train_n": len(train),
                "validation_n":
                    len(validation),
                "train_base_rate":
                    train[
                        TARGET_CLASS
                    ].mean(),
                "validation_base_rate":
                    validation[
                        TARGET_CLASS
                    ].mean(),
                **metrics,
            }
        )

    return rows


def add_deltas(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate improvement versus V3 base."""

    output = results.copy()

    output["delta_auc_vs_base"] = 0.0
    output["delta_brier_vs_base"] = 0.0
    output["delta_log_loss_vs_base"] = 0.0

    for fold_name in output["fold"].unique():

        mask = (
            output["fold"]
            == fold_name
        )

        baseline = output.loc[
            mask
            & (
                output["model"]
                == V3_BASE_NAME
            )
        ].iloc[0]

        output.loc[
            mask,
            "delta_auc_vs_base",
        ] = (
            output.loc[
                mask,
                "roc_auc",
            ]
            - baseline["roc_auc"]
        )

        # Positive = improvement.
        output.loc[
            mask,
            "delta_brier_vs_base",
        ] = (
            baseline["brier_score"]
            - output.loc[
                mask,
                "brier_score",
            ]
        )

        # Positive = improvement.
        output.loc[
            mask,
            "delta_log_loss_vs_base",
        ] = (
            baseline["log_loss"]
            - output.loc[
                mask,
                "log_loss",
            ]
        )

    return output


def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Create across-fold summary."""

    return (
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
            mean_delta_auc_vs_base=(
                "delta_auc_vs_base",
                "mean",
            ),
            mean_delta_brier_vs_base=(
                "delta_brier_vs_base",
                "mean",
            ),
            mean_delta_log_loss_vs_base=(
                "delta_log_loss_vs_base",
                "mean",
            ),
        )
    )


def main() -> None:
    """Run frozen inflation candidate evaluation."""

    print("=" * 72)
    print("FROZEN INFLATION CANDIDATE EVALUATION")
    print("=" * 72)

    print(
        "\nLocked OOS year:",
        LOCKED_OOS_YEAR,
    )

    print("\nV3 base:")
    for feature in V3_BASE_FEATURES:
        print(f"  - {feature}")

    print(f"\n{INFLATION_A_NAME}:")
    for feature in INFLATION_A_FEATURES:
        print(f"  - {feature}")

    print(f"\n{INFLATION_B_NAME}:")
    for feature in INFLATION_B_FEATURES:
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

    rows = []

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

        rows.extend(
            evaluate_fold(fold)
        )

    results = pd.DataFrame(rows)

    results = add_deltas(
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

    columns = [
        "validation_year",
        "model",
        "train_n",
        "validation_n",
        "roc_auc",
        "brier_score",
        "log_loss",
        "accuracy",
        "balanced_accuracy",
        "delta_auc_vs_base",
        "delta_brier_vs_base",
        "delta_log_loss_vs_base",
    ]

    print("\n")
    print("=" * 72)
    print("FOLD RESULTS")
    print("=" * 72)

    print(
        results[
            columns
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