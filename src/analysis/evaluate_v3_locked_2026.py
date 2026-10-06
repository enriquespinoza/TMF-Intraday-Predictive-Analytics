"""
Final locked 2026 out-of-sample evaluation for V3.

FINAL V3
--------
Technical V1
+
PAYEMS-A

This script performs the first model-selection-independent evaluation
on 2026 data.

Important methodological rules
------------------------------
- Final V3 specification must already be frozen.
- No 2026 observation may enter model training.
- A pre-2026 training row is eligible only if its entire 5-day
  forward target ends before 2026.
- 2026 observations require a complete 5-day forward target.
- Technical V1 is included only as the already-frozen benchmark.
- A training-base-rate forecast is included as a naive benchmark.
- All models are evaluated on exactly the same 2026 observations.
- Results must not be used to modify V3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from config.v3_final import (
    FINAL_V3_FEATURES,
    FINAL_V3_NAME,
    LOCKED_OOS_YEAR,
)

from src.models.directional_classifier import (
    HORIZON,
    SELECTED_TECHNICAL_FEATURES,
    TARGET_CLASS,
    TARGET_RETURN,
    add_directional_target,
    build_logistic_model,
)


DATA_FILE = Path(
    "data/features/"
    "TMF_1d_technical_PAYEMS_features.csv"
)

REPORT_DIR = Path(
    "reports"
)

METRICS_FILE = (
    REPORT_DIR
    / "v3_locked_2026_metrics.csv"
)

PREDICTIONS_FILE = (
    REPORT_DIR
    / "v3_locked_2026_predictions.csv"
)

COEFFICIENTS_FILE = (
    REPORT_DIR
    / "v3_locked_2026_coefficients.csv"
)

SPLIT_AUDIT_FILE = (
    REPORT_DIR
    / "v3_locked_2026_split_audit.csv"
)


TECHNICAL_V1_NAME = "technical_v1"
BASE_RATE_NAME = "training_base_rate"

OOS_START = pd.Timestamp(
    f"{LOCKED_OOS_YEAR}-01-01"
)

OOS_END = pd.Timestamp(
    f"{LOCKED_OOS_YEAR + 1}-01-01"
)


def load_data() -> pd.DataFrame:
    """Load the final Technical + PAYEMS dataset."""

    data = pd.read_csv(
        DATA_FILE,
        parse_dates=[
            "market_date",
        ],
    )

    if data.empty:
        raise ValueError(
            "Final V3 dataset is empty."
        )

    if data["market_date"].duplicated().any():
        raise ValueError(
            "Duplicate market dates detected."
        )

    required = {
        "market_date",
        TARGET_RETURN,
        *FINAL_V3_FEATURES,
    }

    missing = (
        required
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            "Final V3 dataset is missing "
            f"columns: {sorted(missing)}"
        )

    data = (
        data.sort_values(
            "market_date"
        )
        .reset_index(drop=True)
    )

    # The 5-day forward return on row t ends
    # at the market date five observations later.
    data[
        "target_end_date"
    ] = (
        data[
            "market_date"
        ].shift(
            -HORIZON
        )
    )

    data = add_directional_target(
        data
    )

    return data


def build_locked_split(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Construct the final training and locked 2026 samples.

    Training:
        market date < 2026
        AND target end date < 2026

    OOS:
        market date in 2026
        AND complete 5-day target exists
    """

    complete_features = (
        data[
            list(
                FINAL_V3_FEATURES
            )
        ]
        .notna()
        .all(axis=1)
    )

    valid_target = (
        data[
            TARGET_RETURN
        ].notna()
        & data[
            "target_end_date"
        ].notna()
    )

    train_mask = (
        (
            data[
                "market_date"
            ]
            < OOS_START
        )
        & (
            data[
                "target_end_date"
            ]
            < OOS_START
        )
        & valid_target
        & complete_features
    )

    oos_mask = (
        (
            data[
                "market_date"
            ]
            >= OOS_START
        )
        & (
            data[
                "market_date"
            ]
            < OOS_END
        )
        & valid_target
        & complete_features
    )

    train = (
        data.loc[
            train_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    oos = (
        data.loc[
            oos_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    if train.empty:
        raise ValueError(
            "No eligible pre-2026 "
            "training observations."
        )

    if oos.empty:
        raise ValueError(
            "No eligible locked 2026 "
            "observations."
        )

    # Hard leakage guardrails.

    if (
        train[
            "market_date"
        ]
        >= OOS_START
    ).any():
        raise AssertionError(
            "2026 market observation entered training."
        )

    if (
        train[
            "target_end_date"
        ]
        >= OOS_START
    ).any():
        raise AssertionError(
            "A training target reaches into 2026."
        )

    if not (
        oos[
            "market_date"
        ].dt.year
        == LOCKED_OOS_YEAR
    ).all():
        raise AssertionError(
            "Non-2026 observation entered locked OOS."
        )

    if (
        oos[
            TARGET_RETURN
        ].isna()
    ).any():
        raise AssertionError(
            "Incomplete target entered locked OOS."
        )

    return (
        train,
        oos,
    )


def safe_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Calculate ROC AUC when both classes exist."""

    if len(
        np.unique(
            y_true
        )
    ) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )


def calculate_metrics(
    model_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Calculate the prespecified locked-OOS metrics."""

    predictions = (
        probabilities
        >= 0.50
    ).astype(int)

    return {
        "model":
            model_name,

        "n":
            len(y_true),

        "roc_auc":
            safe_auc(
                y_true,
                probabilities,
            ),

        "brier_score":
            brier_score_loss(
                y_true,
                probabilities,
            ),

        "log_loss":
            log_loss(
                y_true,
                probabilities,
                labels=[
                    0,
                    1,
                ],
            ),

        "accuracy":
            accuracy_score(
                y_true,
                predictions,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                predictions,
            ),

        "mean_probability":
            float(
                np.mean(
                    probabilities
                )
            ),

        "observed_up_rate":
            float(
                np.mean(
                    y_true
                )
            ),
    }


def fit_model(
    train: pd.DataFrame,
    features: tuple[str, ...],
):
    """Fit the existing frozen logistic-regression pipeline."""

    model = (
        build_logistic_model()
    )

    model.fit(
        train[
            list(
                features
            )
        ],
        train[
            TARGET_CLASS
        ].astype(int),
    )

    return model


def coefficient_table(
    model,
    model_name: str,
    features: tuple[str, ...],
) -> pd.DataFrame:
    """Return standardized logistic-regression coefficients."""

    coefficients = (
        model.named_steps[
            "model"
        ].coef_[0]
    )

    return pd.DataFrame(
        {
            "model":
                model_name,

            "feature":
                list(
                    features
                ),

            "standardized_coefficient":
                coefficients,
        }
    )


def build_split_audit(
    train: pd.DataFrame,
    oos: pd.DataFrame,
) -> pd.DataFrame:
    """Record exactly what was used in the final split."""

    return pd.DataFrame(
        [
            {
                "sample":
                    "training",

                "n":
                    len(train),

                "market_start":
                    train[
                        "market_date"
                    ].min(),

                "market_end":
                    train[
                        "market_date"
                    ].max(),

                "target_end_min":
                    train[
                        "target_end_date"
                    ].min(),

                "target_end_max":
                    train[
                        "target_end_date"
                    ].max(),

                "up_rate":
                    train[
                        TARGET_CLASS
                    ].mean(),
            },

            {
                "sample":
                    "locked_2026",

                "n":
                    len(oos),

                "market_start":
                    oos[
                        "market_date"
                    ].min(),

                "market_end":
                    oos[
                        "market_date"
                    ].max(),

                "target_end_min":
                    oos[
                        "target_end_date"
                    ].min(),

                "target_end_max":
                    oos[
                        "target_end_date"
                    ].max(),

                "up_rate":
                    oos[
                        TARGET_CLASS
                    ].mean(),
            },
        ]
    )


def main() -> None:
    """Run the one-time locked 2026 evaluation."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen V3 locked "
            "2026 out-of-sample evaluation."
        )
    )

    parser.add_argument(
        "--unlock-2026",
        action="store_true",
        help=(
            "Explicitly authorize exposure "
            "of the locked 2026 outcomes."
        ),
    )

    args = parser.parse_args()

    if not args.unlock_2026:
        raise SystemExit(
            "\n2026 remains locked.\n\n"
            "This script exposes the official "
            "out-of-sample results.\n"
            "Run again with:\n\n"
            "python -m "
            "src.analysis.evaluate_v3_locked_2026 "
            "--unlock-2026\n"
        )

    print("=" * 76)
    print(
        "FINAL V3 — LOCKED 2026 OUT-OF-SAMPLE EVALUATION"
    )
    print("=" * 76)

    print(
        "\nThis run unlocks the available "
        "2026 OOS sample."
    )

    print(
        "Results must not be used "
        "to modify V3."
    )

    data = load_data()

    train, oos = (
        build_locked_split(
            data
        )
    )

    split_audit = (
        build_split_audit(
            train,
            oos,
        )
    )

    print("\n")
    print("=" * 76)
    print("SPLIT AUDIT")
    print("=" * 76)

    print(
        split_audit.to_string(
            index=False
        )
    )

    y_train = (
        train[
            TARGET_CLASS
        ]
        .astype(int)
        .to_numpy()
    )

    y_oos = (
        oos[
            TARGET_CLASS
        ]
        .astype(int)
        .to_numpy()
    )

    #
    # Benchmark 1:
    # training-sample base rate.
    #
    train_base_rate = float(
        np.mean(
            y_train
        )
    )

    base_rate_probabilities = (
        np.full(
            len(oos),
            train_base_rate,
            dtype=float,
        )
    )

    #
    # Benchmark 2:
    # frozen Technical V1.
    #
    technical_features = tuple(
        SELECTED_TECHNICAL_FEATURES
    )

    technical_model = (
        fit_model(
            train=train,
            features=technical_features,
        )
    )

    technical_probabilities = (
        technical_model.predict_proba(
            oos[
                list(
                    technical_features
                )
            ]
        )[:, 1]
    )

    #
    # Final frozen V3.
    #
    final_features = tuple(
        FINAL_V3_FEATURES
    )

    final_model = (
        fit_model(
            train=train,
            features=final_features,
        )
    )

    final_probabilities = (
        final_model.predict_proba(
            oos[
                list(
                    final_features
                )
            ]
        )[:, 1]
    )

    metric_rows = [
        calculate_metrics(
            model_name=BASE_RATE_NAME,
            y_true=y_oos,
            probabilities=(
                base_rate_probabilities
            ),
        ),

        calculate_metrics(
            model_name=TECHNICAL_V1_NAME,
            y_true=y_oos,
            probabilities=(
                technical_probabilities
            ),
        ),

        calculate_metrics(
            model_name=FINAL_V3_NAME,
            y_true=y_oos,
            probabilities=(
                final_probabilities
            ),
        ),
    ]

    metrics = pd.DataFrame(
        metric_rows
    )

    technical_row = (
        metrics.loc[
            metrics[
                "model"
            ]
            == TECHNICAL_V1_NAME
        ]
        .iloc[0]
    )

    metrics[
        "delta_auc_vs_technical"
    ] = (
        metrics[
            "roc_auc"
        ]
        - technical_row[
            "roc_auc"
        ]
    )

    # Positive delta means better.
    metrics[
        "delta_brier_vs_technical"
    ] = (
        technical_row[
            "brier_score"
        ]
        - metrics[
            "brier_score"
        ]
    )

    # Positive delta means better.
    metrics[
        "delta_log_loss_vs_technical"
    ] = (
        technical_row[
            "log_loss"
        ]
        - metrics[
            "log_loss"
        ]
    )

    coefficients = pd.concat(
        [
            coefficient_table(
                model=technical_model,
                model_name=TECHNICAL_V1_NAME,
                features=(
                    technical_features
                ),
            ),

            coefficient_table(
                model=final_model,
                model_name=FINAL_V3_NAME,
                features=(
                    final_features
                ),
            ),
        ],
        ignore_index=True,
    )

    predictions = pd.DataFrame(
        {
            "market_date":
                oos[
                    "market_date"
                ].to_numpy(),

            "target_end_date":
                oos[
                    "target_end_date"
                ].to_numpy(),

            "forward_return_5":
                oos[
                    TARGET_RETURN
                ].to_numpy(),

            "target_up_5d":
                y_oos,

            "training_base_rate_probability":
                base_rate_probabilities,

            "technical_v1_probability":
                technical_probabilities,

            "v3_final_probability":
                final_probabilities,
        }
    )

    predictions[
        "v3_minus_technical_probability"
    ] = (
        predictions[
            "v3_final_probability"
        ]
        - predictions[
            "technical_v1_probability"
        ]
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    coefficients.to_csv(
        COEFFICIENTS_FILE,
        index=False,
    )

    split_audit.to_csv(
        SPLIT_AUDIT_FILE,
        index=False,
    )

    print("\n")
    print("=" * 76)
    print("LOCKED 2026 RESULTS")
    print("=" * 76)

    print(
        metrics.to_string(
            index=False,
            float_format=(
                lambda x:
                    f"{x:.6f}"
            ),
        )
    )

    print("\n")
    print("=" * 76)
    print("FINAL V3 TRAINING COEFFICIENTS")
    print("=" * 76)

    print(
        coefficients.loc[
            coefficients[
                "model"
            ]
            == FINAL_V3_NAME
        ].to_string(
            index=False,
            float_format=(
                lambda x:
                    f"{x:.6f}"
            ),
        )
    )

    print("\nSaved:")
    print(
        f"  {METRICS_FILE}"
    )
    print(
        f"  {PREDICTIONS_FILE}"
    )
    print(
        f"  {COEFFICIENTS_FILE}"
    )
    print(
        f"  {SPLIT_AUDIT_FILE}"
    )

    print(
        "\n2026 is now UNLOCKED "
        "for evaluation purposes."
    )

    print(
        "Do not change the frozen V3 "
        "specification based on these results."
    )


if __name__ == "__main__":
    main()