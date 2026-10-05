"""
PAYEMS-A robustness analysis.

This module performs diagnostics on the already-frozen PAYEMS-A
candidate after its initial walk-forward validation.

No feature selection, hyperparameter tuning, or 2026 evaluation is
performed here.

Diagnostics:
1. Standardized coefficient stability
2. Probability-shift analysis
3. Calibration by probability quintile
4. Performance by time since payroll release
5. Moving-block bootstrap of paired V1 vs PAYEMS-A improvements

The same matched sample used in the original frozen PAYEMS evaluation
is retained.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config.payems_candidates import (
    LOCKED_OOS_YEAR,
    PAYEMS_A_FEATURES,
    PAYEMS_A_NAME,
)

from src.analysis.evaluate_payems_candidates import (
    MODEL_SPECS,
    load_evaluation_data,
    prepare_matched_fold,
)

from src.analysis.purged_split import (
    expanding_walk_forward_splits,
)

from src.models.directional_classifier import (
    HORIZON,
    TARGET_CLASS,
    build_logistic_model,
)


TECHNICAL_MODEL_NAME = "technical_v1_matched"

BLOCK_LENGTHS = (5, 10)
N_BOOTSTRAP = 2000
RANDOM_SEED = 42
CALIBRATION_BINS = 5


PREDICTIONS_FILE = Path(
    "reports/PAYEMS_A_robustness_predictions.csv"
)

COEFFICIENTS_FILE = Path(
    "reports/PAYEMS_A_coefficient_stability.csv"
)

PROBABILITY_SHIFT_FILE = Path(
    "reports/PAYEMS_A_probability_shift.csv"
)

CALIBRATION_FILE = Path(
    "reports/PAYEMS_A_calibration.csv"
)

RELEASE_AGE_FILE = Path(
    "reports/PAYEMS_A_release_age_analysis.csv"
)

BOOTSTRAP_FILE = Path(
    "reports/PAYEMS_A_block_bootstrap.csv"
)


def observation_log_loss(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Return observation-level binary log loss."""

    p = np.clip(
        probabilities,
        1e-15,
        1 - 1e-15,
    )

    return -(
        y_true * np.log(p)
        + (1 - y_true) * np.log(1 - p)
    )


def safe_auc(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
) -> float:
    """Calculate AUC when both classes are present."""

    y = np.asarray(y_true)

    if len(np.unique(y)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            np.asarray(probabilities),
        )
    )


def fit_and_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Fit the unchanged logistic pipeline and return
    probabilities plus standardized coefficients.
    """

    model = build_logistic_model()

    model.fit(
        train[list(features)],
        train[TARGET_CLASS].astype(int),
    )

    probabilities = model.predict_proba(
        validation[list(features)]
    )[:, 1]

    coefficients = pd.DataFrame(
        {
            "feature": list(features),
            "standardized_coefficient":
                model.named_steps[
                    "model"
                ].coef_[0],
        }
    )

    return probabilities, coefficients


def generate_predictions_and_coefficients(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate matched V1 and PAYEMS-A predictions."""

    folds = expanding_walk_forward_splits(
        df=df,
        horizon=HORIZON,
    )

    prediction_frames = []
    coefficient_frames = []

    technical_features = tuple(
        MODEL_SPECS[
            TECHNICAL_MODEL_NAME
        ]
    )

    payems_a_features = tuple(
        MODEL_SPECS[
            PAYEMS_A_NAME
        ]
    )

    for fold in folds:

        train, validation = prepare_matched_fold(
            fold
        )

        year = int(
            validation[
                "market_date"
            ].dt.year.iloc[0]
        )

        if year >= LOCKED_OOS_YEAR:
            raise AssertionError(
                "Locked 2026 observations entered "
                "robustness analysis."
            )

        technical_probabilities, technical_coef = (
            fit_and_predict(
                train=train,
                validation=validation,
                features=technical_features,
            )
        )

        payems_probabilities, payems_coef = (
            fit_and_predict(
                train=train,
                validation=validation,
                features=payems_a_features,
            )
        )

        predictions = pd.DataFrame(
            {
                "fold": fold["name"],
                "validation_year": year,
                "market_date":
                    validation[
                        "market_date"
                    ].to_numpy(),
                "target":
                    validation[
                        TARGET_CLASS
                    ].astype(int).to_numpy(),
                "technical_v1_probability":
                    technical_probabilities,
                "payems_a_probability":
                    payems_probabilities,
                "days_since_payroll_release":
                    validation[
                        "days_since_payroll_release"
                    ].to_numpy(),
                "payroll_release_day":
                    validation[
                        "payroll_release_day"
                    ].to_numpy(),
                "payroll_broad_revision_event":
                    validation[
                        "payroll_broad_revision_event"
                    ].to_numpy(),
            }
        )

        predictions[
            "probability_shift"
        ] = (
            predictions[
                "payems_a_probability"
            ]
            - predictions[
                "technical_v1_probability"
            ]
        )

        predictions[
            "absolute_probability_shift"
        ] = predictions[
            "probability_shift"
        ].abs()

        prediction_frames.append(
            predictions
        )

        technical_coef["fold"] = fold["name"]
        technical_coef[
            "validation_year"
        ] = year
        technical_coef["model"] = (
            TECHNICAL_MODEL_NAME
        )

        payems_coef["fold"] = fold["name"]
        payems_coef[
            "validation_year"
        ] = year
        payems_coef["model"] = (
            PAYEMS_A_NAME
        )

        coefficient_frames.extend(
            [
                technical_coef,
                payems_coef,
            ]
        )

    return (
        pd.concat(
            prediction_frames,
            ignore_index=True,
        ),
        pd.concat(
            coefficient_frames,
            ignore_index=True,
        ),
    )


def paired_metrics(
    data: pd.DataFrame,
) -> dict:
    """Calculate paired V1 vs PAYEMS-A metrics."""

    y = data["target"].to_numpy()

    p_v1 = data[
        "technical_v1_probability"
    ].to_numpy()

    p_a = data[
        "payems_a_probability"
    ].to_numpy()

    auc_v1 = safe_auc(y, p_v1)
    auc_a = safe_auc(y, p_a)

    brier_v1 = np.mean(
        (y - p_v1) ** 2
    )

    brier_a = np.mean(
        (y - p_a) ** 2
    )

    log_loss_v1 = np.mean(
        observation_log_loss(
            y,
            p_v1,
        )
    )

    log_loss_a = np.mean(
        observation_log_loss(
            y,
            p_a,
        )
    )

    return {
        "n": len(data),
        "auc_v1": auc_v1,
        "auc_payems_a": auc_a,
        "delta_auc":
            auc_a - auc_v1,
        "brier_v1": brier_v1,
        "brier_payems_a": brier_a,
        "delta_brier":
            brier_v1 - brier_a,
        "log_loss_v1": log_loss_v1,
        "log_loss_payems_a":
            log_loss_a,
        "delta_log_loss":
            log_loss_v1 - log_loss_a,
    }


def build_probability_shift_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize how much PAYEMS-A changes probabilities."""

    rows = []

    for fold_name, group in predictions.groupby(
        "fold",
        sort=False,
    ):

        rows.append(
            {
                "fold": fold_name,
                "validation_year":
                    int(
                        group[
                            "validation_year"
                        ].iloc[0]
                    ),
                "n": len(group),
                "mean_probability_shift":
                    group[
                        "probability_shift"
                    ].mean(),
                "mean_absolute_shift":
                    group[
                        "absolute_probability_shift"
                    ].mean(),
                "median_absolute_shift":
                    group[
                        "absolute_probability_shift"
                    ].median(),
                "p90_absolute_shift":
                    group[
                        "absolute_probability_shift"
                    ].quantile(0.90),
                "max_absolute_shift":
                    group[
                        "absolute_probability_shift"
                    ].max(),
                "share_abs_shift_ge_5pct":
                    (
                        group[
                            "absolute_probability_shift"
                        ]
                        >= 0.05
                    ).mean(),
                "probability_correlation":
                    group[
                        [
                            "technical_v1_probability",
                            "payems_a_probability",
                        ]
                    ].corr().iloc[0, 1],
            }
        )

    return pd.DataFrame(rows)


def build_calibration_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Create probability-quintile calibration tables."""

    rows = []

    model_columns = {
        TECHNICAL_MODEL_NAME:
            "technical_v1_probability",
        PAYEMS_A_NAME:
            "payems_a_probability",
    }

    for fold_name, fold_data in predictions.groupby(
        "fold",
        sort=False,
    ):

        for model_name, probability_column in (
            model_columns.items()
        ):

            data = fold_data.copy()

            # Rank first so duplicate probabilities do not
            # collapse quantile bins.
            ranks = data[
                probability_column
            ].rank(
                method="first"
            )

            data["calibration_bin"] = pd.qcut(
                ranks,
                q=CALIBRATION_BINS,
                labels=False,
            ) + 1

            grouped = (
                data.groupby(
                    "calibration_bin",
                    observed=True,
                )
                .agg(
                    n=("target", "size"),
                    mean_probability=(
                        probability_column,
                        "mean",
                    ),
                    observed_rate=(
                        "target",
                        "mean",
                    ),
                )
                .reset_index()
            )

            grouped["calibration_gap"] = (
                grouped["observed_rate"]
                - grouped["mean_probability"]
            )

            grouped["fold"] = fold_name
            grouped[
                "validation_year"
            ] = int(
                fold_data[
                    "validation_year"
                ].iloc[0]
            )
            grouped["model"] = model_name

            rows.append(grouped)

    return pd.concat(
        rows,
        ignore_index=True,
    )


def release_age_bucket(
    days_since_release: pd.Series,
) -> pd.Series:
    """Bucket observations by payroll-release age."""

    return pd.cut(
        days_since_release,
        bins=[
            -0.1,
            0,
            5,
            15,
            np.inf,
        ],
        labels=[
            "release_day",
            "days_1_5",
            "days_6_15",
            "days_16_plus",
        ],
        include_lowest=True,
    )


def build_release_age_analysis(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Test whether PAYEMS-A improvement is concentrated
    near payroll release dates.
    """

    data = predictions.copy()

    data["release_age_bucket"] = (
        release_age_bucket(
            data[
                "days_since_payroll_release"
            ]
        )
    )

    rows = []

    for (
        fold_name,
        release_bucket,
    ), group in data.groupby(
        [
            "fold",
            "release_age_bucket",
        ],
        observed=True,
        sort=False,
    ):

        metrics = paired_metrics(group)

        rows.append(
            {
                "fold": fold_name,
                "validation_year":
                    int(
                        group[
                            "validation_year"
                        ].iloc[0]
                    ),
                "release_age_bucket":
                    str(release_bucket),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def moving_block_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate moving-block bootstrap indices."""

    if block_length <= 0:
        raise ValueError(
            "block_length must be positive."
        )

    if n < block_length:
        raise ValueError(
            "Sample is shorter than block length."
        )

    blocks_needed = ceil(
        n / block_length
    )

    max_start = (
        n - block_length
    )

    starts = rng.integers(
        low=0,
        high=max_start + 1,
        size=blocks_needed,
    )

    indices = np.concatenate(
        [
            np.arange(
                start,
                start + block_length,
            )
            for start in starts
        ]
    )

    return indices[:n]


def bootstrap_fold(
    fold_data: pd.DataFrame,
    block_length: int,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap one validation fold."""

    rng = np.random.default_rng(seed)

    rows = []

    for _ in range(n_bootstrap):

        indices = moving_block_indices(
            n=len(fold_data),
            block_length=block_length,
            rng=rng,
        )

        sample = fold_data.iloc[
            indices
        ]

        metrics = paired_metrics(
            sample
        )

        rows.append(
            {
                "delta_auc":
                    metrics["delta_auc"],
                "delta_brier":
                    metrics["delta_brier"],
                "delta_log_loss":
                    metrics[
                        "delta_log_loss"
                    ],
            }
        )

    return pd.DataFrame(rows)


def summarize_bootstrap_distribution(
    actual_metrics: dict,
    bootstrap_results: pd.DataFrame,
    scope: str,
    block_length: int,
) -> list[dict]:
    """Summarize confidence intervals for paired deltas."""

    rows = []

    metric_names = [
        "delta_auc",
        "delta_brier",
        "delta_log_loss",
    ]

    for metric in metric_names:

        distribution = (
            bootstrap_results[
                metric
            ].dropna()
        )

        rows.append(
            {
                "scope": scope,
                "block_length":
                    block_length,
                "metric": metric,
                "actual":
                    actual_metrics[metric],
                "ci_2_5":
                    distribution.quantile(
                        0.025
                    ),
                "ci_97_5":
                    distribution.quantile(
                        0.975
                    ),
                "bootstrap_share_positive":
                    (
                        distribution > 0
                    ).mean(),
                "bootstrap_repetitions":
                    len(distribution),
            }
        )

    return rows


def build_block_bootstrap_analysis(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform paired moving-block bootstrap.

    Block sizes 5 and 10 are prespecified before
    inspection of robustness results.
    """

    rows = []

    fold_groups = list(
        predictions.groupby(
            "fold",
            sort=False,
        )
    )

    for block_length in BLOCK_LENGTHS:

        # Individual-fold intervals.
        for fold_number, (
            fold_name,
            fold_data,
        ) in enumerate(
            fold_groups,
            start=1,
        ):

            actual = paired_metrics(
                fold_data
            )

            bootstrap_results = (
                bootstrap_fold(
                    fold_data=fold_data,
                    block_length=block_length,
                    n_bootstrap=N_BOOTSTRAP,
                    seed=(
                        RANDOM_SEED
                        + block_length * 100
                        + fold_number
                    ),
                )
            )

            rows.extend(
                summarize_bootstrap_distribution(
                    actual_metrics=actual,
                    bootstrap_results=(
                        bootstrap_results
                    ),
                    scope=fold_name,
                    block_length=block_length,
                )
            )

        # Across-fold mean improvement.
        actual_fold_metrics = [
            paired_metrics(group)
            for _, group in fold_groups
        ]

        actual_mean = {
            metric: np.nanmean(
                [
                    fold_metrics[
                        metric
                    ]
                    for fold_metrics
                    in actual_fold_metrics
                ]
            )
            for metric in [
                "delta_auc",
                "delta_brier",
                "delta_log_loss",
            ]
        }

        rng = np.random.default_rng(
            RANDOM_SEED
            + block_length * 1000
        )

        bootstrap_mean_rows = []

        for _ in range(N_BOOTSTRAP):

            fold_metric_rows = []

            for _, fold_data in fold_groups:

                indices = moving_block_indices(
                    n=len(fold_data),
                    block_length=(
                        block_length
                    ),
                    rng=rng,
                )

                sample = fold_data.iloc[
                    indices
                ]

                fold_metric_rows.append(
                    paired_metrics(
                        sample
                    )
                )

            bootstrap_mean_rows.append(
                {
                    metric: np.nanmean(
                        [
                            result[metric]
                            for result
                            in fold_metric_rows
                        ]
                    )
                    for metric in [
                        "delta_auc",
                        "delta_brier",
                        "delta_log_loss",
                    ]
                }
            )

        bootstrap_means = pd.DataFrame(
            bootstrap_mean_rows
        )

        rows.extend(
            summarize_bootstrap_distribution(
                actual_metrics=actual_mean,
                bootstrap_results=(
                    bootstrap_means
                ),
                scope="mean_across_folds",
                block_length=block_length,
            )
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Run PAYEMS-A robustness diagnostics."""

    print("=" * 72)
    print("PAYEMS-A ROBUSTNESS ANALYSIS")
    print("=" * 72)

    print(
        "\nLocked OOS year:",
        LOCKED_OOS_YEAR,
    )

    print(
        "Bootstrap block lengths:",
        BLOCK_LENGTHS,
    )

    print(
        "Bootstrap repetitions:",
        N_BOOTSTRAP,
    )

    df = load_evaluation_data()

    predictions, coefficients = (
        generate_predictions_and_coefficients(
            df
        )
    )

    if (
        predictions[
            "validation_year"
        ]
        >= LOCKED_OOS_YEAR
    ).any():
        raise AssertionError(
            "Locked OOS observations found."
        )

    probability_shift = (
        build_probability_shift_summary(
            predictions
        )
    )

    calibration = build_calibration_table(
        predictions
    )

    release_age = (
        build_release_age_analysis(
            predictions
        )
    )

    bootstrap = (
        build_block_bootstrap_analysis(
            predictions
        )
    )

    PREDICTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    coefficients.to_csv(
        COEFFICIENTS_FILE,
        index=False,
    )

    probability_shift.to_csv(
        PROBABILITY_SHIFT_FILE,
        index=False,
    )

    calibration.to_csv(
        CALIBRATION_FILE,
        index=False,
    )

    release_age.to_csv(
        RELEASE_AGE_FILE,
        index=False,
    )

    bootstrap.to_csv(
        BOOTSTRAP_FILE,
        index=False,
    )

    print("\n")
    print("=" * 72)
    print("PAYEMS-A MACRO COEFFICIENTS")
    print("=" * 72)

    macro_coefficients = coefficients.loc[
        (
            coefficients["model"]
            == PAYEMS_A_NAME
        )
        & (
            coefficients["feature"]
            .isin(PAYEMS_A_FEATURES)
        )
    ]

    print(
        macro_coefficients[
            [
                "validation_year",
                "feature",
                "standardized_coefficient",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n")
    print("=" * 72)
    print("PROBABILITY SHIFT")
    print("=" * 72)

    print(
        probability_shift.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n")
    print("=" * 72)
    print("RELEASE-AGE ANALYSIS")
    print("=" * 72)

    print(
        release_age[
            [
                "validation_year",
                "release_age_bucket",
                "n",
                "delta_auc",
                "delta_brier",
                "delta_log_loss",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n")
    print("=" * 72)
    print("BLOCK BOOTSTRAP")
    print("=" * 72)

    print(
        bootstrap.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        "\nSaved robustness outputs "
        "under reports/."
    )


if __name__ == "__main__":
    main()