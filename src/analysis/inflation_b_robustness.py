"""
INFLATION-B robustness analysis.

Compares the frozen V3 base:

    Technical V1 + PAYEMS-A

against frozen INFLATION-B:

    V3 base
    + core_cpi_yoy_pct
    + core_cpi_3m_annualized_pct

Questions addressed
-------------------
1. Are Core CPI coefficient signs stable?
2. Are Brier/log-loss improvements broadly distributed?
3. Does probability calibration improve?
4. Is improvement concentrated near CPI state updates?
5. Does moving-block bootstrap support positive probability-score gains?

Rules
-----
- No feature selection.
- No hyperparameter tuning.
- Same matched observations as frozen inflation evaluation.
- Existing logistic-regression pipeline reused unchanged.
- 2026 remains locked.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config.inflation_candidates import (
    INFLATION_B_FEATURES,
    INFLATION_B_NAME,
    LOCKED_OOS_YEAR,
)

from src.analysis.evaluate_inflation_candidates import (
    MODEL_SPECS,
    V3_BASE_FEATURES,
    V3_BASE_NAME,
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


BLOCK_LENGTHS = (5, 10)
N_BOOTSTRAP = 2000
RANDOM_SEED = 42
CALIBRATION_BINS = 5


PREDICTIONS_FILE = Path(
    "reports/INFLATION_B_robustness_predictions.csv"
)

COEFFICIENTS_FILE = Path(
    "reports/INFLATION_B_coefficient_stability.csv"
)

SCORE_DISTRIBUTION_FILE = Path(
    "reports/INFLATION_B_score_distribution.csv"
)

CALIBRATION_FILE = Path(
    "reports/INFLATION_B_calibration_bins.csv"
)

CALIBRATION_SUMMARY_FILE = Path(
    "reports/INFLATION_B_calibration_summary.csv"
)

UPDATE_AGE_FILE = Path(
    "reports/INFLATION_B_update_age_analysis.csv"
)

BOOTSTRAP_FILE = Path(
    "reports/INFLATION_B_block_bootstrap.csv"
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
    """Calculate AUC only when both classes are present."""

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
    Fit the unchanged logistic pipeline.

    Logistic coefficients are standardized because the existing
    pipeline applies StandardScaler before LogisticRegression.
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
            "feature":
                list(features),
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
    """Generate paired V3-base and INFLATION-B predictions."""

    folds = expanding_walk_forward_splits(
        df=df,
        horizon=HORIZON,
    )

    prediction_frames = []
    coefficient_frames = []

    base_features = tuple(
        MODEL_SPECS[
            V3_BASE_NAME
        ]
    )

    inflation_features = tuple(
        MODEL_SPECS[
            INFLATION_B_NAME
        ]
    )

    for fold in folds:

        train, validation = (
            prepare_matched_fold(
                fold
            )
        )

        validation_year = int(
            validation[
                "market_date"
            ].dt.year.iloc[0]
        )

        if (
            validation_year
            >= LOCKED_OOS_YEAR
        ):
            raise AssertionError(
                "Locked 2026 observations entered "
                "INFLATION-B robustness analysis."
            )

        (
            base_probabilities,
            base_coefficients,
        ) = fit_and_predict(
            train=train,
            validation=validation,
            features=base_features,
        )

        (
            inflation_probabilities,
            inflation_coefficients,
        ) = fit_and_predict(
            train=train,
            validation=validation,
            features=inflation_features,
        )

        predictions = pd.DataFrame(
            {
                "fold":
                    fold["name"],
                "validation_year":
                    validation_year,
                "market_date":
                    validation[
                        "market_date"
                    ].to_numpy(),
                "target":
                    validation[
                        TARGET_CLASS
                    ].astype(int).to_numpy(),
                "inflation_state_date":
                    pd.to_datetime(
                        validation[
                            "inflation_state_date"
                        ],
                        errors="raise",
                    ).to_numpy(),

                "core_cpi_reference_date":
                    pd.to_datetime(
                        validation[
                            "core_cpi_reference_date"
                        ],
                        errors="raise",
                    ).to_numpy(),
                "v3_base_probability":
                    base_probabilities,
                "inflation_b_probability":
                    inflation_probabilities,
            }
        )

        predictions[
            "probability_shift"
        ] = (
            predictions[
                "inflation_b_probability"
            ]
            - predictions[
                "v3_base_probability"
            ]
        )

        predictions[
            "absolute_probability_shift"
        ] = predictions[
            "probability_shift"
        ].abs()

        y = predictions[
            "target"
        ].to_numpy()

        p_base = predictions[
            "v3_base_probability"
        ].to_numpy()

        p_inflation = predictions[
            "inflation_b_probability"
        ].to_numpy()

        predictions[
            "brier_improvement"
        ] = (
            (y - p_base) ** 2
            - (y - p_inflation) ** 2
        )

        predictions[
            "log_loss_improvement"
        ] = (
            observation_log_loss(
                y,
                p_base,
            )
            - observation_log_loss(
                y,
                p_inflation,
            )
        )

        predictions[
            "days_since_cpi_update"
        ] = (
            predictions[
                "market_date"
            ]
            - predictions[
                "inflation_state_date"
            ]
        ).dt.days

        prediction_frames.append(
            predictions
        )

        base_coefficients[
            "fold"
        ] = fold["name"]

        base_coefficients[
            "validation_year"
        ] = validation_year

        base_coefficients[
            "model"
        ] = V3_BASE_NAME

        inflation_coefficients[
            "fold"
        ] = fold["name"]

        inflation_coefficients[
            "validation_year"
        ] = validation_year

        inflation_coefficients[
            "model"
        ] = INFLATION_B_NAME

        coefficient_frames.extend(
            [
                base_coefficients,
                inflation_coefficients,
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
    """Calculate paired V3-base versus INFLATION-B metrics."""

    y = data[
        "target"
    ].to_numpy()

    p_base = data[
        "v3_base_probability"
    ].to_numpy()

    p_inflation = data[
        "inflation_b_probability"
    ].to_numpy()

    auc_base = safe_auc(
        y,
        p_base,
    )

    auc_inflation = safe_auc(
        y,
        p_inflation,
    )

    brier_base = np.mean(
        (y - p_base) ** 2
    )

    brier_inflation = np.mean(
        (y - p_inflation) ** 2
    )

    log_loss_base = np.mean(
        observation_log_loss(
            y,
            p_base,
        )
    )

    log_loss_inflation = np.mean(
        observation_log_loss(
            y,
            p_inflation,
        )
    )

    return {
        "n":
            len(data),
        "auc_base":
            auc_base,
        "auc_inflation_b":
            auc_inflation,
        "delta_auc":
            auc_inflation - auc_base,
        "brier_base":
            brier_base,
        "brier_inflation_b":
            brier_inflation,
        "delta_brier":
            brier_base
            - brier_inflation,
        "log_loss_base":
            log_loss_base,
        "log_loss_inflation_b":
            log_loss_inflation,
        "delta_log_loss":
            log_loss_base
            - log_loss_inflation,
    }


def positive_concentration(
    values: pd.Series,
    fraction: float = 0.05,
) -> float:
    """
    Measure how much positive improvement comes from the largest
    fraction of observations.

    Returns the share of total positive improvement contributed by
    the top fraction of all observations.
    """

    positive = (
        values.loc[
            values > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    if positive.empty:
        return np.nan

    count = max(
        1,
        ceil(
            len(values)
            * fraction
        ),
    )

    denominator = (
        positive.sum()
    )

    if denominator == 0:
        return np.nan

    return float(
        positive.head(
            count
        ).sum()
        / denominator
    )


def build_score_distribution_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Test whether score improvement is broad or driven by
    a handful of observations.
    """

    rows = []

    for fold_name, group in predictions.groupby(
        "fold",
        sort=False,
    ):

        rows.append(
            {
                "fold":
                    fold_name,
                "validation_year":
                    int(
                        group[
                            "validation_year"
                        ].iloc[0]
                    ),
                "n":
                    len(group),

                "mean_absolute_probability_shift":
                    group[
                        "absolute_probability_shift"
                    ].mean(),

                "median_absolute_probability_shift":
                    group[
                        "absolute_probability_shift"
                    ].median(),

                "p90_absolute_probability_shift":
                    group[
                        "absolute_probability_shift"
                    ].quantile(0.90),

                "max_absolute_probability_shift":
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

                "brier_share_positive":
                    (
                        group[
                            "brier_improvement"
                        ]
                        > 0
                    ).mean(),

                "brier_median_improvement":
                    group[
                        "brier_improvement"
                    ].median(),

                "brier_top_5pct_share_of_positive":
                    positive_concentration(
                        group[
                            "brier_improvement"
                        ]
                    ),

                "log_loss_share_positive":
                    (
                        group[
                            "log_loss_improvement"
                        ]
                        > 0
                    ).mean(),

                "log_loss_median_improvement":
                    group[
                        "log_loss_improvement"
                    ].median(),

                "log_loss_top_5pct_share_of_positive":
                    positive_concentration(
                        group[
                            "log_loss_improvement"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_calibration_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Construct within-fold probability-quintile calibration."""

    rows = []

    model_columns = {
        V3_BASE_NAME:
            "v3_base_probability",

        INFLATION_B_NAME:
            "inflation_b_probability",
    }

    for fold_name, fold_data in predictions.groupby(
        "fold",
        sort=False,
    ):

        for (
            model_name,
            probability_column,
        ) in model_columns.items():

            data = fold_data.copy()

            ranks = (
                data[
                    probability_column
                ]
                .rank(
                    method="first"
                )
            )

            data[
                "calibration_bin"
            ] = (
                pd.qcut(
                    ranks,
                    q=CALIBRATION_BINS,
                    labels=False,
                )
                + 1
            )

            grouped = (
                data.groupby(
                    "calibration_bin",
                    observed=True,
                )
                .agg(
                    n=(
                        "target",
                        "size",
                    ),
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

            grouped[
                "calibration_gap"
            ] = (
                grouped[
                    "observed_rate"
                ]
                - grouped[
                    "mean_probability"
                ]
            )

            grouped[
                "absolute_calibration_gap"
            ] = grouped[
                "calibration_gap"
            ].abs()

            grouped["fold"] = (
                fold_name
            )

            grouped[
                "validation_year"
            ] = int(
                fold_data[
                    "validation_year"
                ].iloc[0]
            )

            grouped["model"] = (
                model_name
            )

            rows.append(
                grouped
            )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def build_calibration_summary(
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize calibration using weighted absolute calibration
    error across five within-fold probability bins.
    """

    rows = []

    for (
        fold_name,
        model_name,
    ), group in calibration.groupby(
        [
            "fold",
            "model",
        ],
        sort=False,
    ):

        weights = (
            group["n"]
            / group["n"].sum()
        )

        ece = np.sum(
            weights
            * group[
                "absolute_calibration_gap"
            ]
        )

        rmse = np.sqrt(
            np.sum(
                weights
                * (
                    group[
                        "calibration_gap"
                    ]
                    ** 2
                )
            )
        )

        rows.append(
            {
                "fold":
                    fold_name,
                "validation_year":
                    int(
                        group[
                            "validation_year"
                        ].iloc[0]
                    ),
                "model":
                    model_name,
                "calibration_ece":
                    ece,
                "calibration_rmse":
                    rmse,
            }
        )

    summary = pd.DataFrame(
        rows
    )

    base = (
        summary.loc[
            summary["model"]
            == V3_BASE_NAME,
            [
                "fold",
                "calibration_ece",
                "calibration_rmse",
            ],
        ]
        .rename(
            columns={
                "calibration_ece":
                    "base_calibration_ece",
                "calibration_rmse":
                    "base_calibration_rmse",
            }
        )
    )

    summary = summary.merge(
        base,
        on="fold",
        how="left",
    )

    # Positive delta means INFLATION-B calibration is better.
    summary[
        "delta_calibration_ece"
    ] = (
        summary[
            "base_calibration_ece"
        ]
        - summary[
            "calibration_ece"
        ]
    )

    summary[
        "delta_calibration_rmse"
    ] = (
        summary[
            "base_calibration_rmse"
        ]
        - summary[
            "calibration_rmse"
        ]
    )

    return summary


def cpi_age_bucket(
    days_since_update: pd.Series,
) -> pd.Series:
    """Bucket market observations by age of latest CPI state."""

    return pd.cut(
        days_since_update,
        bins=[
            -0.1,
            0,
            5,
            15,
            np.inf,
        ],
        labels=[
            "update_day",
            "days_1_5",
            "days_6_15",
            "days_16_plus",
        ],
        include_lowest=True,
    )


def build_update_age_analysis(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Test whether INFLATION-B gain occurs only near CPI updates
    or persists through the state cycle.
    """

    data = predictions.copy()

    data[
        "cpi_age_bucket"
    ] = cpi_age_bucket(
        data[
            "days_since_cpi_update"
        ]
    )

    rows = []

    for (
        fold_name,
        age_bucket,
    ), group in data.groupby(
        [
            "fold",
            "cpi_age_bucket",
        ],
        observed=True,
        sort=False,
    ):

        metrics = paired_metrics(
            group
        )

        rows.append(
            {
                "fold":
                    fold_name,
                "validation_year":
                    int(
                        group[
                            "validation_year"
                        ].iloc[0]
                    ),
                "cpi_age_bucket":
                    str(
                        age_bucket
                    ),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


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
            "Sample shorter than block length."
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
    """Bootstrap paired metric differences for one fold."""

    rng = np.random.default_rng(
        seed
    )

    rows = []

    for _ in range(
        n_bootstrap
    ):

        indices = (
            moving_block_indices(
                n=len(
                    fold_data
                ),
                block_length=(
                    block_length
                ),
                rng=rng,
            )
        )

        sample = (
            fold_data.iloc[
                indices
            ]
        )

        metrics = paired_metrics(
            sample
        )

        rows.append(
            {
                "delta_auc":
                    metrics[
                        "delta_auc"
                    ],
                "delta_brier":
                    metrics[
                        "delta_brier"
                    ],
                "delta_log_loss":
                    metrics[
                        "delta_log_loss"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


def summarize_bootstrap(
    actual: dict,
    bootstrap: pd.DataFrame,
    scope: str,
    block_length: int,
) -> list[dict]:
    """Summarize paired bootstrap distributions."""

    rows = []

    for metric in [
        "delta_auc",
        "delta_brier",
        "delta_log_loss",
    ]:

        distribution = (
            bootstrap[
                metric
            ].dropna()
        )

        rows.append(
            {
                "scope":
                    scope,
                "block_length":
                    block_length,
                "metric":
                    metric,
                "actual":
                    actual[
                        metric
                    ],
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
                        distribution
                        > 0
                    ).mean(),
                "bootstrap_repetitions":
                    len(
                        distribution
                    ),
            }
        )

    return rows


def build_block_bootstrap_analysis(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Run 5-day and 10-day paired moving-block bootstrap."""

    rows = []

    fold_groups = list(
        predictions.groupby(
            "fold",
            sort=False,
        )
    )

    for block_length in (
        BLOCK_LENGTHS
    ):

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

            bootstrap = bootstrap_fold(
                fold_data=fold_data,
                block_length=(
                    block_length
                ),
                n_bootstrap=(
                    N_BOOTSTRAP
                ),
                seed=(
                    RANDOM_SEED
                    + block_length * 100
                    + fold_number
                ),
            )

            rows.extend(
                summarize_bootstrap(
                    actual=actual,
                    bootstrap=bootstrap,
                    scope=fold_name,
                    block_length=(
                        block_length
                    ),
                )
            )

        actual_fold_metrics = [
            paired_metrics(
                group
            )
            for _, group
            in fold_groups
        ]

        actual_mean = {
            metric: np.nanmean(
                [
                    result[
                        metric
                    ]
                    for result
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

        for _ in range(
            N_BOOTSTRAP
        ):

            fold_results = []

            for _, fold_data in (
                fold_groups
            ):

                indices = (
                    moving_block_indices(
                        n=len(
                            fold_data
                        ),
                        block_length=(
                            block_length
                        ),
                        rng=rng,
                    )
                )

                sample = (
                    fold_data.iloc[
                        indices
                    ]
                )

                fold_results.append(
                    paired_metrics(
                        sample
                    )
                )

            bootstrap_mean_rows.append(
                {
                    metric: np.nanmean(
                        [
                            result[
                                metric
                            ]
                            for result
                            in fold_results
                        ]
                    )
                    for metric in [
                        "delta_auc",
                        "delta_brier",
                        "delta_log_loss",
                    ]
                }
            )

        bootstrap_means = (
            pd.DataFrame(
                bootstrap_mean_rows
            )
        )

        rows.extend(
            summarize_bootstrap(
                actual=actual_mean,
                bootstrap=(
                    bootstrap_means
                ),
                scope=(
                    "mean_across_folds"
                ),
                block_length=(
                    block_length
                ),
            )
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    """Run frozen INFLATION-B robustness analysis."""

    print("=" * 72)
    print("INFLATION-B ROBUSTNESS ANALYSIS")
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

    (
        predictions,
        coefficients,
    ) = (
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

    score_distribution = (
        build_score_distribution_summary(
            predictions
        )
    )

    calibration = (
        build_calibration_table(
            predictions
        )
    )

    calibration_summary = (
        build_calibration_summary(
            calibration
        )
    )

    update_age = (
        build_update_age_analysis(
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

    score_distribution.to_csv(
        SCORE_DISTRIBUTION_FILE,
        index=False,
    )

    calibration.to_csv(
        CALIBRATION_FILE,
        index=False,
    )

    calibration_summary.to_csv(
        CALIBRATION_SUMMARY_FILE,
        index=False,
    )

    update_age.to_csv(
        UPDATE_AGE_FILE,
        index=False,
    )

    bootstrap.to_csv(
        BOOTSTRAP_FILE,
        index=False,
    )

    print("\n")
    print("=" * 72)
    print("INFLATION-B COEFFICIENTS")
    print("=" * 72)

    macro_coefficients = (
        coefficients.loc[
            (
                coefficients[
                    "model"
                ]
                == INFLATION_B_NAME
            )
            & (
                coefficients[
                    "feature"
                ].isin(
                    INFLATION_B_FEATURES
                )
            )
        ]
    )

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
    print("SCORE IMPROVEMENT DISTRIBUTION")
    print("=" * 72)

    print(
        score_distribution.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n")
    print("=" * 72)
    print("CALIBRATION SUMMARY")
    print("=" * 72)

    print(
        calibration_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n")
    print("=" * 72)
    print("CPI UPDATE-AGE ANALYSIS")
    print("=" * 72)

    print(
        update_age[
            [
                "validation_year",
                "cpi_age_bucket",
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
        "\nSaved INFLATION-B robustness "
        "outputs under reports/."
    )


if __name__ == "__main__":
    main()