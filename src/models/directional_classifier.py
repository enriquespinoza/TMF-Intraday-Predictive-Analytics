"""Directional classification models for TMF 5-day returns."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.purged_split import (
    expanding_walk_forward_splits,
)

INPUT_FILE = Path(
    "data/features/TMF_1d_technical_treasury_features.csv"
)

TARGET_RETURN = "forward_return_5"
TARGET_CLASS = "target_up_5d"

HORIZON = 5

OUTPUT_FILE = Path(
    "reports/TMF_directional_classifier_results.csv"
)
PREDICTIONS_FILE = Path(
    "reports/TMF_directional_predictions.csv"
)
# ---------------------------------------------------------
# Frozen Technical Baseline V1
# ---------------------------------------------------------

# Do not modify this feature set during V2A research.
SELECTED_TECHNICAL_FEATURES = [
    "return_5d",
    "rsi_14",
    "volatility_20d",
    "relative_volume",
    "distance_from_20d_high",
]


# ---------------------------------------------------------
# Existing V1 Treasury benchmark
# ---------------------------------------------------------

# Existing V1 technical features
SELECTED_TECHNICAL_FEATURES = [
    "return_5d",
    "rsi_14",
    "volatility_20d",
    "relative_volume",
    "distance_from_20d_high",
]


# Existing V1 Treasury features
SELECTED_TREASURY_FEATURES = [
    "yield_10y",
    "yield_30y",
    "change_2y_5d_bp",
    "change_30y_5d_bp",
    "curve_10y_2y_bp",
    "curve_30y_2y_bp",
    "yield_30y_volatility_20d",
    "bear_flattening",
]


# Existing V2A empirical curve features
V2A_CURVE_FEATURES = [
    "curve_level",
    "curve_slope_30y_3m_bp",
    "curve_curvature_5y_bp",
    "curve_level_change_5d_bp",
    "curve_slope_change_5d_bp",
    "curve_curvature_change_5d_bp",
    "curve_level_volatility_20d",
    "curve_slope_volatility_20d",
    "curve_curvature_volatility_20d",
]


# NEW V2B Nelson-Siegel features
V2B_NELSON_SIEGEL_FEATURES = [
    "ns_beta0_level",
    "ns_beta1_slope",
    "ns_beta2_curvature",
    "ns_beta0_level_change_5d_bp",
    "ns_beta1_slope_change_5d_bp",
    "ns_beta2_curvature_change_5d_bp",
    "ns_beta0_level_volatility_20d",
    "ns_beta1_slope_volatility_20d",
    "ns_beta2_curvature_volatility_20d",
]


# Model definitions
MODEL_FEATURES = {
    # Frozen V1 benchmarks
    "technical_logistic":
        SELECTED_TECHNICAL_FEATURES,

    "treasury_logistic":
        SELECTED_TREASURY_FEATURES,

    "combined_logistic":
        SELECTED_TECHNICAL_FEATURES
        + SELECTED_TREASURY_FEATURES,

    # V2A empirical curve challengers
    "v2a_curve_logistic":
        V2A_CURVE_FEATURES,

    "technical_v2a_curve_logistic":
        SELECTED_TECHNICAL_FEATURES
        + V2A_CURVE_FEATURES,

    # V2B Nelson-Siegel challengers
    "v2b_nelson_siegel_logistic":
        V2B_NELSON_SIEGEL_FEATURES,

    "technical_v2b_nelson_siegel_logistic":
        SELECTED_TECHNICAL_FEATURES
        + V2B_NELSON_SIEGEL_FEATURES,
}

def add_directional_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create binary 5-day directional target."""

    data = df.copy()

    # Preserve unavailable future returns as NaN.
    data[TARGET_CLASS] = np.where(
        data[TARGET_RETURN].isna(),
        np.nan,
        (
            data[TARGET_RETURN] > 0
        ).astype(float),
    )

    return data


def add_model_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add derived classification features."""

    data = df.copy()

    data["bear_flattening"] = (
        (data["change_2y_5d_bp"] > 0)
        & (data["change_30y_5d_bp"] > 0)
        & (
            data["change_2y_5d_bp"]
            > data["change_30y_5d_bp"]
        )
    ).astype(int)

    return data


def build_logistic_model() -> Pipeline:
    """Build standardized logistic regression pipeline."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def calculate_classification_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict:
    """Calculate probability and classification metrics."""

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                predictions,
            ),
    }


def evaluate_base_rate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict:
    """Forecast the training-set bullish probability."""

    base_probability = (
        train[TARGET_CLASS].mean()
    )

    probabilities = np.full(
        len(validation),
        base_probability,
    )

    return calculate_classification_metrics(
        validation[TARGET_CLASS],
        probabilities,
    )


def evaluate_logistic_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> dict:
    """Train and evaluate logistic regression."""

    model = build_logistic_model()

    model.fit(
        train[features],
        train[TARGET_CLASS].astype(int),
    )

    probabilities = model.predict_proba(
        validation[features]
    )[:, 1]

    return calculate_classification_metrics(
        validation[TARGET_CLASS],
        probabilities,
    )
def generate_walk_forward_predictions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate observation-level validation predictions."""

    folds = expanding_walk_forward_splits(
        df,
        horizon=HORIZON,
    )

    prediction_frames = []

    for fold in folds:

        train = (
            fold["train"]
            .dropna(subset=[TARGET_CLASS])
            .copy()
        )

        validation = (
            fold["validation"]
            .dropna(subset=[TARGET_CLASS])
            .copy()
        )

        fold_name = fold["name"]

        # ---------------------------------
        # Training-set base probability
        # ---------------------------------

        base_probability = (
            train[TARGET_CLASS].mean()
        )

        output = pd.DataFrame(
            {
                "market_date":
                    validation["market_date"].values,

                "fold":
                    fold_name,

                "actual_up":
                    validation[
                        TARGET_CLASS
                    ].astype(int).values,

                "forward_return_5":
                    validation[
                        TARGET_RETURN
                    ].values,

                "base_probability":
                    base_probability,
            }
        )

        # ---------------------------------
        # Generate probability predictions
        # ---------------------------------

        for model_name, features in (
            MODEL_FEATURES.items()
        ):

            model = build_logistic_model()

            model.fit(
                train[features],
                train[
                    TARGET_CLASS
                ].astype(int),
            )

            probabilities = (
                model.predict_proba(
                    validation[features]
                )[:, 1]
            )

            probability_column = (
                model_name.replace(
                    "_logistic",
                    "_probability",
                )
            )

            output[
                probability_column
            ] = probabilities

        prediction_frames.append(
            output
        )

    return pd.concat(
        prediction_frames,
        ignore_index=True,
    )

def run_walk_forward(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run purged chronological classification tests."""

    folds = expanding_walk_forward_splits(
        df,
        horizon=HORIZON,
    )

    results = []

    for fold in folds:

        train = (
            fold["train"]
            .dropna(subset=[TARGET_CLASS])
            .copy()
        )

        validation = (
            fold["validation"]
            .dropna(subset=[TARGET_CLASS])
            .copy()
        )

        fold_name = fold["name"]

        # -------------------------
        # Base-rate benchmark
        # -------------------------

        metrics = evaluate_base_rate(
            train,
            validation,
        )

        results.append(
            {
                "fold": fold_name,
                "model": "base_rate",
                "train_rows": len(train),
                "validation_rows":
                    len(validation),
                "train_up_rate":
                    train[TARGET_CLASS].mean(),
                "validation_up_rate":
                    validation[TARGET_CLASS].mean(),
                **metrics,
            }
        )

        # -------------------------
        # Logistic models
        # -------------------------

        for model_name, features in (
            MODEL_FEATURES.items()
        ):

            metrics = evaluate_logistic_model(
                train,
                validation,
                features,
            )

            results.append(
                {
                    "fold": fold_name,
                    "model": model_name,
                    "train_rows": len(train),
                    "validation_rows":
                        len(validation),
                    "train_up_rate":
                        train[TARGET_CLASS].mean(),
                    "validation_up_rate":
                        validation[
                            TARGET_CLASS
                        ].mean(),
                    **metrics,
                }
            )

    return pd.DataFrame(results)

def main() -> None:
    """Run TMF directional classification experiment."""

    print(
        "Loading TMF feature dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["market_date"],
    )

    df = add_directional_target(df)
    df = add_model_features(df)

    print()
    print("DIRECTIONAL TARGET")
    print()

    print(
        df[TARGET_CLASS]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Missing targets:",
        df[TARGET_CLASS].isna().sum(),
    )

    # ---------------------------------
    # Model-level walk-forward results
    # ---------------------------------

    results = run_walk_forward(df)

    # ---------------------------------
    # Observation-level predictions
    # ---------------------------------

    predictions = (
        generate_walk_forward_predictions(
            df
        )
    )

    # ---------------------------------
    # Save outputs
    # ---------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # ---------------------------------
    # Print model results
    # ---------------------------------

    print()
    print(
        "PURGED DIRECTIONAL WALK-FORWARD RESULTS"
    )
    print()

    print(
        results.to_string(
            index=False
        )
    )

    print()
    print(
        "AVERAGE VALIDATION PERFORMANCE"
    )
    print()

    averages = (
        results.groupby("model")[
            [
                "roc_auc",
                "brier_score",
                "log_loss",
                "accuracy",
                "balanced_accuracy",
            ]
        ]
        .mean()
        .sort_values(
            "roc_auc",
            ascending=False,
        )
    )

    print(
        averages.to_string()
    )

    print()
    print(
        f"Results saved to: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Predictions saved to: "
        f"{PREDICTIONS_FILE}"
    )


if __name__ == "__main__":
    main()