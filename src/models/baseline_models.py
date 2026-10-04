"""Walk-forward baseline models for TMF 5-day returns."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.purged_split import (
    expanding_walk_forward_splits,
)


INPUT_FILE = Path(
    "data/features/TMF_1d_technical_treasury_features.csv"
)

OUTPUT_FILE = Path(
    "reports/TMF_baseline_model_results.csv"
)

TARGET = "forward_return_5"
HORIZON = 5


TECHNICAL_FEATURES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "ema_spread_pct",
    "close_vs_ema9",
    "close_vs_ema21",
    "rsi_14",
    "momentum_5d_pct",
    "momentum_10d_pct",
    "volatility_5d",
    "volatility_20d",
    "range_pct",
    "relative_volume",
    "log_volume",
    "distance_from_20d_high",
    "distance_from_20d_low",
]


TREASURY_FEATURES = [
    "yield_2y",
    "yield_5y",
    "yield_10y",
    "yield_30y",
    "change_2y_1d_bp",
    "change_5y_1d_bp",
    "change_10y_1d_bp",
    "change_30y_1d_bp",
    "change_2y_5d_bp",
    "change_5y_5d_bp",
    "change_10y_5d_bp",
    "change_30y_5d_bp",
    "change_2y_20d_bp",
    "change_5y_20d_bp",
    "change_10y_20d_bp",
    "change_30y_20d_bp",
    "curve_10y_2y_bp",
    "curve_30y_2y_bp",
    "curve_30y_5y_bp",
    "curve_30y_10y_bp",
    "curve_10y_2y_bp_change_5d",
    "curve_30y_2y_bp_change_5d",
    "curve_30y_5y_bp_change_5d",
    "curve_30y_10y_bp_change_5d",
    "yield_10y_volatility_20d",
    "yield_30y_volatility_20d",
]


SELECTED_TECHNICAL_FEATURES = [
    "return_5d",
    "rsi_14",
    "volatility_20d",
    "relative_volume",
    "distance_from_20d_high",
]


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


SELECTED_COMBINED_FEATURES = (
    SELECTED_TECHNICAL_FEATURES
    + SELECTED_TREASURY_FEATURES
)


MODEL_FEATURES = {
    "technical_ridge": TECHNICAL_FEATURES,
    "treasury_ridge": TREASURY_FEATURES,
    "combined_ridge": (
        TECHNICAL_FEATURES
        + TREASURY_FEATURES
    ),
    "selected_technical_ridge":
        SELECTED_TECHNICAL_FEATURES,
    "selected_treasury_ridge":
        SELECTED_TREASURY_FEATURES,
    "selected_combined_ridge":
        SELECTED_COMBINED_FEATURES,
}


def add_model_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add economically motivated model features."""

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


def build_ridge_model() -> Pipeline:
    """Build preprocessing + Ridge regression pipeline."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(alpha=1.0),
            ),
        ]
    )


def calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict:
    """Calculate regression and directional metrics."""

    mse = mean_squared_error(
        y_true,
        predictions,
    )

    directional_accuracy = np.mean(
        np.sign(y_true.to_numpy())
        == np.sign(predictions)
    )

    return {
        "mae": mean_absolute_error(
            y_true,
            predictions,
        ),
        "rmse": np.sqrt(mse),
        "r2": r2_score(
            y_true,
            predictions,
        ),
        "directional_accuracy":
            directional_accuracy,
    }


def evaluate_naive_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict:
    """Forecast validation returns using training mean."""

    train_mean = train[TARGET].mean()

    predictions = np.full(
        len(validation),
        train_mean,
    )

    return calculate_metrics(
        validation[TARGET],
        predictions,
    )


def evaluate_zero_model(
    validation: pd.DataFrame,
) -> dict:
    """Forecast zero return for every observation."""

    predictions = np.zeros(
        len(validation)
    )

    metrics = calculate_metrics(
        validation[TARGET],
        predictions,
    )

    # A zero forecast expresses no directional view.
    metrics["directional_accuracy"] = np.nan

    return metrics


def evaluate_ridge_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> dict:
    """Train Ridge on one fold and evaluate validation."""

    train_sample = train.dropna(
        subset=[TARGET]
    ).copy()

    validation_sample = validation.dropna(
        subset=[TARGET]
    ).copy()

    model = build_ridge_model()

    model.fit(
        train_sample[features],
        train_sample[TARGET],
    )

    predictions = model.predict(
        validation_sample[features]
    )

    return calculate_metrics(
        validation_sample[TARGET],
        predictions,
    )


def run_walk_forward(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate baseline models across purged folds."""

    folds = expanding_walk_forward_splits(
        df,
        horizon=HORIZON,
    )

    results = []

    for fold in folds:

        fold_name = fold["name"]

        train = fold["train"].dropna(
            subset=[TARGET]
        ).copy()

        validation = fold[
            "validation"
        ].dropna(
            subset=[TARGET]
        ).copy()

        # Historical-mean baseline.
        metrics = evaluate_naive_model(
            train,
            validation,
        )

        results.append(
            {
                "fold": fold_name,
                "model": "naive_mean",
                "train_rows": len(train),
                "validation_rows":
                    len(validation),
                **metrics,
            }
        )

        # Zero-return baseline.
        metrics = evaluate_zero_model(
            validation
        )

        results.append(
            {
                "fold": fold_name,
                "model": "zero_return",
                "train_rows": len(train),
                "validation_rows":
                    len(validation),
                **metrics,
            }
        )

        # Ridge models.
        for model_name, features in (
            MODEL_FEATURES.items()
        ):

            metrics = evaluate_ridge_model(
                train=train,
                validation=validation,
                features=features,
            )

            results.append(
                {
                    "fold": fold_name,
                    "model": model_name,
                    "train_rows":
                        len(train),
                    "validation_rows":
                        len(validation),
                    **metrics,
                }
            )

    return pd.DataFrame(results)


def main() -> None:
    """Run TMF baseline modeling experiment."""

    print(
        "Loading TMF feature dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["market_date"],
    )

    # Construct derived model features before
    # creating the walk-forward folds.
    df = add_model_features(df)

    results = run_walk_forward(df)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("PURGED WALK-FORWARD RESULTS")
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
                "mae",
                "rmse",
                "r2",
                "directional_accuracy",
            ]
        ]
        .mean()
        .sort_values("rmse")
    )

    print(
        averages.to_string()
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()