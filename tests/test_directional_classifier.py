import numpy as np
import pandas as pd

from src.models.directional_classifier import (
    MODEL_FEATURES,
    SELECTED_TECHNICAL_FEATURES,
    V2A_CURVE_FEATURES,
    add_directional_target,
    build_logistic_model,
    calculate_classification_metrics,
)


EXPECTED_TECHNICAL_V1_FEATURES = [
    "return_5d",
    "rsi_14",
    "volatility_20d",
    "relative_volume",
    "distance_from_20d_high",
]


EXPECTED_V2A_CURVE_FEATURES = [
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


def test_technical_v1_feature_set_is_frozen():
    """Technical Baseline V1 must remain unchanged."""

    assert (
        SELECTED_TECHNICAL_FEATURES
        == EXPECTED_TECHNICAL_V1_FEATURES
    )


def test_v2a_curve_feature_set_is_prespecified():
    """V2A must contain exactly the planned curve factors."""

    assert (
        V2A_CURVE_FEATURES
        == EXPECTED_V2A_CURVE_FEATURES
    )


def test_v2a_curve_model_uses_only_curve_features():
    """V2A curve challenger must not include technical features."""

    assert (
        MODEL_FEATURES["v2a_curve_logistic"]
        == EXPECTED_V2A_CURVE_FEATURES
    )


def test_technical_v2a_model_combines_exact_feature_sets():
    """Combined challenger must equal Technical V1 + V2A."""

    expected = (
        EXPECTED_TECHNICAL_V1_FEATURES
        + EXPECTED_V2A_CURVE_FEATURES
    )

    assert (
        MODEL_FEATURES[
            "technical_v2a_curve_logistic"
        ]
        == expected
    )


def test_v2a_does_not_modify_existing_models():
    """Existing V1 model definitions must remain available."""

    assert "technical_logistic" in MODEL_FEATURES
    assert "treasury_logistic" in MODEL_FEATURES
    assert "combined_logistic" in MODEL_FEATURES

    assert (
        MODEL_FEATURES["technical_logistic"]
        == EXPECTED_TECHNICAL_V1_FEATURES
    )


def test_directional_target_preserves_missing_returns():
    """Unavailable forward returns must remain unlabeled."""

    df = pd.DataFrame(
        {
            "forward_return_5": [
                0.05,
                -0.02,
                0.00,
                np.nan,
            ]
        }
    )

    result = add_directional_target(df)

    assert result["target_up_5d"].iloc[0] == 1
    assert result["target_up_5d"].iloc[1] == 0
    assert result["target_up_5d"].iloc[2] == 0

    assert pd.isna(
        result["target_up_5d"].iloc[3]
    )


def test_logistic_model_configuration():
    """Classifier architecture must remain fixed for V2A."""

    pipeline = build_logistic_model()

    assert pipeline.named_steps[
        "imputer"
    ].strategy == "median"

    assert pipeline.named_steps[
        "model"
    ].C == 1.0

    assert pipeline.named_steps[
        "model"
    ].max_iter == 1000

    assert pipeline.named_steps[
        "model"
    ].random_state == 42


def test_classification_metrics_are_calculated():
    """Metric function should return all required metrics."""

    y_true = pd.Series(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.10, 0.30, 0.70, 0.90]
    )

    metrics = calculate_classification_metrics(
        y_true,
        probabilities,
    )

    expected = {
        "roc_auc",
        "brier_score",
        "log_loss",
        "accuracy",
        "balanced_accuracy",
    }

    assert expected == set(
        metrics.keys()
    )

    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0

def test_walk_forward_predictions_include_v2a_models():
    """Walk-forward output must include all V1 and V2A probabilities."""

    from src.models.directional_classifier import (
        generate_walk_forward_predictions,
    )

    # Create enough daily observations to cover the
    # 2021-2025 expanding walk-forward folds.
    dates = pd.bdate_range(
        "2021-10-04",
        "2025-12-31",
    )

    rows = len(dates)

    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "market_date": dates,

            "forward_return_5":
                rng.normal(
                    0.0,
                    0.03,
                    rows,
                ),

            "target_up_5d":
                rng.integers(
                    0,
                    2,
                    rows,
                ).astype(float),

            # Frozen Technical V1
            "return_5d":
                rng.normal(0, 0.03, rows),

            "rsi_14":
                rng.uniform(20, 80, rows),

            "volatility_20d":
                rng.uniform(0.01, 0.10, rows),

            "relative_volume":
                rng.uniform(0.5, 2.0, rows),

            "distance_from_20d_high":
                rng.uniform(-0.20, 0.0, rows),

            # Existing Treasury V1
            "yield_10y":
                rng.uniform(2.0, 6.0, rows),

            "yield_30y":
                rng.uniform(2.0, 6.0, rows),

            "change_2y_5d_bp":
                rng.normal(0, 15, rows),

            "change_30y_5d_bp":
                rng.normal(0, 15, rows),

            "curve_10y_2y_bp":
                rng.normal(0, 100, rows),

            "curve_30y_2y_bp":
                rng.normal(50, 100, rows),

            "yield_30y_volatility_20d":
                rng.uniform(1, 15, rows),

            "bear_flattening":
                rng.integers(0, 2, rows),

            # V2A structural curve
            "curve_level":
                rng.uniform(1.0, 6.0, rows),

            "curve_slope_30y_3m_bp":
                rng.normal(50, 100, rows),

            "curve_curvature_5y_bp":
                rng.normal(0, 100, rows),

            "curve_level_change_5d_bp":
                rng.normal(0, 15, rows),

            "curve_slope_change_5d_bp":
                rng.normal(0, 15, rows),

            "curve_curvature_change_5d_bp":
                rng.normal(0, 20, rows),

            "curve_level_volatility_20d":
                rng.uniform(1, 15, rows),

            "curve_slope_volatility_20d":
                rng.uniform(1, 15, rows),

            "curve_curvature_volatility_20d":
                rng.uniform(1, 20, rows),
        }
    )

    predictions = (
        generate_walk_forward_predictions(df)
    )

    expected_columns = {
        "base_probability",
        "technical_probability",
        "treasury_probability",
        "combined_probability",
        "v2a_curve_probability",
        "technical_v2a_curve_probability",
    }

    assert expected_columns.issubset(
        predictions.columns
    )

    # Every generated probability must be valid.
    for column in expected_columns:

        assert predictions[column].notna().all()

        assert predictions[column].between(
            0.0,
            1.0,
        ).all()

    # Confirm the locked 2026 holdout is absent.
    assert (
        predictions["market_date"].max()
        < pd.Timestamp("2026-01-01")
    )