import numpy as np
import pandas as pd

from src.models.baseline_models import (
    build_ridge_model,
    calculate_metrics,
)


def test_ridge_pipeline_predicts():

    X = pd.DataFrame(
        {
            "feature_1": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "feature_2": [
                4.0,
                3.0,
                2.0,
                1.0,
            ],
        }
    )

    y = pd.Series(
        [
            0.01,
            0.02,
            -0.01,
            0.03,
        ]
    )

    model = build_ridge_model()

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)


def test_metrics_perfect_prediction():

    y = pd.Series(
        [
            0.01,
            -0.02,
            0.03,
        ]
    )

    predictions = np.array(
        [
            0.01,
            -0.02,
            0.03,
        ]
    )

    metrics = calculate_metrics(
        y,
        predictions,
    )

    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["r2"] == 1
    assert (
        metrics["directional_accuracy"]
        == 1
    )