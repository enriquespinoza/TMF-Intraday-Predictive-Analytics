import pandas as pd

from src.analysis.feature_research import (
    calculate_feature_relationships,
)


def test_feature_relationships_created():
    df = pd.DataFrame(
        {
            "return_1d": [
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
            ],
            "forward_return_1": [
                0.02,
                0.04,
                0.06,
                0.08,
                0.10,
            ],
        }
    )

    result = calculate_feature_relationships(df)

    row = result[
        (result["feature"] == "return_1d")
        & (
            result["target"]
            == "forward_return_1"
        )
    ]

    assert len(row) == 1


def test_perfect_positive_correlation():
    df = pd.DataFrame(
        {
            "return_1d": [
                1,
                2,
                3,
                4,
                5,
            ],
            "forward_return_1": [
                2,
                4,
                6,
                8,
                10,
            ],
        }
    )

    result = calculate_feature_relationships(df)

    row = result[
        (result["feature"] == "return_1d")
        & (
            result["target"]
            == "forward_return_1"
        )
    ].iloc[0]

    assert abs(
        row["pearson_corr"] - 1.0
    ) < 1e-10

    assert abs(
        row["spearman_corr"] - 1.0
    ) < 1e-10
    