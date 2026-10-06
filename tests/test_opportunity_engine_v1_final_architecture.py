import pandas as pd
import pytest

from src.analysis.analyze_opportunity_engine_v1_final_architecture import (
    add_temporal_partitions,
    build_comparison,
    classify_treasury,
)


def test_treasury_architecture_classification():
    assert (
        classify_treasury(
            "Q1_LOW"
        )
        == "NON_EXTREME"
    )

    assert (
        classify_treasury(
            "Q2"
        )
        == "NON_EXTREME"
    )

    assert (
        classify_treasury(
            "Q3"
        )
        == "NON_EXTREME"
    )

    assert (
        classify_treasury(
            "Q4_HIGH"
        )
        == "EXTREME"
    )


def test_unknown_treasury_quartile_rejected():
    with pytest.raises(
        ValueError,
        match="Unexpected Treasury quartile",
    ):
        classify_treasury(
            "Q5"
        )


def test_blocks_created_before_technical_filter():
    data = pd.DataFrame(
        {
            "market_date":
                pd.date_range(
                    "2025-01-01",
                    periods=40,
                    freq="D",
                ),

            "technical_quartile":
                (
                    ["Q1_LOW"] * 20
                    + ["Q4_HIGH"] * 20
                ),
        }
    )

    result = add_temporal_partitions(
        data
    )

    assert (
        result[
            "chronological_block"
        ].nunique()
        == 4
    )

    q4 = result.loc[
        result[
            "technical_quartile"
        ]
        == "Q4_HIGH"
    ]

    # Technical filtering happens after block assignment.
    assert len(
        q4
    ) == 20


def test_primary_comparison_supported_when_return_and_mae_better():
    data = pd.DataFrame(
        {
            "treasury_architecture_state":
                (
                    ["NON_EXTREME"] * 6
                    + ["EXTREME"] * 6
                ),

            "forward_return_1":
                [0.0] * 12,

            "forward_return_5":
                (
                    [0.02] * 6
                    + [-0.01] * 6
                ),

            "forward_5d_mae":
                (
                    [-0.02] * 6
                    + [-0.05] * 6
                ),

            "forward_5d_mfe":
                (
                    [0.04] * 6
                    + [0.03] * 6
                ),
        }
    )

    result = build_comparison(
        data,
        period_scheme="test",
        period="TEST",
    )

    assert result[
        "return_5d_delta"
    ] == pytest.approx(
        0.03
    )

    assert result[
        "mae_delta"
    ] == pytest.approx(
        0.03
    )

    assert bool(
        result[
            "eligible"
        ]
    ) is True

    assert bool(
        result[
            "directionally_supported"
        ]
    ) is True


def test_better_return_but_worse_mae_is_not_supported():
    data = pd.DataFrame(
        {
            "treasury_architecture_state":
                (
                    ["NON_EXTREME"] * 6
                    + ["EXTREME"] * 6
                ),

            "forward_return_1":
                [0.0] * 12,

            "forward_return_5":
                (
                    [0.03] * 6
                    + [0.01] * 6
                ),

            "forward_5d_mae":
                (
                    [-0.08] * 6
                    + [-0.03] * 6
                ),

            "forward_5d_mfe":
                [0.04] * 12,
        }
    )

    result = build_comparison(
        data,
        period_scheme="test",
        period="TEST",
    )

    assert (
        result[
            "return_5d_delta"
        ]
        > 0
    )

    assert (
        result[
            "mae_delta"
        ]
        < 0
    )

    assert bool(
        result[
            "directionally_supported"
        ]
    ) is False


def test_minimum_sample_required():
    data = pd.DataFrame(
        {
            "treasury_architecture_state":
                (
                    ["NON_EXTREME"] * 4
                    + ["EXTREME"] * 10
                ),

            "forward_return_1":
                [0.0] * 14,

            "forward_return_5":
                (
                    [0.10] * 4
                    + [-0.10] * 10
                ),

            "forward_5d_mae":
                (
                    [-0.01] * 4
                    + [-0.10] * 10
                ),

            "forward_5d_mfe":
                [0.05] * 14,
        }
    )

    result = build_comparison(
        data,
        period_scheme="test",
        period="TEST",
    )

    assert bool(
        result[
            "eligible"
        ]
    ) is False

    assert bool(
        result[
            "directionally_supported"
        ]
    ) is False