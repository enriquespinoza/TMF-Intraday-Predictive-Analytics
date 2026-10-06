import pandas as pd
import pytest

from src.analysis.analyze_opportunity_engine_v1_temporal_robustness import (
    add_temporal_partitions,
    build_contrasts,
    calendar_period,
    filter_cell,
)


def test_calendar_periods():
    assert calendar_period(
        pd.Timestamp(
            "2022-12-01"
        )
    ) == "2022_2023"

    assert calendar_period(
        pd.Timestamp(
            "2023-06-01"
        )
    ) == "2022_2023"

    assert calendar_period(
        pd.Timestamp(
            "2024-06-01"
        )
    ) == "2024"

    assert calendar_period(
        pd.Timestamp(
            "2025-06-01"
        )
    ) == "2025"

    assert calendar_period(
        pd.Timestamp(
            "2026-06-01"
        )
    ) == "2026_YTD"


def test_chronological_blocks_preserve_order():
    data = pd.DataFrame(
        {
            "market_date":
                pd.date_range(
                    "2025-01-01",
                    periods=40,
                    freq="D",
                )
        }
    )

    result = (
        add_temporal_partitions(
            data
        )
    )

    assert result.iloc[
        0
    ][
        "chronological_block"
    ] == "BLOCK_1"

    assert result.iloc[
        -1
    ][
        "chronological_block"
    ] == "BLOCK_4"

    assert (
        result[
            "chronological_block"
        ]
        .nunique()
        == 4
    )


def test_filter_cell():
    data = pd.DataFrame(
        {
            "technical_quartile": [
                "Q1_LOW",
                "Q4_HIGH",
                "Q4_HIGH",
            ],

            "macro_quartile": [
                "Q1_LOW",
                "Q1_LOW",
                "Q4_HIGH",
            ],
        }
    )

    result = filter_cell(
        data,
        {
            "technical_quartile":
                "Q4_HIGH",

            "macro_quartile":
                "Q1_LOW",
        },
    )

    assert len(
        result
    ) == 1


def test_contrast_positive_mae_delta_means_preferred_less_adverse():
    cells = pd.DataFrame(
        {
            "period_scheme": [
                "calendar",
                "calendar",
            ],

            "period": [
                "2025",
                "2025",
            ],

            "cell": [
                "TREASURY_Q2",
                "TREASURY_Q4",
            ],

            "n": [
                20,
                20,
            ],

            "mean_forward_return_5": [
                0.02,
                -0.01,
            ],

            "positive_5d_rate": [
                0.55,
                0.40,
            ],

            "mean_forward_5d_mae": [
                -0.02,
                -0.05,
            ],

            "mean_forward_5d_mfe": [
                0.04,
                0.03,
            ],
        }
    )

    result = build_contrasts(
        cells
    )

    row = result.loc[
        result[
            "hypothesis"
        ]
        == "TREASURY_Q2_BEATS_Q4"
    ].iloc[0]

    assert row[
        "return_5d_delta"
    ] == pytest.approx(
        0.03
    )

    assert row[
        "mae_delta"
    ] == pytest.approx(
        0.03
    )

    assert bool(
        row[
            "directionally_supported"
        ]
    ) is True


def test_contrast_requires_minimum_sample():
    cells = pd.DataFrame(
        {
            "period_scheme": [
                "calendar",
                "calendar",
            ],

            "period": [
                "2025",
                "2025",
            ],

            "cell": [
                "TREASURY_Q2",
                "TREASURY_Q4",
            ],

            "n": [
                4,
                20,
            ],

            "mean_forward_return_5": [
                0.10,
                -0.10,
            ],

            "positive_5d_rate": [
                1.0,
                0.0,
            ],

            "mean_forward_5d_mae": [
                -0.01,
                -0.10,
            ],

            "mean_forward_5d_mfe": [
                0.10,
                0.01,
            ],
        }
    )

    result = build_contrasts(
        cells
    )

    row = result.loc[
        result[
            "hypothesis"
        ]
        == "TREASURY_Q2_BEATS_Q4"
    ].iloc[0]

    assert bool(
        row[
            "eligible"
        ]
    ) is False

    assert bool(
        row[
            "directionally_supported"
        ]
    ) is False