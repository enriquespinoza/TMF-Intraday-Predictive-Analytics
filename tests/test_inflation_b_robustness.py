import numpy as np

from config.inflation_candidates import (
    INFLATION_B_FEATURES,
    INFLATION_B_NAME,
    LOCKED_OOS_YEAR,
)

from src.analysis.evaluate_inflation_candidates import (
    MODEL_SPECS,
    V3_BASE_FEATURES,
)

from src.analysis.inflation_b_robustness import (
    BLOCK_LENGTHS,
    load_evaluation_data,
    moving_block_indices,
)


def test_robustness_uses_frozen_inflation_b():
    expected = (
        V3_BASE_FEATURES
        + tuple(
            INFLATION_B_FEATURES
        )
    )

    assert (
        MODEL_SPECS[
            INFLATION_B_NAME
        ]
        == expected
    )


def test_prespecified_block_lengths():
    assert BLOCK_LENGTHS == (
        5,
        10,
    )


def test_2026_remains_locked():
    assert LOCKED_OOS_YEAR == 2026


def test_loaded_data_excludes_2026():
    df = load_evaluation_data()

    assert (
        df[
            "market_date"
        ].dt.year.max()
        < LOCKED_OOS_YEAR
    )


def test_moving_block_length():
    rng = np.random.default_rng(
        42
    )

    indices = (
        moving_block_indices(
            n=100,
            block_length=5,
            rng=rng,
        )
    )

    assert len(indices) == 100


def test_moving_block_indices_are_valid():
    rng = np.random.default_rng(
        42
    )

    indices = (
        moving_block_indices(
            n=100,
            block_length=10,
            rng=rng,
        )
    )

    assert indices.min() >= 0
    assert indices.max() < 100