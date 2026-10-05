import numpy as np

from config.payems_candidates import (
    LOCKED_OOS_YEAR,
    PAYEMS_A_FEATURES,
    PAYEMS_A_NAME,
)

from src.analysis.evaluate_payems_candidates import (
    MODEL_SPECS,
)

from src.analysis.payems_a_robustness import (
    BLOCK_LENGTHS,
    TECHNICAL_MODEL_NAME,
    load_evaluation_data,
    moving_block_indices,
)


def test_robustness_uses_frozen_payems_a():
    expected = (
        tuple(
            MODEL_SPECS[
                TECHNICAL_MODEL_NAME
            ]
        )
        + tuple(PAYEMS_A_FEATURES)
    )

    assert (
        MODEL_SPECS[
            PAYEMS_A_NAME
        ]
        == expected
    )


def test_prespecified_block_lengths():
    assert BLOCK_LENGTHS == (5, 10)


def test_2026_remains_locked():
    assert LOCKED_OOS_YEAR == 2026


def test_loaded_data_excludes_2026():
    df = load_evaluation_data()

    assert (
        df["market_date"]
        .dt.year
        .max()
        < LOCKED_OOS_YEAR
    )


def test_moving_block_bootstrap_length():
    rng = np.random.default_rng(42)

    indices = moving_block_indices(
        n=100,
        block_length=5,
        rng=rng,
    )

    assert len(indices) == 100


def test_moving_block_indices_in_bounds():
    rng = np.random.default_rng(42)

    indices = moving_block_indices(
        n=100,
        block_length=10,
        rng=rng,
    )

    assert indices.min() >= 0
    assert indices.max() < 100
    