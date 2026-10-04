import pandas as pd

from src.analysis.treasury_regime_episodes import (
    get_episode_starts,
    identify_regime_episodes,
)


def test_episode_identification():

    df = pd.DataFrame(
        {
            "treasury_regime": [
                "bear_flattening",
                "bear_flattening",
                "bear_flattening",
                "bull_steepening",
                "bull_steepening",
                "bear_flattening",
            ]
        }
    )

    result = identify_regime_episodes(df)

    assert result[
        "regime_episode_id"
    ].nunique() == 3


def test_episode_starts():

    df = pd.DataFrame(
        {
            "treasury_regime": [
                "bear_flattening",
                "bear_flattening",
                "bull_steepening",
                "bull_steepening",
                "bear_flattening",
            ]
        }
    )

    result = get_episode_starts(df)

    assert len(result) == 3

    assert result[
        "treasury_regime"
    ].tolist() == [
        "bear_flattening",
        "bull_steepening",
        "bear_flattening",
    ]