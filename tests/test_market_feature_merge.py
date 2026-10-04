import pandas as pd

from src.features.merge_market_features import (
    align_treasury_to_tmf,
)


def create_tmf_sample():
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
            ],
            "close": [
                30.0,
                31.0,
                32.0,
            ],
        }
    )


def create_treasury_sample():
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-02",
                "2026-01-06",
            ],
            "yield_2y": [
                4.00,
                4.10,
            ],
            "yield_5y": [
                4.20,
                4.30,
            ],
            "yield_10y": [
                4.40,
                4.50,
            ],
            "yield_30y": [
                4.60,
                4.70,
            ],
        }
    )


def test_merge_preserves_tmf_rows():

    tmf = create_tmf_sample()
    treasury = create_treasury_sample()

    result = align_treasury_to_tmf(
        tmf,
        treasury,
    )

    assert len(result) == len(tmf)


def test_exact_date_match():

    tmf = create_tmf_sample()
    treasury = create_treasury_sample()

    result = align_treasury_to_tmf(
        tmf,
        treasury,
    )

    first = result.iloc[0]

    assert first["yield_30y"] == 4.60
    assert first["treasury_data_age_days"] == 0


def test_previous_treasury_observation_used():

    tmf = create_tmf_sample()
    treasury = create_treasury_sample()

    result = align_treasury_to_tmf(
        tmf,
        treasury,
    )

    jan_5 = result[
        result["market_date"]
        == pd.Timestamp("2026-01-05")
    ].iloc[0]

    assert jan_5["yield_30y"] == 4.60

    assert (
        jan_5["treasury_observation_date"]
        == pd.Timestamp("2026-01-02")
    )

    assert jan_5["treasury_data_age_days"] == 3


def test_future_treasury_data_not_used():

    tmf = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-05",
            ],
            "close": [
                31.0,
            ],
        }
    )

    treasury = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-06",
            ],
            "yield_2y": [
                4.10,
            ],
            "yield_5y": [
                4.30,
            ],
            "yield_10y": [
                4.50,
            ],
            "yield_30y": [
                4.70,
            ],
        }
    )

    result = align_treasury_to_tmf(
        tmf,
        treasury,
    )

    assert pd.isna(
        result["yield_30y"].iloc[0]
    )