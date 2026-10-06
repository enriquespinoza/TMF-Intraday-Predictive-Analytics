import pandas as pd

from src.analysis.compare_risk_governor_v2_v3 import (
    add_ema_streak,
)


def test_ema_streak_resets_after_recovery():
    data = pd.DataFrame(
        {
            "market_date":
                pd.to_datetime(
                    [
                        "2026-01-02",
                        "2026-01-05",
                        "2026-01-06",
                        "2026-01-07",
                        "2026-01-08",
                        "2026-01-09",
                    ]
                ),

            "below_ema21": [
                False,
                True,
                True,
                True,
                False,
                True,
            ],
        }
    )

    result = add_ema_streak(
        data
    )

    assert result[
        "below_ema21_streak"
    ].tolist() == [
        0,
        1,
        2,
        3,
        0,
        1,
    ]