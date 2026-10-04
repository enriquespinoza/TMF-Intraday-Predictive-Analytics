import pandas as pd

from src.data.load_market_data import load_ohlcv_csv


def test_load_ohlcv_csv(tmp_path):
    path = tmp_path / "tmf.csv"

    sample = pd.DataFrame(
        {
            "Date": [
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
            ],
            "Open": [40.00, 40.50, 41.00],
            "High": [41.00, 41.50, 42.00],
            "Low": [39.50, 40.00, 40.50],
            "Close": [40.75, 41.25, 41.75],
            "Volume": [1_000_000, 1_100_000, 1_200_000],
        }
    )

    sample.to_csv(path, index=False)

    result = load_ohlcv_csv(path)

    assert "timestamp" in result.columns
    assert "close" in result.columns
    assert "volume" in result.columns

    assert len(result) == 3

    assert result["timestamp"].is_monotonic_increasing
    