import csv
import hashlib
import json

import pytest

from scripts.score_forward_forecast import score_forecast


@pytest.fixture
def scoring_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    source_dir = tmp_path / "data/raw/schwab"
    source_dir.mkdir(parents=True)

    source = source_dir / "TMF_1d_test.csv"
    source.write_text("original market data snapshot\n")

    prices = tmp_path / "prices.csv"
    rows = [
        ("2026-10-09", 25.99),
        ("2026-10-12", 26.08),
        ("2026-10-13", 26.38),
        ("2026-10-14", 26.89),
        ("2026-10-15", 28.00),
        ("2026-10-16", 28.04),
    ]

    with prices.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["market_date", "close"])
        writer.writerows(rows)

    forecast = tmp_path / "forecast.json"
    data = {
        "model": "V3_FINAL",
        "issued_at_utc": "2026-10-12T01:10:10+00:00",
        "forecast_origin": "2026-10-09",
        "forecast_target_date": "2026-10-16",
        "horizon_trading_days": 5,
        "tmf_close": 25.99,
        "probability_positive_return": 0.6046196578258294,
        "source_snapshot": source.name,
        "source_sha256": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
    }
    forecast.write_text(json.dumps(data))

    return forecast, prices, tmp_path / "evaluations", source


def modify_forecast(path, **changes):
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


def test_correct_score(scoring_case):
    forecast, prices, output_dir, _ = scoring_case
    output, result = score_forecast(
        forecast, prices, output_dir
    )

    assert output.exists()
    assert result["direction_correct"] is True
    assert result["order_generated"] is False
    assert result["realized_return_pct"] == pytest.approx(
        7.887649, abs=0.00001
    )
    assert result["brier_score"] == pytest.approx(
        0.156326, abs=0.000001
    )


def test_duplicate_evaluation_blocked(scoring_case):
    forecast, prices, output_dir, _ = scoring_case
    score_forecast(forecast, prices, output_dir)

    with pytest.raises(FileExistsError):
        score_forecast(forecast, prices, output_dir)


def test_source_tampering_blocked(scoring_case):
    forecast, prices, output_dir, source = scoring_case
    source.write_text("altered market data\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        score_forecast(forecast, prices, output_dir)


def test_origin_price_mismatch_blocked(scoring_case):
    forecast, prices, output_dir, _ = scoring_case
    modify_forecast(forecast, tmf_close=26.50)

    with pytest.raises(ValueError, match="Origin price"):
        score_forecast(forecast, prices, output_dir)


def test_invalid_probability_blocked(scoring_case):
    forecast, prices, output_dir, _ = scoring_case
    modify_forecast(forecast, probability_positive_return=1.25)

    with pytest.raises(ValueError, match="probability"):
        score_forecast(forecast, prices, output_dir)


def test_incorrect_horizon_blocked(scoring_case):
    forecast, prices, output_dir, _ = scoring_case
    modify_forecast(forecast, horizon_trading_days=4)

    with pytest.raises(ValueError, match="Expected"):
        score_forecast(forecast, prices, output_dir)


def test_missing_target_price_blocked(scoring_case):
    forecast, prices, output_dir, _ = scoring_case
    lines = prices.read_text().splitlines()
    prices.write_text("\n".join(lines[:-1]) + "\n")

    with pytest.raises(ValueError, match="missing"):
        score_forecast(forecast, prices, output_dir)


def test_duplicate_market_date_blocked(scoring_case):
    forecast, prices, output_dir, _ = scoring_case
    with prices.open("a") as handle:
        handle.write("2026-10-16,28.04\n")

    with pytest.raises(ValueError, match="Duplicate"):
        score_forecast(forecast, prices, output_dir)


def test_future_target_blocked(scoring_case):
    forecast, prices, output_dir, _ = scoring_case
    modify_forecast(
        forecast,
        forecast_target_date="2028-10-16",
    )

    with pytest.raises(ValueError, match="not yet safely completed"):
        score_forecast(forecast, prices, output_dir)
