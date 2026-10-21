import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.score_forward_forecast import score_forecast
from scripts.report_forward_results import build_report


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def ledger_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    forecast_dir = tmp_path / "data/forward_test"
    price_dir = tmp_path / "data/raw/schwab"
    evaluation_dir = forecast_dir / "evaluations"

    forecast_dir.mkdir(parents=True)
    price_dir.mkdir(parents=True)

    original = price_dir / "TMF_origin.csv"
    original.write_text("frozen source snapshot\n")

    prices = price_dir / "TMF_prices.csv"

    with prices.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["market_date", "close"])
        writer.writerows([
            ("2026-10-09", 25.99),
            ("2026-10-12", 26.08),
            ("2026-10-13", 26.38),
            ("2026-10-14", 26.89),
            ("2026-10-15", 28.00),
            ("2026-10-16", 28.04),
        ])

    forecast = forecast_dir / "forecast.json"
    forecast.write_text(json.dumps({
        "model": "V3_FINAL",
        "issued_at_utc": "2026-10-12T01:10:10+00:00",
        "forecast_origin": "2026-10-09",
        "forecast_target_date": "2026-10-16",
        "horizon_trading_days": 5,
        "tmf_close": 25.99,
        "probability_positive_return": 0.6046196578258294,
        "training_base_rate": 0.43355325164938735,
        "source_snapshot": original.name,
        "source_sha256": sha256(original),
    }))

    evaluation, _ = score_forecast(
        forecast, prices, evaluation_dir
    )

    return {
        "evaluations": evaluation_dir,
        "forecasts": forecast_dir,
        "prices": price_dir,
        "forecast_file": forecast,
        "price_file": prices,
        "evaluation_file": evaluation,
    }


def run_report(case):
    return build_report(
        case["evaluations"],
        case["forecasts"],
        case["prices"],
    )


def change_evaluation(case, **updates):
    path = case["evaluation_file"]
    data = json.loads(path.read_text())
    data.update(updates)
    path.write_text(json.dumps(data))


def test_empty_ledger(tmp_path):
    result = build_report(
        tmp_path / "evaluations",
        tmp_path / "forecasts",
        tmp_path / "prices",
    )
    assert result == {"count": 0, "records": []}


def test_valid_forecast_metrics(ledger_case):
    result = run_report(ledger_case)

    assert result["count"] == 1
    assert result["directional_accuracy"] == 1
    assert result["mean_brier"] == pytest.approx(
        0.15632561497776423
    )
    assert result["baseline_mean_brier"] == pytest.approx(
        0.3208619187169822
    )
    assert result["mean_tmf_forward_return_pct"] == pytest.approx(
        7.887649095806082
    )


def test_report_does_not_modify_evaluation(ledger_case):
    path = ledger_case["evaluation_file"]
    before = sha256(path)

    run_report(ledger_case)

    assert sha256(path) == before


def test_duplicate_forecast_rejected(ledger_case):
    source = ledger_case["evaluation_file"]
    duplicate = (
        ledger_case["evaluations"] / "duplicate_evaluation.json"
    )
    duplicate.write_bytes(source.read_bytes())

    with pytest.raises(ValueError, match="Duplicate forecast"):
        run_report(ledger_case)


def test_tampered_forecast_rejected(ledger_case):
    ledger_case["forecast_file"].write_text("modified forecast")

    with pytest.raises(ValueError, match="Source hash mismatch"):
        run_report(ledger_case)


def test_tampered_price_data_rejected(ledger_case):
    with ledger_case["price_file"].open("a") as handle:
        handle.write("2026-10-19,27.43\n")

    with pytest.raises(ValueError, match="Source hash mismatch"):
        run_report(ledger_case)


def test_incorrect_brier_rejected(ledger_case):
    change_evaluation(ledger_case, brier_score=0.999)

    with pytest.raises(ValueError, match="Brier score mismatch"):
        run_report(ledger_case)


def test_order_record_rejected(ledger_case):
    change_evaluation(ledger_case, order_generated=True)

    with pytest.raises(ValueError, match="contains an order"):
        run_report(ledger_case)


def test_probability_mismatch_rejected(ledger_case):
    change_evaluation(
        ledger_case,
        predicted_probability=0.75,
    )

    with pytest.raises(ValueError, match="probability mismatch"):
        run_report(ledger_case)
