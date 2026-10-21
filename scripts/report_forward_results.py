import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(evaluation_dir, forecast_dir, price_dir):
    records = []
    seen = set()

    for path in sorted(evaluation_dir.glob("*_evaluation.json")):
        evaluation = json.loads(path.read_text(encoding="utf-8"))

        forecast_path = forecast_dir / evaluation["forecast_file"]
        price_path = price_dir / evaluation["price_source"]

        for source, expected in (
            (forecast_path, evaluation["forecast_sha256"]),
            (price_path, evaluation["price_source_sha256"]),
        ):
            if not source.is_file():
                raise FileNotFoundError(source)
            if file_hash(source) != expected:
                raise ValueError(f"Source hash mismatch: {source}")

        forecast = json.loads(
            forecast_path.read_text(encoding="utf-8")
        )

        key = (
            evaluation["model"],
            evaluation["issued_at_utc"],
            evaluation["forecast_origin"],
            evaluation["forecast_target_date"],
        )
        if key in seen:
            raise ValueError(f"Duplicate forecast: {key}")
        seen.add(key)

        if evaluation["order_generated"] is not False:
            raise ValueError("Evaluation contains an order")

        p = float(forecast["probability_positive_return"])
        baseline = float(forecast["training_base_rate"])
        actual = int(evaluation["actual_positive_return"])

        if not (0 <= p <= 1 and 0 <= baseline <= 1):
            raise ValueError("Invalid prediction probability")
        if actual not in (0, 1):
            raise ValueError("Invalid actual outcome")
        if abs(p - float(evaluation["predicted_probability"])) > 1e-12:
            raise ValueError("Forecast probability mismatch")
        if abs((p - actual) ** 2 - float(evaluation["brier_score"])) > 1e-10:
            raise ValueError("Evaluation Brier score mismatch")
        if evaluation["model"] != forecast["model"]:
            raise ValueError("Model mismatch")

        records.append({
            "forecast": evaluation["forecast_file"],
            "origin": evaluation["forecast_origin"],
            "target": evaluation["forecast_target_date"],
            "probability": p,
            "baseline_probability": baseline,
            "actual": actual,
            "correct": int((p > 0.5) == bool(actual)),
            "brier": (p - actual) ** 2,
            "baseline_brier": (baseline - actual) ** 2,
            "return_pct": float(evaluation["realized_return_pct"]),
        })

    if not records:
        return {"count": 0, "records": []}

    return {
        "count": len(records),
        "directional_accuracy": mean(r["correct"] for r in records),
        "mean_brier": mean(r["brier"] for r in records),
        "baseline_mean_brier": mean(
            r["baseline_brier"] for r in records
        ),
        "mean_tmf_forward_return_pct": mean(
            r["return_pct"] for r in records
        ),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluations",
        type=Path,
        default=Path("data/forward_test/evaluations"),
    )
    parser.add_argument(
        "--forecasts",
        type=Path,
        default=Path("data/forward_test"),
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("data/raw/schwab"),
    )
    args = parser.parse_args()

    result = build_report(
        args.evaluations, args.forecasts, args.prices
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
