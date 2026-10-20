import argparse
import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_forecast(forecast_path, prices_path, output_dir):
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))

    origin = date.fromisoformat(forecast["forecast_origin"])
    target = date.fromisoformat(forecast["forecast_target_date"])
    issued = datetime.fromisoformat(forecast["issued_at_utc"])
    issued = issued.astimezone(timezone.utc)

    if target >= datetime.now(timezone.utc).date():
        raise ValueError("Target date is not yet safely completed")

    if issued.date() > target:
        raise ValueError("Forecast was issued after the target date")

    probability = float(forecast["probability_positive_return"])
    if not 0 <= probability <= 1:
        raise ValueError("Invalid forecast probability")

    # Verify the original historical input snapshot.
    original_snapshot = (
        Path("data/raw/schwab") / forecast["source_snapshot"]
    )

    if not original_snapshot.is_file():
        raise FileNotFoundError(original_snapshot)

    if sha256(original_snapshot) != forecast["source_sha256"]:
        raise ValueError("Original source snapshot hash mismatch")

    prices = {}

    with prices_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            day = date.fromisoformat(row["market_date"])

            if day in prices:
                raise ValueError(f"Duplicate market date: {day}")

            close = float(row["close"])
            if close <= 0:
                raise ValueError(f"Invalid close on {day}")

            prices[day] = close

    if origin not in prices or target not in prices:
        raise ValueError("Origin or target close missing")

    sessions = sorted(
        day for day in prices if origin < day <= target
    )

    if len(sessions) != forecast["horizon_trading_days"]:
        raise ValueError(
            f"Expected {forecast['horizon_trading_days']} sessions, "
            f"found {len(sessions)}"
        )

    original_close = float(forecast["tmf_close"])
    if abs(prices[origin] - original_close) > 0.0001:
        raise ValueError("Origin price does not match forecast")

    target_close = prices[target]
    outcome = int(target_close > original_close)
    predicted_direction = int(probability > 0.5)

    result = {
        "forecast_file": forecast_path.name,
        "forecast_sha256": sha256(forecast_path),
        "model": forecast["model"],
        "issued_at_utc": forecast["issued_at_utc"],
        "forecast_origin": origin.isoformat(),
        "forecast_target_date": target.isoformat(),
        "horizon_trading_days": len(sessions),
        "predicted_probability": probability,
        "origin_close": original_close,
        "target_close": target_close,
        "realized_return_pct": (
            target_close / original_close - 1
        ) * 100,
        "actual_positive_return": outcome,
        "direction_correct": predicted_direction == outcome,
        "brier_score": (probability - outcome) ** 2,
        "price_source": prices_path.name,
        "price_source_sha256": sha256(prices_path),
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "order_generated": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        forecast_path.stem + "_evaluation.json"
    )

    # Exclusive creation: never overwrite an earlier evaluation.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    return output, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", required=True, type=Path)
    parser.add_argument("--prices", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path("data/forward_test/evaluations"),
        type=Path,
    )
    args = parser.parse_args()

    output, result = score_forecast(
        args.forecast, args.prices, args.output_dir
    )

    print("Evaluation saved:", output)
    print("Realized return: {:.4f}%".format(
        result["realized_return_pct"]
    ))
    print("Direction correct:", result["direction_correct"])
    print("Brier score: {:.6f}".format(result["brier_score"]))


if __name__ == "__main__":
    main()
