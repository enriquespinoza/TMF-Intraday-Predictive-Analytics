"""Join a recorded V3 forecast to V4 reports without changing trading state.

This is a reporting adapter, NOT a V4 engine runner or broker interface.
Missing, stale, ambiguous, or invalid V4 inputs never authorize trading.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
FORECAST_DIR = ROOT / "data" / "forward_test"
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = REPORT_DIR / "generated"

DATE_COLUMNS = ("market_date", "session_date", "trading_date", "date", "timestamp")
NY = ZoneInfo("America/New_York")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def market_date(value: str, column: str) -> date:
    """Interpret explicit session dates as dates and timestamps in NY time."""
    raw = str(value).strip()
    if not raw:
        raise ValueError("Blank market date")
    if column != "timestamp":
        return date.fromisoformat(raw[:10])
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include timezone")
    return parsed.astimezone(NY).date()


def load_forecast(path: Path) -> dict:
    forecast = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(forecast, dict):
        raise ValueError("Forecast must be a JSON object")
    if forecast.get("model") != "V3_FINAL":
        raise ValueError("Expected a frozen V3_FINAL forecast")
    if forecast.get("horizon_trading_days") != 5:
        raise ValueError("Expected a five-trading-day forecast")
    origin = date.fromisoformat(forecast["forecast_origin"])
    target = date.fromisoformat(forecast["forecast_target_date"])
    issued = datetime.fromisoformat(
        forecast["issued_at_utc"].replace("Z", "+00:00")
    )
    if issued.tzinfo is None or target <= origin:
        raise ValueError("Invalid issuance time or target date")
    probability = float(forecast["probability_positive_return"])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid V3 directional probability")
    return {
        "model": "V3_FINAL",
        "forecast_origin": origin.isoformat(),
        "forecast_target_date": target.isoformat(),
        "issued_at_utc": issued.isoformat(),
        "horizon_trading_days": 5,
        "probability_positive_return": probability,
        "tmf_close": forecast.get("tmf_close"),
        "training_base_rate": forecast.get("training_base_rate"),
    }


def load_report(path: Path, as_of: date) -> dict:
    """Select exactly one same-day V4 row. Never substitute a stale row."""
    base = {"source": str(path), "status": "unavailable", "data": None}
    if not path.is_file():
        return base

    base["sha256"] = sha256_file(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            date_column = next((c for c in DATE_COLUMNS if c in columns), None)
            if date_column is None:
                raise ValueError("No recognized market-date column")
            rows = list(reader)
        if not rows:
            return {**base, "status": "empty"}
        dated = [(market_date(r[date_column], date_column), r) for r in rows]
    except (OSError, ValueError, csv.Error) as exc:
        return {**base, "status": "invalid_schema", "reason": str(exc)}

    known_dates = [d for d, _ in dated if d <= as_of]
    matching = [r for d, r in dated if d == as_of]
    if len(matching) > 1:
        return {**base, "status": "ambiguous_duplicate_date"}
    if not matching:
        return {
            **base,
            "status": "date_mismatch",
            "latest_available_as_of": max(known_dates).isoformat()
            if known_dates else None,
        }
    return {
        **base,
        "status": "matched",
        "market_date": as_of.isoformat(),
        "data": matching[0],
    }


def build_assessment(
    forecast_path: Path,
    opportunity_path: Path,
    risk_path: Path,
) -> dict:
    forecast_path = Path(forecast_path)
    forecast = load_forecast(forecast_path)
    as_of = date.fromisoformat(forecast["forecast_origin"])
    opportunity = load_report(Path(opportunity_path), as_of)
    risk = load_report(Path(risk_path), as_of)
    return {
        "schema_version": "v3_v4_readonly_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast": forecast,
        "forecast_source": {
            "path": str(forecast_path),
            "sha256": sha256_file(forecast_path),
        },
        "v4_opportunity": opportunity,
        "v4_risk": risk,
        "all_assessments_date_aligned": all(
            a["status"] == "matched" for a in (opportunity, risk)
        ),
        "trading": {
            "status": "NOT_AUTHORIZED_READ_ONLY",
            "orders_generated": False,
            "note": "V3 probability never overrides independent V4 restrictions.",
        },
    }


def latest_forecast(folder: Path) -> Path:
    candidates = sorted(folder.glob("TMF_V3_SCHWAB_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No V3 Schwab forecast found under {folder}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path)
    parser.add_argument(
        "--opportunity-csv", type=Path,
        default=REPORT_DIR / "V4_opportunity_state_machine_v1_daily.csv",
    )
    parser.add_argument(
        "--risk-csv", type=Path,
        default=REPORT_DIR / "V4_risk_governor_v3_daily.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    forecast_path = args.forecast or latest_forecast(FORECAST_DIR)
    assessment = build_assessment(
        forecast_path, args.opportunity_csv, args.risk_csv
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    origin = assessment["forecast"]["forecast_origin"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output_dir / f"TMF_V3_V4_{origin}_{stamp}.json"
    with destination.open("x", encoding="utf-8") as output:
        json.dump(assessment, output, indent=2, allow_nan=False)
        output.write("\n")

    print("TMF V3 + V4 READ-ONLY ASSESSMENT")
    print("Forecast origin:", origin)
    print("V3 P(up):", f"{assessment['forecast']['probability_positive_return']:.2%}")
    for key in ("v4_opportunity", "v4_risk"):
        section = assessment[key]
        print(key + ":", section["status"])
        if section.get("market_date"):
            print("  Report session:", section["market_date"])
        elif section.get("latest_available_as_of"):
            print("  Latest earlier session:", section["latest_available_as_of"])
    print("Trading status: NOT_AUTHORIZED_READ_ONLY")
    print("Saved:", destination)


if __name__ == "__main__":
    main()
