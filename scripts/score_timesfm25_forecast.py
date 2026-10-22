"""Score a matured TimesFM 2.5 price-path forecast against verified Schwab data.

Never changes the forecast or its manifest. Outcome snapshots may be
validated retrospectively at their encoded capture time; that path must
never be used to issue new predictions.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.market_session import FINALIZATION_DELAY
from scripts.timesfm25_integrity import (
    SOURCE_NAME, audit_existing_run, manifest_path, sha256,
)
from scripts.validate_tmf_snapshot_pair import validate_snapshot_pair

ROOT = Path(__file__).resolve().parents[1]
FORECAST_DIR = ROOT / "data/forward_test/timesfm25"
SOURCE_DIR = ROOT / "data/raw/schwab"
HISTORY = ROOT / "data/raw/TMF_1d.csv"
OUTPUT_DIR = FORECAST_DIR / "evaluations"


def _utc(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("Evaluation time must be timezone aware")
    return value.tz_convert("UTC")


def evaluate(session, target_daily, *, now=None, write=False):
    """Validate target closing prices and optionally create an immutable score."""
    now = _utc(now if now is not None else datetime.now(timezone.utc))
    forecast_path = audit_existing_run(
        session, output_dir=FORECAST_DIR,
        source_dir=SOURCE_DIR, history_path=HISTORY,
    )
    if forecast_path is None:
        raise FileNotFoundError(f"No audited TimesFM forecast: {session}")

    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    manifest = manifest_path(FORECAST_DIR, session)
    target = forecast["forecast_target_date"]

    target_daily = Path(target_daily).resolve()
    if target_daily.parent != SOURCE_DIR.resolve():
        raise ValueError("Outcome snapshot is outside Schwab data directory")

    match = SOURCE_NAME.fullmatch(target_daily.name)
    if match is None:
        raise ValueError("Invalid target snapshot filename")

    captured = pd.to_datetime(
        match.group(1), format="%Y%m%dT%H%M%SZ",
        utc=True, errors="raise",
    )
    if captured > now:
        raise ValueError("Outcome snapshot was captured in the future")

    # Critically, do not treat an in-progress target day as an outcome.
    # The existing validator accepts a simulated historical capture time
    # only for *scoring*, never for issuing a forecast.
    validation = validate_snapshot_pair(
        target_daily, now=captured, expected_session=target,
    )
    if validation["market_date"] != target:
        raise ValueError("Outcome snapshot does not end at forecast target")
    if now < captured:
        raise ValueError("Target outcome has not become observable")

    frame = pd.read_csv(target_daily)
    required = {"market_date", "close"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Outcome snapshot missing required columns")

    dates = pd.to_datetime(frame["market_date"], errors="raise").dt.date
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Outcome prices have duplicate or unordered dates")
    closes = pd.to_numeric(frame["close"], errors="raise")
    if not np.isfinite(closes.to_numpy()).all() or (closes <= 0).any():
        raise ValueError("Outcome contains invalid closing prices")

    prices = dict(zip(dates.astype(str), closes.astype(float)))
    expected = forecast["forecast_dates"]
    if any(day not in prices for day in [session, *expected]):
        raise ValueError("Outcome snapshot missing a required trading session")
    if str(dates.iloc[-1]) != target:
        raise ValueError("Outcome snapshot includes a later session")

    origin_close = float(forecast["tmf_close"])
    if abs(prices[session] - origin_close) > 0.001:
        raise ValueError("Outcome snapshot origin close disagrees with forecast")

    actual = np.array([prices[day] for day in expected], dtype=np.float64)
    predicted = np.array(forecast["median_forecast"], dtype=np.float64)
    lower = np.array(forecast["q10_forecast"], dtype=np.float64)
    upper = np.array(forecast["q90_forecast"], dtype=np.float64)

    if any(len(v) != len(expected) for v in (actual, predicted, lower, upper)):
        raise ValueError("Unexpected forecast vector length")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Nonfinite evaluation prices")

    errors = predicted - actual
    baseline_errors = np.full(len(actual), origin_close) - actual
    covered = (lower <= actual) & (actual <= upper)
    predicted_up = bool(predicted[-1] > origin_close)
    actual_up = bool(actual[-1] > origin_close)

    result = {
        "status": "EVALUATED",
        "model": forecast["model"],
        "forecast_file": forecast_path.name,
        "forecast_sha256": sha256(forecast_path),
        "manifest_file": manifest.name,
        "manifest_sha256": sha256(manifest),
        "checkpoint_revision": forecast["checkpoint_revision"],
        "forecast_origin": session,
        "forecast_target_date": target,
        "issued_at_utc": forecast["issued_at_utc"],
        "scored_at_utc": now.isoformat(),
        "horizon_trading_days": len(expected),
        "forecast_dates": list(expected),
        "origin_close": origin_close,
        "actual_closes": actual.tolist(),
        "median_forecast": predicted.tolist(),
        "q10_forecast": lower.tolist(),
        "q90_forecast": upper.tolist(),
        "target_close": float(actual[-1]),
        "realized_return_pct": (actual[-1] / origin_close - 1) * 100,
        "median_implied_return_pct": (predicted[-1] / origin_close - 1) * 100,
        "actual_positive_return": actual_up,
        "median_predicted_positive_return": predicted_up,
        "direction_correct": predicted_up == actual_up,
        "median_mae": float(np.mean(np.abs(errors))),
        "median_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "persistence_mae": float(np.mean(np.abs(baseline_errors))),
        "persistence_rmse": float(np.sqrt(np.mean(baseline_errors ** 2))),
        "target_median_absolute_error": float(abs(errors[-1])),
        "target_persistence_absolute_error": float(abs(baseline_errors[-1])),
        "q10_q90_coverage": float(np.mean(covered)),
        "target_within_q10_q90": bool(covered[-1]),
        "mean_q10_q90_width": float(np.mean(upper - lower)),
        "outcome_snapshot": target_daily.name,
        "outcome_snapshot_sha256": sha256(target_daily),
        "outcome_intraday_sha256": sha256(target_daily.with_name(
            target_daily.name.replace("TMF_1d_", "TMF_5m_", 1)
        )),
        "order_generated": False,
    }

    if write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        destination = OUTPUT_DIR / (
            f"TMF_TIMESFM25_{session}_5d_evaluation.json"
        )
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        return destination, result

    return None, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--daily", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()

    destination, result = evaluate(
        args.session, args.daily, write=args.run,
    )
    print(json.dumps({
        "status": "EVALUATED" if destination else "CHECK_ONLY",
        "evaluation_path": str(destination) if destination else None,
        "direction_correct": result["direction_correct"],
        "median_mae": result["median_mae"],
        "persistence_mae": result["persistence_mae"],
        "q10_q90_coverage": result["q10_q90_coverage"],
        "order_generated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
