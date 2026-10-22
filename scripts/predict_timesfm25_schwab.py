"""Prospective, research-only TimesFM 2.5 five-session TMF price forecasts.

Does not run or modify V3_FINAL; never generates brokerage orders.
Usage:
    python -m scripts.predict_timesfm25_schwab --daily PATH --check
    python -m scripts.predict_timesfm25_schwab --daily PATH --run
Set TIMESFM25_CHECKPOINT_SHA to the audited, immutable HF commit SHA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from scripts.market_session import require_issuance_window
from scripts.validate_tmf_snapshot_pair import validate_snapshot_pair
from src.models.timesfm25_forecaster import (
    HORIZON, MODEL_ID, MODEL_NAME, closing_context,
    forecast_price_path, load_timesfm25_model, validate_revision,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data/raw/schwab"
OUTPUT_DIR = ROOT / "data/forward_test/timesfm25"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trading_dates(origin):
    dates = mcal.get_calendar("NYSE").valid_days(
        start_date=pd.Timestamp(origin) + pd.Timedelta(days=1),
        end_date=pd.Timestamp(origin) + pd.Timedelta(days=21),
    )
    if len(dates) < HORIZON:
        raise ValueError("Unable to find five subsequent NYSE sessions")
    return [str(day.date()) for day in dates[:HORIZON]]


def run(snapshot, *, execute=False, checkpoint_sha=None):
    snapshot = Path(snapshot).resolve()
    if snapshot.parent != SOURCE_DIR.resolve():
        raise ValueError("Snapshot must be within the Schwab snapshot directory")

    # A valid snapshot and market issuance window are mandatory even in
    # check-only mode. No retrospective prediction issuance is permitted.
    validation = validate_snapshot_pair(snapshot)
    origin = validation["market_date"]
    require_issuance_window(origin)

    if not execute:
        return {"status": "CHECK_ONLY", "session": origin,
                "source_snapshot": snapshot.name,
                "order_generated": False}

    revision = validate_revision(
        checkpoint_sha or os.getenv("TIMESFM25_CHECKPOINT_SHA")
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / (
        f"TMF_TIMESFM25_{origin}_{HORIZON}d.json"
    )
    if destination.exists():
        raise FileExistsError(f"TimesFM forecast already exists: {destination}")

    # Reuse V3's established dividend-aware Yahoo/Schwab reconciliation
    # without invoking the V3 predictor or altering its frozen model.
    from scripts.predict_v3_schwab import load_prices
    os.environ["TMF_PINNED_SNAPSHOT"] = str(snapshot)
    prices, selected_snapshot, _ = load_prices()
    if selected_snapshot.resolve() != snapshot:
        raise ValueError("Price loader selected a different snapshot")
    context = closing_context(prices, origin)
    source_sha = file_hash(snapshot)
    intraday = snapshot.with_name(
        snapshot.name.replace("TMF_1d_", "TMF_5m_", 1)
    )
    intraday_sha = file_hash(intraday)
    model = load_timesfm25_model(revision)
    predictions = forecast_price_path(model, context)

    # Check the prospective window a final time immediately before writing.
    require_issuance_window(origin)
    if file_hash(snapshot) != source_sha or file_hash(intraday) != intraday_sha:
        raise ValueError("Schwab input modified during TimesFM inference")

    next_dates = trading_dates(origin)
    issued = datetime.now(timezone.utc).isoformat()
    result = {
        "model": MODEL_NAME,
        "model_id": MODEL_ID,
        "checkpoint_revision": revision,
        "status": "experimental",
        "issued_at_utc": issued,
        "forecast_origin": origin,
        "forecast_target_date": next_dates[-1],
        "forecast_dates": next_dates,
        "horizon_trading_days": HORIZON,
        "tmf_close": float(context[-1]),
        "point_forecast": predictions["point"],
        "median_forecast": predictions["median"],
        "q10_forecast": predictions["q10"],
        "q90_forecast": predictions["q90"],
        "source_snapshot": snapshot.name,
        "source_sha256": source_sha,
        "intraday_sha256": intraday_sha,
        "historical_source_sha256": file_hash(
            ROOT / "data/raw/TMF_1d.csv"
        ),
        "historical_context_size": len(context),
        "forecast_type": "univariate_zero_shot_price_level",
        "order_generated": False,
    }
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return {"status": "GENERATED", "path": str(destination),
            "session": origin, "order_generated": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.daily, execute=args.run), indent=2))


if __name__ == "__main__":
    main()
