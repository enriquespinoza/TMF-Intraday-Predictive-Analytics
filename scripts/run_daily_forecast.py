"""Fail-closed daily TMF V3 forecasting orchestrator."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from config.v3_final import FINAL_V3_NAME
from src.models.directional_classifier import HORIZON
from scripts.validate_tmf_session import FINALIZATION_DELAY
from scripts.validate_tmf_snapshot_pair import validate_snapshot_pair


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/raw/schwab"
OUTPUT_DIR = ROOT / "data/forward_test"
RUN_DIR = OUTPUT_DIR / "runs"
NY = "America/New_York"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session_ready():
    now = pd.Timestamp.now(tz="UTC")
    session = now.tz_convert(NY).date()

    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(
        start_date=session,
        end_date=session,
    )

    if schedule.empty:
        raise ValueError("Today is not an NYSE session")

    market_close = schedule.iloc[0]["market_close"]

    if now < market_close + FINALIZATION_DELAY:
        raise ValueError("Post-close waiting period not complete")

    return session, now


def matching_forecasts(session):
    matches = []

    for path in sorted(
        OUTPUT_DIR.glob(f"TMF_V3_SCHWAB_{session}_*.json")
    ):
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("model") == FINAL_V3_NAME
            and data.get("forecast_origin") == str(session)
            and data.get("horizon_trading_days") == HORIZON
        ):
            matches.append(path)

    if len(matches) > 1:
        raise ValueError("Duplicate forecast records already exist")

    return matches



def audit_existing_run(session):
    """Verify a previously completed forecast and its run manifest."""
    forecasts = matching_forecasts(session)
    manifest = RUN_DIR / (
        f"TMF_V3_{session}_{FINAL_V3_NAME}_{HORIZON}d.json"
    )

    if not forecasts:
        if manifest.exists():
            raise RuntimeError("ORPHANED_MANIFEST: forecast missing")
        return None

    forecast_path = forecasts[0]

    if not manifest.is_file():
        raise RuntimeError(
            "INCOMPLETE_RUN: forecast exists without run manifest"
        )

    record = json.loads(manifest.read_text(encoding="utf-8"))
    forecast = json.loads(
        forecast_path.read_text(encoding="utf-8")
    )

    if (
        record.get("status") != "GENERATED"
        or record.get("session") != str(session)
        or record.get("model") != FINAL_V3_NAME
        or record.get("horizon_trading_days") != HORIZON
        or record.get("forecast") != forecast_path.name
        or record.get("order_generated") is not False
        or forecast.get("order_generated") is not False
        or forecast.get("forecast_target_date")
        != forecast_target(session)
        or forecast.get("source_snapshot")
        != record.get("daily_snapshot")
        or forecast.get("source_sha256")
        != record.get("daily_sha256")
    ):
        raise RuntimeError("INVALID_RUN_MANIFEST: metadata mismatch")

    if digest(forecast_path) != record.get("forecast_sha256"):
        raise RuntimeError("INVALID_RUN_MANIFEST: forecast hash mismatch")

    snapshot_name = record.get("daily_snapshot")

    if (
        not isinstance(snapshot_name, str)
        or Path(snapshot_name).name != snapshot_name
        or not snapshot_name.startswith("TMF_1d_")
        or not snapshot_name.endswith(".csv")
    ):
        raise RuntimeError("INVALID_RUN_MANIFEST: snapshot path")

    daily = DATA_DIR / snapshot_name
    intraday = DATA_DIR / snapshot_name.replace(
        "TMF_1d_", "TMF_5m_", 1
    )

    for path, expected_hash in (
        (daily, record.get("daily_sha256")),
        (intraday, record.get("intraday_sha256")),
    ):
        if not path.is_file() or digest(path) != expected_hash:
            raise RuntimeError(
                f"INVALID_RUN_MANIFEST: source mismatch: {path.name}"
            )

    return forecast_path


def latest_snapshot():
    files = sorted(DATA_DIR.glob("TMF_1d_*.csv"))

    if not files:
        raise FileNotFoundError("No Schwab daily snapshots")

    return files[-1]


def refresh_snapshot():
    before = set(DATA_DIR.glob("TMF_1d_*.csv"))

    subprocess.run(
        [sys.executable, "-m", "scripts.download_schwab_tmf"],
        cwd=ROOT,
        check=True,
    )

    after = set(DATA_DIR.glob("TMF_1d_*.csv"))
    created = sorted(after - before)

    if len(created) != 1:
        raise ValueError(
            "Schwab refresh did not create exactly one new daily snapshot"
        )

    return created[0]


def forecast_target(session):
    days = mcal.get_calendar("NYSE").valid_days(
        start_date=pd.Timestamp(session) + pd.Timedelta(days=1),
        end_date=pd.Timestamp(session) + pd.Timedelta(days=21),
    )

    if len(days) < HORIZON:
        raise ValueError("Unable to resolve forecast target")

    return str(days[HORIZON - 1].date())


def execute_forecast(session, snapshot):
    expected_path = OUTPUT_DIR / (
        f"TMF_V3_SCHWAB_{session}_{FINAL_V3_NAME}_{HORIZON}d.json"
    )

    if expected_path.exists():
        raise FileExistsError(expected_path)

    intraday_path = snapshot.with_name(
        snapshot.name.replace("TMF_1d_", "TMF_5m_", 1)
    )

    daily_hash = digest(snapshot)
    intraday_hash = digest(intraday_path)
    predictor_hash = digest(
        ROOT / "scripts/predict_v3_schwab.py"
    )

    start = datetime.now(timezone.utc)

    env = os.environ.copy()
    env["TMF_PINNED_SNAPSHOT"] = str(snapshot.resolve())
    env["TMF_EXPECTED_SESSION"] = str(session)

    subprocess.run(
        [sys.executable, "-m", "scripts.predict_v3_schwab"],
        cwd=ROOT,
        env=env,
        check=True,
    )

    if not expected_path.is_file():
        raise RuntimeError("Predictor did not create expected forecast")

    result = json.loads(
        expected_path.read_text(encoding="utf-8")
    )

    issued = datetime.fromisoformat(result["issued_at_utc"])

    if issued.tzinfo is None or not (
        start <= issued.astimezone(timezone.utc)
        <= datetime.now(timezone.utc)
    ):
        raise ValueError("Invalid forecast issuance timestamp")

    if (
        result.get("model") != FINAL_V3_NAME
        or result.get("forecast_origin") != str(session)
        or result.get("forecast_target_date")
        != forecast_target(session)
        or result.get("horizon_trading_days") != HORIZON
        or result.get("source_snapshot") != snapshot.name
        or result.get("source_sha256") != daily_hash
        or result.get("order_generated") is not False
    ):
        raise ValueError("Forecast metadata validation failed")

    if (
        digest(snapshot) != daily_hash
        or digest(intraday_path) != intraday_hash
    ):
        raise ValueError("Schwab input changed during prediction")

    matching_forecasts(session)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    record_path = RUN_DIR / (
        f"TMF_V3_{session}_{FINAL_V3_NAME}_{HORIZON}d.json"
    )

    record = {
        "status": "GENERATED",
        "session": str(session),
        "model": FINAL_V3_NAME,
        "horizon_trading_days": HORIZON,
        "forecast": expected_path.name,
        "forecast_sha256": digest(expected_path),
        "daily_snapshot": snapshot.name,
        "daily_sha256": daily_hash,
        "intraday_sha256": intraday_hash,
        "predictor_sha256": predictor_hash,
        "recorded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "order_generated": False,
    }

    with record_path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")

    print("Forecast recorded:", expected_path)
    print("Run manifest:", record_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--daily", type=Path)
    args = parser.parse_args()

    if args.refresh and not args.run:
        parser.error("--refresh requires --run")

    if args.refresh and args.daily:
        parser.error("Choose --refresh or --daily, not both")

    session, _ = session_ready()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.run:
        snapshot = args.daily or latest_snapshot()
        result = validate_snapshot_pair(snapshot)
        print(json.dumps(result, indent=2))
        print("CHECK ONLY — no forecast generated")
        return

    # Atomic directory creation prevents concurrent runs.
    lock_path = OUTPUT_DIR / (
        f".lock_{FINAL_V3_NAME}_{session}_{HORIZON}d"
    )

    try:
        lock_path.mkdir()
    except FileExistsError:
        raise RuntimeError(
            "Forecast lock exists. Check for an active or "
            "interrupted run before investigating the lock."
        )

    try:
        existing = audit_existing_run(session)

        if existing:
            print("SKIPPED — forecast already exists:", existing[0])
            return

        snapshot = (
            refresh_snapshot()
            if args.refresh
            else args.daily or latest_snapshot()
        )

        validation = validate_snapshot_pair(snapshot)

        if validation["market_date"] != str(session):
            raise ValueError("Validated session mismatch")

        print("Snapshot validation: PASSED")
        print("Origin:", session)
        print("Expected target:", forecast_target(session))

        execute_forecast(session, snapshot)

    finally:
        lock_path.rmdir()


if __name__ == "__main__":
    main()
