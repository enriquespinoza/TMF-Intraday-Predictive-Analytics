"""Integrity and recovery controls for immutable TimesFM 2.5 forecasts.

A legacy audit creates a new attestation about an existing file; it does NOT
change the forecast or retroactively claim that its manifest existed at issue.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from scripts.market_session import FINALIZATION_DELAY
from src.models.timesfm25_forecaster import (
    HORIZON, MODEL_ID, MODEL_NAME, validate_revision,
)

SOURCE_NAME = re.compile(r"TMF_1d_(\d{8}T\d{6}Z)\.csv\Z")
MANIFEST_STATES = frozenset({"GENERATED", "AUDITED_LEGACY"})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forecast_path(output_dir, origin):
    return Path(output_dir) / f"TMF_TIMESFM25_{origin}_{HORIZON}d.json"


def manifest_path(output_dir, origin):
    return (Path(output_dir) / "runs" /
            f"TMF_TIMESFM25_{origin}_{HORIZON}d_manifest.json")


@contextmanager
def session_lock(output_dir, origin):
    """A crash leaves a lock requiring manual investigation; never steal it."""
    lock = Path(output_dir) / f".lock_TIMESFM25_{origin}_{HORIZON}d"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(
            "TimesFM forecast lock exists; inspect interrupted run"
        ) from exc
    try:
        yield
    finally:
        lock.rmdir()


def _utc(value, name):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone aware")
    return stamp.tz_convert("UTC")


def _expected_dates(origin):
    sessions = mcal.get_calendar("NYSE").valid_days(
        start_date=pd.Timestamp(origin) + pd.Timedelta(days=1),
        end_date=pd.Timestamp(origin) + pd.Timedelta(days=21),
    )
    if len(sessions) < HORIZON:
        raise ValueError("Unable to resolve forecast target")
    return [str(day.date()) for day in sessions[:HORIZON]]


def _validate_payload(path, source_dir, history_path):
    """Reconcile immutable forecast bytes to market data and origin window."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    origin = data.get("forecast_origin")
    if not isinstance(origin, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", origin):
        raise ValueError("Invalid TimesFM origin")
    if path.name != forecast_path(path.parent, origin).name:
        raise ValueError("TimesFM forecast filename/origin mismatch")
    if (
        data.get("model") != MODEL_NAME
        or data.get("model_id") != MODEL_ID
        or data.get("horizon_trading_days") != HORIZON
        or data.get("forecast_type") != "univariate_zero_shot_price_level"
        or data.get("order_generated") is not False
        or data.get("status") != "experimental"
    ):
        raise ValueError("TimesFM metadata mismatch")
    validate_revision(data.get("checkpoint_revision"))

    dates = _expected_dates(origin)
    if data.get("forecast_dates") != dates or data.get("forecast_target_date") != dates[-1]:
        raise ValueError("TimesFM target-date mismatch")

    for key in ("point_forecast", "median_forecast",
                "q10_forecast", "q90_forecast"):
        values = data.get(key)
        if not isinstance(values, list) or len(values) != HORIZON:
            raise ValueError(f"TimesFM invalid {key} shape")
        if any(not isinstance(v, (int, float)) or isinstance(v, bool)
               or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"TimesFM invalid {key} values")

    if any(not (lo <= mid <= hi) for lo, mid, hi in zip(
        data["q10_forecast"], data["median_forecast"], data["q90_forecast"]
    )):
        raise ValueError("TimesFM quantile ordering failure")

    issued = _utc(data.get("issued_at_utc"), "issued_at_utc")
    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(
        start_date=origin, end_date=pd.Timestamp(origin) + pd.Timedelta(days=14),
    )
    if schedule.empty or str(schedule.index[0].date()) != origin or len(schedule) < 2:
        raise ValueError("Origin is not an NYSE trading session")
    finalized = schedule.iloc[0]["market_close"] + FINALIZATION_DELAY
    next_open = schedule.iloc[1]["market_open"]
    if not (finalized <= issued < next_open):
        raise ValueError("Forecast issuance outside original prospective window")

    source_name = data.get("source_snapshot")
    match = SOURCE_NAME.fullmatch(source_name) if isinstance(source_name, str) else None
    if match is None:
        raise ValueError("Invalid TimesFM source snapshot filename")
    captured = pd.to_datetime(
        match.group(1), utc=True, format="%Y%m%dT%H%M%SZ", errors="raise"
    )
    if not (finalized <= captured <= issued):
        raise ValueError("Snapshot not captured within prospective window")

    daily = Path(source_dir) / source_name
    intraday = Path(source_dir) / source_name.replace("TMF_1d_", "TMF_5m_", 1)
    history = Path(history_path)
    for file, expected in (
        (daily, data.get("source_sha256")),
        (intraday, data.get("intraday_sha256")),
        (history, data.get("historical_source_sha256")),
    ):
        if not file.is_file() or not isinstance(expected, str) or sha256(file) != expected:
            raise ValueError(f"TimesFM source hash mismatch: {file.name}")

    prices = pd.read_csv(daily)
    if not {"market_date", "close"}.issubset(prices.columns) or prices.empty:
        raise ValueError("Invalid daily source schema")
    days = pd.to_datetime(prices["market_date"], errors="raise").dt.date
    if days.duplicated().any() or not days.is_monotonic_increasing or str(days.iloc[-1]) != origin:
        raise ValueError("Snapshot closes beyond or before forecast origin")
    close = data.get("tmf_close")
    if not isinstance(close, (float, int)) or not math.isfinite(close) or close <= 0:
        raise ValueError("Invalid forecast-origin close")
    if abs(float(prices.iloc[-1]["close"]) - close) > 0.001:
        raise ValueError("Forecast-origin closing price mismatch")

    return data


def create_manifest(path, *, source_dir, history_path, state):
    """Create a separate exclusive-write integrity manifest after verification."""
    if state not in MANIFEST_STATES:
        raise ValueError("Unrecognized TimesFM manifest state")
    path = Path(path)
    data = _validate_payload(path, source_dir, history_path)
    manifest = manifest_path(path.parent, data["forecast_origin"])
    manifest.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "status": state,
        "model": MODEL_NAME,
        "origin": data["forecast_origin"],
        "target": data["forecast_target_date"],
        "forecast_file": path.name,
        "forecast_sha256": sha256(path),
        "checkpoint_revision": data["checkpoint_revision"],
        "source_snapshot": data["source_snapshot"],
        "source_sha256": data["source_sha256"],
        "intraday_sha256": data["intraday_sha256"],
        "historical_source_sha256": data["historical_source_sha256"],
        "issued_at_utc": data["issued_at_utc"],
        "manifest_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_limitation": (
            "Verified recorded timestamps and available source hashes; "
            "no independent timestamp attestation"
        ) if state == "AUDITED_LEGACY" else None,
        "order_generated": False,
    }
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return manifest


def audit_existing_run(origin, *, output_dir, source_dir, history_path):
    """Fail closed on orphan, incomplete, altered, or inconsistent artifacts."""
    forecast = forecast_path(output_dir, origin)
    manifest = manifest_path(output_dir, origin)
    if not forecast.is_file():
        if manifest.exists():
            raise RuntimeError("ORPHANED_MANIFEST: forecast file missing")
        return None
    if not manifest.is_file():
        raise RuntimeError("INCOMPLETE_RUN: TimesFM manifest is missing")

    data = _validate_payload(forecast, source_dir, history_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {
        "model": MODEL_NAME,
        "origin": origin,
        "target": data["forecast_target_date"],
        "forecast_file": forecast.name,
        "forecast_sha256": sha256(forecast),
        "checkpoint_revision": data["checkpoint_revision"],
        "source_snapshot": data["source_snapshot"],
        "source_sha256": data["source_sha256"],
        "intraday_sha256": data["intraday_sha256"],
        "historical_source_sha256": data["historical_source_sha256"],
        "issued_at_utc": data["issued_at_utc"],
        "order_generated": False,
    }
    if record.get("status") not in MANIFEST_STATES:
        raise RuntimeError("INVALID_RUN_MANIFEST: status mismatch")
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"INVALID_RUN_MANIFEST: {key} mismatch")
    return forecast
