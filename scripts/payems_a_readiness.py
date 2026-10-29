"""Fail-closed PAYEMS-A release readiness for frozen V3 inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from config.payems_candidates import PAYEMS_A_FEATURES

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "config/payems_a_refresh_schedule.json"
EVENTS = ROOT / "data/features/PAYEMS_event_features.csv"
SIDECAR = EVENTS.with_suffix(".refresh.json")
NY = "America/New_York"

def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require_payems_ready(origin, *, events_path=EVENTS,
                         schedule_path=SCHEDULE, sidecar_path=SIDECAR):
    """Refuse stale releases when a scheduled BLS date has already occurred.

    Does not accept a schedule alone as proof of publication; the matching
    active CSV must have been explicitly staged and activated.
    """
    day = str(pd.Timestamp(origin).date())
    schedule = json.loads(Path(schedule_path).read_text(encoding="utf-8"))
    events = pd.read_csv(events_path)
    if events.empty or "event_release_date" not in events:
        raise ValueError("PAYEMS event source missing")
    dates = pd.to_datetime(events["event_release_date"], errors="raise").dt.date
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("PAYEMS event dates invalid")
    available = events.loc[dates <= pd.Timestamp(day).date()]
    if available.empty:
        raise ValueError("No PAYEMS event available for forecast date")
    latest = available.iloc[-1]
    if not set(PAYEMS_A_FEATURES).issubset(events.columns):
        raise ValueError("PAYEMS-A frozen columns missing")
    for name in PAYEMS_A_FEATURES:
        val = float(latest[name])
        if not __import__("math").isfinite(val):
            raise ValueError("PAYEMS-A invalid input " + name)
    due = [item for item in schedule["events"] if item["release_date"] <= day]
    if len({item["release_date"] for item in due}) != len(due):
        raise ValueError("Duplicate PAYEMS release schedule entries")
    if due:
        item = max(due, key=lambda x: x["release_date"])
        release = item["release_date"]
        if item["release_timezone"] != NY or item["release_time"] != "08:30":
            raise ValueError("Unapproved PAYEMS release schedule")
        if str(latest["event_release_date"]) != release:
            raise ValueError(
                "PAYEMS_A_REFRESH_REQUIRED: scheduled release " + release +
                " has no activated event row; do not use stale macro inputs"
            )
        if not Path(sidecar_path).is_file():
            raise ValueError("PAYEMS_A_REFRESH_REQUIRED: missing activation attestation")
        attestation = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
        if (attestation["release_date"] != release
                or attestation["active_sha256"] != _hash(events_path)):
            raise ValueError("PAYEMS_A_REFRESH_REQUIRED: source integrity mismatch")
    return str(latest["event_release_date"])
