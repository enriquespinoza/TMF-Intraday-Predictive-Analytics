"""Stage and activate point-in-time PAYEMS-A release updates, without model changes.

--stage: fetch ALFRED PAYEMS revision history, reconstruct exactly ONE event
         using frozen labor_features, and write immutable candidate/attestation.
--activate: revalidate all staged hashes and baseline; atomically replace
            PAYEMS_event_features.csv and write an integrity sidecar.
No automatic activation. V3 guards scheduled release dates separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.payems_candidates import PAYEMS_A_FEATURES
from src.data.download_macro_vintages import (
    get_fred_api_key, request_vintage_observations,
    parse_vintage_observations, validate_vintage_data,
)
from src.features.labor_features import build_payroll_event_features

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data/features/PAYEMS_event_features.csv"
SCHEDULE = ROOT / "config/payems_a_refresh_schedule.json"
STAGE_DIR = ROOT / "data/raw/payems_a_refresh"
SIDECAR = EVENTS.with_suffix(".refresh.json")
NY = "America/New_York"
REQUIRED = tuple(PAYEMS_A_FEATURES)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def schedule_for(day, *, schedule=SCHEDULE):
    obj = json.loads(Path(schedule).read_text(encoding="utf-8"))
    matches = [e for e in obj["events"] if e["release_date"] == day]
    if len(matches) != 1:
        raise ValueError("Release must be explicitly present exactly once in approved schedule")
    event = matches[0]
    if event["release_timezone"] != NY or event["release_time"] != "08:30":
        raise ValueError("Unapproved release clock")
    return event

def release_utc(event):
    return pd.Timestamp(
        event["release_date"] + " " + event["release_time"],
        tz=NY,
    ).tz_convert("UTC")

def now_utc(now=None):
    stamp = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Current time must be timezone-aware")
    return stamp.tz_convert("UTC")

def validate_baseline(baseline, event):
    if baseline.empty or "event_release_date" not in baseline:
        raise ValueError("Missing baseline PAYEMS event history")
    dates = pd.to_datetime(baseline["event_release_date"], errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Baseline event dates duplicated or unsorted")
    last = str(dates.iloc[-1].date())
    if last >= event["release_date"]:
        raise ValueError("Baseline already includes release (or later)")
    if not set(REQUIRED).issubset(baseline.columns):
        raise ValueError("Frozen PAYEMS-A fields are missing")
    return last

def create_candidate(baseline, vintage, event):
    validate_baseline(baseline, event)
    vintage = vintage.copy()
    for column in ("reference_date", "vintage_date"):
        vintage[column] = pd.to_datetime(vintage[column], errors="raise")
    validate_vintage_data(vintage, "nonfarm_payrolls")
    if set(vintage["series_id"]) != {"PAYEMS"} or set(vintage["series_name"]) != {"nonfarm_payrolls"}:
        raise ValueError("Expected only PAYEMS vintages")
    if vintage["vintage_date"].max().date().isoformat() != event["release_date"]:
        raise ValueError("Newest PAYEMS vintage must be the approved release date")
    newest = vintage.loc[
        vintage["vintage_date"] == pd.Timestamp(event["release_date"])
    ]
    if newest.empty or not (
        newest["reference_date"] == pd.Timestamp(event["reference_month"])
    ).any():
        raise ValueError("PAYEMS release is missing expected initial reference period")
    if not (
        newest.loc[newest["reference_date"] == pd.Timestamp(event["reference_month"]),
                   "value_as_known"].notna().all()
    ):
        raise ValueError("Missing initial PAYEMS value")
    features = build_payroll_event_features(vintage, event["release_date"])
    if str(features["reference_date"].date()) != event["reference_month"]:
        raise ValueError("Latest PAYEMS period mismatches schedule")
    row = {name: None for name in baseline.columns}
    row.update({
        "event_name": "employment_situation",
        "event_reference_date": event["reference_month"],
        "event_release_date": event["release_date"],
        "event_release_time": event["release_time"],
        "event_release_timezone": NY,
    })
    for field in features:
        if field in row:
            value = features[field]
            row[field] = value.isoformat() if isinstance(value, pd.Timestamp) else value
    # Only three frozen operational features are allowed to change V3 inference.
    for feature in REQUIRED:
        value = pd.to_numeric(pd.Series([row[feature]]), errors="raise").iloc[0]
        if not pd.notna(value) or not __import__('math').isfinite(float(value)):
            raise ValueError(f"Missing/nonfinite PAYEMS-A feature: {feature}")
    # Standardized research-only columns are not model inputs; leave empty
    # rather than substitute guessed z-scores.
    candidate = pd.concat([baseline, pd.DataFrame([row], columns=baseline.columns)],
                          ignore_index=True)
    if len(candidate) != len(baseline) + 1:
        raise ValueError("Candidate must append exactly one row")
    return candidate

def stage(day, *, now=None, baseline_path=EVENTS, schedule_path=SCHEDULE,
          stage_dir=STAGE_DIR, vintage_data=None):
    event = schedule_for(day, schedule=schedule_path)
    clock = now_utc(now)
    if clock < release_utc(event):
        raise ValueError("Cannot stage PAYEMS before official scheduled release")
    baseline_path = Path(baseline_path)
    baseline = pd.read_csv(baseline_path)
    last = validate_baseline(baseline, event)
    if vintage_data is None:
        api_key = get_fred_api_key()
        payload = request_vintage_observations(
            series_id="PAYEMS", api_key=api_key,
            realtime_start="2021-09-01",
            realtime_end=day,
            observation_start="2021-09-01",
        )
        vintage_data = parse_vintage_observations(
            payload, series_name="nonfarm_payrolls", series_id="PAYEMS"
        )
    candidate = create_candidate(baseline, vintage_data, event)
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    name = f"PAYEMS_A_{day}"
    vintage_path = stage_dir / f"{name}_vintages.csv"
    candidate_path = stage_dir / f"{name}_candidate.csv"
    manifest_path = stage_dir / f"{name}_manifest.json"
    if any(x.exists() for x in (vintage_path, candidate_path, manifest_path)):
        raise FileExistsError("PAYEMS release already staged; immutable records cannot be overwritten")
    vintage_data.to_csv(vintage_path, index=False, mode="x")
    candidate.to_csv(candidate_path, index=False, mode="x")
    manifest = {
        "schema_version": "payems_a_refresh_v1",
        "status": "STAGED_NOT_ACTIVATED",
        "release_date": day,
        "reference_month": event["reference_month"],
        "scheduled_release_utc": release_utc(event).isoformat(),
        "fetched_at_utc": clock.isoformat(),
        "last_previous_release": last,
        "baseline_sha256": digest(baseline_path),
        "vintage_sha256": digest(vintage_path),
        "candidate_sha256": digest(candidate_path),
        "schedule_sha256": digest(schedule_path),
        "source": "FRED_ALFRED_PAYEMS_output_type_3",
        "source_url": "https://api.stlouisfed.org/fred/series/observations",
        "active_model_changes": False,
        "frozen_features": list(REQUIRED),
        "order_generated": False,
        "review_required": True,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest

def activate(day, *, baseline_path=EVENTS, schedule_path=SCHEDULE,
             stage_dir=STAGE_DIR, sidecar_path=SIDECAR):
    """Activate a reviewed candidate; disallow another active release without review."""
    event = schedule_for(day, schedule=schedule_path)
    stage_dir = Path(stage_dir)
    prefix = stage_dir / f"PAYEMS_A_{day}"
    manifest_path = Path(str(prefix) + "_manifest.json")
    vintage_path = Path(str(prefix) + "_vintages.csv")
    candidate_path = Path(str(prefix) + "_candidate.csv")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline_path = Path(baseline_path)
    if manifest["status"] != "STAGED_NOT_ACTIVATED" or manifest["release_date"] != day:
        raise ValueError("Invalid stage status/release")
    for path, field in ((vintage_path, "vintage_sha256"),
                        (candidate_path, "candidate_sha256"),
                        (baseline_path, "baseline_sha256"),
                        (schedule_path, "schedule_sha256")):
        if digest(path) != manifest[field]:
            raise ValueError(f"Staged or active source hash mismatch: {path}")
    baseline = pd.read_csv(baseline_path)
    expected = create_candidate(baseline, pd.read_csv(vintage_path), event)
    candidate = pd.read_csv(candidate_path)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True), candidate.reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=0, atol=1e-10,
    )
    if Path(sidecar_path).exists():
        prior = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
        if digest(baseline_path) != prior["active_sha256"]:
            raise ValueError("Previous PAYEMS active sidecar is invalid")
    # Crash-safe activation: sidecar prepared with the exact destination bytes.
    attestation = {
        "schema_version": "payems_a_activation_v1",
        "release_date": day,
        "active_sha256": digest(candidate_path),
        "stage_manifest_sha256": digest(manifest_path),
        "candidate_sha256": manifest["candidate_sha256"],
        "activated_at_utc": now_utc().isoformat(),
        "research_only": True,
    }
    dest = baseline_path.resolve()
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as temp:
        tmp = Path(temp.name)
        temp.write(candidate_path.read_bytes())
    try:
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    sidecar = Path(sidecar_path)
    sidecar.write_text(json.dumps(attestation, indent=2) + "\n", encoding="utf-8")
    return attestation

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stage", action="store_true")
    mode.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    if args.stage:
        output = stage(args.release)
    else:
        output = activate(args.release)
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
