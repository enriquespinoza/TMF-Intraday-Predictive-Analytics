"""Research-only scheduled TMF V3_FINAL / TimesFM 2.5 paired experiment cycle.

Designed for Windows Task Scheduler now and a future top-level orchestrator.
Run from the analytics venv after close+30 minutes, before next NYSE open.
Only the established, independently audited model and scoring CLIs may write
forecasts/evaluations. There are NO trading/ordering API calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from scripts.market_session import resolve_market_session
from scripts.payems_a_readiness import require_payems_ready
from scripts.run_daily_forecast import audit_existing_run as audit_v3
from scripts.timesfm25_integrity import audit_existing_run as audit_timesfm
from scripts.validate_tmf_snapshot_pair import validate_snapshot_pair

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/schwab"
V3_DIR = ROOT / "data/forward_test"
TIMESFM_DIR = V3_DIR / "timesfm25"
EVENT_DIR = ROOT / "reports/automation/events"
HISTORY = ROOT / "data/raw/TMF_1d.csv"
CHECKPOINT_SHA = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
DEFAULT_TIMESFM_PYTHON = Path("C:/Dev/.venvs/tmf-timesfm25/Scripts/python.exe")
TIMEOUT = {"preflight": 90, "download": 240, "v3": 600,
           "timesfm": 2400, "score": 600}

class ResearchCycleError(RuntimeError):
    """A stage failed without exposing credential-bearing subprocess output."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_child(stage, python, module, *options, env=None, timeout=None):
    """Never log a Schwab OAuth authorization URL, token or secret."""
    cmd = [str(python), "-m", module, *[str(option) for option in options]]
    try:
        result = subprocess.run(
            cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            errors="replace", timeout=timeout or TIMEOUT["score"], check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ResearchCycleError(f"{stage}: subprocess timeout") from exc
    if result.returncode != 0:
        raise ResearchCycleError(
            f"{stage}: child exited {result.returncode}; "
            "investigate manually without exposing OAuth logs"
        )
    return result.stdout


def require_interpreter(path):
    path = Path(path)
    if not path.is_file():
        raise ResearchCycleError(f"TimesFM interpreter not found: {path}")
    return path


def original_source(forecast):
    payload = json.loads(Path(forecast).read_text(encoding="utf-8"))
    filename = payload.get("source_snapshot")
    if (not isinstance(filename, str) or Path(filename).name != filename
            or not filename.startswith("TMF_1d_") or not filename.endswith(".csv")):
        raise ResearchCycleError("Invalid forecast source name")
    source = SOURCE / filename
    if not source.is_file() or digest(source) != payload.get("source_sha256"):
        raise ResearchCycleError("Existing forecast source integrity failed")
    return source


def existing_paired_snapshot(session):
    """Retain the original market snapshot if an issuance was interrupted."""
    v3 = audit_v3(session)
    tf = audit_timesfm(
        str(session), output_dir=TIMESFM_DIR, source_dir=SOURCE,
        history_path=HISTORY,
    )
    paths = [original_source(path) for path in (v3, tf) if path is not None]
    if len(paths) == 2 and paths[0].resolve() != paths[1].resolve():
        raise ResearchCycleError("Previously issued model sources differ")
    return (paths[0] if paths else None), v3 is not None, tf is not None


def download_one_snapshot(session, analytics_python):
    before = set(SOURCE.glob("TMF_1d_*.csv"))
    run_child("Schwab download", analytics_python, "scripts.download_schwab_tmf",
              timeout=TIMEOUT["download"])
    after = set(SOURCE.glob("TMF_1d_*.csv"))
    created = sorted(after - before)
    if len(created) != 1:
        raise ResearchCycleError(
            "Schwab refresh must create exactly one new daily snapshot"
        )
    snapshot = created[0]
    validation = validate_snapshot_pair(snapshot, expected_session=session)
    if validation["market_date"] != session or validation["order_generated"] is not False:
        raise ResearchCycleError("Schwab snapshot not approved")
    return snapshot


def matching_maturities(session):
    """Return only frozen forecasts with the current exact target session."""
    v3 = []
    tf = []
    for directory, glob, collection in (
        (V3_DIR, "TMF_V3_SCHWAB_*_V3_FINAL_5d.json", v3),
        (TIMESFM_DIR, "TMF_TIMESFM25_*_5d.json", tf),
    ):
        for path in sorted(directory.glob(glob)):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("forecast_target_date") == session:
                collection.append((payload["forecast_origin"], path))
    return v3, tf


def existing_score_valid(path, forecast):
    """Verify write-once evaluation hashes without rewriting its source."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("forecast_sha256") != digest(forecast)
            or record.get("order_generated") is not False):
        raise ResearchCycleError("Existing evaluation manifest mismatch")
    name = record.get("outcome_snapshot", record.get("price_source"))
    expected = record.get(
        "outcome_snapshot_sha256", record.get("price_source_sha256")
    )
    if (not isinstance(name, str) or Path(name).name != name
            or not expected or digest(SOURCE / name) != expected):
        raise ResearchCycleError("Existing evaluation source integrity failed")


def score_maturities(session, snapshot, analytics_python, timesfm_python, events):
    """Each scorer runs only for its actual five-NYSE-session target."""
    v3, tf = matching_maturities(session)
    for origin, forecast in v3:
        dest = V3_DIR / "evaluations" / f"{forecast.stem}_evaluation.json"
        audit_v3(origin)
        if dest.exists():
            existing_score_valid(dest, forecast)
            events.append(f"V3_SCORE_ALREADY_VERIFIED:{origin}")
        elif datetime.now(timezone.utc).date().isoformat() <= session:
            # Frozen V3 scorer requires the calendar to advance beyond the
            # target date in UTC. A second nightly scheduler pass handles it.
            events.append(f"V3_SCORE_DEFERRED_UTC_CUTOFF:{origin}")
        else:
            run_child("V3 score", analytics_python,
                      "scripts.score_forward_forecast",
                      "--forecast", forecast, "--prices", snapshot,
                      timeout=TIMEOUT["score"])
            existing_score_valid(dest, forecast)
            events.append(f"V3_SCORED:{origin}")
    for origin, forecast in tf:
        dest = TIMESFM_DIR / "evaluations" / (
            f"TMF_TIMESFM25_{origin}_5d_evaluation.json"
        )
        audit_timesfm(origin, output_dir=TIMESFM_DIR, source_dir=SOURCE,
                      history_path=HISTORY)
        if dest.exists():
            existing_score_valid(dest, forecast)
            events.append(f"TIMESFM_SCORE_ALREADY_VERIFIED:{origin}")
        else:
            run_child("TimesFM score", timesfm_python,
                      "scripts.score_timesfm25_forecast",
                      "--session", origin, "--daily", snapshot, "--check",
                      timeout=TIMEOUT["score"])
            run_child("TimesFM score", timesfm_python,
                      "scripts.score_timesfm25_forecast",
                      "--session", origin, "--daily", snapshot, "--run",
                      timeout=TIMEOUT["score"])
            existing_score_valid(dest, forecast)
            events.append(f"TIMESFM_SCORED:{origin}")


def save_event(session, status, actions, *, stage=None, snapshot=None):
    """An append-only operational receipt, separate from research outcomes."""
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "tmf_daily_research_automation_v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "session": str(session) if session is not None else None,
        "status": status,
        "stage": stage,
        "actions": list(actions),
        "source_snapshot": snapshot.name if snapshot else None,
        "source_sha256": digest(snapshot) if snapshot and snapshot.is_file() else None,
        "order_generated": False,
    }
    label = str(session) if session else "no_session"
    file = EVENT_DIR / f"TMF_RESEARCH_{label}_{uuid4().hex}.json"
    with file.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps(record, indent=2))
    return file


def run_cycle(analytics_python, timesfm_python, *, dry_run=False):
    actions = []
    stage = "market_window"
    snapshot = None
    window = resolve_market_session()
    if window.status != "READY":
        # Closed NYSE holiday, prefinalization, or missed prospective window.
        if not dry_run:
            save_event(window.session, "SKIPPED_NO_PROSPECTIVE_WINDOW",
                       actions, stage=stage)
        return "SKIPPED_NO_PROSPECTIVE_WINDOW"

    session = str(window.session)
    if dry_run:
        print(json.dumps({
            "status": "DRY_RUN_ONLY", "session": session,
            "finalized_at_utc": str(window.finalized_at_utc),
            "next_open_utc": str(window.next_open_utc),
            "model_specifications_changed": False,
            "order_generated": False,
        }, indent=2))
        return "DRY_RUN_ONLY"

    try:
        stage = "preflight"
        timesfm_python = require_interpreter(timesfm_python)
        run_child("TimesFM environment", timesfm_python,
                  "scripts.audit_timesfm25_run", "--help",
                  timeout=TIMEOUT["preflight"])
        # Prevent a half-pair when a scheduled payroll release remains stale.
        require_payems_ready(session)
        actions.append("PAYEMS_READY")

        stage = "existing_issuance_audit"
        snapshot, has_v3, has_tf = existing_paired_snapshot(session)
        if snapshot is None:
            stage = "schwab_download"
            snapshot = download_one_snapshot(session, analytics_python)
            actions.append("NEW_SCHWAB_SNAPSHOT")
        else:
            validation = validate_snapshot_pair(snapshot, expected_session=session)
            if validation["order_generated"] is not False:
                raise ResearchCycleError("Snapshot validation output unsafe")
            actions.append("REUSED_AUDITED_SNAPSHOT")

        stage = "v3_issuance"
        if not has_v3:
            run_child("V3 issuance", analytics_python,
                      "scripts.run_daily_forecast",
                      "--run", "--daily", snapshot,
                      timeout=TIMEOUT["v3"])
            actions.append("V3_ISSUED")
        else:
            actions.append("V3_ALREADY_AUDITED")

        stage = "timesfm_issuance"
        if not has_tf:
            env = os.environ.copy()
            if env.get("TIMESFM25_CHECKPOINT_SHA") not in (None, CHECKPOINT_SHA):
                raise ResearchCycleError("Unexpected TimesFM checkpoint override")
            env["TIMESFM25_CHECKPOINT_SHA"] = CHECKPOINT_SHA
            run_child("TimesFM issuance", timesfm_python,
                      "scripts.predict_timesfm25_schwab",
                      "--daily", snapshot, "--run",
                      env=env, timeout=TIMEOUT["timesfm"])
            actions.append("TIMESFM_ISSUED")
        else:
            actions.append("TIMESFM_ALREADY_AUDITED")

        stage = "paired_source_audit"
        verified_snapshot, has_v3, has_tf = existing_paired_snapshot(session)
        if not (has_v3 and has_tf and verified_snapshot == snapshot):
            raise ResearchCycleError("Strict matched-snapshot pair not established")
        actions.append("STRICT_PAIRED_SNAPSHOT_VERIFIED")

        stage = "matured_scoring"
        score_maturities(session, snapshot, analytics_python,
                         timesfm_python, actions)

        stage = "complete"
        save_event(session, "COMPLETED", actions, stage=stage, snapshot=snapshot)
        return "COMPLETED"
    except Exception as exc:
        # Never include subprocess stderr, stdout, OAuth redirects, or tokens.
        actions.append(f"FAILED:{stage}:{type(exc).__name__}")
        save_event(session, "FAILED_REQUIRES_REVIEW", actions,
                   stage=stage, snapshot=snapshot)
        raise ResearchCycleError(
            f"Cycle failed in {stage} ({type(exc).__name__}); "
            "see local step status and investigate source/credentials"
        ) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesfm-python", type=Path,
                        default=DEFAULT_TIMESFM_PYTHON)
    parser.add_argument("--dry-run", action="store_true",
                        help="Read market calendar only; no downloads or writes")
    args = parser.parse_args()
    result = run_cycle(sys.executable, args.timesfm_python,
                       dry_run=args.dry_run)
    if result == "FAILED_REQUIRES_REVIEW":
        sys.exit(1)


if __name__ == "__main__":
    main()
