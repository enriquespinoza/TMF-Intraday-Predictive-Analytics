"""Prospective, independent V4 Opportunity Engine V1 research ledger.

Issue: compute/record one V4 state from independently produced, timestamped
component JSON files during the ORIGINAL prospective post-close window.
Score: separately evaluate the five completed NYSE sessions from an
independently validated outcome snapshot. Report: summarize audited scores.

No historical state backfill, no retrospective issuance, no V3/TimesFM writes,
no allocation, no broker interaction or executable instruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import pandas as pd
import pandas_market_calendars as mcal

from scripts.market_session import require_issuance_window
from scripts.timesfm25_integrity import SOURCE_NAME
from scripts.validate_tmf_snapshot_pair import validate_snapshot_pair
from src.signals.opportunity_state_machine import (
    OpportunityInputsV1, evaluate_opportunity_v1, load_policy_v1,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data/raw/schwab"
COMPONENT_DIR = ROOT / "data/raw/v4_live"
POLICY_PATH = ROOT / "config/opportunity_engine_v1_policy.json"
LEDGER_DIR = ROOT / "data/forward_test/v4_opportunity"
EVAL_DIR = LEDGER_DIR / "evaluations"
ROLE_FIELDS = {
    "risk_governor_v3": None,
    "technical_opportunity_v1": "technical_opportunity_score",
    "treasury_confirmation_v1": "treasury_confirmation_score",
}
SCHEMA = "v4_opportunity_prospective_v1"
CALENDAR = mcal.get_calendar("NYSE")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Timestamp must be timezone aware")
    return stamp.tz_convert("UTC")


def now_utc(now=None):
    return utc(now if now is not None else datetime.now(timezone.utc))


def strict_session(session):
    if not isinstance(session, str) or len(session) != 10:
        raise ValueError("Origin must be ISO YYYY-MM-DD")
    day = pd.Timestamp(session).date()
    if day.isoformat() != session:
        raise ValueError("Origin must be ISO YYYY-MM-DD")
    return session


def horizon(origin):
    days = CALENDAR.valid_days(
        start_date=origin,
        end_date=pd.Timestamp(origin) + pd.Timedelta(days=21),
    )
    dates = [str(day.date()) for day in days]
    if not dates or dates[0] != origin or len(dates) < 6:
        raise ValueError("Origin is not a valid NYSE market session")
    return dates[1:6]


def local_file(path, root, suffix):
    path = Path(path).resolve()
    root = Path(root).resolve()
    if path.parent != root or path.suffix.lower() != suffix:
        raise ValueError(f"Input must be a {suffix} file directly under {root}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def no_future_fields(obj):
    """Reject research target columns from supposedly contemporaneous evidence."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower().startswith((
                "forward_", "future_", "target_return", "actual_return",
            )):
                raise ValueError("Future outcome fields prohibited in V4 evidence")
            no_future_fields(value)
    elif isinstance(obj, list):
        for value in obj:
            no_future_fields(value)


def finite_score(value, role):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 100):
        raise ValueError(f"Invalid {role} score (expected 0..100)")
    return float(value)


def component(path, role, origin, snapshot_name, snapshot_hash, *, issued, finalized):
    path = local_file(path, COMPONENT_DIR, ".json")
    data = read_json(path)
    no_future_fields(data)
    if data.get("schema_version") != "v4_live_component_v1":
        raise ValueError("Invalid V4 component schema")
    if data.get("role") != role or data.get("market_date") != origin:
        raise ValueError(f"V4 {role} date/role mismatch")
    if data.get("source_snapshot") != snapshot_name:
        raise ValueError(f"V4 {role} snapshot mismatch")
    if data.get("source_sha256") != snapshot_hash:
        raise ValueError(f"V4 {role} snapshot digest mismatch")
    observed_at = utc(data.get("available_at_utc"))
    if not finalized <= observed_at <= issued:
        raise ValueError(f"V4 {role} evidence unavailable at issuance")
    if role == "risk_governor_v3":
        if type(data.get("new_entries_allowed")) is not bool:
            raise ValueError("Risk governor entry permission must be boolean")
        state = data.get("exposure_state")
        if state not in ("GREEN", "YELLOW", "RED", "EMERGENCY_EXIT"):
            raise ValueError("Unknown risk exposure state")
        if data["new_entries_allowed"] and state != "GREEN":
            raise ValueError("Non-green risk cannot permit a new entry")
        value = data["new_entries_allowed"]
    else:
        value = finite_score(data.get(ROLE_FIELDS[role]), role)
    return {
        "role": role, "file": path.name, "sha256": digest(path),
        "available_at_utc": observed_at.isoformat(), "value": value,
        "risk_exposure_state": data.get("exposure_state") if role == "risk_governor_v3" else None,
    }


def paths_for(origin):
    strict_session(origin)
    return (
        LEDGER_DIR / f"TMF_V4_OPPORTUNITY_{origin}_5d.json",
        LEDGER_DIR / "runs" / f"TMF_V4_OPPORTUNITY_{origin}_5d_manifest.json",
    )


def _build_issue(origin, daily, risk, technical, treasury, *, now=None):
    origin = strict_session(origin)
    issued = now_utc(now)
    window = require_issuance_window(origin, now=issued)
    daily = local_file(daily, SOURCE_DIR, ".csv")
    validation = validate_snapshot_pair(daily, now=issued, expected_session=origin)
    if validation.get("status") != "POST_CLOSE_SNAPSHOT_VALIDATED":
        raise ValueError("Not a validated post-close source")
    if validation.get("market_date") != origin:
        raise ValueError("Forecast source does not match origin")
    snap_hash = digest(daily)
    inputs = {}
    for role, path in (
        ("risk_governor_v3", risk),
        ("technical_opportunity_v1", technical),
        ("treasury_confirmation_v1", treasury),
    ):
        inputs[role] = component(
            path, role, origin, daily.name, snap_hash,
            issued=issued, finalized=utc(window.finalized_at_utc),
        )
    policy = load_policy_v1(POLICY_PATH)
    decision = evaluate_opportunity_v1(OpportunityInputsV1(
        risk_entry_blocked=not inputs["risk_governor_v3"]["value"],
        technical_opportunity_score=inputs["technical_opportunity_v1"]["value"],
        treasury_confirmation_score=inputs["treasury_confirmation_v1"]["value"],
    ), policy)
    target_days = horizon(origin)
    intraday = daily.with_name(daily.name.replace("TMF_1d_", "TMF_5m_", 1))
    if not intraday.is_file():
        raise FileNotFoundError(intraday)
    record = {
        "schema_version": SCHEMA,
        "status": "ISSUED",
        "model": "V4_OPPORTUNITY_ENGINE_V1",
        "forecast_origin": origin,
        "forecast_target_date": target_days[-1],
        "forecast_dates": target_days,
        "horizon_trading_days": 5,
        "issued_at_utc": issued.isoformat(),
        "origin_close": float(validation["daily_close"]),
        "source_snapshot": daily.name,
        "source_sha256": snap_hash,
        "source_intraday_sha256": digest(intraday),
        "policy_file": POLICY_PATH.name,
        "policy_sha256": digest(POLICY_PATH),
        "components": inputs,
        "decision": {
            "opportunity_state": decision.state.value,
            "entry_candidate": decision.entry_candidate,
            "risk_entry_blocked": decision.risk_entry_blocked,
            "technical_opportunity_score": decision.technical_opportunity_score,
            "treasury_confirmation_score": decision.treasury_confirmation_score,
            "reasons": list(decision.reasons),
        },
        "source_attestation": "LOCAL_COMPONENT_HASHES_ONLY",
        "trading_authorization": "NOT_AUTHORIZED_RESEARCH_ONLY",
        "order_generated": False,
    }
    return record


def issue(origin, daily, risk, technical, treasury, *, now=None, write=False):
    record = _build_issue(origin, daily, risk, technical, treasury, now=now)
    path, manifest_path = paths_for(origin)
    if path.exists() or manifest_path.exists():
        raise FileExistsError("V4 opportunity issuance already exists; never overwrite")
    if not write:
        return None, record
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    manifest = {
        "schema_version": SCHEMA,
        "record_file": path.name,
        "record_sha256": digest(path),
        "forecast_origin": origin,
        "issued_at_utc": record["issued_at_utc"],
        "order_generated": False,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return path, record


def audit(origin):
    """Audit the complete saved state, policy, component source and origin pair."""
    path, manifest_path = paths_for(origin)
    if not path.exists() and not manifest_path.exists():
        return None
    if not path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Incomplete V4 opportunity record/manifest")
    record = read_json(path)
    manifest = read_json(manifest_path)
    if record.get("schema_version") != SCHEMA or record.get("status") != "ISSUED":
        raise ValueError("Invalid V4 record schema/status")
    if record.get("model") != "V4_OPPORTUNITY_ENGINE_V1":
        raise ValueError("Invalid V4 model identity")
    if (record.get("order_generated") is not False
            or record.get("trading_authorization") != "NOT_AUTHORIZED_RESEARCH_ONLY"):
        raise ValueError("V4 record cannot authorize execution")
    for field, expected in (
        ("record_file", path.name),
        ("record_sha256", digest(path)),
        ("forecast_origin", origin),
        ("issued_at_utc", record.get("issued_at_utc")),
        ("order_generated", False),
        ("schema_version", SCHEMA),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"V4 manifest mismatch: {field}")
    if (record.get("forecast_origin") != origin
            or record.get("forecast_dates") != horizon(origin)
            or record.get("forecast_target_date") != horizon(origin)[-1]
            or record.get("horizon_trading_days") != 5):
        raise ValueError("Invalid V4 session/horizon")
    issued = utc(record.get("issued_at_utc"))
    if issued > now_utc():
        raise ValueError("V4 record was issued in the future")
    # Historical verification of issuance window; NEVER used to issue a state.
    window = require_issuance_window(origin, now=issued)
    daily = local_file(SOURCE_DIR / record.get("source_snapshot", ""), SOURCE_DIR, ".csv")
    if digest(daily) != record.get("source_sha256"):
        raise ValueError("V4 origin snapshot modified")
    snapshot_match = SOURCE_NAME.fullmatch(daily.name)
    if snapshot_match is None:
        raise ValueError("Invalid V4 source snapshot name")
    captured = pd.to_datetime(
        snapshot_match.group(1), format="%Y%m%dT%H%M%SZ",
        utc=True, errors="raise",
    )
    if captured > issued:
        raise ValueError("V4 snapshot captured after issuance")
    original = validate_snapshot_pair(daily, now=captured, expected_session=origin)
    if original.get("status") != "POST_CLOSE_SNAPSHOT_VALIDATED":
        raise ValueError("V4 original snapshot validation failed")
    if abs(float(original["daily_close"]) - float(record.get("origin_close"))) > 0.0001:
        raise ValueError("V4 origin closing price mismatch")
    intraday = daily.with_name(daily.name.replace("TMF_1d_", "TMF_5m_", 1))
    if digest(intraday) != record.get("source_intraday_sha256"):
        raise ValueError("V4 origin intraday source modified")
    if record.get("policy_file") != POLICY_PATH.name:
        raise ValueError("V4 policy source mismatch")
    if digest(POLICY_PATH) != record.get("policy_sha256"):
        raise ValueError("V4 frozen policy was modified")
    sources = record.get("components")
    if not isinstance(sources, dict) or set(sources) != set(ROLE_FIELDS):
        raise ValueError("Missing V4 component records")
    for role in ROLE_FIELDS:
        saved = sources[role]
        current = component(
            COMPONENT_DIR / saved["file"], role, origin,
            daily.name, digest(daily),
            issued=issued, finalized=utc(window.finalized_at_utc),
        )
        if current != saved:
            raise ValueError(f"V4 {role} source/decision modified")
    policy = load_policy_v1(POLICY_PATH)
    evaluated = evaluate_opportunity_v1(OpportunityInputsV1(
        risk_entry_blocked=not sources["risk_governor_v3"]["value"],
        technical_opportunity_score=sources["technical_opportunity_v1"]["value"],
        treasury_confirmation_score=sources["treasury_confirmation_v1"]["value"],
    ), policy)
    expected_decision = {
        "opportunity_state": evaluated.state.value,
        "entry_candidate": evaluated.entry_candidate,
        "risk_entry_blocked": evaluated.risk_entry_blocked,
        "technical_opportunity_score": evaluated.technical_opportunity_score,
        "treasury_confirmation_score": evaluated.treasury_confirmation_score,
        "reasons": list(evaluated.reasons),
    }
    if record.get("decision") != expected_decision:
        raise ValueError("V4 frozen decision replay mismatch")
    return record


def score(origin, outcome_daily, *, now=None, write=False):
    """Score only after a complete 5-NYSE-session outcome snapshot exists."""
    record = audit(origin)
    if record is None:
        raise FileNotFoundError(f"No prospective V4 opportunity issuance: {origin}")
    now = now_utc(now)
    outcome_daily = local_file(outcome_daily, SOURCE_DIR, ".csv")
    match = SOURCE_NAME.fullmatch(outcome_daily.name)
    if match is None:
        raise ValueError("Invalid outcome snapshot filename")
    captured = pd.to_datetime(
        match.group(1), format="%Y%m%dT%H%M%SZ", utc=True, errors="raise"
    )
    if captured > now:
        raise ValueError("Outcome snapshot captured in future")
    validated = validate_snapshot_pair(
        outcome_daily, now=captured,
        expected_session=record["forecast_target_date"],
    )
    if validated["market_date"] != record["forecast_target_date"]:
        raise ValueError("V4 outcome is not the forecast target date")
    prices = pd.read_csv(outcome_daily)
    if prices.empty or not {"market_date", "close"}.issubset(prices.columns):
        raise ValueError("V4 outcome snapshot missing closes")
    dates = pd.to_datetime(prices["market_date"], errors="raise").dt.date.astype(str)
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Invalid V4 outcome session order")
    closes = pd.to_numeric(prices["close"], errors="raise")
    if any(not math.isfinite(float(x)) or float(x) <= 0 for x in closes):
        raise ValueError("Invalid V4 outcome prices")
    if dates.iloc[-1] != record["forecast_target_date"]:
        raise ValueError("V4 outcome has wrong final market date")
    close_by_date = dict(zip(dates, closes.astype(float)))
    required = [origin, *record["forecast_dates"]]
    if any(day not in close_by_date for day in required):
        raise ValueError("V4 outcome missing intermediate trading dates")
    origin_close = float(record["origin_close"])
    if abs(close_by_date[origin] - origin_close) > 0.001:
        raise ValueError("V4 outcome origin close differs from issuance")
    actual_closes = [close_by_date[day] for day in record["forecast_dates"]]
    returns = [(price / origin_close - 1) * 100 for price in actual_closes]
    evaluation = {
        "schema_version": SCHEMA,
        "status": "EVALUATED",
        "model": "V4_OPPORTUNITY_ENGINE_V1",
        "forecast_origin": origin,
        "forecast_target_date": record["forecast_target_date"],
        "issued_at_utc": record["issued_at_utc"],
        "scored_at_utc": now.isoformat(),
        "record_file": paths_for(origin)[0].name,
        "record_sha256": digest(paths_for(origin)[0]),
        "state": record["decision"]["opportunity_state"],
        "entry_candidate": record["decision"]["entry_candidate"],
        "origin_close": origin_close,
        "outcome_snapshot": outcome_daily.name,
        "outcome_snapshot_sha256": digest(outcome_daily),
        "outcome_intraday_sha256": digest(outcome_daily.with_name(
            outcome_daily.name.replace("TMF_1d_", "TMF_5m_", 1)
        )),
        "forecast_dates": record["forecast_dates"],
        "actual_closes": actual_closes,
        "five_session_return_pct": returns[-1],
        "positive_five_session_return": returns[-1] > 0,
        "close_path_mae_pct": min(returns),
        "close_path_mfe_pct": max(returns),
        "order_generated": False,
        "note": "State-conditional descriptive outcome, NOT a simulated strategy trade.",
    }
    dest = EVAL_DIR / f"TMF_V4_OPPORTUNITY_{origin}_5d_evaluation.json"
    if write:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("x", encoding="utf-8") as handle:
            json.dump(evaluation, handle, indent=2, allow_nan=False)
            handle.write("\n")
    return dest if write else None, evaluation


def report():
    groups = defaultdict(list)
    for path in sorted(EVAL_DIR.glob("TMF_V4_OPPORTUNITY_*_5d_evaluation.json")):
        evaluation = read_json(path)
        origin = evaluation.get("forecast_origin")
        if not isinstance(origin, str):
            raise ValueError("V4 evaluation lacks origin")
        verified_now = now_utc()
        scored_at = utc(evaluation.get("scored_at_utc"))
        source_name = evaluation.get("outcome_snapshot")
        match = SOURCE_NAME.fullmatch(source_name) if isinstance(source_name, str) else None
        if match is None:
            raise ValueError("V4 evaluation outcome filename invalid")
        captured = pd.to_datetime(
            match.group(1), format="%Y%m%dT%H%M%SZ",
            utc=True, errors="raise",
        )
        if not captured <= scored_at <= verified_now:
            raise ValueError("V4 evaluation timestamp precedes outcome or is future")
        _, rebuilt = score(
            origin, SOURCE_DIR / source_name,
            now=verified_now, write=False,
        )
        # Timestamp is a previously recorded audit field, not a score input.
        rebuilt["scored_at_utc"] = evaluation["scored_at_utc"]
        if rebuilt != evaluation:
            raise ValueError(f"V4 evaluation does not match audited data: {path.name}")
        groups[evaluation["state"]].append(evaluation)
    summary = []
    for state, records in sorted(groups.items()):
        summary.append({
            "state": state,
            "n": len(records),
            "mean_five_session_return_pct": mean(
                x["five_session_return_pct"] for x in records),
            "positive_five_session_rate": mean(
                x["positive_five_session_return"] for x in records),
            "mean_close_path_mae_pct": mean(
                x["close_path_mae_pct"] for x in records),
            "mean_close_path_mfe_pct": mean(
                x["close_path_mfe_pct"] for x in records),
            "sample_warning": "VERY_SMALL_SAMPLE" if len(records) < 10 else None,
        })
    return {
        "status": "MATURED_V4_EVALUATIONS" if summary else "NO_MATURED_V4_EVALUATIONS",
        "matured_count": sum(len(v) for v in groups.values()),
        "by_state": summary,
        "research_only": True,
        "note": (
            "Overlapping five-session windows are not independent. "
            "Descriptive state-conditioned returns are NOT trade P&L, "
            "proof of profitable entry timing, or evidence of execution authority."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p_issue = commands.add_parser("issue")
    p_issue.add_argument("--origin", required=True)
    p_issue.add_argument("--daily", type=Path, required=True)
    for flag in ("risk", "technical", "treasury"):
        p_issue.add_argument("--" + flag, required=True, type=Path)
    mode = p_issue.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    p_score = commands.add_parser("score")
    p_score.add_argument("--origin", required=True)
    p_score.add_argument("--daily", type=Path, required=True)
    mode = p_score.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    commands.add_parser("report")
    args = parser.parse_args()
    if args.command == "issue":
        dest, result = issue(
            args.origin, args.daily, args.risk, args.technical, args.treasury,
            write=args.run,
        )
        print(json.dumps({"path": str(dest) if dest else None, **result}, indent=2))
    elif args.command == "score":
        dest, result = score(args.origin, args.daily, write=args.run)
        print(json.dumps({"path": str(dest) if dest else None, **result}, indent=2))
    else:
        print(json.dumps(report(), indent=2))


if __name__ == "__main__":
    main()
