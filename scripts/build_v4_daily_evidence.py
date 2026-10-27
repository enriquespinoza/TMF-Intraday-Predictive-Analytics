"""Point-in-time, research-only V4 live component evidence generator.

Consumes *independently timestamped* market and Treasury history CSVs.
Reuses the frozen technical and Treasury scorers and Risk Governor V3.
Does not issue a V4 ledger state, V3/TimesFM forecast or broker order.
An explicit --run writes three unique, hash-linked upstream component
JSONs; --check is read-only. Missing attested evidence fails closed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from scripts import v4_opportunity_ledger as ledger
from scripts.market_session import require_issuance_window
from scripts.validate_tmf_snapshot_pair import validate_snapshot_pair
from src.features.technical_features import build_technical_features
from src.signals.technical_opportunity import (
    add_technical_opportunity_features, score_current_row as technical_current,
)
from src.signals.treasury_confirmation import (
    add_treasury_confirmation_features, score_current_row as treasury_current,
)
from src.signals.opportunity_state_machine import (
    OpportunityInputsV1, evaluate_opportunity_v1, load_policy_v1,
)
from src.risk.risk_governor_v2 import GovernorMemory, ExposureState
from src.risk.risk_governor_v3 import RiskInputsV3, evaluate_risk_v3

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data/raw/v4_inputs"
SOURCE_DIR = ledger.SOURCE_DIR
COMPONENT_DIR = ledger.COMPONENT_DIR
POLICY_PATH = ledger.POLICY_PATH

MARKET_COLUMNS = {
    "market_date", "open", "high", "low", "close", "volume",
    "available_at_utc",
}
TREASURY_COLUMNS = {
    "reference_date", "yield_30y", "real_yield_10y", "available_at_utc",
}
ROLE_FILENAMES = {
    "risk_governor_v3": "risk",
    "technical_opportunity_v1": "technical",
    "treasury_confirmation_v1": "treasury",
}
REQUIRED_RISK = (
    "return_1d", "volatility_5d", "volatility_20d", "ema_21",
    "change_30y_1d_bp", "change_30y_5d_bp",
    "real_yield_10y_change_5d_bp",
)


def _finite(value, label, *, positive=False):
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"Invalid or missing {label}")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _read_history(path, required, date_column):
    path = ledger.local_file(path, INPUT_DIR, ".csv")
    frame = pd.read_csv(path)
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"Missing required point-in-time source fields: {path.name}")
    frame[date_column] = pd.to_datetime(
        frame[date_column], errors="raise"
    ).dt.date.astype(str)
    if frame[date_column].duplicated().any():
        raise ValueError("Duplicated source dates")
    if not frame[date_column].is_monotonic_increasing:
        raise ValueError("Source rows must be chronological")
    times = pd.to_datetime(frame["available_at_utc"], utc=True, errors="raise")
    if times.isna().any() or any(
        pd.Timestamp(raw).tzinfo is None for raw in frame["available_at_utc"]
    ):
        raise ValueError("Availability timestamps must be explicit and timezone-aware")
    frame["_available"] = times
    return path, frame


def _reject_unavailable(frame, date_column, now, origin):
    if any(date > origin for date in frame[date_column]):
        raise ValueError("Source contains a future-reference-date observation")
    if any(frame["_available"] > now):
        raise ValueError("Source includes values unavailable at issuance")
    if any(frame["_available"].dt.date.astype(str) < frame[date_column]):
        raise ValueError("Source timestamp precedes its reference date")


def _market(origin, market_history, daily, now):
    path, frame = _read_history(market_history, MARKET_COLUMNS, "market_date")
    _reject_unavailable(frame, "market_date", now, origin)
    if frame["market_date"].iloc[-1] != origin:
        raise ValueError("Market source must end at current prospective origin")
    if len(frame) < 90:
        raise ValueError("Insufficient complete market history for V4 63-prior-row scoring")
    # No missing NYSE sessions: shifting five rows must mean five sessions.
    scheduled = ledger.CALENDAR.valid_days(
        start_date=frame["market_date"].iloc[0], end_date=origin,
    )
    if [str(x.date()) for x in scheduled] != frame["market_date"].tolist():
        raise ValueError("Market history has gaps/non-NYSE dates")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(dtype=float)
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError(f"Invalid market {column}")
    if any(frame["high"] < frame[["open", "close", "low"]].max(axis=1)):
        raise ValueError("Market high violates OHLC")
    if any(frame["low"] > frame[["open", "close", "high"]].min(axis=1)):
        raise ValueError("Market low violates OHLC")
    # A dated market row must have become available during its own
    # post-close window. This is an attestation, not a trusted external clock.
    for date, stamp in zip(frame["market_date"], frame["_available"]):
        require_issuance_window(date, now=stamp)
    # Compare all recent overlapping Schwab closes; this fails if history
    # uses a different split/dividend price basis than the live snapshot.
    sample = pd.read_csv(daily)
    if sample.empty or not {"market_date", "close"}.issubset(sample.columns):
        raise ValueError("Schwab daily snapshot missing market closes")
    sample["market_date"] = pd.to_datetime(
        sample["market_date"], errors="raise"
    ).dt.date.astype(str)
    if sample["market_date"].duplicated().any():
        raise ValueError("Duplicate Schwab daily market dates")
    overlap = frame[["market_date", "close"]].merge(
        sample[["market_date", "close"]], on="market_date",
        how="inner", suffixes=("_history", "_schwab"), validate="one_to_one",
    ).tail(10)
    if len(overlap) < 10:
        raise ValueError("Fewer than ten overlapping Schwab/source market sessions")
    if any(abs(overlap["close_history"] - overlap["close_schwab"]) > 0.001):
        raise ValueError("Market history not reconciled to Schwab price basis")
    return path, frame


def _treasury(origin, treasury_history, now):
    path, frame = _read_history(
        treasury_history, TREASURY_COLUMNS, "reference_date",
    )
    _reject_unavailable(frame, "reference_date", now, origin)
    if len(frame) < 70:
        raise ValueError("Insufficient valid Treasury observations")
    for column in ("yield_30y", "real_yield_10y"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if any(not math.isfinite(x) for x in frame[column]):
            raise ValueError(f"Missing Treasury data: {column}")
    if frame["reference_date"].iloc[-1] > origin:
        raise ValueError("Future Treasury reference date")
    # Old releases are acceptable only for the current market date when
    # exchange/Treasury calendars differ; large gaps fail below.
    frame["change_30y_1d_bp"] = frame["yield_30y"].diff() * 100
    frame["change_30y_5d_bp"] = frame["yield_30y"].diff(5) * 100
    frame["real_yield_10y_change_5d_bp"] = (
        frame["real_yield_10y"].diff(5) * 100
    )
    return path, frame


def _align_asof(market, treasury):
    # Find the last Treasury release whose *reference date* and *genuine
    # recorded availability timestamp* both precede each market decision.
    treasury_rows = []
    for _, row in market.iterrows():
        eligible = treasury.loc[
            (treasury["reference_date"] <= row["market_date"])
            & (treasury["_available"] <= row["_available"])
        ]
        if eligible.empty:
            treasury_rows.append(None)
        else:
            treasury_rows.append(eligible.iloc[-1])
    combined = market.copy()
    for column in (
        "reference_date", "change_30y_1d_bp", "change_30y_5d_bp",
        "real_yield_10y_change_5d_bp",
    ):
        combined[column] = [
            r[column] if r is not None else float("nan")
            for r in treasury_rows
        ]
    combined["treasury_age_days"] = [
        (pd.Timestamp(day) - pd.Timestamp(ref)).days if isinstance(ref, str)
        else float("nan")
        for day, ref in zip(combined["market_date"], combined["reference_date"])
    ]
    return combined


def _compute(origin, daily, market_history, treasury_history, *, now=None):
    origin = ledger.strict_session(origin)
    issued = ledger.now_utc(now)
    require_issuance_window(origin, now=issued)
    daily = ledger.local_file(daily, ledger.SOURCE_DIR, ".csv")
    if ledger.SOURCE_NAME.fullmatch(daily.name) is None:
        raise ValueError("Invalid Schwab snapshot filename")
    validation = validate_snapshot_pair(
        daily, now=issued, expected_session=origin,
    )
    if validation.get("status") != "POST_CLOSE_SNAPSHOT_VALIDATED":
        raise ValueError("Schwab pair failed validation")
    if validation.get("market_date") != origin:
        raise ValueError("Schwab market date mismatch")
    market_path, market = _market(origin, market_history, daily, issued)
    treasury_path, treasury = _treasury(origin, treasury_history, issued)
    aligned = _align_asof(market, treasury)
    # Fail closed on stale latest yield data rather than pretending a
    # missing current Treasury publication is a current observation.
    age = _finite(aligned["treasury_age_days"].iloc[-1], "Treasury age")
    if age < 0 or age > 4:
        raise ValueError("Latest point-in-time Treasury evidence is stale")
    # Reuse existing frozen V4 formulas; no threshold calibration.
    technical = build_technical_features(aligned.copy())
    below = technical["close"] < technical["ema_21"]
    streak = []
    count = 0
    for is_below in below:
        count = count + 1 if is_below else 0
        streak.append(count)
    technical["below_ema21_streak"] = streak
    scored_technical = add_technical_opportunity_features(technical)
    # Frozen Treasury scoring is on individual *Treasury observations*,
    # never a forward-filled market-day grid (which would alter percentiles).
    scored_treasury_history = add_treasury_confirmation_features(
        treasury[["reference_date", "change_30y_1d_bp",
                  "change_30y_5d_bp", "real_yield_10y_change_5d_bp"]]
        .rename(columns={"reference_date": "market_date"})
    )
    latest_treasury_ref = aligned["reference_date"].iloc[-1]
    scored_treasury = scored_treasury_history.loc[
        scored_treasury_history["market_date"] == latest_treasury_ref
    ]
    if len(scored_treasury) != 1:
        raise ValueError("No exactly matched Treasury score for point-in-time as-of")

    tech_result = technical_current(scored_technical.iloc[-1])
    treasury_result = treasury_current(scored_treasury.iloc[-1])
    # Hypothetical flat research state. Replay Risk Governor V3 from
    # chronological as-of source features, explicitly recording the
    # initialization assumption; no actual position is inferred.
    memory = GovernorMemory()
    valid_risk_days = 0
    decision = None
    for i in range(len(technical)):
        row = technical.iloc[i]
        trow = aligned.iloc[i]
        vals = [row.get("return_1d"), row.get("volatility_5d"),
                row.get("volatility_20d"), row.get("ema_21"),
                trow.get("change_30y_1d_bp"), trow.get("change_30y_5d_bp"),
                trow.get("real_yield_10y_change_5d_bp")]
        if not all(pd.notna(x) and math.isfinite(float(x)) for x in vals):
            continue
        if trow["treasury_age_days"] > 4:
            # Cannot assert an independently continuous governor state
            # across unavailable/stale observations.
            raise ValueError("Risk memory replay interrupted by stale Treasury data")
        ratio = _finite(
            row["volatility_5d"] / row["volatility_20d"],
            "realized volatility ratio", positive=True,
        )
        inputs = RiskInputsV3(
            tmf_return_1d_pct=float(row["return_1d"]) * 100,
            yield_30y_change_1d_bp=float(trow["change_30y_1d_bp"]),
            yield_30y_change_5d_bp=float(trow["change_30y_5d_bp"]),
            real_yield_10y_change_5d_bp=float(
                trow["real_yield_10y_change_5d_bp"]
            ),
            volatility_ratio_5d_20d=ratio,
            below_ema21=bool(row["close"] < row["ema_21"]),
            below_ema21_streak=int(streak[i]),
            critical_data_stale=False,
            position_open=False,
        )
        decision, memory = evaluate_risk_v3(inputs, memory=memory)
        valid_risk_days += 1
    if decision is None or valid_risk_days < 63:
        raise ValueError("Risk Governor replay lacks 63 complete sessions")
    # All current inputs must be available, not a stale earlier decision.
    latest = aligned.iloc[-1]
    if any(
        not pd.notna(latest[col]) or not math.isfinite(float(latest[col]))
        for col in (
            "change_30y_1d_bp", "change_30y_5d_bp",
            "real_yield_10y_change_5d_bp",
        )
    ):
        raise ValueError("Current Treasury risk inputs unavailable")
    policy = load_policy_v1(ledger.POLICY_PATH)
    opportunity = evaluate_opportunity_v1(OpportunityInputsV1(
        risk_entry_blocked=decision.entry_blocked,
        technical_opportunity_score=tech_result.technical_opportunity_score,
        treasury_confirmation_score=treasury_result.treasury_confirmation_score,
    ), policy)
    if float(validation["daily_close"]) != float(
        pd.read_csv(daily).iloc[-1]["close"]
    ):
        raise ValueError("Schwab close changed during evidence calculation")
    snap_hash = ledger.digest(daily)
    sources = {
        "market": {"filename": market_path.name, "sha256": ledger.digest(market_path)},
        "treasury": {
            "filename": treasury_path.name, "sha256": ledger.digest(treasury_path),
            "latest_reference_date": latest["reference_date"],
            "latest_age_days": int(age),
        },
    }
    common = {
        "schema_version": "v4_live_component_v1",
        "market_date": origin,
        "source_snapshot": daily.name,
        "source_sha256": snap_hash,
        "available_at_utc": issued.isoformat(),
        "generator": "build_v4_daily_evidence_v1",
        "point_in_time_sources": sources,
        "risk_assumption": (
            "hypothetical_flat_no_position; governor replay initialized "
            "GREEN from earliest complete historical risk session"
        ),
        "trading_authorization": "NOT_AUTHORIZED_RESEARCH_ONLY",
    }
    records = {
        "risk_governor_v3": {
            **common, "role": "risk_governor_v3",
            "new_entries_allowed": decision.new_entries_allowed,
            "exposure_state": decision.exposure_state.value,
            "raw_exposure_state": decision.raw_exposure_state.value,
            "entry_block_reasons": list(decision.entry_block_reasons),
            "exposure_reasons": list(decision.exposure_reasons),
            "recovery_streak": memory.recovery_streak,
            "risk_replay_sessions": valid_risk_days,
        },
        "technical_opportunity_v1": {
            **common, "role": "technical_opportunity_v1",
            "technical_opportunity_score":
                float(tech_result.technical_opportunity_score),
            "subscores": {
                "pullback_5d": tech_result.pullback_5d_score,
                "rsi_14": tech_result.rsi_score,
                "distance_20d_high": tech_result.distance_high_score,
                "stabilization_1d": tech_result.stabilization_score,
                "ema_pullback": tech_result.ema_pullback_score,
            },
        },
        "treasury_confirmation_v1": {
            **common, "role": "treasury_confirmation_v1",
            "treasury_confirmation_score":
                float(treasury_result.treasury_confirmation_score),
            "subscores": {
                "yield_30y_1d": treasury_result.yield_30y_1d_score,
                "yield_30y_5d": treasury_result.yield_30y_5d_score,
                "real_yield_10y_5d": treasury_result.real_yield_10y_5d_score,
            },
        },
    }
    return {
        "status": "CHECK_ONLY",
        "forecast_origin": origin,
        "snapshot": daily.name,
        "snapshot_sha256": snap_hash,
        "treasury_reference_date": latest["reference_date"],
        "risk_exposure_state": decision.exposure_state.value,
        "new_entries_allowed": decision.new_entries_allowed,
        "technical_opportunity_score": tech_result.technical_opportunity_score,
        "treasury_confirmation_score": treasury_result.treasury_confirmation_score,
        "v4_opportunity_state_preview": opportunity.state.value,
        "output_components": records,
        "order_generated": False,
        "research_only": True,
    }


def build(origin, daily, market_history, treasury_history, *, now=None, write=False):
    result = _compute(
        origin, daily, market_history, treasury_history, now=now,
    )
    names = {
        role: ledger.COMPONENT_DIR / (
            f"TMF_V4_{role}_{origin}_"
            f"{Path(result['snapshot']).stem.removeprefix('TMF_1d_')}.json"
        )
        for role in ROLE_FILENAMES
    }
    if any(path.exists() for path in names.values()):
        raise FileExistsError("V4 component evidence already exists; no overwrite")
    if write:
        ledger.COMPONENT_DIR.mkdir(parents=True, exist_ok=True)
        for role, path in names.items():
            with path.open("x", encoding="utf-8") as output:
                json.dump(result["output_components"][role], output,
                          indent=2, allow_nan=False)
                output.write("\n")
        result["status"] = "EVIDENCE_WRITTEN_LEDGER_NOT_YET_ISSUED"
    result["component_files"] = {role: str(path) for role, path in names.items()}
    # Audit-before-issue remains a separate explicit step.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--market-history", type=Path, required=True)
    parser.add_argument("--treasury-history", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = build(
        args.origin, args.daily, args.market_history, args.treasury_history,
        write=args.run,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
