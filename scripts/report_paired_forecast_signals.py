"""Read-only paired V3/TimesFM research monitor for *issued* forecasts.

Does not issue a prediction or score an unmatured five-session outcome.
Verified daily Schwab snapshots can supply intermediate TimesFM diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from scripts.run_daily_forecast import audit_existing_run as audit_v3
from scripts.timesfm25_integrity import (
    SOURCE_NAME, audit_existing_run as audit_timesfm, sha256,
)
from scripts.validate_tmf_snapshot_pair import validate_snapshot_pair

ROOT = Path(__file__).resolve().parents[1]
V3_DIR = ROOT / "data/forward_test"
TIMESFM_DIR = V3_DIR / "timesfm25"
SOURCE_DIR = ROOT / "data/raw/schwab"
HISTORY_PATH = ROOT / "data/raw/TMF_1d.csv"


def _json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Forecast must be a JSON object")
    return data


def _number(value, label):
    if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError(f"Invalid numeric value: {label}")
    return float(value)


def _direction(value):
    return "UP" if value > 0 else "DOWN" if value < 0 else "FLAT"


def _pair_status(v3_direction, timesfm_direction):
    if "FLAT" in (v3_direction, timesfm_direction):
        return "NEUTRAL"
    if v3_direction == timesfm_direction:
        return "AGREE_" + v3_direction
    return "DISAGREE"


def build_paired_report(origin, *, observation_daily=None, now=None):
    """Audit each frozen forecast then optionally inspect realized daily bars.

    No forecast, manifest, input data or evaluation file is written.
    """
    origin = str(pd.Timestamp(origin).date())
    v3_path = audit_v3(origin)
    timesfm_path = audit_timesfm(
        origin, output_dir=TIMESFM_DIR,
        source_dir=SOURCE_DIR, history_path=HISTORY_PATH,
    )
    if v3_path is None or timesfm_path is None:
        missing = [
            name for name, path in
            (("V3", v3_path), ("TimesFM", timesfm_path))
            if path is None
        ]
        raise RuntimeError("Missing audited paired forecast: " + ", ".join(missing))

    v3 = _json(v3_path)
    tf = _json(timesfm_path)
    if (
        v3.get("model") != "V3_FINAL"
        or tf.get("model") != "TIMESFM_2P5_200M"
        or v3.get("order_generated") is not False
        or tf.get("order_generated") is not False
        or v3.get("horizon_trading_days") != 5
        or tf.get("horizon_trading_days") != 5
        or v3.get("forecast_origin") != origin
        or tf.get("forecast_origin") != origin
    ):
        raise ValueError("Paired forecast model/origin mismatch")

    target = tf.get("forecast_target_date")
    if v3.get("forecast_target_date") != target:
        raise ValueError("Paired forecast target mismatch")
    # The October 21 pilot forecasts used separate independently audited
    # Schwab captures. Preserve that distinction instead of treating them
    # as a strict same-snapshot matched experiment.
    v3_source = v3.get("source_snapshot")
    tf_source = tf.get("source_snapshot")
    v3_source_hash = v3.get("source_sha256")
    tf_source_hash = tf.get("source_sha256")
    if not all(
        isinstance(x, str) and x
        for x in (v3_source, tf_source, v3_source_hash, tf_source_hash)
    ):
        raise ValueError("Missing paired input provenance")
    if v3_source == tf_source and v3_source_hash != tf_source_hash:
        raise ValueError("Same snapshot filename has different recorded hashes")
    same_snapshot = (
        v3_source == tf_source and v3_source_hash == tf_source_hash
    )

    v3_close = _number(v3.get("tmf_close"), "V3 origin price")
    tf_close = _number(tf.get("tmf_close"), "TimesFM origin price")
    if min(v3_close, tf_close) <= 0 or abs(v3_close - tf_close) > 0.001:
        raise ValueError("Paired forecast origin closing prices differ")

    p = _number(v3.get("probability_positive_return"), "V3 probability")
    if not 0 <= p <= 1:
        raise ValueError("Invalid V3 probability")
    med = tf.get("median_forecast")
    lower = tf.get("q10_forecast")
    upper = tf.get("q90_forecast")
    dates = tf.get("forecast_dates")
    if not all(isinstance(seq, list) and len(seq) == 5 for seq in
               (med, lower, upper, dates)):
        raise ValueError("TimesFM forecast vectors malformed")
    med = [_number(x, "median") for x in med]
    lower = [_number(x, "q10") for x in lower]
    upper = [_number(x, "q90") for x in upper]
    if any(not (0 < lo <= md <= hi) for lo, md, hi
           in zip(lower, med, upper)):
        raise ValueError("TimesFM quantile interval invalid")
    if dates[-1] != target or len(set(dates)) != 5:
        raise ValueError("TimesFM horizon dates invalid")

    v3_direction = _direction(p - 0.5)
    tf_return = (med[-1] / tf_close - 1) * 100
    tf_direction = _direction(tf_return)
    result = {
        "status": (
            "PAIRED_FORECASTS" if same_snapshot
            else "CROSS_SNAPSHOT_HISTORICAL_COMPARISON"
        ),
        "source_alignment": (
            "SAME_VALIDATED_SNAPSHOT" if same_snapshot
            else "DIFFERENT_AUDITED_SNAPSHOTS"
        ),
        "forecast_origin": origin,
        "forecast_target_date": target,
        "horizon_trading_days": 5,
        "input_snapshot": v3_source if same_snapshot else None,
        "input_snapshot_sha256": v3_source_hash if same_snapshot else None,
        "input_snapshots": {
            "v3": {"filename": v3_source, "sha256": v3_source_hash},
            "timesfm": {"filename": tf_source, "sha256": tf_source_hash},
        },
        "origin_close": tf_close,
        "v3": {
            "forecast_sha256": sha256(v3_path),
            "probability_positive_return": p,
            "training_base_rate": _number(
                v3.get("training_base_rate"), "V3 base rate"
            ),
            "direction_at_50pct": v3_direction,
        },
        "timesfm": {
            "forecast_sha256": sha256(timesfm_path),
            "checkpoint_revision": tf["checkpoint_revision"],
            "median_target_close": med[-1],
            "median_implied_return_pct": tf_return,
            "q10_target_close": lower[-1],
            "q90_target_close": upper[-1],
            "median_direction": tf_direction,
        },
        "directional_relation": _pair_status(v3_direction, tf_direction),
        "observation": None,
        "order_generated": False,
        "research_note": (
            "Opposing directions are observations, not trade signals. "
            "V3 probability and TimesFM quantiles are not equivalent. "
            "Partial-path diagnostics do not establish five-day predictive skill. "
            + (
                "The models used different separately audited snapshots; "
                "this is a historical comparison, not a strict same-source pair."
                if not same_snapshot else
                "Both models used the same validated Schwab snapshot."
            )
        ),
    }

    if observation_daily is None:
        return result

    # Observation eligibility is determined by the live session validator.
    # The caller cannot substitute arbitrary intraday data or retroactive
    # snapshots for a newly issued signal.
    observation_daily = Path(observation_daily).resolve()
    if observation_daily.parent != SOURCE_DIR.resolve():
        raise ValueError("Observation snapshot must be in Schwab source directory")
    if SOURCE_NAME.fullmatch(observation_daily.name) is None:
        raise ValueError("Observation snapshot name is invalid")
    validation = validate_snapshot_pair(observation_daily, now=now)
    observed_session = validation["market_date"]

    prices = pd.read_csv(observation_daily)
    if prices.empty or not {"market_date", "close"}.issubset(prices.columns):
        raise ValueError("Observation snapshot missing market closes")
    series_dates = pd.to_datetime(
        prices["market_date"], errors="raise"
    ).dt.date.astype(str)
    if series_dates.duplicated().any() or not series_dates.is_monotonic_increasing:
        raise ValueError("Observation dates duplicated or unordered")
    closes = pd.to_numeric(prices["close"], errors="raise").astype(float)
    if not all(math.isfinite(x) and x > 0 for x in closes):
        raise ValueError("Observation prices contain invalid values")
    if series_dates.iloc[-1] > observed_session:
        raise ValueError("Observation contains future session prices")
    if series_dates.iloc[-1] != observed_session:
        raise ValueError("Observation snapshot session mismatch")

    observed = dict(zip(series_dates, closes))
    if origin not in observed:
        raise ValueError("Origin missing from observation snapshot")
    if abs(observed[origin] - tf_close) > 0.001:
        raise ValueError("Observation origin differs from forecast origin")
    if observed_session < origin:
        raise ValueError("Observation is older than forecast origin")

    realized_dates = [day for day in dates if day <= observed_session]
    if any(day not in observed for day in realized_dates):
        raise ValueError("Gap in completed TimesFM forecast sessions")
    if any(day in observed for day in dates[len(realized_dates):]):
        raise ValueError("Observation contains future session prices")

    rows = []
    for i, day in enumerate(realized_dates):
        actual = float(observed[day])
        rows.append({
            "session": day,
            "actual_close": actual,
            "median_close": med[i],
            "q10_close": lower[i],
            "q90_close": upper[i],
            "median_absolute_error": abs(actual - med[i]),
            "persistence_absolute_error": abs(actual - tf_close),
            "within_q10_q90": lower[i] <= actual <= upper[i],
        })

    count = len(rows)
    status = "NO_OUTCOMES_YET" if count == 0 else (
        "TARGET_REACHED_AWAITING_OFFICIAL_EVALUATION"
        if count == 5 else "PARTIAL_OUTCOMES"
    )
    result["observation"] = {
        "status": status,
        "snapshot": observation_daily.name,
        "snapshot_sha256": sha256(observation_daily),
        "intraday_sha256": sha256(observation_daily.with_name(
            observation_daily.name.replace("TMF_1d_", "TMF_5m_", 1)
        )),
        "completed_session": observed_session,
        "snapshot_close_difference": validation["close_difference"],
        "observed_steps": count,
        "rows": rows,
        "timesfm_path_mae_to_date": (
            sum(row["median_absolute_error"] for row in rows) / count
            if count else None
        ),
        "persistence_path_mae_to_date": (
            sum(row["persistence_absolute_error"] for row in rows) / count
            if count else None
        ),
        "timesfm_quantile_coverage_to_date": (
            sum(row["within_q10_q90"] for row in rows) / count
            if count else None
        ),
        "v3_five_session_outcome": "NOT_SCORED_IN_THIS_REPORT",
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Read-only matched V3/TimesFM forecast and interim diagnostics."
    )
    parser.add_argument("--origin", required=True, help="Forecast origin YYYY-MM-DD")
    parser.add_argument(
        "--daily", type=Path,
        help="Validated post-close Schwab daily snapshot for interim diagnostics",
    )
    args = parser.parse_args()
    print(json.dumps(
        build_paired_report(args.origin, observation_daily=args.daily),
        indent=2,
    ))


if __name__ == "__main__":
    main()
