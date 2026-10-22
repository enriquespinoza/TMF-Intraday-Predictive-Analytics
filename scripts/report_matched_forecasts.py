"""Read-only matched V3/TimesFM research evaluation report.

Never creates or scores forecasts. Only matured, independently recorded
and source-verified evaluation pairs enter the matched sample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def safe_path(directory, name):
    if (not isinstance(name, str) or not name
            or Path(name).name != name or name in (".", "..")
            or "/" in name or "\\" in name):
        raise ValueError("Invalid artifact filename")
    return Path(directory) / name


def verified_file(directory, name, digest, label):
    path = safe_path(directory, name)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(digest, str) or len(digest) != 64 or sha256(path) != digest:
        raise ValueError(f"{label} SHA-256 mismatch: {path.name}")
    return path


def number(x, label):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"Invalid numeric {label}")
    return float(x)


def close(a, b, label, tol=1e-6):
    if abs(number(a, label) - number(b, label)) > tol:
        raise ValueError(f"{label} mismatch")


def verify_evaluation_time(e, as_of):
    stamp = e.get("scored_at_utc")
    if not isinstance(stamp, str):
        raise ValueError("Evaluation lacks scoring timestamp")
    try:
        scored = datetime.fromisoformat(stamp)
        issued = datetime.fromisoformat(e["issued_at_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid evaluation timestamps") from exc
    if scored.tzinfo is None or issued.tzinfo is None:
        raise ValueError("Evaluation timestamps must be timezone aware")
    if not (issued <= scored <= as_of):
        raise ValueError("Evaluation timestamp is future or before issuance")
    return scored


def key_from(evaluation):
    origin = evaluation.get("forecast_origin")
    target = evaluation.get("forecast_target_date")
    if not all(isinstance(v, str) and len(v) == 10 for v in (origin, target)):
        raise ValueError("Missing forecast origin/target")
    return (origin, target)


def collect(directory, label):
    records = {}
    for path in sorted(Path(directory).glob("*_evaluation.json")):
        data = read_json(path)
        key = key_from(data)
        if key in records:
            raise ValueError(f"Duplicate {label} evaluation: {key}")
        records[key] = (path, data)
    return records


def check_v3(entry, forecast_dir, source_dir, as_of):
    _, e = entry
    verify_evaluation_time(e, as_of)
    if e.get("model") != "V3_FINAL" or e.get("order_generated") is not False:
        raise ValueError("Invalid V3 evaluation model or order flag")
    forecast_path = verified_file(forecast_dir, e.get("forecast_file"),
                                  e.get("forecast_sha256"), "V3 forecast")
    price_path = verified_file(source_dir, e.get("price_source"),
                               e.get("price_source_sha256"), "V3 outcome")
    f = read_json(forecast_path)
    if f.get("model") != "V3_FINAL" or f.get("order_generated") is not False:
        raise ValueError("Invalid V3 forecast")
    for field in ("issued_at_utc", "forecast_origin", "forecast_target_date",
                  "horizon_trading_days"):
        if f.get(field) != e.get(field):
            raise ValueError(f"V3 {field} mismatch")
    if e.get("horizon_trading_days") != 5:
        raise ValueError("V3 horizon mismatch")
    verified_file(source_dir, f.get("source_snapshot"),
                  f.get("source_sha256"), "V3 origin")
    p = number(f.get("probability_positive_return"), "V3 probability")
    base = number(f.get("training_base_rate"), "V3 base rate")
    if not (0 <= p <= 1 and 0 <= base <= 1):
        raise ValueError("V3 probability out of range")
    actual = e.get("actual_positive_return")
    if type(actual) is not int or actual not in (0, 1):
        raise ValueError("V3 actual outcome must be binary")
    close(e.get("predicted_probability"), p, "V3 probability", 1e-12)
    close(e.get("brier_score"), (p - actual)**2, "V3 Brier", 1e-10)
    if e.get("direction_correct") is not (int(p > 0.5) == actual):
        raise ValueError("V3 direction mismatch")
    origin_close = number(e.get("origin_close"), "V3 origin close")
    target_close = number(e.get("target_close"), "V3 target close")
    if origin_close <= 0 or target_close <= 0:
        raise ValueError("V3 nonpositive close")
    close(origin_close, f.get("tmf_close"), "V3 origin close", 1e-4)
    close(e.get("realized_return_pct"),
          (target_close/origin_close-1)*100, "V3 return", 1e-7)
    if int(target_close > origin_close) != actual:
        raise ValueError("V3 price direction mismatch")
    price_rows = {}
    with price_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            date = row["market_date"]
            if date in price_rows:
                raise ValueError("V3 duplicate outcome dates")
            price_rows[date] = float(row["close"])
    for key, expected in ((e["forecast_origin"], origin_close),
                          (e["forecast_target_date"], target_close)):
        close(price_rows.get(key), expected, "V3 outcome snapshot close", 1e-4)
    return {
        "origin_close": origin_close, "target_close": target_close,
        "actual": bool(actual), "probability": p,
        "brier": (p-actual)**2, "baseline_brier": (base-actual)**2,
        "direction_correct": int(p > 0.5) == actual,
    }


def check_timesfm(entry, forecast_dir, source_dir, as_of):
    _, e = entry
    verify_evaluation_time(e, as_of)
    if e.get("status") != "EVALUATED" or e.get("model") != "TIMESFM_2P5_200M":
        raise ValueError("Invalid TimesFM evaluation")
    if e.get("order_generated") is not False or e.get("horizon_trading_days") != 5:
        raise ValueError("Invalid TimesFM evaluation horizon or order flag")
    forecast_path = verified_file(forecast_dir, e.get("forecast_file"),
                                  e.get("forecast_sha256"), "TimesFM forecast")
    manifest_path = verified_file(Path(forecast_dir) / "runs", e.get("manifest_file"),
                                  e.get("manifest_sha256"), "TimesFM manifest")
    outcome_path = verified_file(source_dir, e.get("outcome_snapshot"),
                                 e.get("outcome_snapshot_sha256"), "TimesFM outcome")
    verified_file(source_dir, e.get("outcome_snapshot").replace("TMF_1d_", "TMF_5m_", 1),
                  e.get("outcome_intraday_sha256"), "TimesFM five-minute outcome")
    f = read_json(forecast_path)
    manifest = read_json(manifest_path)
    for field, expected in (
        ("forecast_sha256", sha256(forecast_path)),
        ("forecast_file", forecast_path.name),
        ("origin", e["forecast_origin"]),
        ("target", e["forecast_target_date"]),
        ("issued_at_utc", e["issued_at_utc"]),
        ("checkpoint_revision", e["checkpoint_revision"]),
        ("order_generated", False),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"TimesFM manifest {field} mismatch")
    if manifest.get("status") not in ("GENERATED", "AUDITED_LEGACY"):
        raise ValueError("TimesFM manifest status mismatch")
    for field in ("model", "forecast_origin", "forecast_target_date",
                  "horizon_trading_days", "issued_at_utc", "checkpoint_revision"):
        if f.get(field) != e.get(field):
            raise ValueError(f"TimesFM {field} mismatch")
    if f.get("order_generated") is not False:
        raise ValueError("TimesFM forecast contains an order")
    for field in ("forecast_dates", "median_forecast", "q10_forecast", "q90_forecast"):
        if f.get(field) != e.get(field):
            raise ValueError(f"TimesFM {field} mismatch")
    dates = e.get("forecast_dates")
    if not isinstance(dates, list) or len(dates) != 5 or len(set(dates)) != 5:
        raise ValueError("Invalid TimesFM forecast dates")
    if dates[-1] != e.get("forecast_target_date"):
        raise ValueError("TimesFM target sequence mismatch")
    origin_close = number(e.get("origin_close"), "TimesFM origin close")
    close(origin_close, f.get("tmf_close"), "TimesFM origin close", 1e-4)
    actuals = e.get("actual_closes")
    if not isinstance(actuals, list) or len(actuals) != 5:
        raise ValueError("Invalid TimesFM actual closes")
    actual = [number(x, "actual close") for x in actuals]
    med = [number(x, "median close") for x in e["median_forecast"]]
    lo = [number(x, "q10 close") for x in e["q10_forecast"]]
    hi = [number(x, "q90 close") for x in e["q90_forecast"]]
    if any(min(a,m,l,h) <= 0 or not (l <= m <= h) for a,m,l,h in zip(actual, med, lo, hi)):
        raise ValueError("Invalid TimesFM prices or quantile order")
    close(e.get("target_close"), actual[-1], "TimesFM target", 1e-8)
    close(e.get("realized_return_pct"), (actual[-1]/origin_close-1)*100,
          "TimesFM return", 1e-7)
    errors = [abs(m-a) for m,a in zip(med, actual)]
    base_errors = [abs(origin_close-a) for a in actual]
    cover = sum(l <= a <= h for a,l,h in zip(actual,lo,hi))/5
    scores = {
        "median_mae": mean(errors),
        "median_rmse": math.sqrt(mean(x*x for x in errors)),
        "persistence_mae": mean(base_errors),
        "persistence_rmse": math.sqrt(mean(x*x for x in base_errors)),
        "target_median_absolute_error": errors[-1],
        "target_persistence_absolute_error": base_errors[-1],
        "q10_q90_coverage": cover,
        "mean_q10_q90_width": mean(h-l for h,l in zip(hi,lo)),
    }
    for name, value in scores.items():
        close(e.get(name), value, f"TimesFM {name}", 1e-8)
    direction = med[-1] > origin_close
    actual_positive = actual[-1] > origin_close
    for field, value in (("median_predicted_positive_return", direction),
                         ("actual_positive_return", actual_positive),
                         ("direction_correct", direction == actual_positive),
                         ("target_within_q10_q90", lo[-1] <= actual[-1] <= hi[-1])):
        if e.get(field) is not value:
            raise ValueError(f"TimesFM {field} mismatch")
    # Independently verify the outcome snapshot closes used in the evaluation.
    rows = {}
    with outcome_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            day = row["market_date"]
            if day in rows:
                raise ValueError("TimesFM duplicate outcome dates")
            rows[day] = float(row["close"])
    for day, expected in [(e["forecast_origin"], origin_close), *zip(dates, actual)]:
        close(rows.get(day), expected, "TimesFM outcome snapshot close", 1e-4)
    return {
        "origin_close": origin_close,
        "target_close": actual[-1],
        "actual": actual_positive,
        "median_target": med[-1],
        "direction_correct": direction == actual_positive,
        **scores,
    }


def build_report(v3_evaluations, timesfm_evaluations, v3_forecasts,
                 timesfm_forecasts, source_dir, *, as_of=None):
    """Return matched evaluated records; unmatched evaluations remain visible."""
    as_of = as_of or datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        raise ValueError("Report time must be timezone aware")
    v3 = collect(v3_evaluations, "V3")
    tf = collect(timesfm_evaluations, "TimesFM")
    matched = sorted(set(v3) & set(tf))
    records = []
    for key in matched:
        a = check_v3(v3[key], v3_forecasts, source_dir, as_of)
        b = check_timesfm(tf[key], timesfm_forecasts, source_dir, as_of)
        close(a["origin_close"], b["origin_close"], "Cross-model origin", 0.001)
        close(a["target_close"], b["target_close"], "Cross-model target", 0.001)
        if a["actual"] is not b["actual"]:
            raise ValueError("Cross-model realized direction mismatch")
        records.append({
            "origin": key[0], "target": key[1],
            "origin_close": a["origin_close"], "actual_target_close": b["target_close"],
            "realized_positive": a["actual"],
            "v3_probability_positive": a["probability"],
            "v3_direction_correct": a["direction_correct"],
            "v3_brier": a["brier"],
            "v3_training_base_rate_brier": a["baseline_brier"],
            "timesfm_median_target": b["median_target"],
            "timesfm_direction_correct": b["direction_correct"],
            "timesfm_path_mae": b["median_mae"],
            "timesfm_path_rmse": b["median_rmse"],
            "timesfm_target_absolute_error": b["target_median_absolute_error"],
            "persistence_path_mae": b["persistence_mae"],
            "persistence_path_rmse": b["persistence_rmse"],
            "persistence_target_absolute_error": b["target_persistence_absolute_error"],
            "timesfm_q10_q90_coverage": b["q10_q90_coverage"],
        })
    result = {
        "status": "MATCHED_EVALUATIONS" if records else "NO_MATCHED_EVALUATIONS",
        "matched_count": len(records),
        "unmatched_v3": [list(k) for k in sorted(set(v3) - set(tf))],
        "unmatched_timesfm": [list(k) for k in sorted(set(tf) - set(v3))],
        "records": records,
        "summary": None,
        "research_note": (
            "Forecast samples can overlap; accuracy and interval coverage are descriptive, "
            "not evidence of statistically independent predictive skill. "
            "V3 Brier is a probability metric; TimesFM MAE is a price metric."
        ),
    }
    if records:
        result["summary"] = {
            "v3_directional_accuracy": mean(int(r["v3_direction_correct"]) for r in records),
            "v3_mean_brier": mean(r["v3_brier"] for r in records),
            "v3_base_rate_mean_brier": mean(r["v3_training_base_rate_brier"] for r in records),
            "timesfm_directional_accuracy": mean(int(r["timesfm_direction_correct"]) for r in records),
            "timesfm_mean_path_mae": mean(r["timesfm_path_mae"] for r in records),
            "persistence_mean_path_mae": mean(r["persistence_path_mae"] for r in records),
            "timesfm_mean_target_absolute_error": mean(
                r["timesfm_target_absolute_error"] for r in records),
            "persistence_mean_target_absolute_error": mean(
                r["persistence_target_absolute_error"] for r in records),
            "timesfm_mean_q10_q90_coverage": mean(r["timesfm_q10_q90_coverage"] for r in records),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    base = Path("data/forward_test")
    parser.add_argument("--v3-evaluations", type=Path, default=base / "evaluations")
    parser.add_argument("--timesfm-evaluations", type=Path,
                        default=base / "timesfm25/evaluations")
    parser.add_argument("--v3-forecasts", type=Path, default=base)
    parser.add_argument("--timesfm-forecasts", type=Path,
                        default=base / "timesfm25")
    parser.add_argument("--sources", type=Path, default=Path("data/raw/schwab"))
    args = parser.parse_args()
    result = build_report(args.v3_evaluations, args.timesfm_evaluations,
                          args.v3_forecasts, args.timesfm_forecasts, args.sources)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
