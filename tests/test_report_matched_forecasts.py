"""Offline safety tests for matching previously scored V3 and TimesFM predictions."""

import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean

import pytest

from scripts.report_matched_forecasts import build_report, sha256

ORIGIN = "2026-10-21"
TARGET = "2026-10-28"
DAYS = ["2026-10-22", "2026-10-23", "2026-10-26", "2026-10-27", TARGET]
ACTUALS = [28.15, 28.27, 28.39, 28.49, 28.90]
MED = [28.086, 28.289, 28.269, 28.378, 28.376]
LOW = [27.074, 27.004, 26.749, 26.739, 26.536]
HIGH = [28.865, 29.254, 29.51, 29.941, 29.996]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


def mutate(path, **updates):
    d = json.loads(path.read_text())
    d.update(updates)
    write_json(path, d)


@pytest.fixture
def case(tmp_path):
    root = tmp_path
    v3 = root / "forward"
    tf = v3 / "timesfm25"
    sources = root / "sources"
    ve = v3 / "evaluations"
    te = tf / "evaluations"
    for d in (v3, tf, sources, ve, te, tf / "runs"):
        d.mkdir(parents=True, exist_ok=True)
    origin_source = sources / "TMF_1d_20261021T225435Z.csv"
    origin_source.write_text("previous verified market snapshot", encoding="utf-8")
    target_source = sources / "TMF_1d_20261028T210000Z.csv"
    with target_source.open("w", newline="") as h:
        writer = csv.writer(h)
        writer.writerow(["market_date", "close"])
        writer.writerow([ORIGIN, 28.22])
        writer.writerows(zip(DAYS, ACTUALS))
    intraday = sources / "TMF_5m_20261028T210000Z.csv"
    intraday.write_text("previously verified intraday snapshot", encoding="utf-8")

    v3_f = write_json(v3 / "TMF_V3_SCHWAB_2026-10-21_V3_FINAL_5d.json", {
        "model": "V3_FINAL", "order_generated": False,
        "issued_at_utc": "2026-10-21T22:54:47+00:00",
        "forecast_origin": ORIGIN, "forecast_target_date": TARGET,
        "horizon_trading_days": 5, "tmf_close": 28.22,
        "probability_positive_return": 0.5073,
        "training_base_rate": 0.43355,
        "source_snapshot": origin_source.name,
        "source_sha256": sha256(origin_source),
    })
    v3_e = write_json(ve / "TMF_V3_SCHWAB_2026-10-21_V3_FINAL_5d_evaluation.json", {
        "model": "V3_FINAL", "order_generated": False,
        "forecast_file": v3_f.name, "forecast_sha256": sha256(v3_f),
        "price_source": target_source.name,
        "price_source_sha256": sha256(target_source),
        "issued_at_utc": "2026-10-21T22:54:47+00:00",
        "scored_at_utc": "2026-10-28T22:00:00+00:00",
        "forecast_origin": ORIGIN, "forecast_target_date": TARGET,
        "horizon_trading_days": 5, "origin_close": 28.22,
        "target_close": ACTUALS[-1], "actual_positive_return": 1,
        "predicted_probability": 0.5073,
        "brier_score": (0.5073-1)**2,
        "direction_correct": True,
        "realized_return_pct": (ACTUALS[-1]/28.22-1)*100,
    })
    tf_f = write_json(tf / "TMF_TIMESFM25_2026-10-21_5d.json", {
        "model": "TIMESFM_2P5_200M", "order_generated": False,
        "issued_at_utc": "2026-10-22T13:13:06+00:00",
        "forecast_origin": ORIGIN, "forecast_target_date": TARGET,
        "horizon_trading_days": 5, "checkpoint_revision": "a"*40,
        "forecast_dates": DAYS, "tmf_close": 28.22,
        "median_forecast": MED, "q10_forecast": LOW, "q90_forecast": HIGH,
    })
    manifest = write_json(tf / "runs/TMF_TIMESFM25_2026-10-21_5d_manifest.json", {
        "status": "AUDITED_LEGACY", "forecast_sha256": sha256(tf_f),
        "forecast_file": tf_f.name, "origin": ORIGIN, "target": TARGET,
        "issued_at_utc": "2026-10-22T13:13:06+00:00",
        "checkpoint_revision": "a"*40, "order_generated": False,
    })
    diffs = [abs(a-b) for a,b in zip(ACTUALS, MED)]
    baseline_diffs = [abs(a-28.22) for a in ACTUALS]
    cover = [lo <= a <= hi for a,lo,hi in zip(ACTUALS, LOW, HIGH)]
    tf_e = write_json(te / "TMF_TIMESFM25_2026-10-21_5d_evaluation.json", {
        "status": "EVALUATED", "model": "TIMESFM_2P5_200M", "order_generated": False,
        "forecast_file": tf_f.name, "forecast_sha256": sha256(tf_f),
        "manifest_file": manifest.name, "manifest_sha256": sha256(manifest),
        "outcome_snapshot": target_source.name, "outcome_snapshot_sha256": sha256(target_source),
        "outcome_intraday_sha256": sha256(intraday),
        "forecast_origin": ORIGIN, "forecast_target_date": TARGET,
        "horizon_trading_days": 5, "checkpoint_revision": "a"*40,
        "issued_at_utc": "2026-10-22T13:13:06+00:00",
        "scored_at_utc": "2026-10-28T22:00:00+00:00",
        "origin_close": 28.22, "target_close": ACTUALS[-1],
        "actual_closes": ACTUALS, "forecast_dates": DAYS,
        "median_forecast": MED, "q10_forecast": LOW, "q90_forecast": HIGH,
        "realized_return_pct": (ACTUALS[-1]/28.22-1)*100,
        "median_mae": mean(diffs),
        "median_rmse": (mean(d*d for d in diffs))**.5,
        "persistence_mae": mean(baseline_diffs),
        "persistence_rmse": (mean(d*d for d in baseline_diffs))**.5,
        "target_median_absolute_error": diffs[-1],
        "target_persistence_absolute_error": baseline_diffs[-1],
        "q10_q90_coverage": mean(cover),
        "mean_q10_q90_width": mean(hi-lo for lo,hi in zip(LOW,HIGH)),
        "actual_positive_return": True,
        "median_predicted_positive_return": True,
        "direction_correct": True,
        "target_within_q10_q90": cover[-1],
    })
    return {
        "v3e": ve, "tfe": te, "v3": v3, "tf": tf, "sources": sources,
        "v3forecast": v3_f, "v3eval": v3_e,
        "tfforecast": tf_f, "tfeval": tf_e,
        "manifest": manifest, "outcome": target_source,
    }


def report(case):
    return build_report(case["v3e"], case["tfe"], case["v3"],
                        case["tf"], case["sources"],
                        as_of=datetime(2026, 10, 29, tzinfo=timezone.utc))


def test_no_matches_is_not_an_early_score(tmp_path):
    result = build_report(tmp_path/"none", tmp_path/"none2",
                          tmp_path/"v3", tmp_path/"tf", tmp_path/"sources")
    assert result["status"] == "NO_MATCHED_EVALUATIONS"
    assert result["matched_count"] == 0
    assert result["summary"] is None


def test_matched_forecasts_report_recalculated_metrics(case):
    r = report(case)
    assert r["matched_count"] == 1
    assert r["records"][0]["origin"] == ORIGIN
    assert r["summary"]["v3_mean_brier"] == pytest.approx((0.5073-1)**2)
    assert r["summary"]["timesfm_mean_path_mae"] < r["summary"]["persistence_mean_path_mae"]
    assert r["summary"]["timesfm_mean_q10_q90_coverage"] == pytest.approx(1)


def test_unmatched_not_reported_as_matched(case):
    case["tfeval"].unlink()
    r = report(case)
    assert r["matched_count"] == 0
    assert r["unmatched_v3"] == [[ORIGIN, TARGET]]


def test_no_files_modified_by_report(case):
    before = {str(p): sha256(p) for d in (case["v3"], case["tf"], case["sources"])
              for p in d.rglob("*") if p.is_file()}
    report(case)
    after = {str(p): sha256(p) for d in (case["v3"], case["tf"], case["sources"])
             for p in d.rglob("*") if p.is_file()}
    assert after == before


def test_rejects_altered_timesfm_forecast(case):
    mutate(case["tfforecast"], median_forecast=[29]*5)
    with pytest.raises(ValueError, match="TimesFM forecast SHA-256 mismatch"):
        report(case)


def test_rejects_altered_manifest(case):
    mutate(case["manifest"], status="CHANGED")
    with pytest.raises(ValueError, match="TimesFM manifest SHA-256 mismatch"):
        report(case)


def test_rejects_altered_outcome(case):
    case["outcome"].write_text("modified market price")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        report(case)


def test_rejects_duplicate_evaluations(case):
    duplicate = case["tfe"] / "copy_evaluation.json"
    duplicate.write_bytes(case["tfeval"].read_bytes())
    with pytest.raises(ValueError, match="Duplicate TimesFM evaluation"):
        report(case)


def test_rejects_brier_tamper(case):
    mutate(case["v3eval"], brier_score=.999)
    with pytest.raises(ValueError, match="V3 Brier mismatch"):
        report(case)


def test_rejects_timesfm_metric_tamper(case):
    mutate(case["tfeval"], median_mae=.999)
    with pytest.raises(ValueError, match="TimesFM median_mae mismatch"):
        report(case)


def test_rejects_inconsistent_outcomes(case):
    mutate(case["v3eval"], target_close=29.3,
           realized_return_pct=(29.3/28.22-1)*100)
    with pytest.raises(ValueError, match="V3 outcome snapshot close mismatch"):
        report(case)


def test_rejects_invalid_v3_baseline(case):
    mutate(case["v3forecast"], training_base_rate=1.7)
    with pytest.raises(ValueError, match="V3 forecast SHA-256 mismatch"):
        report(case)


def test_rejects_path_traversal(case):
    mutate(case["v3eval"], forecast_file="../secret.json")
    with pytest.raises(ValueError, match="Invalid artifact filename"):
        report(case)


def test_rejects_future_scoring_timestamp(case):
    with pytest.raises(ValueError, match="future"):
        build_report(case["v3e"], case["tfe"], case["v3"],
                     case["tf"], case["sources"],
                     as_of=datetime(2026, 10, 22, tzinfo=timezone.utc))


def test_rejects_manifest_metadata_tamper_even_with_changed_eval_hash(case):
    mutate(case["manifest"], origin="2026-10-20")
    mutate(case["tfeval"], manifest_sha256=sha256(case["manifest"]))
    with pytest.raises(ValueError, match="TimesFM manifest origin mismatch"):
        report(case)
