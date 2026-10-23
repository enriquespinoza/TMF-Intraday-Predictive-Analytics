"""Deterministic paired V3/TimesFM research monitor tests; no Schwab API."""

import json

import pandas as pd
import pytest

from scripts import report_paired_forecast_signals as paired

ORIGIN = "2026-10-21"
DAYS = ["2026-10-22", "2026-10-23", "2026-10-26",
        "2026-10-27", "2026-10-28"]


@pytest.fixture
def case(tmp_path, monkeypatch):
    sources = tmp_path / "schwab"
    sources.mkdir()
    v3dir = tmp_path / "forward"
    v3dir.mkdir()
    tfdir = v3dir / "timesfm25"
    tfdir.mkdir()
    history = tmp_path / "history.csv"
    history.write_text("history")

    origin_snapshot = sources / "TMF_1d_20261021T225435Z.csv"
    origin_snapshot.write_text("original source")
    v3 = v3dir / "v3.json"
    tf = tfdir / "tf.json"
    original = {
        "forecast_origin": ORIGIN,
        "forecast_target_date": DAYS[-1],
        "horizon_trading_days": 5,
        "source_snapshot": origin_snapshot.name,
        "source_sha256": paired.sha256(origin_snapshot),
        "tmf_close": 28.22,
        "order_generated": False,
    }
    v3.write_text(json.dumps({
        **original, "model": "V3_FINAL",
        "probability_positive_return": 0.5073,
        "training_base_rate": 0.4336,
    }))
    tf.write_text(json.dumps({
        **original, "model": "TIMESFM_2P5_200M",
        "checkpoint_revision": "a" * 40,
        "forecast_dates": DAYS,
        "median_forecast": [28.086, 28.289, 28.269, 28.378, 28.376],
        "q10_forecast": [27.074, 27.004, 26.749, 26.739, 26.536],
        "q90_forecast": [28.865, 29.254, 29.51, 29.941, 29.996],
    }))

    monkeypatch.setattr(paired, "SOURCE_DIR", sources)
    monkeypatch.setattr(paired, "V3_DIR", v3dir)
    monkeypatch.setattr(paired, "TIMESFM_DIR", tfdir)
    monkeypatch.setattr(paired, "HISTORY_PATH", history)
    monkeypatch.setattr(paired, "audit_v3", lambda origin: v3)
    monkeypatch.setattr(paired, "audit_timesfm", lambda origin, **kwargs: tf)

    snap = sources / "TMF_1d_20261022T221138Z.csv"
    bars = sources / "TMF_5m_20261022T221138Z.csv"
    pd.DataFrame({
        "market_date": [ORIGIN, DAYS[0]],
        "close": [28.22, 29.78],
    }).to_csv(snap, index=False)
    bars.write_text("validated synthetic companion")
    monkeypatch.setattr(
        paired, "validate_snapshot_pair",
        lambda path, now=None: {
            "market_date": "2026-10-22", "close_difference": 0.02
        },
    )
    return {"v3": v3, "tf": tf, "snapshot": snap,
            "source": origin_snapshot, "tfdir": tfdir}


def change(path, **values):
    data = json.loads(path.read_text())
    data.update(values)
    path.write_text(json.dumps(data))


def test_direction_divergence(case):
    change(case["v3"], probability_positive_return=0.392)
    change(case["tf"], median_forecast=[29, 29, 29, 29, 30.18],
           tmf_close=28.22)
    report = paired.build_paired_report(ORIGIN)
    assert report["directional_relation"] == "DISAGREE"
    assert report["order_generated"] is False
    assert report["observation"] is None


def test_matched_prospective_without_observations(case):
    result = paired.build_paired_report(ORIGIN)
    assert result["directional_relation"] == "AGREE_UP"
    assert result["forecast_target_date"] == DAYS[-1]


def test_partial_observation_outside_q90(case):
    before = case["tf"].read_bytes()
    result = paired.build_paired_report(
        ORIGIN, observation_daily=case["snapshot"]
    )
    obs = result["observation"]
    assert obs["status"] == "PARTIAL_OUTCOMES"
    assert obs["observed_steps"] == 1
    assert obs["rows"][0]["actual_close"] == 29.78
    assert obs["rows"][0]["within_q10_q90"] is False
    assert obs["timesfm_path_mae_to_date"] == pytest.approx(1.694)
    assert obs["v3_five_session_outcome"] == "NOT_SCORED_IN_THIS_REPORT"
    assert case["tf"].read_bytes() == before


def test_snapshot_mismatch_rejected(case):
    change(case["tf"], source_sha256="f" * 64)
    with pytest.raises(ValueError, match="different input"):
        paired.build_paired_report(ORIGIN)


def test_origin_mismatch_rejected(case):
    change(case["tf"], tmf_close=999)
    with pytest.raises(ValueError, match="origin closing prices"):
        paired.build_paired_report(ORIGIN)


def test_incorrect_target_rejected(case):
    change(case["tf"], forecast_target_date="2026-10-29")
    with pytest.raises(ValueError, match="target mismatch"):
        paired.build_paired_report(ORIGIN)


def test_missing_intermediate_date_rejected(case):
    change(case["tf"], forecast_dates=DAYS[:-1] + [DAYS[-2]])
    with pytest.raises(ValueError, match="horizon dates invalid"):
        paired.build_paired_report(ORIGIN)


def test_outcome_snapshot_outside_dir_rejected(case, tmp_path):
    outside = tmp_path / "TMF_1d_20261022T221138Z.csv"
    outside.write_bytes(case["snapshot"].read_bytes())
    with pytest.raises(ValueError, match="Schwab source directory"):
        paired.build_paired_report(ORIGIN, observation_daily=outside)


def test_missing_origin_observation_rejected(case):
    pd.DataFrame({
        "market_date": [DAYS[0]], "close": [29.78],
    }).to_csv(case["snapshot"], index=False)
    with pytest.raises(ValueError, match="Origin missing"):
        paired.build_paired_report(
            ORIGIN, observation_daily=case["snapshot"]
        )


def test_future_session_gap_rejected(case):
    df = pd.DataFrame({
        "market_date": [ORIGIN, DAYS[0], DAYS[2]],
        "close": [28.22, 29.78, 30.1],
    })
    df.to_csv(case["snapshot"], index=False)
    with pytest.raises(ValueError, match="future session"):
        paired.build_paired_report(
            ORIGIN, observation_daily=case["snapshot"]
        )
