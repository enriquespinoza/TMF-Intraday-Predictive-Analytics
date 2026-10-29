"""PAYEMS-A 2026 prospective release refresh tests (offline/synthetic)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import refresh_payems_a as refresh
from scripts.payems_a_readiness import require_payems_ready


def files(tmp_path):
    path = tmp_path / "PAYEMS_event_features.csv"
    pd.DataFrame([{
        "event_name": "employment_situation",
        "event_reference_date": "2026-09-01",
        "event_release_date": "2026-10-02",
        "event_release_time": "08:30",
        "event_release_timezone": "America/New_York",
        "reference_date": "2026-09-01",
        "payroll_level": 1010.0,
        "payroll_change_1m": 10.0,
        "payroll_revision_1m": 0.0,
        "payroll_revision_2m": 0.0,
        "payroll_revision_total": 0.0,
        "payroll_initial_periods": 1,
        "payroll_revised_periods": 0,
        "payroll_broad_revision_event": 0,
        "payroll_change_z": 0.0,
        "payroll_revision_z": 0.0,
    }]).to_csv(path, index=False)
    policy = tmp_path / "schedule.json"
    policy.write_text(json.dumps({"events": [{
        "reference_month": "2026-10-01",
        "release_date": "2026-11-06",
        "release_time": "08:30",
        "release_timezone": "America/New_York",
    }]}), encoding="utf-8")
    return path, policy, tmp_path / "stage", path.with_suffix(".refresh.json")


def vintages():
    def row(ref, vintage, value):
        return {
            "series_name": "nonfarm_payrolls", "series_id": "PAYEMS",
            "reference_date": ref, "vintage_date": vintage,
            "value_as_known": value,
        }
    return pd.DataFrame([
        row("2026-08-01", "2026-09-04", 1000),
        row("2026-09-01", "2026-10-02", 1010),
        row("2026-08-01", "2026-11-06", 1001),
        row("2026-09-01", "2026-11-06", 1012),
        row("2026-10-01", "2026-11-06", 1025),
    ])


def test_pre_release_stays_on_existing_event(tmp_path):
    events, policy, stage, sidecar = files(tmp_path)
    assert require_payems_ready(
        "2026-11-05", events_path=events, schedule_path=policy,
        sidecar_path=sidecar) == "2026-10-02"


def test_missing_update_fails_closed_on_release_date(tmp_path):
    events, policy, stage, sidecar = files(tmp_path)
    with pytest.raises(ValueError, match="PAYEMS_A_REFRESH_REQUIRED"):
        require_payems_ready(
            "2026-11-06", events_path=events, schedule_path=policy,
            sidecar_path=sidecar)


def test_stage_before_release_rejected(tmp_path):
    events, policy, stage, sidecar = files(tmp_path)
    with pytest.raises(ValueError, match="before official"):
        refresh.stage(
            "2026-11-06", baseline_path=events, schedule_path=policy,
            stage_dir=stage, now="2026-11-06T12:00:00Z",
            vintage_data=vintages())
    assert not stage.exists()


def test_staged_candidate_is_append_only_and_immutable(tmp_path):
    events, policy, stage, sidecar = files(tmp_path)
    old = events.read_bytes()
    result = refresh.stage(
        "2026-11-06", baseline_path=events, schedule_path=policy,
        stage_dir=stage, now="2026-11-06T14:00:00Z",
        vintage_data=vintages())
    assert result["status"] == "STAGED_NOT_ACTIVATED"
    assert events.read_bytes() == old
    candidate = pd.read_csv(stage / "PAYEMS_A_2026-11-06_candidate.csv")
    assert len(candidate) == 2
    assert candidate.iloc[-1]["payroll_change_1m"] == 13
    assert candidate.iloc[-1]["payroll_revision_total"] == 3
    assert candidate.iloc[-1]["payroll_broad_revision_event"] == 0
    with pytest.raises(FileExistsError):
        refresh.stage(
            "2026-11-06", baseline_path=events, schedule_path=policy,
            stage_dir=stage, now="2026-11-06T14:00:00Z",
            vintage_data=vintages())


def test_activation_requires_original_baseline_hash(tmp_path):
    events, policy, stage, sidecar = files(tmp_path)
    refresh.stage(
        "2026-11-06", baseline_path=events, schedule_path=policy,
        stage_dir=stage, now="2026-11-06T14:00:00Z",
        vintage_data=vintages())
    with events.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        refresh.activate(
            "2026-11-06", baseline_path=events,
            schedule_path=policy, stage_dir=stage, sidecar_path=sidecar)
    assert not sidecar.exists()


def test_activation_enables_gate_and_detects_tampering(tmp_path):
    events, policy, stage, sidecar = files(tmp_path)
    refresh.stage(
        "2026-11-06", baseline_path=events, schedule_path=policy,
        stage_dir=stage, now="2026-11-06T14:00:00Z",
        vintage_data=vintages())
    result = refresh.activate(
        "2026-11-06", baseline_path=events, schedule_path=policy,
        stage_dir=stage, sidecar_path=sidecar)
    assert result["release_date"] == "2026-11-06"
    assert require_payems_ready(
        "2026-11-06", events_path=events, schedule_path=policy,
        sidecar_path=sidecar) == "2026-11-06"
    with events.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="integrity mismatch"):
        require_payems_ready(
            "2026-11-06", events_path=events,
            schedule_path=policy, sidecar_path=sidecar)


def test_wrong_reference_month_rejected(tmp_path):
    events, policy, stage, sidecar = files(tmp_path)
    data = vintages()
    data.loc[data["reference_date"] == "2026-10-01", "reference_date"] = "2026-07-01"
    with pytest.raises(ValueError, match="expected initial reference period"):
        refresh.stage(
            "2026-11-06", baseline_path=events, schedule_path=policy,
            stage_dir=stage, now="2026-11-06T14:00:00Z",
            vintage_data=data)
