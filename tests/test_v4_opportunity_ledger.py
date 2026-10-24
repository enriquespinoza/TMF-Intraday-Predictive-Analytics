"""V4 independent prospective ledger tests; all data synthetic, no network."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import v4_opportunity_ledger as ledger


@pytest.fixture
def case(tmp_path, monkeypatch):
    source_dir = tmp_path / "schwab"
    component_dir = tmp_path / "components"
    output = tmp_path / "ledger"
    for directory in (source_dir, component_dir):
        directory.mkdir()
    policy = tmp_path / "opportunity_engine_v1_policy.json"
    policy.write_text(json.dumps({
        "technical": {"strong_setup_threshold": 43.77},
        "treasury": {"extreme_confirmation_threshold": 79.72},
    }))
    monkeypatch.setattr(ledger, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(ledger, "COMPONENT_DIR", component_dir)
    monkeypatch.setattr(ledger, "LEDGER_DIR", output)
    monkeypatch.setattr(ledger, "EVAL_DIR", output / "evaluations")
    monkeypatch.setattr(ledger, "POLICY_PATH", policy)
    monkeypatch.setattr(ledger, "horizon", lambda origin: [
        "2026-10-26", "2026-10-27", "2026-10-28",
        "2026-10-29", "2026-10-30",
    ])
    monkeypatch.setattr(
        ledger, "require_issuance_window",
        lambda origin, now=None: SimpleNamespace(
            finalized_at_utc=pd.Timestamp("2026-10-23T20:30:00Z")
        )
    )
    def validate(path, now=None, expected_session=None):
        session = expected_session or "2026-10-23"
        if session == "2026-10-30":
            return {
                "status": "POST_CLOSE_SNAPSHOT_VALIDATED",
                "market_date": "2026-10-30", "daily_close": 31.2,
            }
        return {
            "status": "POST_CLOSE_SNAPSHOT_VALIDATED",
            "market_date": "2026-10-23", "daily_close": 29.36,
        }
    monkeypatch.setattr(ledger, "validate_snapshot_pair", validate)
    daily = source_dir / "TMF_1d_20261023T214235Z.csv"
    dates = [
        "2026-10-22", "2026-10-23", "2026-10-26", "2026-10-27",
        "2026-10-28", "2026-10-29", "2026-10-30",
    ]
    closes = [29.78, 29.36, 29.7, 29.6, 30.1, 30.5, 31.2]
    pd.DataFrame({"market_date": dates[:2], "close": closes[:2]}).to_csv(
        daily, index=False,
    )
    intraday = source_dir / "TMF_5m_20261023T214235Z.csv"
    intraday.write_text("test bars")
    source_sha = ledger.digest(daily)
    paths = {}
    evidence = {
        "risk_governor_v3": {
            "new_entries_allowed": True, "exposure_state": "GREEN",
        },
        "technical_opportunity_v1": {
            "technical_opportunity_score": 70.0,
        },
        "treasury_confirmation_v1": {
            "treasury_confirmation_score": 30.0,
        },
    }
    for role, fields in evidence.items():
        path = component_dir / f"{role}.json"
        path.write_text(json.dumps({
            "schema_version": "v4_live_component_v1",
            "role": role,
            "market_date": "2026-10-23",
            "source_snapshot": daily.name,
            "source_sha256": source_sha,
            "available_at_utc": "2026-10-23T21:43:00Z",
            **fields,
        }))
        paths[role] = path
    outcome = source_dir / "TMF_1d_20261030T214235Z.csv"
    pd.DataFrame({"market_date": dates, "close": closes}).to_csv(
        outcome, index=False,
    )
    (source_dir / "TMF_5m_20261030T214235Z.csv").write_text("target bars")
    return {
        "daily": daily,
        "risk": paths["risk_governor_v3"],
        "technical": paths["technical_opportunity_v1"],
        "treasury": paths["treasury_confirmation_v1"],
        "outcome": outcome,
        "root": output,
        "issue_time": pd.Timestamp("2026-10-23T21:45:00Z"),
    }


def mutate(path, **kwargs):
    data = json.loads(path.read_text())
    data.update(kwargs)
    path.write_text(json.dumps(data))


def issue(case, **kwargs):
    return ledger.issue(
        "2026-10-23", case["daily"], case["risk"],
        case["technical"], case["treasury"],
        now=case["issue_time"], **kwargs,
    )


def test_check_produces_no_artifacts(case):
    dest, result = issue(case, write=False)
    assert dest is None
    assert result["decision"]["opportunity_state"] == "ENTRY_CANDIDATE"
    assert result["order_generated"] is False
    assert not case["root"].exists()


def test_write_once_and_audit(case, monkeypatch):
    issue(case, write=True)
    monkeypatch.setattr(ledger, "now_utc", lambda now=None:
                        pd.Timestamp("2026-10-23T22:00:00Z") if now is None
                        else ledger.utc(now))
    assert ledger.audit("2026-10-23")["decision"]["entry_candidate"] is True
    with pytest.raises(FileExistsError):
        issue(case, write=True)


def test_blocked_risk_veto(case):
    mutate(case["risk"], new_entries_allowed=False, exposure_state="RED")
    _, result = issue(case)
    assert result["decision"]["opportunity_state"] == "NO_TRADE"
    assert result["decision"]["risk_entry_blocked"] is True


def test_unavailable_evidence_is_rejected(case):
    mutate(case["technical"], available_at_utc="2026-10-23T21:46:00Z")
    with pytest.raises(ValueError, match="unavailable at issuance"):
        issue(case)


def test_stale_previous_date_rejected(case):
    mutate(case["technical"], market_date="2026-10-22")
    with pytest.raises(ValueError, match="date/role mismatch"):
        issue(case)


def test_future_labels_not_allowed_in_evidence(case):
    mutate(case["technical"], forward_return_5=0.5)
    with pytest.raises(ValueError, match="Future outcome"):
        issue(case)


def test_non_green_entry_permission_rejected(case):
    mutate(case["risk"], exposure_state="YELLOW")
    with pytest.raises(ValueError, match="Non-green"):
        issue(case)


def test_source_hash_mismatch_rejected(case):
    mutate(case["treasury"], source_sha256="f" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        issue(case)


def test_score_requires_prospective_issuance(case):
    with pytest.raises(FileNotFoundError, match="No prospective"):
        ledger.score("2026-10-23", case["outcome"])


def test_scoring_and_state_report(case, monkeypatch):
    issue(case, write=True)
    monkeypatch.setattr(ledger, "now_utc", lambda now=None:
                        pd.Timestamp("2026-10-30T22:00:00Z") if now is None
                        else ledger.utc(now))
    dest, score = ledger.score("2026-10-23", case["outcome"], write=True)
    assert dest.exists()
    assert score["state"] == "ENTRY_CANDIDATE"
    assert score["five_session_return_pct"] == pytest.approx(
        (31.2 / 29.36 - 1) * 100,
    )
    assert ledger.report()["matured_count"] == 1
    with pytest.raises(FileExistsError):
        ledger.score("2026-10-23", case["outcome"], write=True)


def test_outcome_earlier_than_target_rejected(case, monkeypatch):
    issue(case, write=True)
    monkeypatch.setattr(ledger, "now_utc", lambda now=None:
                        pd.Timestamp("2026-10-30T22:00:00Z") if now is None
                        else ledger.utc(now))
    with pytest.raises(ValueError, match="target"):
        ledger.score("2026-10-23", case["daily"])


def test_tampered_issued_record_rejected(case, monkeypatch):
    path, _ = issue(case, write=True)
    mutate(path, decision={"opportunity_state": "WATCH"})
    monkeypatch.setattr(ledger, "now_utc", lambda now=None:
                        pd.Timestamp("2026-10-30T22:00:00Z") if now is None
                        else ledger.utc(now))
    with pytest.raises(ValueError, match="manifest mismatch"):
        ledger.audit("2026-10-23")


def test_missing_component_file_fails_audit(case, monkeypatch):
    issue(case, write=True)
    case["treasury"].unlink()
    monkeypatch.setattr(ledger, "now_utc", lambda now=None:
                        pd.Timestamp("2026-10-30T22:00:00Z") if now is None
                        else ledger.utc(now))
    with pytest.raises(FileNotFoundError):
        ledger.audit("2026-10-23")
