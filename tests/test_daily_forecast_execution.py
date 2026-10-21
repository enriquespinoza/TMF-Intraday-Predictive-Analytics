import json
from datetime import datetime, timezone

import pytest

from scripts import run_daily_forecast as engine


SESSION = "2026-10-20"


@pytest.fixture
def execution_case(tmp_path, monkeypatch):
    root = tmp_path
    output = root / "data/forward_test"
    output.mkdir(parents=True)

    schwab = root / "data/raw/schwab"
    schwab.mkdir(parents=True)

    scripts = root / "scripts"
    scripts.mkdir()

    predictor = scripts / "predict_v3_schwab.py"
    predictor.write_text("frozen predictor test")

    daily = schwab / "TMF_1d_test.csv"
    intraday = schwab / "TMF_5m_test.csv"

    daily.write_text("frozen daily snapshot")
    intraday.write_text("frozen intraday snapshot")

    monkeypatch.setattr(engine, "ROOT", root)
    monkeypatch.setattr(engine, "OUTPUT_DIR", output)
    monkeypatch.setattr(engine, "RUN_DIR", output / "runs")

    forecast_path = output / (
        f"TMF_V3_SCHWAB_{SESSION}_"
        f"{engine.FINAL_V3_NAME}_{engine.HORIZON}d.json"
    )

    return {
        "daily": daily,
        "intraday": intraday,
        "forecast": forecast_path,
        "output": output,
    }


def install_fake_predictor(monkeypatch, case, changes=None,
                           modify_daily=False, write_output=True):
    changes = changes or {}

    def fake_run(*args, **kwargs):
        if modify_daily:
            case["daily"].write_text("changed during execution")

        if not write_output:
            return None

        result = {
            "model": engine.FINAL_V3_NAME,
            "issued_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "forecast_origin": SESSION,
            "forecast_target_date":
                engine.forecast_target(SESSION),
            "horizon_trading_days": engine.HORIZON,
            "source_snapshot": case["daily"].name,
            "source_sha256": engine.digest(
                case["daily"]
            ) if not modify_daily else changes.get(
                "original_hash", ""
            ),
            "order_generated": False,
        }

        result.update(changes)

        case["forecast"].write_text(
            json.dumps(result), encoding="utf-8"
        )

        return None

    monkeypatch.setattr(engine.subprocess, "run", fake_run)


def test_valid_prediction_creates_manifest(
    execution_case, monkeypatch
):
    install_fake_predictor(monkeypatch, execution_case)

    engine.execute_forecast(
        SESSION, execution_case["daily"]
    )

    manifests = list(
        (execution_case["output"] / "runs").glob("*.json")
    )

    assert len(manifests) == 1

    record = json.loads(manifests[0].read_text())

    assert record["status"] == "GENERATED"
    assert record["order_generated"] is False
    assert record["forecast_sha256"] == engine.digest(
        execution_case["forecast"]
    )


def test_missing_predictor_output_rejected(
    execution_case, monkeypatch
):
    install_fake_predictor(
        monkeypatch, execution_case,
        write_output=False
    )

    with pytest.raises(
        RuntimeError, match="expected forecast"
    ):
        engine.execute_forecast(
            SESSION, execution_case["daily"]
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "UNAPPROVED_MODEL"},
        {"forecast_origin": "2026-10-19"},
        {"forecast_target_date": "2026-10-28"},
        {"source_sha256": "invalid_hash"},
    ],
)
def test_invalid_forecast_metadata_rejected(
    execution_case, monkeypatch, changes
):
    install_fake_predictor(
        monkeypatch, execution_case, changes
    )

    with pytest.raises(
        ValueError, match="metadata validation failed"
    ):
        engine.execute_forecast(
            SESSION, execution_case["daily"]
        )

    assert not (
        execution_case["output"] / "runs"
    ).exists()


def test_snapshot_modified_during_prediction_rejected(
    execution_case, monkeypatch
):
    original_hash = engine.digest(
        execution_case["daily"]
    )

    install_fake_predictor(
        monkeypatch,
        execution_case,
        changes={"original_hash": original_hash},
        modify_daily=True,
    )

    with pytest.raises(
        ValueError, match="input changed"
    ):
        engine.execute_forecast(
            SESSION, execution_case["daily"]
        )
