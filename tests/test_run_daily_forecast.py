import json
import sys

import pandas as pd
import pytest

from scripts import run_daily_forecast as engine


SESSION = "2026-10-20"
NOW = pd.Timestamp("2026-10-21T02:35:56Z")


@pytest.fixture
def isolated_engine(tmp_path, monkeypatch):
    output = tmp_path / "forward_test"
    output.mkdir()

    monkeypatch.setattr(engine, "OUTPUT_DIR", output)
    monkeypatch.setattr(
        engine, "session_ready", lambda: (SESSION, NOW)
    )

    return output


def fail_if_called(*args, **kwargs):
    raise AssertionError("Unauthorized operation attempted")


def test_check_only_does_not_generate_forecast(
    isolated_engine, monkeypatch
):
    snapshot = isolated_engine / "snapshot.csv"
    snapshot.write_text("test")

    monkeypatch.setattr(
        engine, "validate_snapshot_pair",
        lambda path: {
            "status": "POST_CLOSE_SNAPSHOT_VALIDATED",
            "market_date": SESSION,
        },
    )
    monkeypatch.setattr(
        engine, "execute_forecast", fail_if_called
    )
    monkeypatch.setattr(
        engine, "refresh_snapshot", fail_if_called
    )
    monkeypatch.setattr(
        sys, "argv",
        ["run_daily_forecast", "--daily", str(snapshot)],
    )

    engine.main()

    assert not list(
        isolated_engine.glob("TMF_V3_SCHWAB_*.json")
    )


def test_forecast_without_manifest_blocks_refresh(
    isolated_engine, monkeypatch
):
    existing = isolated_engine / (
        f"TMF_V3_SCHWAB_{SESSION}_existing.json"
    )
    existing.write_text(json.dumps({
        "model": engine.FINAL_V3_NAME,
        "forecast_origin": SESSION,
        "horizon_trading_days": engine.HORIZON,
    }))

    monkeypatch.setattr(
        engine, "refresh_snapshot", fail_if_called
    )
    monkeypatch.setattr(
        engine, "execute_forecast", fail_if_called
    )
    monkeypatch.setattr(
        sys, "argv",
        ["run_daily_forecast", "--run", "--refresh"],
    )

    with pytest.raises(RuntimeError, match="INCOMPLETE_RUN"):
        engine.main()

    assert existing.exists()
    assert not list(isolated_engine.glob(".lock_*"))


def test_concurrent_run_blocked(
    isolated_engine, monkeypatch
):
    lock = isolated_engine / (
        f".lock_{engine.FINAL_V3_NAME}_"
        f"{SESSION}_{engine.HORIZON}d"
    )
    lock.mkdir()

    monkeypatch.setattr(
        engine, "refresh_snapshot", fail_if_called
    )
    monkeypatch.setattr(
        sys, "argv",
        ["run_daily_forecast", "--run", "--refresh"],
    )

    with pytest.raises(RuntimeError, match="lock exists"):
        engine.main()

    assert lock.exists()


def test_invalid_snapshot_blocks_forecast(
    isolated_engine, monkeypatch
):
    snapshot = isolated_engine / "snapshot.csv"
    snapshot.write_text("invalid")

    monkeypatch.setattr(
        engine, "latest_snapshot", lambda: snapshot
    )

    def reject_snapshot(*args, **kwargs):
        raise ValueError("Invalid Schwab snapshot")

    monkeypatch.setattr(
        engine, "validate_snapshot_pair", reject_snapshot
    )
    monkeypatch.setattr(
        engine, "execute_forecast", fail_if_called
    )
    monkeypatch.setattr(
        sys, "argv",
        ["run_daily_forecast", "--run"],
    )

    with pytest.raises(ValueError, match="Invalid Schwab"):
        engine.main()

    assert not list(isolated_engine.glob(".lock_*"))


def test_duplicate_forecast_records_rejected(
    isolated_engine
):
    for name in ["first", "second"]:
        path = isolated_engine / (
            f"TMF_V3_SCHWAB_{SESSION}_{name}.json"
        )
        path.write_text(json.dumps({
            "model": engine.FINAL_V3_NAME,
            "forecast_origin": SESSION,
            "horizon_trading_days": engine.HORIZON,
        }))

    with pytest.raises(ValueError, match="Duplicate"):
        engine.matching_forecasts(SESSION)


def test_refresh_without_run_rejected(
    isolated_engine, monkeypatch
):
    monkeypatch.setattr(
        sys, "argv",
        ["run_daily_forecast", "--refresh"],
    )

    with pytest.raises(SystemExit) as error:
        engine.main()

    assert error.value.code == 2


def test_completed_run_skips_without_refresh(
    isolated_engine, monkeypatch, capsys
):
    completed = isolated_engine / "completed_forecast.json"
    completed.write_text("{}")

    monkeypatch.setattr(
        engine, "audit_existing_run",
        lambda session: completed,
    )
    monkeypatch.setattr(engine, "refresh_snapshot", fail_if_called)
    monkeypatch.setattr(engine, "execute_forecast", fail_if_called)
    monkeypatch.setattr(
        sys, "argv",
        ["run_daily_forecast", "--run", "--refresh"],
    )

    engine.main()

    assert "SKIPPED" in capsys.readouterr().out
    assert completed.exists()
    assert not list(isolated_engine.glob(".lock_*"))
