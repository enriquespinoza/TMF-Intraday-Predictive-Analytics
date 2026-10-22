"""TimesFM manifest and crash-recovery tests (no model download)."""

import json

import pytest

from scripts import timesfm25_integrity as integrity
from src.models.timesfm25_forecaster import MODEL_NAME


ORIGIN = "2026-10-21"


@pytest.fixture
def record(tmp_path, monkeypatch):
    output = tmp_path / "timesfm25"
    output.mkdir()
    forecast = integrity.forecast_path(output, ORIGIN)
    forecast.write_text('{"immutable":true}', encoding="utf-8")
    source = tmp_path / "schwab"
    source.mkdir()
    daily = source / "TMF_1d_test.csv"
    daily.write_text("a")
    history = tmp_path / "TMF_1d.csv"
    history.write_text("history")

    payload = {
        "forecast_origin": ORIGIN, "forecast_target_date": "2026-10-28",
        "checkpoint_revision": "a" * 40, "source_snapshot": daily.name,
        "source_sha256": integrity.sha256(daily),
        "intraday_sha256": "b" * 64,
        "historical_source_sha256": integrity.sha256(history),
        "issued_at_utc": "2026-10-22T13:13:06+00:00",
    }
    monkeypatch.setattr(
        integrity, "_validate_payload",
        lambda path, source_dir, history_path: payload,
    )
    return output, forecast, source, history


def test_exclusive_manifest_and_existing_audit(record):
    output, forecast, source, history = record
    manifest = integrity.create_manifest(
        forecast, source_dir=source, history_path=history,
        state="GENERATED",
    )
    assert json.loads(manifest.read_text())["status"] == "GENERATED"
    assert integrity.audit_existing_run(
        ORIGIN, output_dir=output, source_dir=source,
        history_path=history,
    ) == forecast
    with pytest.raises(FileExistsError):
        integrity.create_manifest(
            forecast, source_dir=source, history_path=history,
            state="GENERATED",
        )


def test_incomplete_run_blocks_recovery(record):
    output, _, source, history = record
    with pytest.raises(RuntimeError, match="INCOMPLETE_RUN"):
        integrity.audit_existing_run(
            ORIGIN, output_dir=output, source_dir=source,
            history_path=history,
        )


def test_modified_forecast_rejected(record):
    output, forecast, source, history = record
    integrity.create_manifest(
        forecast, source_dir=source, history_path=history,
        state="GENERATED",
    )
    forecast.write_text("mutated", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forecast_sha256"):
        integrity.audit_existing_run(
            ORIGIN, output_dir=output, source_dir=source,
            history_path=history,
        )


def test_orphan_manifest_rejected(record):
    output, forecast, source, history = record
    integrity.create_manifest(
        forecast, source_dir=source, history_path=history,
        state="GENERATED",
    )
    forecast.unlink()
    with pytest.raises(RuntimeError, match="ORPHANED_MANIFEST"):
        integrity.audit_existing_run(
            ORIGIN, output_dir=output, source_dir=source,
            history_path=history,
        )


def test_duplicate_lock_rejected(record):
    output, _, _, _ = record
    with integrity.session_lock(output, ORIGIN):
        with pytest.raises(RuntimeError, match="lock exists"):
            with integrity.session_lock(output, ORIGIN):
                pass
    assert not list(output.glob(".lock*"))
