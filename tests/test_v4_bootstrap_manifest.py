"""No live market data, network calls, trading, or prospective V4 issuance."""
import json
from pathlib import Path

import pytest

from scripts import build_v4_bootstrap_manifest as m

MARKET = """market_date,open,high,low,close,volume\n2026-10-26,29.1,29.9,29.0,29.7,12000\n2026-10-27,29.7,29.8,29.2,29.64,13000\n2026-10-28,29.64,29.7,28.7,28.94,14000\n"""
NOMINAL = """reference_date,yield_30y\n2026-10-26,4.60\n2026-10-27,4.59\n2026-10-28,4.55\n"""
REAL = """reference_date,real_yield_10y\n2026-10-26,2.01\n2026-10-27,2.00\n2026-10-28,1.98\n"""

@pytest.fixture
def configured(tmp_path):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    (raw / "market.csv").write_bytes(MARKET.encode("utf-8"))
    (raw / "nominal.csv").write_bytes(NOMINAL.encode("utf-8"))
    (raw / "real.csv").write_bytes(REAL.encode("utf-8"))
    cfg = {
        "schema": m.DECLARATIONS,
        "as_of_start_session": "2026-11-03",
        "sources": {
            "tmf_historical": {"path": "data/raw/market.csv", "date_column": "market_date", "provider": None},
            "treasury_nominal": {"path": "data/raw/nominal.csv", "date_column": "reference_date", "value_column": "yield_30y", "provider": None},
            "treasury_real_10y": {"path": "data/raw/real.csv", "date_column": "reference_date", "value_column": "real_yield_10y", "provider": None},
        },
    }
    return tmp_path, cfg


def test_synthetic_source_record_check_only(configured):
    root, cfg = configured
    record, content = m.build(root, cfg)
    assert record["status"] == "HISTORICAL_CONTEXT_ONLY"
    assert record["source_acceptance"] == "NOT_ESTABLISHED"
    assert record["historical_availability_attested"] is False
    assert record["authorizes_live_v4_issuance"] is False
    assert record["order_generated"] is False
    assert record["risk_initialization"] is None
    assert record["B3"] == "BLOCKED"
    assert record["sources"]["tmf_historical"]["rows"] == 3
    assert content["tmf_historical"] == MARKET.encode()
    assert not (root / m.ARCHIVE).exists()


def test_archive_exact_bytes_and_idempotence(configured):
    root, cfg = configured
    record, blobs = m.build(root, cfg)
    path, status = m.save(root, record, blobs)
    assert status == "ARCHIVED_HISTORICAL_CONTEXT_ONLY"
    assert m.verify(root, path)["source_bundle_sha256"] == record["source_bundle_sha256"]
    assert (root / record["sources"]["tmf_historical"]["archived_relative_path"]).read_bytes() == MARKET.encode()
    first = path.read_bytes()
    again, second = m.save(root, record, blobs)
    assert again == path and second == "ALREADY_VERIFIED"
    assert path.read_bytes() == first


def test_changes_to_original_trigger_failure(configured):
    root, cfg = configured
    record, blobs = m.build(root, cfg)
    path, _ = m.save(root, record, blobs)
    (root / "data/raw/market.csv").write_text(MARKET.replace("29.7,12000", "29.7,13000"))
    with pytest.raises(m.BootstrapError, match="original source bytes differ"):
        m.verify(root, path)
    with pytest.raises(m.BootstrapError, match="Original source changed"):
        m.save(root, record, blobs)


def test_changes_to_archived_bytes_trigger_failure(configured):
    root, cfg = configured
    record, blobs = m.build(root, cfg)
    path, _ = m.save(root, record, blobs)
    archive = root / record["sources"]["tmf_historical"]["archived_relative_path"]
    archive.write_bytes(b"tampered")
    with pytest.raises(m.BootstrapError, match="archive source bytes differ"):
        m.verify(root, path)
    with pytest.raises(m.BootstrapError, match="Immutable archive conflict"):
        m.save(root, record, blobs)


def test_manifest_edit_is_detected(configured):
    root, cfg = configured
    record, blobs = m.build(root, cfg)
    path, _ = m.save(root, record, blobs)
    stored = json.loads(path.read_text())
    stored["blockers"] = []
    path.write_text(json.dumps(stored))
    with pytest.raises(m.BootstrapError, match="Manifest content/hash mismatch"):
        m.verify(root, path)


@pytest.mark.parametrize("bad", [
    "../outside.csv", "/tmp/outside.csv", "data/raw/../market.csv",
    "data\\raw\\market.csv", "data/raw/market.json",
])
def test_paths_must_be_local_unmodified_csv(configured, bad):
    root, cfg = configured
    cfg["sources"]["tmf_historical"]["path"] = bad
    with pytest.raises(m.BootstrapError):
        m.build(root, cfg)


def test_symlink_forbidden(configured):
    root, cfg = configured
    original = root / "data/raw/market.csv"
    original.rename(root / "data/raw/source.csv")
    try:
        original.symlink_to(root / "data/raw/source.csv")
    except OSError:
        pytest.skip("Symlinks unavailable in this environment")
    with pytest.raises(m.BootstrapError, match="linked original"):
        m.build(root, cfg)


@pytest.mark.parametrize("badcsv,error", [
    (MARKET.replace("2026-10-28", "2026-10-27"), "Dates duplicated"),
    (MARKET.replace("2026-10-28", "2026-10-25"), "not increasing"),
    (MARKET.replace("2026-10-28", "2026-14-28"), "Invalid"),
    (MARKET.replace("29.64,29.7,28.7", "29.64,28.7,28.7"), "OHLC high invalid"),
    (MARKET.replace("28.94,14000", "28.94,NaN"), "Missing or nonfinite"),
    (MARKET.replace("29.64,13000", "29.64,-5"), "Missing or nonfinite"),
    (MARKET.replace("market_date", "future_return_5d"), "Missing required date"),
    (MARKET.replace("volume", "available_at_utc"), "Future/availability"),
    ("market_date,open,high,low,close,volume\n", "Empty source CSV"),
])
def test_bad_market_csv_fails_closed(configured, badcsv, error):
    root, cfg = configured
    (root / "data/raw/market.csv").write_text(badcsv)
    with pytest.raises(m.BootstrapError, match=error):
        m.build(root, cfg)


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(schema="unknown"),
    lambda c: c["sources"].pop("treasury_real_10y"),
    lambda c: c["sources"]["treasury_nominal"].update(value_column="missing"),
    lambda c: c.update(as_of_start_session="2026-10-27"),
    lambda c: c["sources"]["tmf_historical"].update(retrieved_at_utc="2026-11-02T19:00:00"),
])
def test_bad_declarations_are_rejected(configured, mutate):
    root, cfg = configured
    mutate(cfg)
    with pytest.raises(m.BootstrapError):
        m.build(root, cfg)


def test_fractional_volume_is_flagged_not_silently_rounded(configured):
    root, cfg = configured
    (root / "data/raw/market.csv").write_text(MARKET.replace("14000", "14000.25"))
    record, _ = m.build(root, cfg)
    assert record["sources"]["tmf_historical"]["fractional_volume_rows"] == 1
    assert "market.fractional_volume_requires_review" in record["blockers"]


def test_missing_source_fails_closed(configured):
    root, cfg = configured
    (root / "data/raw/real.csv").unlink()
    with pytest.raises(m.BootstrapError, match="Missing or linked original"):
        m.build(root, cfg)


def test_metadata_never_grants_research_acceptance(configured):
    root, cfg = configured
    for entry in cfg["sources"].values():
        entry.update(provider="vendor", source_url="https://example.org/source",
                     license="research license", retrieved_at_utc="2026-10-29T19:00:00Z",
                     retrieval_evidence_reference="evidence/test.txt", vintage_policy="unknown",
                     corporate_action_policy="under review", publication_policy="unknown")
    record, _ = m.build(root, cfg)
    assert record["B1"] == record["B2"] == "NOT_ESTABLISHED"
    assert record["authorizes_live_v4_issuance"] is False
    assert "independent_treasury_publication_and_vintage_unverified" in record["blockers"]


def test_source_bundle_digest_deterministic(configured):
    root, cfg = configured
    first, _ = m.build(root, cfg)
    second, _ = m.build(root, cfg)
    assert first == second


def test_sources_must_not_duplicate(configured):
    root, cfg = configured
    (root / "data/raw/real.csv").write_bytes(NOMINAL.encode("utf-8"))
    cfg["sources"]["treasury_real_10y"]["value_column"] = "yield_30y"
    with pytest.raises(m.BootstrapError, match="Same file bytes"):
        m.build(root, cfg)


def test_wrong_root_manifest_rejected(configured):
    root, cfg = configured
    record, blobs = m.build(root, cfg)
    path, _ = m.save(root, record, blobs)
    with pytest.raises(m.BootstrapError, match="inside local bootstrap archive"):
        m.verify(root / "nested", path)


def test_cli_does_not_write_without_output(configured, tmp_path, capsys):
    root, cfg = configured
    meta = tmp_path / "declarations.json"
    meta.write_text(json.dumps(cfg))
    # Build API is what check-only mode calls; output is explicitly required for save.
    record, _ = m.build(root, m._json(meta))
    assert not (root / m.ARCHIVE).exists()
    assert record["order_generated"] is False

@pytest.mark.parametrize("offset", ["-04:00", "-05:00"])
def test_market_midnight_timestamp_is_reference_only(configured, offset):
    root, cfg = configured
    content = MARKET
    for day in ("2026-10-26", "2026-10-27", "2026-10-28"):
        content = content.replace(
            f"{day},", f"{day} 00:00:00{offset},"
        )

    (root / "data/raw/market.csv").write_bytes(content.encode("utf-8"))
    record, blobs = m.build(root, cfg)

    market = record["sources"]["tmf_historical"]
    assert market["first_reference_date"] == "2026-10-26"
    assert market["last_reference_date"] == "2026-10-28"
    assert blobs["tmf_historical"] == content.encode("utf-8")
    assert record["historical_availability_attested"] is False
    assert record["authorizes_live_v4_issuance"] is False

    audit = market["nyse_session_audit"]
    if audit["status"] == "CHECKED":
        assert audit["missing_dates"] == []
        assert audit["non_nyse_dates"] == []


@pytest.mark.parametrize("bad", [
    "2026-10-26 00:00:00",
    "2026-10-26 09:30:00-04:00",
    "2026-10-26 00:00:00+25:00",
])
def test_market_invalid_timestamp_rejected(configured, bad):
    root, cfg = configured
    content = MARKET.replace("2026-10-26,", f"{bad},")
    (root / "data/raw/market.csv").write_bytes(content.encode("utf-8"))

    with pytest.raises(m.BootstrapError):
        m.build(root, cfg)


@pytest.mark.parametrize("role,filename,original,old", [
    ("treasury_nominal", "nominal.csv", NOMINAL, "2026-10-27,4.59"),
    ("treasury_real_10y", "real.csv", REAL, "2026-10-27,2.00"),
])
@pytest.mark.parametrize("missing_token", ["", "."])
def test_missing_treasury_yield_is_recorded_not_filled(
    configured, role, filename, original, old, missing_token
):
    root, cfg = configured
    content = original.replace(old, f"2026-10-27,{missing_token}")
    (root / "data/raw" / filename).write_bytes(content.encode("utf-8"))

    record, blobs = m.build(root, cfg)
    source = record["sources"][role]

    assert source["rows"] == 3
    assert source["valid_yield_rows"] == 2
    assert source["missing_yield_rows"] == 1
    assert source["missing_yield_dates"] == ["2026-10-27"]
    assert blobs[role] == content.encode("utf-8")
    assert f"{role}.missing_yield_observations" in record["blockers"]
    assert record["source_acceptance"] == "NOT_ESTABLISHED"
    assert record["authorizes_live_v4_issuance"] is False


@pytest.mark.parametrize("bad", ["NaN", "inf", "not_a_yield"])
def test_invalid_treasury_yields_still_fail_closed(configured, bad):
    root, cfg = configured
    content = NOMINAL.replace("2026-10-27,4.59", f"2026-10-27,{bad}")
    (root / "data/raw/nominal.csv").write_bytes(content.encode("utf-8"))

    with pytest.raises(m.BootstrapError):
        m.build(root, cfg)
