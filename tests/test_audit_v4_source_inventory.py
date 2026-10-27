"""Contract tests for read-only V4 inventory."""
import csv

from scripts.audit_v4_source_inventory import inspect_csv, inventory


def test_inventory_missing_sources_never_authorizes(tmp_path):
    report = inventory(tmp_path)
    assert report["readiness"] == "REQUIRES_INDEPENDENT_POINT_IN_TIME_VERIFICATION"
    assert report["v4_issued_record_count"] == 0
    assert report["files"]["tmf_historical"]["status"] == "MISSING"
    assert report["research_only"] is True


def test_reference_only_does_not_claim_prospective(tmp_path):
    path = tmp_path / "history.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["market_date", "close"])
        writer.writeheader()
        writer.writerow({"market_date": "2026-10-26", "close": "29.70"})
    report = inspect_csv(path)
    assert report["row_count"] == 1
    assert report["last_reference_date"] == "2026-10-26"
    assert report["status"] == "HISTORICAL_REFERENCE_ONLY"
    assert report["has_strict_availability_column"] is False


def test_timestamp_column_is_not_verification(tmp_path):
    path = tmp_path / "history.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["reference_date", "available_at_utc", "yield_30y"],
        )
        writer.writeheader()
        writer.writerow({
            "reference_date": "2026-10-23",
            "available_at_utc": "2026-10-23T21:00:00Z",
            "yield_30y": "4.40",
        })
    report = inspect_csv(path)
    assert report["status"] == "TIMESTAMP_FIELDS_PRESENT_UNVERIFIED"
    assert report["strict_availability_rows"] == 1


def test_partial_availability_not_ready(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text(
        "market_date,available_at_utc,close\n"
        "2026-10-23,2026-10-23T21:00:00Z,29.36\n"
        "2026-10-26,,29.70\n",
        encoding="utf-8",
    )
    report = inspect_csv(path)
    assert report["status"] == "INCOMPLETE_AVAILABILITY_FIELDS"
    assert report["row_count"] == 2
