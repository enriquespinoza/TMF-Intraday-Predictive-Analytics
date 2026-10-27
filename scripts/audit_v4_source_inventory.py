"""Read-only schema, coverage and provenance inventory for V4 source readiness.

This command NEVER constructs available_at_utc, writes datasets, or issues
forecasts. A filename capture timestamp does not attest every historical row.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "tmf_historical": "data/raw/TMF_1d.csv",
    "treasury_nominal": "data/raw/treasury_yields_daily.csv",
    "treasury_real_10y": "data/raw/macro/real_yield_10y.csv",
    "v4_risk_features": "data/features/TMF_1d_risk_governor_features.csv",
    "v4_market_attested": "data/raw/v4_inputs/market_history.csv",
    "v4_treasury_attested": "data/raw/v4_inputs/treasury_history.csv",
}
DATE_FIELDS = (
    "market_date", "reference_date", "timestamp", "date", "session_date",
    "trading_date",
)
TIME_FIELDS = (
    "available_at_utc", "release_timestamp", "realtime_start",
    "vintage_date", "publication_time", "released_at",
)
STRICT_TIME_FIELD = "available_at_utc"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for chunk in iter(lambda: inp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_csv(path):
    path = Path(path)
    base = {
        "file": str(path),
        "exists": path.is_file(),
        "row_count": 0,
        "first_reference_date": None,
        "last_reference_date": None,
        "date_field": None,
        "available_time_fields": [],
        "has_strict_availability_column": False,
        "strict_availability_rows": 0,
        "status": "MISSING",
    }
    if not path.is_file():
        return base
    base["sha256"] = sha256(path)
    try:
        with path.open(newline="", encoding="utf-8-sig") as inp:
            reader = csv.DictReader(inp)
            if not reader.fieldnames:
                raise ValueError("Missing CSV header")
            fields = tuple(reader.fieldnames)
            base["columns"] = fields
            dcol = next((f for f in DATE_FIELDS if f in fields), None)
            base["date_field"] = dcol
            base["available_time_fields"] = [
                f for f in TIME_FIELDS if f in fields
            ]
            base["has_strict_availability_column"] = STRICT_TIME_FIELD in fields
            first = last = None
            for row in reader:
                base["row_count"] += 1
                if dcol and row.get(dcol):
                    value = row[dcol].strip()[:10]
                    try:
                        parsed = date.fromisoformat(value)
                    except ValueError:
                        raise ValueError(
                            f"Invalid {dcol} date on row {base['row_count'] + 1}"
                        ) from None
                    if first is None or parsed < first:
                        first = parsed
                    if last is None or parsed > last:
                        last = parsed
                if row.get(STRICT_TIME_FIELD) and row[STRICT_TIME_FIELD].strip():
                    base["strict_availability_rows"] += 1
            base["first_reference_date"] = first.isoformat() if first else None
            base["last_reference_date"] = last.isoformat() if last else None
            if not base["row_count"]:
                base["status"] = "EMPTY"
            elif STRICT_TIME_FIELD not in fields:
                base["status"] = "HISTORICAL_REFERENCE_ONLY"
            elif base["strict_availability_rows"] < base["row_count"]:
                base["status"] = "INCOMPLETE_AVAILABILITY_FIELDS"
            else:
                base["status"] = "TIMESTAMP_FIELDS_PRESENT_UNVERIFIED"
    except (OSError, ValueError, csv.Error) as exc:
        base["status"] = "INVALID_CSV"
        base["reason"] = str(exc)
    return base


def inventory(root=ROOT):
    root = Path(root)
    result = {
        "audit": "V4_SOURCE_INVENTORY_READ_ONLY",
        "research_only": True,
        "observations_do_not_imply_prospective_issuance": True,
        "files": {key: inspect_csv(root / rel) for key, rel in FILES.items()},
        "schwab_snapshots": [],
        "v4_live_component_count": 0,
        "v4_issued_record_count": 0,
        "v4_evaluation_count": 0,
    }
    for path in sorted((root / "data/raw/schwab").glob("TMF_1d_*.csv")):
        # Full row inspection ensures lengths and reference dates are observed,
        # not inferred from capture-name or filesystem timestamps.
        result["schwab_snapshots"].append(inspect_csv(path))
    result["v4_live_component_count"] = len(list(
        (root / "data/raw/v4_live").glob("*.json")
    ))
    result["v4_issued_record_count"] = len(list(
        (root / "data/forward_test/v4_opportunity").glob(
            "TMF_V4_OPPORTUNITY_*_5d.json"
        )
    ))
    result["v4_evaluation_count"] = len(list(
        (root / "data/forward_test/v4_opportunity/evaluations").glob(
            "TMF_V4_OPPORTUNITY_*_5d_evaluation.json"
        )
    ))
    result["readiness"] = (
        "REQUIRES_INDEPENDENT_POINT_IN_TIME_VERIFICATION"
    )
    result["note"] = (
        "Timestamp-column presence alone never proves authenticity, as-of "
        "availability, split/dividend parity, or risk-state continuity; "
        "this inventory does not authorize retrospective V4 issuance."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(inventory(args.root), indent=2))


if __name__ == "__main__":
    main()
