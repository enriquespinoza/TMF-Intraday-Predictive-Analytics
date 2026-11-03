"""Research-only, fail-closed V4 historical source inventory and byte archive.

No prospective V4 issuance, historical availability attestation, governor replay,
model inference or brokerage operations. The recorded source declarations are
unverified statements until an independent B1/B2 review accepts the evidence.

Run (read-only):
    python -m scripts.build_v4_bootstrap_manifest --metadata PATH
Run (write-once local archive + manifest, explicitly opted in):
    python -m scripts.build_v4_bootstrap_manifest --metadata PATH --output
Verify an existing immutable archive, manifest, and source paths:
    python -m scripts.build_v4_bootstrap_manifest --verify-existing PATH
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "tmf_v4_bootstrap_source_manifest_v1"
DECLARATIONS = "tmf_v4_bootstrap_source_declarations_v1"
ROLES = ("tmf_historical", "treasury_nominal", "treasury_real_10y")
ARCHIVE = Path("data/raw/v4_bootstrap")
ACCEPTANCE = "NOT_ESTABLISHED"
MARKET_COLS = ("open", "high", "low", "close", "volume")
PROHIBITED_FIELDS = (
    "available_at_utc", "forward_return", "future_return", "future_close",
    "target_return", "label_5d", "actual_positive_return",
)


class BootstrapError(ValueError):
    """Unverifiable or invalid source evidence; research acceptance is blocked."""


def _canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _json(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BootstrapError(f"Cannot read JSON: {path}") from exc
    if not isinstance(data, dict):
        raise BootstrapError("Expected JSON object")
    return data


def _iso_date(raw, label):
    if not isinstance(raw, str) or len(raw) != 10:
        raise BootstrapError(f"{label} must be an ISO YYYY-MM-DD date")
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise BootstrapError(f"Invalid {label}: {raw}") from exc
    if parsed.isoformat() != raw:
        raise BootstrapError(f"Invalid {label}: {raw}")
    return parsed


def _market_session_date(raw, label):
    """Parse a market reference date, not an availability timestamp."""
    if isinstance(raw, str) and len(raw) == 10:
        return _iso_date(raw, label)

    if not isinstance(raw, str):
        raise BootstrapError(f"Invalid market session timestamp: {label}")

    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise BootstrapError(
            f"Invalid market session timestamp: {label}"
        ) from exc

    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise BootstrapError(f"{label} requires an explicit UTC offset")

    if (stamp.hour, stamp.minute, stamp.second, stamp.microsecond) != (0, 0, 0, 0):
        raise BootstrapError(f"{label} must be a midnight session tag")

    if raw not in (stamp.isoformat(sep=" "), stamp.isoformat(sep="T")):
        raise BootstrapError(f"{label} must use a canonical ISO timestamp")

    return stamp.date()


def _utc(raw, label):
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise BootstrapError(f"{label} must be an ISO timestamp or null")
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BootstrapError(f"Invalid {label}") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise BootstrapError(f"{label} requires an explicit UTC offset")
    return stamp.astimezone(timezone.utc).isoformat()


def _local_source(root, raw):
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise BootstrapError("Source must be a relative local path under data/raw")
    if "\\" in raw or any(part in ("..", ".") for part in Path(raw).parts):
        raise BootstrapError("Source path contains traversal or Windows separators")
    base = (root / "data/raw").resolve()
    path = root / raw
    if path.suffix.lower() != ".csv":
        raise BootstrapError("Source must be a CSV file")
    if path.is_symlink() or not path.is_file():
        raise BootstrapError(f"Missing or linked original source: {raw}")
    resolved = path.resolve()
    if not resolved.is_relative_to(base) or ARCHIVE.as_posix() in resolved.as_posix():
        raise BootstrapError("Source must be outside the bootstrap archive and under data/raw")
    return resolved


def _value(raw, label, *, positive=False):
    try:
        number = float(raw)
    except (ValueError, TypeError) as exc:
        raise BootstrapError(f"Invalid number for {label}") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise BootstrapError(f"Missing or nonfinite value for {label}")
    return number


def _observations(original, *, role, date_column, value_column):
    try:
        text = original.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BootstrapError("Source CSV must be UTF-8 readable; original bytes are preserved") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = reader.fieldnames
    if not fields or date_column not in fields:
        raise BootstrapError(f"Missing required date field: {date_column}")
    if any(field is None or not field for field in fields) or len(set(fields)) != len(fields):
        raise BootstrapError("Blank or duplicated CSV header")
    for field in fields:
        if any(flag in field.lower() for flag in PROHIBITED_FIELDS):
            raise BootstrapError(f"Future/availability field forbidden in historical context: {field}")
    if role == "tmf_historical":
        required = set(MARKET_COLS)
        if not required.issubset(fields):
            raise BootstrapError(f"Market CSV must include {sorted(required)}")
    elif not value_column or value_column not in fields:
        raise BootstrapError(f"Treasury CSV lacks declared value_column: {value_column}")
    first = last = None
    prior = None
    count = 0
    fractional_volumes = 0
    missing_yield_dates = []
    valid_yield_rows = 0
    for n, row in enumerate(reader, start=2):
        if None in row or any(v is None for v in row.values()):
            raise BootstrapError(f"Malformed CSV row {n}")
        day = (
            _market_session_date(row[date_column], f"row {n} {date_column}")
            if role == "tmf_historical"
            else _iso_date(row[date_column], f"row {n} {date_column}")
        )
        if prior is not None and day <= prior:
            raise BootstrapError(f"Dates duplicated or not increasing at row {n}")
        prior = day
        first = first or day
        last = day
        count += 1
        if role == "tmf_historical":
            values = {key: _value(row[key], f"row {n} {key}", positive=True) for key in MARKET_COLS}
            if values["high"] < max(values["open"], values["low"], values["close"]):
                raise BootstrapError(f"OHLC high invalid at row {n}")
            if values["low"] > min(values["open"], values["high"], values["close"]):
                raise BootstrapError(f"OHLC low invalid at row {n}")
            fractional_volumes += not values["volume"].is_integer()
        else:
            raw_yield = row[value_column].strip()
            if raw_yield in ("", "."):
                missing_yield_dates.append(day.isoformat())
            else:
                _value(raw_yield, f"row {n} {value_column}")
                valid_yield_rows += 1
    if not count:
        raise BootstrapError("Empty source CSV")
    if role != "tmf_historical" and not valid_yield_rows:
        raise BootstrapError("Treasury source contains no valid yield observations")
    return {
        "rows": count,
        "first_reference_date": first.isoformat(),
        "last_reference_date": last.isoformat(),
        "date_column": date_column,
        "value_column": value_column,
        "fractional_volume_rows": fractional_volumes if role == "tmf_historical" else None,
        "valid_yield_rows": valid_yield_rows if role != "tmf_historical" else None,
        "missing_yield_rows": len(missing_yield_dates) if role != "tmf_historical" else None,
        "missing_yield_dates": missing_yield_dates if role != "tmf_historical" else None,
    }


def _missing_sessions(first, last, observed_bytes, date_column):
    """Use the NYSE calendar if available; never certify continuity otherwise."""
    try:
        import pandas_market_calendars as calendars
    except ImportError:
        return {"status": "NOT_CHECKED_CALENDAR_DEPENDENCY_UNAVAILABLE", "missing_dates": None}
    fields = csv.DictReader(io.StringIO(observed_bytes.decode("utf-8-sig"), newline=""))
    present = {
        _market_session_date(
            row[date_column], f"NYSE {date_column}"
        ).isoformat()
        for row in fields
    }
    expected = [d.date().isoformat() for d in calendars.get_calendar("NYSE").valid_days(
        start_date=first, end_date=last,
    )]
    missing = [d for d in expected if d not in present]
    extra = sorted(present - set(expected))
    return {"status": "CHECKED", "missing_dates": missing, "non_nyse_dates": extra}


def _metadata_field(value, label):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise BootstrapError(f"{label} must be a nonempty string or null")
    return value.strip()


def _source(root, role, info, asof):
    if not isinstance(info, dict):
        raise BootstrapError(f"{role} declaration must be an object")
    declared_path = info.get("path")
    path = _local_source(root, declared_path)
    payload = path.read_bytes()
    original_sha = _digest(payload)
    if path.read_bytes() != payload:
        raise BootstrapError(f"Source modified while reading: {declared_path}")
    date_column = _metadata_field(info.get("date_column"), f"{role}.date_column")
    value_column = _metadata_field(info.get("value_column"), f"{role}.value_column")
    if not date_column:
        raise BootstrapError(f"{role}.date_column is required")
    observations = _observations(payload, role=role, date_column=date_column, value_column=value_column)
    if asof is not None and observations["last_reference_date"] >= asof:
        raise BootstrapError(f"{role} contains reference data on/after planned first live session")
    retrieved = _utc(info.get("retrieved_at_utc"), f"{role}.retrieved_at_utc")
    if retrieved and datetime.fromisoformat(retrieved).date() < _iso_date(observations["last_reference_date"], "last_reference_date"):
        raise BootstrapError(f"{role}: recorded retrieval time predates the newest source observation")
    filename = f"{role}/{original_sha}.csv"
    source = {
        "role": role,
        "path": declared_path,
        "sha256": original_sha,
        "original_byte_size": len(payload),
        "archived_relative_path": (ARCHIVE / "sources" / filename).as_posix(),
        **observations,
        "provider": _metadata_field(info.get("provider"), f"{role}.provider"),
        "source_url": _metadata_field(info.get("source_url"), f"{role}.source_url"),
        "license": _metadata_field(info.get("license"), f"{role}.license"),
        "retrieved_at_utc": retrieved,
        "retrieval_evidence_reference": _metadata_field(
            info.get("retrieval_evidence_reference"), f"{role}.retrieval_evidence_reference"
        ),
        "corporate_action_policy": _metadata_field(
            info.get("corporate_action_policy"), f"{role}.corporate_action_policy"
        ) if role == "tmf_historical" else None,
        "vintage_policy": _metadata_field(info.get("vintage_policy"), f"{role}.vintage_policy") if role != "tmf_historical" else None,
        "publication_policy": _metadata_field(info.get("publication_policy"), f"{role}.publication_policy") if role != "tmf_historical" else None,
        "independently_verified": False,
        "historical_available_at_attested": False,
    }
    if role == "tmf_historical":
        source["nyse_session_audit"] = _missing_sessions(
            observations["first_reference_date"], observations["last_reference_date"],
            payload, date_column,
        )
    missing = [key for key in (
        "provider", "source_url", "license", "retrieved_at_utc",
        "retrieval_evidence_reference",
        "corporate_action_policy" if role == "tmf_historical" else "vintage_policy",
    ) if not source.get(key)]
    if role != "tmf_historical" and not source.get("publication_policy"):
        missing.append("publication_policy")
    source["unresolved_metadata"] = missing
    return source, payload


def build(root, declarations):
    root = Path(root).resolve()
    if declarations.get("schema") != DECLARATIONS:
        raise BootstrapError("Invalid source declaration schema")
    asof = declarations.get("as_of_start_session")
    if asof is not None:
        _iso_date(asof, "as_of_start_session")
    sources = declarations.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(ROLES):
        raise BootstrapError(f"Exactly three source roles required: {ROLES}")
    records = {}
    original_bytes = {}
    for role in ROLES:
        record, content = _source(root, role, sources[role], asof)
        records[role] = record
        original_bytes[role] = content
    if len({records[r]["sha256"] for r in ROLES}) != len(ROLES):
        raise BootstrapError("Same file bytes used for multiple distinct source roles")
    blockers = [f"{role}.{missing}" for role in ROLES for missing in records[role]["unresolved_metadata"]]
    for role in ("treasury_nominal", "treasury_real_10y"):
        if records[role]["missing_yield_rows"]:
            blockers.append(f"{role}.missing_yield_observations")
    nyse = records["tmf_historical"]["nyse_session_audit"]
    if nyse["status"] != "CHECKED":
        blockers.append("market.nyse_calendar_not_checked")
    elif nyse["missing_dates"] or nyse["non_nyse_dates"]:
        blockers.append("market.nyse_session_alignment_unresolved")
    if records["tmf_historical"]["rows"] < 90:
        blockers.append("market.insufficient_90_session_history")
    if records["tmf_historical"]["fractional_volume_rows"]:
        blockers.append("market.fractional_volume_requires_review")
    blockers.extend([
        "independent_market_adjustment_and_schwab_parity_unverified",
        "independent_treasury_publication_and_vintage_unverified",
        "historical_observation_availability_not_attested",
        "B3_risk_memory_replay_not_implemented",
    ])
    record = {
        "schema": SCHEMA,
        "status": "HISTORICAL_CONTEXT_ONLY",
        "source_acceptance": ACCEPTANCE,
        "as_of_start_session": asof,
        "research_only": True,
        "sources": records,
        "blockers": blockers,
        "B1": ACCEPTANCE,
        "B2": ACCEPTANCE,
        "B3": "BLOCKED",
        "B4": "BLOCKED",
        "B5": "BLOCKED",
        "risk_initialization": None,
        "historical_availability_attested": False,
        "authorizes_live_v4_issuance": False,
        "order_generated": False,
    }
    record["source_bundle_sha256"] = _digest(_canonical(record))
    return record, original_bytes


def _safe_archive_path(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to((root / ARCHIVE).resolve()):
        raise BootstrapError("Archive path escapes bootstrap output directory")
    if path.is_symlink():
        raise BootstrapError("Symlink archive paths are forbidden")
    return path


def _write_once(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise BootstrapError(f"Linked archive path forbidden: {path}")
    try:
        with path.open("xb") as output:
            output.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise BootstrapError(f"Immutable archive conflict/tampering: {path}")
    if path.read_bytes() != payload:
        raise BootstrapError(f"Archive verification failed: {path}")


def verify(root, manifest_path):
    root = Path(root).resolve()
    path = Path(manifest_path).resolve()
    if not path.is_relative_to((root / ARCHIVE).resolve()):
        raise BootstrapError("Manifest must be inside local bootstrap archive")
    saved = _json(path)
    if saved.get("schema") != SCHEMA or saved.get("source_acceptance") != ACCEPTANCE:
        raise BootstrapError("Invalid or unauthorized source manifest")
    if saved.get("authorizes_live_v4_issuance") is not False or saved.get("order_generated") is not False:
        raise BootstrapError("Source manifest must not authorize trading or live V4")
    without_digest = {k: v for k, v in saved.items() if k not in ("created_at_utc", "source_bundle_sha256")}
    if _digest(_canonical(without_digest)) != saved.get("source_bundle_sha256"):
        raise BootstrapError("Manifest content/hash mismatch")
    for role in ROLES:
        source = saved["sources"][role]
        original = _local_source(root, source["path"])
        archive = _safe_archive_path(root, source["archived_relative_path"])
        for name, filename in (("original", original), ("archive", archive)):
            if not filename.is_file() or _digest(filename.read_bytes()) != source["sha256"]:
                raise BootstrapError(f"{role}: {name} source bytes differ from manifest")
    return saved


def save(root, record, data):
    root = Path(root).resolve()
    for role in ROLES:
        archive = _safe_archive_path(root, record["sources"][role]["archived_relative_path"])
        current_path = _local_source(root, record["sources"][role]["path"])
        if (_digest(data[role]) != record["sources"][role]["sha256"]
                or _digest(current_path.read_bytes()) != record["sources"][role]["sha256"]):
            raise BootstrapError(f"Original source changed for {role}")
        _write_once(archive, data[role])
    destination = root / ARCHIVE / "manifests" / (
        f"TMF_V4_BOOTSTRAP_SOURCE_{record['source_bundle_sha256'][:24]}.json"
    )
    if destination.exists():
        existing = verify(root, destination)
        if {k:v for k,v in existing.items() if k != "created_at_utc"} != record:
            raise BootstrapError("Existing immutable manifest does not match current sources")
        return destination, "ALREADY_VERIFIED"
    stamped = {**record, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_once(destination, json.dumps(stamped, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
    verify(root, destination)
    return destination, "ARCHIVED_HISTORICAL_CONTEXT_ONLY"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--metadata", type=Path, help="Local declarations JSON (check-only unless --output)")
    options.add_argument("--verify-existing", type=Path, help="Verify an archived manifest and all source bytes")
    parser.add_argument("--output", action="store_true", help="Explicitly save original-byte archives and write-once manifest")
    args = parser.parse_args()
    if not args.metadata and not args.verify_existing:
        parser.error("Specify --metadata or --verify-existing")
    if args.verify_existing:
        if args.output:
            parser.error("--output is not valid with --verify-existing")
        result = verify(args.root, args.verify_existing)
        print(json.dumps({"status": "VERIFIED_HISTORICAL_SOURCE_ONLY", "source_bundle_sha256": result["source_bundle_sha256"], "authorizes_live_v4_issuance": False}, indent=2))
        return
    record, data = build(args.root, _json(args.metadata))
    if args.output:
        destination, status = save(args.root, record, data)
        print(json.dumps({"status": status, "manifest": str(destination), "blockers": record["blockers"], "authorizes_live_v4_issuance": False}, indent=2))
    else:
        print(json.dumps({"status": "CHECK_ONLY", **record}, indent=2))


if __name__ == "__main__":
    main()
