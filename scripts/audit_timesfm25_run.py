"""Audit a TimesFM 2.5 forecast without changing its original JSON."""

import argparse
import json
import re
from pathlib import Path

from scripts.timesfm25_integrity import (
    audit_existing_run, create_manifest, forecast_path,
    manifest_path, session_lock,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/forward_test/timesfm25"
SOURCE_DIR = ROOT / "data/raw/schwab"
HISTORY = ROOT / "data/raw/TMF_1d.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--adopt-legacy", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.session):
        parser.error("--session must be YYYY-MM-DD")

    if args.adopt_legacy:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with session_lock(OUTPUT_DIR, args.session):
            path = forecast_path(OUTPUT_DIR, args.session)
            existing_manifest = manifest_path(OUTPUT_DIR, args.session)
            if not path.is_file() or existing_manifest.exists():
                raise RuntimeError(
                    "Legacy adoption requires a forecast with no manifest"
                )
            create_manifest(
                path, source_dir=SOURCE_DIR,
                history_path=HISTORY, state="AUDITED_LEGACY",
            )

    verified = audit_existing_run(
        args.session, output_dir=OUTPUT_DIR,
        source_dir=SOURCE_DIR, history_path=HISTORY,
    )
    if verified is None:
        raise RuntimeError("No TimesFM forecast found for that session")
    print(json.dumps({
        "status": "AUDIT_PASSED",
        "forecast": str(verified),
        "manifest": str(manifest_path(OUTPUT_DIR, args.session)),
        "order_generated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
