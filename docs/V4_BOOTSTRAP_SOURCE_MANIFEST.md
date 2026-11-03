# V4-BOOT-001 reconstructed local source manifest (research-only)

This component is a **source archive and evidence inventory**, not the B3 Risk Governor historical bootstrap itself. Its schema is `tmf_v4_bootstrap_source_manifest_v1`, **not** the prospective V4 opportunity ledger or final B3 bootstrap state manifest. It makes no historical point-in-time assertions and never authorizes V4 issuance.

## Input and evidence policy

Provide three independently identifiable historical source CSVs under the **isolated checkout's** `data/raw/` folder (which Git ignores): TMF market OHLCV, nominal 30-year Treasury yield, and real 10-year Treasury yield. Do not take existing private or licensed market data from another machine without confirming licensing and provenance. Preserve original source files and source acquisition evidence unchanged. Data copied from another checkout are not newly acquired independent evidence.

The required input declaration is JSON. Use the following template and replace *only* documented facts. Keep unknown fields `null` rather than inventing historical timestamps, corporate-action policies, publication schedules, or vintage evidence:

```json
{
  "schema": "tmf_v4_bootstrap_source_declarations_v1",
  "as_of_start_session": null,
  "sources": {
    "tmf_historical": {
      "path": "data/raw/TMF_1d.csv",
      "date_column": "market_date",
      "provider": null,
      "source_url": null,
      "license": null,
      "retrieved_at_utc": null,
      "retrieval_evidence_reference": null,
      "corporate_action_policy": null
    },
    "treasury_nominal": {
      "path": "data/raw/treasury_yields_daily.csv",
      "date_column": "date",
      "value_column": "yield_30y",
      "provider": null,
      "source_url": null,
      "license": null,
      "retrieved_at_utc": null,
      "retrieval_evidence_reference": null,
      "vintage_policy": null,
      "publication_policy": null
    },
    "treasury_real_10y": {
      "path": "data/raw/macro/real_yield_10y.csv",
      "date_column": "date",
      "value_column": "real_yield_10y",
      "provider": null,
      "source_url": null,
      "license": null,
      "retrieved_at_utc": null,
      "retrieval_evidence_reference": null,
      "vintage_policy": null,
      "publication_policy": null
    }
  }
}
```

Column names in this template are **illustrative** and must be reconciled against the actual unmodified CSV headers. The TMF CSV must contain the columns `open`, `high`, `low`, `close`, `volume` plus the declared date column. Treasury source files need their real date and yield column names. Every required row is checked for ascending, unique ISO reference dates and valid numeric values. The TMF historical CSV may be declined if it contains future-return or `available_at_utc` columns, because this is a historical-context-only source contract; such columns must be handled by a distinct validated live-evidence interface, not a reinterpretation of the historical file.

`retrieved_at_utc`, if supplied, must be an actual independently evidenced retrieval timestamp with UTC offset; it is **not** an observation release time and does not establish historical point-in-time availability. Claims about licensing, corporate actions, vintage and publication timing require human review.

## Commands (from the isolated V4 checkout)

```bash
cd /c/Dev/TMF-V4-Research

# Only after copying the declaration template to a local ignored path.
# Checks and prints the metadata/row inventory; writes nothing.
python -m scripts.build_v4_bootstrap_manifest \
  --metadata data/raw/v4_bootstrap_declarations.json

# Explicit local, ignored, write-once original-byte archive and manifest.
# Still NOT accepted for B1/B2, even if all fields are populated.
python -m scripts.build_v4_bootstrap_manifest \
  --metadata data/raw/v4_bootstrap_declarations.json --output

# Verify a previously emitted manifest and all local original + archive bytes.
python -m scripts.build_v4_bootstrap_manifest \
  --verify-existing data/raw/v4_bootstrap/manifests/TMF_V4_BOOTSTRAP_SOURCE_<digest>.json

python -m pytest tests/test_v4_bootstrap_manifest.py -q
python -m pytest -q
```

CLI `--output` is opt-in and never overwrites a manifest or archived source. If existing bytes differ, it raises an error; it does **not** repair or replace them. Content-addressed originals and write-once manifests live under ignored `data/raw/v4_bootstrap/`. A `--check` run (the default) creates nothing. Running with missing source files fails closed. No instructions in this document authorize fetching, generating, or altering licensed historical files.

## Limitations / gates

- Input `retrieval_evidence_reference` is a *claimed reference*, not verified evidence. This builder records claims and hashes exact local source bytes, but **does not independently verify vendor identity, licenses, retrieval-time authenticity, corporate-action adjustment, Treasury publication dates, Treasury vintages, source cross-parity, or historical PIT validity**.
- NYSE session continuity is checked when `pandas_market_calendars` is installed; otherwise the manifest explicitly records that the calendar check could not be performed. Missing/extra NYSE dates remain blockers. A continuous reference-date sequence alone is not a provenance attestation.
- The manifest is always `HISTORICAL_CONTEXT_ONLY`, `B1/B2=NOT_ESTABLISHED`, `B3/B4/B5=BLOCKED`, `risk_initialization=null`, and `authorizes_live_v4_issuance=false`.
- B3 hypothetical-flat GovernorMemory replay is **not** included. The older `docs/V4_HISTORICAL_BOOTSTRAP_GATE.md` remains the authority for future B3–B5 validation.
- The current isolated checkout may not include any raw historical data because Git ignores `data/raw/*`. This builder cannot work on real source files until the data and documentation are independently supplied.
- No changes to existing V3_FINAL, TimesFM, V4 daily live evidence generator, opportunity ledger, scheduler, frozen policies, or trading APIs.

## Environment and evaluation results

Reconstruction was written and tested separately from the user's Windows checkout. The developer-side suite uses synthetic CSV fixtures only. Run the project's full suite in the isolated Windows checkout and report all failures, including the two known data-dependent PAYEMS/inflation dataset exceptions. Do not create fake feature files or alter those tests to make a full-suite result appear green.
