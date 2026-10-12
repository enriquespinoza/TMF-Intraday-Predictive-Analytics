# V3 + V4 read-only assessment integration

This adapter joins a **previously recorded** frozen V3 five-trading-day forecast JSON to two independent V4 daily assessment CSVs. It does **not** calculate V3 probabilities, recompute V4 states, refresh Schwab market data, authorize exposure, or call a broker.

## Inputs

- `data/forward_test/TMF_V3_SCHWAB_*.json` — locally generated V3 forecast (excluded from Git).
- `reports/V4_opportunity_state_machine_v1_daily.csv` — independent V4 opportunity observations.
- `reports/V4_risk_governor_v3_daily.csv` — independent V4 risk observations.

The V4 CSVs **must** have one unambiguous `market_date`, `session_date`, `trading_date`, `date`, or timezone-aware `timestamp` per session. The adapter preserves every returned column as a source record instead of interpreting policy-specific flags.

## Usage

From the repository root:

```bash
python -m pytest -v tests/test_integrate_v3_v4_assessments.py
python -m src.analysis.integrate_v3_v4_assessments
```

To specify explicit inputs:

```bash
python -m src.analysis.integrate_v3_v4_assessments \
  --forecast data/forward_test/TMF_V3_SCHWAB_2026-10-09_20261012T011010Z.json \
  --opportunity-csv reports/V4_opportunity_state_machine_v1_daily.csv \
  --risk-csv reports/V4_risk_governor_v3_daily.csv
```

Reports are saved as unique, timestamped JSON files in `reports/generated/` (ignored by Git). The original model and report inputs are never rewritten.

## Fail-closed statuses

- `matched`: exactly one record at the forecast origin.
- `date_mismatch`: no same-day record; an older report is **not** promoted to current.
- `unavailable`: source file absent.
- `empty`: no report observations.
- `invalid_schema`: missing date column or invalid date/timestamp.
- `ambiguous_duplicate_date`: multiple records for a session; none is selected.

Output always includes `trading.status=NOT_AUTHORIZED_READ_ONLY`, `orders_generated=false`, source SHA-256 hashes, independent V4 status objects, and a boolean date-alignment flag. It is a research report, **not** a recommendation or execution request.

## Scope limits

The remote GitHub `main` branch currently does not contain the user's local V3/V4 engine modules or frozen configuration. This integration is intentionally source-agnostic so it can be reviewed on `main` without claiming those models are present or reimplementing their logic.

Before live V4 calculations can be incorporated, publish or selectively port the reviewed V4 engine code, verify output column meanings, add point-in-time availability checks for underlying Treasury/macro data, test feature parity, and run the complete local suite. A same-day report date alone does not establish a correct point-in-time calculation.
