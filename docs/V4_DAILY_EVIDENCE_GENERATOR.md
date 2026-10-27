# V4 daily evidence generator — source contract and limits

The new `scripts/build_v4_daily_evidence.py` computes Risk Governor V3, Technical Opportunity V1 and Treasury Confirmation V1 from independently timestamped daily market and Treasury source histories. Its outputs are three upstream JSON component files for the existing independent `scripts/v4_opportunity_ledger.py issue` command.

**This is a fail-closed research adapter, not a broker, price forecasting model or automated data source.** V3_FINAL and TimesFM source files, weights, forecasts and evaluations are never modified.

## Required point-in-time data

Both source CSV files must be stored directly under `data/raw/v4_inputs/` and represent actual contemporaneously collected evidence, not exported/relabelled historical files.

Market-history CSV required columns:

```
market_date,open,high,low,close,volume,available_at_utc
```

- At least 90 complete, strictly chronological NYSE market dates, ending on the origin.
- Prices must use the same corporate-action-adjusted basis as the validated Schwab origin snapshot; at least the last 10 overlapping daily closes must agree within $0.001.
- Each historical row must carry its real, timezone-aware post-close source availability timestamp inside that day's prospective issuance window. A newly generated timestamp for old data is **not** acceptable.
- No gaps, non-NYSE dates, future dates, negative/missing prices, or impossible OHLC ranges.
- Frozen Technical Opportunity V1 computes historical RSI, five-day returns, 20-day-high distance, EMA21 streak and prior-only rolling percentile scores from this history.

Treasury-history CSV required columns:

```
reference_date,yield_30y,real_yield_10y,available_at_utc
```

- At least 70 chronological valid Treasury observation dates with explicit source availability timestamps.
- Yields are in **percentage points**, e.g., `4.30`, not `0.043`; the generator calculates basis-point changes by multiplying percentage-point differences by 100.
- The independent data publisher's as-of availability time is essential. FRED observation dates alone are not sufficient evidence of what had been published by the decision timestamp. Historical research exports without per-row availability timestamps are not suitable for this adapter.
- The latest eligible Treasury reference must be no more than four calendar days older than the origin. Missing or stale dates fail rather than inventing yield values.
- Treasury percentile ranks use successive **valid Treasury observations**, not forward-filled duplicates on the TMF calendar.
- The Risk Governor replay joins each market decision only to Treasury releases available at that historical decision time.

Risk Governor V3 is replayed in **hypothetical flat-position research mode**, starting from the explicit baseline `GovernorMemory()` (GREEN, zero recovery streak) at the first complete historical risk row. This does not reconstruct a real portfolio, intraday FORCE_EXIT signal or live position memory. No such trading state may be claimed from its output. Any stale gap in the required Risk Governor replay or fewer than 63 eligible sessions fails closed.

The adapter binds the generated JSON source hashes to the underlying CSV histories and the exact Schwab snapshot. The existing ledger now validates those underlying hashes again when issuing/auditing the state. These hashes are local file-integrity proofs, **not independently trusted publication timestamps**.

## Usage: only for a genuinely open prospective NYSE window

```bash
source /c/Dev/.venvs/tmf-analytics/Scripts/activate

python -m scripts.build_v4_daily_evidence \
  --origin YYYY-MM-DD \
  --daily data/raw/schwab/TMF_1d_<capture-id>.csv \
  --market-history data/raw/v4_inputs/<attested-market-history>.csv \
  --treasury-history data/raw/v4_inputs/<attested-treasury-history>.csv \
  --check
```

If source history was separately verified, the current-session window is still open, and the dry run succeeds, replace `--check` with `--run`. The tool writes three immutable JSON components directly under `data/raw/v4_live/` and prints their precise filenames.

Then review and separately issue the V4 state:

```bash
python -m scripts.v4_opportunity_ledger issue \
  --origin YYYY-MM-DD \
  --daily data/raw/schwab/TMF_1d_<capture-id>.csv \
  --risk data/raw/v4_live/<generated-risk-json>.json \
  --technical data/raw/v4_live/<generated-technical-json>.json \
  --treasury data/raw/v4_live/<generated-treasury-json>.json \
  --check
```

The same command with `--run` writes the immutable state and SHA-256 manifest. V4 is not auto-issued by the generator.

**No retroactive issuance:** Do not run these commands for October 21–26 if the original prospective window is closed. The ordinary Schwab downloader returns only a short lookback (26 daily sessions in the October 26 example), which is **not enough** for a 90-session point-in-time market history; do not fabricate missing rows or timestamps. Until genuine attested histories exist, V4 must continue to report **unavailable** while V3 and TimesFM remain unchanged.

## Test and audit commands

```bash
python -m pytest -q tests/test_build_v4_daily_evidence.py tests/test_v4_opportunity_ledger.py
python -m pytest -q
python -m pip check
python -m scripts.v4_opportunity_ledger report
```

This component is deliberately stricter than historical research. For any later production-grade system, add independently verifiable archival timestamps and release-vintage history, intraday emergency signals, complete risk position memory, and a separate documented external data ingest flow. Do not retrofit those assumptions into old forecasts.
