# PAYEMS-A prospective refresh runbook (V3_FINAL frozen)

## Scope and model invariants

V3_FINAL still uses the five frozen technical features plus the three frozen PAYEMS-A inputs: `payroll_change_1m`, `payroll_revision_total`, and `payroll_broad_revision_event`. Its training cutoff remains December 31, 2025. The research forecasting paths and five-session outcomes remain immutable. The refresh changes **only** the current operational event source.

BLS official release schedule: https://www.bls.gov/schedule/news_release/empsit.htm

The next published schedule entry is **2026-11-06 08:30 America/New_York** for October 2026 employment. The BLS scheduled date does **not** prove the data was published. Verify that the Employment Situation report actually appeared and ALFRED has an event vintage for November 6 before activating. If BLS changes the release schedule, update the explicit versioned schedule and tests *before* release processing. Never invent or backdate observation timestamps.

## Before the release

```bash
cd /c/Dev/TMF-Intraday-Predictive-Analytics
git pull --ff-only origin feature/timesfm-2p5-research
source /c/Dev/.venvs/tmf-analytics/Scripts/activate
python -m pytest -q tests/test_refresh_payems_a.py
python -m pytest -q
python -m pip check
```

Check the baseline `data/features/PAYEMS_event_features.csv` latest row is October 2, 2026 and contains three finite frozen PAYEMS-A fields. Ensure `FRED_API_KEY` is configured locally in `.env`. Do not commit API keys.

## On November 6 after 8:30 AM ET and official release confirmation

1. Verify the reference period is October 2026 at BLS and source availability is genuine.
2. Stage the ALFRED source:
   ```bash
   python -m scripts.refresh_payems_a --release 2026-11-06 --stage
   ```
3. Review `data/raw/payems_a_refresh/PAYEMS_A_2026-11-06_manifest.json`, `..._vintages.csv`, and `..._candidate.csv`. Confirm initial October payroll level, published changes/revisions, release clock, frozen feature values, source hashes, preserved preceding event history and no future vintage data. Staging never changes the active event CSV.
4. Rerun the focused and full regression tests. Only after manual approval activate once:
   ```bash
   python -m scripts.refresh_payems_a --release 2026-11-06 --activate
   ```
5. Verify `data/features/PAYEMS_event_features.refresh.json` is present, contains matching `active_sha256`, and `python -m scripts.payems_a_readiness --help` is **not** the supported entry point (it is a library validator). Use a read-only Python assertion:
   ```bash
   python -c "from scripts.payems_a_readiness import require_payems_ready; print(require_payems_ready('2026-11-06'))"
   ```
   It must print `2026-11-06`.

Only after the 2026-11-06 NYSE market close +30 minutes, and before the next NYSE open, issue a new V3 forecast with a newly validated post-close Schwab snapshot using the existing `scripts.run_daily_forecast` orchestrator. Its predictor refuses to use stale PAYEMS features if the approved scheduled release has become due without a matching activated, hashed event row. Never reissue an earlier forecast.

## Fail-closed cases

* No BLS confirmation, no ALFRED release vintage, wrong reference month or publication date: **do not activate**.
* Staging before scheduled release: rejected.
* Already staged artifacts: rejected rather than silently overwritten.
* Baseline, schedule, candidate or vintage hash differs from staged evidence: activation rejected.
* Frozen features missing/nonfinite, newer reference not released, or incorrect release clock: rejected.
* Latest event is stale after the scheduled due date: new V3 forecast rejected with `PAYEMS_A_REFRESH_REQUIRED`.
* Historical origin forecasts cannot be reissued and previous V3/TimesFM forecasts and outcomes must never be rewritten.

For future monthly releases, review and extend `config/payems_a_refresh_schedule.json` in a separate audited commit before the corresponding release. The activation creates an integrity sidecar for operational inference, not a new training sample.

## Important technical limitation

The new refresh pipeline is an operational guard and staged provenance chain; it **does not independently establish that the scheduled BLS release actually occurred**. Its production activation must be preceded by external BLS confirmation. The tests use synthetic vintage data; no November 6 real vintage can be validated before that date. There is no trading authorization.
