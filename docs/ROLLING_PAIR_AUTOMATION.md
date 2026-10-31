# TMF rolling paired-experiment automation — pilot

## Purpose and boundary

Research-only provisional scheduling *inside the TMF repository*, to be
superseded by the Top-Level Investment Research Orchestrator. This automation
must never place trades, alter frozen V3_FINAL or TimesFM 2.5 specifications,
create retroactive forecasts, overwrite evidence, silently repair interrupted
locks, or backfill missed model origins. It reuses existing audited APIs.

**Status:** Code committed on an isolated feature branch; unattended runs
are not verified until local tests, OAuth preflight, Task Scheduler dry run,
and at least one supervised real-session acceptance test succeed.

## Statistical acceptance

Target 80 **matched-source paired five-session origins**, with two recorded
model forecasts and eventual evaluation per origin. Each paired origin is
one paired comparison, not two independent trials. Do not describe 80 samples
as a +/-3 percentage-point precision target. Under an IID 50% binomial
conservative approximation, n=80 has about +/-11 percentage points 95%
half-width; n around 1,067 is needed for +/-3 points. Five-session targets
overlap and effective sample sizes may be lower. Include HAC or block bootstrap
intervals, coverage, Brier, MAE vs persistence, and per-regime analyses later.
No tuning on the 2026 holdout.

## Environment requirements

Windows Eastern Standard Time system timezone; power on; user signed in; one
dedicated repo at C:\Dev\TMF-Intraday-Predictive-Analytics; pinned Python
interpreters C:\Dev\.venvs\tmf-analytics\Scripts\python.exe and
C:\Dev\.venvs\tmf-timesfm25\Scripts\python.exe. Have a pre-authorized Schwab
token whose refresh works **without interactive browser authentication**;
`scripts.download_schwab_tmf` uses schwab-py `easy_client`, so a missing or
expired token requiring browser authorization will fail closed in the scheduler.
The script captures but does not print OAuth child output.

TimesFM checkpoint revision is pinned to
`1d952420fba87f3c6dee4f240de0f1a0fbc790e3`. The model must already be
cached and runnable without human authentication or download consent.
Credentials remain in local environment/token paths, never Git or logs.

## Daily schedule (weekday, Eastern local time)

* 5:30 PM ET: after NYSE close +30m; run once. If the day is not an NYSE
  session, calendar check skips. Reuse original snapshot on retry; otherwise
  download one daily + 5m Schwab pair, validate dates/78 RTH bars on regular
  days and closing-price tolerance, audit PAYEMS-A readiness, run frozen V3
  and TimesFM once each, verify strict source SHA equality, score the matured
  TimesFM observation. V3 score deferred if UTC calendar is still the target
  date.
* 8:15 PM ET: retry same idempotent cycle. Already issued forecasts are
  audited and skipped. Score the matured V3 origin now that it is the next
  UTC calendar date. No new snapshot if original matched pair exists.
* If a market half-day, fixed time still lies after finalization. The market
  calendar governs whether the window is really READY.
* A missed forecast window is logged as skipped. Never issue historical
  forecasts to fill a gap.

Two tasks share the same Task Scheduler name with two trigger times, with
overlap disabled. The Windows task runs only while the specific user is
logged on, due to user-scoped OAuth access. A powered-off, signed-out, or
disconnected device is NOT an unattended service. If truly always-on
issuance is required, first design a separately authorized always-on host
with secure secret storage and source licensing.

## Install **after local regression tests**

```bash
cd /c/Dev/TMF-Intraday-Predictive-Analytics
git -c gc.auto=0 fetch origin feature/tmf-daily-research-automation
git switch feature/tmf-daily-research-automation
git -c gc.auto=0 pull --ff-only origin feature/tmf-daily-research-automation

source /c/Dev/.venvs/tmf-analytics/Scripts/activate
python -m pytest -q tests/test_automate_daily_research.py
python -m pytest -q
python -m pip check
python -m scripts.automate_daily_research --dry-run
```

The `--dry-run` flag checks only the NYSE calendar. No Schwab data is fetched,
no files are written, no forecasts or scores are created.

In a separate Windows PowerShell terminal under the intended interactive
Windows account, inspect and then run the opt-in installer:

```powershell
cd C:\Dev\TMF-Intraday-Predictive-Analytics
Get-TimeZone
Get-Content .\scripts\install_daily_research_task.ps1
.\scripts\install_daily_research_task.ps1
Get-ScheduledTask -TaskName 'TMF V3 TimesFM Rolling Research' |
    Get-ScheduledTaskInfo
```

Do **not** register until OAuth access and both Python environments are
validated. On the next NYSE session, supervise both triggers and verify
two append-only receipts in `reports/automation/events` plus the original
research artifacts. Confirm no OAuth prompt, no missed-pair origin, no
unmatched hashes, no order generation, no attempts to rescore existing files.

To disable the Task Scheduler job:

```powershell
Disable-ScheduledTask -TaskName 'TMF V3 TimesFM Rolling Research'
```

To inspect recent result codes:

```powershell
Get-ScheduledTaskInfo -TaskName 'TMF V3 TimesFM Rolling Research'
```

## Recovery contract

* Unavailable Schwab OAuth / stale snapshot / missing five-minute bars:
  **fail, no synthetic input or partial price approximation**.
* If V3 exists but TimesFM does not, use the **same original validated
  snapshot** for TimesFM when the prospective window is still open.
* Missing or invalid forecast manifest: **fail**, manual repair and forensic
  review; never replace or erase the record.
* Missing PAYEMS-A activated release on/after approved date: **fail**.
* Re-run after failure only during the same valid market window. Log each
  attempt as a new operational receipt, separate from immutable forecasts.
* If the pre-UTC 5:30 PM V3 score has not become eligible, mark deferred, not
  failed; 8:15 PM handles it. Existing evaluations are checked by forecast
  and outcome source hashes instead of being overwritten.
* If jobs cannot run, the ledger must show a missing origin; never claim
  the experiment was completed.
* The local runner's status receipts are ignored by Git; immutable source
  snapshots and forecast/evaluation artifacts stay in existing local dirs.

## Handoff to the Top-Level Investment Research Orchestrator

Retain the validated source, forecast, score, and manifest contracts.
Replace only local trigger/orchestration. Before enabling a new scheduler,
disable the old task, verify there is no overlapping issuance, and conduct
parallel *read-only* comparisons of expected next origins and artifact paths.
