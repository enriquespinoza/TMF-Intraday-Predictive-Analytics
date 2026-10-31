# V4 historical bootstrap: controlled provenance and research-only initialization

**Status:** PROPOSED / NOT AUTHORIZED FOR PROSPECTIVE ISSUANCE  
**Scope:** TMF Technical Opportunity V1, Treasury Confirmation V1, Risk Governor V3, and Opportunity Engine V1.  
**Isolation:** Do not change V3_FINAL, TimesFM 2.5, their scheduler, their artifacts, V4 frozen thresholds, or the existing strict live generator and ledger.

## Findings from source inventory

The existing read-only inventory reports sufficient historical row counts but no qualified, independently evidenced publication/availability timestamps per observation. The existing `scripts/build_v4_daily_evidence.py` therefore correctly rejects the historical CSVs as live evidence. A Schwab 26-session snapshot cannot substitute for a 90+ session technical and Treasury initialization history. Historical Risk Governor feature tables include future outcome fields and cannot be used directly as contemporaneous inputs.

The bootstrap is designed to distinguish two separate assertions:

1. **Historical computational context (HISTORY_BOOTSTRAP).** An audited, frozen set of prices and Treasury observations already known before the first V4 origin can initialize indicators and hypothetical research-only risk memory. Its original source, retrieval timestamp, correction/revision policy, adjustment convention and byte hashes are retained. The bootstrap audit never asserts when a historical value first became available.
2. **Prospective contemporary evidence (LIVE_ATTESTED).** Exact Schwab source snapshots and independently acquired Treasury values used at a new V4 decision have verifiable, genuine acquisition or publication timestamps at or before the decision. Neither a filename nor a newly added timestamp for a historical row is sufficient proof of point-in-time availability.

A replay computed with currently available history is an **initialization reconstruction**, not proof of a historically actionable trading strategy or a historical point-in-time backtest. The first eligible prospective V4 origin is not guaranteed and may be later than the bootstrap audit.

## Required immutable bootstrap manifest (version 1)

```json
{
  "schema": "tmf_v4_bootstrap_manifest_v1",
  "status": "HISTORICAL_CONTEXT_ONLY",
  "research_only": true,
  "created_at_utc": "<actual recorded generation timestamp>",
  "as_of_start_session": "<first permitted NEW V4 prospective session>",
  "market_source": {
    "path": "<local immutable original file>",
    "sha256": "<original byte digest>",
    "first_reference_date": "<YYYY-MM-DD>",
    "last_reference_date": "<YYYY-MM-DD>",
    "rows": 0,
    "retrieved_at_utc": "<actual fetch time, if independently evidenced>",
    "corporate_action_policy": "<explicit declared verified convention>"
  },
  "treasury_sources": [
    {
      "series": "DGS30 or separately qualified 30y",
      "path": "<local immutable original file>",
      "sha256": "<original byte digest>",
      "vintage_policy": "<explicit documented revision and vintage coverage>",
      "publication_policy": "<evidence for available date/timing>"
    },
    {
      "series": "DFII10 or independently qualified real 10y",
      "path": "<local immutable original file>",
      "sha256": "<original byte digest>",
      "vintage_policy": "<explicit documented revision and vintage coverage>",
      "publication_policy": "<evidence for available date/timing>"
    }
  ],
  "risk_initialization": {
    "mode": "HYPOTHETICAL_FLAT_RECONSTRUCTION",
    "starting_memory": {"state": "GREEN", "recovery_streak": 0},
    "ending_memory": {"state": "<audited computed state>", "recovery_streak": 0},
    "replay_source_hashes": ["<hashes>"],
    "replay_algorithm_hash": "<hash>",
    "policy_hash": "<hash>",
    "replayed_sessions": 0,
    "rejected_gap_count": 0
  },
  "historical_availability_attested": false,
  "authorizes_live_v4_issuance": false,
  "order_generated": false
}
```

Store original-byte checksums and source metadata in a write-once `data/raw/v4_bootstrap/` record. Do not modify original snapshots or reinterpret their timestamps. The manifest schema above is a design contract; no issued manifest exists yet.

## Gate sequence before implementation or activation

1. **Independent source refresh:** obtain new named, versioned TMF daily OHLCV, nominal 30-year and 10-year real Treasury series; retain old files unchanged. Record actual retrieval metadata, license, first/last dates, missing days, source SHA-256, publication policy, and as-of-vintage policy. Do not use a current revised FRED series as evidence of what was published historically.
2. **Market basis reconciliation:** determine split, distribution, raw-vs-adjusted price policy, verify all matching historical Schwab sessions, and document any unresolved differences. Recent 10-session parity alone does not establish earlier corporate-action parity.
3. **Treasury eligibility:** match observation reference dates and genuine availability where known, explicitly flag reconstructed/unknown timing in bootstrap rows, and separately capture current prospective source at origin. Reject stale or unavailable real/nominal yields, silent forward-fill, and post-origin revisions.
4. **Bootstrap computation:** calculate frozen indicators from reference-date chronological history; retain exact function/policy hashes, lookback limits, prior-only percentile behavior, and explicit hypothetical-flat Risk Governor V3 replay. Save ending `GovernorMemory(state, recovery_streak)` and indicator warmup state. Audit continuity across skipped dates; do not silently reset hysteresis.
5. **Cutover consistency:** freeze history before the first prospective origin; when new session data becomes available, extend the state once with genuine live evidence. Compare full replay and checkpoint replay outcomes on qualified fixtures; reject any mismatch. Prohibit importing historical `forward_return_*` fields into V4 components.
6. **Read-only end-to-end rehearsal:** use `--check` without issuing live components/states, test missing source/time, tampering, Treasury late release, weekend gaps, dividend changes, and crash recovery; run full regression suite.
7. **Independent authorization:** only after verified market parity, Treasury vintage/release timing, state continuity, source-hash attestations and reviewed audit evidence may a new `bootstrap_v1` pathway be connected to `scripts/build_v4_daily_evidence.py`. The ledger remains a separate explicit issuance gate. No retroactive V4 states.

## Architecture boundary

The existing strict `market_history.csv` and `treasury_history.csv` live contract must remain untouched. A future versioned bootstrap adapter shall accept a **bootstrap manifest and genuine current-session incremental evidence** instead of pretending each historical row has a dated `available_at_utc`. Its output must include a `historical_context_only` provenance label, historical data SHA-256s, complete state-memory checkpoints, and the *actual* current-session source availability. Change the live generator only after tests and independent authorization.

The planned Top-Level Investment Research Orchestrator will eventually own schedule/retries and artifact handoffs. The currently installed V3/TimesFM scheduler must not be changed to trigger V4 while `authorizes_live_v4_issuance=false`.

## Validation milestones

- **B0**: source inventory (4/4 tests); status REQUIRES_INDEPENDENT_POINT_IN_TIME_VERIFICATION.
- **B1**: independent refreshed market + Treasury files hashed and documented.
- **B2**: corporate-action and nominal/real yield timing audits accepted.
- **B3**: reproducible memory replay, no lookahead, identical checkpoint/full replay.
- **B4**: validated same-session, genuine prospective evidence and read-only V4 computation.
- **B5**: explicit authorization for first new V4 state; independently written state and evaluation five sessions later.

**Decision:** Historical sufficiency does not establish publication-time correctness. Until B1–B4 pass, **V4 issuance stays prohibited**; existing V3/TimesFM experiment issuance and scoring continue independently.
