# V4-BOOT-001 — local reconstruction implementation report

**Status:** SOFTWARE RECONSTRUCTION / RESEARCH NOT ACCEPTED.

This package was rebuilt from the GitHub V4 bootstrap-gate specification after the uncommitted Codex Cloud changes were unavailable. It is **not** a recovered copy of the original Codex implementation and does not claim the original 29 test cases were reproduced.

## Added files

- `scripts/build_v4_bootstrap_manifest.py` — source-only manifest builder, local exact-byte archive, `--metadata` read-only check, `--output` write-once archive, `--verify-existing` integrity verification.
- `tests/test_v4_bootstrap_manifest.py` — deterministic synthetic tests without real market data or brokerage access.
- `docs/V4_BOOTSTRAP_SOURCE_MANIFEST.md` — source declaration schema, safety restrictions, and usage.
- `reports/v4_bootstrap/SHA256SUMS` — SHA-256 digests of the above source files in this reconstruction package, not historical data.

## Validation performed outside user's Windows checkout

- Synthetic tests: **32 passed** (Python 3.13 in sandbox; no genuine market input).
- Patch application: **passed** `git apply --check` and application in a separate throwaway Git repo; synthetic tests also passed after patch application.
- Full TMF repository regression suite: **NOT RUN for this reconstruction**. Latest user-supplied isolated checkout baseline was Python 3.11.4, `pip check` clean, 534 passed, 2 FileNotFoundError due to ignored PAYEMS feature datasets, 7 skipped, 2 warnings.
- B1/B2: **NOT_ESTABLISHED** — independent original market/Treasury sources, acquisition metadata, adjustments, source-parity, publication timing and vintage evidence unverified.
- B3/B4/B5: **BLOCKED** — no risk-memory replay, live integration, or prospective state issuance.

## Important scope limits

An archived source hash is not vendor provenance, a real acquisition timestamp, an audited price-adjustment policy, or observation-level point-in-time publication evidence. `retrieval_evidence_reference` is preserved as unverified declaration text. This source-only schema is distinct from the final bootstrap manifest that must eventually include independently audited Risk Governor replay state. Never interpret this package as trading permission.

## Local next steps

1. In `C:/Dev/TMF-V4-Research`, inspect `git status --short`, apply the supplied patch using `git apply --check` followed by `git apply`, and run `python -m pytest tests/test_v4_bootstrap_manifest.py -q` with the local Python interpreter.
2. Rerun `python -m pytest -q` and document every failure without weakening existing tests. The two missing PAYEMS/inflation feature CSVs must not be fabricated.
3. Review the new code and documentation before committing. The package was not applied to or tested against the user's own private Windows checkout.
4. Qualify historical source bytes and metadata independently before attempting actual source inventory with `--metadata`. Until qualified, do not activate V4 issuance.

## Windows validation — 2026-11-02

- Environment: Windows, Python 3.11.4.
- Bootstrap suite: 31 passed, 1 skipped, 0 failed.
- One test skipped: symlinks unavailable in this environment.
- Synthetic CSV fixtures use exact UTF-8 bytes to avoid Windows newline translation.
- B1/B2: NOT_ESTABLISHED.
- B3–B5: BLOCKED.
- No V4 prospective issuance or trading authorized.

## Full Windows regression — 2026-11-02

Command: python -m pytest -q --tb=short

Results:
- 565 passed
- 2 failed
- 8 skipped
- 2 deprecation warnings

Both failures are FileNotFoundError exceptions involving:
- data/features/TMF_1d_technical_PAYEMS_inflation_features.csv
- data/features/TMF_1d_technical_PAYEMS_features.csv

These failures also occurred in the pre-implementation baseline.
No new regression was observed.

Full-suite acceptance: BLOCKED — missing existing datasets.
Bootstrap-specific validation: 31 passed, 1 skipped.

B1/B2: NOT_ESTABLISHED.
B3–B5: BLOCKED.
Prospective V4 issuance: NOT AUTHORIZED.
