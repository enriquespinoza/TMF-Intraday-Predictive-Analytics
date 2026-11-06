# Offline CI audit integration (AUDIT-004)

The workflow consumes the shared `research-ci-audit-v1` runner at exact top-level repository commit `0532fac27d99b408641ee7ea63a6fac8f6ad066d`. The runner and evidence schema are maintained centrally instead of duplicated in the TMF repository.

The workflow triggers on PRs, pushes to main, weekdays and manual dispatch. It runs existing offline pytest tests with network access disabled by pytest-socket, retaining JUnit results, sanitized (after review) pytest logs, a JSON summary and SHA-256 evidence hashes. Failed/missing/empty JUnit reports or nonzero pytest outcomes fail the job. No API keys or raw licensed market data are passed to hosted runners.

**This is software-health validation, not TMF empirical research confirmation.** The summary always sets `classification=SOFTWARE_TEST_ONLY`, `research_acceptance=NOT_ESTABLISHED`, `independent_review=NOT_PERFORMED` and `broker_submission_permitted=false`. TMF full OHLCV equivalence, fractional-volume anomaly resolution, platform execution attestation, V3/V4 forecasting and prospective predictive performance remain independently gated.

Hosted audit artifacts expire after 90 days and do not constitute a permanent signed evidence archive. Later work will implement missed-run detection, encrypted long-term audit storage, strict dependency locking, independent data-source review and release approval rules. A shared-runner update must be an explicit reviewed commit-pin change.
