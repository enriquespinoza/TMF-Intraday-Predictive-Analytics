# TimesFM 2.5 — TMF five-session research lane

This feature branch adds a **separate**, experimental univariate forecast of TMF's
next five NYSE closing prices. It does not replace or invoke the frozen
`V3_FINAL` directional predictor. It must not be connected to trade execution.

## Environment (Windows or Ubuntu)

Use a dedicated Python environment for TimesFM/PyTorch: substantial framework
dependencies may conflict with the existing validated V3 environment.
Install and resolve platform-specific packages first:

```bash
python -m pip install -r requirements-timesfm25.in
python -m pip check
python -m pytest -q tests/test_timesfm25_forecaster.py
```

TimesFM 2.5 inference is based on `google/timesfm-2.5-200m-pytorch`.
Download weights only after checking the model card/license and recording
the immutable **40-character Hugging Face model repository commit SHA**
as `TIMESFM25_CHECKPOINT_SHA`. Never replace this with `main` or an
unverified revision. Cache pretrained weights on persistent storage.

## Workflow

1. Use the existing Schwab downloader to create an immutable, post-close
   daily + five-minute snapshot pair. Never reuse incomplete snapshots.
2. Validate the NYSE session and that the next session has **not opened**.
3. To validate without loading model weights:

   ```bash
   python -m scripts.predict_timesfm25_schwab --daily data/raw/schwab/TMF_1d_<snapshot-id>.csv --check
   ```

4. Only in a valid prospective window, after passing full tests:

   ```bash
   export TIMESFM25_CHECKPOINT_SHA=<40-character-model-revision-sha>
   python -m scripts.predict_timesfm25_schwab --daily data/raw/schwab/TMF_1d_<snapshot-id>.csv --run
   ```

The explicit `--run` flag is required to write a forecast. The output is
an exclusive-create JSON record under `data/forward_test/timesfm25/`, not
the V3 forward-test ledger. Existing records cannot be silently overwritten.

## Interpretation and evaluation

The median and q10/q90 are **model forecasts** of future TMF closing prices.
They are not calibrated probabilities of a positive five-day return and do
not represent guaranteed confidence intervals. Compare five-session
median absolute error, final price error, directional hit rate, and interval
coverage against a no-change baseline in a walk-forward evaluation.
Keep overlapping five-session outcomes and any retrospective runs
separate from timestamped prospective predictions.

## Proxmox limits

No automatic timer, OAuth migration, or portfolio execution wiring is
authorized by this experimental module. Ensure a single writer per model
and session, off-host backups, session-locking, Linux dependency validation,
and process-failure notifications before unattended use.

## Immutable run manifests and adoption of Forecast 001

New forecasts have per-origin locks and a separate exclusive-write manifest.
An interrupted forecast without a manifest is an `INCOMPLETE_RUN` error,
not an invitation to regenerate or overwrite. Do not delete stale locks or
forecast files without inspecting and preserving them.

Forecast 001 (origin 2026-10-21) predates the manifest implementation.
Verify original snapshot and history hashes first. The following will
attest the existing record, **without modifying its forecast bytes**:

```bash
python -m scripts.audit_timesfm25_run --session 2026-10-21 --adopt-legacy
python -m scripts.audit_timesfm25_run --session 2026-10-21 --audit
```

The manifest records `AUDITED_LEGACY`: it does not claim the original
forecast had a manifest at issuance or prove independent timestamp provenance.
Only run adoption once; subsequent checks must use `--audit`.
Never check credentials or OAuth token files into Git.
