"""Offline-testable TimesFM 2.5 adapter for TMF daily close forecasts.

This module intentionally does not import TimesFM/PyTorch at module import time.
V3_FINAL remains independent and unmodified.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
MODEL_NAME = "TIMESFM_2P5_200M"
CONTEXT_LENGTH = 512
MIN_CONTEXT = 32
HORIZON = 5


def validate_revision(revision: str) -> str:
    """Require an immutable Hugging Face commit SHA, not a moving branch."""
    if not isinstance(revision, str) or not re.fullmatch(
        r"[0-9a-fA-F]{40}", revision
    ):
        raise ValueError("TimesFM checkpoint revision must be a 40-character SHA")
    return revision.lower()


def closing_context(prices: pd.DataFrame, origin, max_context=CONTEXT_LENGTH):
    """Extract only closes available through the exact forecast origin."""
    if not {"market_date", "close"}.issubset(prices.columns):
        raise ValueError("Historical context lacks market_date or close")
    if max_context < MIN_CONTEXT or max_context > 16384:
        raise ValueError("Invalid context size")

    dates = pd.to_datetime(prices["market_date"], errors="raise").dt.date
    if dates.duplicated().any():
        raise ValueError("Duplicate market dates in context")
    if not dates.is_monotonic_increasing:
        raise ValueError("Context dates must be chronological")
    if len(dates) < MIN_CONTEXT:
        raise ValueError("Insufficient historical closes")
    if str(dates.iloc[-1]) != str(origin):
        raise ValueError("Context endpoint differs from validated origin")

    closes = pd.to_numeric(prices["close"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(closes).all() or (closes <= 0).any():
        raise ValueError("Nonpositive or nonfinite TMF close")

    return closes[-max_context:].astype(np.float32)


def load_timesfm25_model(revision: str):
    """Load the selected checkpoint lazily; no model download during tests."""
    revision = validate_revision(revision)
    try:
        import timesfm
    except ImportError as exc:
        raise RuntimeError(
            "Install timesfm[torch]==2.0.2 in the separate TimesFM environment"
        ) from exc

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        MODEL_ID, revision=revision, torch_compile=False
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=CONTEXT_LENGTH,
            max_horizon=128,
            per_core_batch_size=1,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=False,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


def forecast_price_path(model, context, horizon=HORIZON):
    """Return checked median and 10/90 quantile paths; not P(price rises)."""
    context = np.asarray(context, dtype=np.float32)
    if horizon != HORIZON or context.ndim != 1 or len(context) < MIN_CONTEXT:
        raise ValueError("Invalid TimesFM context or horizon")
    if not np.isfinite(context).all() or (context <= 0).any():
        raise ValueError("Invalid TimesFM price context")

    point, quantiles = model.forecast(horizon=horizon, inputs=[context])
    point = np.asarray(point, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)

    # TimesFM 2.5: quantile index 0 = mean; 1..9 = q10..q90.
    if point.shape != (1, horizon) or quantiles.shape != (1, horizon, 10):
        raise ValueError("Unexpected TimesFM 2.5 forecast shapes")
    if not np.isfinite(point).all() or not np.isfinite(quantiles).all():
        raise ValueError("Nonfinite TimesFM forecasts")
    if (point <= 0).any() or (quantiles <= 0).any():
        raise ValueError("Nonpositive TimesFM price forecasts")
    if np.any(np.diff(quantiles[0, :, 1:], axis=1) < -1e-6):
        raise ValueError("TimesFM quantile crossing")

    return {
        "point": point[0].tolist(),
        "median": quantiles[0, :, 5].tolist(),
        "q10": quantiles[0, :, 1].tolist(),
        "q90": quantiles[0, :, 9].tolist(),
    }
