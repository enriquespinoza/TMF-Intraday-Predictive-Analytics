"""Deterministic TimesFM 2.5 adapter and runner tests; no weights or API calls."""

import numpy as np
import pandas as pd
import pytest

from src.models import timesfm25_forecaster as adapter
from scripts import predict_timesfm25_schwab as runner


def make_prices(n=60):
    return pd.DataFrame({
        "market_date": pd.bdate_range("2026-01-02", periods=n),
        "close": np.linspace(22.0, 28.0, n),
    })


def test_context_uses_origin_and_caps_history():
    prices = make_prices()
    origin = str(prices["market_date"].iloc[-1].date())
    history = adapter.closing_context(prices, origin, max_context=40)
    assert history.dtype == np.float32
    assert len(history) == 40
    assert history[-1] == pytest.approx(28.0)


def test_context_rejects_future_endpoint():
    prices = make_prices()
    with pytest.raises(ValueError, match="endpoint"):
        adapter.closing_context(prices, "2026-02-01")


def test_context_rejects_duplicates():
    prices = make_prices()
    prices.loc[15, "market_date"] = prices.loc[14, "market_date"]
    with pytest.raises(ValueError, match="Duplicate"):
        adapter.closing_context(prices, "2026-03-26")


def test_context_rejects_nonpositive():
    prices = make_prices()
    prices.loc[12, "close"] = 0
    with pytest.raises(ValueError, match="Nonpositive"):
        adapter.closing_context(
            prices, str(prices["market_date"].iloc[-1].date())
        )


def test_revision_must_be_immutable():
    assert adapter.validate_revision("A" * 40) == "a" * 40
    for bad in ("main", "v2.5", "", None, "f" * 39):
        with pytest.raises(ValueError, match="40-character"):
            adapter.validate_revision(bad)


class FakeTimesFM:
    def __init__(self, bad=False):
        self.bad = bad

    def forecast(self, *, horizon, inputs):
        assert horizon == 5
        assert len(inputs) == 1
        point = np.full((1, 5), 30.0)
        quantiles = np.empty((1, 5, 10))
        quantiles[:, :, 0] = 30.1
        quantiles[:, :, 1:] = np.linspace(28.0, 32.0, 9)
        if self.bad:
            quantiles[0, 0, 9] = 20
        return point, quantiles


def test_forecast_shapes_and_quantiles():
    result = adapter.forecast_price_path(
        FakeTimesFM(), np.linspace(25, 28, 40)
    )
    assert len(result["point"]) == 5
    assert result["q10"][0] == pytest.approx(28)
    assert result["median"][0] == pytest.approx(30)
    assert result["q90"][0] == pytest.approx(32)


def test_quantile_crossing_rejected():
    with pytest.raises(ValueError, match="crossing"):
        adapter.forecast_price_path(
            FakeTimesFM(bad=True), np.linspace(25, 28, 40)
        )


def test_runner_check_only_never_downloads_weights(
    tmp_path, monkeypatch
):
    snapshot_dir = tmp_path / "schwab"
    snapshot_dir.mkdir()
    daily = snapshot_dir / "TMF_1d_test.csv"
    daily.write_text("synthetic")

    monkeypatch.setattr(runner, "SOURCE_DIR", snapshot_dir)
    monkeypatch.setattr(
        runner, "validate_snapshot_pair",
        lambda path: {"market_date": "2026-10-21"},
    )
    monkeypatch.setattr(
        runner, "require_issuance_window",
        lambda origin: None,
    )
    monkeypatch.setattr(
        runner, "load_timesfm25_model",
        lambda revision: pytest.fail("Unexpected model download"),
    )

    result = runner.run(daily, execute=False)
    assert result["status"] == "CHECK_ONLY"
    assert result["order_generated"] is False


def test_runner_rejects_snapshot_outside_directory(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "SOURCE_DIR", tmp_path / "schwab")
    invalid = tmp_path / "outside.csv"
    invalid.write_text("synthetic")
    with pytest.raises(ValueError, match="snapshot directory"):
        runner.run(invalid)


def test_runner_rejects_unpinned_weights_before_loading(
    tmp_path, monkeypatch
):
    snapshot_dir = tmp_path / "schwab"
    snapshot_dir.mkdir()
    daily = snapshot_dir / "TMF_1d_test.csv"
    daily.write_text("synthetic")
    monkeypatch.setattr(runner, "SOURCE_DIR", snapshot_dir)
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(
        runner, "validate_snapshot_pair",
        lambda path: {"market_date": "2026-10-21"},
    )
    monkeypatch.setattr(
        runner, "require_issuance_window",
        lambda origin: None,
    )
    monkeypatch.delenv("TIMESFM25_CHECKPOINT_SHA", raising=False)
    with pytest.raises(ValueError, match="40-character"):
        runner.run(daily, execute=True)
