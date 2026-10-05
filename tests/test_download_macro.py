"""Tests for raw macroeconomic data ingestion."""

from __future__ import annotations

import pandas as pd

from config.macro_series import MACRO_SERIES
from src.data.download_macro import (
    download_fred_series,
    save_raw_series,
)


def test_macro_registry_contains_expected_series():
    """Registry should contain the frozen V3 raw-series universe."""

    assert len(MACRO_SERIES) == 24

    expected = {
        "nonfarm_payrolls",
        "unemployment_rate",
        "quits_rate",
        "retail_sales",
        "cpi",
        "core_cpi",
        "breakeven_10y",
        "real_yield_10y",
        "real_yield_30y",
    }

    assert expected.issubset(
        MACRO_SERIES.keys()
    )


def test_macro_registry_required_fields():
    """Every configured series should contain required metadata."""

    required = {
        "series_id",
        "name",
        "category",
        "source",
        "frequency",
        "units",
        "transformation",
        "requires_release_alignment",
        "requires_vintage_tracking",
    }

    for name, config in MACRO_SERIES.items():
        missing = required - config.keys()

        assert not missing, (
            f"{name} missing fields: {missing}"
        )


def test_daily_market_series_do_not_require_release_alignment():
    """Market-observed daily series should use observation-date logic."""

    market_series = [
        "breakeven_5y",
        "breakeven_10y",
        "forward_inflation_5y5y",
        "real_yield_5y",
        "real_yield_10y",
        "real_yield_20y",
        "real_yield_30y",
    ]

    for name in market_series:
        assert (
            MACRO_SERIES[name][
                "requires_release_alignment"
            ]
            is False
        )


def test_monthly_macro_series_require_release_alignment():
    """Economic releases must not be aligned by reference date alone."""

    release_series = [
        "nonfarm_payrolls",
        "unemployment_rate",
        "job_openings",
        "quits_rate",
        "retail_sales",
        "industrial_production",
        "cpi",
        "core_cpi",
        "pce_price_index",
        "core_pce_price_index",
    ]

    for name in release_series:
        assert (
            MACRO_SERIES[name][
                "requires_release_alignment"
            ]
            is True
        )


def test_save_raw_series(
    tmp_path,
    monkeypatch,
):
    """Raw saver should preserve observation and metadata fields."""

    import src.data.download_macro as module

    monkeypatch.setattr(
        module,
        "RAW_MACRO_DIR",
        tmp_path,
    )

    data = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-02-01",
                ]
            ),
            "value": [
                100.0,
                101.0,
            ],
        }
    )

    config = MACRO_SERIES[
        "retail_sales"
    ]

    output_path = save_raw_series(
        name="retail_sales",
        config=config,
        data=data,
    )

    saved = pd.read_csv(
        output_path
    )

    expected_columns = [
        "reference_date",
        "value",
        "series_name",
        "series_id",
        "category",
        "source",
        "frequency",
        "units",
    ]

    assert list(saved.columns) == expected_columns
    assert len(saved) == 2
    assert (
        saved["series_id"].iloc[0]
        == "RSAFS"
    )


def test_download_fred_series_schema(
    monkeypatch,
):
    """Downloader should normalize FRED output."""

    import src.data.download_macro as module

    sample = pd.DataFrame(
        {
            "TEST": [
                1.0,
                2.0,
            ]
        },
        index=pd.to_datetime(
            [
                "2025-01-01",
                "2025-02-01",
            ]
        ),
    )

    sample.index.name = "DATE"

    def fake_data_reader(
        series_id,
        source,
        start,
    ):
        assert series_id == "TEST"
        assert source == "fred"

        return sample.copy()

    monkeypatch.setattr(
        module.web,
        "DataReader",
        fake_data_reader,
    )

    result = download_fred_series(
        "TEST"
    )

    assert list(result.columns) == [
        "reference_date",
        "value",
    ]

    assert len(result) == 2
    assert result["value"].tolist() == [
        1.0,
        2.0,
    ]

    assert pd.api.types.is_datetime64_any_dtype(
        result["reference_date"]
    )