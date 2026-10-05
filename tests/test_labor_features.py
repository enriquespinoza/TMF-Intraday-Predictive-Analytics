import pandas as pd

from src.features.labor_features import (
    build_payroll_event_features,
    add_payroll_standardized_features,
    merge_payroll_features_to_market,
    payroll_revision_features,
    payroll_state_as_of,
)


def make_payems_vintages() -> pd.DataFrame:
    """
    Synthetic PAYEMS history modeled after the real
    2021-10-08, 2021-11-05, and 2021-12-03 releases.
    """

    return pd.DataFrame(
        {
            "series_name": [
                "nonfarm_payrolls",
                "nonfarm_payrolls",
                "nonfarm_payrolls",
                "nonfarm_payrolls",
                "nonfarm_payrolls",
                "nonfarm_payrolls",
            ],
            "series_id": [
                "PAYEMS",
                "PAYEMS",
                "PAYEMS",
                "PAYEMS",
                "PAYEMS",
                "PAYEMS",
            ],
            "reference_date": [
                "2021-09-01",
                "2021-09-01",
                "2021-10-01",
                "2021-09-01",
                "2021-10-01",
                "2021-11-01",
            ],
            "vintage_date": [
                "2021-10-08",
                "2021-11-05",
                "2021-11-05",
                "2021-12-03",
                "2021-12-03",
                "2021-12-03",
            ],
            "value_as_known": [
                147553,
                147788,
                148319,
                147855,
                148401,
                148611,
            ],
        }
    )


def test_payroll_state_before_november_release():
    vintages = make_payems_vintages()

    result = payroll_state_as_of(
        vintage_data=vintages,
        as_of_date="2021-11-04",
    )

    assert len(result) == 1

    september = result.iloc[0]

    assert (
        september["reference_date"]
        == pd.Timestamp("2021-09-01")
    )

    assert september["value_as_known"] == 147553

    assert pd.isna(
        september["payroll_change_1m"]
    )


def test_payroll_state_after_november_release():
    vintages = make_payems_vintages()

    result = payroll_state_as_of(
        vintage_data=vintages,
        as_of_date="2021-11-05",
    )

    assert len(result) == 2

    september = result.loc[
        result["reference_date"]
        == pd.Timestamp("2021-09-01")
    ].iloc[0]

    october = result.loc[
        result["reference_date"]
        == pd.Timestamp("2021-10-01")
    ].iloc[0]

    assert september["value_as_known"] == 147788

    assert october["value_as_known"] == 148319

    assert october["payroll_change_1m"] == 531


def test_payroll_revision_on_november_release():
    vintages = make_payems_vintages()

    result = payroll_revision_features(
        vintage_data=vintages,
        event_date="2021-11-05",
    )

    assert result["payroll_revision_1m"] == 235
    assert result["payroll_revision_2m"] == 0
    assert result["payroll_revision_total"] == 235


def test_build_payroll_event_features_november_release():
    vintages = make_payems_vintages()

    result = build_payroll_event_features(
        vintage_data=vintages,
        event_date="2021-11-05",
    )

    assert (
        result["reference_date"]
        == pd.Timestamp("2021-10-01")
    )

    assert result["payroll_level"] == 148319
    assert result["payroll_change_1m"] == 531

    assert result["payroll_revision_1m"] == 235
    assert result["payroll_revision_2m"] == 0
    assert result["payroll_revision_total"] == 235

def test_multiple_initial_releases_same_event_not_counted_as_revisions():
    """
    Simulate the December 16, 2025 shutdown release.

    October and November are both initial PAYEMS observations
    on the same event date. Only September is a revision.
    """

    vintages = pd.DataFrame(
        {
            "series_name": [
                "nonfarm_payrolls",
                "nonfarm_payrolls",
                "nonfarm_payrolls",
                "nonfarm_payrolls",
            ],
            "series_id": [
                "PAYEMS",
                "PAYEMS",
                "PAYEMS",
                "PAYEMS",
            ],
            "reference_date": [
                "2025-09-01",
                "2025-09-01",
                "2025-10-01",
                "2025-11-01",
            ],
            "vintage_date": [
                "2025-11-20",
                "2025-12-16",
                "2025-12-16",
                "2025-12-16",
            ],
            "value_as_known": [
                158548,
                158500,
                158408,
                158449,
            ],
        }
    )

    revisions = payroll_revision_features(
        vintage_data=vintages,
        event_date="2025-12-16",
    )

    features = build_payroll_event_features(
        vintage_data=vintages,
        event_date="2025-12-16",
    )

    #
    # September revision:
    # 158500 - 158548 = -48
    #
    assert revisions["payroll_revision_1m"] == -48
    assert revisions["payroll_revision_2m"] == 0
    assert revisions["payroll_revision_total"] == -48

    #
    # October and November are both initial observations.
    #
    assert features["payroll_level"] == 158449

    #
    # November - October:
    # 158449 - 158408 = +41
    #
    assert features["payroll_change_1m"] == 41

def test_payroll_features_backward_merge_to_market():
    market = pd.DataFrame(
        {
            "timestamp": [
                "2021-10-07",
                "2021-11-04",
                "2021-11-05",
                "2021-11-08",
            ],
            "close": [
                30.0,
                31.0,
                32.0,
                33.0,
            ],
        }
    )

    payroll = pd.DataFrame(
        {
            "event_name": [
                "employment_situation",
                "employment_situation",
            ],
            "event_reference_date": [
                "2021-09-01",
                "2021-10-01",
            ],
            "event_release_date": [
                "2021-10-08",
                "2021-11-05",
            ],
            "event_release_time": [
                "08:30",
                "08:30",
            ],
            "event_release_timezone": [
                "America/New_York",
                "America/New_York",
            ],
            "reference_date": [
                "2021-09-01",
                "2021-10-01",
            ],
            "payroll_level": [
                147553,
                148319,
            ],
            "payroll_change_1m": [
                float("nan"),
                531,
            ],
            "payroll_revision_1m": [
                0,
                235,
            ],
            "payroll_revision_2m": [
                0,
                0,
            ],
            "payroll_revision_total": [
                0,
                235,
            ],
        }
    )

    result = merge_payroll_features_to_market(
        market_data=market,
        payroll_features=payroll,
    )

    oct_7 = result.loc[
        result["market_date"]
        == pd.Timestamp("2021-10-07")
    ].iloc[0]

    nov_4 = result.loc[
        result["market_date"]
        == pd.Timestamp("2021-11-04")
    ].iloc[0]

    nov_5 = result.loc[
        result["market_date"]
        == pd.Timestamp("2021-11-05")
    ].iloc[0]

    nov_8 = result.loc[
        result["market_date"]
        == pd.Timestamp("2021-11-08")
    ].iloc[0]

    # Before the first PAYEMS release, no payroll state exists.
    assert pd.isna(
        oct_7["event_release_date"]
    )

    # November 4 still sees the October 8 release.
    assert (
        nov_4["event_release_date"]
        == pd.Timestamp("2021-10-08")
    )

    assert nov_4["payroll_level"] == 147553
    assert nov_4["payroll_release_day"] == 0

    # November 5 receives the new 08:30 release.
    assert (
        nov_5["event_release_date"]
        == pd.Timestamp("2021-11-05")
    )

    assert nov_5["payroll_level"] == 148319
    assert nov_5["payroll_change_1m"] == 531
    assert nov_5["payroll_revision_1m"] == 235
    assert nov_5["payroll_revision_total"] == 235
    assert nov_5["days_since_payroll_release"] == 0
    assert nov_5["payroll_release_day"] == 1

    # November 8 carries the November 5 state forward.
    assert (
        nov_8["event_release_date"]
        == pd.Timestamp("2021-11-05")
    )

    assert nov_8["payroll_change_1m"] == 531
    assert nov_8["payroll_revision_total"] == 235
    assert nov_8["days_since_payroll_release"] == 3
    assert nov_8["payroll_release_day"] == 0

    # No future information.
    available = result[
        "event_release_date"
    ].notna()

    assert (
        result.loc[
            available,
            "event_release_date",
        ]
        <= result.loc[
            available,
            "market_date",
        ]
    ).all()

def test_broad_revision_excluded_from_revision_zscore_distribution():
    events = pd.DataFrame(
        {
            "event_release_date": [
                "2022-01-01",
                "2022-02-01",
                "2022-03-01",
                "2022-04-01",
            ],
            "payroll_change_1m": [
                100.0,
                110.0,
                120.0,
                130.0,
            ],
            "payroll_revision_total": [
                10.0,
                20.0,
                1000.0,
                30.0,
            ],
            "payroll_broad_revision_event": [
                0,
                0,
                1,
                0,
            ],
        }
    )

    result = add_payroll_standardized_features(
        event_features=events,
        min_history=2,
    )

    broad = result.iloc[2]
    later_regular = result.iloc[3]

    #
    # Broad revision magnitude is preserved.
    #
    assert broad["payroll_revision_total"] == 1000

    assert (
        broad["payroll_broad_revision_event"]
        == 1
    )

    #
    # But it is not assigned a normal-revision z-score.
    #
    assert pd.isna(
        broad["payroll_revision_z"]
    )

    #
    # The later regular revision is standardized against
    # ONLY the prior regular revisions:
    #
    # prior regular values = [10, 20]
    # mean = 15
    # sample std = ~7.0711
    #
    # (30 - 15) / 7.0711 = ~2.1213
    #
    assert abs(
        later_regular["payroll_revision_z"]
        - 2.1213203435596424
    ) < 1e-9

def test_payroll_zscores_use_prior_events_only():
    events = pd.DataFrame(
        {
            "event_release_date": [
                "2022-01-01",
                "2022-02-01",
                "2022-03-01",
            ],
            "payroll_change_1m": [
                100.0,
                110.0,
                120.0,
            ],
            "payroll_revision_total": [
                10.0,
                20.0,
                30.0,
            ],
            "payroll_broad_revision_event": [
                0,
                0,
                0,
            ],
        }
    )

    result = add_payroll_standardized_features(
        event_features=events,
        min_history=2,
    )

    third = result.iloc[2]

    #
    # Prior payroll changes are [100, 110].
    #
    assert abs(
        third["payroll_change_z"]
        - 2.1213203435596424
    ) < 1e-9

    #
    # Prior revisions are [10, 20].
    #
    assert abs(
        third["payroll_revision_z"]
        - 2.1213203435596424
    ) < 1e-9