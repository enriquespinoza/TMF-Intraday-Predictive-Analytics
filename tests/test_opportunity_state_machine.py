import pytest

from src.signals.opportunity_state_machine import (
    OpportunityInputsV1,
    OpportunityPolicyV1,
    OpportunityState,
    evaluate_opportunity_v1,
    load_policy_v1,
)


POLICY = OpportunityPolicyV1(
    technical_strong_setup_threshold=(
        43.769841
    ),

    treasury_extreme_confirmation_threshold=(
        79.728836
    ),
)


def test_frozen_policy_loads():
    policy = load_policy_v1()

    assert (
        policy.technical_strong_setup_threshold
        == pytest.approx(
            43.769841,
            abs=1e-6,
        )
    )

    assert (
        policy.treasury_extreme_confirmation_threshold
        == pytest.approx(
            79.728836,
            abs=1e-6,
        )
    )


def test_risk_block_has_veto_authority():
    decision = evaluate_opportunity_v1(
        OpportunityInputsV1(
            risk_entry_blocked=True,

            technical_opportunity_score=90.0,

            treasury_confirmation_score=50.0,

            macro_regime_score=100.0,

            market_confirmation_score=100.0,
        ),

        policy=POLICY,
    )

    assert (
        decision.state
        == OpportunityState.NO_TRADE
    )

    assert (
        decision.entry_candidate
        is False
    )


def test_weak_technical_setup_is_watch():
    decision = evaluate_opportunity_v1(
        OpportunityInputsV1(
            risk_entry_blocked=False,

            technical_opportunity_score=40.0,

            treasury_confirmation_score=50.0,
        ),

        policy=POLICY,
    )

    assert (
        decision.state
        == OpportunityState.WATCH
    )

    assert (
        decision.technical_strong_setup
        is False
    )


def test_strong_technical_non_extreme_treasury_is_entry_candidate():
    decision = evaluate_opportunity_v1(
        OpportunityInputsV1(
            risk_entry_blocked=False,

            technical_opportunity_score=60.0,

            treasury_confirmation_score=70.0,
        ),

        policy=POLICY,
    )

    assert (
        decision.state
        == OpportunityState.ENTRY_CANDIDATE
    )

    assert (
        decision.entry_candidate
        is True
    )

    assert (
        decision.technical_strong_setup
        is True
    )

    assert (
        decision.treasury_extreme_confirmation
        is False
    )


def test_strong_technical_extreme_treasury_is_do_not_chase():
    decision = evaluate_opportunity_v1(
        OpportunityInputsV1(
            risk_entry_blocked=False,

            technical_opportunity_score=60.0,

            treasury_confirmation_score=90.0,
        ),

        policy=POLICY,
    )

    assert (
        decision.state
        == OpportunityState.ARMED_DO_NOT_CHASE
    )

    assert (
        decision.entry_candidate
        is False
    )

    assert (
        decision.treasury_extreme_confirmation
        is True
    )


def test_technical_threshold_is_inclusive():
    decision = evaluate_opportunity_v1(
        OpportunityInputsV1(
            risk_entry_blocked=False,

            technical_opportunity_score=(
                POLICY
                .technical_strong_setup_threshold
            ),

            treasury_confirmation_score=50.0,
        ),

        policy=POLICY,
    )

    assert (
        decision.state
        == OpportunityState.ENTRY_CANDIDATE
    )


def test_treasury_threshold_is_inclusive():
    decision = evaluate_opportunity_v1(
        OpportunityInputsV1(
            risk_entry_blocked=False,

            technical_opportunity_score=60.0,

            treasury_confirmation_score=(
                POLICY
                .treasury_extreme_confirmation_threshold
            ),
        ),

        policy=POLICY,
    )

    assert (
        decision.state
        == OpportunityState.ARMED_DO_NOT_CHASE
    )


def test_macro_does_not_change_state():
    low_macro = evaluate_opportunity_v1(
        OpportunityInputsV1(
            risk_entry_blocked=False,

            technical_opportunity_score=60.0,

            treasury_confirmation_score=60.0,

            macro_regime_score=0.0,
        ),

        policy=POLICY,
    )

    high_macro = evaluate_opportunity_v1(
        OpportunityInputsV1(
            risk_entry_blocked=False,

            technical_opportunity_score=60.0,

            treasury_confirmation_score=60.0,

            macro_regime_score=100.0,
        ),

        policy=POLICY,
    )

    assert (
        low_macro.state
        == high_macro.state
        == OpportunityState.ENTRY_CANDIDATE
    )


def test_market_confirmation_does_not_change_state():
    low_market = evaluate_opportunity_v1(
        OpportunityInputsV1(
            risk_entry_blocked=False,

            technical_opportunity_score=60.0,

            treasury_confirmation_score=60.0,

            market_confirmation_score=0.0,
        ),

        policy=POLICY,
    )

    high_market = evaluate_opportunity_v1(
        OpportunityInputsV1(
            risk_entry_blocked=False,

            technical_opportunity_score=60.0,

            treasury_confirmation_score=60.0,

            market_confirmation_score=100.0,
        ),

        policy=POLICY,
    )

    assert (
        low_market.state
        == high_market.state
        == OpportunityState.ENTRY_CANDIDATE
    )


def test_missing_technical_score_is_no_trade():
    decision = evaluate_opportunity_v1(
        OpportunityInputsV1(
            risk_entry_blocked=False,

            technical_opportunity_score=None,

            treasury_confirmation_score=50.0,
        ),

        policy=POLICY,
    )

    assert (
        decision.state
        == OpportunityState.NO_TRADE
    )


def test_missing_treasury_score_is_no_trade():
    decision = evaluate_opportunity_v1(
        OpportunityInputsV1(
            risk_entry_blocked=False,

            technical_opportunity_score=60.0,

            treasury_confirmation_score=None,
        ),

        policy=POLICY,
    )

    assert (
        decision.state
        == OpportunityState.NO_TRADE
    )


def test_invalid_score_rejected():
    with pytest.raises(
        ValueError,
        match="between 0 and 100",
    ):
        evaluate_opportunity_v1(
            OpportunityInputsV1(
                risk_entry_blocked=False,

                technical_opportunity_score=101.0,

                treasury_confirmation_score=50.0,
            ),

            policy=POLICY,
        )