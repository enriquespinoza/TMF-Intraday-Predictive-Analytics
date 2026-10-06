import pytest

from src.execution.position_lifecycle import (
    ExecutionAction,
    ExecutionInputsV1,
    ExecutionPolicyV1,
    PositionMemoryV1,
    apply_execution_fill,
    calculate_drawdown_from_entry_pct,
    evaluate_execution_v1,
    load_execution_policy_v1,
    transaction_cost_fraction,
)

from src.signals.opportunity_state_machine import (
    OpportunityState,
)


POLICY = ExecutionPolicyV1(
    entry_target_exposure=1.0,
    transaction_cost_bps_per_unit_turnover=5.0,
    execution_timing="next_session_open",
    drawdown_anchor=(
        "initial_position_cycle_entry_price"
    ),
)


def flat_position():
    return PositionMemoryV1(
        current_exposure=0.0,
        entry_anchor_price=None,
    )


def full_position():
    return PositionMemoryV1(
        current_exposure=1.0,
        entry_anchor_price=100.0,
    )


def half_position():
    return PositionMemoryV1(
        current_exposure=0.5,
        entry_anchor_price=100.0,
    )


def test_frozen_policy_loads():
    policy = load_execution_policy_v1()

    assert (
        policy.entry_target_exposure
        == pytest.approx(1.0)
    )

    assert (
        policy.transaction_cost_bps_per_unit_turnover
        == pytest.approx(5.0)
    )

    assert (
        policy.execution_timing
        == "next_session_open"
    )


def test_entry_candidate_opens_position():
    decision = evaluate_execution_v1(
        ExecutionInputsV1(
            opportunity_state=(
                OpportunityState.ENTRY_CANDIDATE
            ),
            risk_entry_blocked=False,
            risk_max_exposure=1.0,
        ),
        flat_position(),
        POLICY,
    )

    assert (
        decision.action
        == ExecutionAction.OPEN_POSITION
    )

    assert (
        decision.target_exposure
        == pytest.approx(1.0)
    )


def test_watch_does_not_open_position():
    decision = evaluate_execution_v1(
        ExecutionInputsV1(
            opportunity_state=(
                OpportunityState.WATCH
            ),
            risk_entry_blocked=False,
            risk_max_exposure=1.0,
        ),
        flat_position(),
        POLICY,
    )

    assert (
        decision.action
        == ExecutionAction.HOLD_FLAT
    )


def test_do_not_chase_does_not_open_position():
    decision = evaluate_execution_v1(
        ExecutionInputsV1(
            opportunity_state=(
                OpportunityState.ARMED_DO_NOT_CHASE
            ),
            risk_entry_blocked=False,
            risk_max_exposure=1.0,
        ),
        flat_position(),
        POLICY,
    )

    assert (
        decision.action
        == ExecutionAction.HOLD_FLAT
    )


def test_risk_entry_block_vetoes_entry():
    decision = evaluate_execution_v1(
        ExecutionInputsV1(
            opportunity_state=(
                OpportunityState.ENTRY_CANDIDATE
            ),
            risk_entry_blocked=True,
            risk_max_exposure=1.0,
        ),
        flat_position(),
        POLICY,
    )

    assert (
        decision.action
        == ExecutionAction.HOLD_FLAT
    )


def test_watch_does_not_exit_existing_position():
    decision = evaluate_execution_v1(
        ExecutionInputsV1(
            opportunity_state=(
                OpportunityState.WATCH
            ),
            risk_entry_blocked=False,
            risk_max_exposure=1.0,
        ),
        full_position(),
        POLICY,
    )

    assert (
        decision.action
        == ExecutionAction.HOLD_POSITION
    )

    assert (
        decision.target_exposure
        == pytest.approx(1.0)
    )


def test_do_not_chase_does_not_exit_existing_position():
    decision = evaluate_execution_v1(
        ExecutionInputsV1(
            opportunity_state=(
                OpportunityState.ARMED_DO_NOT_CHASE
            ),
            risk_entry_blocked=False,
            risk_max_exposure=1.0,
        ),
        full_position(),
        POLICY,
    )

    assert (
        decision.action
        == ExecutionAction.HOLD_POSITION
    )


def test_risk_reduces_full_to_half():
    decision = evaluate_execution_v1(
        ExecutionInputsV1(
            opportunity_state=(
                OpportunityState.WATCH
            ),
            risk_entry_blocked=True,
            risk_max_exposure=0.5,
        ),
        full_position(),
        POLICY,
    )

    assert (
        decision.action
        == ExecutionAction.REDUCE_POSITION
    )

    assert (
        decision.target_exposure
        == pytest.approx(0.5)
    )


def test_risk_restores_half_to_full_without_new_entry_signal():
    decision = evaluate_execution_v1(
        ExecutionInputsV1(
            opportunity_state=(
                OpportunityState.WATCH
            ),
            risk_entry_blocked=True,
            risk_max_exposure=1.0,
        ),
        half_position(),
        POLICY,
    )

    assert (
        decision.action
        == ExecutionAction.RESTORE_POSITION
    )

    assert (
        decision.target_exposure
        == pytest.approx(1.0)
    )


def test_force_exit_has_highest_authority():
    decision = evaluate_execution_v1(
        ExecutionInputsV1(
            opportunity_state=(
                OpportunityState.ENTRY_CANDIDATE
            ),
            risk_entry_blocked=False,
            risk_max_exposure=1.0,
            risk_force_exit=True,
        ),
        full_position(),
        POLICY,
    )

    assert (
        decision.action
        == ExecutionAction.EXIT_POSITION
    )

    assert (
        decision.target_exposure
        == pytest.approx(0.0)
    )


def test_cooldown_blocks_new_entry():
    decision = evaluate_execution_v1(
        ExecutionInputsV1(
            opportunity_state=(
                OpportunityState.ENTRY_CANDIDATE
            ),
            risk_entry_blocked=False,
            risk_max_exposure=1.0,
            risk_cooldown_active=True,
        ),
        flat_position(),
        POLICY,
    )

    assert (
        decision.action
        == ExecutionAction.HOLD_FLAT
    )


def test_new_entry_sets_anchor():
    decision = evaluate_execution_v1(
        ExecutionInputsV1(
            opportunity_state=(
                OpportunityState.ENTRY_CANDIDATE
            ),
            risk_entry_blocked=False,
            risk_max_exposure=1.0,
        ),
        flat_position(),
        POLICY,
    )

    updated = apply_execution_fill(
        flat_position(),
        decision,
        execution_price=50.0,
    )

    assert (
        updated.current_exposure
        == pytest.approx(1.0)
    )

    assert (
        updated.entry_anchor_price
        == pytest.approx(50.0)
    )


def test_reduction_preserves_entry_anchor():
    decision = evaluate_execution_v1(
        ExecutionInputsV1(
            opportunity_state=(
                OpportunityState.WATCH
            ),
            risk_entry_blocked=True,
            risk_max_exposure=0.5,
        ),
        full_position(),
        POLICY,
    )

    updated = apply_execution_fill(
        full_position(),
        decision,
        execution_price=90.0,
    )

    assert (
        updated.current_exposure
        == pytest.approx(0.5)
    )

    assert (
        updated.entry_anchor_price
        == pytest.approx(100.0)
    )


def test_restoration_preserves_entry_anchor():
    decision = evaluate_execution_v1(
        ExecutionInputsV1(
            opportunity_state=(
                OpportunityState.WATCH
            ),
            risk_entry_blocked=False,
            risk_max_exposure=1.0,
        ),
        half_position(),
        POLICY,
    )

    updated = apply_execution_fill(
        half_position(),
        decision,
        execution_price=110.0,
    )

    assert (
        updated.current_exposure
        == pytest.approx(1.0)
    )

    assert (
        updated.entry_anchor_price
        == pytest.approx(100.0)
    )


def test_full_exit_clears_anchor():
    decision = evaluate_execution_v1(
        ExecutionInputsV1(
            opportunity_state=(
                OpportunityState.WATCH
            ),
            risk_entry_blocked=True,
            risk_max_exposure=0.0,
        ),
        full_position(),
        POLICY,
    )

    updated = apply_execution_fill(
        full_position(),
        decision,
        execution_price=90.0,
    )

    assert (
        updated.current_exposure
        == pytest.approx(0.0)
    )

    assert (
        updated.entry_anchor_price
        is None
    )


def test_drawdown_from_original_entry():
    position = PositionMemoryV1(
        current_exposure=0.5,
        entry_anchor_price=100.0,
    )

    drawdown = (
        calculate_drawdown_from_entry_pct(
            position,
            current_price=96.0,
        )
    )

    assert drawdown == pytest.approx(
        -4.0
    )


def test_transaction_cost_full_entry():
    cost = transaction_cost_fraction(
        current_exposure=0.0,
        target_exposure=1.0,
        policy=POLICY,
    )

    assert cost == pytest.approx(
        0.0005
    )


def test_transaction_cost_half_reduction():
    cost = transaction_cost_fraction(
        current_exposure=1.0,
        target_exposure=0.5,
        policy=POLICY,
    )

    assert cost == pytest.approx(
        0.00025
    )