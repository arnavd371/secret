import pytest

from app.session.state import (
    InMemorySessionStateStore,
    ProblemSessionState,
    TurnOutcome,
    apply_turn_outcome,
)


def test_new_problem_resets_ladder():
    state = ProblemSessionState(session_id="s1", problem_id="p1", attempt_count=3, hint_ladder_level=3)
    next_state = apply_turn_outcome(state, "p2", TurnOutcome.NEW_PROBLEM)
    assert next_state.problem_id == "p2"
    assert next_state.attempt_count == 0
    assert next_state.hint_ladder_level == 0


def test_problem_change_resets_even_without_explicit_new_problem_outcome():
    """If the caller passes a different problem_id, the ladder resets
    regardless of the outcome label passed in."""
    state = ProblemSessionState(session_id="s1", problem_id="p1", attempt_count=3, hint_ladder_level=3)
    next_state = apply_turn_outcome(state, "p2", TurnOutcome.ATTEMPTED_INCORRECT)
    assert next_state.problem_id == "p2"
    assert next_state.attempt_count == 1  # the incorrect outcome still applies on the fresh state
    assert next_state.hint_ladder_level == 0  # not escalated yet: only one attempt so far


def test_single_incorrect_attempt_does_not_escalate_ladder():
    state = ProblemSessionState(session_id="s1", problem_id="p1", attempt_count=0, hint_ladder_level=0)
    next_state = apply_turn_outcome(state, "p1", TurnOutcome.ATTEMPTED_INCORRECT)
    assert next_state.attempt_count == 1
    assert next_state.hint_ladder_level == 0


def test_repeated_incorrect_attempts_escalate_ladder():
    state = ProblemSessionState(session_id="s1", problem_id="p1", attempt_count=1, hint_ladder_level=0)
    next_state = apply_turn_outcome(state, "p1", TurnOutcome.ATTEMPTED_INCORRECT)
    assert next_state.attempt_count == 2
    assert next_state.hint_ladder_level == 1

    next_state = apply_turn_outcome(next_state, "p1", TurnOutcome.ATTEMPTED_INCORRECT)
    assert next_state.attempt_count == 3
    assert next_state.hint_ladder_level == 2


def test_ladder_does_not_escalate_past_max():
    state = ProblemSessionState(session_id="s1", problem_id="p1", attempt_count=5, hint_ladder_level=4)
    next_state = apply_turn_outcome(state, "p1", TurnOutcome.ATTEMPTED_INCORRECT)
    assert next_state.hint_ladder_level == 4


def test_correct_attempt_deescalates_and_resets_attempt_count():
    state = ProblemSessionState(session_id="s1", problem_id="p1", attempt_count=3, hint_ladder_level=2)
    next_state = apply_turn_outcome(state, "p1", TurnOutcome.ATTEMPTED_CORRECT)
    assert next_state.attempt_count == 0
    assert next_state.hint_ladder_level == 1


def test_correct_attempt_does_not_deescalate_below_zero():
    state = ProblemSessionState(session_id="s1", problem_id="p1", attempt_count=1, hint_ladder_level=0)
    next_state = apply_turn_outcome(state, "p1", TurnOutcome.ATTEMPTED_CORRECT)
    assert next_state.hint_ladder_level == 0


def test_explicit_hint_request_escalates_immediately_regardless_of_attempt_count():
    state = ProblemSessionState(session_id="s1", problem_id="p1", attempt_count=0, hint_ladder_level=0)
    next_state = apply_turn_outcome(state, "p1", TurnOutcome.HINT_REQUESTED)
    assert next_state.hint_ladder_level == 1
    assert next_state.attempt_count == 0  # hint requests don't count as attempts


def test_no_change_outcome_is_a_no_op():
    state = ProblemSessionState(session_id="s1", problem_id="p1", attempt_count=2, hint_ladder_level=1)
    next_state = apply_turn_outcome(state, "p1", TurnOutcome.NO_CHANGE)
    assert next_state == state


@pytest.mark.asyncio
async def test_in_memory_store_roundtrip():
    store = InMemorySessionStateStore()
    fresh = await store.get("session-a", "problem-1")
    assert fresh.attempt_count == 0
    assert fresh.hint_ladder_level == 0

    updated = fresh.model_copy(update={"attempt_count": 2, "hint_ladder_level": 1})
    await store.save(updated)

    fetched = await store.get("session-a", "problem-1")
    assert fetched.attempt_count == 2
    assert fetched.hint_ladder_level == 1
