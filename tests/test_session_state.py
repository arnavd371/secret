import pytest

from app.models.contracts import Action, ActionType, IntentType
from app.session.state import InMemorySessionStateStore, ProblemSessionState, advance_session_state


def _hint(level: int) -> Action:
    return Action(action_type=ActionType.HINT, level=level, reason="test")


def _question() -> Action:
    return Action(action_type=ActionType.QUESTION, move="diagnostic_probe", reason="test")


def _explain() -> Action:
    return Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")


def test_new_problem_resets_ladder_and_attempt_count():
    state = ProblemSessionState(session_id="s1", problem_id="p1", attempt_count=3, hint_ladder_level=3)
    next_state = advance_session_state(state, "p2", IntentType.SOLVE_REQUEST, _question())
    assert next_state.problem_id == "p2"
    # the reset happens first, then this turn's solve_request attempt is counted
    assert next_state.attempt_count == 1
    assert next_state.hint_ladder_level == 0


def test_same_problem_solve_request_increments_attempt_count():
    state = ProblemSessionState(session_id="s1", problem_id="p1", attempt_count=1, hint_ladder_level=0)
    next_state = advance_session_state(state, "p1", IntentType.SOLVE_REQUEST, _question())
    assert next_state.attempt_count == 2
    assert next_state.hint_ladder_level == 0  # QUESTION doesn't touch the ladder


def test_hint_action_sets_ladder_level_to_the_actions_level():
    state = ProblemSessionState(session_id="s1", problem_id="p1", attempt_count=1, hint_ladder_level=0)
    next_state = advance_session_state(state, "p1", IntentType.SOLVE_REQUEST, _hint(1))
    assert next_state.hint_ladder_level == 1
    assert next_state.attempt_count == 2

    next_state = advance_session_state(next_state, "p1", IntentType.SOLVE_REQUEST, _hint(2))
    assert next_state.hint_ladder_level == 2
    assert next_state.attempt_count == 3


def test_non_hint_non_solve_action_does_not_change_attempt_count_or_ladder():
    state = ProblemSessionState(session_id="s1", problem_id="p1", attempt_count=2, hint_ladder_level=1)
    next_state = advance_session_state(state, "p1", IntentType.CHECK_WORK, _explain())
    assert next_state.attempt_count == 2
    assert next_state.hint_ladder_level == 1


def test_problem_id_none_does_not_reset_state():
    """A problem_id of None (e.g. a general_chat turn with no active
    problem) should not be treated as a problem switch."""
    state = ProblemSessionState(session_id="s1", problem_id="p1", attempt_count=2, hint_ladder_level=1)
    next_state = advance_session_state(state, None, IntentType.CHECK_WORK, _explain())
    assert next_state.problem_id == "p1"
    assert next_state.attempt_count == 2
    assert next_state.hint_ladder_level == 1


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
