"""
Session-scoped state for the hint ladder, keyed by (session_id, problem_id).

Two concerns live here:
  - Storage: an interface with a Redis-backed implementation (for real
    deployments) and an in-memory implementation (for tests / dev without
    Redis wired up), so callers never depend on which one is active.
  - Escalation policy: `apply_turn_outcome`, a pure function that decides
    how attempt_count/hint_ladder_level change given what happened on a
    turn. This is intentionally NOT "increment every time" — see the
    TurnOutcome branches below.

Memory/mastery persistence *beyond* this session-scoped hint ladder state
is out of scope for Phase 1.
TODO(Phase 5): long-term mastery persistence across sessions.
"""

from __future__ import annotations

import abc
import json
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

MAX_HINT_LADDER_LEVEL = 4


class TurnOutcome(str, Enum):
    """What happened as a result of this turn, used to update the ladder."""

    NEW_PROBLEM = "new_problem"
    ATTEMPTED_INCORRECT = "attempted_incorrect"
    ATTEMPTED_CORRECT = "attempted_correct"
    HINT_REQUESTED = "hint_requested"
    NO_CHANGE = "no_change"


class ProblemSessionState(BaseModel):
    session_id: str
    problem_id: Optional[str] = None
    attempt_count: int = Field(default=0, ge=0)
    hint_ladder_level: int = Field(default=0, ge=0, le=MAX_HINT_LADDER_LEVEL)


def apply_turn_outcome(
    state: ProblemSessionState, problem_id: Optional[str], outcome: TurnOutcome
) -> ProblemSessionState:
    """
    Pure function: given the previous state, the problem_id of the CURRENT
    turn, and what happened, return the next state.

    Rules:
      - If the problem changed, the ladder resets to 0/0 regardless of
        outcome (a new problem is always a fresh start).
      - ATTEMPTED_INCORRECT: increments attempt_count. Escalates the
        ladder only once the student has struggled for more than one
        attempt (attempt_count >= 2) rather than on every single miss, so
        a single wrong try doesn't immediately jump to a strong hint.
      - ATTEMPTED_CORRECT: resets attempt_count (fresh start on retries)
        and de-escalates the ladder by one level, rewarding demonstrated
        understanding rather than leaving them stuck at a high hint level.
      - HINT_REQUESTED: an explicit ask for more help escalates
        immediately, independent of attempt_count.
      - NO_CHANGE / NEW_PROBLEM: no further mutation beyond the reset
        already applied above.
    """
    if problem_id is not None and problem_id != state.problem_id:
        state = ProblemSessionState(session_id=state.session_id, problem_id=problem_id, attempt_count=0, hint_ladder_level=0)
        if outcome == TurnOutcome.NEW_PROBLEM:
            return state
        # fall through: apply this turn's outcome on top of the fresh state

    if outcome == TurnOutcome.NEW_PROBLEM:
        return state.model_copy(update={"attempt_count": 0, "hint_ladder_level": 0})

    if outcome == TurnOutcome.ATTEMPTED_INCORRECT:
        new_attempt_count = state.attempt_count + 1
        new_level = state.hint_ladder_level
        if new_attempt_count >= 2:
            new_level = min(state.hint_ladder_level + 1, MAX_HINT_LADDER_LEVEL)
        return state.model_copy(update={"attempt_count": new_attempt_count, "hint_ladder_level": new_level})

    if outcome == TurnOutcome.ATTEMPTED_CORRECT:
        return state.model_copy(
            update={"attempt_count": 0, "hint_ladder_level": max(state.hint_ladder_level - 1, 0)}
        )

    if outcome == TurnOutcome.HINT_REQUESTED:
        return state.model_copy(
            update={"hint_ladder_level": min(state.hint_ladder_level + 1, MAX_HINT_LADDER_LEVEL)}
        )

    return state  # NO_CHANGE


class SessionStateStore(abc.ABC):
    @abc.abstractmethod
    async def get(self, session_id: str, problem_id: Optional[str]) -> ProblemSessionState: ...

    @abc.abstractmethod
    async def save(self, state: ProblemSessionState) -> None: ...


class InMemorySessionStateStore(SessionStateStore):
    """Used in tests and in dev environments where Redis isn't wired up
    yet. Same interface as the Redis-backed store so callers are agnostic."""

    def __init__(self) -> None:
        self._store: dict[str, ProblemSessionState] = {}

    def _key(self, session_id: str) -> str:
        return session_id

    async def get(self, session_id: str, problem_id: Optional[str]) -> ProblemSessionState:
        existing = self._store.get(self._key(session_id))
        if existing is None:
            return ProblemSessionState(session_id=session_id, problem_id=problem_id)
        return existing

    async def save(self, state: ProblemSessionState) -> None:
        self._store[self._key(state.session_id)] = state


class RedisSessionStateStore(SessionStateStore):
    """Redis-backed implementation, keyed by session_id (problem_id is
    stored as a field so a problem switch is detected by the caller/
    apply_turn_outcome rather than by key structure)."""

    def __init__(self, redis_client) -> None:  # type: ignore[no-untyped-def]
        self._redis = redis_client

    def _key(self, session_id: str) -> str:
        return f"tutor:session_state:{session_id}"

    async def get(self, session_id: str, problem_id: Optional[str]) -> ProblemSessionState:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            return ProblemSessionState(session_id=session_id, problem_id=problem_id)
        return ProblemSessionState(**json.loads(raw))

    async def save(self, state: ProblemSessionState) -> None:
        await self._redis.set(self._key(state.session_id), state.model_dump_json())
