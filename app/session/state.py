"""
Session-scoped state for the hint ladder, keyed by (session_id, problem_id).

Two concerns live here:
  - Storage: an interface with a Redis-backed implementation (for real
    deployments) and an in-memory implementation (for tests / dev without
    Redis wired up), so callers never depend on which one is active.
  - Escalation: `advance_session_state`, a pure function that updates
    attempt_count/hint_ladder_level after a turn.

Per spec §1.5's pseudocode, the decision policy itself computes the next
hint level as `min(hint_ladder_level + 1, cap)` — the ladder escalates by
construction every time a HINT action is chosen. So the session store's
job on write-back is simple and derived directly from what the policy
already decided, not a second independent heuristic:
  - a new problem_id always resets attempt_count and hint_ladder_level to 0
  - a solve_request turn counts as another attempt (attempt_count += 1)
  - if the chosen Action was a HINT, hint_ladder_level becomes that
    action's level (which the policy already capped correctly)
  - otherwise hint_ladder_level is left as-is

De-escalation per spec §7.2's ladder table ("drop to level N-1 if next
attempt is correct") requires knowing whether an attempt was actually
correct, which needs the Math Solver + CAS agent (§2.2) — that agent does
not exist yet (Phase 2). Escalation-only behavior is what's implementable
without CAS; de-escalation is a documented TODO here, not a bug.
TODO(Phase 2 / CAS agent): once attempt correctness is verifiable, add
de-escalation using the §7.2 table's per-level drop rules.

Memory/mastery persistence *beyond* this session-scoped hint ladder state
is out of scope for Phase 1.
TODO(Phase 5): long-term mastery persistence across sessions.
"""

from __future__ import annotations

import abc
import json
from typing import Optional

from pydantic import BaseModel, Field

from app.models.contracts import Action, ActionType, IntentType, MAX_HINT_LADDER_LEVEL


class ProblemSessionState(BaseModel):
    session_id: str
    problem_id: Optional[str] = None
    attempt_count: int = Field(default=0, ge=0)
    hint_ladder_level: int = Field(default=0, ge=0, le=MAX_HINT_LADDER_LEVEL)


def advance_session_state(
    state: ProblemSessionState,
    problem_id: Optional[str],
    intent: IntentType,
    action: Action,
) -> ProblemSessionState:
    if problem_id is not None and problem_id != state.problem_id:
        state = ProblemSessionState(session_id=state.session_id, problem_id=problem_id)

    attempt_count = state.attempt_count + 1 if intent == IntentType.SOLVE_REQUEST else state.attempt_count
    hint_ladder_level = action.level if action.action_type == ActionType.HINT else state.hint_ladder_level

    return state.model_copy(update={"attempt_count": attempt_count, "hint_ladder_level": hint_ladder_level})


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

    async def get(self, session_id: str, problem_id: Optional[str]) -> ProblemSessionState:
        existing = self._store.get(session_id)
        if existing is None:
            return ProblemSessionState(session_id=session_id, problem_id=problem_id)
        return existing

    async def save(self, state: ProblemSessionState) -> None:
        self._store[state.session_id] = state


class RedisSessionStateStore(SessionStateStore):
    """Redis-backed implementation, keyed by session_id (problem_id is
    stored as a field so a problem switch is detected by
    advance_session_state rather than by key structure)."""

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
