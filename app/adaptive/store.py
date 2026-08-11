"""
Adaptive Learning Engine storage (spec §12), same store-interface pattern
as app.memory.store and app.session.state: an ABC with an in-memory
implementation for tests/dev; a real deployment swaps in a persistent
implementation without changing any caller.
"""

from __future__ import annotations

import abc
from typing import Optional

from app.adaptive.models import ReviewState


class ReviewStateStore(abc.ABC):
    @abc.abstractmethod
    async def get(self, student_id: str, subtopic_id: str) -> Optional[ReviewState]: ...

    @abc.abstractmethod
    async def save(self, state: ReviewState) -> None: ...

    @abc.abstractmethod
    async def get_all_for_student(self, student_id: str) -> list[ReviewState]: ...


class InMemoryReviewStateStore(ReviewStateStore):
    def __init__(self) -> None:
        self._states: dict[tuple[str, str], ReviewState] = {}

    async def get(self, student_id: str, subtopic_id: str) -> Optional[ReviewState]:
        return self._states.get((student_id, subtopic_id))

    async def save(self, state: ReviewState) -> None:
        self._states[(state.student_id, state.subtopic_id)] = state

    async def get_all_for_student(self, student_id: str) -> list[ReviewState]:
        return [state for (sid, _), state in self._states.items() if sid == student_id]


_default_review_state_store: Optional[ReviewStateStore] = None


def get_default_review_state_store() -> ReviewStateStore:
    """Process-wide singleton, mirroring app.memory.store's pattern."""
    global _default_review_state_store
    if _default_review_state_store is None:
        _default_review_state_store = InMemoryReviewStateStore()
    return _default_review_state_store
