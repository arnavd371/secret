"""
Review-queue storage: unlike this codebase's append-only logs, entries
here get real in-place status updates (pending -> resolved/appealed), so
the store interface supports update-by-id, not just add-and-list.
"""

from __future__ import annotations

import abc
from typing import Optional

from app.review_queue.models import ReviewQueueEntry, ReviewStatus


class ReviewQueueStore(abc.ABC):
    @abc.abstractmethod
    async def add(self, entry: ReviewQueueEntry) -> None: ...

    @abc.abstractmethod
    async def get(self, entry_id: str) -> Optional[ReviewQueueEntry]: ...

    @abc.abstractmethod
    async def update(self, entry: ReviewQueueEntry) -> None: ...

    @abc.abstractmethod
    async def list_pending(self, student_id: Optional[str] = None) -> list[ReviewQueueEntry]: ...

    @abc.abstractmethod
    async def get_all_for_student(self, student_id: str) -> list[ReviewQueueEntry]: ...

    @abc.abstractmethod
    async def erase_student(self, student_id: str) -> int: ...


class InMemoryReviewQueueStore(ReviewQueueStore):
    def __init__(self) -> None:
        self._entries: dict[str, ReviewQueueEntry] = {}

    async def add(self, entry: ReviewQueueEntry) -> None:
        self._entries[entry.entry_id] = entry

    async def get(self, entry_id: str) -> Optional[ReviewQueueEntry]:
        return self._entries.get(entry_id)

    async def update(self, entry: ReviewQueueEntry) -> None:
        self._entries[entry.entry_id] = entry

    async def list_pending(self, student_id: Optional[str] = None) -> list[ReviewQueueEntry]:
        pending = [e for e in self._entries.values() if e.status == ReviewStatus.PENDING]
        if student_id is not None:
            pending = [e for e in pending if e.student_id == student_id]
        return sorted(pending, key=lambda e: e.created_at)

    async def get_all_for_student(self, student_id: str) -> list[ReviewQueueEntry]:
        """Phase 19 (GDPR export): every entry regardless of status, not
        just the pending ones list_pending returns."""
        matching = [e for e in self._entries.values() if e.student_id == student_id]
        return sorted(matching, key=lambda e: e.created_at)

    async def erase_student(self, student_id: str) -> int:
        ids = [entry_id for entry_id, e in self._entries.items() if e.student_id == student_id]
        for entry_id in ids:
            del self._entries[entry_id]
        return len(ids)


_default_review_queue_store: Optional[ReviewQueueStore] = None


def get_default_review_queue_store() -> ReviewQueueStore:
    """Process-wide singleton, mirroring app.memory.store's pattern."""
    global _default_review_queue_store
    if _default_review_queue_store is None:
        _default_review_queue_store = InMemoryReviewQueueStore()
    return _default_review_queue_store
