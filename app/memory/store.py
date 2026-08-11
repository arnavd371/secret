"""
Memory Agent storage (spec §2.2, §4): read/write the per-student mastery
map and misconception registry. Same store-interface pattern as
app/session/state.py — an ABC with an in-memory implementation for tests/
dev; a real deployment swaps in a Postgres-backed implementation against
the schema in spec §4.2 without changing any caller.
"""

from __future__ import annotations

import abc
from typing import Optional

from app.memory.models import MisconceptionRegistryEntry, SubtopicMastery


class MemoryStore(abc.ABC):
    @abc.abstractmethod
    async def get_mastery(self, student_id: str, subtopic_id: str) -> Optional[SubtopicMastery]: ...

    @abc.abstractmethod
    async def save_mastery(self, mastery: SubtopicMastery) -> None: ...

    @abc.abstractmethod
    async def get_misconceptions(self, student_id: str) -> list[MisconceptionRegistryEntry]: ...

    @abc.abstractmethod
    async def save_misconception(self, entry: MisconceptionRegistryEntry) -> None: ...


class InMemoryMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._mastery: dict[tuple[str, str], SubtopicMastery] = {}
        self._misconceptions: dict[tuple[str, str], MisconceptionRegistryEntry] = {}

    async def get_mastery(self, student_id: str, subtopic_id: str) -> Optional[SubtopicMastery]:
        return self._mastery.get((student_id, subtopic_id))

    async def save_mastery(self, mastery: SubtopicMastery) -> None:
        self._mastery[(mastery.student_id, mastery.subtopic_id)] = mastery

    async def get_misconceptions(self, student_id: str) -> list[MisconceptionRegistryEntry]:
        return [entry for (sid, _), entry in self._misconceptions.items() if sid == student_id]

    async def save_misconception(self, entry: MisconceptionRegistryEntry) -> None:
        self._misconceptions[(entry.student_id, entry.misconception_id)] = entry


_default_memory_store: Optional[MemoryStore] = None


def get_default_memory_store() -> MemoryStore:
    """Process-wide singleton, mirroring app.knowledge.retriever's pattern."""
    global _default_memory_store
    if _default_memory_store is None:
        _default_memory_store = InMemoryMemoryStore()
    return _default_memory_store
