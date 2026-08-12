"""
AI-assistance disclosure log storage (spec §11): append-only by design —
a disclosure record documents what happened and must never be edited or
deleted after the fact, unlike the keyed-upsert stores elsewhere in this
codebase (SubtopicMastery, ReviewState, IAProjectState all represent
current state and get overwritten on update; a DisclosureEntry represents
a historical event and only ever gets added to).
"""

from __future__ import annotations

import abc
from typing import Optional

from app.ia_supervisor.models import DisclosureEntry


class DisclosureStore(abc.ABC):
    @abc.abstractmethod
    async def add(self, entry: DisclosureEntry) -> None: ...

    @abc.abstractmethod
    async def get_all(self, student_id: str, project_id: str) -> list[DisclosureEntry]: ...


class InMemoryDisclosureStore(DisclosureStore):
    def __init__(self) -> None:
        self._entries: list[DisclosureEntry] = []

    async def add(self, entry: DisclosureEntry) -> None:
        self._entries.append(entry)

    async def get_all(self, student_id: str, project_id: str) -> list[DisclosureEntry]:
        matching = [
            e for e in self._entries if e.student_id == student_id and e.project_id == project_id
        ]
        return sorted(matching, key=lambda e: e.timestamp)


_default_disclosure_store: Optional[DisclosureStore] = None


def get_default_disclosure_store() -> DisclosureStore:
    """Process-wide singleton, mirroring app.memory.store's pattern."""
    global _default_disclosure_store
    if _default_disclosure_store is None:
        _default_disclosure_store = InMemoryDisclosureStore()
    return _default_disclosure_store
