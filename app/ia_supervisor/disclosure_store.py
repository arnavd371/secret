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

    @abc.abstractmethod
    async def get_all_for_student(self, student_id: str) -> list[DisclosureEntry]: ...

    # Deliberately no erase_student() here (Phase 19, GDPR export/erasure):
    # this store's own docstring above already establishes disclosure
    # entries as records that "must never be edited or deleted after the
    # fact." An AI-use academic-integrity disclosure log is exactly the
    # kind of record GDPR Article 17(3)'s legal-obligation exemption
    # covers - a school may need to retain it regardless of an erasure
    # request. app.privacy.gdpr documents this exemption explicitly
    # rather than silently including or silently skipping it.


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

    async def get_all_for_student(self, student_id: str) -> list[DisclosureEntry]:
        matching = [e for e in self._entries if e.student_id == student_id]
        return sorted(matching, key=lambda e: e.timestamp)


_default_disclosure_store: Optional[DisclosureStore] = None


def get_default_disclosure_store() -> DisclosureStore:
    """Process-wide singleton, mirroring app.memory.store's pattern."""
    global _default_disclosure_store
    if _default_disclosure_store is None:
        _default_disclosure_store = InMemoryDisclosureStore()
    return _default_disclosure_store
