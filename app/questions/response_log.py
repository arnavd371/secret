"""
Per-template response history (spec §9.7's prerequisite: "online IRT
parameter recalibration from response history" needs a real history to
recalibrate from). Append-only, same architecture as
app.ia_supervisor.disclosure_store — a response record documents an
event that happened and is never edited or deleted after the fact.

Scoped to template_id, not individual item_id: a parametric template's
many sampled instances (different numbers, same underlying difficulty
profile) are treated as one calibration unit, consistent with how
ItemTemplate.difficulty_band is itself declared once per template, not
once per sampled instance.
"""

from __future__ import annotations

import abc
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class ItemResponseRecord(BaseModel):
    template_id: str
    student_id: str
    correct: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResponseLogStore(abc.ABC):
    @abc.abstractmethod
    async def add(self, record: ItemResponseRecord) -> None: ...

    @abc.abstractmethod
    async def get_all(self, template_id: str) -> list[ItemResponseRecord]: ...

    @abc.abstractmethod
    async def get_all_for_student(self, student_id: str) -> list[ItemResponseRecord]: ...

    @abc.abstractmethod
    async def erase_student(self, student_id: str) -> int: ...


class InMemoryResponseLogStore(ResponseLogStore):
    def __init__(self) -> None:
        self._records: list[ItemResponseRecord] = []

    async def add(self, record: ItemResponseRecord) -> None:
        self._records.append(record)

    async def get_all(self, template_id: str) -> list[ItemResponseRecord]:
        return [r for r in self._records if r.template_id == template_id]

    async def get_all_for_student(self, student_id: str) -> list[ItemResponseRecord]:
        """Phase 19 (GDPR export): this store is normally queried by
        template (for recalibration); export needs the orthogonal
        by-student view instead."""
        return [r for r in self._records if r.student_id == student_id]

    async def erase_student(self, student_id: str) -> int:
        """Real deletion. Erasing a student's individual responses does
        reduce the sample a template's future recalibration draws from
        (app.questions.irt_recalibration) - an accepted, documented
        tradeoff of honoring erasure, not an oversight."""
        before = len(self._records)
        self._records = [r for r in self._records if r.student_id != student_id]
        return before - len(self._records)


_default_response_log_store: Optional[ResponseLogStore] = None


def get_default_response_log_store() -> ResponseLogStore:
    """Process-wide singleton, mirroring app.memory.store's pattern."""
    global _default_response_log_store
    if _default_response_log_store is None:
        _default_response_log_store = InMemoryResponseLogStore()
    return _default_response_log_store
