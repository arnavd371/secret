"""
Memory Agent storage (spec §2.2, §4): read/write the per-student mastery
map and misconception registry. Same store-interface pattern as
app/session/state.py — an ABC with an in-memory implementation for tests/
dev; a real deployment swaps in a Postgres-backed implementation against
the schema in spec §4.2 without changing any caller.
"""

from __future__ import annotations

import abc
import json
from typing import Optional

import aiosqlite

from app.memory.models import MisconceptionRegistryEntry, SubtopicMastery
from app.storage.schema import SqliteBackedStore


class MemoryStore(abc.ABC):
    @abc.abstractmethod
    async def get_mastery(self, student_id: str, subtopic_id: str) -> Optional[SubtopicMastery]: ...

    @abc.abstractmethod
    async def save_mastery(self, mastery: SubtopicMastery) -> None: ...

    @abc.abstractmethod
    async def get_all_mastery(self, student_id: str) -> list[SubtopicMastery]: ...

    @abc.abstractmethod
    async def get_misconceptions(self, student_id: str) -> list[MisconceptionRegistryEntry]: ...

    @abc.abstractmethod
    async def save_misconception(self, entry: MisconceptionRegistryEntry) -> None: ...

    @abc.abstractmethod
    async def erase_student(self, student_id: str) -> int: ...


class InMemoryMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._mastery: dict[tuple[str, str], SubtopicMastery] = {}
        self._misconceptions: dict[tuple[str, str], MisconceptionRegistryEntry] = {}

    async def get_mastery(self, student_id: str, subtopic_id: str) -> Optional[SubtopicMastery]:
        return self._mastery.get((student_id, subtopic_id))

    async def save_mastery(self, mastery: SubtopicMastery) -> None:
        self._mastery[(mastery.student_id, mastery.subtopic_id)] = mastery

    async def get_all_mastery(self, student_id: str) -> list[SubtopicMastery]:
        """Phase 18's consolidation batch job (app/memory/consolidation.py)
        needs to enumerate every subtopic a student has a record for,
        not just look one up by name — a real production store backs
        this with "select * where student_id = X", same as
        get_misconceptions already does."""
        return [m for (sid, _), m in self._mastery.items() if sid == student_id]

    async def get_misconceptions(self, student_id: str) -> list[MisconceptionRegistryEntry]:
        return [entry for (sid, _), entry in self._misconceptions.items() if sid == student_id]

    async def save_misconception(self, entry: MisconceptionRegistryEntry) -> None:
        self._misconceptions[(entry.student_id, entry.misconception_id)] = entry

    async def erase_student(self, student_id: str) -> int:
        """Phase 19 (GDPR erasure): real deletion, not a soft-delete flag
        - every mastery and misconception record keyed to this student is
        removed outright. Returns the real count of records erased."""
        mastery_keys = [k for k in self._mastery if k[0] == student_id]
        misconception_keys = [k for k in self._misconceptions if k[0] == student_id]
        for k in mastery_keys:
            del self._mastery[k]
        for k in misconception_keys:
            del self._misconceptions[k]
        return len(mastery_keys) + len(misconception_keys)


class SqliteMemoryStore(MemoryStore, SqliteBackedStore):
    """File-based persistence for the MVP: mastery and misconception
    records survive a process restart, in two tables mirroring the two
    dicts InMemoryMemoryStore keeps in memory."""

    async def get_mastery(self, student_id: str, subtopic_id: str) -> Optional[SubtopicMastery]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT data FROM mastery WHERE student_id = ? AND subtopic_id = ?", (student_id, subtopic_id)
            ) as cursor:
                row = await cursor.fetchone()
        return SubtopicMastery(**json.loads(row[0])) if row else None

    async def save_mastery(self, mastery: SubtopicMastery) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO mastery (student_id, subtopic_id, data) VALUES (?, ?, ?) "
                "ON CONFLICT(student_id, subtopic_id) DO UPDATE SET data = excluded.data",
                (mastery.student_id, mastery.subtopic_id, mastery.model_dump_json()),
            )
            await db.commit()

    async def get_all_mastery(self, student_id: str) -> list[SubtopicMastery]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT data FROM mastery WHERE student_id = ?", (student_id,)) as cursor:
                rows = await cursor.fetchall()
        return [SubtopicMastery(**json.loads(row[0])) for row in rows]

    async def get_misconceptions(self, student_id: str) -> list[MisconceptionRegistryEntry]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT data FROM misconceptions WHERE student_id = ?", (student_id,)
            ) as cursor:
                rows = await cursor.fetchall()
        return [MisconceptionRegistryEntry(**json.loads(row[0])) for row in rows]

    async def save_misconception(self, entry: MisconceptionRegistryEntry) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO misconceptions (student_id, misconception_id, data) VALUES (?, ?, ?) "
                "ON CONFLICT(student_id, misconception_id) DO UPDATE SET data = excluded.data",
                (entry.student_id, entry.misconception_id, entry.model_dump_json()),
            )
            await db.commit()

    async def erase_student(self, student_id: str) -> int:
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            mastery_cursor = await db.execute("DELETE FROM mastery WHERE student_id = ?", (student_id,))
            misconception_cursor = await db.execute(
                "DELETE FROM misconceptions WHERE student_id = ?", (student_id,)
            )
            await db.commit()
            return mastery_cursor.rowcount + misconception_cursor.rowcount


_default_memory_store: Optional[MemoryStore] = None


def get_default_memory_store() -> MemoryStore:
    """Process-wide singleton, mirroring app.knowledge.retriever's pattern."""
    global _default_memory_store
    if _default_memory_store is None:
        _default_memory_store = InMemoryMemoryStore()
    return _default_memory_store
