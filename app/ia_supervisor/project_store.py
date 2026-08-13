"""
IA/EE project state storage (spec §11), same store-interface pattern as
app.memory.store and app.adaptive.store: an ABC with an in-memory
implementation for tests/dev; a real deployment swaps in a persistent
implementation without changing any caller.
"""

from __future__ import annotations

import abc
import json
from typing import Optional

import aiosqlite

from app.ia_supervisor.models import IAProjectState
from app.storage.schema import SqliteBackedStore


class IAProjectStateStore(abc.ABC):
    @abc.abstractmethod
    async def get(self, student_id: str, project_id: str) -> Optional[IAProjectState]: ...

    @abc.abstractmethod
    async def save(self, state: IAProjectState) -> None: ...

    @abc.abstractmethod
    async def get_all_for_student(self, student_id: str) -> list[IAProjectState]: ...

    @abc.abstractmethod
    async def erase_student(self, student_id: str) -> int: ...


class InMemoryIAProjectStateStore(IAProjectStateStore):
    def __init__(self) -> None:
        self._states: dict[tuple[str, str], IAProjectState] = {}

    async def get(self, student_id: str, project_id: str) -> Optional[IAProjectState]:
        return self._states.get((student_id, project_id))

    async def save(self, state: IAProjectState) -> None:
        self._states[(state.student_id, state.project_id)] = state

    async def get_all_for_student(self, student_id: str) -> list[IAProjectState]:
        """Phase 19 (GDPR export): every IA/EE project this student has,
        not just one looked up by id - needed to enumerate what data
        exists about them at all."""
        return [state for (sid, _), state in self._states.items() if sid == student_id]

    async def erase_student(self, student_id: str) -> int:
        keys = [k for k in self._states if k[0] == student_id]
        for k in keys:
            del self._states[k]
        return len(keys)


class SqliteIAProjectStateStore(IAProjectStateStore, SqliteBackedStore):
    """File-based persistence for the MVP: a project's stage survives a
    process restart instead of resetting to topic_selection."""

    async def get(self, student_id: str, project_id: str) -> Optional[IAProjectState]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT data FROM ia_project_state WHERE student_id = ? AND project_id = ?",
                (student_id, project_id),
            ) as cursor:
                row = await cursor.fetchone()
        return IAProjectState(**json.loads(row[0])) if row else None

    async def save(self, state: IAProjectState) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO ia_project_state (student_id, project_id, data) VALUES (?, ?, ?) "
                "ON CONFLICT(student_id, project_id) DO UPDATE SET data = excluded.data",
                (state.student_id, state.project_id, state.model_dump_json()),
            )
            await db.commit()

    async def get_all_for_student(self, student_id: str) -> list[IAProjectState]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT data FROM ia_project_state WHERE student_id = ?", (student_id,)
            ) as cursor:
                rows = await cursor.fetchall()
        return [IAProjectState(**json.loads(row[0])) for row in rows]

    async def erase_student(self, student_id: str) -> int:
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("DELETE FROM ia_project_state WHERE student_id = ?", (student_id,))
            await db.commit()
            return cursor.rowcount


_default_ia_project_store: Optional[IAProjectStateStore] = None


def get_default_ia_project_store() -> IAProjectStateStore:
    """Process-wide singleton, mirroring app.memory.store's pattern."""
    global _default_ia_project_store
    if _default_ia_project_store is None:
        _default_ia_project_store = InMemoryIAProjectStateStore()
    return _default_ia_project_store
