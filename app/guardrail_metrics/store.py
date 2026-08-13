"""
Append-only guardrail signal log - same architecture as
app.questions.response_log and app.ia_supervisor.disclosure_store: a
turn's guardrail signals document what actually happened and are never
edited or deleted after the fact.
"""

from __future__ import annotations

import abc
import json
from typing import Optional

import aiosqlite

from app.guardrail_metrics.models import GuardrailTurnSignals
from app.storage.schema import SqliteBackedStore


class GuardrailMetricsStore(abc.ABC):
    @abc.abstractmethod
    async def add(self, record: GuardrailTurnSignals) -> None: ...

    @abc.abstractmethod
    async def get_all(self) -> list[GuardrailTurnSignals]: ...


class InMemoryGuardrailMetricsStore(GuardrailMetricsStore):
    def __init__(self) -> None:
        self._records: list[GuardrailTurnSignals] = []

    async def add(self, record: GuardrailTurnSignals) -> None:
        self._records.append(record)

    async def get_all(self) -> list[GuardrailTurnSignals]:
        return list(self._records)


class SqliteGuardrailMetricsStore(GuardrailMetricsStore, SqliteBackedStore):
    """File-based persistence for the MVP: guardrail telemetry
    accumulates across restarts instead of resetting, so
    compute_guardrail_metrics has a real, growing sample to aggregate
    over rather than only ever seeing the current process's traffic."""

    async def add(self, record: GuardrailTurnSignals) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO guardrail_signals (turn_id, data) VALUES (?, ?)",
                (record.turn_id, record.model_dump_json()),
            )
            await db.commit()

    async def get_all(self) -> list[GuardrailTurnSignals]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT data FROM guardrail_signals") as cursor:
                rows = await cursor.fetchall()
        return [GuardrailTurnSignals(**json.loads(row[0])) for row in rows]


_default_guardrail_metrics_store: Optional[GuardrailMetricsStore] = None


def get_default_guardrail_metrics_store() -> GuardrailMetricsStore:
    """Process-wide singleton, mirroring app.memory.store's pattern."""
    global _default_guardrail_metrics_store
    if _default_guardrail_metrics_store is None:
        _default_guardrail_metrics_store = InMemoryGuardrailMetricsStore()
    return _default_guardrail_metrics_store
