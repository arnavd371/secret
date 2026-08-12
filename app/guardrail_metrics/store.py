"""
Append-only guardrail signal log - same architecture as
app.questions.response_log and app.ia_supervisor.disclosure_store: a
turn's guardrail signals document what actually happened and are never
edited or deleted after the fact.
"""

from __future__ import annotations

import abc
from typing import Optional

from app.guardrail_metrics.models import GuardrailTurnSignals


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


_default_guardrail_metrics_store: Optional[GuardrailMetricsStore] = None


def get_default_guardrail_metrics_store() -> GuardrailMetricsStore:
    """Process-wide singleton, mirroring app.memory.store's pattern."""
    global _default_guardrail_metrics_store
    if _default_guardrail_metrics_store is None:
        _default_guardrail_metrics_store = InMemoryGuardrailMetricsStore()
    return _default_guardrail_metrics_store
