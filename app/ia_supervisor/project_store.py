"""
IA/EE project state storage (spec §11), same store-interface pattern as
app.memory.store and app.adaptive.store: an ABC with an in-memory
implementation for tests/dev; a real deployment swaps in a persistent
implementation without changing any caller.
"""

from __future__ import annotations

import abc
from typing import Optional

from app.ia_supervisor.models import IAProjectState


class IAProjectStateStore(abc.ABC):
    @abc.abstractmethod
    async def get(self, student_id: str, project_id: str) -> Optional[IAProjectState]: ...

    @abc.abstractmethod
    async def save(self, state: IAProjectState) -> None: ...


class InMemoryIAProjectStateStore(IAProjectStateStore):
    def __init__(self) -> None:
        self._states: dict[tuple[str, str], IAProjectState] = {}

    async def get(self, student_id: str, project_id: str) -> Optional[IAProjectState]:
        return self._states.get((student_id, project_id))

    async def save(self, state: IAProjectState) -> None:
        self._states[(state.student_id, state.project_id)] = state


_default_ia_project_store: Optional[IAProjectStateStore] = None


def get_default_ia_project_store() -> IAProjectStateStore:
    """Process-wide singleton, mirroring app.memory.store's pattern."""
    global _default_ia_project_store
    if _default_ia_project_store is None:
        _default_ia_project_store = InMemoryIAProjectStateStore()
    return _default_ia_project_store
