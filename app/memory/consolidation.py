"""
Memory consolidation batch job: a real, deterministic batch pass over a
student's persisted mastery and misconception records that brings their
at-rest state in line with what decay has already done to them by
read-time computation. No model call anywhere in this module - it
reuses Phase 5's own decay/node_state math (app.memory.decay,
app.memory.node_state) rather than introducing new formulas.

Two real operations, run together by `run_memory_consolidation`:
  - A misconception whose decayed strength has fallen to (or below) the
    same activity threshold Phase 5's context assembly already uses to
    decide whether to surface it is marked remediated, so it stops
    being carried forward indefinitely as "active" once it's genuinely
    decayed away.
  - A mastery record's stored `node_state` is recomputed from its
    current decayed mastery and persisted if it's drifted from what's
    stored - e.g. a record last written while "mastered" that hasn't
    been practiced in over a month should be persisted as DECAYED, not
    left reading "mastered" until the next read-time computation
    happens to notice.

This is a batch job in the literal sense (meant to be run periodically
over a student, or all students, not on the hot request path) - nothing
in app.orchestrator.handle_turn calls it per-turn.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from app.memory.context_assembly import MISCONCEPTION_ACTIVE_THRESHOLD
from app.memory.decay import decayed_misconception_strength
from app.memory.models import MisconceptionRegistryEntry, SubtopicMastery
from app.memory.node_state import compute_node_state
from app.memory.store import MemoryStore


class ConsolidationReport(BaseModel):
    student_id: str
    misconceptions_remediated: list[str] = Field(default_factory=list)
    mastery_node_states_updated: list[str] = Field(default_factory=list)


async def consolidate_misconceptions(
    store: MemoryStore, student_id: str, now: Optional[datetime] = None
) -> list[MisconceptionRegistryEntry]:
    """Returns the entries newly marked remediated this pass."""
    now = now or datetime.now(timezone.utc)
    newly_remediated: list[MisconceptionRegistryEntry] = []

    for entry in await store.get_misconceptions(student_id):
        if entry.remediated_at is not None:
            continue  # already remediated, nothing to consolidate
        current_strength = decayed_misconception_strength(entry.decayed_strength, entry.last_observed_at, now)
        if current_strength <= MISCONCEPTION_ACTIVE_THRESHOLD:
            entry.decayed_strength = current_strength
            entry.remediated_at = now
            await store.save_misconception(entry)
            newly_remediated.append(entry)

    return newly_remediated


async def consolidate_mastery_node_states(
    store: MemoryStore, student_id: str, now: Optional[datetime] = None
) -> list[SubtopicMastery]:
    """Returns the mastery records whose persisted node_state was stale
    and has now been corrected."""
    now = now or datetime.now(timezone.utc)
    updated: list[SubtopicMastery] = []

    for mastery in await store.get_all_mastery(student_id):
        real_node_state = compute_node_state(mastery, now)
        if real_node_state != mastery.node_state:
            mastery.node_state = real_node_state
            mastery.updated_at = now
            await store.save_mastery(mastery)
            updated.append(mastery)

    return updated


async def run_memory_consolidation(
    store: MemoryStore, student_id: str, now: Optional[datetime] = None
) -> ConsolidationReport:
    remediated = await consolidate_misconceptions(store, student_id, now)
    updated = await consolidate_mastery_node_states(store, student_id, now)
    return ConsolidationReport(
        student_id=student_id,
        misconceptions_remediated=[e.misconception_id for e in remediated],
        mastery_node_states_updated=[m.subtopic_id for m in updated],
    )
