"""
Typed contracts for the Student Memory System (spec §4).

Persisted, per-(student, subtopic) mastery state (§4.2/§4.3), a per-
student misconception registry (§4.6), and the deterministic
budgeted-context-assembly output (§4.12) that gets injected into the
Tutor prompt's STUDENT MASTERY CONTEXT / ACTIVE MISCONCEPTIONS slots.

Not modeled here (later-phase non-goals):
  - Episodic event log (§4.1's "episodic_events" layer) — this phase
    persists current-state mastery/misconceptions directly, not an
    append-only history of every attempt.
  - Learner preferences (§4.2's learner_preferences) — not read by
    anything yet.
  - Automatic misconception *detection* populating this registry (spec
    §8's Misconception Diagnostician Agent, a separate later phase) —
    the registry here is real, tested storage/decay with an explicit
    write API, nothing auto-detects and calls it yet.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class NodeState(str, Enum):
    UNSEEN = "unseen"
    INTRODUCED = "introduced"
    PRACTICING = "practicing"
    CONSOLIDATING = "consolidating"
    MASTERED = "mastered"
    DECAYED = "decayed"


class SubtopicMastery(BaseModel):
    student_id: str
    subtopic_id: str
    p_mastery_bkt: float = Field(default=0.10, ge=0.0, le=1.0)
    theta_irt: float = 0.0
    se_theta: float = Field(default=1.0, gt=0.0)
    attempts_total: int = Field(default=0, ge=0)
    attempts_correct: int = Field(default=0, ge=0)
    node_state: NodeState = NodeState.UNSEEN
    last_practiced_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MisconceptionRegistryEntry(BaseModel):
    student_id: str
    misconception_id: str
    occurrences: int = Field(default=1, ge=1)
    first_observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decayed_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    remediated_at: Optional[datetime] = None


class MemoryReadContext(BaseModel):
    """Deterministic, fixed-shape output of budgeted context assembly
    (§4.12) — what actually gets injected into the Tutor prompt."""

    subtopic_id: Optional[str] = None
    node_state: Optional[NodeState] = None
    effective_mastery: Optional[float] = None
    active_misconception_ids: list[str] = Field(default_factory=list)
    rendered_text: str = ""
