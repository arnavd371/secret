"""
Typed contracts for the Adaptive Learning Engine (spec §12): spaced-
repetition scheduling state, per (student, subtopic), independent of
Phase 5's BKT/IRT mastery state. Mastery answers "how well does this
student know this subtopic"; this answers "when will they next forget
it" — related but genuinely different questions, so this is deliberately
a separate model/store rather than more fields bolted onto
SubtopicMastery.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ReviewGrade(str, Enum):
    """This build's grading only has a binary correct/incorrect signal
    (Phase 4's grader doesn't produce FSRS's usual 4-button Again/Hard/
    Good/Easy scale) — an honest, documented simplification, same
    spirit as Phase 4's binary awarded/not-awarded marking instead of a
    continuous per-node score."""

    AGAIN = "again"
    GOOD = "good"


class ReviewState(BaseModel):
    student_id: str
    subtopic_id: str
    stability: float = Field(gt=0.0)
    difficulty: float = Field(ge=1.0, le=10.0)
    reps: int = Field(default=0, ge=0)
    lapses: int = Field(default=0, ge=0)
    last_reviewed_at: Optional[datetime] = None
    due_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
