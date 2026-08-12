"""
Typed contracts for the human-in-the-loop review/appeals queue (spec
§10.10's "human-in-the-loop review/appeals queue persistence" — this
build's real, if necessarily infrastructure-light, version of it).

Unlike the append-only logs elsewhere in this codebase (app.ia_supervisor.
disclosure_store, app.questions.response_log), a review entry has a real
lifecycle — pending, then resolved or appealed — so this store supports
genuine in-place status updates rather than only ever appending.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ReviewStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    APPEALED = "appealed"


class ReviewReason(str, Enum):
    # Phase 4: the grader's own confidence rubric came back LOW (no final
    # answer found, zero marks awarded, or a flag was raised).
    LOW_CONFIDENCE_GRADING = "low_confidence_grading"
    # Phase 4's real unsupported-answer detector: a correct final answer
    # with too little supporting work shown.
    UNSUPPORTED_ANSWER_FLAG = "unsupported_answer_flag"
    # Phase 6: the Verifier/Critic call failed or was unparseable and the
    # turn fell back to the conservative static check instead of a real
    # model critique.
    CRITIC_DEGRADED = "critic_degraded"


class ReviewQueueEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: f"REVIEW-{uuid.uuid4().hex[:12]}")
    turn_id: str
    student_id: str
    reason: ReviewReason
    summary: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: ReviewStatus = ReviewStatus.PENDING
    resolution_note: Optional[str] = None
    resolved_at: Optional[datetime] = None
