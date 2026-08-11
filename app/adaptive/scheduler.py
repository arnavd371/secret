"""
Scheduling orchestration (spec §12): updates FSRS state after a real
grading event, and answers "what's due for review right now" — the
query the exam_prep decision branch and the orchestrator both consult.

No LLM call anywhere in this module; scheduling is entirely deterministic
math over persisted state, same "no generative model in the critical
path" posture as app.cas.solver and app.examiner.grader.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from app.adaptive.fsrs import initial_state, next_interval_days, update_state
from app.adaptive.models import ReviewGrade, ReviewState
from app.adaptive.store import ReviewStateStore


def _days_between(a: datetime, b: datetime) -> float:
    return abs((b - a).total_seconds()) / 86400.0


async def record_review(
    store: ReviewStateStore, student_id: str, subtopic_id: str, correct: bool, now: Optional[datetime] = None
) -> ReviewState:
    now = now or datetime.now(timezone.utc)
    grade = ReviewGrade.GOOD if correct else ReviewGrade.AGAIN

    existing = await store.get(student_id, subtopic_id)
    if existing is None:
        stability, difficulty = initial_state(grade)
        reps = 1
        lapses = 0 if correct else 1
    else:
        days_elapsed = _days_between(existing.last_reviewed_at, now) if existing.last_reviewed_at else 0.0
        stability, difficulty = update_state(existing.stability, existing.difficulty, days_elapsed, grade)
        reps = existing.reps + 1
        lapses = existing.lapses + (0 if correct else 1)

    interval_days = next_interval_days(stability)
    due_at = now + timedelta(days=interval_days)

    state = ReviewState(
        student_id=student_id,
        subtopic_id=subtopic_id,
        stability=stability,
        difficulty=difficulty,
        reps=reps,
        lapses=lapses,
        last_reviewed_at=now,
        due_at=due_at,
    )
    await store.save(state)
    return state


async def get_due_subtopics(
    store: ReviewStateStore, student_id: str, now: Optional[datetime] = None
) -> list[ReviewState]:
    """Every reviewed subtopic whose due_at has passed, most-overdue
    first. A subtopic never reviewed at all isn't "due" by this engine —
    it hasn't entered spaced repetition yet (that's Phase 5's
    unseen/introduced node-state territory, a deliberate scope line, not
    this engine's job to bootstrap)."""
    now = now or datetime.now(timezone.utc)
    all_states = await store.get_all_for_student(student_id)
    due = [s for s in all_states if s.due_at <= now]
    due.sort(key=lambda s: s.due_at)
    return due


async def most_overdue_subtopic(
    store: ReviewStateStore, student_id: str, now: Optional[datetime] = None
) -> Optional[str]:
    due = await get_due_subtopics(store, student_id, now)
    return due[0].subtopic_id if due else None
