"""
Real operations over the review queue (spec §10.10): enqueue, resolve,
appeal. All pure state transitions over a ReviewQueueStore, no model
call, no I/O beyond the store itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.review_queue.models import ReviewQueueEntry, ReviewReason, ReviewStatus
from app.review_queue.store import ReviewQueueStore


async def enqueue_review(
    store: ReviewQueueStore, turn_id: str, student_id: str, reason: ReviewReason, summary: str
) -> ReviewQueueEntry:
    entry = ReviewQueueEntry(turn_id=turn_id, student_id=student_id, reason=reason, summary=summary)
    await store.add(entry)
    return entry


async def resolve_review(store: ReviewQueueStore, entry_id: str, resolution_note: str) -> Optional[ReviewQueueEntry]:
    """None if entry_id doesn't exist. A non-PENDING entry can still be
    re-resolved (e.g. a resolved entry later found to need a different
    outcome) — this doesn't reject that, it just overwrites the
    resolution, same as any other real correction."""
    entry = await store.get(entry_id)
    if entry is None:
        return None
    entry.status = ReviewStatus.RESOLVED
    entry.resolution_note = resolution_note
    entry.resolved_at = datetime.now(timezone.utc)
    await store.update(entry)
    return entry


async def appeal_review(store: ReviewQueueStore, entry_id: str, appeal_note: str) -> Optional[ReviewQueueEntry]:
    """A resolved entry the student (or a teacher) disputes — moves back
    to a real, distinct APPEALED status rather than silently reopening
    as PENDING, so the queue can tell "never reviewed" apart from
    "reviewed, then disputed" when a human looks at it."""
    entry = await store.get(entry_id)
    if entry is None:
        return None
    entry.status = ReviewStatus.APPEALED
    entry.resolution_note = appeal_note
    entry.resolved_at = datetime.now(timezone.utc)
    await store.update(entry)
    return entry
