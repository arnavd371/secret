import pytest

from app.review_queue.models import ReviewReason, ReviewStatus
from app.review_queue.queue import appeal_review, enqueue_review, resolve_review
from app.review_queue.store import InMemoryReviewQueueStore


@pytest.mark.asyncio
async def test_enqueue_creates_a_pending_entry():
    store = InMemoryReviewQueueStore()
    entry = await enqueue_review(store, "turn-1", "stu-1", ReviewReason.LOW_CONFIDENCE_GRADING, "no final answer")

    assert entry.status == ReviewStatus.PENDING
    assert entry.turn_id == "turn-1"
    fetched = await store.get(entry.entry_id)
    assert fetched == entry


@pytest.mark.asyncio
async def test_list_pending_excludes_resolved_entries():
    store = InMemoryReviewQueueStore()
    e1 = await enqueue_review(store, "t1", "stu-1", ReviewReason.LOW_CONFIDENCE_GRADING, "a")
    await enqueue_review(store, "t2", "stu-1", ReviewReason.CRITIC_DEGRADED, "b")

    await resolve_review(store, e1.entry_id, "confirmed fine")

    pending = await store.list_pending("stu-1")
    assert len(pending) == 1
    assert pending[0].reason == ReviewReason.CRITIC_DEGRADED


@pytest.mark.asyncio
async def test_list_pending_is_scoped_per_student():
    store = InMemoryReviewQueueStore()
    await enqueue_review(store, "t1", "stu-1", ReviewReason.LOW_CONFIDENCE_GRADING, "a")
    await enqueue_review(store, "t2", "stu-2", ReviewReason.LOW_CONFIDENCE_GRADING, "b")

    assert len(await store.list_pending("stu-1")) == 1
    assert len(await store.list_pending("stu-2")) == 1
    assert len(await store.list_pending()) == 2


@pytest.mark.asyncio
async def test_resolve_sets_status_and_note_and_timestamp():
    store = InMemoryReviewQueueStore()
    entry = await enqueue_review(store, "t1", "stu-1", ReviewReason.UNSUPPORTED_ANSWER_FLAG, "check coverage")

    resolved = await resolve_review(store, entry.entry_id, "confirmed correct, method shown verbally")

    assert resolved.status == ReviewStatus.RESOLVED
    assert resolved.resolution_note == "confirmed correct, method shown verbally"
    assert resolved.resolved_at is not None


@pytest.mark.asyncio
async def test_resolve_unknown_entry_returns_none():
    store = InMemoryReviewQueueStore()
    assert await resolve_review(store, "nonexistent", "note") is None


@pytest.mark.asyncio
async def test_appeal_moves_a_resolved_entry_to_appealed_not_back_to_pending():
    store = InMemoryReviewQueueStore()
    entry = await enqueue_review(store, "t1", "stu-1", ReviewReason.LOW_CONFIDENCE_GRADING, "a")
    await resolve_review(store, entry.entry_id, "first resolution")

    appealed = await appeal_review(store, entry.entry_id, "student disputes this")

    assert appealed.status == ReviewStatus.APPEALED
    assert appealed.resolution_note == "student disputes this"
    # Appealed entries are not "pending" - a human still needs to look,
    # but the queue can tell "never reviewed" apart from "disputed".
    pending = await store.list_pending("stu-1")
    assert appealed not in pending


@pytest.mark.asyncio
async def test_appeal_unknown_entry_returns_none():
    store = InMemoryReviewQueueStore()
    assert await appeal_review(store, "nonexistent", "note") is None
