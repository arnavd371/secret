"""
Tests for the persistence-facing scheduling orchestration: record_review,
get_due_subtopics, most_overdue_subtopic. All against InMemoryReviewStateStore
with explicit `now` timestamps, so due/not-due behavior is deterministic.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.adaptive.scheduler import get_due_subtopics, most_overdue_subtopic, record_review
from app.adaptive.store import InMemoryReviewStateStore

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_first_review_creates_a_new_state():
    store = InMemoryReviewStateStore()
    state = await record_review(store, "stu-1", "algebra.quadratics", True, NOW)
    assert state.reps == 1
    assert state.lapses == 0
    assert state.last_reviewed_at == NOW
    assert state.due_at > NOW


@pytest.mark.asyncio
async def test_a_fresh_review_is_not_immediately_due():
    store = InMemoryReviewStateStore()
    await record_review(store, "stu-1", "algebra.quadratics", True, NOW)
    due = await get_due_subtopics(store, "stu-1", NOW)
    assert due == []


@pytest.mark.asyncio
async def test_a_review_becomes_due_after_its_interval_elapses():
    store = InMemoryReviewStateStore()
    state = await record_review(store, "stu-1", "algebra.quadratics", True, NOW)
    due = await get_due_subtopics(store, "stu-1", state.due_at + timedelta(hours=1))
    assert [d.subtopic_id for d in due] == ["algebra.quadratics"]


@pytest.mark.asyncio
async def test_a_failed_review_becomes_due_sooner_than_a_successful_one():
    store = InMemoryReviewStateStore()
    good = await record_review(store, "stu-1", "topic.a", True, NOW)
    bad = await record_review(store, "stu-1", "topic.b", False, NOW)
    assert bad.due_at < good.due_at


@pytest.mark.asyncio
async def test_repeated_reviews_update_the_same_state_not_a_duplicate():
    store = InMemoryReviewStateStore()
    first = await record_review(store, "stu-1", "algebra.quadratics", True, NOW)
    second = await record_review(store, "stu-1", "algebra.quadratics", True, first.due_at)
    assert second.reps == 2
    assert second.stability > first.stability
    all_states = await store.get_all_for_student("stu-1")
    assert len(all_states) == 1


@pytest.mark.asyncio
async def test_most_overdue_subtopic_picks_the_earliest_due_date():
    store = InMemoryReviewStateStore()
    await record_review(store, "stu-1", "topic.a", True, NOW)
    bad = await record_review(store, "stu-1", "topic.b", False, NOW)

    result = await most_overdue_subtopic(store, "stu-1", bad.due_at + timedelta(days=100))
    assert result == "topic.b"


@pytest.mark.asyncio
async def test_most_overdue_subtopic_is_none_when_nothing_is_due():
    store = InMemoryReviewStateStore()
    await record_review(store, "stu-1", "topic.a", True, NOW)
    result = await most_overdue_subtopic(store, "stu-1", NOW)
    assert result is None


@pytest.mark.asyncio
async def test_never_reviewed_subtopic_is_never_due():
    store = InMemoryReviewStateStore()
    far_future = NOW + timedelta(days=3650)
    due = await get_due_subtopics(store, "stu-1", far_future)
    assert due == []


@pytest.mark.asyncio
async def test_due_states_are_scoped_per_student():
    store = InMemoryReviewStateStore()
    state = await record_review(store, "stu-1", "topic.a", False, NOW)
    due_for_other_student = await get_due_subtopics(store, "stu-2", state.due_at + timedelta(days=1))
    assert due_for_other_student == []
