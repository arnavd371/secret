"""
Tests for the persistence-facing scheduling orchestration: record_review,
get_due_subtopics, most_overdue_subtopic. All against InMemoryReviewStateStore
with explicit `now` timestamps, so due/not-due behavior is deterministic.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.adaptive.models import ReviewUrgency
from app.adaptive.scheduler import (
    MILD_RETRIEVABILITY_THRESHOLD,
    SIGNIFICANT_RETRIEVABILITY_THRESHOLD,
    compute_review_urgency,
    get_due_subtopics,
    get_due_subtopics_with_urgency,
    most_overdue_subtopic,
    record_review,
)
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


# ---------------------------------------------------------------------------
# Phase 15: mastery-threshold review bands (real retrievability-based
# urgency, not a flat days-overdue count)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_urgency_right_at_due_at_is_mild():
    store = InMemoryReviewStateStore()
    state = await record_review(store, "stu-1", "topic.a", True, NOW)
    # due_at is defined as the point where retrievability = 0.9 (desired
    # retention) - well above the MILD threshold.
    urgency, r = compute_review_urgency(state, state.due_at)
    assert urgency == ReviewUrgency.MILD
    assert r == pytest.approx(0.9, abs=0.01)


@pytest.mark.asyncio
async def test_urgency_far_past_due_is_critical():
    store = InMemoryReviewStateStore()
    state = await record_review(store, "stu-1", "topic.a", True, NOW)
    urgency, r = compute_review_urgency(state, state.due_at + timedelta(days=365))
    assert urgency == ReviewUrgency.CRITICAL
    assert r < SIGNIFICANT_RETRIEVABILITY_THRESHOLD


@pytest.mark.asyncio
async def test_urgency_bands_are_ordered_by_time_elapsed():
    store = InMemoryReviewStateStore()
    state = await record_review(store, "stu-1", "topic.a", True, NOW)

    _, r_mild = compute_review_urgency(state, state.due_at)
    _, r_significant = compute_review_urgency(state, state.due_at + timedelta(days=5))
    _, r_critical = compute_review_urgency(state, state.due_at + timedelta(days=50))
    assert r_mild > r_significant > r_critical


@pytest.mark.asyncio
async def test_lower_stability_item_is_more_urgent_at_equal_days_overdue():
    """The real point of retrievability-based ranking over a flat
    days-overdue count: two items overdue by the same number of days can
    be at genuinely different forgetting risk if their stabilities
    differ."""
    store = InMemoryReviewStateStore()
    stable = await record_review(store, "stu-1", "topic.stable", True, NOW)  # stability 2.5
    fragile = await record_review(store, "stu-1", "topic.fragile", False, NOW)  # stability 0.5

    query_time = NOW + timedelta(days=30)
    _, r_stable = compute_review_urgency(stable, query_time)
    _, r_fragile = compute_review_urgency(fragile, query_time)
    assert r_fragile < r_stable


@pytest.mark.asyncio
async def test_get_due_subtopics_with_urgency_ranks_most_forgotten_first():
    store = InMemoryReviewStateStore()
    await record_review(store, "stu-1", "topic.stable", True, NOW)
    await record_review(store, "stu-1", "topic.fragile", False, NOW)

    query_time = NOW + timedelta(days=30)
    ranked = await get_due_subtopics_with_urgency(store, "stu-1", query_time)

    assert [state.subtopic_id for state, _u, _r in ranked] == ["topic.fragile", "topic.stable"]
    # retrievability strictly ascending (most urgent first)
    assert ranked[0][2] < ranked[1][2]


@pytest.mark.asyncio
async def test_get_due_subtopics_with_urgency_excludes_not_yet_due_items():
    store = InMemoryReviewStateStore()
    await record_review(store, "stu-1", "topic.a", True, NOW)
    ranked = await get_due_subtopics_with_urgency(store, "stu-1", NOW)
    assert ranked == []


def test_retrievability_thresholds_are_ordered():
    assert 0.0 < SIGNIFICANT_RETRIEVABILITY_THRESHOLD < MILD_RETRIEVABILITY_THRESHOLD < 1.0
