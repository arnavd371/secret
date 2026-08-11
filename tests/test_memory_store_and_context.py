from datetime import datetime, timedelta, timezone

import pytest

from app.examiner.models import ConfidenceTier
from app.memory.context_assembly import assemble_memory_context
from app.memory.models import MisconceptionRegistryEntry, NodeState, SubtopicMastery
from app.memory.store import InMemoryMemoryStore
from app.memory.write_policy import should_write_mastery_update


@pytest.mark.asyncio
async def test_memory_store_roundtrip():
    store = InMemoryMemoryStore()
    assert await store.get_mastery("s1", "topic.a") is None

    mastery = SubtopicMastery(student_id="s1", subtopic_id="topic.a", p_mastery_bkt=0.6, attempts_total=3)
    await store.save_mastery(mastery)

    fetched = await store.get_mastery("s1", "topic.a")
    assert fetched is not None
    assert fetched.p_mastery_bkt == 0.6
    assert fetched.attempts_total == 3


@pytest.mark.asyncio
async def test_memory_store_is_scoped_per_student_and_subtopic():
    store = InMemoryMemoryStore()
    await store.save_mastery(SubtopicMastery(student_id="s1", subtopic_id="topic.a", p_mastery_bkt=0.9))
    await store.save_mastery(SubtopicMastery(student_id="s2", subtopic_id="topic.a", p_mastery_bkt=0.1))
    await store.save_mastery(SubtopicMastery(student_id="s1", subtopic_id="topic.b", p_mastery_bkt=0.5))

    assert (await store.get_mastery("s1", "topic.a")).p_mastery_bkt == 0.9
    assert (await store.get_mastery("s2", "topic.a")).p_mastery_bkt == 0.1
    assert (await store.get_mastery("s1", "topic.b")).p_mastery_bkt == 0.5


@pytest.mark.asyncio
async def test_misconception_registry_roundtrip():
    store = InMemoryMemoryStore()
    assert await store.get_misconceptions("s1") == []

    entry = MisconceptionRegistryEntry(student_id="s1", misconception_id="MISC-CALC-014")
    await store.save_misconception(entry)

    entries = await store.get_misconceptions("s1")
    assert len(entries) == 1
    assert entries[0].misconception_id == "MISC-CALC-014"


@pytest.mark.asyncio
async def test_saving_misconception_again_updates_not_duplicates():
    store = InMemoryMemoryStore()
    await store.save_misconception(MisconceptionRegistryEntry(student_id="s1", misconception_id="MISC-CALC-014", occurrences=1))
    await store.save_misconception(MisconceptionRegistryEntry(student_id="s1", misconception_id="MISC-CALC-014", occurrences=2))

    entries = await store.get_misconceptions("s1")
    assert len(entries) == 1
    assert entries[0].occurrences == 2


# ---------------------------------------------------------------------------
# Write policy
# ---------------------------------------------------------------------------


def test_high_and_medium_confidence_are_write_eligible():
    assert should_write_mastery_update(ConfidenceTier.HIGH) is True
    assert should_write_mastery_update(ConfidenceTier.MEDIUM) is True


def test_low_confidence_is_not_write_eligible():
    assert should_write_mastery_update(ConfidenceTier.LOW) is False


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


def test_no_mastery_record_gives_explicit_no_history_text():
    context = assemble_memory_context(None, [])
    assert "no mastery history" in context.rendered_text
    assert context.subtopic_id is None


def test_context_includes_node_state_and_effective_mastery():
    now = datetime.now(timezone.utc)
    mastery = SubtopicMastery(student_id="s1", subtopic_id="topic.a", p_mastery_bkt=0.5, attempts_total=4, last_practiced_at=now)
    context = assemble_memory_context(mastery, [], now=now)
    assert context.subtopic_id == "topic.a"
    assert context.node_state == NodeState.PRACTICING
    assert context.effective_mastery == pytest.approx(0.5, abs=0.01)
    assert "topic.a" in context.rendered_text


def test_context_sorts_misconceptions_by_decayed_strength_and_caps_at_three():
    now = datetime.now(timezone.utc)
    mastery = SubtopicMastery(student_id="s1", subtopic_id="topic.a", attempts_total=1, last_practiced_at=now)
    misconceptions = [
        MisconceptionRegistryEntry(student_id="s1", misconception_id="MISC-A", decayed_strength=0.5, last_observed_at=now),
        MisconceptionRegistryEntry(student_id="s1", misconception_id="MISC-B", decayed_strength=0.9, last_observed_at=now),
        MisconceptionRegistryEntry(student_id="s1", misconception_id="MISC-C", decayed_strength=0.7, last_observed_at=now),
        MisconceptionRegistryEntry(student_id="s1", misconception_id="MISC-D", decayed_strength=0.3, last_observed_at=now),
    ]
    context = assemble_memory_context(mastery, misconceptions, now=now, max_misconceptions=3)
    assert context.active_misconception_ids == ["MISC-B", "MISC-C", "MISC-A"]


def test_context_excludes_remediated_misconceptions():
    now = datetime.now(timezone.utc)
    mastery = SubtopicMastery(student_id="s1", subtopic_id="topic.a", attempts_total=1, last_practiced_at=now)
    misconceptions = [
        MisconceptionRegistryEntry(
            student_id="s1", misconception_id="MISC-A", decayed_strength=0.9, last_observed_at=now, remediated_at=now
        ),
    ]
    context = assemble_memory_context(mastery, misconceptions, now=now)
    assert context.active_misconception_ids == []


def test_context_excludes_fully_decayed_misconceptions():
    now = datetime.now(timezone.utc)
    mastery = SubtopicMastery(student_id="s1", subtopic_id="topic.a", attempts_total=1, last_practiced_at=now)
    stale = now - timedelta(days=365)  # far past tau=30, strength decays to ~0
    misconceptions = [
        MisconceptionRegistryEntry(student_id="s1", misconception_id="MISC-A", decayed_strength=1.0, last_observed_at=stale),
    ]
    context = assemble_memory_context(mastery, misconceptions, now=now)
    assert context.active_misconception_ids == []


def test_rendered_text_respects_word_budget():
    now = datetime.now(timezone.utc)
    mastery = SubtopicMastery(student_id="s1", subtopic_id="topic.a", attempts_total=1, last_practiced_at=now)
    misconceptions = [
        MisconceptionRegistryEntry(student_id="s1", misconception_id=f"MISC-{i}", decayed_strength=0.9, last_observed_at=now)
        for i in range(3)
    ]
    context = assemble_memory_context(mastery, misconceptions, now=now, word_budget=5)
    assert len(context.rendered_text.split()) <= 6  # 5 words + possible "..." token
