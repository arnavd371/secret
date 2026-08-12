"""
Tests for the memory consolidation batch job: real decay/node_state math
reused from Phase 5, checked against hand-computable scenarios.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.memory.consolidation import (
    consolidate_mastery_node_states,
    consolidate_misconceptions,
    run_memory_consolidation,
)
from app.memory.models import MisconceptionRegistryEntry, NodeState, SubtopicMastery
from app.memory.store import InMemoryMemoryStore

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_a_fully_decayed_misconception_is_remediated():
    store = InMemoryMemoryStore()
    await store.save_misconception(
        MisconceptionRegistryEntry(
            student_id="stu-1",
            misconception_id="MISC-CALC-010",
            decayed_strength=0.3,
            last_observed_at=NOW - timedelta(days=120),
        )
    )

    remediated = await consolidate_misconceptions(store, "stu-1", NOW)

    assert len(remediated) == 1
    assert remediated[0].misconception_id == "MISC-CALC-010"
    assert remediated[0].remediated_at == NOW


@pytest.mark.asyncio
async def test_a_recently_reinforced_misconception_is_not_remediated():
    store = InMemoryMemoryStore()
    await store.save_misconception(
        MisconceptionRegistryEntry(
            student_id="stu-1",
            misconception_id="MISC-ALG-003",
            decayed_strength=1.0,
            last_observed_at=NOW - timedelta(days=1),
        )
    )

    remediated = await consolidate_misconceptions(store, "stu-1", NOW)
    assert remediated == []


@pytest.mark.asyncio
async def test_an_already_remediated_misconception_is_left_alone():
    store = InMemoryMemoryStore()
    await store.save_misconception(
        MisconceptionRegistryEntry(
            student_id="stu-1",
            misconception_id="MISC-CALC-010",
            decayed_strength=0.3,
            last_observed_at=NOW - timedelta(days=120),
            remediated_at=NOW - timedelta(days=10),
        )
    )

    remediated = await consolidate_misconceptions(store, "stu-1", NOW)
    assert remediated == []


@pytest.mark.asyncio
async def test_a_stale_mastered_record_recomputes_to_decayed():
    store = InMemoryMemoryStore()
    await store.save_mastery(
        SubtopicMastery(
            student_id="stu-1",
            subtopic_id="calc.chain_rule",
            p_mastery_bkt=0.95,
            attempts_total=5,
            attempts_correct=5,
            node_state=NodeState.MASTERED,
            last_practiced_at=NOW - timedelta(days=60),
        )
    )

    updated = await consolidate_mastery_node_states(store, "stu-1", NOW)

    assert len(updated) == 1
    assert updated[0].node_state == NodeState.DECAYED
    persisted = await store.get_mastery("stu-1", "calc.chain_rule")
    assert persisted.node_state == NodeState.DECAYED


@pytest.mark.asyncio
async def test_a_correctly_labeled_record_is_not_rewritten():
    store = InMemoryMemoryStore()
    await store.save_mastery(
        SubtopicMastery(
            student_id="stu-1",
            subtopic_id="calc.chain_rule",
            p_mastery_bkt=0.95,
            attempts_total=5,
            attempts_correct=5,
            node_state=NodeState.MASTERED,
            last_practiced_at=NOW - timedelta(days=1),
        )
    )

    updated = await consolidate_mastery_node_states(store, "stu-1", NOW)
    assert updated == []


@pytest.mark.asyncio
async def test_run_memory_consolidation_combines_both_and_is_scoped_per_student():
    store = InMemoryMemoryStore()
    await store.save_misconception(
        MisconceptionRegistryEntry(
            student_id="stu-1",
            misconception_id="MISC-CALC-010",
            decayed_strength=0.3,
            last_observed_at=NOW - timedelta(days=120),
        )
    )
    await store.save_mastery(
        SubtopicMastery(
            student_id="stu-1",
            subtopic_id="calc.chain_rule",
            p_mastery_bkt=0.95,
            attempts_total=5,
            attempts_correct=5,
            node_state=NodeState.MASTERED,
            last_practiced_at=NOW - timedelta(days=60),
        )
    )
    # A second student's records must not be touched by stu-1's run.
    await store.save_mastery(
        SubtopicMastery(
            student_id="stu-2",
            subtopic_id="calc.chain_rule",
            p_mastery_bkt=0.95,
            attempts_total=5,
            attempts_correct=5,
            node_state=NodeState.MASTERED,
            last_practiced_at=NOW - timedelta(days=60),
        )
    )

    report = await run_memory_consolidation(store, "stu-1", NOW)

    assert report.misconceptions_remediated == ["MISC-CALC-010"]
    assert report.mastery_node_states_updated == ["calc.chain_rule"]

    stu2_record = await store.get_mastery("stu-2", "calc.chain_rule")
    assert stu2_record.node_state == NodeState.MASTERED  # untouched


@pytest.mark.asyncio
async def test_get_all_mastery_is_scoped_per_student():
    store = InMemoryMemoryStore()
    await store.save_mastery(SubtopicMastery(student_id="stu-1", subtopic_id="topic.a"))
    await store.save_mastery(SubtopicMastery(student_id="stu-1", subtopic_id="topic.b"))
    await store.save_mastery(SubtopicMastery(student_id="stu-2", subtopic_id="topic.a"))

    stu1_records = await store.get_all_mastery("stu-1")
    assert {r.subtopic_id for r in stu1_records} == {"topic.a", "topic.b"}
