"""
Real round-trip tests for every Sqlite*Store: write through one store
instance, then read back through a *fresh* instance (a brand-new object,
no shared in-process state) against the same tmp_path db file. That's
the actual risk area persistence adds over the in-memory stores this
codebase already has thorough business-logic coverage for - a fresh
instance forces every read to go through real SQL, not a warm Python
dict, so this genuinely proves the data survived to disk.
"""

from __future__ import annotations

import pytest

from app.adaptive.models import ReviewState
from app.adaptive.store import SqliteReviewStateStore
from app.guardrail_metrics.models import GuardrailTurnSignals
from app.guardrail_metrics.store import SqliteGuardrailMetricsStore
from app.ia_supervisor.disclosure_store import SqliteDisclosureStore
from app.ia_supervisor.models import DisclosureAssistanceType, DisclosureEntry, IAProjectState, IAStage
from app.ia_supervisor.project_store import SqliteIAProjectStateStore
from app.memory.models import MisconceptionRegistryEntry, SubtopicMastery
from app.memory.store import SqliteMemoryStore
from app.questions.response_log import ItemResponseRecord, SqliteResponseLogStore
from app.review_queue.models import ReviewQueueEntry, ReviewReason, ReviewStatus
from app.review_queue.store import SqliteReviewQueueStore
from app.session.state import ProblemSessionState, SqliteSessionStateStore


def _db_path(tmp_path) -> str:
    return str(tmp_path / "tutor.db")


# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_state_survives_a_fresh_store_instance(tmp_path):
    path = _db_path(tmp_path)
    await SqliteSessionStateStore(path).save(
        ProblemSessionState(session_id="s1", problem_id="p1", attempt_count=2, hint_ladder_level=1)
    )

    reloaded = await SqliteSessionStateStore(path).get("s1", "p1")
    assert reloaded.attempt_count == 2
    assert reloaded.hint_ladder_level == 1


@pytest.mark.asyncio
async def test_session_state_get_on_unknown_session_returns_a_fresh_default(tmp_path):
    state = await SqliteSessionStateStore(_db_path(tmp_path)).get("never-seen", "p1")
    assert state.attempt_count == 0
    assert state.session_id == "never-seen"


@pytest.mark.asyncio
async def test_session_state_save_upserts_not_duplicates(tmp_path):
    path = _db_path(tmp_path)
    store = SqliteSessionStateStore(path)
    await store.save(ProblemSessionState(session_id="s1", attempt_count=1))
    await store.save(ProblemSessionState(session_id="s1", attempt_count=5))

    reloaded = await SqliteSessionStateStore(path).get("s1", None)
    assert reloaded.attempt_count == 5


# ---------------------------------------------------------------------------
# memory (mastery + misconceptions)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mastery_survives_a_fresh_store_instance(tmp_path):
    path = _db_path(tmp_path)
    await SqliteMemoryStore(path).save_mastery(
        SubtopicMastery(student_id="stu-1", subtopic_id="calc.chain_rule", p_mastery_bkt=0.75)
    )

    reloaded = await SqliteMemoryStore(path).get_mastery("stu-1", "calc.chain_rule")
    assert reloaded is not None
    assert reloaded.p_mastery_bkt == 0.75


@pytest.mark.asyncio
async def test_get_all_mastery_and_misconceptions_scoped_per_student(tmp_path):
    path = _db_path(tmp_path)
    store = SqliteMemoryStore(path)
    await store.save_mastery(SubtopicMastery(student_id="stu-1", subtopic_id="a"))
    await store.save_mastery(SubtopicMastery(student_id="stu-1", subtopic_id="b"))
    await store.save_mastery(SubtopicMastery(student_id="stu-2", subtopic_id="a"))
    await store.save_misconception(MisconceptionRegistryEntry(student_id="stu-1", misconception_id="MISC-1"))

    fresh = SqliteMemoryStore(path)
    assert len(await fresh.get_all_mastery("stu-1")) == 2
    assert len(await fresh.get_all_mastery("stu-2")) == 1
    assert len(await fresh.get_misconceptions("stu-1")) == 1


@pytest.mark.asyncio
async def test_memory_erase_student_really_deletes_and_returns_the_count(tmp_path):
    path = _db_path(tmp_path)
    store = SqliteMemoryStore(path)
    await store.save_mastery(SubtopicMastery(student_id="stu-1", subtopic_id="a"))
    await store.save_misconception(MisconceptionRegistryEntry(student_id="stu-1", misconception_id="MISC-1"))

    erased = await SqliteMemoryStore(path).erase_student("stu-1")
    assert erased == 2
    assert await SqliteMemoryStore(path).get_all_mastery("stu-1") == []


# ---------------------------------------------------------------------------
# adaptive review state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_state_survives_a_fresh_store_instance(tmp_path):
    path = _db_path(tmp_path)
    await SqliteReviewStateStore(path).save(
        ReviewState(student_id="stu-1", subtopic_id="calc.chain_rule", stability=5.0, difficulty=3.0)
    )

    reloaded = await SqliteReviewStateStore(path).get("stu-1", "calc.chain_rule")
    assert reloaded is not None
    assert reloaded.stability == 5.0


@pytest.mark.asyncio
async def test_review_state_erase_student(tmp_path):
    path = _db_path(tmp_path)
    await SqliteReviewStateStore(path).save(
        ReviewState(student_id="stu-1", subtopic_id="a", stability=1.0, difficulty=5.0)
    )
    erased = await SqliteReviewStateStore(path).erase_student("stu-1")
    assert erased == 1
    assert await SqliteReviewStateStore(path).get_all_for_student("stu-1") == []


# ---------------------------------------------------------------------------
# IA project state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ia_project_state_survives_a_fresh_store_instance(tmp_path):
    path = _db_path(tmp_path)
    await SqliteIAProjectStateStore(path).save(
        IAProjectState(student_id="stu-1", project_id="ia-1", stage=IAStage.METHODOLOGY)
    )

    reloaded = await SqliteIAProjectStateStore(path).get("stu-1", "ia-1")
    assert reloaded is not None
    assert reloaded.stage == IAStage.METHODOLOGY


@pytest.mark.asyncio
async def test_ia_project_state_get_all_for_student_and_erase(tmp_path):
    path = _db_path(tmp_path)
    store = SqliteIAProjectStateStore(path)
    await store.save(IAProjectState(student_id="stu-1", project_id="ia-1", stage=IAStage.TOPIC_SELECTION))
    await store.save(IAProjectState(student_id="stu-1", project_id="ia-2", stage=IAStage.DRAFTING))

    assert len(await SqliteIAProjectStateStore(path).get_all_for_student("stu-1")) == 2
    erased = await SqliteIAProjectStateStore(path).erase_student("stu-1")
    assert erased == 2


# ---------------------------------------------------------------------------
# disclosure log (append-only, no erase)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disclosure_log_is_append_only_and_survives_a_fresh_instance(tmp_path):
    path = _db_path(tmp_path)
    store = SqliteDisclosureStore(path)
    await store.add(
        DisclosureEntry(
            student_id="stu-1",
            project_id="ia-1",
            stage=IAStage.TOPIC_SELECTION,
            assistance_type=DisclosureAssistanceType.COACHING,
            summary="first",
        )
    )
    await store.add(
        DisclosureEntry(
            student_id="stu-1",
            project_id="ia-1",
            stage=IAStage.DRAFTING,
            assistance_type=DisclosureAssistanceType.GHOSTWRITING_REQUEST_REFUSED,
            summary="second",
        )
    )

    fresh = SqliteDisclosureStore(path)
    entries = await fresh.get_all("stu-1", "ia-1")
    assert [e.summary for e in entries] == ["first", "second"]  # sorted by timestamp
    assert len(await fresh.get_all_for_student("stu-1")) == 2
    assert not hasattr(fresh, "erase_student")


# ---------------------------------------------------------------------------
# review queue (real lifecycle: pending -> resolved/appealed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_queue_entry_survives_a_fresh_store_instance(tmp_path):
    path = _db_path(tmp_path)
    entry = ReviewQueueEntry(
        turn_id="turn-1", student_id="stu-1", reason=ReviewReason.LOW_CONFIDENCE_GRADING, summary="x"
    )
    await SqliteReviewQueueStore(path).add(entry)

    reloaded = await SqliteReviewQueueStore(path).get(entry.entry_id)
    assert reloaded is not None
    assert reloaded.status == ReviewStatus.PENDING


@pytest.mark.asyncio
async def test_review_queue_update_persists_the_real_status_transition(tmp_path):
    path = _db_path(tmp_path)
    entry = ReviewQueueEntry(
        turn_id="turn-1", student_id="stu-1", reason=ReviewReason.CRITIC_DEGRADED, summary="x"
    )
    store = SqliteReviewQueueStore(path)
    await store.add(entry)

    entry.status = ReviewStatus.RESOLVED
    entry.resolution_note = "looked fine"
    await store.update(entry)

    reloaded = await SqliteReviewQueueStore(path).get(entry.entry_id)
    assert reloaded.status == ReviewStatus.RESOLVED
    assert reloaded.resolution_note == "looked fine"
    # a resolved entry must drop out of list_pending
    assert reloaded.entry_id not in [e.entry_id for e in await SqliteReviewQueueStore(path).list_pending()]


@pytest.mark.asyncio
async def test_review_queue_erase_student(tmp_path):
    path = _db_path(tmp_path)
    await SqliteReviewQueueStore(path).add(
        ReviewQueueEntry(turn_id="t1", student_id="stu-1", reason=ReviewReason.LOW_CONFIDENCE_GRADING, summary="x")
    )
    erased = await SqliteReviewQueueStore(path).erase_student("stu-1")
    assert erased == 1


# ---------------------------------------------------------------------------
# response log
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_log_survives_a_fresh_store_instance(tmp_path):
    path = _db_path(tmp_path)
    store = SqliteResponseLogStore(path)
    await store.add(ItemResponseRecord(template_id="T1", student_id="stu-1", correct=True))
    await store.add(ItemResponseRecord(template_id="T1", student_id="stu-2", correct=False))

    fresh = SqliteResponseLogStore(path)
    records = await fresh.get_all("T1")
    assert len(records) == 2
    assert len(await fresh.get_all_for_student("stu-1")) == 1


@pytest.mark.asyncio
async def test_response_log_erase_student(tmp_path):
    path = _db_path(tmp_path)
    await SqliteResponseLogStore(path).add(ItemResponseRecord(template_id="T1", student_id="stu-1", correct=True))
    erased = await SqliteResponseLogStore(path).erase_student("stu-1")
    assert erased == 1


# ---------------------------------------------------------------------------
# guardrail metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guardrail_signals_survive_a_fresh_store_instance(tmp_path):
    path = _db_path(tmp_path)
    store = SqliteGuardrailMetricsStore(path)
    await store.add(GuardrailTurnSignals(turn_id="turn-1", leak_check_triggered=True, fell_back_to_template=True))
    await store.add(GuardrailTurnSignals(turn_id="turn-2", critic_verdict="pass"))

    fresh = SqliteGuardrailMetricsStore(path)
    records = await fresh.get_all()
    assert {r.turn_id for r in records} == {"turn-1", "turn-2"}


# ---------------------------------------------------------------------------
# cross-store: several stores sharing one db file don't collide
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_stores_share_one_db_file_without_colliding(tmp_path):
    path = _db_path(tmp_path)
    await SqliteSessionStateStore(path).save(ProblemSessionState(session_id="s1", attempt_count=3))
    await SqliteMemoryStore(path).save_mastery(SubtopicMastery(student_id="stu-1", subtopic_id="a"))
    await SqliteReviewQueueStore(path).add(
        ReviewQueueEntry(turn_id="t1", student_id="stu-1", reason=ReviewReason.LOW_CONFIDENCE_GRADING, summary="x")
    )

    assert (await SqliteSessionStateStore(path).get("s1", None)).attempt_count == 3
    assert len(await SqliteMemoryStore(path).get_all_mastery("stu-1")) == 1
    assert len(await SqliteReviewQueueStore(path).list_pending("stu-1")) == 1
