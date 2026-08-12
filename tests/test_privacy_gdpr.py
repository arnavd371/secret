"""
Tests for GDPR export/erasure: real data across every relevant store,
with the disclosure log's deliberate retention exemption explicitly
verified, not just assumed.
"""

import pytest

from app.adaptive.scheduler import record_review
from app.adaptive.store import InMemoryReviewStateStore
from app.ia_supervisor.disclosure_store import InMemoryDisclosureStore
from app.ia_supervisor.models import DisclosureAssistanceType, DisclosureEntry, IAProjectState, IAStage
from app.ia_supervisor.project_store import InMemoryIAProjectStateStore
from app.memory.models import MisconceptionRegistryEntry, SubtopicMastery
from app.memory.store import InMemoryMemoryStore
from app.privacy.gdpr import erase_student_data, export_student_data
from app.questions.response_log import InMemoryResponseLogStore, ItemResponseRecord
from app.review_queue.models import ReviewReason
from app.review_queue.queue import enqueue_review
from app.review_queue.store import InMemoryReviewQueueStore

STUDENT = "stu-privacy-1"
OTHER_STUDENT = "stu-privacy-2"


class _Stores:
    def __init__(self) -> None:
        self.memory = InMemoryMemoryStore()
        self.review = InMemoryReviewStateStore()
        self.ia_project = InMemoryIAProjectStateStore()
        self.disclosure = InMemoryDisclosureStore()
        self.review_queue = InMemoryReviewQueueStore()
        self.response_log = InMemoryResponseLogStore()


async def _seed(stores: _Stores, student_id: str) -> None:
    await stores.memory.save_mastery(SubtopicMastery(student_id=student_id, subtopic_id="calc.chain_rule"))
    await stores.memory.save_misconception(
        MisconceptionRegistryEntry(student_id=student_id, misconception_id="MISC-CALC-010")
    )
    await record_review(stores.review, student_id, "calc.chain_rule", True)
    await stores.ia_project.save(
        IAProjectState(student_id=student_id, project_id="ia-1", stage=IAStage.TOPIC_SELECTION)
    )
    await stores.disclosure.add(
        DisclosureEntry(
            student_id=student_id,
            project_id="ia-1",
            stage=IAStage.TOPIC_SELECTION,
            assistance_type=DisclosureAssistanceType.COACHING,
            summary="coaching given",
        )
    )
    await enqueue_review(stores.review_queue, "turn-1", student_id, ReviewReason.LOW_CONFIDENCE_GRADING, "x")
    await stores.response_log.add(ItemResponseRecord(template_id="T1", student_id=student_id, correct=True))


async def _export(stores: _Stores, student_id: str):
    return await export_student_data(
        student_id,
        memory_store=stores.memory,
        review_store=stores.review,
        ia_project_store=stores.ia_project,
        disclosure_store=stores.disclosure,
        review_queue_store=stores.review_queue,
        response_log_store=stores.response_log,
    )


async def _erase(stores: _Stores, student_id: str):
    return await erase_student_data(
        student_id,
        memory_store=stores.memory,
        review_store=stores.review,
        ia_project_store=stores.ia_project,
        review_queue_store=stores.review_queue,
        response_log_store=stores.response_log,
    )


@pytest.mark.asyncio
async def test_export_includes_every_store_real_data():
    stores = _Stores()
    await _seed(stores, STUDENT)

    export = await _export(stores, STUDENT)

    assert len(export.mastery) == 1
    assert len(export.misconceptions) == 1
    assert len(export.review_states) == 1
    assert len(export.ia_projects) == 1
    assert len(export.disclosure_entries) == 1
    assert len(export.review_queue_entries) == 1
    assert len(export.item_responses) == 1
    assert export.mastery[0]["subtopic_id"] == "calc.chain_rule"


@pytest.mark.asyncio
async def test_export_is_scoped_to_the_requested_student():
    stores = _Stores()
    await _seed(stores, STUDENT)
    await _seed(stores, OTHER_STUDENT)

    export = await _export(stores, STUDENT)

    assert len(export.mastery) == 1  # not 2 - not the other student's record too


@pytest.mark.asyncio
async def test_erasure_removes_data_from_every_store_except_disclosure():
    stores = _Stores()
    await _seed(stores, STUDENT)

    report = await _erase(stores, STUDENT)

    assert report.mastery_and_misconceptions_erased == 2  # mastery + misconception
    assert report.review_states_erased == 1
    assert report.ia_projects_erased == 1
    assert report.review_queue_entries_erased == 1
    assert report.item_responses_erased == 1
    assert report.disclosure_log_retained is True
    assert report.total_records_erased == 6


@pytest.mark.asyncio
async def test_erasure_actually_deletes_not_just_reports():
    stores = _Stores()
    await _seed(stores, STUDENT)
    await _erase(stores, STUDENT)

    export_after = await _export(stores, STUDENT)
    assert export_after.mastery == []
    assert export_after.misconceptions == []
    assert export_after.review_states == []
    assert export_after.ia_projects == []
    assert export_after.review_queue_entries == []
    assert export_after.item_responses == []


@pytest.mark.asyncio
async def test_disclosure_log_survives_erasure():
    stores = _Stores()
    await _seed(stores, STUDENT)
    await _erase(stores, STUDENT)

    export_after = await _export(stores, STUDENT)
    assert len(export_after.disclosure_entries) == 1


@pytest.mark.asyncio
async def test_erasure_does_not_touch_other_students_data():
    stores = _Stores()
    await _seed(stores, STUDENT)
    await _seed(stores, OTHER_STUDENT)

    await _erase(stores, STUDENT)

    other_export = await _export(stores, OTHER_STUDENT)
    assert len(other_export.mastery) == 1
    assert len(other_export.review_states) == 1


@pytest.mark.asyncio
async def test_erasure_of_a_student_with_no_data_reports_zero():
    stores = _Stores()
    report = await _erase(stores, "nobody-ever-seen")
    assert report.total_records_erased == 0
