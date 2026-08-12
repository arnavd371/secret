import pytest

from app.ia_supervisor.disclosure import render_disclosure_statement
from app.ia_supervisor.disclosure_store import InMemoryDisclosureStore
from app.ia_supervisor.models import DisclosureAssistanceType, DisclosureEntry, IAStage
from app.ia_supervisor.project_store import InMemoryIAProjectStateStore
from app.ia_supervisor.models import IAProjectState


@pytest.mark.asyncio
async def test_project_store_roundtrip():
    store = InMemoryIAProjectStateStore()
    assert await store.get("stu-1", "proj-1") is None

    state = IAProjectState(student_id="stu-1", project_id="proj-1", stage=IAStage.METHODOLOGY)
    await store.save(state)

    fetched = await store.get("stu-1", "proj-1")
    assert fetched.stage == IAStage.METHODOLOGY


@pytest.mark.asyncio
async def test_project_store_is_scoped_per_project():
    store = InMemoryIAProjectStateStore()
    await store.save(IAProjectState(student_id="stu-1", project_id="ia-bio", stage=IAStage.DRAFTING))
    await store.save(IAProjectState(student_id="stu-1", project_id="ee-history", stage=IAStage.TOPIC_SELECTION))

    assert (await store.get("stu-1", "ia-bio")).stage == IAStage.DRAFTING
    assert (await store.get("stu-1", "ee-history")).stage == IAStage.TOPIC_SELECTION


@pytest.mark.asyncio
async def test_disclosure_store_is_append_only_and_chronological():
    store = InMemoryDisclosureStore()
    first = DisclosureEntry(
        student_id="stu-1", project_id="proj-1", stage=IAStage.TOPIC_SELECTION,
        assistance_type=DisclosureAssistanceType.COACHING, summary="first",
    )
    second = DisclosureEntry(
        student_id="stu-1", project_id="proj-1", stage=IAStage.RESEARCH_QUESTION,
        assistance_type=DisclosureAssistanceType.COACHING, summary="second",
    )
    await store.add(second)  # added out of order
    await store.add(first)

    entries = await store.get_all("stu-1", "proj-1")
    assert [e.summary for e in entries] == ["second", "first"] or [e.summary for e in entries] == ["first", "second"]
    # Chronological by timestamp, not insertion order - both were created
    # microseconds apart in this test, so just assert count and scoping
    # here; ordering-by-timestamp is exercised precisely below.
    assert len(entries) == 2


@pytest.mark.asyncio
async def test_disclosure_store_orders_by_timestamp_not_insertion_order():
    import datetime

    store = InMemoryDisclosureStore()
    later = DisclosureEntry(
        student_id="stu-1", project_id="proj-1", stage=IAStage.DRAFTING,
        assistance_type=DisclosureAssistanceType.COACHING, summary="later",
        timestamp=datetime.datetime(2026, 6, 2, tzinfo=datetime.timezone.utc),
    )
    earlier = DisclosureEntry(
        student_id="stu-1", project_id="proj-1", stage=IAStage.TOPIC_SELECTION,
        assistance_type=DisclosureAssistanceType.COACHING, summary="earlier",
        timestamp=datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc),
    )
    await store.add(later)
    await store.add(earlier)

    entries = await store.get_all("stu-1", "proj-1")
    assert [e.summary for e in entries] == ["earlier", "later"]


@pytest.mark.asyncio
async def test_disclosure_store_is_scoped_per_student_and_project():
    store = InMemoryDisclosureStore()
    await store.add(DisclosureEntry(
        student_id="stu-1", project_id="proj-1", stage=IAStage.TOPIC_SELECTION,
        assistance_type=DisclosureAssistanceType.COACHING, summary="stu-1's entry",
    ))
    await store.add(DisclosureEntry(
        student_id="stu-2", project_id="proj-1", stage=IAStage.TOPIC_SELECTION,
        assistance_type=DisclosureAssistanceType.COACHING, summary="stu-2's entry",
    ))

    entries = await store.get_all("stu-1", "proj-1")
    assert len(entries) == 1
    assert entries[0].summary == "stu-1's entry"


def test_render_disclosure_statement_with_no_entries():
    text = render_disclosure_statement("stu-1", "proj-1", [])
    assert "proj-1" in text
    assert "No AI-assisted interactions" in text


def test_render_disclosure_statement_includes_every_entry():
    entries = [
        DisclosureEntry(
            student_id="stu-1", project_id="proj-1", stage=IAStage.RESEARCH_QUESTION,
            assistance_type=DisclosureAssistanceType.COACHING, summary="Discussed narrowing the RQ.",
        ),
        DisclosureEntry(
            student_id="stu-1", project_id="proj-1", stage=IAStage.DRAFTING,
            assistance_type=DisclosureAssistanceType.GHOSTWRITING_REQUEST_REFUSED,
            summary="Student asked for the introduction to be written; declined.",
        ),
    ]
    text = render_disclosure_statement("stu-1", "proj-1", entries)
    assert "research_question" in text
    assert "drafting" in text
    assert "Discussed narrowing the RQ." in text
    assert "declined" in text.lower()
    assert "coaching feedback provided" in text.lower()
