import pytest

from app.questions.response_log import InMemoryResponseLogStore, ItemResponseRecord


@pytest.mark.asyncio
async def test_response_log_is_append_only_and_scoped_per_template():
    store = InMemoryResponseLogStore()
    await store.add(ItemResponseRecord(template_id="T1", student_id="s1", correct=True))
    await store.add(ItemResponseRecord(template_id="T1", student_id="s2", correct=False))
    await store.add(ItemResponseRecord(template_id="T2", student_id="s1", correct=True))

    t1_records = await store.get_all("T1")
    t2_records = await store.get_all("T2")

    assert len(t1_records) == 2
    assert len(t2_records) == 1


@pytest.mark.asyncio
async def test_empty_template_returns_no_records():
    store = InMemoryResponseLogStore()
    assert await store.get_all("nonexistent") == []
