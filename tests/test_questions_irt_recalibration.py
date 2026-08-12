"""
Real-math assertions for IRT recalibration: the Rasch-model logit
relationship between empirical pass rate and difficulty, hand-checkable
at a few reference points.
"""

import math

import pytest

from app.questions.irt_recalibration import (
    DEFAULT_MIN_RESPONSES,
    compute_empirical_difficulty,
    recalibrate_template_difficulty,
)
from app.questions.response_log import InMemoryResponseLogStore, ItemResponseRecord


def test_fifty_percent_pass_rate_gives_zero_difficulty():
    assert compute_empirical_difficulty(0.5) == pytest.approx(0.0, abs=1e-9)


def test_higher_pass_rate_gives_lower_difficulty():
    assert compute_empirical_difficulty(0.9) < compute_empirical_difficulty(0.5) < compute_empirical_difficulty(0.1)


def test_known_reference_point():
    # b = ln((1-p)/p); p=0.25 -> ln(3) ≈ 1.0986
    assert compute_empirical_difficulty(0.25) == pytest.approx(math.log(3), abs=1e-6)


def test_extreme_pass_rates_are_clamped_not_infinite():
    assert math.isfinite(compute_empirical_difficulty(1.0))
    assert math.isfinite(compute_empirical_difficulty(0.0))


async def _seed(store, template_id, correctness: list[bool]):
    for i, correct in enumerate(correctness):
        await store.add(ItemResponseRecord(template_id=template_id, student_id=f"s{i}", correct=correct))


@pytest.mark.asyncio
async def test_below_minimum_sample_size_returns_none():
    store = InMemoryResponseLogStore()
    await _seed(store, "T1", [True] * (DEFAULT_MIN_RESPONSES - 1))
    result = await recalibrate_template_difficulty("T1", store)
    assert result is None


@pytest.mark.asyncio
async def test_at_minimum_sample_size_recalibrates():
    store = InMemoryResponseLogStore()
    await _seed(store, "T1", [True] * DEFAULT_MIN_RESPONSES)
    result = await recalibrate_template_difficulty("T1", store)
    assert result is not None
    assert result.sample_size == DEFAULT_MIN_RESPONSES
    assert result.pass_rate == 1.0
    assert result.recalibrated_b < 0  # easy


@pytest.mark.asyncio
async def test_mostly_wrong_responses_recalibrate_harder():
    store = InMemoryResponseLogStore()
    await _seed(store, "T1", [True, True, True] + [False] * 12)
    result = await recalibrate_template_difficulty("T1", store)
    assert result.pass_rate == pytest.approx(0.2)
    assert result.recalibrated_b > 0  # hard


@pytest.mark.asyncio
async def test_recalibration_is_scoped_per_template():
    store = InMemoryResponseLogStore()
    await _seed(store, "T-easy", [True] * DEFAULT_MIN_RESPONSES)
    await _seed(store, "T-hard", [False] * DEFAULT_MIN_RESPONSES)

    easy = await recalibrate_template_difficulty("T-easy", store)
    hard = await recalibrate_template_difficulty("T-hard", store)
    assert easy.recalibrated_b < hard.recalibrated_b


@pytest.mark.asyncio
async def test_custom_min_responses_threshold():
    store = InMemoryResponseLogStore()
    await _seed(store, "T1", [True] * 3)
    assert await recalibrate_template_difficulty("T1", store, min_responses=5) is None
    assert await recalibrate_template_difficulty("T1", store, min_responses=3) is not None
