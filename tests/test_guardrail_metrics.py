"""
Tests for the guardrail metrics recorder: real extraction from
TutorResponse-shaped ui_metadata dicts (synthetic, standing in for real
turns - matching this phase's own scope: infrastructure any deployment
could feed live traffic into, verified here without one), a real
append-only store, and real aggregation math.
"""

import pytest

from app.guardrail_metrics.aggregate import compute_guardrail_metrics
from app.guardrail_metrics.extraction import extract_guardrail_signals
from app.guardrail_metrics.models import GuardrailTurnSignals
from app.guardrail_metrics.store import InMemoryGuardrailMetricsStore


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def test_extraction_of_an_approved_non_templated_turn():
    ui_metadata = {
        "action_type": "explain",
        "templated": False,
        "critique_verdict": "pass",
        "critic_degraded": False,
        "grounding_score": 0.87,
    }
    signals = extract_guardrail_signals("turn-1", ui_metadata)
    assert signals.fell_back_to_template is False
    assert signals.fallback_reason is None
    assert signals.leak_check_triggered is False
    assert signals.critic_verdict == "pass"
    assert signals.critic_degraded is False
    assert signals.grounding_score == 0.87


def test_extraction_of_a_leak_check_fallback():
    ui_metadata = {"action_type": "hint", "level": 1, "templated": True, "fallback_reason": "leak_check"}
    signals = extract_guardrail_signals("turn-2", ui_metadata)
    assert signals.fell_back_to_template is True
    assert signals.leak_check_triggered is True
    assert signals.fallback_reason == "leak_check"
    assert signals.critic_verdict is None  # critic never ran - draft was discarded first


def test_extraction_of_a_critic_block_fallback_is_not_counted_as_a_leak_check():
    ui_metadata = {"action_type": "explain", "templated": True, "fallback_reason": "critic_block"}
    signals = extract_guardrail_signals("turn-3", ui_metadata)
    assert signals.fell_back_to_template is True
    assert signals.leak_check_triggered is False
    assert signals.fallback_reason == "critic_block"


def test_extraction_of_a_turn_with_no_grounding_check_at_all():
    """No retrieved_chunks means grounding was never checked - the
    signal must be None, not a misleading 0.0."""
    ui_metadata = {"action_type": "hint", "templated": False, "critique_verdict": "pass", "critic_degraded": False}
    signals = extract_guardrail_signals("turn-4", ui_metadata)
    assert signals.grounding_score is None


def test_extraction_of_a_critic_degraded_turn():
    ui_metadata = {
        "action_type": "explain",
        "templated": False,
        "critique_verdict": "pass",
        "critic_degraded": True,
        "grounding_score": None,
    }
    signals = extract_guardrail_signals("turn-5", ui_metadata)
    assert signals.critic_degraded is True


def test_extraction_ignores_a_stray_fallback_reason_when_not_actually_templated():
    """Defensive: fallback_reason should only ever be read when
    templated is True, even if a caller's dict is malformed."""
    ui_metadata = {"action_type": "explain", "templated": False, "fallback_reason": "leak_check"}
    signals = extract_guardrail_signals("turn-6", ui_metadata)
    assert signals.fell_back_to_template is False
    assert signals.fallback_reason is None
    assert signals.leak_check_triggered is False


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _signal(**overrides) -> GuardrailTurnSignals:
    defaults = dict(turn_id="t", leak_check_triggered=False, fell_back_to_template=False)
    defaults.update(overrides)
    return GuardrailTurnSignals(**defaults)


def test_aggregate_on_empty_batch_reports_zero_not_an_error():
    report = compute_guardrail_metrics([])
    assert report.total_turns == 0
    assert report.leak_check_trigger_rate == 0.0
    assert report.fallback_rate == 0.0
    assert report.average_grounding_score is None
    assert report.critic_verdict_distribution == {}


def test_leak_check_trigger_rate_is_computed_over_all_turns():
    records = [
        _signal(leak_check_triggered=True, fell_back_to_template=True, fallback_reason="leak_check"),
        _signal(leak_check_triggered=False, fell_back_to_template=False, critic_verdict="pass"),
        _signal(leak_check_triggered=False, fell_back_to_template=False, critic_verdict="pass"),
        _signal(leak_check_triggered=False, fell_back_to_template=False, critic_verdict="pass"),
    ]
    report = compute_guardrail_metrics(records)
    assert report.total_turns == 4
    assert report.leak_check_trigger_rate == 0.25


def test_fallback_rate_counts_every_fallback_cause():
    records = [
        _signal(fell_back_to_template=True, fallback_reason="leak_check", leak_check_triggered=True),
        _signal(fell_back_to_template=True, fallback_reason="critic_block"),
        _signal(fell_back_to_template=True, fallback_reason="grounding_failed"),
        _signal(fell_back_to_template=False, critic_verdict="pass"),
    ]
    report = compute_guardrail_metrics(records)
    assert report.fallback_rate == 0.75
    assert report.leak_check_trigger_rate == 0.25  # only 1 of the 3 fallbacks was a leak-check


def test_critic_verdict_distribution_is_over_turns_the_critic_actually_ran_on():
    records = [
        _signal(critic_verdict="pass"),
        _signal(critic_verdict="pass"),
        _signal(critic_verdict="revise"),
        _signal(critic_verdict="block", fell_back_to_template=True, fallback_reason="critic_block"),
        _signal(critic_verdict=None, fell_back_to_template=True, fallback_reason="leak_check", leak_check_triggered=True),
    ]
    report = compute_guardrail_metrics(records)
    assert report.total_turns == 5
    assert report.critic_ran_count == 4  # the leak-check turn never reached the critic
    assert report.critic_verdict_distribution["pass"] == pytest.approx(0.5)
    assert report.critic_verdict_distribution["revise"] == pytest.approx(0.25)
    assert report.critic_verdict_distribution["block"] == pytest.approx(0.25)


def test_average_grounding_score_only_counts_turns_that_had_a_score():
    records = [
        _signal(grounding_score=0.9),
        _signal(grounding_score=0.7),
        _signal(grounding_score=None),  # no retrieved_chunks this turn - not a 0.0
    ]
    report = compute_guardrail_metrics(records)
    assert report.grounding_checked_count == 2
    assert report.average_grounding_score == pytest.approx(0.8)


def test_critic_degraded_rate_is_computed_over_all_turns():
    records = [_signal(critic_degraded=True), _signal(critic_degraded=False), _signal(critic_degraded=False), _signal(critic_degraded=False)]
    report = compute_guardrail_metrics(records)
    assert report.critic_degraded_rate == 0.25


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_is_append_only_and_returns_everything_added():
    store = InMemoryGuardrailMetricsStore()
    await store.add(_signal(turn_id="a"))
    await store.add(_signal(turn_id="b"))
    records = await store.get_all()
    assert [r.turn_id for r in records] == ["a", "b"]


@pytest.mark.asyncio
async def test_store_and_aggregate_compose_end_to_end():
    store = InMemoryGuardrailMetricsStore()
    await store.add(_signal(turn_id="a", leak_check_triggered=True, fell_back_to_template=True, fallback_reason="leak_check"))
    await store.add(_signal(turn_id="b", critic_verdict="pass"))
    report = compute_guardrail_metrics(await store.get_all())
    assert report.total_turns == 2
    assert report.leak_check_trigger_rate == 0.5
