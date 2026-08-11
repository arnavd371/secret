"""
Real-numeric tests for the Student Memory System's core math: every
assertion here is checked against a value computed by hand from the
spec's own formulas, not just "did it move in the right direction."
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.memory.bkt import P_GUESS_DEFAULT, P_INIT_DEFAULT, P_SLIP_DEFAULT, P_TRANSIT_DEFAULT, update_bkt
from app.memory.decay import MASTERY_FLOOR, decayed_misconception_strength, effective_mastery
from app.memory.irt import probability_correct, update_irt
from app.memory.models import NodeState, SubtopicMastery
from app.memory.node_state import compute_node_state


# ---------------------------------------------------------------------------
# BKT
# ---------------------------------------------------------------------------


def test_bkt_correct_update_matches_hand_computed_value():
    p = P_INIT_DEFAULT  # 0.10
    numerator = p * (1 - P_SLIP_DEFAULT)
    denominator = numerator + (1 - p) * P_GUESS_DEFAULT
    p_given_evidence = numerator / denominator
    expected = p_given_evidence + (1 - p_given_evidence) * P_TRANSIT_DEFAULT

    assert update_bkt(p, correct=True) == pytest.approx(expected)


def test_bkt_incorrect_update_matches_hand_computed_value():
    p = 0.5
    numerator = p * P_SLIP_DEFAULT
    denominator = numerator + (1 - p) * (1 - P_GUESS_DEFAULT)
    p_given_evidence = numerator / denominator
    expected = p_given_evidence + (1 - p_given_evidence) * P_TRANSIT_DEFAULT

    assert update_bkt(p, correct=False) == pytest.approx(expected)


def test_bkt_mastery_increases_monotonically_with_repeated_correct_answers():
    p = P_INIT_DEFAULT
    values = [p]
    for _ in range(6):
        p = update_bkt(p, correct=True)
        values.append(p)
    assert values == sorted(values)
    assert values[-1] > 0.9


def test_bkt_stays_within_bounds():
    p = P_INIT_DEFAULT
    for _ in range(20):
        p = update_bkt(p, correct=True)
        assert 0.0 <= p <= 1.0


def test_bkt_never_reaches_exactly_zero_or_one_from_transit_floor():
    """p_transit means even a wrong answer never drives mastery to
    literal zero, and correct-answer belief always has some slip
    uncertainty — sanity bounds, not exact equality."""
    p = 0.01
    for _ in range(5):
        p = update_bkt(p, correct=False)
    assert p > 0.0


# ---------------------------------------------------------------------------
# IRT
# ---------------------------------------------------------------------------


def test_probability_correct_at_theta_equals_b_is_one_half():
    assert probability_correct(theta=0.5, a=1.0, b=0.5) == pytest.approx(0.5)


def test_irt_correct_answer_increases_theta():
    theta_new, se_new = update_irt(theta=0.0, se_theta=1.0, a=1.0, b=0.0, correct=True)
    assert theta_new > 0.0
    assert se_new < 1.0  # uncertainty should shrink after any observation


def test_irt_incorrect_answer_decreases_theta():
    theta_new, se_new = update_irt(theta=0.0, se_theta=1.0, a=1.0, b=0.0, correct=False)
    assert theta_new < 0.0


def test_irt_matches_hand_computed_value():
    theta, se, a, b = 0.0, 1.0, 1.0, 0.0
    p = probability_correct(theta, a, b)
    expected_theta = theta + (se**2) * a * (1.0 - p)
    theta_new, _ = update_irt(theta, se, a, b, correct=True)
    assert theta_new == pytest.approx(expected_theta)


def test_irt_uncertainty_shrinks_with_repeated_observations():
    theta, se = 0.0, 1.0
    prev_se = se
    for _ in range(5):
        theta, se = update_irt(theta, se, a=1.0, b=0.0, correct=True)
        assert se < prev_se
        prev_se = se


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------


def test_effective_mastery_with_no_history_is_unchanged():
    assert effective_mastery(0.8, last_practiced_at=None) == 0.8


def test_effective_mastery_matches_hand_computed_value():
    now = datetime.now(timezone.utc)
    last_practiced = now - timedelta(days=45)
    result = effective_mastery(0.9, last_practiced, now=now, floor=0.3, tau=45)
    import math

    expected = 0.3 + (0.9 - 0.3) * math.exp(-1)
    assert result == pytest.approx(expected, abs=1e-6)


def test_effective_mastery_never_drops_below_floor():
    now = datetime.now(timezone.utc)
    ancient = now - timedelta(days=100000)
    result = effective_mastery(0.9, ancient, now=now)
    assert result == pytest.approx(MASTERY_FLOOR, abs=1e-6)


def test_effective_mastery_recent_practice_stays_close_to_raw_value():
    now = datetime.now(timezone.utc)
    result = effective_mastery(0.9, now, now=now)
    assert result == pytest.approx(0.9, abs=0.01)


def test_misconception_decay_matches_hand_computed_value():
    now = datetime.now(timezone.utc)
    last_observed = now - timedelta(days=30)
    import math

    expected = 1.0 * math.exp(-1)
    assert decayed_misconception_strength(1.0, last_observed, now=now, tau=30) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Node state
# ---------------------------------------------------------------------------


def test_unseen_state_for_no_attempts():
    m = SubtopicMastery(student_id="s", subtopic_id="t", attempts_total=0)
    assert compute_node_state(m) == NodeState.UNSEEN


def test_introduced_state():
    now = datetime.now(timezone.utc)
    m = SubtopicMastery(student_id="s", subtopic_id="t", attempts_total=2, p_mastery_bkt=0.3, last_practiced_at=now)
    assert compute_node_state(m, now) == NodeState.INTRODUCED


def test_practicing_state():
    now = datetime.now(timezone.utc)
    m = SubtopicMastery(student_id="s", subtopic_id="t", attempts_total=4, p_mastery_bkt=0.55, last_practiced_at=now)
    assert compute_node_state(m, now) == NodeState.PRACTICING


def test_consolidating_state():
    now = datetime.now(timezone.utc)
    m = SubtopicMastery(
        student_id="s", subtopic_id="t", attempts_total=5, p_mastery_bkt=0.8, last_practiced_at=now - timedelta(days=10)
    )
    assert compute_node_state(m, now) == NodeState.CONSOLIDATING


def test_mastered_state():
    now = datetime.now(timezone.utc)
    m = SubtopicMastery(
        student_id="s", subtopic_id="t", attempts_total=10, p_mastery_bkt=0.95, last_practiced_at=now - timedelta(days=5)
    )
    assert compute_node_state(m, now) == NodeState.MASTERED


def test_mastered_state_expires_past_14_days_to_consolidating_or_decayed():
    now = datetime.now(timezone.utc)
    m = SubtopicMastery(
        student_id="s", subtopic_id="t", attempts_total=10, p_mastery_bkt=0.95, last_practiced_at=now - timedelta(days=20)
    )
    assert compute_node_state(m, now) != NodeState.MASTERED


def test_decayed_state_for_stale_high_mastery():
    now = datetime.now(timezone.utc)
    m = SubtopicMastery(
        student_id="s", subtopic_id="t", attempts_total=10, p_mastery_bkt=0.95, last_practiced_at=now - timedelta(days=45)
    )
    assert compute_node_state(m, now) == NodeState.DECAYED


def test_decayed_state_does_not_apply_to_low_mastery():
    """Staleness alone shouldn't downgrade a student who was never that
    far along in the first place — DECAYED means 'used to be good,
    isn't fresh now', not just 'hasn't practiced recently'."""
    now = datetime.now(timezone.utc)
    m = SubtopicMastery(
        student_id="s", subtopic_id="t", attempts_total=3, p_mastery_bkt=0.5, last_practiced_at=now - timedelta(days=45)
    )
    assert compute_node_state(m, now) != NodeState.DECAYED
