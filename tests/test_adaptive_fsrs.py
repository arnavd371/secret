"""
Real-math assertions against the FSRS-lite scheduling formulas — every
value here is independently hand-checkable, and the monotonicity
properties (more successes grow stability, a lapse shrinks it) are
exactly what a spaced-repetition scheduler must guarantee to be correct
at all.
"""

from app.adaptive.fsrs import (
    DESIRED_RETENTION,
    FORGETTING_CURVE_FACTOR,
    INITIAL_DIFFICULTY,
    MIN_DIFFICULTY,
    MAX_DIFFICULTY,
    initial_state,
    next_interval_days,
    retrievability,
    update_state,
)
from app.adaptive.models import ReviewGrade


def test_retrievability_is_1_at_zero_elapsed_time():
    assert retrievability(0, stability=5.0) == 1.0


def test_retrievability_equals_desired_retention_at_t_equals_stability():
    # By construction: R(S, S) = (1 + S/(9*S))**-1 = (1 + 1/9)**-1 = 0.9
    stability = 7.0
    assert abs(retrievability(stability, stability) - DESIRED_RETENTION) < 1e-9


def test_retrievability_decreases_as_time_elapses():
    stability = 5.0
    r1 = retrievability(1, stability)
    r2 = retrievability(10, stability)
    r3 = retrievability(100, stability)
    assert r1 > r2 > r3


def test_retrievability_is_higher_for_higher_stability_at_the_same_elapsed_time():
    assert retrievability(10, stability=20.0) > retrievability(10, stability=5.0)


def test_initial_state_good_grade_gives_higher_stability_than_again():
    s_good, d_good = initial_state(ReviewGrade.GOOD)
    s_again, d_again = initial_state(ReviewGrade.AGAIN)
    assert s_good > s_again
    assert d_good == INITIAL_DIFFICULTY == d_again


def test_successful_review_grows_stability():
    stability, difficulty = initial_state(ReviewGrade.GOOD)
    new_stability, _ = update_state(stability, difficulty, days_elapsed=stability, grade=ReviewGrade.GOOD)
    assert new_stability > stability


def test_failed_review_shrinks_stability():
    stability, difficulty = initial_state(ReviewGrade.GOOD)
    new_stability, _ = update_state(stability, difficulty, days_elapsed=stability, grade=ReviewGrade.AGAIN)
    assert new_stability < stability


def test_failed_review_raises_difficulty():
    stability, difficulty = initial_state(ReviewGrade.GOOD)
    _, new_difficulty = update_state(stability, difficulty, days_elapsed=stability, grade=ReviewGrade.AGAIN)
    assert new_difficulty > difficulty


def test_difficulty_stays_within_bounds_after_many_failures():
    stability, difficulty = initial_state(ReviewGrade.AGAIN)
    for _ in range(50):
        stability, difficulty = update_state(stability, difficulty, days_elapsed=0.1, grade=ReviewGrade.AGAIN)
    assert MIN_DIFFICULTY <= difficulty <= MAX_DIFFICULTY


def test_repeated_successful_reviews_at_recommended_intervals_grow_stability_monotonically():
    stability, difficulty = initial_state(ReviewGrade.GOOD)
    stabilities = [stability]
    for _ in range(6):
        interval = next_interval_days(stability)
        stability, difficulty = update_state(stability, difficulty, interval, ReviewGrade.GOOD)
        stabilities.append(stability)
    assert stabilities == sorted(stabilities)
    assert stabilities[-1] > stabilities[0]


def test_lower_retrievability_success_grows_stability_more_than_a_near_certain_recall():
    # Successfully recalling an item you were more likely to have
    # forgotten (waited longer, so lower retrievability) is real evidence
    # of stronger memory formation, and should grow stability more than
    # recalling something you reviewed a moment ago (retrievability ~1).
    stability, difficulty = initial_state(ReviewGrade.GOOD)
    grown_from_near_certain, _ = update_state(stability, difficulty, days_elapsed=0.01, grade=ReviewGrade.GOOD)
    grown_from_uncertain, _ = update_state(stability, difficulty, days_elapsed=stability * 3, grade=ReviewGrade.GOOD)
    assert grown_from_uncertain > grown_from_near_certain


def test_next_interval_days_scales_linearly_with_stability():
    assert next_interval_days(10.0) == 2 * next_interval_days(5.0)


def test_next_interval_days_matches_the_retrievability_inversion():
    stability = 4.0
    interval = next_interval_days(stability)
    assert abs(retrievability(interval, stability) - DESIRED_RETENTION) < 1e-9


def test_higher_desired_retention_target_gives_a_shorter_interval():
    stability = 5.0
    assert next_interval_days(stability, desired_retention=0.95) < next_interval_days(stability, desired_retention=0.8)


def test_forgetting_curve_factor_is_nine_by_construction():
    # Not a "real" test of behavior, just pins the documented constant so
    # a silent edit here can't quietly break the R(S,S)=0.9 property above.
    assert FORGETTING_CURVE_FACTOR == 9.0
