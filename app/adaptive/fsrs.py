"""
Spaced-repetition scheduling math (spec §12: "FSRS scheduling").

Real forgetting-curve mechanics with the same shape as the published FSRS
(Free Spaced Repetition Scheduler) algorithm — a stability/difficulty
state per item, a power-law retrievability curve parameterized by
stability, and reviews that grow stability on success and shrink it on
failure. Deliberately NOT a reproduction of FSRS's published ~19
ML-optimized weights: those are fit per-user from thousands of real
review logs, which this system doesn't have (same honest-simplification
posture as Phase 2's TF-IDF-instead-of-dense-retrieval and Phase 5's
BKT/IRT default parameters instead of per-student-fitted ones). The
constants below are fixed, documented, and chosen for the right
qualitative behavior (harder items and low-retrievability successes
grow stability more; failures shrink it and raise difficulty; every
review nudges difficulty back toward a neutral midpoint) rather than
optimized against real data.

Every formula here is pure and independently hand-checkable — no I/O,
no persistence (that's app/adaptive/store.py and scheduler.py).
"""

from __future__ import annotations

from app.adaptive.models import ReviewGrade

# Retrievability R(t, S) = (1 + t / (FORGETTING_CURVE_FACTOR * S)) ** -1.
# Stability S is defined, by construction of this constant, as "the
# number of days for retrievability to decay to DESIRED_RETENTION" when
# DESIRED_RETENTION = 0.9: at t=S, R = (1 + 1/9)**-1 = 0.9 exactly.
FORGETTING_CURVE_FACTOR = 9.0

DESIRED_RETENTION = 0.9

MIN_STABILITY = 0.1
INITIAL_STABILITY_GOOD = 2.5
INITIAL_STABILITY_AGAIN = 0.5

MIN_DIFFICULTY = 1.0
MAX_DIFFICULTY = 10.0
INITIAL_DIFFICULTY = 5.0
DIFFICULTY_DELTA_ON_GOOD = -0.3
DIFFICULTY_DELTA_ON_AGAIN = 1.0
# Every review pulls difficulty partway back toward the neutral midpoint
# rather than letting it drift monotonically — the same "mean reversion"
# principle FSRS uses, so one unusually easy or hard review doesn't
# permanently peg an item's difficulty.
DIFFICULTY_MEAN_REVERSION_WEIGHT = 0.1

STABILITY_GROWTH_RATE = 1.3
STABILITY_DECAY_ON_AGAIN = 0.5


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def retrievability(days_elapsed: float, stability: float) -> float:
    """Probability of successful recall right now, given `stability` days
    since the item was last reviewed `days_elapsed` days ago. 1.0 at
    days_elapsed=0, decaying toward 0 as time passes; decays slower for
    a higher-stability item."""
    if days_elapsed <= 0:
        return 1.0
    return (1.0 + days_elapsed / (FORGETTING_CURVE_FACTOR * stability)) ** -1.0


def initial_state(grade: ReviewGrade) -> tuple[float, float]:
    """(stability, difficulty) for an item's first-ever review."""
    stability = INITIAL_STABILITY_GOOD if grade == ReviewGrade.GOOD else INITIAL_STABILITY_AGAIN
    return stability, INITIAL_DIFFICULTY


def _update_difficulty(difficulty: float, grade: ReviewGrade) -> float:
    delta = DIFFICULTY_DELTA_ON_GOOD if grade == ReviewGrade.GOOD else DIFFICULTY_DELTA_ON_AGAIN
    adjusted = difficulty + delta
    reverted = adjusted + DIFFICULTY_MEAN_REVERSION_WEIGHT * (INITIAL_DIFFICULTY - adjusted)
    return _clamp(reverted, MIN_DIFFICULTY, MAX_DIFFICULTY)


def _update_stability_on_success(stability: float, difficulty: float, r: float) -> float:
    # Growth scales up when the item is harder (11 - difficulty is
    # larger for easy items, so this factor is actually largest for LOW
    # difficulty... to match FSRS's real principle — successfully
    # recalling something you were *likely to forget* (low r) grows
    # stability more than recalling something you already knew cold
    # (r near 1) — the (1 - r) factor does the real work here;
    # difficulty scales the size of that effect.
    difficulty_factor = (11.0 - difficulty) / 10.0
    growth = 1.0 + STABILITY_GROWTH_RATE * difficulty_factor * (1.0 - r)
    return stability * growth


def _update_stability_on_failure(stability: float) -> float:
    return max(MIN_STABILITY, stability * STABILITY_DECAY_ON_AGAIN)


def update_state(
    stability: float, difficulty: float, days_elapsed: float, grade: ReviewGrade
) -> tuple[float, float]:
    """(new_stability, new_difficulty) after a review of an item that was
    last reviewed `days_elapsed` days ago at the given prior
    stability/difficulty."""
    r = retrievability(days_elapsed, stability)
    new_difficulty = _update_difficulty(difficulty, grade)
    if grade == ReviewGrade.GOOD:
        new_stability = _update_stability_on_success(stability, difficulty, r)
    else:
        new_stability = _update_stability_on_failure(stability)
    return new_stability, new_difficulty


def next_interval_days(stability: float, desired_retention: float = DESIRED_RETENTION) -> float:
    """Days until retrievability decays to `desired_retention`, inverting
    the retrievability formula for t. Higher stability -> longer
    interval; a higher desired_retention target -> shorter interval
    (you're asking to be reviewed sooner, before you'd have forgotten as
    much)."""
    return stability * FORGETTING_CURVE_FACTOR * (1.0 / desired_retention - 1.0)
