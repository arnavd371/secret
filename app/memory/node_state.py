"""
Node state thresholds (spec §4.3's table), derived from p_mastery_bkt,
attempt count, and recency. Evaluated in explicit precedence order since
the spec's table bands aren't fully disjoint (e.g. "mastered" and
"decayed" both describe p_mastery_bkt >= 0.7-0.9 territory, distinguished
only by recency) — staleness is checked first as a catch-all so a
student who was once mastered but hasn't practiced in a month is
correctly reported as decayed, not still mastered.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.memory.models import NodeState, SubtopicMastery

MASTERED_MIN_MASTERY = 0.9
MASTERED_MAX_RECENCY_DAYS = 14
CONSOLIDATING_MIN_MASTERY = 0.7
CONSOLIDATING_MAX_RECENCY_DAYS = 21
PRACTICING_MIN_MASTERY = 0.4
PRACTICING_MIN_ATTEMPTS = 3
DECAYED_STALENESS_DAYS = 30


def compute_node_state(mastery: SubtopicMastery, now: Optional[datetime] = None) -> NodeState:
    if mastery.attempts_total == 0:
        return NodeState.UNSEEN

    now = now or datetime.now(timezone.utc)
    days_since_practice = (
        (now - mastery.last_practiced_at).total_seconds() / 86400.0 if mastery.last_practiced_at else None
    )
    p = mastery.p_mastery_bkt

    if days_since_practice is not None and days_since_practice > DECAYED_STALENESS_DAYS and p >= CONSOLIDATING_MIN_MASTERY:
        return NodeState.DECAYED

    if p >= MASTERED_MIN_MASTERY and days_since_practice is not None and days_since_practice <= MASTERED_MAX_RECENCY_DAYS:
        return NodeState.MASTERED

    if (
        CONSOLIDATING_MIN_MASTERY <= p < MASTERED_MIN_MASTERY
        and days_since_practice is not None
        and days_since_practice <= CONSOLIDATING_MAX_RECENCY_DAYS
    ):
        return NodeState.CONSOLIDATING

    if mastery.attempts_total >= PRACTICING_MIN_ATTEMPTS and PRACTICING_MIN_MASTERY <= p < CONSOLIDATING_MIN_MASTERY:
        return NodeState.PRACTICING

    if mastery.attempts_total <= 2 and p < PRACTICING_MIN_MASTERY:
        return NodeState.INTRODUCED

    # Combinations the table doesn't explicitly cover (e.g. >=3 attempts
    # but still p < 0.4) fall back to the closest matching band by
    # attempt count rather than defaulting to UNSEEN, which would be
    # actively wrong once attempts_total > 0.
    return NodeState.PRACTICING if mastery.attempts_total >= PRACTICING_MIN_ATTEMPTS else NodeState.INTRODUCED
