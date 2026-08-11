"""
Decay / forgetting functions (spec §4.4, §4.6). Both applied at read-time
(not a background job), exponential decay ported exactly from the spec's
formulas.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

# Spec §4.4 defaults.
MASTERY_FLOOR = 0.3
MASTERY_DECAY_TAU_DAYS = 45

# Spec §4.6 default.
MISCONCEPTION_DECAY_TAU_DAYS = 30


def _days_elapsed(since: datetime, now: Optional[datetime] = None) -> float:
    now = now or datetime.now(timezone.utc)
    return (now - since).total_seconds() / 86400.0


def effective_mastery(
    p_mastery_bkt: float,
    last_practiced_at: Optional[datetime],
    now: Optional[datetime] = None,
    floor: float = MASTERY_FLOOR,
    tau: float = MASTERY_DECAY_TAU_DAYS,
) -> float:
    """
    Spec §4.4:
        decay_factor = exp(-days_elapsed / tau)
        effective_p_mastery = floor + (p_mastery_bkt - floor) * decay_factor

    A subtopic never practiced has nothing to decay from — returns the
    raw value (its p_init prior) unchanged.
    """
    if last_practiced_at is None:
        return p_mastery_bkt
    days_elapsed = _days_elapsed(last_practiced_at, now)
    decay_factor = math.exp(-days_elapsed / tau)
    return floor + (p_mastery_bkt - floor) * decay_factor


def decayed_misconception_strength(
    decayed_strength: float,
    last_observed_at: datetime,
    now: Optional[datetime] = None,
    tau: float = MISCONCEPTION_DECAY_TAU_DAYS,
) -> float:
    """Spec §4.6: decayed_strength_new = decayed_strength_old * exp(-days_since_last_observed / tau)."""
    days_elapsed = _days_elapsed(last_observed_at, now)
    return decayed_strength * math.exp(-days_elapsed / tau)
