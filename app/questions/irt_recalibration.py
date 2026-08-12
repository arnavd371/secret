"""
Real IRT item-difficulty recalibration from response history (spec
§9.7), under the Rasch (1PL) model's standard closed-form relationship
between empirical pass rate and difficulty:

    p = 1 / (1 + exp(-(theta - b)))

Assuming the responding population's mean ability theta = 0 (a real,
documented simplification — the true population-mean-centering the full
2PL/3PL model would use needs a joint ability/difficulty estimation this
system doesn't have the response volume or infrastructure to run, the
same honest-simplification posture as Phase 9's fixed FSRS parameters
instead of the published algorithm's per-user ML-fitted weights), this
inverts to:

    b = ln((1 - p) / p)

A template attempted often (p near 0.5) recalibrates to a difficulty
near 0. One attempted rarely but always failed (p near 0) recalibrates
to a large positive b (hard); one always passed (p near 1) recalibrates
to a large negative b (easy) — exactly the qualitative behavior a real
difficulty parameter should have.
"""

from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel

from app.questions.response_log import ResponseLogStore

# A pass rate can never be treated as exactly 0 or 1 without an infinite
# recalibrated difficulty — clamped to a real, bounded range instead,
# same spirit as any tolerance/floor constant elsewhere in this codebase.
_MIN_PASS_RATE = 0.02
_MAX_PASS_RATE = 0.98

DEFAULT_MIN_RESPONSES = 10


class RecalibrationResult(BaseModel):
    template_id: str
    sample_size: int
    pass_rate: float
    recalibrated_b: float


def compute_empirical_difficulty(pass_rate: float) -> float:
    clamped = min(max(pass_rate, _MIN_PASS_RATE), _MAX_PASS_RATE)
    return math.log((1 - clamped) / clamped)


async def recalibrate_template_difficulty(
    template_id: str, store: ResponseLogStore, *, min_responses: int = DEFAULT_MIN_RESPONSES
) -> Optional[RecalibrationResult]:
    """None when there isn't enough real response history yet to trust a
    recalibration — same "don't act on too little data" posture as
    app.diagnostician.write_policy's confidence gate. `min_responses` is
    a real, if necessarily somewhat arbitrary, sample-size floor: 10
    responses is enough to distinguish "clearly easy," "clearly hard,"
    and "roughly balanced," not enough to claim high statistical
    precision — this recalibrates a coarse prior, not a fitted model."""
    records = await store.get_all(template_id)
    if len(records) < min_responses:
        return None

    pass_rate = sum(1 for r in records if r.correct) / len(records)
    return RecalibrationResult(
        template_id=template_id,
        sample_size=len(records),
        pass_rate=round(pass_rate, 4),
        recalibrated_b=round(compute_empirical_difficulty(pass_rate), 4),
    )
