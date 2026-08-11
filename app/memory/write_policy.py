"""
Memory write policy (spec §4.7): mastery updates should only be trusted
on sufficiently confident evidence — "only on attempts with
assessment_confidence >= 0.75 from parsing (avoid corrupting mastery with
misread OCR)."

This system has no OCR confidence signal yet (multimodal ingestion is a
Phase 7 non-goal). The closest real evidence-quality signal actually
available is the Grader's own ConfidenceTier (Phase 4, spec §10.11),
used here as the write gate instead of a fabricated substitute — a LOW-
confidence grading (couldn't find a final answer, or flagged unsupported/
inconsistent) shouldn't be trusted to update a persistent mastery model
any more than a badly-OCR'd submission should.
"""

from __future__ import annotations

from app.examiner.models import ConfidenceTier

_WRITE_ELIGIBLE_TIERS = (ConfidenceTier.HIGH, ConfidenceTier.MEDIUM)


def should_write_mastery_update(confidence: ConfidenceTier) -> bool:
    return confidence in _WRITE_ELIGIBLE_TIERS
