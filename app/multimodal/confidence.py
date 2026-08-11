"""
Composite confidence score + three-tier gate (spec §3.2) for a completed
OCR transcription. Combines three independent, real signals rather than
trusting the vision model's own self-reported confidence (which most
providers don't even expose):

  - length: a suspiciously short transcription (a handful of characters)
    usually means the model gave up or the image was near-blank, not
    that the work was genuinely short
  - expression parseability: does at least one line actually parse as
    math (app/multimodal/expression_parse.py), independent of the
    model's fluency
  - LaTeX well-formedness: are math delimiters balanced in the raw OCR
    output, before normalization stripped them — an unbalanced count is
    a strong, cheap signal that the transcription is truncated or
    garbled

Deviation from a literal reading of spec §3.2: the spec's exact
per-signal weights aren't reproduced verbatim here (the source blueprint
doesn't pin down specific numeric weights either — it specifies the
concept of a composite score and three-tier gating). The weights below
are a documented, considered choice: parseability is weighted heaviest
because it's the strongest real signal ("is this actually math"), length
and well-formedness are weaker, complementary signals.

Three-tier gate, same HIGH/MEDIUM/LOW vocabulary Phase 4 already uses
for grading confidence (app.examiner.models.ConfidenceTier):
  - HIGH: safe to feed straight into Phase 4 grading with no student
    confirmation needed.
  - MEDIUM: usable, but the student should be asked to confirm the
    transcription before it's graded.
  - LOW: not usable as-is; ask the student to retype or retake the
    photo instead.
"""

from __future__ import annotations

from app.examiner.models import ConfidenceTier
from app.multimodal.models import ConfidenceBreakdown

WEIGHT_LENGTH = 0.2
WEIGHT_PARSEABLE = 0.5
WEIGHT_LATEX_WELL_FORMED = 0.3

# A transcription at or above this many characters gets full credit for
# the length signal; below it, credit scales down linearly. Calibrated
# to "a single short line of working" (e.g. "dy/dx = 5(2x+1)^4 * 2" is
# 22 characters), not a full multi-step solution.
LENGTH_SIGNAL_FULL_CREDIT_CHARS = 20

HIGH_CONFIDENCE_THRESHOLD = 0.75
MEDIUM_CONFIDENCE_THRESHOLD = 0.45


def _latex_well_formed(raw_text: str) -> bool:
    return (
        raw_text.count(r"\(") == raw_text.count(r"\)")
        and raw_text.count(r"\[") == raw_text.count(r"\]")
        and raw_text.count("{") == raw_text.count("}")
    )


def _length_signal(normalized_text: str) -> float:
    length = len(normalized_text.strip())
    if length == 0:
        return 0.0
    return min(1.0, length / LENGTH_SIGNAL_FULL_CREDIT_CHARS)


def score_confidence(
    *, raw_ocr_text: str, normalized_text: str, expression_parseable: bool
) -> ConfidenceBreakdown:
    length_signal = _length_signal(normalized_text)
    parseable_signal = 1.0 if expression_parseable else 0.0
    # An empty transcription has trivially balanced (zero) delimiters —
    # that's "well-formed" in a vacuous sense, not a real positive
    # signal, so it must not earn credit here.
    well_formed_signal = 1.0 if raw_ocr_text.strip() and _latex_well_formed(raw_ocr_text) else 0.0

    composite = (
        WEIGHT_LENGTH * length_signal
        + WEIGHT_PARSEABLE * parseable_signal
        + WEIGHT_LATEX_WELL_FORMED * well_formed_signal
    )

    if composite >= HIGH_CONFIDENCE_THRESHOLD:
        tier = ConfidenceTier.HIGH
    elif composite >= MEDIUM_CONFIDENCE_THRESHOLD:
        tier = ConfidenceTier.MEDIUM
    else:
        tier = ConfidenceTier.LOW

    return ConfidenceBreakdown(
        ocr_length_signal=length_signal,
        expression_parseable_signal=parseable_signal,
        latex_well_formed_signal=well_formed_signal,
        composite_score=round(composite, 4),
        tier=tier,
    )
