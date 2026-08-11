from app.examiner.models import ConfidenceTier
from app.multimodal.confidence import score_confidence


def test_long_well_formed_parseable_transcription_is_high_confidence():
    breakdown = score_confidence(
        raw_ocr_text=r"\frac{dy}{dx} = 5(2x + 1)^4 \cdot 2",
        normalized_text="dy/dx = 5*(2*x + 1)**4 * 2",
        expression_parseable=True,
    )
    assert breakdown.tier == ConfidenceTier.HIGH
    assert breakdown.composite_score > 0.75


def test_empty_transcription_is_low_confidence():
    breakdown = score_confidence(raw_ocr_text="", normalized_text="", expression_parseable=False)
    assert breakdown.tier == ConfidenceTier.LOW
    assert breakdown.composite_score == 0.0


def test_unparseable_but_well_formed_transcription_is_not_high_confidence():
    breakdown = score_confidence(
        raw_ocr_text="I am not sure what this says",
        normalized_text="I am not sure what this says",
        expression_parseable=False,
    )
    assert breakdown.tier != ConfidenceTier.HIGH


def test_unbalanced_latex_delimiters_lower_the_well_formed_signal():
    breakdown = score_confidence(
        raw_ocr_text=r"\(x^2 + 1 unbalanced",
        normalized_text="x^2 + 1 unbalanced",
        expression_parseable=False,
    )
    assert breakdown.latex_well_formed_signal == 0.0


def test_short_transcription_gets_partial_length_credit():
    breakdown = score_confidence(raw_ocr_text="x=1", normalized_text="x=1", expression_parseable=True)
    assert 0.0 < breakdown.ocr_length_signal < 1.0
