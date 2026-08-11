from app.diagnostician.models import DiagnosisMethod, DiagnosisResult
from app.diagnostician.write_policy import MODEL_INFERENCE_CONFIDENCE_THRESHOLD, should_write_diagnosis


def test_no_misconception_id_is_never_written():
    result = DiagnosisResult(misconception_id=None, confidence=1.0, method=None)
    assert should_write_diagnosis(result) is False


def test_pattern_match_at_full_confidence_is_written():
    result = DiagnosisResult(misconception_id="MISC-CALC-010", confidence=1.0, method=DiagnosisMethod.PATTERN_MATCH)
    assert should_write_diagnosis(result) is True


def test_model_inference_above_threshold_is_written():
    result = DiagnosisResult(
        misconception_id="MISC-CALC-010",
        confidence=MODEL_INFERENCE_CONFIDENCE_THRESHOLD + 0.01,
        method=DiagnosisMethod.MODEL_INFERENCE,
    )
    assert should_write_diagnosis(result) is True


def test_model_inference_below_threshold_is_not_written():
    result = DiagnosisResult(
        misconception_id="MISC-CALC-010",
        confidence=MODEL_INFERENCE_CONFIDENCE_THRESHOLD - 0.01,
        method=DiagnosisMethod.MODEL_INFERENCE,
    )
    assert should_write_diagnosis(result) is False
