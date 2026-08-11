"""
Orchestrates the two-tier diagnosis (spec §8): try the real pattern
detectors first (app.diagnostician.detectors), and only fall back to the
model-inference call (app.diagnostician.model_fallback) when nothing
matched. Called from the orchestrator only after a check_work submission
has already been graded and confirmed wrong — grading a correct
submission has nothing to diagnose.
"""

from __future__ import annotations

from typing import Optional

from app.cas.extraction import MathTask
from app.cas.models import CASResult, CASStatus
from app.diagnostician.detectors import detect_pattern_misconception
from app.diagnostician.model_fallback import diagnose_via_model
from app.diagnostician.models import DiagnosisMethod, DiagnosisResult
from app.examiner.models import StepType, WorkStep
from app.examiner.segmentation import segment_work
from app.llm.client import ModelRouter


def _final_answer_value(work_steps: list[WorkStep]) -> Optional[str]:
    final_steps = [s for s in work_steps if s.step_type == StepType.FINAL_ANSWER]
    if not final_steps:
        return None
    # raw_text (not normalized_expr) is used deliberately: it may state
    # multiple comma-separated candidate values on one line (e.g.
    # "x = -2, x = 2"), which detect_pattern_misconception's own root-set
    # matching for SOLVE tasks needs, and normalized_expr only captures
    # the last one.
    return final_steps[-1].raw_text


async def diagnose_misconception(
    router: ModelRouter, math_task: MathTask, cas_result: CASResult, student_work: str
) -> DiagnosisResult:
    if cas_result.status != CASStatus.OK:
        return DiagnosisResult(evidence="no CAS-verified ground truth available for this problem")

    work_steps = segment_work(student_work, given_expression=math_task.expression)
    student_value = _final_answer_value(work_steps)
    if student_value is None:
        return DiagnosisResult(evidence="no final answer found in the submission to diagnose")

    pattern = detect_pattern_misconception(
        math_task.operation, math_task.expression, math_task.variable, student_value
    )
    if pattern is not None:
        misconception_id, evidence = pattern
        return DiagnosisResult(
            misconception_id=misconception_id,
            confidence=1.0,
            method=DiagnosisMethod.PATTERN_MATCH,
            evidence=evidence,
        )

    return await diagnose_via_model(
        router, math_task.operation.value, math_task.expression, cas_result.result_exact or "", student_value
    )
