"""
Grader / Examiner Agent orchestration (spec §2.2, §10): segments the
student's submission, aligns it against the mark scheme, detects
unsupported answers, scores confidence, and produces a grounded comment —
all deterministic, no model call in the critical path (matching the
CAS agent's own "no generative model in the critical path" principle:
correctness of *marks awarded* should never depend on an LLM's judgment).
"""

from __future__ import annotations

from typing import Optional

from app.examiner.alignment import align_and_award, compute_first_error_step_index, detect_unsupported_answer
from app.examiner.comment import generate_examiner_comment
from app.examiner.models import ConfidenceTier, MarkResult, StepType
from app.examiner.segmentation import segment_work
from app.questions.models import MarkScheme

# Spec §10.11 confidence rubric, adapted to this phase's binary (awarded/
# not-awarded) mark model rather than a continuous per-node match score:
#   High   — every mark awarded, no flags raised.
#   Medium — partial credit, no flags.
#   Low    — no final answer found at all, zero marks awarded, or any
#            flag raised (e.g. unsupported_correct_answer).


def _compute_confidence(total_awarded: int, total_available: int, flags: list[str], has_final_answer: bool) -> ConfidenceTier:
    if not has_final_answer or flags:
        return ConfidenceTier.LOW
    if total_available > 0 and total_awarded == total_available:
        return ConfidenceTier.HIGH
    if total_awarded > 0:
        return ConfidenceTier.MEDIUM
    return ConfidenceTier.LOW


def grade_submission(item_id: str, mark_scheme: MarkScheme, student_work: str, given_expression: Optional[str] = None) -> MarkResult:
    work_steps = segment_work(student_work, given_expression=given_expression)
    breakdown = align_and_award(work_steps, mark_scheme)

    total_awarded = sum(award.marks_awarded for award in breakdown)
    total_available = sum(award.marks_available for award in breakdown)
    method_marks = sum(award.marks_awarded for award in breakdown if award.type == "M")
    method_marks_available = sum(award.marks_available for award in breakdown if award.type == "M")
    accuracy_marks = sum(award.marks_awarded for award in breakdown if award.type == "A")

    first_error_step_index = compute_first_error_step_index(work_steps, mark_scheme)

    flags: list[str] = []
    unsupported_flag = detect_unsupported_answer(work_steps, mark_scheme, method_marks, method_marks_available)
    if unsupported_flag:
        flags.append(unsupported_flag)

    has_final_answer = any(step.step_type == StepType.FINAL_ANSWER for step in work_steps)
    confidence = _compute_confidence(total_awarded, total_available, flags, has_final_answer)

    mark_result = MarkResult(
        item_id=item_id,
        total_awarded=total_awarded,
        total_available=total_available,
        breakdown=breakdown,
        method_marks=method_marks,
        accuracy_marks=accuracy_marks,
        first_error_step_index=first_error_step_index,
        flags=flags,
        confidence=confidence,
    )
    mark_result.comment = generate_examiner_comment(mark_result, mark_scheme)
    return mark_result
