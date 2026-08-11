"""
Alignment algorithm (spec §10.3), scoped to what a single-canonical-path,
no-per-node-intermediate-value mark scheme (Phase 3's MarkScheme) actually
supports:

  - "A" (accuracy) nodes carry a CAS-computed `expected_value` and are
    awarded via real symbolic equivalence against a matching WorkStep —
    the same `sympy.simplify(a - b) == 0` check used by
    app.cas.solver.verify_claim, not string matching. A `solve` result
    with multiple roots ("x = -2, x = 2") is handled as a set: every
    expected root must be matched by *some* candidate value somewhere in
    the submission (not necessarily the same step, and a step itself may
    state more than one candidate value on one line, e.g. "x = 2, x = -2").
  - "M" (method) nodes have no machine-checkable expected value (that
    needs the fuller per-node solution-graph of spec §8.3, a later
    phase). They're awarded by a documented, honest heuristic: at least
    one ALGEBRAIC_MANIPULATION step exists that isn't just a
    restatement of the given — i.e. the student showed *some*
    intermediate work, not that it was necessarily the *specific*
    method the mark node names.
  - "First error" localization (spec §10.3/§10.4) is similarly limited:
    without per-node expected values, the most defensible thing this
    phase can report is whether the final stated answer is correct, and
    if not, which step stated it (or that no final answer was given at
    all). This is a real, honest simplification of "localize the exact
    node where things diverge," not a fabricated substitute for it.

Unsupported/lucky-answer detection (spec §10.5) is implemented for real:
a correct final answer with method-mark coverage below the threshold is
flagged, using the spec's own default 0.4 coverage threshold.
"""

from __future__ import annotations

from typing import Optional

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from app.examiner.models import MarkAward, StepType, WorkStep
from app.questions.models import MarkScheme, MarkSchemeNode

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)

# Spec §10.5 default: "coverage < UNSUPPORTED_THRESHOLD (default 0.4)".
UNSUPPORTED_ANSWER_COVERAGE_THRESHOLD = 0.4


def _expressions_equivalent(a: str, b: str) -> bool:
    try:
        expr_a = parse_expr(a, transformations=_TRANSFORMATIONS)
        expr_b = parse_expr(b, transformations=_TRANSFORMATIONS)
        return sympy.simplify(expr_a - expr_b) == 0
    except Exception:  # noqa: BLE001
        return False


def _split_roots(value: str) -> list[str]:
    """"x = -2, x = 2" -> ["-2", "2"]. A single value with no comma
    (and possibly no "var =" prefix) is returned as a one-item list."""
    return [part.rsplit("=", 1)[-1].strip() for part in value.split(",") if part.strip()]


def _candidate_values(work_step: WorkStep) -> list[str]:
    """A step's raw text may itself state multiple comma-separated
    candidate values on one line (e.g. "x = 2, x = -2") even though
    `normalized_expr` (used for single-value comparisons elsewhere) only
    captures the last one — extract all of them here for root-set
    matching."""
    return _split_roots(work_step.raw_text) if "=" in work_step.raw_text else []


def _is_multi_root(expected_value: str) -> bool:
    return "," in expected_value


def _find_matching_step_for_single_value(expected_value: str, checkable_steps: list[WorkStep]) -> Optional[WorkStep]:
    for step in checkable_steps:
        if step.normalized_expr and _expressions_equivalent(step.normalized_expr, expected_value):
            return step
    return None


def _all_roots_covered(expected_value: str, checkable_steps: list[WorkStep]) -> tuple[bool, Optional[int]]:
    expected_roots = _split_roots(expected_value)
    all_candidates: list[tuple[str, int]] = []
    for step in checkable_steps:
        all_candidates.extend((value, step.step_index) for value in _candidate_values(step))

    matched_indices: list[int] = []
    for root in expected_roots:
        match = next((idx for value, idx in all_candidates if _expressions_equivalent(value, root)), None)
        if match is None:
            return False, None
        matched_indices.append(match)
    return True, min(matched_indices) if matched_indices else None


def _award_accuracy_node(node: MarkSchemeNode, work_steps: list[WorkStep]) -> MarkAward:
    checkable_steps = [
        s for s in work_steps if s.step_type in (StepType.ALGEBRAIC_MANIPULATION, StepType.FINAL_ANSWER)
    ]
    expected_value = node.expected_value or ""

    if _is_multi_root(expected_value):
        covered, matched_index = _all_roots_covered(expected_value, checkable_steps)
        if covered:
            return MarkAward(
                node_id=node.id, type=node.type, marks_available=node.marks, marks_awarded=node.marks,
                matched_step_index=matched_index, reason="all expected roots matched across the submission",
            )
    else:
        match = _find_matching_step_for_single_value(expected_value, checkable_steps)
        if match is not None:
            return MarkAward(
                node_id=node.id, type=node.type, marks_available=node.marks, marks_awarded=node.marks,
                matched_step_index=match.step_index, reason=f"step {match.step_index} matches the expected value",
            )

    return MarkAward(
        node_id=node.id, type=node.type, marks_available=node.marks, marks_awarded=0,
        matched_step_index=None, reason="no step matched the expected value",
    )


def _award_method_node(node: MarkSchemeNode, work_steps: list[WorkStep]) -> MarkAward:
    manipulation_steps = [s for s in work_steps if s.step_type == StepType.ALGEBRAIC_MANIPULATION]
    if manipulation_steps:
        return MarkAward(
            node_id=node.id,
            type=node.type,
            marks_available=node.marks,
            marks_awarded=node.marks,
            matched_step_index=manipulation_steps[0].step_index,
            reason="intermediate algebraic work shown",
        )
    return MarkAward(
        node_id=node.id,
        type=node.type,
        marks_available=node.marks,
        marks_awarded=0,
        matched_step_index=None,
        reason="no intermediate working shown",
    )


def align_and_award(work_steps: list[WorkStep], mark_scheme: MarkScheme) -> list[MarkAward]:
    breakdown: list[MarkAward] = []
    for node in mark_scheme.nodes:
        if node.type == "A" and node.expected_value is not None:
            breakdown.append(_award_accuracy_node(node, work_steps))
        else:
            breakdown.append(_award_method_node(node, work_steps))
    return breakdown


def _is_final_answer_correct(work_steps: list[WorkStep], accuracy_node: MarkSchemeNode) -> bool:
    expected_value = accuracy_node.expected_value or ""
    checkable_steps = [
        s for s in work_steps if s.step_type in (StepType.ALGEBRAIC_MANIPULATION, StepType.FINAL_ANSWER)
    ]
    if _is_multi_root(expected_value):
        covered, _ = _all_roots_covered(expected_value, checkable_steps)
        return covered
    return _find_matching_step_for_single_value(expected_value, checkable_steps) is not None


def compute_first_error_step_index(work_steps: list[WorkStep], mark_scheme: MarkScheme) -> Optional[int]:
    accuracy_node = next((n for n in mark_scheme.nodes if n.type == "A" and n.expected_value), None)
    if accuracy_node is None:
        return None

    final_steps = [s for s in work_steps if s.step_type == StepType.FINAL_ANSWER]
    if not final_steps:
        return len(work_steps) if work_steps else None  # no final answer given at all

    if _is_final_answer_correct(work_steps, accuracy_node):
        return None  # correct — no error to localize

    return final_steps[-1].step_index


def detect_unsupported_answer(
    work_steps: list[WorkStep], mark_scheme: MarkScheme, method_marks_awarded: int, method_marks_available: int
) -> Optional[str]:
    accuracy_node = next((n for n in mark_scheme.nodes if n.type == "A" and n.expected_value), None)
    if accuracy_node is None or method_marks_available == 0:
        return None

    final_steps = [s for s in work_steps if s.step_type == StepType.FINAL_ANSWER]
    if not final_steps:
        return None

    if not _is_final_answer_correct(work_steps, accuracy_node):
        return None

    coverage = method_marks_awarded / method_marks_available
    if coverage < UNSUPPORTED_ANSWER_COVERAGE_THRESHOLD:
        return "unsupported_correct_answer"
    return None
