"""
Tests for the Grader/Examiner core: alignment, mark awarding, unsupported-
answer detection, and confidence scoring — all checked against real CAS
ground truth, not fixture data that could drift from what SymPy computes.
"""

from app.cas.solver import differentiate, solve_equation
from app.examiner.grader import grade_submission
from app.examiner.models import ConfidenceTier
from app.questions.mark_scheme import build_mark_scheme


def _chain_rule_mark_scheme():
    cas_result = differentiate("x**2 * sin(x)", "x")
    return build_mark_scheme("ITEM-TEST", cas_result), cas_result


def test_full_credit_for_fully_correct_and_supported_work():
    mark_scheme, _ = _chain_rule_mark_scheme()
    work = (
        "u = x**2, v = sin(x)\n"
        "u_prime = 2*x\n"
        "v_prime = cos(x)\n"
        "therefore dy/dx = 2*x*sin(x) + x**2*cos(x)"
    )
    result = grade_submission("ITEM-TEST", mark_scheme, work)
    assert result.total_awarded == result.total_available == 2
    assert result.method_marks == 1
    assert result.accuracy_marks == 1
    assert result.flags == []
    assert result.confidence == ConfidenceTier.HIGH
    assert result.first_error_step_index is None


def test_correct_answer_algebraically_rearranged_still_gets_full_credit():
    """The accuracy check is symbolic equivalence, not string equality —
    a differently-ordered but equal expression must still be awarded."""
    mark_scheme, _ = _chain_rule_mark_scheme()
    work = "u = x**2, v = sin(x)\nu_prime = 2*x\ntherefore dy/dx = x**2*cos(x) + 2*x*sin(x)"
    result = grade_submission("ITEM-TEST", mark_scheme, work)
    accuracy_award = next(a for a in result.breakdown if a.type == "A")
    assert accuracy_award.marks_awarded == accuracy_award.marks_available


def test_unsupported_correct_answer_flagged_and_loses_method_mark():
    mark_scheme, _ = _chain_rule_mark_scheme()
    work = "therefore dy/dx = 2*x*sin(x) + x**2*cos(x)"
    result = grade_submission("ITEM-TEST", mark_scheme, work)
    assert "unsupported_correct_answer" in result.flags
    assert result.method_marks == 0
    assert result.accuracy_marks == 1
    assert result.confidence == ConfidenceTier.LOW


def test_wrong_final_answer_loses_accuracy_mark_and_localizes_error():
    mark_scheme, _ = _chain_rule_mark_scheme()
    work = "u = x**2, v = sin(x)\ntherefore dy/dx = 2*x*cos(x)"  # wrong
    result = grade_submission("ITEM-TEST", mark_scheme, work)
    accuracy_award = next(a for a in result.breakdown if a.type == "A")
    assert accuracy_award.marks_awarded == 0
    assert result.first_error_step_index is not None
    assert result.confidence in (ConfidenceTier.MEDIUM, ConfidenceTier.LOW)


def test_no_working_and_no_answer_gets_zero_marks():
    mark_scheme, _ = _chain_rule_mark_scheme()
    result = grade_submission("ITEM-TEST", mark_scheme, "I don't know how to start")
    assert result.total_awarded == 0
    assert result.confidence == ConfidenceTier.LOW


def test_comment_is_grounded_in_the_actual_breakdown():
    mark_scheme, _ = _chain_rule_mark_scheme()
    work = "u = x**2, v = sin(x)\nu_prime = 2*x\ntherefore dy/dx = 2*x*sin(x) + x**2*cos(x)"
    result = grade_submission("ITEM-TEST", mark_scheme, work)
    assert f"{result.total_awarded}/{result.total_available}" in result.comment
    for award in result.breakdown:
        assert award.node_id in result.comment


def test_quadratic_solve_mark_scheme_grades_correctly():
    cas_result = solve_equation("x**2 - 4 = 0", "x")
    mark_scheme = build_mark_scheme("ITEM-QUAD", cas_result)
    work = "factor: (x-2)(x+2) = 0\ntherefore x = 2, x = -2"
    result = grade_submission("ITEM-QUAD", mark_scheme, work)
    accuracy_award = next(a for a in result.breakdown if a.type == "A")
    assert accuracy_award.marks_awarded == accuracy_award.marks_available


def test_given_expression_step_does_not_count_as_method_work():
    """A step that's just a restatement of the given (not real working)
    should not, by itself, satisfy the method-mark heuristic — it's
    classified separately from ALGEBRAIC_MANIPULATION."""
    mark_scheme, cas_result = _chain_rule_mark_scheme()
    work = "y = x**2*sin(x)\ntherefore dy/dx = 2*x*sin(x) + x**2*cos(x)"
    result = grade_submission("ITEM-TEST", mark_scheme, work, given_expression="x**2*sin(x)")
    method_award = next(a for a in result.breakdown if a.type == "M")
    assert method_award.marks_awarded == 0
    assert "unsupported_correct_answer" in result.flags
