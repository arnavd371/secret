from app.cas.models import CASOperation
from app.cas.extraction import extract_math_task


def test_extracts_differentiate_from_natural_phrasing():
    task = extract_math_task("Can you differentiate x^2 * sin(x) for me?")
    assert task is not None
    assert task.operation == CASOperation.DIFFERENTIATE
    assert task.expression == "x^2 * sin(x)"


def test_extracts_derivative_of_phrasing():
    task = extract_math_task("What is the derivative of 3x^3 - 2x?")
    assert task is not None
    assert task.operation == CASOperation.DIFFERENTIATE
    assert task.expression == "3x^3 - 2x"


def test_extracts_solve_with_equation():
    task = extract_math_task("solve 2x + 3 = 7")
    assert task is not None
    assert task.operation == CASOperation.SOLVE
    assert "=" in task.expression


def test_extracts_solve_for_named_variable():
    task = extract_math_task("Please solve x^2 - 4 = 0 for y")
    assert task is not None
    assert task.variable == "y"


def test_solve_without_equals_sign_is_not_extracted_as_solve():
    """'solve' without an '=' isn't a checkable equation — should fall
    through to no extraction (or another operation), not a malformed
    solve task."""
    task = extract_math_task("solve this problem for me")
    assert task is None or task.operation != CASOperation.SOLVE


def test_extracts_integrate_with_respect_to_variable():
    task = extract_math_task("integrate 2*t with respect to t please")
    assert task is not None
    assert task.operation == CASOperation.INTEGRATE
    assert task.variable == "t"
    assert task.expression == "2*t"


def test_extracts_simplify():
    task = extract_math_task("simplify (x+1)^2 - (x^2+2x+1)")
    assert task is not None
    assert task.operation == CASOperation.SIMPLIFY


def test_extracts_evaluate_at_point():
    task = extract_math_task("evaluate x^2 + 1 at x=2")
    assert task is not None
    assert task.operation == CASOperation.EVALUATE
    assert task.at == 2.0


def test_no_extraction_for_unrelated_text():
    assert extract_math_task("I don't know what to do here") is None


def test_no_extraction_for_greeting():
    assert extract_math_task("hi, can you help me study today?") is None


# ---------------------------------------------------------------------------
# Phase 12: definite integrals, determinant
# ---------------------------------------------------------------------------


def test_extracts_definite_integral_with_bounds():
    task = extract_math_task("integrate x**2 from 0 to 2")
    assert task is not None
    assert task.operation == CASOperation.INTEGRATE
    assert task.expression == "x**2"
    assert task.at == 0.0
    assert task.upper_at == 2.0


def test_extracts_definite_integral_with_negative_bounds():
    task = extract_math_task("integrate x from -1 to 1")
    assert task.at == -1.0
    assert task.upper_at == 1.0


def test_extracts_definite_integral_with_respect_to_variable():
    task = extract_math_task("integrate t**2 from 0 to 3 with respect to t")
    assert task.variable == "t"
    assert task.at == 0.0
    assert task.upper_at == 3.0


def test_indefinite_integral_extraction_leaves_bounds_unset():
    task = extract_math_task("integrate x**2 with respect to x")
    assert task.operation == CASOperation.INTEGRATE
    assert task.at is None
    assert task.upper_at is None


def test_extracts_determinant_request():
    task = extract_math_task("what's the determinant of [[1,2],[3,4]]")
    assert task is not None
    assert task.operation == CASOperation.DETERMINANT
    assert task.expression == "[[1,2],[3,4]]"
