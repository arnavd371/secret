"""
Real-math assertions against the Math Solver + CAS agent — every result
asserted here is independently checkable by hand, not just "did it run."
"""

import pytest

from app.cas.models import CASOperation, CASStatus
from app.cas.solver import (
    determinant,
    differentiate,
    evaluate,
    evaluate_piecewise,
    integrate,
    matrix_multiply,
    run_cas_operation,
    run_cas_operation_async,
    simplify_expr,
    solve_equation,
    verify_claim,
)


def test_differentiate_product_rule():
    result = differentiate("x**2 * sin(x)", "x")
    assert result.status == CASStatus.OK
    assert result.operation == CASOperation.DIFFERENTIATE
    # d/dx[x^2 sin(x)] = x^2 cos(x) + 2x sin(x), in whatever equivalent
    # arrangement SymPy settles on — verify algebraically, not by string
    # equality with one particular arrangement.
    assert verify_claim(result, "2*x*sin(x) + x**2*cos(x)")
    assert "product_rule" in result.steps[0]


def test_differentiate_chain_rule_detected():
    result = differentiate("sin(3*x + 1)", "x")
    assert result.status == CASStatus.OK
    assert verify_claim(result, "3*cos(3*x + 1)")
    assert "chain_rule" in result.steps[0]


def test_differentiate_simple_power_rule():
    result = differentiate("x**3", "x")
    assert verify_claim(result, "3*x**2")


def test_solve_linear_equation():
    result = solve_equation("2*x + 3 = 7", "x")
    assert result.status == CASStatus.OK
    assert result.result_decimal_at == pytest.approx(2.0)
    assert verify_claim(result, "x = 2")
    assert not verify_claim(result, "x = 3")


def test_solve_quadratic_equation_no_single_decimal():
    result = solve_equation("x**2 - 4 = 0", "x")
    assert result.status == CASStatus.OK
    # two roots -> no single result_decimal_at
    assert result.result_decimal_at is None
    assert "-2" in result.result_exact
    assert "2" in result.result_exact


def test_integrate_power_rule_includes_constant():
    result = integrate("2*x", "x")
    assert result.status == CASStatus.OK
    assert "+ C" in result.result_exact
    assert verify_claim(result, "x**2 + C")


def test_simplify_identity_to_zero():
    result = simplify_expr("(x+1)**2 - (x**2 + 2*x + 1)")
    assert result.status == CASStatus.OK
    assert result.result_decimal_at == pytest.approx(0.0)


def test_evaluate_at_point():
    result = evaluate("x**2 + 1", "x", 2.0)
    assert result.status == CASStatus.OK
    assert result.result_decimal_at == pytest.approx(5.0)


def test_malformed_expression_is_unverifiable_not_a_crash():
    result = differentiate("this is not math ###", "x")
    assert result.status == CASStatus.UNVERIFIABLE
    assert result.domain_notes  # a reason is recorded


def test_unsolvable_equation_is_unverifiable():
    # transcendental, no closed-form solution -> SymPy raises NotImplementedError
    result = solve_equation("x - cos(x) = 0", "x")
    assert result.status == CASStatus.UNVERIFIABLE


def test_run_cas_operation_dispatches_by_string():
    result = run_cas_operation("differentiate", "x**2", "x")
    assert result.status == CASStatus.OK
    assert verify_claim(result, "2*x")


def test_run_cas_operation_unsupported_operation_is_unverifiable():
    result = run_cas_operation("factorize_into_primes", "x**2", "x")  # type: ignore[arg-type]
    assert result.status == CASStatus.UNVERIFIABLE


@pytest.mark.asyncio
async def test_run_cas_operation_async_matches_sync():
    result = await run_cas_operation_async("differentiate", "x**2", "x")
    assert result.status == CASStatus.OK
    assert verify_claim(result, "2*x")


@pytest.mark.asyncio
async def test_run_cas_operation_async_times_out_gracefully():
    result = await run_cas_operation_async("differentiate", "x**2", "x", timeout_seconds=0.0)
    assert result.status == CASStatus.UNVERIFIABLE
    assert "timed out" in result.domain_notes[0]


def test_verify_claim_rejects_when_cas_result_itself_unverifiable():
    bad = differentiate("### not math", "x")
    assert verify_claim(bad, "anything") is False


def test_verify_claim_rejects_unparseable_claim():
    good = differentiate("x**2", "x")
    assert verify_claim(good, "not an expression either ###") is False


# ---------------------------------------------------------------------------
# Phase 12: definite integrals, matrices, piecewise functions
# ---------------------------------------------------------------------------


def test_definite_integral_of_x_squared_from_0_to_2():
    result = integrate("x**2", "x", lower=0, upper=2)
    assert result.status == CASStatus.OK
    # ∫[0,2] x^2 dx = [x^3/3] = 8/3, exactly, not "8/3 + C"
    assert result.result_exact == "8/3"
    assert result.result_decimal_at == pytest.approx(8 / 3)


def test_definite_integral_negative_bounds():
    # ∫[-1,1] x dx = 0 by symmetry
    result = integrate("x", "x", lower=-1, upper=1)
    assert result.result_exact == "0"


def test_indefinite_integral_still_unaffected_by_the_bounds_extension():
    result = integrate("x**2", "x")
    assert result.result_exact == "x**3/3 + C"


def test_determinant_2x2():
    result = determinant("[[1,2],[3,4]]")
    assert result.status == CASStatus.OK
    assert result.result_exact == "-2"


def test_determinant_3x3():
    # Identity matrix, det = 1
    result = determinant("[[1,0,0],[0,1,0],[0,0,1]]")
    assert result.result_exact == "1"


def test_determinant_of_a_non_square_matrix_is_unverifiable():
    result = determinant("[[1,2,3],[4,5,6]]")
    assert result.status == CASStatus.UNVERIFIABLE


def test_determinant_with_symbolic_entries():
    result = determinant("[[x,1],[0,x]]")
    assert result.result_exact == "x**2"


def test_matrix_multiply_2x2():
    result = matrix_multiply("[[1,2],[3,4]] * [[5,6],[7,8]]")
    assert result.status == CASStatus.OK
    assert result.result_exact == "[[19, 22], [43, 50]]"


def test_matrix_multiply_incompatible_dimensions_is_unverifiable():
    result = matrix_multiply("[[1,2,3]] * [[1,2]]")
    assert result.status == CASStatus.UNVERIFIABLE


def test_evaluate_piecewise_selects_the_matching_branch():
    negative = evaluate_piecewise("x**2 if x < 0; 2*x if x >= 0", "x", -3)
    positive = evaluate_piecewise("x**2 if x < 0; 2*x if x >= 0", "x", 3)
    assert negative.result_exact == "9"
    assert positive.result_exact == "6"


def test_evaluate_piecewise_default_branch_with_no_condition():
    result = evaluate_piecewise("1/x if x != 0; 0", "x", 0)
    assert result.result_exact == "0"


def test_evaluate_piecewise_with_no_matching_branch_is_unverifiable():
    result = evaluate_piecewise("x**2 if x < 0", "x", 5)
    assert result.status == CASStatus.UNVERIFIABLE


def test_run_cas_operation_dispatches_determinant_and_definite_integral():
    det_result = run_cas_operation(CASOperation.DETERMINANT, "[[2,0],[0,3]]")
    assert det_result.result_exact == "6"

    integral_result = run_cas_operation(CASOperation.INTEGRATE, "x**2", "x", 0, 2)
    assert integral_result.result_exact == "8/3"
