"""
Real-math assertions against the Math Solver + CAS agent — every result
asserted here is independently checkable by hand, not just "did it run."
"""

import pytest

from app.cas.models import CASOperation, CASStatus
from app.cas.solver import (
    differentiate,
    evaluate,
    integrate,
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
