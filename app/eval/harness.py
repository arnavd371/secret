"""
Offline eval harness: runs every hand-authored scenario in
app.eval.scenarios against the real decision policy, real CAS solver,
and real grader, and reports an EvalReport. No LLM call anywhere in
this module - the three subsystems it exercises are exactly this
codebase's deterministic core, which is what makes a fast, free,
fully-repeatable offline harness possible at all (spec's generative
paths - authoring, critique, diagnosis - are not eval'd here; they were
each already covered by mocked-provider tests at build time, per this
project's standing test convention).

Each scenario's expected value is checked by real computation (SymPy
symbolic equivalence, root-set comparison, matrix equality, or the real
grader's own mark totals), never by string equality against a
CAS/grader's raw output - a cosmetically different but mathematically
equal result must still pass.
"""

from __future__ import annotations

import ast
from typing import Optional

import sympy

from app.cas.models import CASStatus
from app.cas.solver import (
    determinant,
    differentiate,
    evaluate,
    evaluate_piecewise,
    integrate,
    matrix_multiply,
    simplify_expr,
    solve_equation,
)
from app.eval.models import EvalCaseResult, EvalCategory, EvalReport
from app.eval.scenarios import (
    CAS_SCENARIOS,
    DECISION_POLICY_SCENARIOS,
    GRADING_SCENARIOS,
    CASScenario,
    DecisionScenario,
    GradingScenario,
)
from app.examiner.grader import grade_submission
from app.models.contracts import DecisionSignals
from app.policy.decision import decide_pedagogical_action
from app.questions.mark_scheme import build_mark_scheme


def _symbolic_equal(actual: Optional[str], expected: str) -> bool:
    """True SymPy equivalence, not string equality - "2*x*sin(x) +
    x**2*cos(x)" and "x**2*cos(x) + 2*x*sin(x)" must both pass."""
    if actual is None:
        return False
    try:
        return sympy.simplify(sympy.sympify(actual) - sympy.sympify(expected)) == 0
    except Exception:  # noqa: BLE001
        return False


def _roots_equal(result_exact: Optional[str], expected_csv: str) -> bool:
    """`result_exact` is solve_equation's own "var = sol, var = sol"
    format; `expected_csv` is a plain comma-separated list of expected
    root values. Compared as sets, since solution order isn't part of
    the contract."""
    if result_exact is None:
        return False
    try:
        got = {sympy.simplify(sympy.sympify(part.strip().split("=", 1)[-1])) for part in result_exact.split(",")}
        want = {sympy.simplify(sympy.sympify(part.strip())) for part in expected_csv.split(",")}
        return got == want
    except Exception:  # noqa: BLE001
        return False


def _matrices_equal(result_exact: Optional[str], expected_literal: str) -> bool:
    """`result_exact` is matrix_multiply/determinant's own
    str(Matrix.tolist()) form - a real Python list literal, safely
    parsed with ast.literal_eval (never `eval`)."""
    if result_exact is None:
        return False
    try:
        got = sympy.Matrix(ast.literal_eval(result_exact))
        want = sympy.Matrix(ast.literal_eval(expected_literal))
        if got.shape != want.shape:
            return False
        return sympy.simplify(got - want) == sympy.zeros(*got.shape)
    except Exception:  # noqa: BLE001
        return False


def _run_cas_scenario(scenario: CASScenario) -> EvalCaseResult:
    try:
        if scenario.operation == "differentiate":
            result = differentiate(scenario.expression, scenario.variable)
            ok = _symbolic_equal(result.result_exact, scenario.expected)
        elif scenario.operation == "integrate":
            result = integrate(scenario.expression, scenario.variable)
            exact = result.result_exact[: -len(" + C")] if result.result_exact and result.result_exact.endswith(" + C") else result.result_exact
            ok = _symbolic_equal(exact, scenario.expected)
        elif scenario.operation == "definite_integrate":
            result = integrate(scenario.expression, scenario.variable, lower=scenario.lower, upper=scenario.upper)
            ok = _symbolic_equal(result.result_exact, scenario.expected)
        elif scenario.operation == "solve":
            result = solve_equation(scenario.expression, scenario.variable)
            ok = _roots_equal(result.result_exact, scenario.expected)
        elif scenario.operation == "simplify":
            result = simplify_expr(scenario.expression)
            ok = _symbolic_equal(result.result_exact, scenario.expected)
        elif scenario.operation == "evaluate":
            result = evaluate(scenario.expression, scenario.variable, scenario.at)
            ok = _symbolic_equal(result.result_exact, scenario.expected)
        elif scenario.operation == "determinant":
            result = determinant(scenario.expression)
            ok = _symbolic_equal(result.result_exact, scenario.expected)
        elif scenario.operation == "matrix_multiply":
            result = matrix_multiply(scenario.expression)
            ok = _matrices_equal(result.result_exact, scenario.expected)
        elif scenario.operation == "piecewise":
            result = evaluate_piecewise(scenario.pieces_text or "", scenario.variable, scenario.at)
            ok = _symbolic_equal(result.result_exact, scenario.expected)
        else:
            return EvalCaseResult(
                name=scenario.name,
                category=EvalCategory.CAS,
                passed=False,
                detail=f"unknown scenario operation {scenario.operation!r}",
            )

        passed = ok and result.status == CASStatus.OK
        detail = "" if passed else f"status={result.status.value} result_exact={result.result_exact!r} expected~={scenario.expected!r}"
        return EvalCaseResult(name=scenario.name, category=EvalCategory.CAS, passed=passed, detail=detail)
    except Exception as exc:  # noqa: BLE001
        return EvalCaseResult(name=scenario.name, category=EvalCategory.CAS, passed=False, detail=f"error: {exc}")


def _run_decision_scenario(scenario: DecisionScenario) -> EvalCaseResult:
    try:
        signals = DecisionSignals(**scenario.signals)
        action = decide_pedagogical_action(signals)
        passed = action.action_type == scenario.expected_action_type
        detail = "" if passed else f"got {action.action_type.value}, expected {scenario.expected_action_type.value}"
        return EvalCaseResult(name=scenario.name, category=EvalCategory.DECISION_POLICY, passed=passed, detail=detail)
    except Exception as exc:  # noqa: BLE001
        return EvalCaseResult(
            name=scenario.name, category=EvalCategory.DECISION_POLICY, passed=False, detail=f"error: {exc}"
        )


def _run_grading_scenario(scenario: GradingScenario) -> EvalCaseResult:
    try:
        item_id = f"EVAL-{scenario.name}"
        if scenario.seed_operation == "differentiate":
            cas_result = differentiate(scenario.seed_expression)
        elif scenario.seed_operation == "solve":
            cas_result = solve_equation(scenario.seed_expression)
        else:
            return EvalCaseResult(
                name=scenario.name,
                category=EvalCategory.GRADING,
                passed=False,
                detail=f"unknown seed_operation {scenario.seed_operation!r}",
            )

        mark_scheme = build_mark_scheme(item_id, cas_result)
        result = grade_submission(item_id, mark_scheme, scenario.student_work)
        passed = (
            result.total_awarded == scenario.expected_total_awarded
            and result.total_available == scenario.expected_total_available
        )
        detail = (
            ""
            if passed
            else f"got {result.total_awarded}/{result.total_available}, "
            f"expected {scenario.expected_total_awarded}/{scenario.expected_total_available}"
        )
        return EvalCaseResult(name=scenario.name, category=EvalCategory.GRADING, passed=passed, detail=detail)
    except Exception as exc:  # noqa: BLE001
        return EvalCaseResult(name=scenario.name, category=EvalCategory.GRADING, passed=False, detail=f"error: {exc}")


def run_eval_suite() -> EvalReport:
    """Synchronous and side-effect-free: every scenario exercises a pure
    function or a real-but-in-memory computation, so the whole suite
    runs in well under a second and needs no event loop, no store setup,
    and no API key."""
    results: list[EvalCaseResult] = []
    results.extend(_run_cas_scenario(s) for s in CAS_SCENARIOS)
    results.extend(_run_decision_scenario(s) for s in DECISION_POLICY_SCENARIOS)
    results.extend(_run_grading_scenario(s) for s in GRADING_SCENARIOS)
    return EvalReport(results=results)
