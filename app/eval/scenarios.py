"""
Hand-authored golden scenarios for the offline eval harness. These are
deliberately distinct from pytest unit tests: they exist to be run
repeatedly across code changes and tracked as a numeric pass rate, not
to assert a single behaviour once. Every expected value here is either
mathematically self-evident (checkable by hand) or independently
verified via SymPy in the harness itself - nothing is copied from a
model's output.

Three categories, matching the three deterministic (no-LLM-call)
subsystems in this codebase: CAS solving, the pure decision policy, and
grading. All three can be eval'd with zero network/model dependency,
which is exactly what makes an offline harness possible here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.models.contracts import (
    ActionType,
    AssessmentMode,
    DecisionSignals,
    FrustrationLevel,
    IntegrityRisk,
    IntentType,
)


@dataclass(frozen=True)
class CASScenario:
    name: str
    operation: str  # one of: differentiate, integrate, solve, simplify, evaluate,
    #                  definite_integrate, determinant, matrix_multiply, piecewise
    expression: str = ""  # unused (empty) for piecewise scenarios, which use pieces_text instead
    variable: str = "x"
    at: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    pieces_text: Optional[str] = None
    expected: str = ""  # value the CAS result must be equivalent to


@dataclass(frozen=True)
class DecisionScenario:
    name: str
    signals: dict = field(default_factory=dict)
    expected_action_type: ActionType = ActionType.QUESTION


@dataclass(frozen=True)
class GradingScenario:
    name: str
    seed_expression: str  # expression differentiate()/solve_equation() is run on
    seed_operation: str  # "differentiate" or "solve"
    student_work: str
    expected_total_awarded: int
    expected_total_available: int


CAS_SCENARIOS: list[CASScenario] = [
    CASScenario(
        name="power_rule_derivative",
        operation="differentiate",
        expression="x**3",
        expected="3*x**2",
    ),
    CASScenario(
        name="product_rule_derivative",
        operation="differentiate",
        expression="x**2 * sin(x)",
        expected="2*x*sin(x) + x**2*cos(x)",
    ),
    CASScenario(
        name="indefinite_integral_power",
        operation="integrate",
        expression="2*x",
        expected="x**2",
    ),
    CASScenario(
        name="definite_integral_exact_fraction",
        operation="definite_integrate",
        expression="x**2",
        lower=0,
        upper=2,
        expected="8/3",
    ),
    CASScenario(
        name="quadratic_solve_two_roots",
        operation="solve",
        expression="x**2 - 4 = 0",
        expected="2, -2",
    ),
    CASScenario(
        name="simplify_trig_identity",
        operation="simplify",
        expression="sin(x)**2 + cos(x)**2",
        expected="1",
    ),
    CASScenario(
        name="evaluate_at_point",
        operation="evaluate",
        expression="x**2 + 1",
        at=3,
        expected="10",
    ),
    CASScenario(
        name="two_by_two_determinant",
        operation="determinant",
        expression="[[1,2],[3,4]]",
        expected="-2",
    ),
    CASScenario(
        name="two_by_two_matrix_multiply",
        operation="matrix_multiply",
        expression="[[1,0],[0,1]] * [[5,6],[7,8]]",
        expected="[[5,6],[7,8]]",
    ),
    CASScenario(
        name="piecewise_below_boundary",
        operation="piecewise",
        pieces_text="x**2 if x < 0; 2*x if x >= 0",
        variable="x",
        at=-2,
        expected="4",
    ),
    CASScenario(
        name="piecewise_at_boundary",
        operation="piecewise",
        pieces_text="x**2 if x < 0; 2*x if x >= 0",
        variable="x",
        at=0,
        expected="0",
    ),
]


def _signals(**overrides) -> dict:
    defaults = dict(
        intent=IntentType.SOLVE_REQUEST,
        mastery_estimate=0.5,
        assessment_mode=AssessmentMode.PRACTICE,
        integrity_risk=IntegrityRisk.LOW,
        attempt_count=1,
        frustration_signal=FrustrationLevel.NONE,
        hint_ladder_level=0,
    )
    defaults.update(overrides)
    return defaults


DECISION_POLICY_SCENARIOS: list[DecisionScenario] = [
    DecisionScenario(
        name="high_integrity_risk_always_refuses",
        signals=_signals(integrity_risk=IntegrityRisk.HIGH),
        expected_action_type=ActionType.REFUSE,
    ),
    DecisionScenario(
        name="integrity_gate_wins_over_challenge_shortcut",
        signals=_signals(integrity_risk=IntegrityRisk.HIGH, mastery_estimate=0.99, attempt_count=1),
        expected_action_type=ActionType.REFUSE,
    ),
    DecisionScenario(
        name="high_mastery_first_attempt_triggers_challenge",
        signals=_signals(mastery_estimate=0.9, attempt_count=1),
        expected_action_type=ActionType.CHALLENGE,
    ),
    DecisionScenario(
        name="graded_take_home_elevated_risk_refuses",
        signals=_signals(
            assessment_mode=AssessmentMode.GRADED_TAKE_HOME,
            integrity_risk=IntegrityRisk.MEDIUM,
            intent=IntentType.SOLVE_REQUEST,
        ),
        expected_action_type=ActionType.REFUSE,
    ),
    DecisionScenario(
        name="first_attempt_solve_request_gets_diagnostic_question",
        signals=_signals(intent=IntentType.SOLVE_REQUEST, mastery_estimate=0.5, attempt_count=1),
        expected_action_type=ActionType.QUESTION,
    ),
    DecisionScenario(
        name="mid_attempt_solve_request_gets_hint",
        signals=_signals(intent=IntentType.SOLVE_REQUEST, mastery_estimate=0.5, attempt_count=2),
        expected_action_type=ActionType.HINT,
    ),
    DecisionScenario(
        name="check_work_intent_explains_with_error_localization",
        signals=_signals(intent=IntentType.CHECK_WORK),
        expected_action_type=ActionType.EXPLAIN,
    ),
    DecisionScenario(
        name="high_frustration_overrides_to_supportive_scaffold",
        signals=_signals(intent=IntentType.SOLVE_REQUEST, frustration_signal=FrustrationLevel.HIGH, attempt_count=3),
        expected_action_type=ActionType.SUPPORTIVE_SCAFFOLD,
    ),
]


GRADING_SCENARIOS: list[GradingScenario] = [
    GradingScenario(
        name="chain_rule_fully_correct_and_supported",
        seed_expression="x**2 * sin(x)",
        seed_operation="differentiate",
        student_work=(
            "u = x**2, v = sin(x)\n"
            "u_prime = 2*x\n"
            "v_prime = cos(x)\n"
            "therefore dy/dx = 2*x*sin(x) + x**2*cos(x)"
        ),
        expected_total_awarded=2,
        expected_total_available=2,
    ),
    GradingScenario(
        name="chain_rule_unsupported_correct_answer",
        seed_expression="x**2 * sin(x)",
        seed_operation="differentiate",
        student_work="therefore dy/dx = 2*x*sin(x) + x**2*cos(x)",
        expected_total_awarded=1,
        expected_total_available=2,
    ),
    GradingScenario(
        name="chain_rule_wrong_final_answer",
        seed_expression="x**2 * sin(x)",
        seed_operation="differentiate",
        student_work="u = x**2, v = sin(x)\ntherefore dy/dx = 2*x*cos(x)",
        expected_total_awarded=1,
        expected_total_available=2,
    ),
    GradingScenario(
        name="chain_rule_no_working_no_answer",
        seed_expression="x**2 * sin(x)",
        seed_operation="differentiate",
        student_work="I don't know how to start",
        expected_total_awarded=0,
        expected_total_available=2,
    ),
    GradingScenario(
        name="quadratic_solve_both_roots_given",
        seed_expression="x**2 - 4 = 0",
        seed_operation="solve",
        student_work="factor: (x-2)(x+2) = 0\ntherefore x = 2, x = -2",
        expected_total_awarded=2,
        expected_total_available=2,
    ),
]
