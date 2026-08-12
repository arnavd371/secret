"""
Tests for the offline eval harness: real scenarios run against the real
decision policy / CAS solver / grader, no mocking (there is nothing to
mock - this subsystem is entirely deterministic, no LLM calls). The
regression gate is tested against synthetic reports built by hand so a
genuine regression can be constructed and confirmed to be caught.
"""

from app.eval.harness import run_eval_suite
from app.eval.models import EvalCaseResult, EvalCategory, EvalReport
from app.eval.regression_gate import check_regression
from app.eval.scenarios import CAS_SCENARIOS, DECISION_POLICY_SCENARIOS, GRADING_SCENARIOS


def test_every_hand_authored_scenario_currently_passes():
    """The golden dataset itself must be internally consistent - every
    expected value was derived from real computation, not guessed."""
    report = run_eval_suite()
    failed = report.failed_cases()
    assert failed == [], f"unexpected failures: {[(c.name, c.detail) for c in failed]}"
    assert report.pass_rate == 1.0


def test_report_covers_every_declared_scenario():
    report = run_eval_suite()
    assert report.total == len(CAS_SCENARIOS) + len(DECISION_POLICY_SCENARIOS) + len(GRADING_SCENARIOS)


def test_report_is_broken_down_by_category():
    report = run_eval_suite()
    rates = report.category_pass_rates()
    assert set(rates.keys()) == {"cas", "decision_policy", "grading"}
    for rate in rates.values():
        assert rate == 1.0


def test_cas_scenario_catches_a_real_regression():
    """A deliberately wrong expected value must fail, proving the
    checker isn't vacuously true - not just that the real scenarios
    happen to pass."""
    from app.eval.harness import _run_cas_scenario
    from app.eval.scenarios import CASScenario

    broken = CASScenario(name="deliberately_wrong", operation="differentiate", expression="x**2", expected="x**3")
    result = _run_cas_scenario(broken)
    assert result.passed is False
    assert result.category == EvalCategory.CAS


def test_decision_scenario_catches_a_real_regression():
    from app.eval.harness import _run_decision_scenario
    from app.eval.scenarios import DecisionScenario
    from app.models.contracts import ActionType

    broken = DecisionScenario(
        name="deliberately_wrong",
        signals=dict(
            intent="solve_request",
            mastery_estimate=0.5,
            assessment_mode="practice",
            integrity_risk="high",
            attempt_count=1,
            frustration_signal="none",
            hint_ladder_level=0,
        ),
        expected_action_type=ActionType.CHALLENGE,  # actually REFUSE, integrity gate wins
    )
    result = _run_decision_scenario(broken)
    assert result.passed is False


def test_grading_scenario_catches_a_real_regression():
    from app.eval.harness import _run_grading_scenario
    from app.eval.scenarios import GradingScenario

    broken = GradingScenario(
        name="deliberately_wrong",
        seed_expression="x**2 * sin(x)",
        seed_operation="differentiate",
        student_work="I don't know how to start",
        expected_total_awarded=2,  # actually 0
        expected_total_available=2,
    )
    result = _run_grading_scenario(broken)
    assert result.passed is False


def test_regression_gate_passes_when_current_matches_baseline():
    report = run_eval_suite()
    gate = check_regression(report, report)
    assert gate.passed is True
    assert gate.regressions == []


def test_regression_gate_fails_when_a_category_pass_rate_drops():
    baseline = EvalReport(
        results=[
            EvalCaseResult(name="a", category=EvalCategory.CAS, passed=True),
            EvalCaseResult(name="b", category=EvalCategory.CAS, passed=True),
        ]
    )
    current = EvalReport(
        results=[
            EvalCaseResult(name="a", category=EvalCategory.CAS, passed=True),
            EvalCaseResult(name="b", category=EvalCategory.CAS, passed=False),
        ]
    )
    gate = check_regression(current, baseline)
    assert gate.passed is False
    assert gate.regressions[0].category == "cas"
    assert gate.regressions[0].baseline_pass_rate == 1.0
    assert gate.regressions[0].current_pass_rate == 0.5


def test_regression_gate_ignores_an_improvement():
    baseline = EvalReport(
        results=[
            EvalCaseResult(name="a", category=EvalCategory.GRADING, passed=True),
            EvalCaseResult(name="b", category=EvalCategory.GRADING, passed=False),
        ]
    )
    current = EvalReport(
        results=[
            EvalCaseResult(name="a", category=EvalCategory.GRADING, passed=True),
            EvalCaseResult(name="b", category=EvalCategory.GRADING, passed=True),
        ]
    )
    gate = check_regression(current, baseline)
    assert gate.passed is True


def test_regression_gate_only_flags_the_category_that_actually_regressed():
    baseline = EvalReport(
        results=[
            EvalCaseResult(name="a", category=EvalCategory.CAS, passed=True),
            EvalCaseResult(name="b", category=EvalCategory.GRADING, passed=True),
        ]
    )
    current = EvalReport(
        results=[
            EvalCaseResult(name="a", category=EvalCategory.CAS, passed=False),
            EvalCaseResult(name="b", category=EvalCategory.GRADING, passed=True),
        ]
    )
    gate = check_regression(current, baseline)
    assert len(gate.regressions) == 1
    assert gate.regressions[0].category == "cas"


def test_empty_report_has_perfect_pass_rate_by_convention():
    """An empty scenario set is vacuously 100% - the gate should never
    divide by zero or treat "nothing ran" as a failure."""
    report = EvalReport(results=[])
    assert report.pass_rate == 1.0
    assert report.pass_rate_for(EvalCategory.CAS) == 1.0


def test_baseline_round_trips_through_disk(tmp_path):
    from app.eval.baseline import baseline_exists, load_baseline, save_baseline

    path = tmp_path / "baseline.json"
    assert baseline_exists(path) is False

    report = run_eval_suite()
    save_baseline(report, path)
    assert baseline_exists(path) is True

    loaded = load_baseline(path)
    assert loaded.total == report.total
    assert loaded.pass_rate == report.pass_rate
    assert {r.name for r in loaded.results} == {r.name for r in report.results}


def test_load_baseline_without_a_file_raises_a_clear_error(tmp_path):
    from app.eval.baseline import load_baseline

    missing = tmp_path / "does_not_exist.json"
    try:
        load_baseline(missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "run" in str(exc).lower()
