"""
Tests for the real dependency-graph stage executor. The concurrency
tests measure actual wall-clock time against asyncio.sleep-based stages
to prove stages genuinely run in parallel, not just that the code
"looks async" — a wrong (sequential) implementation would fail these on
timing alone, not just on some abstract property.
"""

import asyncio
import time

import pytest

from app.planner.executor import PlanCycleError, run_plan
from app.planner.models import ExecutionPlan


async def _sleep_and_return(seconds: float, value):
    await asyncio.sleep(seconds)
    return value


@pytest.mark.asyncio
async def test_independent_stages_run_concurrently_not_sequentially():
    stages = {
        "a": (lambda: _sleep_and_return(0.2, "a"), []),
        "b": (lambda: _sleep_and_return(0.2, "b"), []),
    }
    start = time.monotonic()
    results, _ = await run_plan(stages)
    elapsed = time.monotonic() - start

    assert results == {"a": "a", "b": "b"}
    # Sequential would take >= 0.4s; concurrent should be close to 0.2s.
    assert elapsed < 0.35


@pytest.mark.asyncio
async def test_dependent_stage_waits_for_its_dependency():
    order: list[str] = []

    async def first():
        await asyncio.sleep(0.1)
        order.append("first")
        return "first-done"

    async def second():
        order.append("second")
        return "second-done"

    stages = {"first": (first, []), "second": (second, ["first"])}
    results, _ = await run_plan(stages)

    assert order == ["first", "second"]
    assert results == {"first": "first-done", "second": "second-done"}


@pytest.mark.asyncio
async def test_diamond_dependency_graph_resolves_correctly():
    #   a
    #  / \
    # b   c
    #  \ /
    #   d
    order: list[str] = []

    def make(name, deps_ready_check=None):
        async def _fn():
            order.append(name)
            return name
        return _fn

    stages = {
        "a": (make("a"), []),
        "b": (make("b"), ["a"]),
        "c": (make("c"), ["a"]),
        "d": (make("d"), ["b", "c"]),
    }
    results, _ = await run_plan(stages)

    assert order[0] == "a"
    assert order[-1] == "d"
    assert set(order[1:3]) == {"b", "c"}
    assert results == {"a": "a", "b": "b", "c": "c", "d": "d"}


@pytest.mark.asyncio
async def test_a_stage_failure_is_isolated_from_its_siblings():
    async def ok():
        return "fine"

    async def bad():
        raise ValueError("boom")

    stages = {"ok": (ok, []), "bad": (bad, [])}
    results, plan = await run_plan(stages)

    assert results["ok"] == "fine"
    assert results["bad"] is None
    bad_outcome = next(s for s in plan.stages if s.name == "bad")
    assert bad_outcome.error is not None
    assert "boom" in bad_outcome.error
    ok_outcome = next(s for s in plan.stages if s.name == "ok")
    assert ok_outcome.error is None


@pytest.mark.asyncio
async def test_a_failed_dependency_still_unblocks_its_dependents():
    """A stage's dependency failing (returning None, with an error
    recorded) must not deadlock the plan — the dependent still runs once
    the dependency has *completed* (successfully or not)."""
    async def bad():
        raise ValueError("boom")

    async def dependent():
        return "ran anyway"

    stages = {"bad": (bad, []), "dependent": (dependent, ["bad"])}
    results, plan = await run_plan(stages)

    assert results["dependent"] == "ran anyway"
    assert len(plan.stages) == 2


@pytest.mark.asyncio
async def test_cycle_is_detected_not_hung():
    async def noop():
        return None

    with pytest.raises(PlanCycleError):
        await run_plan({"x": (noop, ["y"]), "y": (noop, ["x"])})


@pytest.mark.asyncio
async def test_self_dependency_is_detected():
    async def noop():
        return None

    with pytest.raises(PlanCycleError):
        await run_plan({"x": (noop, ["x"])})


@pytest.mark.asyncio
async def test_unregistered_dependency_is_detected():
    async def noop():
        return None

    with pytest.raises(PlanCycleError):
        await run_plan({"x": (noop, ["nonexistent"])})


@pytest.mark.asyncio
async def test_empty_plan_returns_empty_results():
    results, plan = await run_plan({})
    assert results == {}
    assert plan.stages == []
    assert plan == ExecutionPlan(stages=[], total_duration_ms=plan.total_duration_ms)


@pytest.mark.asyncio
async def test_execution_plan_records_real_per_stage_timing():
    async def slow():
        await asyncio.sleep(0.05)
        return "done"

    _, plan = await run_plan({"slow": (slow, [])})
    assert plan.stages[0].duration_ms >= 45  # real measured time, not a placeholder
    assert plan.total_duration_ms >= 45
