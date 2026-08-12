"""
Real dependency-graph stage executor (spec §6, the Planner Agent).

Given a set of named async stages, each with a declared list of
dependency stage names, `run_plan` runs every stage whose dependencies
have already completed concurrently (via `asyncio.gather`), then repeats
until every stage has run — a genuine topological execution, not a fixed
sequence. Wherever a turn's independent side effects don't actually
depend on each other (see app/orchestrator/handle_turn.py's post-grading
writes: Phase 5's mastery write, Phase 8's diagnosis, Phase 9's review
record all key only off an already-completed grading, never off one
another), this replaces "always await them one at a time, in this exact
order" with "run whatever's ready, as soon as it's ready."

A stage that raises is caught and recorded on its StageOutcome rather
than aborting the whole plan or cancelling sibling stages — these are
independent side effects, and one failing is not a reason to skip the
others (the caller decides what a None/error result means, same
graceful-degradation posture used everywhere else in this codebase).

A missing dependency name or a genuine cycle is a real configuration
bug, not a data problem, so that raises immediately rather than being
silently swallowed or hanging forever.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Optional

from app.planner.models import ExecutionPlan, StageOutcome

StageFn = Callable[[], Awaitable[Any]]
# stage name -> (coroutine factory, dependency stage names)
StageGraph = dict[str, tuple[StageFn, list[str]]]


class PlanCycleError(RuntimeError):
    """Raised when the declared stage graph can't make progress: a
    dependency cycle, or a dependency name that was never registered as
    a stage. The executor refuses to silently deadlock on either."""


async def _run_one(name: str, fn: StageFn, depends_on: list[str]) -> tuple[str, Any, StageOutcome]:
    start = time.monotonic()
    error: Optional[str] = None
    value: Any = None
    try:
        value = await fn()
    except Exception as exc:  # noqa: BLE001 - captured per-stage, isolated from siblings
        error = f"{type(exc).__name__}: {exc}"
    duration_ms = (time.monotonic() - start) * 1000
    outcome = StageOutcome(name=name, depends_on=depends_on, duration_ms=round(duration_ms, 3), error=error)
    return name, value, outcome


async def run_plan(stages: StageGraph) -> tuple[dict[str, Any], ExecutionPlan]:
    """Returns (results keyed by stage name, the real ExecutionPlan that
    was executed). A stage whose dependency list references a name never
    present in `stages` — or a genuine cycle — raises PlanCycleError
    rather than hanging: neither is recoverable at runtime."""
    all_names = set(stages)
    for name, (_, deps) in stages.items():
        unknown = [d for d in deps if d not in all_names]
        if unknown:
            raise PlanCycleError(f"stage {name!r} depends on unregistered stage(s) {unknown}")

    remaining = dict(stages)
    results: dict[str, Any] = {}
    outcomes: list[StageOutcome] = []
    plan_start = time.monotonic()

    while remaining:
        ready = [name for name, (_, deps) in remaining.items() if all(d in results for d in deps)]
        if not ready:
            raise PlanCycleError(f"no stage is ready to run; a dependency cycle exists among {list(remaining)}")

        batch = await asyncio.gather(*(_run_one(name, fn, deps) for name, (fn, deps) in ((n, remaining[n]) for n in ready)))
        for name, value, outcome in batch:
            del remaining[name]
            results[name] = value
            outcomes.append(outcome)

    total_duration_ms = round((time.monotonic() - plan_start) * 1000, 3)
    return results, ExecutionPlan(stages=outcomes, total_duration_ms=total_duration_ms)
