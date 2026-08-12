"""
Typed contracts for the Planner Agent (spec §6): a real, serializable
record of a stage graph that actually ran, stored on
Blackboard.execution_plan (a Phase-1-era stub this phase finally
populates for real, same precedent as Phase 5 upgrading
student_state_snapshot and Phase 8 upgrading diagnosis_result).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class StageOutcome(BaseModel):
    name: str
    depends_on: list[str] = Field(default_factory=list)
    duration_ms: float
    # None on success. A stage that raised is recorded here rather than
    # aborting the whole plan — these stages are independent side effects
    # (a mastery write, a diagnosis, a review record), and one failing is
    # not a reason to skip the other two, same graceful-degradation
    # posture as every other phase's model-call failure handling.
    error: Optional[str] = None


class ExecutionPlan(BaseModel):
    stages: list[StageOutcome]
    total_duration_ms: float
