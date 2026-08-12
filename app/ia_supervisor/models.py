"""
Typed contracts for the IA/EE Supervisor Agent (spec §11): coaching a
student through their Internal Assessment or Extended Essay without ever
producing the submittable content itself (spec §1.5's own framing:
"never full ghostwriting").

Three real, separate concerns, matching the spec's own description of
this agent ("state machine, guard architecture, disclosure logging"):
  - IAStage / IAProjectState: which real stage of IA/EE work a given
    project is at (app/ia_supervisor/state_machine.py owns the
    transition logic).
  - the ghostwriting guard (app/ia_supervisor/guard.py) doesn't need its
    own model here — it returns a plain Optional[str] evidence string.
  - DisclosureEntry: one append-only record of an AI-assisted
    interaction on a project, real enough to render into the AI-use
    disclosure statement IB policy requires students to include with
    their submission (app/ia_supervisor/disclosure.py).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class IAStage(str, Enum):
    TOPIC_SELECTION = "topic_selection"
    RESEARCH_QUESTION = "research_question"
    METHODOLOGY = "methodology"
    ANALYSIS = "analysis"
    DRAFTING = "drafting"
    REVISION = "revision"
    # A genuine terminal state (app/ia_supervisor/state_machine.py): once
    # reached, coaching closes rather than continuing indefinitely.
    COMPLETE = "complete"


class IAProjectState(BaseModel):
    student_id: str
    project_id: str
    stage: IAStage = IAStage.TOPIC_SELECTION
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DisclosureAssistanceType(str, Enum):
    COACHING = "coaching"
    GHOSTWRITING_REQUEST_REFUSED = "ghostwriting_request_refused"
    PROJECT_ALREADY_COMPLETE = "project_already_complete"


class DisclosureEntry(BaseModel):
    student_id: str
    project_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage: IAStage
    assistance_type: DisclosureAssistanceType
    summary: str
