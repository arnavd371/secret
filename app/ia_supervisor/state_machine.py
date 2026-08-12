"""
Real state-machine transition logic (spec §11) for an IA/EE project's
stage. Deliberately permissive about lateral/backward movement — real
IA/EE work is iterative, not a strict pipeline (a student legitimately
revisits methodology after starting analysis, or drafts an introduction
before finalizing a research question) — but COMPLETE is a genuine
terminal state: once reached, no further transition is possible, matching
IAStage.COMPLETE's own contract in app.ia_supervisor.models. This is the
one place "state machine" earns its name rather than being a label on a
single mutable field: the terminal-state rule is enforced here, not left
to callers to remember.

Pure function, no I/O, unit-tested exactly like app.policy.decision and
app.memory.node_state.
"""

from __future__ import annotations

from typing import Optional

from app.ia_supervisor.models import IAStage


def advance_stage(current: Optional[IAStage], classified: Optional[IAStage]) -> IAStage:
    if current == IAStage.COMPLETE:
        return IAStage.COMPLETE

    if current is None:
        return classified or IAStage.TOPIC_SELECTION

    return classified or current
