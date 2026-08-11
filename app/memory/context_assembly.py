"""
Budgeted context assembly (spec §4.12): "This assembly is deterministic
(no LLM call) — a scoring/sorting function over the LearnerModel fields...
producing a fixed-shape MemoryReadContext object injected into the Tutor/
Grader system prompt as a structured block."

Priority order, adapted to what this phase actually persists (no
prerequisite-skill graph traversal or session-summary compression exist
yet — both later-phase non-goals):
  1. Current subtopic mastery + node_state (always included)
  2. Active, non-remediated misconceptions for the subtopic, sorted by
     decayed strength, capped at 3 (spec §4.12's own cap)
A word-count budget stands in for the spec's token budget — close enough
for a short structured block, and avoids adding a tokenizer dependency
just to enforce ~800 tokens.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.memory.decay import decayed_misconception_strength, effective_mastery
from app.memory.models import MemoryReadContext, MisconceptionRegistryEntry, SubtopicMastery
from app.memory.node_state import compute_node_state

MAX_ACTIVE_MISCONCEPTIONS = 3
# A misconception decayed below this strength is treated as no longer
# active enough to surface in the prompt, even if never formally
# remediated.
MISCONCEPTION_ACTIVE_THRESHOLD = 0.1
DEFAULT_WORD_BUDGET = 200

_NO_MASTERY_HISTORY = "(no mastery history for this subtopic yet)"


def assemble_memory_context(
    mastery: Optional[SubtopicMastery],
    misconceptions: list[MisconceptionRegistryEntry],
    now: Optional[datetime] = None,
    max_misconceptions: int = MAX_ACTIVE_MISCONCEPTIONS,
    word_budget: int = DEFAULT_WORD_BUDGET,
) -> MemoryReadContext:
    now = now or datetime.now(timezone.utc)

    if mastery is None:
        return MemoryReadContext(rendered_text=_NO_MASTERY_HISTORY)

    node_state = compute_node_state(mastery, now)
    eff_mastery = effective_mastery(mastery.p_mastery_bkt, mastery.last_practiced_at, now)

    active = [
        (entry, decayed_misconception_strength(entry.decayed_strength, entry.last_observed_at, now))
        for entry in misconceptions
        if entry.remediated_at is None
    ]
    active = [(entry, strength) for entry, strength in active if strength > MISCONCEPTION_ACTIVE_THRESHOLD]
    active.sort(key=lambda pair: pair[1], reverse=True)
    top_misconceptions = [entry.misconception_id for entry, _ in active[:max_misconceptions]]

    lines = [
        f"Subtopic {mastery.subtopic_id}: mastery={eff_mastery:.2f} ({node_state.value}), "
        f"{mastery.attempts_total} attempts ({mastery.attempts_correct} correct)."
    ]
    if top_misconceptions:
        lines.append("Active misconceptions: " + ", ".join(top_misconceptions) + ".")

    rendered_text = " ".join(lines)
    words = rendered_text.split()
    if len(words) > word_budget:
        rendered_text = " ".join(words[:word_budget]) + "..."

    return MemoryReadContext(
        subtopic_id=mastery.subtopic_id,
        node_state=node_state,
        effective_mastery=eff_mastery,
        active_misconception_ids=top_misconceptions,
        rendered_text=rendered_text,
    )
