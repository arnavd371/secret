"""
Templated, non-LLM fallback responses.

Used whenever the Tutor agent's model call fails, times out, or its draft
fails the structural leak-check. These are plain string templates — no
model call, no I/O — so they always succeed and always respect the action
contract by construction.
"""

from __future__ import annotations

from app.models.contracts import Action, ActionType, TutorResponse

_REFUSAL_TEXT = (
    "I can't help complete this one for you directly — it looks like it's "
    "part of a graded assessment or exam-style setting. I'm happy to "
    "explain the underlying concept instead, or help you practice with a "
    "similar problem."
)

_HINT_FALLBACKS = {
    0: "Here's a nudge: think about which concept or formula this problem is asking you to apply.",
    1: "Try identifying what's given and what's being asked — write those down separately before doing any algebra.",
    2: "Consider which rule or technique connects the given information to what you're solving for.",
    3: "You're close to the method — set up the first line of working using that rule, then see what simplifies.",
    4: (
        "You've worked through several hints on this one. Give it one more "
        "attempt with the approach above, and if it's still not clicking, "
        "let me know and I'll walk through the full solution with you."
    ),
}

_QUESTION_FALLBACK = "What do you think the first step should be, based on what's given in the problem?"

_EXPLAIN_FALLBACK = (
    "Let's go over the underlying concept step by step. Could you tell me "
    "which part is unclear — the setup, the method, or interpreting the "
    "result — so I can focus the explanation there?"
)

_CHALLENGE_FALLBACK = (
    "You've got a solid handle on this. Try this variation: change one of "
    "the given conditions and work out how the solution method would need "
    "to adapt."
)

_SUPPORTIVE_SCAFFOLD_FALLBACK = (
    "This one's tricky, that's completely normal. Let's slow down and just "
    "focus on the very first small step — what does the problem give you "
    "to start with?"
)


def build_refusal_response(action: Action) -> TutorResponse:
    return TutorResponse(
        text=_REFUSAL_TEXT,
        citations=[],
        ui_metadata={"action_type": ActionType.REFUSE.value, "reason": action.reason, "templated": True},
    )


def get_fallback_response(action: Action) -> TutorResponse:
    if action.action_type == ActionType.REFUSE:
        return build_refusal_response(action)

    if action.action_type == ActionType.HINT:
        level = action.level if action.level is not None else 0
        text = _HINT_FALLBACKS.get(level, _HINT_FALLBACKS[0])
    elif action.action_type == ActionType.QUESTION:
        text = _QUESTION_FALLBACK
    elif action.action_type == ActionType.EXPLAIN:
        text = _EXPLAIN_FALLBACK
    elif action.action_type == ActionType.CHALLENGE:
        text = _CHALLENGE_FALLBACK
    elif action.action_type == ActionType.SUPPORTIVE_SCAFFOLD:
        text = _SUPPORTIVE_SCAFFOLD_FALLBACK
    else:  # pragma: no cover - exhaustive over ActionType
        text = _EXPLAIN_FALLBACK

    return TutorResponse(
        text=text,
        citations=[],
        ui_metadata={"action_type": action.action_type.value, "level": action.level, "templated": True},
    )
