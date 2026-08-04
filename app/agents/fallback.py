"""
Templated, non-LLM fallback responses.

Used whenever the Tutor agent's model call fails, times out, or its draft
fails the structural leak-check. These are plain string templates — no
model call, no I/O — so they always succeed and always respect the action
contract by construction.
"""

from __future__ import annotations

from app.models.contracts import Action, ActionType, TutorResponse

_OFFER_TEXT = {
    "concept_explanation": "explain the underlying concept instead",
    "strategy_coaching_only": "help with timing and exam strategy, but not the content itself",
    "analog_practice_problem": "help you practice with a similar problem",
    "ia_methodology_coaching": "help with your research question, methodology, or structure — not write it for you",
}


def _describe_offer(offer) -> str:  # noqa: ANN001 - Action.offer is str | list[str] | None
    if offer is None:
        return "explain the underlying concept instead"
    if isinstance(offer, str):
        return _OFFER_TEXT.get(offer, offer)
    return " or ".join(_OFFER_TEXT.get(o, o) for o in offer)


def build_refusal_response(action: Action) -> TutorResponse:
    text = (
        "I can't help complete this one for you directly given the context here. "
        f"I'm happy to {_describe_offer(action.offer)}."
    )
    return TutorResponse(
        text=text,
        citations=[],
        ui_metadata={"action_type": ActionType.REFUSE.value, "reason": action.reason, "templated": True},
    )


_HINT_FALLBACKS = {
    1: "Here's a nudge: restate what the problem is actually asking for, and which concept category it falls under — no method yet.",
    2: "Think about which specific technique or theorem applies here. Don't apply it yet — just name it to yourself first.",
    3: "Try applying that technique to the first step of the actual problem, and see what the expression simplifies to.",
    4: (
        "You've worked through several hints on this one. Here's a fully worked example of the same *type* of "
        "problem with different numbers — walk through it, then try your original problem once more with that "
        "method in hand."
    ),
}

_QUESTION_FALLBACKS = {
    "diagnostic_probe": "What do you think the first step should be, based on what's given in the problem?",
    "retrieval_practice": "Before we move on, can you recall the key rule or formula this topic depends on?",
}

_EXPLAIN_FALLBACKS = {
    "error_localization_explanation": (
        "Let's find exactly where this went off track. Walk me through your steps one at a time, and I'll "
        "point out where the reasoning starts to diverge."
    ),
    "direct_explanation": (
        "Let's go over the underlying concept step by step. Could you tell me which part is unclear — the "
        "setup, the method, or interpreting the result — so I can focus the explanation there?"
    ),
    "general_response": (
        "I want to make sure I help with the right thing — could you tell me a bit more about what you're "
        "working on?"
    ),
}

_CHALLENGE_FALLBACK = (
    "You've got a solid handle on this. Try this variation: change one of the given conditions and work out "
    "how the solution method would need to adapt."
)

_SUPPORTIVE_SCAFFOLD_FALLBACK = (
    "This one's tricky, that's completely normal. Let's slow down and just focus on the very first small "
    "step — what does the problem give you to start with?"
)


def get_fallback_response(action: Action) -> TutorResponse:
    if action.action_type == ActionType.REFUSE:
        return build_refusal_response(action)

    if action.action_type == ActionType.HINT:
        level = action.level if action.level is not None else 1
        text = _HINT_FALLBACKS.get(level, _HINT_FALLBACKS[1])
    elif action.action_type == ActionType.QUESTION:
        text = _QUESTION_FALLBACKS.get(action.move or "", _QUESTION_FALLBACKS["diagnostic_probe"])
    elif action.action_type == ActionType.EXPLAIN:
        text = _EXPLAIN_FALLBACKS.get(action.move or "", _EXPLAIN_FALLBACKS["general_response"])
    elif action.action_type == ActionType.CHALLENGE:
        text = _CHALLENGE_FALLBACK
    elif action.action_type == ActionType.SUPPORTIVE_SCAFFOLD:
        text = _SUPPORTIVE_SCAFFOLD_FALLBACK
    else:  # pragma: no cover - exhaustive over ActionType
        text = _EXPLAIN_FALLBACKS["general_response"]

    return TutorResponse(
        text=text,
        citations=[],
        ui_metadata={"action_type": action.action_type.value, "level": action.level, "templated": True},
    )
