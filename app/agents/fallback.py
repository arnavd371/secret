"""
Templated, non-LLM fallback responses.

Used whenever the Tutor agent's model call fails, times out, or its draft
fails the structural leak-check. These are plain string templates — no
model call, no I/O — so they always succeed and always respect the action
contract by construction.
"""

from __future__ import annotations

from typing import Optional

from app.cas.models import CASResult
from app.models.contracts import Action, ActionType, TutorResponse
from app.multimodal.models import IngestionResult, IntakeRejectionReason
from app.questions.models import GeneratedItem

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


def get_fallback_response(action: Action, *, challenge_item: Optional[GeneratedItem] = None) -> TutorResponse:
    if action.action_type == ActionType.REFUSE:
        return build_refusal_response(action)

    if action.action_type == ActionType.HINT:
        level = action.level if action.level is not None else 1
        text = _HINT_FALLBACKS.get(level, _HINT_FALLBACKS[1])
    elif action.action_type == ActionType.QUESTION:
        if action.move == "retrieval_practice" and challenge_item is not None:
            text = _describe_review_item(challenge_item)
        else:
            text = _QUESTION_FALLBACKS.get(action.move or "", _QUESTION_FALLBACKS["diagnostic_probe"])
    elif action.action_type == ActionType.EXPLAIN:
        text = _EXPLAIN_FALLBACKS.get(action.move or "", _EXPLAIN_FALLBACKS["general_response"])
    elif action.action_type == ActionType.CHALLENGE:
        text = _describe_challenge_item(challenge_item) if challenge_item is not None else _CHALLENGE_FALLBACK
    elif action.action_type == ActionType.SUPPORTIVE_SCAFFOLD:
        text = _SUPPORTIVE_SCAFFOLD_FALLBACK
    else:  # pragma: no cover - exhaustive over ActionType
        text = _EXPLAIN_FALLBACKS["general_response"]

    return TutorResponse(
        text=text,
        citations=[],
        ui_metadata={"action_type": action.action_type.value, "level": action.level, "templated": True},
    )


def _describe_challenge_item(item: GeneratedItem) -> str:
    """Presents the CAS-verified generated item's stem directly — safe by
    construction, since it never states the item's own answer."""
    return (
        "You've got a solid handle on this — here's a harder one to try: "
        f"{item.rendered_stem} Give it a go, and let me know what you get."
    )


def _describe_review_item(item: GeneratedItem) -> str:
    """Same safety property as _describe_challenge_item (the item's own
    answer is never stated), different framing: this is spaced-repetition
    retrieval practice (Phase 9), not a harder extension problem."""
    return (
        f"This one's due for review: {item.rendered_stem} "
        "Try it from memory before checking your work."
    )


def build_cas_unverifiable_response(action: Action) -> TutorResponse:
    """
    Spec §1.4: "if CAS verification fails or times out, the response
    degrades to a hint-only reply and the failure is logged." Used when a
    math task was extracted for this turn but the CAS agent could not
    compute/verify a result for it (status=unverifiable).
    """
    text = (
        "I wasn't able to confidently verify this computation, so I don't want to state a specific result "
        "that might be wrong. Let's work through it together instead. What's the first step you'd try?"
    )
    return TutorResponse(
        text=text,
        citations=[],
        ui_metadata={"action_type": action.action_type.value, "cas_status": "unverifiable", "templated": True},
    )


_INTAKE_REJECTION_TEXT = {
    IntakeRejectionReason.UNSUPPORTED_FORMAT: "that file format isn't supported (please use PNG or JPEG)",
    IntakeRejectionReason.CORRUPT_IMAGE: "the image couldn't be read (it may be corrupted or empty)",
    IntakeRejectionReason.TOO_LARGE_BYTES: "that file is too large to upload",
    IntakeRejectionReason.DIMENSIONS_TOO_SMALL: "that image is too small to read reliably",
    IntakeRejectionReason.DIMENSIONS_TOO_LARGE: "that image is too large to process",
}


def build_multimodal_confirmation_response(action: Action, ingestion: IngestionResult) -> TutorResponse:
    """
    Spec §3.2's three-tier confidence gate, surfaced as a real response:
    used whenever a photo of student work couldn't be turned into
    trusted `student_work` text on its own — either intake rejected it
    outright, the vision call itself failed, or the OCR transcription's
    confidence tier came back MEDIUM or LOW. HIGH-confidence
    transcriptions never reach this function; they're graded directly.
    """
    if ingestion.rejected:
        reason_text = _INTAKE_REJECTION_TEXT.get(ingestion.rejection_reason, "I couldn't process that image")
        text = (
            f"I couldn't use that photo because {reason_text}. Could you try again with a clearer photo, "
            "or type your work instead?"
        )
    elif ingestion.student_work:
        text = (
            "Here's what I read from your photo:\n\n"
            f"{ingestion.student_work}\n\n"
            "Does that look right? Reply to confirm it, or send a corrected version, and I'll grade it."
        )
    else:
        text = (
            "I couldn't confidently read that photo. Could you retake it with better lighting or focus, "
            "or type your work instead?"
        )

    return TutorResponse(
        text=text,
        citations=[],
        ui_metadata={
            "action_type": action.action_type.value,
            "multimodal_requires_confirmation": True,
            "templated": True,
        },
    )


def build_cas_grounded_response(action: Action, cas_result: CASResult) -> TutorResponse:
    """
    Spec §1.4: "The LLM-authored narrative is discarded/regenerated if it
    disagrees with the CAS result." Rather than discard-and-retry (an
    extra model call), Phase 2 substitutes a minimal response built
    directly from the CAS ground truth — safe because EXPLAIN/CHALLENGE
    are the only action types permitted to state a final answer at all,
    and this text states nothing the CAS agent didn't itself compute.
    """
    text = (
        f"Let me verify that computation directly: for {action.move or 'this step'}, the CAS-checked result "
        f"is `{cas_result.result_exact}`. The draft explanation didn't match this, so here's the confirmed "
        "result instead of an unverified one."
    )
    return TutorResponse(
        text=text,
        citations=[],
        ui_metadata={
            "action_type": action.action_type.value,
            "cas_status": cas_result.status.value,
            "cas_grounded": True,
            "templated": True,
        },
    )
