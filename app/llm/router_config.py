"""
The single place in the whole codebase that names a concrete model.

Every agent calls `ModelRouter.call(capability=...)` with a capability name
(e.g. "intent_classify"). No agent, template, or orchestrator file may import
a provider SDK directly or hardcode a model string — grep for a raw model
name outside this file and it's a bug.

Phase 1 only needed two capabilities; the shape (capability -> ModelSpec)
is what every later phase's agents plug into without rework — Phase 6
adds a third, "critic_check", for the independent Verifier/Critic pass
(spec §2.2 puts this in the same fast/cheap tier as Router/Intent, since
it's a checklist-style pass, not full generation). Phase 7 adds a fourth,
"math_ocr", for the multimodal ingestion pipeline's vision call (spec
§3.2 and §14.2: no specialized math-OCR service like Mathpix is wired up,
so a general vision-capable model is the documented, acceptable fallback).
Phase 8 adds a fifth, "misconception_diagnose", for the Misconception
Diagnostician's model-inference fallback (spec §8) — same fast/cheap
tier as critic_check, since it's a classify-against-a-fixed-catalog call,
not open-ended generation. Phase 13 adds a sixth, "item_variant_author",
for LLM-authored question variants (spec §9.6) — full generation tier
like tutor_generate, since authoring a genuinely new problem is an
open-ended creative task, not a classification one; every claim it makes
is independently CAS-verified before ever being served (app/questions/
llm_variant.py), so this capability's own output is never trusted on its
own.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    MOCK = "mock"  # used by tests / offline dev; never selected in prod config


@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model: str
    max_tokens: int = 1024
    temperature: float = 0.3
    timeout_seconds: float = 8.0


# capability -> ModelSpec. This is the ONLY place model identifiers appear.
CAPABILITY_MODEL_MAP: dict[str, ModelSpec] = {
    "intent_classify": ModelSpec(
        provider=Provider.ANTHROPIC,
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        temperature=0.0,
        timeout_seconds=4.0,
    ),
    "tutor_generate": ModelSpec(
        provider=Provider.ANTHROPIC,
        model="claude-sonnet-5",
        max_tokens=1024,
        temperature=0.4,
        timeout_seconds=8.0,
    ),
    "critic_check": ModelSpec(
        provider=Provider.ANTHROPIC,
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        temperature=0.0,
        timeout_seconds=3.0,
    ),
    "math_ocr": ModelSpec(
        provider=Provider.ANTHROPIC,
        model="claude-sonnet-5",
        max_tokens=1024,
        temperature=0.0,
        timeout_seconds=12.0,
    ),
    "misconception_diagnose": ModelSpec(
        provider=Provider.ANTHROPIC,
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        temperature=0.0,
        timeout_seconds=4.0,
    ),
    "item_variant_author": ModelSpec(
        provider=Provider.ANTHROPIC,
        model="claude-sonnet-5",
        max_tokens=512,
        temperature=0.6,
        timeout_seconds=8.0,
    ),
}


def get_model_spec(capability: str) -> ModelSpec:
    try:
        return CAPABILITY_MODEL_MAP[capability]
    except KeyError as exc:
        raise ValueError(
            f"Unknown capability '{capability}'. Register it in "
            f"app/llm/router_config.CAPABILITY_MODEL_MAP first."
        ) from exc
