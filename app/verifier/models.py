"""
Typed contracts for the Verifier / Critic Agent (spec §2.2, §13.5).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CritiqueVerdict(str, Enum):
    PASS = "pass"
    REVISE = "revise"
    BLOCK = "block"


class CritiqueResult(BaseModel):
    verdict: CritiqueVerdict
    violations: list[str] = Field(default_factory=list)
    # True when the real critic model call failed/timed out and this
    # result came from the conservative static-checks-only fallback
    # instead (spec §2.2: "On critic timeout, apply conservative static
    # checks only... and pass with a critic_degraded=true flag").
    critic_degraded: bool = False
