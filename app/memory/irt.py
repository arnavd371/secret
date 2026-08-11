"""
Item Response Theory (spec §4.3): slow-updating, cross-topic ability
estimate. 2-parameter logistic model, online single-step MAP update, ported
exactly from the spec's pseudocode.
"""

from __future__ import annotations

import math

_EPSILON = 1e-6


def probability_correct(theta: float, a: float, b: float) -> float:
    """P(correct | theta, a, b) = 1 / (1 + exp(-a*(theta - b)))"""
    return 1.0 / (1.0 + math.exp(-a * (theta - b)))


def update_irt(theta: float, se_theta: float, a: float, b: float, correct: bool) -> tuple[float, float]:
    """
    Spec §4.3:
        theta_new = theta + SE^2 * a * (correct - P)
        SE_new^2  = 1 / (1/SE^2 + a^2 * P * (1-P))
    """
    p = probability_correct(theta, a, b)
    p_clamped = min(max(p, _EPSILON), 1 - _EPSILON)
    observed = 1.0 if correct else 0.0

    theta_new = theta + (se_theta**2) * a * (observed - p)

    information = (a**2) * p_clamped * (1 - p_clamped)
    se_theta_new_sq = 1.0 / (1.0 / (se_theta**2) + information)
    se_theta_new = math.sqrt(se_theta_new_sq)

    return theta_new, se_theta_new
