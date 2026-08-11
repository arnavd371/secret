"""
Bayesian Knowledge Tracing (spec §4.3): fast-updating, per-subtopic latent
mastery. Formulas ported exactly from the spec's pseudocode.
"""

from __future__ import annotations

# Spec §4.3 defaults, calibrated offline from item-response data in the
# real system; used as flat defaults here since no such calibration data
# exists yet.
P_INIT_DEFAULT = 0.10
P_TRANSIT_DEFAULT = 0.15
P_SLIP_DEFAULT = 0.10
P_GUESS_DEFAULT = 0.20

_EPSILON = 1e-9


def update_bkt(
    p_mastery_t: float,
    correct: bool,
    p_slip: float = P_SLIP_DEFAULT,
    p_guess: float = P_GUESS_DEFAULT,
    p_transit: float = P_TRANSIT_DEFAULT,
) -> float:
    """
    Spec §4.3:
        if correct:
            p_given_evidence = (p*(1-slip)) / (p*(1-slip) + (1-p)*guess)
        else:
            p_given_evidence = (p*slip) / (p*slip + (1-p)*(1-guess))
        p_next = p_given_evidence + (1 - p_given_evidence) * transit
    """
    p = min(max(p_mastery_t, 0.0), 1.0)

    if correct:
        numerator = p * (1 - p_slip)
        denominator = numerator + (1 - p) * p_guess
    else:
        numerator = p * p_slip
        denominator = numerator + (1 - p) * (1 - p_guess)

    p_given_evidence = numerator / denominator if denominator > _EPSILON else p
    p_next = p_given_evidence + (1 - p_given_evidence) * p_transit
    return min(max(p_next, 0.0), 1.0)
