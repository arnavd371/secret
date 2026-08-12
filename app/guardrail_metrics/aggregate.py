"""
Real aggregation over a batch of GuardrailTurnSignals records - the
"any deployment could feed live traffic into this" half of Phase 21.
Every rate here is computed over the denominator that actually applies
(e.g. the critic verdict distribution is over turns the critic actually
ran on, not over all turns), so a metric never gets silently diluted or
inflated by turns that couldn't have produced that signal at all.
"""

from __future__ import annotations

from collections import Counter

from app.guardrail_metrics.models import GuardrailMetricsReport, GuardrailTurnSignals


def compute_guardrail_metrics(records: list[GuardrailTurnSignals]) -> GuardrailMetricsReport:
    total = len(records)
    if total == 0:
        return GuardrailMetricsReport(
            total_turns=0,
            leak_check_trigger_rate=0.0,
            fallback_rate=0.0,
            critic_degraded_rate=0.0,
        )

    leak_check_count = sum(1 for r in records if r.leak_check_triggered)
    fallback_count = sum(1 for r in records if r.fell_back_to_template)
    critic_degraded_count = sum(1 for r in records if r.critic_degraded)

    verdict_counts = Counter(r.critic_verdict for r in records if r.critic_verdict is not None)
    critic_ran_count = sum(verdict_counts.values())
    verdict_distribution = (
        {verdict: count / critic_ran_count for verdict, count in verdict_counts.items()}
        if critic_ran_count > 0
        else {}
    )

    grounding_scores = [r.grounding_score for r in records if r.grounding_score is not None]
    average_grounding_score = sum(grounding_scores) / len(grounding_scores) if grounding_scores else None

    return GuardrailMetricsReport(
        total_turns=total,
        leak_check_trigger_rate=leak_check_count / total,
        fallback_rate=fallback_count / total,
        critic_degraded_rate=critic_degraded_count / total,
        critic_verdict_distribution=verdict_distribution,
        critic_ran_count=critic_ran_count,
        average_grounding_score=average_grounding_score,
        grounding_checked_count=len(grounding_scores),
    )
