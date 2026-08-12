"""
The orchestrator: wires Router/Intent -> Memory read -> session state ->
decision policy -> grounding (CAS+retrieval for EXPLAIN, item generation
for CHALLENGE, real grading for a check_work submission) -> Tutor agent
(or hard-gated refusal / a fully-graded response) -> session state update
-> Memory write-back.

`handle_turn` is the only place these components are composed together.
Every component it calls already owns its own fallback behavior (see
router_agent.py, tutor_agent.py, cas/solver.py, questions/generator.py,
examiner/grader.py) — the orchestrator's job is sequencing and state, not
catching stray exceptions from components that should have handled their
own failure modes already. The one thing it enforces itself is the
REFUSE hard-gate: that path returns immediately and never touches the
Tutor agent (or CAS/retrieval/generation/grading/memory) at all.

Grounding is matched to what each action type is actually permitted to
do per spec §1.5/§7.7: EXPLAIN may state a syllabus-specific claim or a
final answer, so it gets CAS verification + retrieval. CHALLENGE poses a
new problem "instead of full solve" — it never states an answer — so it
gets a real, CAS-verified extension item from the Question Generation
Engine instead. A check_work turn that includes `student_work` text is
graded for real by the Examiner (Phase 4): the orchestrator builds the
mark scheme from the same CAS ground truth Phase 2 already computes, and
returns the grounded examiner comment directly, bypassing the Tutor LLM
call entirely. Without `student_work`, check_work behaves exactly as it
did before Phase 4.

Memory (Phase 5, spec §4) is read at the start of every turn with a
resolvable topic_hint — the real, persisted mastery estimate for that
subtopic replaces the flat DEFAULT_MASTERY_ESTIMATE prior (used only when
no record exists yet), and a deterministic MemoryReadContext is injected
into the Tutor prompt's STUDENT MASTERY CONTEXT slot. A completed
check_work grading writes a real BKT/IRT update back, gated by the
grading's own confidence tier (see app.memory.write_policy) — a caller-
supplied `mastery_estimate` always overrides the memory-derived value
(useful for tests/simulation and callers who already have their own
signal), matching how the parameter already worked in earlier phases.

Planner Agent (Phase 11, spec §6): the orchestrator still runs a fixed
linear sequence for almost every stage — deliberately, because almost
every stage genuinely depends on the previous one's output (you can't
classify intent before reading raw_input, decide an action before
classifying intent, or generate before deciding an action). The one
place with real, independent side effects is the post-grading writes: a
completed check_work grading feeds Phase 5's mastery write, Phase 8's
misconception diagnosis, and Phase 9's review record, none of which
depend on each other or on one another's output. Those three now run as
a real concurrent stage graph via app.planner.executor.run_plan rather
than three sequential awaits, and the actual executed plan (per-stage
timing, dependencies, any isolated failure) is stored on
Blackboard.execution_plan for real — not a rewrite of every branch's
sequencing, which would mean inventing parallelism between stages that
are genuinely sequential.

Adaptive Learning Engine (Phase 9, spec §12): on an exam_prep turn, the
real FSRS due-review queue (app.adaptive.scheduler) is consulted before
the decision policy runs, resolving DecisionSignals.has_due_review for
real rather than leaving it at its default. When something is due, a
real practice item is generated for that exact subtopic via the same
Question Generation Engine CHALLENGE already uses, and bound to the
QUESTION action the same way an extension item is bound to CHALLENGE.
Every graded check_work submission (typed or photographed) also updates
FSRS state for its subtopic, same hook point as Phase 5's mastery write
and Phase 8's misconception write.

IA/EE Supervisor (Phase 10, spec §11): on an ia_ee_help turn, the real
ghostwriting guard (app.ia_supervisor.guard) and the real project stage
machine (app.ia_supervisor.state_machine) both run before the decision
policy, resolving DecisionSignals.ia_ghostwriting_request_detected/
ia_project_complete/ia_stage for real. `problem_id` doubles as the IA/EE
project's own id for this intent — the same parameter every other intent
already threads through for hint-ladder/session-state purposes, reused
here rather than adding a second project-identifying parameter. Every
ia_ee_help turn writes exactly one real DisclosureEntry (coaching
allowed, ghostwriting refused, or project-already-complete), regardless
of which branch the decision resolves to — the disclosure log's job is
to document what happened, not just the successful path.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.adaptive.scheduler import most_overdue_subtopic, record_review
from app.adaptive.store import ReviewStateStore, get_default_review_state_store
from app.agents import router_agent, tutor_agent
from app.agents.fallback import build_multimodal_confirmation_response, build_refusal_response
from app.cas.extraction import MathTask, extract_math_task
from app.cas.models import CASResult, CASStatus
from app.cas.solver import run_cas_operation_async
from app.diagnostician.catalog import describe
from app.diagnostician.diagnose import diagnose_misconception
from app.diagnostician.models import DiagnosisResult
from app.diagnostician.write_policy import should_write_diagnosis
from app.examiner.grader import grade_submission
from app.examiner.models import ConfidenceTier, MarkResult
from app.ia_supervisor.disclosure_store import DisclosureStore, get_default_disclosure_store
from app.ia_supervisor.guard import detect_ghostwriting_request
from app.ia_supervisor.models import DisclosureAssistanceType, DisclosureEntry, IAProjectState, IAStage
from app.ia_supervisor.project_store import IAProjectStateStore, get_default_ia_project_store
from app.ia_supervisor.stage_classifier import classify_stage
from app.ia_supervisor.state_machine import advance_stage
from app.knowledge.retriever import KnowledgeBase, get_default_knowledge_base
from app.knowledge.schemas import RetrievedChunk
from app.llm.client import ModelRouter
from app.planner.executor import StageGraph, run_plan
from app.memory.bkt import update_bkt
from app.memory.context_assembly import assemble_memory_context
from app.memory.decay import effective_mastery
from app.memory.irt import update_irt
from app.memory.models import MemoryReadContext, MisconceptionRegistryEntry, SubtopicMastery
from app.memory.store import MemoryStore, get_default_memory_store
from app.memory.write_policy import should_write_mastery_update
from app.models.contracts import ActionType, Blackboard, DecisionSignals, IntentType, RawInput, TutorResponse
from app.multimodal.pipeline import ingest_image
from app.orchestrator.signals import DEFAULT_MASTERY_ESTIMATE, estimate_frustration, estimate_safety_result
from app.policy.decision import decide_pedagogical_action
from app.questions.generator import (
    ItemGenerationError,
    generate_item_async,
    select_template_for_topic,
    topic_has_known_template,
)
from app.questions.llm_variant import generate_llm_variant
from app.questions.mark_scheme import build_mark_scheme
from app.questions.response_log import ItemResponseRecord, ResponseLogStore, get_default_response_log_store
from app.review_queue.models import ReviewReason
from app.review_queue.queue import enqueue_review
from app.review_queue.store import ReviewQueueStore, get_default_review_queue_store
from app.questions.models import GeneratedItem
from app.session.state import InMemorySessionStateStore, ProblemSessionState, SessionStateStore, advance_session_state

logger = logging.getLogger(__name__)

# Generic IRT item-difficulty defaults used for check_work-driven mastery
# updates. Real per-item (a, b) parameters (spec §9.7) exist only for
# Question Generation Engine items with a known template; an arbitrary
# CAS-extracted check_work problem has no calibrated difficulty, so a
# neutral discrimination/difficulty pair is used rather than fabricating
# a specific one. TODO(later phase): thread real per-item IRT parameters
# through once check_work problems are linked back to a specific item.
_GENERIC_IRT_DISCRIMINATION = 1.0
_GENERIC_IRT_DIFFICULTY = 0.0


async def _ground_explain_turn(
    raw_input: str, topic_hint: Optional[str], knowledge_base: KnowledgeBase
) -> tuple[Optional[CASResult], list[RetrievedChunk]]:
    cas_result: Optional[CASResult] = None
    math_task = extract_math_task(raw_input)
    if math_task is not None:
        cas_result = await run_cas_operation_async(
            math_task.operation, math_task.expression, math_task.variable, math_task.at, upper_at=math_task.upper_at
        )

    retrieved_chunks = knowledge_base.retrieve(raw_input, topic_hint=topic_hint)
    return cas_result, retrieved_chunks


async def _generate_challenge_item(
    topic_hint: Optional[str],
    router: Optional[ModelRouter] = None,
    response_log_store: Optional[ResponseLogStore] = None,
) -> Optional[GeneratedItem]:
    # Phase 13 (spec §9.6): only tried when there's genuinely no good
    # parametric template for this topic — not as a redundant, costlier
    # alternative to a template that already works. Falls through to the
    # generic default template below if the LLM can't produce a
    # CAS-verified item either.
    if router is not None and not topic_has_known_template(topic_hint):
        llm_item = await generate_llm_variant(topic_hint, router)
        if llm_item is not None:
            return llm_item

    template_id = select_template_for_topic(topic_hint)
    try:
        # Phase 14 (spec §9.7): a real recalibrated difficulty is used
        # in place of the template's hand-set prior once enough response
        # history exists for this exact template.
        return await generate_item_async(template_id, response_log_store=response_log_store)
    except ItemGenerationError as exc:
        logger.warning("Question generation failed for template %s (%s); Tutor falls back to a generic extension prompt", template_id, exc)
        return None


async def _grade_check_work_turn(
    turn_id: str, raw_input: str, student_work: str
) -> Optional[tuple[MarkResult, MathTask, CASResult]]:
    """Returns (MarkResult, MathTask, CASResult) if the problem in
    `raw_input` was extractable and CAS-verifiable, else None — meaning
    the caller should fall back to the normal Tutor-generated EXPLAIN
    path instead of grading blind. The MathTask/CASResult are returned
    alongside the grade (not just used internally) because the
    Misconception Diagnostician (Phase 8) needs the same CAS ground
    truth to generate its wrong-answer hypotheses against."""
    math_task = extract_math_task(raw_input)
    if math_task is None:
        return None

    cas_result = await run_cas_operation_async(
        math_task.operation, math_task.expression, math_task.variable, math_task.at, upper_at=math_task.upper_at
    )
    if cas_result.status != CASStatus.OK:
        return None

    mark_scheme = build_mark_scheme(f"check-{turn_id}", cas_result)
    mark_result = grade_submission(f"check-{turn_id}", mark_scheme, student_work, given_expression=math_task.expression)
    return mark_result, math_task, cas_result


async def _diagnose_and_write_misconception(
    router: ModelRouter, student_id: str, math_task: MathTask, cas_result: CASResult, student_work: str, memory_store: MemoryStore
) -> DiagnosisResult:
    diagnosis = await diagnose_misconception(router, math_task, cas_result, student_work)
    if not should_write_diagnosis(diagnosis):
        return diagnosis

    now = datetime.now(timezone.utc)
    existing_entries = await memory_store.get_misconceptions(student_id)
    existing = next((e for e in existing_entries if e.misconception_id == diagnosis.misconception_id), None)
    if existing is not None:
        existing.occurrences += 1
        existing.last_observed_at = now
        # A repeat diagnosis reinforces rather than overwrites: never
        # let a fresh observation's strength read as weaker than what
        # was already on record.
        existing.decayed_strength = max(existing.decayed_strength, diagnosis.confidence)
        entry = existing
    else:
        entry = MisconceptionRegistryEntry(
            student_id=student_id,
            misconception_id=diagnosis.misconception_id,
            occurrences=1,
            first_observed_at=now,
            last_observed_at=now,
            decayed_strength=diagnosis.confidence,
        )
    await memory_store.save_misconception(entry)
    return diagnosis


async def _load_memory(
    student_id: str, topic_hint: Optional[str], memory_store: MemoryStore
) -> tuple[Optional[SubtopicMastery], MemoryReadContext]:
    if not topic_hint:
        return None, MemoryReadContext(rendered_text="(no topic identified for this turn)")
    mastery = await memory_store.get_mastery(student_id, topic_hint)
    misconceptions = await memory_store.get_misconceptions(student_id)
    relevant_misconceptions = [m for m in misconceptions]  # registry is small; no per-subtopic filter modeled yet
    context = assemble_memory_context(mastery, relevant_misconceptions)
    return mastery, context


async def _write_mastery_from_grading(
    student_id: str, topic_hint: Optional[str], mark_result: MarkResult, memory_store: MemoryStore
) -> None:
    if not topic_hint or not should_write_mastery_update(mark_result.confidence):
        return

    now = datetime.now(timezone.utc)
    existing = await memory_store.get_mastery(student_id, topic_hint)
    mastery = existing or SubtopicMastery(student_id=student_id, subtopic_id=topic_hint)

    correct = mark_result.total_available > 0 and mark_result.total_awarded == mark_result.total_available
    mastery.p_mastery_bkt = update_bkt(mastery.p_mastery_bkt, correct)
    mastery.theta_irt, mastery.se_theta = update_irt(
        mastery.theta_irt, mastery.se_theta, _GENERIC_IRT_DISCRIMINATION, _GENERIC_IRT_DIFFICULTY, correct
    )
    mastery.attempts_total += 1
    if correct:
        mastery.attempts_correct += 1
    mastery.last_practiced_at = now
    mastery.updated_at = now

    await memory_store.save_mastery(mastery)


async def handle_turn(
    raw_input: str,
    session_id: str,
    student_id: str,
    problem_id: Optional[str] = None,
    *,
    router: Optional[ModelRouter] = None,
    session_store: Optional[SessionStateStore] = None,
    knowledge_base: Optional[KnowledgeBase] = None,
    memory_store: Optional[MemoryStore] = None,
    review_store: Optional[ReviewStateStore] = None,
    ia_project_store: Optional[IAProjectStateStore] = None,
    ia_disclosure_store: Optional[DisclosureStore] = None,
    response_log_store: Optional[ResponseLogStore] = None,
    review_queue_store: Optional[ReviewQueueStore] = None,
    mastery_estimate: Optional[float] = None,
    student_work: Optional[str] = None,
    student_work_image: Optional[bytes] = None,
    responding_to_template_id: Optional[str] = None,
) -> Blackboard:
    """
    Returns the fully populated Blackboard for the turn (not just the bare
    TutorResponse) so callers/tests can assert on the intermediate
    IntentResult/DecisionSignals/Action as well as the final response —
    the response text itself is at `blackboard.final_response.text`.

    `student_work` is optional, separate free text (e.g. the student's
    typed working) graded against the problem extracted from `raw_input`
    when the turn's intent is check_work. Omit it and check_work behaves
    exactly as in earlier phases.

    `student_work_image` (Phase 7, spec §3.2) is an optional photo of the
    student's work, used only when `student_work` text wasn't already
    supplied directly. It's run through the real multimodal ingestion
    pipeline (app/multimodal/pipeline.py): a HIGH-confidence transcription
    is graded exactly as if the student had typed it; anything else
    (intake rejection, an OCR outage, or a MEDIUM/LOW-confidence
    transcription) short-circuits with a templated response asking the
    student to confirm what was read, retype, or retake the photo — the
    Tutor LLM is never asked to grade an unconfirmed transcription.

    `mastery_estimate`, if given explicitly, overrides the real
    memory-derived mastery for this turn (useful for tests/simulation, or
    a caller that already has its own signal). Left as None (the
    default), the real persisted mastery for the turn's topic is used.

    For an ia_ee_help turn, `problem_id` is reused as the IA/EE project's
    own id (see the module docstring) — pass a stable per-project value
    across turns so stage tracking and the disclosure log stay scoped to
    the right project.

    `responding_to_template_id` (Phase 14, spec §9.7): pass the
    `template_id` of a previously served GeneratedItem when this turn's
    check_work submission is the student's response to it, so the
    grading outcome gets logged as a real response record
    (app/questions/response_log.py) feeding future recalibration for
    that template. Omit it (the default) for check_work submissions on
    an ad-hoc, non-generated problem — nothing is logged in that case.

    Human-in-the-loop review (Phase 17, spec §10.10): a real entry is
    queued (app/review_queue/) whenever this turn's grading came back
    LOW confidence or flagged, or the Verifier/Critic degraded to its
    static fallback — signals every earlier phase already computes for
    real, just not previously surfaced anywhere a human could act on.
    """
    router = router or ModelRouter()
    session_store = session_store or InMemorySessionStateStore()
    knowledge_base = knowledge_base or get_default_knowledge_base()
    memory_store = memory_store or get_default_memory_store()
    review_store = review_store or get_default_review_state_store()
    ia_project_store = ia_project_store or get_default_ia_project_store()
    ia_disclosure_store = ia_disclosure_store or get_default_disclosure_store()
    response_log_store = response_log_store or get_default_response_log_store()
    review_queue_store = review_queue_store or get_default_review_queue_store()

    blackboard = Blackboard(
        turn_id=str(uuid.uuid4()),
        session_id=session_id,
        student_id=student_id,
        raw_input=RawInput(text=raw_input),
    )

    intent_result = await router_agent.classify_intent(raw_input, router)
    blackboard.intent_result = intent_result

    safety_result = estimate_safety_result(raw_input, intent_result.assessment_mode_guess)
    blackboard.safety_result = safety_result

    mastery_record, memory_context = await _load_memory(student_id, intent_result.topic_hint, memory_store)
    blackboard.student_state_snapshot = memory_context

    if mastery_estimate is not None:
        resolved_mastery_estimate = mastery_estimate
    elif mastery_record is not None:
        resolved_mastery_estimate = effective_mastery(mastery_record.p_mastery_bkt, mastery_record.last_practiced_at)
    else:
        resolved_mastery_estimate = DEFAULT_MASTERY_ESTIMATE

    prev_state = await session_store.get(session_id, problem_id)
    problem_changed = problem_id is not None and problem_id != prev_state.problem_id
    current_state = (
        ProblemSessionState(session_id=session_id, problem_id=problem_id) if problem_changed else prev_state
    )

    # Adaptive Learning Engine (Phase 9, spec §12): resolved here, before
    # the pure decision policy runs, exactly like mastery_estimate above —
    # decide_pedagogical_action never touches a store directly.
    due_subtopic_id: Optional[str] = None
    if intent_result.intent == IntentType.EXAM_PREP:
        due_subtopic_id = await most_overdue_subtopic(review_store, student_id)

    # IA/EE Supervisor (Phase 10, spec §11): same convention — the real
    # guard and state-machine transition both run here, before the
    # policy, so decide_pedagogical_action only ever sees pre-resolved
    # booleans/enums.
    ia_project_id: Optional[str] = None
    ia_ghostwriting_evidence: Optional[str] = None
    ia_new_stage: Optional[IAStage] = None
    if intent_result.intent == IntentType.IA_EE_HELP:
        ia_project_id = problem_id or f"default-ia-project:{student_id}"
        ia_ghostwriting_evidence = detect_ghostwriting_request(raw_input)
        existing_ia_project = await ia_project_store.get(student_id, ia_project_id)
        classified_stage = classify_stage(raw_input)
        ia_new_stage = advance_stage(existing_ia_project.stage if existing_ia_project else None, classified_stage)
        await ia_project_store.save(IAProjectState(student_id=student_id, project_id=ia_project_id, stage=ia_new_stage))

    signals = DecisionSignals(
        intent=intent_result.intent,
        mastery_estimate=resolved_mastery_estimate,
        assessment_mode=intent_result.assessment_mode_guess,
        integrity_risk=safety_result.integrity_risk,
        attempt_count=current_state.attempt_count,
        frustration_signal=estimate_frustration(raw_input),
        hint_ladder_level=current_state.hint_ladder_level,
        has_due_review=due_subtopic_id is not None,
        ia_ghostwriting_request_detected=ia_ghostwriting_evidence is not None,
        ia_project_complete=ia_new_stage == IAStage.COMPLETE,
        ia_stage=ia_new_stage,
    )

    action = decide_pedagogical_action(signals)
    blackboard.decision_action = action

    # Disclosure logging (Phase 10, spec §11): exactly one real entry per
    # ia_ee_help turn, regardless of which branch the decision resolved
    # to — the log documents what happened, including a refusal.
    if intent_result.intent == IntentType.IA_EE_HELP:
        if action.reason == "ia_ghostwriting_guard_tripped":
            assistance_type = DisclosureAssistanceType.GHOSTWRITING_REQUEST_REFUSED
            summary = f"Request declined (matched pattern: {ia_ghostwriting_evidence})."
        elif action.reason == "ia_project_already_complete":
            assistance_type = DisclosureAssistanceType.PROJECT_ALREADY_COMPLETE
            summary = "Project already marked complete; no further assistance given."
        else:
            assistance_type = DisclosureAssistanceType.COACHING
            summary = f"Coaching feedback provided for the {ia_new_stage.value} stage."

        disclosure_entry = DisclosureEntry(
            student_id=student_id,
            project_id=ia_project_id,
            stage=ia_new_stage,
            assistance_type=assistance_type,
            summary=summary,
        )
        await ia_disclosure_store.add(disclosure_entry)
        blackboard.ia_disclosure_entry = disclosure_entry

    # Hard gate: REFUSE short-circuits here. The Tutor agent (and CAS/
    # retrieval/generation/grading) is never invoked for a REFUSE action,
    # by construction.
    if action.action_type == ActionType.REFUSE:
        response = build_refusal_response(action)
        blackboard.final_response = response
        return blackboard

    # A photographed submission is turned into text before grading can even
    # be attempted. Only runs when the caller didn't already supply typed
    # student_work directly (typed text always wins over re-OCRing a photo).
    if intent_result.intent == IntentType.CHECK_WORK and student_work_image is not None and not student_work:
        ingestion = await ingest_image(router, student_work_image)
        blackboard.ingestion_result = ingestion
        if ingestion.rejected or ingestion.requires_confirmation or not ingestion.student_work:
            response = build_multimodal_confirmation_response(action, ingestion)
            blackboard.final_response = response
            new_state = advance_session_state(current_state, problem_id, intent_result.intent, action)
            await session_store.save(new_state)
            return blackboard
        student_work = ingestion.student_work

    # Real grading path: a check_work turn with student_work attached is
    # graded directly, bypassing the Tutor LLM call entirely. Falls
    # through to the normal Tutor-generated path if the problem wasn't
    # extractable/verifiable — grading blind isn't an option.
    if intent_result.intent == IntentType.CHECK_WORK and student_work:
        grading = await _grade_check_work_turn(blackboard.turn_id, raw_input, student_work)
        if grading is not None:
            mark_result, math_task, cas_result = grading
            blackboard.mark_result = mark_result

            # Planner Agent (Phase 11, spec §6): the mastery write
            # (Phase 5), review record (Phase 9), and diagnosis (Phase 8)
            # all key only off mark_result — none depends on either of
            # the others — so they run as a real concurrent stage graph
            # instead of three sequential awaits.
            correct = mark_result.total_available > 0 and mark_result.total_awarded == mark_result.total_available
            diagnosis_needed = mark_result.total_available > 0 and mark_result.total_awarded < mark_result.total_available

            post_grading_stages: StageGraph = {
                "mastery_write": (
                    lambda: _write_mastery_from_grading(student_id, intent_result.topic_hint, mark_result, memory_store),
                    [],
                ),
            }
            if intent_result.topic_hint:
                # Every graded attempt is a real spaced-repetition review,
                # regardless of the grading's own confidence tier —
                # unlike the mastery write above, this isn't gated on
                # confidence, since the review itself genuinely happened
                # even when extracting a clean mark from it didn't go
                # perfectly.
                post_grading_stages["review_record"] = (
                    lambda: record_review(review_store, student_id, intent_result.topic_hint, correct),
                    [],
                )
            if diagnosis_needed:
                # Only worth running when marks were actually missed — a
                # fully correct submission has nothing to diagnose.
                post_grading_stages["diagnosis"] = (
                    lambda: _diagnose_and_write_misconception(
                        router, student_id, math_task, cas_result, student_work, memory_store
                    ),
                    [],
                )
            if mark_result.confidence == ConfidenceTier.LOW or mark_result.flags:
                # Phase 17 (spec §10.10): a real grading-confidence signal
                # every earlier phase already computes, surfaced to a
                # human review queue for the first time here.
                review_reason = (
                    ReviewReason.UNSUPPORTED_ANSWER_FLAG
                    if "unsupported_correct_answer" in mark_result.flags
                    else ReviewReason.LOW_CONFIDENCE_GRADING
                )
                review_summary = (
                    f"{mark_result.total_awarded}/{mark_result.total_available} marks, "
                    f"confidence={mark_result.confidence.value}, flags={mark_result.flags}"
                )
                post_grading_stages["review_queue_write"] = (
                    lambda: enqueue_review(review_queue_store, blackboard.turn_id, student_id, review_reason, review_summary),
                    [],
                )
            if responding_to_template_id is not None:
                # Phase 14 (spec §9.7): a real response record feeding
                # future IRT recalibration for this exact template.
                # Independent of the other three writes above, so it
                # joins the same concurrent stage batch rather than
                # being a fourth sequential await.
                post_grading_stages["response_log_write"] = (
                    lambda: response_log_store.add(
                        ItemResponseRecord(
                            template_id=responding_to_template_id, student_id=student_id, correct=correct
                        )
                    ),
                    [],
                )

            post_grading_results, execution_plan = await run_plan(post_grading_stages)
            blackboard.execution_plan = execution_plan.model_dump(mode="json")

            response_text = mark_result.comment
            diagnosis = post_grading_results.get("diagnosis")
            if diagnosis is not None:
                blackboard.diagnosis_result = diagnosis
                if should_write_diagnosis(diagnosis):
                    response_text += f" This looks like a specific, recognizable error: {describe(diagnosis.misconception_id)}"

            response = TutorResponse(
                text=response_text,
                citations=[],
                ui_metadata={
                    "action_type": action.action_type.value,
                    "graded": True,
                    "total_awarded": mark_result.total_awarded,
                    "total_available": mark_result.total_available,
                    "confidence": mark_result.confidence.value,
                    "templated": True,
                },
            )
            blackboard.final_response = response
            new_state = advance_session_state(current_state, problem_id, intent_result.intent, action)
            await session_store.save(new_state)
            return blackboard

    cas_result: Optional[CASResult] = None
    retrieved_chunks: list[RetrievedChunk] = []
    challenge_item: Optional[GeneratedItem] = None

    if action.action_type == ActionType.EXPLAIN:
        cas_result, retrieved_chunks = await _ground_explain_turn(raw_input, intent_result.topic_hint, knowledge_base)
        blackboard.cas_result = cas_result
        blackboard.retrieved_chunks = retrieved_chunks
    elif action.action_type == ActionType.CHALLENGE:
        challenge_item = await _generate_challenge_item(intent_result.topic_hint, router, response_log_store)
        blackboard.generated_item = challenge_item
    elif action.action_type == ActionType.QUESTION and action.move == "retrieval_practice" and due_subtopic_id is not None:
        # Phase 9: a real spaced-repetition item for the actual due
        # subtopic, reusing Phase 3's CAS-verified generator rather than
        # inventing a separate item source for review practice.
        challenge_item = await _generate_challenge_item(due_subtopic_id, router, response_log_store)
        blackboard.generated_item = challenge_item

    response = await tutor_agent.generate(
        action,
        raw_input,
        router,
        cas_result=cas_result,
        retrieved_chunks=retrieved_chunks,
        challenge_item=challenge_item,
        memory_context=memory_context,
    )
    blackboard.draft_response = response
    blackboard.final_response = response

    if response.ui_metadata.get("critic_degraded"):
        # Phase 17 (spec §10.10): the Verifier/Critic fell back to its
        # conservative static check (Phase 6) rather than a real model
        # critique — a real signal that this turn's response wasn't
        # independently reviewed the way it normally would be.
        await enqueue_review(
            review_queue_store,
            blackboard.turn_id,
            student_id,
            ReviewReason.CRITIC_DEGRADED,
            f"Verifier/Critic degraded to static fallback for action_type={action.action_type.value}",
        )

    new_state = advance_session_state(current_state, problem_id, intent_result.intent, action)
    await session_store.save(new_state)

    return blackboard
