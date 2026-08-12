from app.ia_supervisor.models import IAStage
from app.ia_supervisor.stage_classifier import classify_stage
from app.ia_supervisor.state_machine import advance_stage


def test_classify_topic_selection():
    assert classify_stage("I am stuck choosing a topic for my IA") == IAStage.TOPIC_SELECTION


def test_classify_research_question():
    assert classify_stage("is this a good research question?") == IAStage.RESEARCH_QUESTION


def test_classify_methodology():
    assert classify_stage("what feedback do you have on my methodology?") == IAStage.METHODOLOGY


def test_classify_analysis():
    assert classify_stage("how should I structure my analysis section?") == IAStage.ANALYSIS


def test_classify_drafting():
    assert classify_stage("can you look at my introduction draft?") == IAStage.DRAFTING


def test_classify_revision_takes_priority_over_drafting_vocabulary():
    # Contains "draft" but is clearly a revision request.
    assert classify_stage("can you review my draft and give feedback?") == IAStage.REVISION


def test_classify_complete_requires_genuine_submission_language():
    assert classify_stage("I already submitted my IA") == IAStage.COMPLETE


def test_finishing_a_draft_is_not_classified_as_complete():
    # "finished" alone (without submission language) must not prematurely
    # close coaching — it means the draft is done, not the project.
    assert classify_stage("I finished my draft, can you check it?") != IAStage.COMPLETE


def test_unrecognized_text_classifies_as_none():
    assert classify_stage("thanks, that's helpful") is None


def test_advance_stage_starts_at_classified_stage_for_a_new_project():
    assert advance_stage(None, IAStage.METHODOLOGY) == IAStage.METHODOLOGY


def test_advance_stage_defaults_to_topic_selection_for_a_brand_new_unclassified_turn():
    assert advance_stage(None, None) == IAStage.TOPIC_SELECTION


def test_advance_stage_keeps_current_stage_when_nothing_classified():
    assert advance_stage(IAStage.METHODOLOGY, None) == IAStage.METHODOLOGY


def test_advance_stage_moves_forward():
    assert advance_stage(IAStage.TOPIC_SELECTION, IAStage.METHODOLOGY) == IAStage.METHODOLOGY


def test_advance_stage_allows_lateral_backward_movement():
    # Real IA/EE work is iterative - moving back to methodology after
    # starting analysis is legitimate, not an error.
    assert advance_stage(IAStage.ANALYSIS, IAStage.METHODOLOGY) == IAStage.METHODOLOGY


def test_complete_is_a_genuine_terminal_state():
    assert advance_stage(IAStage.COMPLETE, IAStage.TOPIC_SELECTION) == IAStage.COMPLETE
    assert advance_stage(IAStage.COMPLETE, None) == IAStage.COMPLETE
