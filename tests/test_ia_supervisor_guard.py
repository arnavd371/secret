from app.ia_supervisor.guard import detect_ghostwriting_request


def test_write_my_introduction_is_detected():
    assert detect_ghostwriting_request("can you write my introduction for me") is not None


def test_write_the_conclusion_section_is_detected():
    assert detect_ghostwriting_request("write the conclusion section please") is not None


def test_give_me_a_research_question_is_detected():
    assert detect_ghostwriting_request("give me a research question about photosynthesis") is not None


def test_do_my_ia_for_me_is_detected():
    assert detect_ghostwriting_request("do my IA for me, I am out of time") is not None


def test_finish_the_essay_for_me_is_detected():
    assert detect_ghostwriting_request("can you finish the essay for me") is not None


def test_write_it_for_me_is_detected():
    assert detect_ghostwriting_request("write it for me, I dont have time") is not None


def test_asking_for_research_question_feedback_is_not_flagged():
    text = "is this a good research question: how does temperature affect enzyme activity?"
    assert detect_ghostwriting_request(text) is None


def test_asking_for_methodology_feedback_is_not_flagged():
    assert detect_ghostwriting_request("what feedback do you have on my methodology?") is None


def test_asking_how_to_structure_analysis_is_not_flagged():
    assert detect_ghostwriting_request("how should I structure my analysis section?") is None


def test_asking_to_review_a_draft_is_not_flagged():
    assert detect_ghostwriting_request("can you review my draft introduction and tell me what to improve?") is None


def test_asking_for_topic_advice_is_not_flagged():
    assert detect_ghostwriting_request("I am stuck choosing a topic for my IA, any advice?") is None


def test_evidence_string_is_descriptive():
    evidence = detect_ghostwriting_request("write my conclusion")
    assert evidence is not None
    assert "write" in evidence.lower()
