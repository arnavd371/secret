from app.knowledge.schemas import DocType, RetrievedChunk
from app.verifier.grounding import check_grounding


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="C1",
        doc_type=DocType.FORMULA_BOOKLET_ENTRY,
        subtopic_id="calculus.differentiation.chain_rule",
        citation="Formula booklet, Calculus: Chain rule",
        text=text,
        score=0.95,
        authority_tier=1.0,
    )


def test_no_chunks_is_trivially_grounded():
    result = check_grounding("Any text at all here.", None)
    assert result.grounded is True
    assert result.score == 1.0


def test_empty_chunk_list_is_trivially_grounded():
    result = check_grounding("Any text at all here.", [])
    assert result.grounded is True


def test_related_text_is_grounded():
    chunk = _chunk("Chain rule: dy/dx = dy/du times du/dx, used to differentiate composite functions")
    result = check_grounding(
        "The chain rule lets you differentiate composite functions using dy/du and du/dx.", [chunk]
    )
    assert result.grounded is True
    assert result.score > 0.1


def test_unrelated_text_is_not_grounded():
    chunk = _chunk("Chain rule: dy/dx = dy/du times du/dx, used to differentiate composite functions")
    result = check_grounding("The capital of France is Paris and bananas are yellow.", [chunk])
    assert result.grounded is False
    assert result.ungrounded_sentences


def test_mixed_grounded_and_ungrounded_sentences():
    chunk = _chunk("Chain rule: dy/dx = dy/du times du/dx, used to differentiate composite functions")
    draft = (
        "The chain rule lets you differentiate composite functions using dy/du and du/dx. "
        "Unrelated: bananas are a good source of potassium."
    )
    result = check_grounding(draft, [chunk])
    assert len(result.ungrounded_sentences) == 1
    assert "bananas" in result.ungrounded_sentences[0].lower()


def test_empty_draft_is_trivially_grounded():
    chunk = _chunk("Some content")
    result = check_grounding("", [chunk])
    assert result.grounded is True
