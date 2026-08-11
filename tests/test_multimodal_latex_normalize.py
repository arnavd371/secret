from app.multimodal.latex_normalize import normalize_transcription


def test_strips_markdown_code_fences():
    result = normalize_transcription("```latex\ny = x^2\n```")
    assert result.normalized_text == "y = x^2"


def test_strips_inline_and_display_math_delimiters():
    result = normalize_transcription(r"The derivative is \(2x\) which simplifies to \[2x\]")
    assert "\\(" not in result.normalized_text
    assert "\\)" not in result.normalized_text
    assert "2x" in result.normalized_text


def test_strips_dollar_delimiters():
    result = normalize_transcription("solve $x^2 - 4 = 0$ for x")
    assert "$" not in result.normalized_text


def test_resolves_dfrac_and_tfrac_aliases_to_frac():
    result = normalize_transcription(r"\dfrac{1}{2} and \tfrac{3}{4}")
    assert "\\dfrac" not in result.normalized_text
    assert "\\tfrac" not in result.normalized_text
    assert result.normalized_text.count("\\frac") == 2


def test_collapses_repeated_blank_lines_but_keeps_line_structure():
    result = normalize_transcription("step 1\n\n\n\nstep 2")
    assert result.normalized_text == "step 1\n\nstep 2"


def test_counts_latex_commands():
    result = normalize_transcription(r"\frac{1}{2} + \sin(x)")
    assert result.latex_command_count == 2


def test_empty_input_produces_empty_output():
    result = normalize_transcription("")
    assert result.normalized_text == ""
    assert result.latex_command_count == 0
