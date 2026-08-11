from app.multimodal.expression_parse import _latex_to_plain, parse_expression


def test_plain_algebraic_equation_is_parseable():
    result = parse_expression("x^2 + 3*x - 5 = 0")
    assert result.parseable is True
    assert result.parsed_expression is not None


def test_latex_fraction_is_parseable():
    result = parse_expression(r"\frac{1}{2} + x")
    assert result.parseable is True


def test_latex_sqrt_is_parseable():
    result = parse_expression(r"\sqrt{16} + 2")
    assert result.parseable is True


def test_named_function_is_parseable():
    result = parse_expression(r"\sin(x) + \cos(x)")
    assert result.parseable is True


def test_equation_with_derivative_notation_is_parseable():
    # dy/dx is itself a valid symbolic expression (two symbols divided),
    # and the right-hand side is real chain-rule differentiation output.
    result = parse_expression("dy/dx = 5*(2*x + 1)**4 * 2")
    assert result.parseable is True


def test_pure_prose_is_not_parseable():
    result = parse_expression("I think this is right but I am not sure")
    assert result.parseable is False
    assert result.parse_error is not None


def test_empty_transcription_is_not_parseable():
    result = parse_expression("")
    assert result.parseable is False


def test_nested_fraction_converts_correctly():
    plain = _latex_to_plain(r"\frac{1}{\frac{1}{x}}")
    assert plain == "((1)/(((1)/(x))))"


def test_greek_letters_and_infinity_convert():
    plain = _latex_to_plain(r"\pi + \theta + \infty")
    assert plain == "pi + theta + oo"


def test_braced_exponent_converts_to_python_power():
    plain = _latex_to_plain("x^{2}")
    assert plain == "x**(2)"
