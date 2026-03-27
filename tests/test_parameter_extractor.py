import pytest

from src.services.parameter_extractor import ParameterExtractor


@pytest.fixture
def extractor():
    return ParameterExtractor()


# ----------------------------
# can_fallback_to_llm
# ----------------------------

def test_can_fallback_add_number_false_with_vague_placeholder(extractor):
    assert extractor.can_fallback_to_llm(
        "Add 10 and something",
        "fn_add_numbers",
        "b",
        "number",
    ) is False


def test_can_fallback_add_number_false_without_function_intent(extractor):
    assert extractor.can_fallback_to_llm(
        "hello there",
        "fn_add_numbers",
        "a",
        "number",
    ) is False


def test_can_fallback_greet_name_false_for_empty_prompt(extractor):
    assert extractor.can_fallback_to_llm(
        "",
        "fn_greet",
        "name",
        "string",
    ) is False


def test_can_fallback_greet_name_false_for_greet_me(extractor):
    assert extractor.can_fallback_to_llm(
        "greet me",
        "fn_greet",
        "name",
        "string",
    ) is False


def test_can_fallback_greet_name_false_for_greet_please(extractor):
    assert extractor.can_fallback_to_llm(
        "greet please",
        "fn_greet",
        "name",
        "string",
    ) is False


# ----------------------------
# extract_parameter - valid
# ----------------------------

def test_extract_add_parameter_a_success(extractor):
    value = extractor.extract_parameter(
        "Add 2 and 3",
        "fn_add_numbers",
        "a",
        "number",
    )
    assert value == 2.0


def test_extract_add_parameter_b_success(extractor):
    value = extractor.extract_parameter(
        "Add 2 and 3",
        "fn_add_numbers",
        "b",
        "number",
    )
    assert value == 3.0


def test_extract_add_negative_and_decimal(extractor):
    value_a = extractor.extract_parameter(
        "Add -2.5 and 3",
        "fn_add_numbers",
        "a",
        "number",
    )
    value_b = extractor.extract_parameter(
        "Add -2.5 and 3",
        "fn_add_numbers",
        "b",
        "number",
    )

    assert value_a == -2.5
    assert value_b == 3.0


def test_extract_greet_name_success(extractor):
    value = extractor.extract_parameter(
        "Greet Alice",
        "fn_greet",
        "name",
        "string",
    )
    assert value == "Alice"


def test_extract_reverse_text_with_quotes_success(extractor):
    value = extractor.extract_parameter(
        'Reverse "hello world"',
        "fn_reverse_string",
        "text",
        "string",
    )
    assert value == "hello world"


def test_extract_sqrt_value_success(extractor):
    value = extractor.extract_parameter(
        "sqrt 2.25",
        "fn_square_root",
        "value",
        "number",
    )
    assert value == 2.25


# ----------------------------
# extract_parameter - incomplete / blocked
# ----------------------------

def test_extract_add_missing_b_returns_none_or_raises(extractor):
    try:
        value = extractor.extract_parameter(
            "Add 7",
            "fn_add_numbers",
            "b",
            "number",
        )
        assert value is None
    except ValueError as exc:
        assert "Missing enough information" in str(exc)


def test_extract_greet_me_returns_none_or_raises(extractor):
    try:
        value = extractor.extract_parameter(
            "greet me",
            "fn_greet",
            "name",
            "string",
        )
        assert value is None
    except ValueError:
        pass


def test_extract_greet_please_returns_none_or_raises(extractor):
    try:
        value = extractor.extract_parameter(
            "greet please",
            "fn_greet",
            "name",
            "string",
        )
        assert value is None
    except ValueError:
        pass


def test_extract_reverse_without_text_returns_none_or_raises(extractor):
    try:
        value = extractor.extract_parameter(
            "reverse",
            "fn_reverse_string",
            "text",
            "string",
        )
        assert value is None
    except ValueError:
        pass


def test_extract_sqrt_without_value_returns_none_or_raises(extractor):
    try:
        value = extractor.extract_parameter(
            "sqrt",
            "fn_square_root",
            "value",
            "number",
        )
        assert value is None
    except ValueError:
        pass


# ----------------------------
# boolean safety
# ----------------------------

def test_boolean_does_not_match_true_inside_structure(extractor):
    value = extractor.extract_parameter(
        "structure",
        "fn_set_flag",
        "enabled",
        "boolean",
    )
    assert value is None


def test_boolean_true_explicit_match(extractor):
    value = extractor.extract_parameter(
        "Set enabled to true",
        "fn_set_flag",
        "enabled",
        "boolean",
    )
    assert value is True


def test_boolean_false_explicit_match(extractor):
    value = extractor.extract_parameter(
        "Set enabled to false",
        "fn_set_flag",
        "enabled",
        "boolean",
    )
    assert value is False