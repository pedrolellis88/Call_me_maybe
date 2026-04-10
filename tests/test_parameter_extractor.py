import pytest

from src.services.parameter_extractor import ParameterExtractor


@pytest.fixture
def extractor() -> ParameterExtractor:
    """Create a parameter extractor for tests."""
    return ParameterExtractor()


def test_can_fallback_add_second_number_false_when_only_one_number_present(
    extractor: ParameterExtractor,
) -> None:
    """Reject fallback when the second add argument is missing."""
    assert extractor.can_fallback_to_llm(
        "What is the sum of 7?",
        "fn_add_numbers",
        "b",
        "number",
    ) is False


def test_can_fallback_greet_name_false_for_empty_prompt(
    extractor: ParameterExtractor,
) -> None:
    """Reject fallback for an empty greeting prompt."""
    assert extractor.can_fallback_to_llm(
        "",
        "fn_greet",
        "name",
        "string",
    ) is False


def test_extract_add_parameter_a_success(
    extractor: ParameterExtractor,
) -> None:
    """Extract the first numeric add parameter."""
    value = extractor.extract_parameter(
        "What is the sum of 2 and 3?",
        "fn_add_numbers",
        "a",
        "number",
    )
    assert value == 2.0


def test_extract_add_parameter_b_success(
    extractor: ParameterExtractor,
) -> None:
    """Extract the second numeric add parameter."""
    value = extractor.extract_parameter(
        "What is the sum of 2 and 3?",
        "fn_add_numbers",
        "b",
        "number",
    )
    assert value == 3.0


def test_extract_add_large_numbers_success(
    extractor: ParameterExtractor,
) -> None:
    """Extract large numeric values for the add function."""
    value_a = extractor.extract_parameter(
        "What is the sum of 265 and 345?",
        "fn_add_numbers",
        "a",
        "number",
    )
    value_b = extractor.extract_parameter(
        "What is the sum of 265 and 345?",
        "fn_add_numbers",
        "b",
        "number",
    )

    assert value_a == 265.0
    assert value_b == 345.0


def test_extract_greet_name_success_shrek(
    extractor: ParameterExtractor,
) -> None:
    """Extract the greeting target name from a basic prompt."""
    value = extractor.extract_parameter(
        "Greet shrek",
        "fn_greet",
        "name",
        "string",
    )
    assert value == "shrek"


def test_extract_greet_name_success_john(
    extractor: ParameterExtractor,
) -> None:
    """Extract another greeting target name."""
    value = extractor.extract_parameter(
        "Greet john",
        "fn_greet",
        "name",
        "string",
    )
    assert value == "john"


def test_extract_reverse_string_parameter_hello(
    extractor: ParameterExtractor,
) -> None:
    """Extract the reverse-string parameter from the quoted prompt."""
    value = extractor.extract_parameter(
        "Reverse the string 'hello'",
        "fn_reverse_string",
        "s",
        "string",
    )
    assert value == "hello"


def test_extract_reverse_string_parameter_world(
    extractor: ParameterExtractor,
) -> None:
    """Extract another reverse-string parameter from the quoted prompt."""
    value = extractor.extract_parameter(
        "Reverse the string 'world'",
        "fn_reverse_string",
        "s",
        "string",
    )
    assert value == "world"


def test_extract_square_root_value_16(
    extractor: ParameterExtractor,
) -> None:
    """Extract the numeric square-root argument."""
    value = extractor.extract_parameter(
        "What is the square root of 16?",
        "fn_get_square_root",
        "a",
        "number",
    )
    assert value == 16.0


def test_extract_square_root_value_144(
    extractor: ParameterExtractor,
) -> None:
    """Extract another square-root argument."""
    value = extractor.extract_parameter(
        "Calculate the square root of 144",
        "fn_get_square_root",
        "a",
        "number",
    )
    assert value == 144.0


def test_extract_regex_source_string_success(
    extractor: ParameterExtractor,
) -> None:
    """Extract the source string for the regex substitution prompt."""
    value = extractor.extract_parameter(
        'Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS',
        "fn_substitute_string_with_regex",
        "source_string",
        "string",
    )
    assert value == "Hello 34 I'm 233 years old"


def test_extract_regex_pattern_for_numbers_success(
    extractor: ParameterExtractor,
) -> None:
    """Extract the regex pattern for the numbers substitution prompt."""
    value = extractor.extract_parameter(
        'Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS',
        "fn_substitute_string_with_regex",
        "regex",
        "string",
    )
    assert value == r"\d"


def test_extract_regex_replacement_for_numbers_success(
    extractor: ParameterExtractor,
) -> None:
    """Extract the replacement text for the numbers substitution prompt."""
    value = extractor.extract_parameter(
        'Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS',
        "fn_substitute_string_with_regex",
        "replacement",
        "string",
    )
    assert value == "NUMBERS"


def test_extract_regex_source_string_for_vowels_success(
    extractor: ParameterExtractor,
) -> None:
    """Extract the source string for the vowels substitution prompt."""
    value = extractor.extract_parameter(
        "Replace all vowels in 'Programming is fun' with asterisks",
        "fn_substitute_string_with_regex",
        "source_string",
        "string",
    )
    assert value == "Programming is fun"


def test_extract_regex_pattern_for_vowels_success(
    extractor: ParameterExtractor,
) -> None:
    """Extract the vowel regex for the vowels substitution prompt."""
    value = extractor.extract_parameter(
        "Replace all vowels in 'Programming is fun' with asterisks",
        "fn_substitute_string_with_regex",
        "regex",
        "string",
    )
    assert value == "[AEIOUaeiou]"


def test_extract_regex_replacement_for_vowels_success(
    extractor: ParameterExtractor,
) -> None:
    """Extract the replacement text for the vowels substitution prompt."""
    value = extractor.extract_parameter(
        "Replace all vowels in 'Programming is fun' with asterisks",
        "fn_substitute_string_with_regex",
        "replacement",
        "string",
    )
    assert value == "*"


def test_extract_regex_source_string_for_word_substitution_success(
    extractor: ParameterExtractor,
) -> None:
    """Extract the source string for the word substitution prompt."""
    value = extractor.extract_parameter(
        "Substitute the word 'cat' with 'dog' in "
        "'The cat sat on the mat with another cat'",
        "fn_substitute_string_with_regex",
        "source_string",
        "string",
    )
    assert value == "The cat sat on the mat with another cat"


def test_extract_regex_pattern_for_word_substitution_success(
    extractor: ParameterExtractor,
) -> None:
    """Extract the literal regex pattern for the word substitution prompt."""
    value = extractor.extract_parameter(
        "Substitute the word 'cat' with 'dog' in "
        "'The cat sat on the mat with another cat'",
        "fn_substitute_string_with_regex",
        "regex",
        "string",
    )
    assert value == "cat"


def test_extract_regex_replacement_for_word_substitution_success(
    extractor: ParameterExtractor,
) -> None:
    """Extract the replacement text for the word substitution prompt."""
    value = extractor.extract_parameter(
        "Substitute the word 'cat' with 'dog' in "
        "'The cat sat on the mat with another cat'",
        "fn_substitute_string_with_regex",
        "replacement",
        "string",
    )
    assert value == "dog"


def test_extract_add_missing_b_returns_none_or_raises(
    extractor: ParameterExtractor,
) -> None:
    """Handle missing second add parameter."""
    try:
        value = extractor.extract_parameter(
            "What is the sum of 7?",
            "fn_add_numbers",
            "b",
            "number",
        )
        assert value is None
    except ValueError as exc:
        assert "Missing enough information" in str(exc)


def test_extract_reverse_without_text_returns_none_or_raises(
    extractor: ParameterExtractor,
) -> None:
    """Handle reverse calls without text."""
    try:
        value = extractor.extract_parameter(
            "Reverse the string",
            "fn_reverse_string",
            "s",
            "string",
        )
        assert value is None
    except ValueError:
        pass


def test_extract_square_root_without_value_returns_none_or_raises(
    extractor: ParameterExtractor,
) -> None:
    """Handle square root calls without a value."""
    try:
        value = extractor.extract_parameter(
            "What is the square root?",
            "fn_get_square_root",
            "a",
            "number",
        )
        assert value is None
    except ValueError:
        pass
