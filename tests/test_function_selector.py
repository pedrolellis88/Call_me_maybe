from typing import Any

import pytest

from src.services.function_selector import FunctionSelector


class FakeConstrainedDecoder:
    """Test double for constrained decoder behavior."""

    next_result: dict[str, Any] | str | None = None
    next_exception: Exception | None = None

    def __init__(self, functions: list[dict[str, Any]]) -> None:
        """Store available functions for the fake decoder."""
        self.functions = functions

    def generate_call(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
    ) -> dict[str, Any] | str | None:
        """Return the prepared fake result or raise the prepared error."""
        del prompt
        del functions

        if FakeConstrainedDecoder.next_exception is not None:
            exc = FakeConstrainedDecoder.next_exception
            FakeConstrainedDecoder.next_exception = None
            raise exc

        result = FakeConstrainedDecoder.next_result
        FakeConstrainedDecoder.next_result = None
        return result


def make_functions() -> list[dict[str, Any]]:
    """Build a reusable list of test functions aligned with the base JSONs."""
    return [
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers together and return their sum.",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
        },
        {
            "name": "fn_greet",
            "description": "Generate a greeting message for a person by name.",
            "parameters": {
                "name": {"type": "string"},
            },
        },
        {
            "name": "fn_reverse_string",
            "description": "Reverse a string and return the reversed result.",
            "parameters": {
                "s": {"type": "string"},
            },
        },
        {
            "name": "fn_get_square_root",
            "description": "Calculate the square root of a number.",
            "parameters": {
                "a": {"type": "number"},
            },
        },
        {
            "name": "fn_substitute_string_with_regex",
            "description": (
                "Replace all occurrences matching a regex pattern in a string."
            ),
            "parameters": {
                "source_string": {"type": "string"},
                "regex": {"type": "string"},
                "replacement": {"type": "string"},
            },
        },
    ]


@pytest.fixture
def selector() -> FunctionSelector:
    """Build a selector configured with a fake decoder."""
    functions = make_functions()
    return FunctionSelector(
        functions=functions,
        decoder=FakeConstrainedDecoder(functions),
    )


def test_normalizes_add_result_from_decoder(
    selector: FunctionSelector,
) -> None:
    """Normalize decoder output for add-number prompts."""
    FakeConstrainedDecoder.next_result = {
        "fn_name": "fn_add_numbers",
        "args": {
            "a": 2.0,
            "b": 3.0,
        },
    }

    result = selector.select_and_extract(
        "What is the sum of 2 and 3?"
    ).model_dump()

    assert result == {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0,
        },
        "error": None,
    }


def test_normalizes_greet_result_from_decoder(
    selector: FunctionSelector,
) -> None:
    """Normalize decoder output for greet prompts."""
    FakeConstrainedDecoder.next_result = {
        "fn_name": "fn_greet",
        "args": {
            "name": "john",
        },
    }

    result = selector.select_and_extract("Greet john").model_dump()

    assert result == {
        "prompt": "Greet john",
        "name": "fn_greet",
        "parameters": {
            "name": "john",
        },
        "error": None,
    }


def test_normalizes_reverse_result_from_decoder(
    selector: FunctionSelector,
) -> None:
    """Normalize decoder output for reverse-string prompts."""
    FakeConstrainedDecoder.next_result = {
        "fn_name": "fn_reverse_string",
        "args": {
            "s": "hello",
        },
    }

    result = selector.select_and_extract(
        "Reverse the string 'hello'"
    ).model_dump()

    assert result == {
        "prompt": "Reverse the string 'hello'",
        "name": "fn_reverse_string",
        "parameters": {
            "s": "hello",
        },
        "error": None,
    }


def test_normalizes_square_root_result_from_decoder(
    selector: FunctionSelector,
) -> None:
    """Normalize decoder output for square-root prompts."""
    FakeConstrainedDecoder.next_result = {
        "fn_name": "fn_get_square_root",
        "args": {
            "a": 144.0,
        },
    }

    result = selector.select_and_extract(
        "Calculate the square root of 144"
    ).model_dump()

    assert result == {
        "prompt": "Calculate the square root of 144",
        "name": "fn_get_square_root",
        "parameters": {
            "a": 144.0,
        },
        "error": None,
    }


def test_normalizes_regex_substitution_result_from_decoder(
    selector: FunctionSelector,
) -> None:
    """Normalize decoder output for regex-substitution prompts."""
    FakeConstrainedDecoder.next_result = {
        "fn_name": "fn_substitute_string_with_regex",
        "args": {
            "source_string": "Programming is fun",
            "regex": "[AEIOUaeiou]",
            "replacement": "*",
        },
    }

    result = selector.select_and_extract(
        "Replace all vowels in 'Programming is fun' with asterisks"
    ).model_dump()

    assert result == {
        "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
        "name": "fn_substitute_string_with_regex",
        "parameters": {
            "source_string": "Programming is fun",
            "regex": "[AEIOUaeiou]",
            "replacement": "*",
        },
        "error": None,
    }


def test_preserves_prompt_in_success_case(
    selector: FunctionSelector,
) -> None:
    """Keep the original prompt on success."""
    FakeConstrainedDecoder.next_result = {
        "fn_name": "fn_greet",
        "args": {
            "name": "shrek",
        },
    }

    result = selector.select_and_extract("Greet shrek").model_dump()

    assert result["prompt"] == "Greet shrek"
    assert result["name"] == "fn_greet"
    assert result["parameters"] == {"name": "shrek"}
    assert result["error"] is None


def test_preserves_prompt_in_error_case(
    selector: FunctionSelector,
) -> None:
    """Keep the original prompt on failure."""
    FakeConstrainedDecoder.next_exception = ValueError(
        "Could not determine a valid target function from the prompt."
    )

    result = selector.select_and_extract("???").model_dump()

    assert result["prompt"] == "???"
    assert result["name"] is None
    assert result["parameters"] == {}
    assert (
        result["error"]
        == "Could not determine a valid target function from the prompt."
    )


def test_error_field_is_none_on_success(
    selector: FunctionSelector,
) -> None:
    """Keep the error field empty for successful results."""
    FakeConstrainedDecoder.next_result = {
        "fn_name": "fn_add_numbers",
        "args": {
            "a": 265.0,
            "b": 345.0,
        },
    }

    result = selector.select_and_extract(
        "What is the sum of 265 and 345?"
    ).model_dump()

    assert result["error"] is None


def test_handles_generic_exception_as_structured_error(
    selector: FunctionSelector,
) -> None:
    """Convert generic exceptions into structured errors."""
    FakeConstrainedDecoder.next_exception = RuntimeError(
        "Unexpected decoder failure"
    )

    result = selector.select_and_extract("some prompt").model_dump()

    assert result == {
        "prompt": "some prompt",
        "name": None,
        "parameters": {},
        "error": "Unexpected decoder failure",
    }


def test_returns_structured_error_when_decoder_returns_non_dict(
    selector: FunctionSelector,
) -> None:
    """Reject non-dict decoder results."""
    FakeConstrainedDecoder.next_result = "not a dict"

    result = selector.select_and_extract("bad result").model_dump()

    assert result == {
        "prompt": "bad result",
        "name": None,
        "parameters": {},
        "error": "Decoder returned a non-dict result",
    }


def test_returns_structured_error_when_decoder_returns_invalid_fn_name(
    selector: FunctionSelector,
) -> None:
    """Reject decoder results with invalid function names."""
    FakeConstrainedDecoder.next_result = {
        "fn_name": 123,
        "args": {},
    }

    result = selector.select_and_extract("bad name").model_dump()

    assert result == {
        "prompt": "bad name",
        "name": None,
        "parameters": {},
        "error": "Decoder returned an invalid function name",
    }


def test_returns_structured_error_when_decoder_returns_invalid_args(
    selector: FunctionSelector,
) -> None:
    """Reject decoder results with invalid arguments."""
    FakeConstrainedDecoder.next_result = {
        "fn_name": "fn_add_numbers",
        "args": ["not", "a", "dict"],
    }

    result = selector.select_and_extract("bad parameters").model_dump()

    assert result == {
        "prompt": "bad parameters",
        "name": None,
        "parameters": {},
        "error": "Decoder returned invalid parameters",
    }


def test_clears_parameters_when_fn_name_is_none(
    selector: FunctionSelector,
) -> None:
    """Clear parameters when no function name is returned."""
    FakeConstrainedDecoder.next_result = {
        "fn_name": None,
        "args": {"unexpected": "value"},
    }

    result = selector.select_and_extract("unknown").model_dump()

    assert result == {
        "prompt": "unknown",
        "name": None,
        "parameters": {},
        "error": None,
    }
