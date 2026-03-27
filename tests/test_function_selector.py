import pytest

from src.services.function_selector import FunctionSelector


class FakeConstrainedDecoder:
    next_result = None
    next_exception = None

    def __init__(self, functions):
        self.functions = functions

    def generate_call(self, prompt, functions):
        if FakeConstrainedDecoder.next_exception is not None:
            exc = FakeConstrainedDecoder.next_exception
            FakeConstrainedDecoder.next_exception = None
            raise exc

        result = FakeConstrainedDecoder.next_result
        FakeConstrainedDecoder.next_result = None
        return result


@pytest.fixture
def selector():
    functions = [
        {
            "name": "fn_add_numbers",
            "parameters": {
                "a": "number",
                "b": "number",
            },
        },
        {
            "name": "fn_greet",
            "parameters": {
                "name": "string",
            },
        },
        {
            "name": "fn_ping",
            "parameters": {},
        },
    ]
    return FunctionSelector(
        functions=functions,
        decoder=FakeConstrainedDecoder(functions),
    )


def test_returns_valid_result_without_error_field(selector):
    FakeConstrainedDecoder.next_result = {
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0,
        },
    }

    result = selector.select_and_extract("Add 2 and 3").model_dump()

    assert result == {
        "prompt": "Add 2 and 3",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0,
        },
        "error": None,
    }


def test_returns_structured_error_for_incomplete_prompt(selector):
    FakeConstrainedDecoder.next_exception = ValueError(
        "Missing enough information to extract parameter 'b' for function 'fn_add_numbers'."
    )

    result = selector.select_and_extract("Add 7").model_dump()

    assert result == {
        "prompt": "Add 7",
        "name": None,
        "parameters": {},
        "error": "Missing enough information to extract parameter 'b' for function 'fn_add_numbers'.",
    }


def test_returns_structured_error_for_prompt_without_clear_intent(selector):
    FakeConstrainedDecoder.next_exception = ValueError(
        "Could not determine a valid target function from the prompt."
    )

    result = selector.select_and_extract("???").model_dump()

    assert result == {
        "prompt": "???",
        "name": None,
        "parameters": {},
        "error": "Could not determine a valid target function from the prompt.",
    }


def test_preserves_prompt_in_success_case(selector):
    FakeConstrainedDecoder.next_result = {
        "name": "fn_greet",
        "parameters": {
            "name": "Alice",
        },
    }

    result = selector.select_and_extract("Greet Alice").model_dump()

    assert result["prompt"] == "Greet Alice"
    assert result["name"] == "fn_greet"
    assert result["parameters"] == {"name": "Alice"}
    assert result["error"] is None


def test_preserves_prompt_in_error_case(selector):
    FakeConstrainedDecoder.next_exception = ValueError("Some extraction error")

    result = selector.select_and_extract("bad prompt").model_dump()

    assert result["prompt"] == "bad prompt"
    assert result["name"] is None
    assert result["parameters"] == {}
    assert result["error"] == "Some extraction error"


def test_error_field_only_exists_when_there_is_an_error(selector):
    FakeConstrainedDecoder.next_result = {
        "name": "fn_add_numbers",
        "parameters": {
            "a": 1.0,
            "b": 2.0,
        },
    }

    ok_result = selector.select_and_extract("Add 1 and 2").model_dump()
    assert ok_result["error"] is None

    FakeConstrainedDecoder.next_exception = ValueError("boom")
    error_result = selector.select_and_extract("bad").model_dump()

    assert error_result["error"] == "boom"


def test_uses_decoder_result_even_with_empty_parameters(selector):
    FakeConstrainedDecoder.next_result = {
        "name": "fn_ping",
        "parameters": {},
    }

    result = selector.select_and_extract("Ping").model_dump()

    assert result == {
        "prompt": "Ping",
        "name": "fn_ping",
        "parameters": {},
        "error": None,
    }


def test_handles_generic_exception_as_structured_error(selector):
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


def test_returns_structured_error_when_decoder_returns_non_dict(selector):
    FakeConstrainedDecoder.next_result = "not a dict"

    result = selector.select_and_extract("bad result").model_dump()

    assert result == {
        "prompt": "bad result",
        "name": None,
        "parameters": {},
        "error": "Decoder returned a non-dict result",
    }


def test_returns_structured_error_when_decoder_returns_invalid_name(selector):
    FakeConstrainedDecoder.next_result = {
        "name": 123,
        "parameters": {},
    }

    result = selector.select_and_extract("bad name").model_dump()

    assert result == {
        "prompt": "bad name",
        "name": None,
        "parameters": {},
        "error": "Decoder returned an invalid function name",
    }


def test_returns_structured_error_when_decoder_returns_invalid_parameters(selector):
    FakeConstrainedDecoder.next_result = {
        "name": "fn_add_numbers",
        "parameters": ["not", "a", "dict"],
    }

    result = selector.select_and_extract("bad parameters").model_dump()

    assert result == {
        "prompt": "bad parameters",
        "name": None,
        "parameters": {},
        "error": "Decoder returned invalid parameters",
    }


def test_clears_parameters_when_name_is_none(selector):
    FakeConstrainedDecoder.next_result = {
        "name": None,
        "parameters": {"unexpected": "value"},
    }

    result = selector.select_and_extract("unknown").model_dump()

    assert result == {
        "prompt": "unknown",
        "name": None,
        "parameters": {},
        "error": None,
    }
