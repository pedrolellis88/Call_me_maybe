import pytest

from src.services.function_selector import FunctionSelector


class FakeConstrainedDecoder:
    next_result = None
    next_exception = None

    def __init__(self, functions):
        self.functions = functions

    def decode(self, prompt):
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
    return FunctionSelector(functions=functions, decoder=FakeConstrainedDecoder(functions))


def test_returns_valid_result_without_error_field(selector):
    FakeConstrainedDecoder.next_result = {
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0,
        },
    }

    result = selector.select_and_extract("Add 2 and 3")

    assert result == {
        "prompt": "Add 2 and 3",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0,
        },
    }
    assert "error" not in result


def test_returns_structured_error_for_incomplete_prompt(selector):
    FakeConstrainedDecoder.next_exception = ValueError(
        "Missing enough information to extract parameter 'b' for function 'fn_add_numbers'."
    )

    result = selector.select_and_extract("Add 7")

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

    result = selector.select_and_extract("???")

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

    result = selector.select_and_extract("Greet Alice")

    assert result["prompt"] == "Greet Alice"
    assert result["name"] == "fn_greet"
    assert result["parameters"] == {"name": "Alice"}
    assert "error" not in result


def test_preserves_prompt_in_error_case(selector):
    FakeConstrainedDecoder.next_exception = ValueError("Some extraction error")

    result = selector.select_and_extract("bad prompt")

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

    ok_result = selector.select_and_extract("Add 1 and 2")
    assert "error" not in ok_result

    FakeConstrainedDecoder.next_exception = ValueError("boom")
    error_result = selector.select_and_extract("bad")

    assert error_result["error"] == "boom"


def test_uses_decoder_result_even_with_empty_parameters(selector):
    FakeConstrainedDecoder.next_result = {
        "name": "fn_ping",
        "parameters": {},
    }

    result = selector.select_and_extract("Ping")

    assert result == {
        "prompt": "Ping",
        "name": "fn_ping",
        "parameters": {},
    }


def test_handles_generic_exception_as_structured_error(selector):
    FakeConstrainedDecoder.next_exception = RuntimeError("Unexpected decoder failure")

    result = selector.select_and_extract("some prompt")

    assert result == {
        "prompt": "some prompt",
        "name": None,
        "parameters": {},
        "error": "Unexpected decoder failure",
    }