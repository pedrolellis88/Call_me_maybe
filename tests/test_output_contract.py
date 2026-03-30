from typing import Any

from src.services.function_selector import FunctionSelector


class FakeDecoder:
    """Test double for decoder behavior."""

    next_result: dict[str, Any] | None = None
    next_exception: Exception | None = None

    def __init__(self, functions: list[dict[str, Any]]) -> None:
        """Store available functions for the fake decoder."""
        self.functions = functions

    def generate_call(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Return the prepared result or raise the prepared exception."""
        del prompt
        del functions
        if FakeDecoder.next_exception is not None:
            exc = FakeDecoder.next_exception
            FakeDecoder.next_exception = None
            raise exc

        result = FakeDecoder.next_result
        FakeDecoder.next_result = None
        return result


def make_functions() -> list[dict[str, Any]]:
    """Build a reusable list of test functions."""
    return [
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
    ]


def test_valid_result_contract() -> None:
    """Validate the success output contract."""
    selector = FunctionSelector(
        functions=make_functions(),
        decoder=FakeDecoder(make_functions()),
    )

    FakeDecoder.next_result = {
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


def test_invalid_result_contract() -> None:
    """Validate the error output contract for missing data."""
    selector = FunctionSelector(
        functions=make_functions(),
        decoder=FakeDecoder(make_functions()),
    )

    FakeDecoder.next_exception = ValueError(
        "Missing enough information to extract parameter 'b' "
        "for function 'fn_add_numbers'."
    )

    result = selector.select_and_extract("Add 7").model_dump()

    assert result == {
        "prompt": "Add 7",
        "name": None,
        "parameters": {},
        "error": (
            "Missing enough information to extract parameter 'b' "
            "for function 'fn_add_numbers'."
        ),
    }


def test_unclear_intent_result_contract() -> None:
    """Validate the error output contract for unclear intent."""
    selector = FunctionSelector(
        functions=make_functions(),
        decoder=FakeDecoder(make_functions()),
    )

    FakeDecoder.next_exception = ValueError(
        "Could not determine a valid target function from the prompt."
    )

    result = selector.select_and_extract("???").model_dump()

    assert result == {
        "prompt": "???",
        "name": None,
        "parameters": {},
        "error": (
            "Could not determine a valid target function "
            "from the prompt."
        ),
    }
