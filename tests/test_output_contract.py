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
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
        },
        {
            "name": "fn_greet",
            "parameters": {
                "name": {"type": "string"},
            },
        },
    ]


def test_selector_normalizes_success_result() -> None:
    """Validate selector normalization for a successful decoder result."""
    selector = FunctionSelector(
        functions=make_functions(),
        decoder=FakeDecoder(make_functions()),
    )

    FakeDecoder.next_result = {
        "fn_name": "fn_add_numbers",
        "args": {
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


def test_selector_returns_structured_error_for_missing_data() -> None:
    """Validate selector error handling for missing data."""
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


def test_selector_returns_structured_error_for_unclear_intent() -> None:
    """Validate selector error handling for unclear intent."""
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
