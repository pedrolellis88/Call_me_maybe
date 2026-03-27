from src.services.function_selector import FunctionSelector


class FakeDecoder:
    next_result = None
    next_exception = None

    def __init__(self, functions):
        self.functions = functions

    def decode(self, prompt):
        if FakeDecoder.next_exception is not None:
            exc = FakeDecoder.next_exception
            FakeDecoder.next_exception = None
            raise exc

        result = FakeDecoder.next_result
        FakeDecoder.next_result = None
        return result


def make_functions():
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


def test_valid_result_contract():
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


def test_invalid_result_contract():
    selector = FunctionSelector(
        functions=make_functions(),
        decoder=FakeDecoder(make_functions()),
    )

    FakeDecoder.next_exception = ValueError(
        "Missing enough information to extract parameter 'b' for function 'fn_add_numbers'."
    )

    result = selector.select_and_extract("Add 7")

    assert result == {
        "prompt": "Add 7",
        "name": None,
        "parameters": {},
        "error": "Missing enough information to extract parameter 'b' for function 'fn_add_numbers'.",
    }