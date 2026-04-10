from typing import Any

import pytest

import src.services.pipeline as pipeline_module
from src.models.selection_result import SelectionResult


def make_function(
    name: str,
    description: str,
    parameters: dict[str, Any],
    returns_type: str = "string",
    returns_description: str = "Return value",
) -> dict[str, Any]:
    """Build a function definition for tests."""
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "returns": {
            "type": returns_type,
            "description": returns_description,
        },
    }


class FakeSelector:
    """Provide canned selection results for pipeline tests."""

    responses: dict[str, SelectionResult] = {}

    def __init__(self, functions: list[dict[str, Any]]) -> None:
        """Store available functions for the fake selector."""
        self.functions = functions

    def select_and_extract(self, prompt: str) -> SelectionResult:
        """Return the predefined selection result for a prompt."""
        return self.responses[prompt]


def test_run_pipeline_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Write a successful pipeline result to output."""
    functions_raw: list[dict[str, Any]] = [
        make_function(
            "fn_add_numbers",
            "Add two numbers together and return their sum.",
            {
                "a": {
                    "type": "number",
                    "description": "First number",
                },
                "b": {
                    "type": "number",
                    "description": "Second number",
                },
            },
            returns_type="number",
            returns_description="Sum of the two numbers",
        )
    ]
    prompts_raw: list[dict[str, str]] = [
        {"prompt": "What is the sum of 2 and 3?"}
    ]

    captured_output: dict[str, Any] = {}

    def fake_read_json_file(path: object) -> list[dict[str, Any]]:
        """Return fake input data based on the requested path."""
        if "functions" in str(path):
            return functions_raw
        return prompts_raw

    def fake_write_json_file(path: object, data: Any) -> None:
        """Capture written output data for assertions."""
        captured_output["path"] = path
        captured_output["data"] = data

    FakeSelector.responses = {
        "What is the sum of 2 and 3?": SelectionResult(
            prompt="What is the sum of 2 and 3?",
            name="fn_add_numbers",
            parameters={"a": 2.0, "b": 3.0},
        )
    }

    monkeypatch.setattr(pipeline_module, "read_json_file", fake_read_json_file)
    monkeypatch.setattr(
        pipeline_module,
        "write_json_file",
        fake_write_json_file,
    )
    monkeypatch.setattr(pipeline_module, "FunctionSelector", FakeSelector)

    pipeline_module.run_pipeline("functions.json", "input.json", "output.json")

    assert captured_output["data"] == [
        {
            "prompt": "What is the sum of 2 and 3?",
            "fn_name": "fn_add_numbers",
            "args": {"a": 2.0, "b": 3.0},
        }
    ]


def test_run_pipeline_keeps_null_call_in_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep prompts with no valid function as null calls in output."""
    functions_raw: list[dict[str, Any]] = [
        make_function(
            "fn_add_numbers",
            "Add two numbers together and return their sum.",
            {
                "a": {
                    "type": "number",
                    "description": "First number",
                },
                "b": {
                    "type": "number",
                    "description": "Second number",
                },
            },
            returns_type="number",
            returns_description="Sum of the two numbers",
        )
    ]
    prompts_raw: list[dict[str, str]] = [{"prompt": "???"}]

    captured_output: dict[str, Any] = {}

    def fake_read_json_file(path: object) -> list[dict[str, Any]]:
        """Return fake input data based on the requested path."""
        if "functions" in str(path):
            return functions_raw
        return prompts_raw

    def fake_write_json_file(path: object, data: Any) -> None:
        """Capture written output data for assertions."""
        del path
        captured_output["data"] = data

    FakeSelector.responses = {
        "???": SelectionResult(
            prompt="???",
            name=None,
            parameters={},
            error=(
                "Could not determine a valid target function "
                "from the prompt."
            ),
        )
    }

    monkeypatch.setattr(pipeline_module, "read_json_file", fake_read_json_file)
    monkeypatch.setattr(
        pipeline_module,
        "write_json_file",
        fake_write_json_file,
    )
    monkeypatch.setattr(pipeline_module, "FunctionSelector", FakeSelector)

    pipeline_module.run_pipeline("functions.json", "input.json", "output.json")

    assert captured_output["data"] == [
        {
            "prompt": "???",
            "fn_name": None,
            "args": {},
        }
    ]


def test_run_pipeline_processes_multiple_prompts_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep batch order and serialize each result in scale format."""
    functions_raw: list[dict[str, Any]] = [
        make_function(
            "fn_greet",
            "Generate a greeting message for a person by name.",
            {
                "name": {
                    "type": "string",
                    "description": "Name of the person",
                },
            },
            returns_type="string",
            returns_description="Greeting message",
        ),
        make_function(
            "fn_reverse_string",
            "Reverse a string and return the reversed result.",
            {
                "s": {
                    "type": "string",
                    "description": "String to reverse",
                },
            },
            returns_type="string",
            returns_description="Reversed string",
        ),
    ]
    prompts_raw: list[dict[str, str]] = [
        {"prompt": "Greet john"},
        {"prompt": "Reverse the string 'world'"},
    ]

    captured_output: dict[str, Any] = {}

    def fake_read_json_file(path: object) -> list[dict[str, Any]]:
        """Return fake input data based on the requested path."""
        if "functions" in str(path):
            return functions_raw
        return prompts_raw

    def fake_write_json_file(path: object, data: Any) -> None:
        """Capture written output data for assertions."""
        del path
        captured_output["data"] = data

    FakeSelector.responses = {
        "Greet john": SelectionResult(
            prompt="Greet john",
            name="fn_greet",
            parameters={"name": "john"},
        ),
        "Reverse the string 'world'": SelectionResult(
            prompt="Reverse the string 'world'",
            name="fn_reverse_string",
            parameters={"s": "world"},
        ),
    }

    monkeypatch.setattr(pipeline_module, "read_json_file", fake_read_json_file)
    monkeypatch.setattr(
        pipeline_module,
        "write_json_file",
        fake_write_json_file,
    )
    monkeypatch.setattr(pipeline_module, "FunctionSelector", FakeSelector)

    pipeline_module.run_pipeline("functions.json", "input.json", "output.json")

    assert captured_output["data"] == [
        {
            "prompt": "Greet john",
            "fn_name": "fn_greet",
            "args": {"name": "john"},
        },
        {
            "prompt": "Reverse the string 'world'",
            "fn_name": "fn_reverse_string",
            "args": {"s": "world"},
        },
    ]


def test_run_pipeline_serializes_regex_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serialize the regex substitution case in the final output format."""
    functions_raw: list[dict[str, Any]] = [
        make_function(
            "fn_substitute_string_with_regex",
            "Replace all occurrences matching a regex pattern in a string.",
            {
                "source_string": {
                    "type": "string",
                    "description": "Original source string",
                },
                "regex": {
                    "type": "string",
                    "description": "Regex pattern to apply",
                },
                "replacement": {
                    "type": "string",
                    "description": "Replacement text",
                },
            },
            returns_type="string",
            returns_description="Updated string",
        )
    ]
    prompts_raw: list[dict[str, str]] = [
        {
            "prompt": (
                'Replace all numbers in "Hello 34 I\'m 233 years old" '
                "with NUMBERS"
            )
        }
    ]

    captured_output: dict[str, Any] = {}

    def fake_read_json_file(path: object) -> list[dict[str, Any]]:
        """Return fake input data based on the requested path."""
        if "functions" in str(path):
            return functions_raw
        return prompts_raw

    def fake_write_json_file(path: object, data: Any) -> None:
        """Capture written output data for assertions."""
        del path
        captured_output["data"] = data

    FakeSelector.responses = {
        'Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS': (
            SelectionResult(
                prompt=(
                    'Replace all numbers in "Hello 34 I\'m 233 years old" '
                    "with NUMBERS"
                ),
                name="fn_substitute_string_with_regex",
                parameters={
                    "source_string": "Hello 34 I'm 233 years old",
                    "regex": r"\d",
                    "replacement": "NUMBERS",
                },
            )
        )
    }

    monkeypatch.setattr(pipeline_module, "read_json_file", fake_read_json_file)
    monkeypatch.setattr(
        pipeline_module,
        "write_json_file",
        fake_write_json_file,
    )
    monkeypatch.setattr(pipeline_module, "FunctionSelector", FakeSelector)

    pipeline_module.run_pipeline("functions.json", "input.json", "output.json")

    assert captured_output["data"] == [
        {
            "prompt": 'Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS', # noqa
            "fn_name": "fn_substitute_string_with_regex",
            "args": {
                "source_string": "Hello 34 I'm 233 years old",
                "regex": r"\d",
                "replacement": "NUMBERS",
            },
        }
    ]
