import src.services.pipeline as pipeline_module
from src.models.selection_result import SelectionResult


def make_function(
    name,
    description,
    parameters,
    returns_type="string",
    returns_description="Return value",
):
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
    responses = {}

    def __init__(self, functions):
        self.functions = functions

    def select_and_extract(self, prompt):
        return self.responses[prompt]


def test_run_pipeline_success(monkeypatch):
    functions_raw = [
        make_function(
            "fn_add_numbers",
            "Add two numbers",
            {
                "a": {
                    "type": "number",
                    "required": True,
                    "description": "First number",
                },
                "b": {
                    "type": "number",
                    "required": True,
                    "description": "Second number",
                },
            },
            returns_type="number",
            returns_description="Sum of the two numbers",
        )
    ]
    prompts_raw = [{"prompt": "Add 2 and 3"}]

    captured_output = {}

    def fake_read_json_file(path):
        if "functions" in str(path):
            return functions_raw
        return prompts_raw

    def fake_write_json_file(path, data):
        captured_output["path"] = path
        captured_output["data"] = data

    FakeSelector.responses = {
        "Add 2 and 3": SelectionResult(
            prompt="Add 2 and 3",
            name="fn_add_numbers",
            parameters={"a": 2.0, "b": 3.0},
        )
    }

    monkeypatch.setattr(pipeline_module, "read_json_file", fake_read_json_file)
    monkeypatch.setattr(pipeline_module, "write_json_file", fake_write_json_file)
    monkeypatch.setattr(pipeline_module, "FunctionSelector", FakeSelector)

    pipeline_module.run_pipeline("functions.json", "input.json", "output.json")

    assert captured_output["data"] == [
        {
            "prompt": "Add 2 and 3",
            "name": "fn_add_numbers",
            "parameters": {"a": 2.0, "b": 3.0},
        }
    ]


def test_run_pipeline_selection_error_does_not_abort(monkeypatch):
    functions_raw = [
        make_function(
            "fn_add_numbers",
            "Add two numbers",
            {
                "a": {
                    "type": "number",
                    "required": True,
                    "description": "First number",
                },
                "b": {
                    "type": "number",
                    "required": True,
                    "description": "Second number",
                },
            },
            returns_type="number",
            returns_description="Sum of the two numbers",
        ),
        make_function(
            "fn_greet",
            "Greet someone",
            {
                "name": {
                    "type": "string",
                    "required": True,
                    "description": "Name of the person",
                },
            },
            returns_type="string",
            returns_description="Greeting message",
        ),
    ]
    prompts_raw = [
        {"prompt": "Add 7"},
        {"prompt": "Greet Alice"},
    ]

    captured_output = {}

    def fake_read_json_file(path):
        if "functions" in str(path):
            return functions_raw
        return prompts_raw

    def fake_write_json_file(path, data):
        captured_output["data"] = data

    FakeSelector.responses = {
        "Add 7": SelectionResult(
            prompt="Add 7",
            name=None,
            parameters={},
            error="Missing enough information to extract parameter 'b' for function 'fn_add_numbers'.",
        ),
        "Greet Alice": SelectionResult(
            prompt="Greet Alice",
            name="fn_greet",
            parameters={"name": "Alice"},
        ),
    }

    monkeypatch.setattr(pipeline_module, "read_json_file", fake_read_json_file)
    monkeypatch.setattr(pipeline_module, "write_json_file", fake_write_json_file)
    monkeypatch.setattr(pipeline_module, "FunctionSelector", FakeSelector)

    pipeline_module.run_pipeline("functions.json", "input.json", "output.json")

    assert captured_output["data"] == [
        {
            "prompt": "Add 7",
            "name": None,
            "parameters": {},
        },
        {
            "prompt": "Greet Alice",
            "name": "fn_greet",
            "parameters": {"name": "Alice"},
        },
    ]


def test_run_pipeline_error_does_not_abort_batch(monkeypatch):
    functions_raw = [
        make_function(
            "fn_add_numbers",
            "Add two numbers",
            {
                "a": {
                    "type": "number",
                    "required": True,
                    "description": "First number",
                },
                "b": {
                    "type": "number",
                    "required": True,
                    "description": "Second number",
                },
            },
            returns_type="number",
            returns_description="Sum of the two numbers",
        ),
        make_function(
            "fn_ping",
            "Ping",
            {},
            returns_type="string",
            returns_description="Ping response",
        ),
    ]
    prompts_raw = [
        {"prompt": "Add 7 and maybe something"},
        {"prompt": "Ping"},
    ]

    captured_output = {}

    def fake_read_json_file(path):
        if "functions" in str(path):
            return functions_raw
        return prompts_raw

    def fake_write_json_file(path, data):
        captured_output["data"] = data

    FakeSelector.responses = {
        "Add 7 and maybe something": SelectionResult(
            prompt="Add 7 and maybe something",
            name=None,
            parameters={},
            error="Missing enough information to extract parameter 'b' for function 'fn_add_numbers'.",
        ),
        "Ping": SelectionResult(
            prompt="Ping",
            name="fn_ping",
            parameters={},
        ),
    }

    monkeypatch.setattr(pipeline_module, "read_json_file", fake_read_json_file)
    monkeypatch.setattr(pipeline_module, "write_json_file", fake_write_json_file)
    monkeypatch.setattr(pipeline_module, "FunctionSelector", FakeSelector)

    pipeline_module.run_pipeline("functions.json", "input.json", "output.json")

    assert captured_output["data"] == [
        {
            "prompt": "Add 7 and maybe something",
            "name": None,
            "parameters": {},
        },
        {
            "prompt": "Ping",
            "name": "fn_ping",
            "parameters": {},
        },
    ]


def test_run_pipeline_unclear_intent_is_kept_in_output_as_null_call(monkeypatch):
    functions_raw = [
        make_function(
            "fn_add_numbers",
            "Add two numbers",
            {
                "a": {
                    "type": "number",
                    "required": True,
                    "description": "First number",
                },
                "b": {
                    "type": "number",
                    "required": True,
                    "description": "Second number",
                },
            },
            returns_type="number",
            returns_description="Sum of the two numbers",
        )
    ]
    prompts_raw = [{"prompt": "???"}]

    captured_output = {}

    def fake_read_json_file(path):
        if "functions" in str(path):
            return functions_raw
        return prompts_raw

    def fake_write_json_file(path, data):
        captured_output["data"] = data

    FakeSelector.responses = {
        "???": SelectionResult(
            prompt="???",
            name=None,
            parameters={},
            error="Could not determine a valid target function from the prompt.",
        )
    }

    monkeypatch.setattr(pipeline_module, "read_json_file", fake_read_json_file)
    monkeypatch.setattr(pipeline_module, "write_json_file", fake_write_json_file)
    monkeypatch.setattr(pipeline_module, "FunctionSelector", FakeSelector)

    pipeline_module.run_pipeline("functions.json", "input.json", "output.json")

    assert captured_output["data"] == [
        {
            "prompt": "???",
            "name": None,
            "parameters": {},
        }
    ]
