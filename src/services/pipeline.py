from pathlib import Path

from src.file_io.reader import read_json_file
from src.file_io.writer import write_json_file
from src.models.function_call_result import FunctionCallResult
from src.models.function_definition import FunctionDefinition
from src.models.prompt_input import PromptInput
from src.services.function_selector import FunctionSelector


def run_pipeline(
    functions_definition_path: str,
    input_path: str,
    output_path: str,
) -> None:
    """Run the full function-calling pipeline."""
    functions_raw = read_json_file(Path(functions_definition_path))
    prompts_raw = read_json_file(Path(input_path))

    functions = [FunctionDefinition(**item) for item in functions_raw]
    prompts = [PromptInput(**item) for item in prompts_raw]

    selector = FunctionSelector([fn.model_dump() for fn in functions])

    results: list[FunctionCallResult] = []

    for prompt in prompts:
        selection = selector.select_and_extract(prompt.prompt)

        result = FunctionCallResult(
            prompt=selection.prompt,
            name=selection.name,
            parameters=selection.parameters if selection.name is not None else {}, # noqa
        )
        results.append(result)

    write_json_file(
        Path(output_path),
        [result.model_dump() for result in results],
    )
