from pathlib import Path

from src.file_io.reader import read_json_file
from src.file_io.writer import write_json_file
from src.models.function_call_result import FunctionCallResult
from src.models.function_definition import FunctionDefinition
from src.models.prompt_input import PromptInput
from src.services.argument_extractor import ArgumentExtractor
from src.services.function_selector import FunctionSelector
from src.services.schema_validator import SchemaValidator


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

    selector = FunctionSelector()
    extractor = ArgumentExtractor()

    results = []

    for prompt in prompts:
        raw = selector.select(prompt.prompt, functions)

        if raw["error"] is not None or raw["name"] is None:
            results.append(
                {
                    "prompt": prompt.prompt,
                    "name": None,
                    "parameters": {},
                    "error": raw["error"],
                }
            )
            continue

        try:
            fn = SchemaValidator.find_function(raw["name"], functions)
            params = extractor.extract(raw, fn)

            result = FunctionCallResult(
                prompt=prompt.prompt,
                name=fn.name,
                parameters=params,
            )
            results.append(result.model_dump())
        except ValueError as exc:
            results.append(
                {
                    "prompt": prompt.prompt,
                    "name": raw["name"],
                    "parameters": {},
                    "error": str(exc),
                }
            )

    write_json_file(Path(output_path), results)