from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from src.errors import InputFileError, ValidationError
from src.file_io import read_json_file, write_json_file
from src.models import FunctionCallResult, FunctionDefinition, PromptInput


def run_pipeline(
    functions_definition_path: Path,
    input_path: Path,
    output_path: Path,
) -> None:
    """Run the project pipeline."""
    raw_functions = read_json_file(functions_definition_path)
    raw_inputs = read_json_file(input_path)

    if not isinstance(raw_functions, list):
        raise InputFileError("Function definitions file must contain a JSON array.") # noqa

    if not isinstance(raw_inputs, list):
        raise InputFileError("Input prompts file must contain a JSON array.") # noqa

    try:
        functions = [FunctionDefinition.model_validate(item) for item in raw_functions] # noqa
        prompts = [PromptInput.model_validate(item) for item in raw_inputs]
    except PydanticValidationError as exc:
        raise ValidationError(f"Invalid input structure: {exc}") from exc

    results = []
    fallback_name = functions[0].name if functions else "undefined_function"

    for prompt in prompts:
        result = FunctionCallResult(
            prompt=prompt.prompt,
            name=fallback_name,
            parameters={},
        )
        results.append(result.model_dump())

    write_json_file(output_path, results)
