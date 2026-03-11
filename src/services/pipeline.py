from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from src.errors import InputFileError, ValidationError
from src.io import read_json_file, write_json_file
from src.models import FunctionDefinition, PromptInput


def run_pipeline(
    functions_definition_path: Path,
    input_path: Path,
    output_path: Path,
) -> None:
    """Run the project pipeline."""
    raw_functions = read_json_file(functions_definition_path)
    raw_inputs = read_json_file(input_path)

    try:
        functions = [FunctionDefinition.model_validate(item) for item in raw_functions] # noqa
        prompts = [PromptInput.model_validate(item) for item in raw_inputs]
    except PydanticValidationError as exc:
        raise ValidationError(f"Invalid input structure: {exc}") from exc
    except TypeError as exc:
        raise InputFileError(f"Input data must be a JSON array: {exc}") from exc # noqa

    # Dia 1: só confirmar leitura/validação e gerar placeholder.
    results = []
    fallback_name = functions[0].name if functions else "undefined_function"

    for prompt in prompts:
        results.append(
            {
                "prompt": prompt.prompt,
                "name": fallback_name,
                "parameters": {},
            }
        )

    write_json_file(output_path, results)
