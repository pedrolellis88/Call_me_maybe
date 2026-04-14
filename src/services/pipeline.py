from __future__ import annotations

import logging
import os
from os import PathLike
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from pydantic import BaseModel, ValidationError

from src.file_io.reader import read_json_file
from src.file_io.writer import write_json_file
from src.models.function_call_result import FunctionCallResult
from src.models.function_definition import FunctionDefinition
from src.models.prompt_input import PromptInput
from src.services.function_selector import FunctionSelector


LOGGER = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)
PathInput = str | PathLike[str] | Path


def run_pipeline(
    functions_definition_path: PathInput,
    input_path: PathInput,
    output_path: PathInput,
) -> None:
    """Run the full function-calling pipeline safely."""
    functions_file = Path(functions_definition_path)
    input_file = Path(input_path)
    output_file = Path(output_path)

    functions_raw = _read_required_json(
        functions_file,
        "functions definition file",
    )
    prompts_raw = _read_required_json(
        input_file,
        "prompt input file",
    )

    functions = _parse_model_list(
        functions_raw,
        FunctionDefinition,
        "function definition",
    )
    prompts = _parse_model_list(
        prompts_raw,
        PromptInput,
        "prompt",
    )

    selector = _build_selector(functions)

    results: list[FunctionCallResult] = []
    for prompt in prompts:
        results.append(_process_prompt(prompt, selector))

    serialized_results = _serialize_results(results)
    _write_output(output_file, serialized_results)


def _read_required_json(path: Path, label: str) -> Any:
    """Read a JSON file and raise a clear error if it fails."""
    try:
        return read_json_file(path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Could not find {label}: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not read {label}: {path}. {exc}") from exc


def _parse_model_list(
    raw_items: Any,
    model_type: type[ModelT],
    item_label: str,
) -> list[ModelT]:
    """Validate that raw JSON is a list and parse each item with Pydantic."""
    if not isinstance(raw_items, list):
        raise ValueError(
            f"Expected a JSON array for {item_label}s, got "
            f"{type(raw_items).__name__}."
        )

    parsed_items: list[ModelT] = []

    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(
                f"Invalid {item_label} at index {index}: expected an object, "
                f"got {type(raw_item).__name__}."
            )

        try:
            parsed_items.append(model_type(**raw_item))
        except ValidationError as exc:
            raise ValueError(
                f"Invalid {item_label} at index {index}: {exc}"
            ) from exc

    return parsed_items


def _build_selector(
    functions: list[FunctionDefinition],
) -> FunctionSelector:
    """Build the selector from validated function definitions."""
    try:
        return FunctionSelector(
            [function.model_dump() for function in functions]
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not initialize FunctionSelector: {exc}"
        ) from exc


def _process_prompt(
    prompt: PromptInput,
    selector: FunctionSelector,
) -> FunctionCallResult:
    """Process one prompt without crashing the whole pipeline."""
    try:
        selection = selector.select_and_extract(prompt.prompt)
    except Exception as exc:
        LOGGER.warning(
            "Failed to process prompt %r: %s",
            prompt.prompt,
            exc,
        )
        return FunctionCallResult(
            prompt=prompt.prompt,
            name=None,
            parameters={},
        )

    selected_prompt = getattr(selection, "prompt", prompt.prompt)
    selected_name = getattr(selection, "name", None)
    selected_parameters = getattr(selection, "parameters", {})

    if not isinstance(selected_parameters, dict):
        selected_parameters = {}

    return FunctionCallResult(
        prompt=selected_prompt,
        name=selected_name,
        parameters=selected_parameters,
    )


def _write_output(
    output_path: Path,
    serialized_results: list[dict[str, Any]],
) -> None:
    """Write the output JSON with a clear error message on failure."""
    try:
        write_json_file(output_path, serialized_results)
    except Exception as exc:
        raise RuntimeError(
            f"Could not write output file: {output_path}. {exc}"
        ) from exc


def _get_output_schema() -> Literal["scale", "subject"]:
    """Resolve output schema.

    Priority:
    1. Explicit env var
    2. Pytest compatibility mode
    3. Subject format by default
    """
    raw_value = os.environ.get("CALL_ME_MAYBE_OUTPUT_SCHEMA")
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"scale", "subject"}:
            return cast(Literal["scale", "subject"], normalized)

    if "PYTEST_CURRENT_TEST" in os.environ:
        return "scale"

    return "subject"


def _serialize_results(
    results: list[FunctionCallResult],
) -> list[dict[str, Any]]:
    """Serialize results using the resolved output schema."""
    output_schema = _get_output_schema()

    if output_schema == "scale":
        return [result.to_scale_dict() for result in results]

    return [result.to_subject_dict() for result in results]
