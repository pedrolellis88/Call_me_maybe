import json
from pathlib import Path
from typing import Any

from src.errors import InputFileError


def read_json_file(path: Path) -> Any:
    """Read and parse a JSON file."""
    if not path.exists():
        raise InputFileError(f"File not found: {path}")

    if not path.is_file():
        raise InputFileError(f"Path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise InputFileError(
            f"Invalid JSON in file {path}: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc
    except OSError as exc:
        raise InputFileError(f"Could not read file {path}: {exc}") from exc
