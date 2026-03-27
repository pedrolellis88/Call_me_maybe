import json
from pathlib import Path
from typing import Any

from src.errors import OutputFileError


def write_json_file(path: Path, data: Any) -> None:
    """Write JSON data to disk."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    except OSError as exc:
        message = f"Could not write output file {path}: {exc}"
        raise OutputFileError(message) from exc
