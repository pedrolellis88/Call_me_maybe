from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import (
    DEFAULT_FUNCTIONS_DEFINITION,
    DEFAULT_INPUT_FILE,
    DEFAULT_OUTPUT_FILE,
)
from src.errors import ProjectError
from src.services.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="call-me-maybe",
        description="Translate natural language prompts into function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to the output JSON file.",
    )
    return parser


def main() -> int:
    """Run the CLI and return the process exit code."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_pipeline(
            functions_definition_path=args.functions_definition,
            input_path=args.input,
            output_path=args.output,
        )
    except KeyboardInterrupt:
        print("Execution interrupted by user.", file=sys.stderr)
        return 130
    except (ProjectError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    return 0
