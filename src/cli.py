import argparse
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
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_pipeline(
            functions_definition_path=args.functions_definition,
            input_path=args.input,
            output_path=args.output,
        )
    except ProjectError as exc:
        print(f"Error: {exc}")
        return 1

    print("Done.")
    return 0
