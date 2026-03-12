from typing import List

from src.models.function_definition import FunctionDefinition


class SchemaValidator:
    """Provide helpers to validate function schemas."""

    @staticmethod
    def find_function(
        name: str,
        functions: List[FunctionDefinition],
    ) -> FunctionDefinition:
        """Find a function definition by its name."""
        for fn in functions:
            if fn.name == name:
                return fn

        raise ValueError(f"Function not found: {name}")
