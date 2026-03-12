from typing import Any, Dict

from src.models.function_definition import FunctionDefinition


class ArgumentExtractor:
    """Validate and convert extracted parameters."""

    def extract(
        self,
        raw_output: Dict[str, Any],
        function: FunctionDefinition,
    ) -> Dict[str, Any]:
        """Extract and cast parameters based on the function schema."""
        params = raw_output.get("parameters", {})
        validated: Dict[str, Any] = {}

        for name, definition in function.parameters.items():
            if name not in params:
                raise ValueError(f"Missing parameter: {name}")

            value = params[name]

            if definition.type == "number":
                validated[name] = float(value)
            elif definition.type == "string":
                validated[name] = str(value)
            elif definition.type == "boolean":
                validated[name] = bool(value)
            else:
                validated[name] = value

        return validated
