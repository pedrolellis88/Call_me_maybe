from typing import Any, Dict

from src.models.function_definition import FunctionDefinition


class ArgumentExtractor:
    """Validate and convert extracted parameters based on a function schema."""

    def extract(
        self,
        raw_output: Dict[str, Any],
        function: FunctionDefinition,
    ) -> Dict[str, Any]:
        """
        Extract and cast parameters based on the function schema.

        Supported raw formats:
        - {"parameters": {...}}
        - {"args": {...}}

        Returns only the parameters declared in the function schema.
        Raises ValueError when required parameters are missing or invalid.
        """
        params = self._get_params_dict(raw_output)
        validated: Dict[str, Any] = {}

        for name, definition in function.parameters.items():
            if name not in params:
                raise ValueError(
                    f"Missing parameter '{name}' for function '{function.name}'"  # noqa  
                )

            raw_value = params[name]
            validated[name] = self._cast_value(
                value=raw_value,
                expected_type=definition.type,
                parameter_name=name,
                function_name=function.name,
            )

        return validated

    def _get_params_dict(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Return the parameters dictionary from supported output formats."""
        if "parameters" in raw_output:
            params = raw_output["parameters"]
        elif "args" in raw_output:
            params = raw_output["args"]
        else:
            raise ValueError(
                "Missing parameter container in model output. "
                "Expected 'parameters' or 'args'."
            )

        if not isinstance(params, dict):
            raise ValueError(
                "Invalid parameter container type. "
                "'parameters'/'args' must be a JSON object."
            )

        return params

    def _cast_value(
        self,
        value: Any,
        expected_type: str,
        parameter_name: str,
        function_name: str,
    ) -> Any:
        """Cast a raw value to the type declared in the function schema."""
        if expected_type == "number":
            return self._cast_number(
                value=value,
                parameter_name=parameter_name,
                function_name=function_name,
            )

        if expected_type == "string":
            return self._cast_string(value)

        if expected_type == "boolean":
            return self._cast_boolean(
                value=value,
                parameter_name=parameter_name,
                function_name=function_name,
            )

        return value

    def _cast_number(
        self,
        value: Any,
        parameter_name: str,
        function_name: str,
    ) -> float:
        """Convert a value to float, raising a clear error on failure."""
        if isinstance(value, bool):
            raise ValueError(
                f"Invalid number for parameter '{parameter_name}' "
                f"in function '{function_name}': booleans are not valid numbers"  # noqa  
            )

        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid number for parameter '{parameter_name}' "
                f"in function '{function_name}': {value!r}"
            ) from exc

    def _cast_string(self, value: Any) -> str:
        """Convert a value to string."""
        return str(value)

    def _cast_boolean(
        self,
        value: Any,
        parameter_name: str,
        function_name: str,
    ) -> bool:
        """Convert a value to boolean safely."""
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()

            if normalized in {"true", "1", "yes", "y"}:
                return True
            if normalized in {"false", "0", "no", "n"}:
                return False

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False

        raise ValueError(
            f"Invalid boolean for parameter '{parameter_name}' "
            f"in function '{function_name}': {value!r}"
        )
