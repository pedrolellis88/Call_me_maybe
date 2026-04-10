from __future__ import annotations

import os
from typing import Any

from src.models.selection_result import SelectionResult


class FunctionSelector:
    """Select a function call and validate extracted parameters."""

    def __init__(
        self,
        functions: list[dict[str, Any]],
        decoder: Any | None = None,
    ) -> None:
        self.functions = functions
        self.function_map = self._build_function_map(functions)
        self.debug_enabled = self._is_debug_enabled()

        if decoder is not None:
            self.decoder = decoder
        else:
            from src.llm.constrained_decoder import ConstrainedDecoder

            self.decoder = ConstrainedDecoder()

    def _is_debug_enabled(self) -> bool:
        """Return whether selector debug logging is enabled."""
        raw_value = os.environ.get("CALL_ME_MAYBE_DEBUG_SELECTOR", "0")
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}

    def _debug(self, message: str) -> None:
        """Print a debug message when selector debugging is enabled."""
        if self.debug_enabled:
            print(f"[FUNCTION_SELECTOR] {message}")

    def _build_function_map(
        self,
        functions: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Build a validated function-name map."""
        function_map: dict[str, dict[str, Any]] = {}

        for function in functions:
            if not isinstance(function, dict):
                continue

            raw_name = function.get("name")
            if not isinstance(raw_name, str):
                continue

            name = raw_name.strip()
            if not name:
                continue

            if name in function_map:
                raise ValueError(f"Duplicate function name in schema: {name}")

            function_map[name] = function

        return function_map

    def _call_decoder(self, prompt: str) -> Any:
        """Call the decoder using the first compatible public method."""
        candidate_methods = (
            "generate_call",
            "decode_function_call",
            "select_and_extract",
            "decode",
            "generate",
        )

        for method_name in candidate_methods:
            method = getattr(self.decoder, method_name, None)
            if callable(method):
                return method(prompt, self.functions)

        raise AttributeError(
            "Decoder does not expose a compatible call method"
        )

    def _normalize_decoder_output(
        self,
        result: Any,
    ) -> tuple[str | None, dict[str, Any], str | None]:
        """
        Normalize decoder output to the internal shape.

        Returns:
            tuple[name, parameters, error]
        """
        if not isinstance(result, dict):
            return None, {}, "Decoder returned a non-dict result"

        raw_name = result.get("name")
        if raw_name is None:
            raw_name = result.get("fn_name")
        if raw_name is None:
            raw_name = result.get("function")

        raw_parameters = result.get("parameters")
        if raw_parameters is None:
            raw_parameters = result.get("args")
        if raw_parameters is None:
            raw_parameters = result.get("arguments")
        if raw_parameters is None:
            raw_parameters = {}

        raw_error = result.get("error")
        if isinstance(raw_error, str):
            return None, {}, raw_error

        if raw_name is not None and not isinstance(raw_name, str):
            return None, {}, "Decoder returned an invalid function name"

        if not isinstance(raw_parameters, dict):
            return None, {}, "Decoder returned invalid parameters"

        normalized_name = (
            raw_name.strip() if isinstance(raw_name, str) else None
        )
        if normalized_name == "":
            normalized_name = None

        return normalized_name, raw_parameters, None

    def _coerce_parameter_value(
        self,
        value: Any,
        expected_type: str,
    ) -> tuple[Any, str | None]:
        """Coerce a parameter value to the schema-declared type."""
        normalized_type = expected_type.strip().lower()

        if normalized_type in {"string", "str", "text"}:
            if isinstance(value, str):
                return value, None
            return str(value), None

        if normalized_type in {"number", "float"}:
            if isinstance(value, bool):
                return None, "Boolean is not a valid number"
            if isinstance(value, (int, float)):
                return float(value), None
            if isinstance(value, str):
                stripped = value.strip()
                try:
                    return float(stripped), None
                except ValueError:
                    return None, f"Invalid numeric value: {value!r}"
            return None, f"Invalid numeric value: {value!r}"

        if normalized_type in {"int", "integer"}:
            if isinstance(value, bool):
                return None, "Boolean is not a valid integer"
            if isinstance(value, int):
                return value, None
            if isinstance(value, float):
                if value.is_integer():
                    return int(value), None
                return None, f"Non-integer numeric value: {value!r}"
            if isinstance(value, str):
                stripped = value.strip()
                try:
                    parsed = float(stripped)
                except ValueError:
                    return None, f"Invalid integer value: {value!r}"

                if not parsed.is_integer():
                    return None, f"Non-integer numeric value: {value!r}"

                return int(parsed), None
            return None, f"Invalid integer value: {value!r}"

        if normalized_type in {"boolean", "bool"}:
            if isinstance(value, bool):
                return value, None
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {
                    "true",
                    "1",
                    "yes",
                    "on",
                    "enable",
                    "enabled",
                }:
                    return True, None
                if lowered in {
                    "false",
                    "0",
                    "no",
                    "off",
                    "disable",
                    "disabled",
                }:
                    return False, None
            return None, f"Invalid boolean value: {value!r}"

        return value, None

    def _validate_enum(
        self,
        value: Any,
        spec: dict[str, Any],
    ) -> str | None:
        """Validate enum membership when present."""
        enum_values = spec.get("enum")
        if not isinstance(enum_values, list):
            return None

        if value not in enum_values:
            return (
                f"Value {value!r} is not one of the allowed enum values: "
                f"{enum_values!r}"
            )

        return None

    def _is_required_parameter(
        self,
        parameter_name: str,
        spec: Any,
    ) -> bool:
        """Return whether a parameter should be treated as required."""
        if isinstance(spec, str):
            _ = parameter_name
            return True

        if not isinstance(spec, dict):
            return True

        raw_required = spec.get("required")
        if isinstance(raw_required, bool):
            return raw_required

        if "default" in spec:
            return False

        _ = parameter_name
        return True

    def _normalize_parameter_spec(self, spec: Any) -> dict[str, Any] | None:
        """Normalize a parameter spec into dict form."""
        if isinstance(spec, dict):
            return spec

        if isinstance(spec, str):
            return {"type": spec}

        return None

    def _validate_parameters(
        self,
        function_schema: dict[str, Any],
        parameters: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        """Keep expected parameters, coerce types, validate enum, and ensure required ones."""  # noqa
        expected_parameters = function_schema.get("parameters", {})

        if not isinstance(expected_parameters, dict):
            return {}, "Function schema has invalid parameters definition"

        cleaned_parameters: dict[str, Any] = {}

        for key, raw_spec in expected_parameters.items():
            if key not in parameters:
                continue

            spec = self._normalize_parameter_spec(raw_spec)
            if spec is None:
                return {}, f"Invalid schema for parameter: {key}"

            raw_type = spec.get("type")
            if not isinstance(raw_type, str):
                return {}, f"Missing or invalid type for parameter: {key}"

            coerced_value, coerce_error = self._coerce_parameter_value(
                parameters[key],
                raw_type,
            )
            if coerce_error is not None:
                return {}, f"Invalid value for parameter {key}: {coerce_error}"

            enum_error = self._validate_enum(coerced_value, spec)
            if enum_error is not None:
                return {}, f"Invalid value for parameter {key}: {enum_error}"

            cleaned_parameters[key] = coerced_value

        missing_required = [
            key
            for key, spec in expected_parameters.items()
            if self._is_required_parameter(key, spec)
            and key not in cleaned_parameters
        ]
        if missing_required:
            return {}, (
                "Missing required parameters: " + ", ".join(missing_required)
            )

        return cleaned_parameters, None

    def select_and_extract(self, prompt: str) -> SelectionResult:
        """Run the decoder, validate the chosen function, and clean parameters."""  # noqa
        try:
            self._debug(f"prompt={prompt!r}")
            result = self._call_decoder(prompt)
            self._debug(f"raw_decoder_output={result!r}")

            name, parameters, normalize_error = self._normalize_decoder_output(
                result
            )
            if normalize_error is not None:
                self._debug(f"normalize_error={normalize_error}")
                return SelectionResult(
                    prompt=prompt,
                    name=None,
                    parameters={},
                    error=normalize_error,
                )

            if name is None:
                self._debug("decoder did not determine a function")
                return SelectionResult(
                    prompt=prompt,
                    name=None,
                    parameters={},
                    error=None,
                )

            if name not in self.function_map:
                self._debug(f"unknown_function={name!r}")
                return SelectionResult(
                    prompt=prompt,
                    name=None,
                    parameters={},
                    error=f"Decoder selected unknown function: {name}",
                )

            validated_parameters, validation_error = self._validate_parameters(
                self.function_map[name],
                parameters,
            )
            if validation_error is not None:
                self._debug(
                    "validation_error="
                    f"{validation_error}; name={name!r}; parameters={parameters!r}"  # noqa
                )
                return SelectionResult(
                    prompt=prompt,
                    name=None,
                    parameters={},
                    error=validation_error,
                )

            self._debug(
                f"ok name={name!r}; validated_parameters={validated_parameters!r}"  # noqa
            )
            return SelectionResult(
                prompt=prompt,
                name=name,
                parameters=validated_parameters,
            )

        except Exception as exc:
            self._debug(f"exception={exc!r}")
            return SelectionResult(
                prompt=prompt,
                name=None,
                parameters={},
                error=str(exc),
            )
