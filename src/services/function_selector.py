# src/services/function_selector.py

from __future__ import annotations

import os
import re
from typing import Any

from src.models.selection_result import SelectionResult
from src.services.parameter_extractor import ParameterExtractor


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
        self.parameter_extractor = ParameterExtractor()

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

    def _normalize_parameter_spec(
        self,
        spec: Any,
    ) -> dict[str, Any] | None:
        """Normalize a parameter spec into dict form."""
        if isinstance(spec, dict):
            return spec

        if isinstance(spec, str):
            return {"type": spec}

        return None

    def _extract_parameter_types(
        self,
        function_schema: dict[str, Any],
    ) -> dict[str, str]:
        """Build a simple name -> type mapping for extraction."""
        expected_parameters = function_schema.get("parameters", {})
        if not isinstance(expected_parameters, dict):
            return {}

        parameter_types: dict[str, str] = {}

        for key, raw_spec in expected_parameters.items():
            spec = self._normalize_parameter_spec(raw_spec)
            if spec is None:
                continue

            raw_type = spec.get("type")
            if isinstance(raw_type, str) and raw_type.strip():
                parameter_types[key] = raw_type.strip()

        return parameter_types

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

    def _tokenize_text(self, value: str) -> set[str]:
        """Tokenize text into lowercase alphanumeric fragments."""
        return set(re.findall(r"[a-z0-9_]+", value.lower()))

    def _schema_text(self, function_schema: dict[str, Any]) -> str:
        """Build a searchable text blob from schema fields."""
        name = function_schema.get("name", "")
        description = function_schema.get("description", "")
        parameters = function_schema.get("parameters", {})

        parts: list[str] = []
        if isinstance(name, str):
            parts.append(name)
        if isinstance(description, str):
            parts.append(description)

        if isinstance(parameters, dict):
            parts.extend(parameters.keys())

        return " ".join(parts).lower()

    def _score_function_for_prompt(
        self,
        prompt: str,
        function_schema: dict[str, Any],
    ) -> int:
        """Compute a light lexical score for fallback selection."""
        prompt_tokens = self._tokenize_text(prompt)
        schema_text = self._schema_text(function_schema)
        schema_tokens = self._tokenize_text(schema_text)

        score = 0

        for token in prompt_tokens:
            if token in schema_tokens:
                score += 1

        if "template" in prompt.lower() and "template" in schema_text:
            score += 6

        if "sql" in prompt.lower() and "sql" in schema_text:
            score += 6

        if "query" in prompt.lower() and "query" in schema_text:
            score += 4

        if "database" in prompt.lower() and "database" in schema_text:
            score += 4

        if "read" in prompt.lower() and "read" in schema_text:
            score += 5

        if "file" in prompt.lower() and "file" in schema_text:
            score += 5

        if "encoding" in prompt.lower() and "encoding" in schema_text:
            score += 4

        return score

    def _best_scored_function(
        self,
        prompt: str,
        minimum_score: int = 1,
    ) -> str | None:
        """Pick the best lexical fallback above a minimum score."""
        best_name: str | None = None
        best_score = minimum_score - 1

        for name, function_schema in self.function_map.items():
            score = self._score_function_for_prompt(prompt, function_schema)
            if score > best_score:
                best_score = score
                best_name = name

        return best_name

    def _find_function_by_required_terms(
        self,
        required_terms: set[str],
        preferred_terms: set[str] | None = None,
    ) -> str | None:
        """Find the schema entry that best matches required/preferred terms."""
        preferred_terms = preferred_terms or set()

        best_name: str | None = None
        best_score = -1

        for name, function_schema in self.function_map.items():
            schema_text = self._schema_text(function_schema)
            schema_tokens = self._tokenize_text(schema_text)

            if not required_terms.issubset(schema_tokens):
                continue

            score = 0
            for term in required_terms:
                if term in schema_tokens:
                    score += 5
            for term in preferred_terms:
                if term in schema_tokens:
                    score += 2

            if score > best_score:
                best_score = score
                best_name = name

        return best_name

    def _select_fallback_function(
        self,
        prompt: str,
    ) -> str | None:
        """Select a fallback function when the decoder misses the intent."""
        lowered_prompt = prompt.strip().lower()

        if re.search(
            r"\b(?:execute\s+sql\s+query|run\s+the\s+query|sql\s+query)\b",
            lowered_prompt,
        ):
            sql_name = self._find_function_by_required_terms(
                required_terms={"query"},
                preferred_terms={"sql", "database", "execute"},
            )
            if sql_name is not None:
                return sql_name

        if (
            re.search(r"^\s*read\b", lowered_prompt)
            or "file at" in lowered_prompt
            or "encoding" in lowered_prompt
        ):
            read_name = self._find_function_by_required_terms(
                required_terms={"read"},
                preferred_terms={"file", "path", "encoding"},
            )
            if read_name is not None:
                return read_name

        if re.search(r"^\s*format\s+template:", lowered_prompt):
            template_name = self._find_function_by_required_terms(
                required_terms={"template"},
                preferred_terms={"format"},
            )
            if template_name is not None:
                return template_name

        return self._best_scored_function(prompt, minimum_score=3)

    def _merge_parameter_values(
        self,
        expected_type: str,
        decoder_value: Any,
        extracted_value: Any,
    ) -> Any:
        """Prefer local extraction for strings; keep decoder when appropriate."""  # noqa
        normalized_type = expected_type.strip().lower()

        if extracted_value is None:
            return decoder_value

        if decoder_value is None:
            return extracted_value

        if normalized_type in {"string", "str", "text"}:
            return extracted_value

        return decoder_value

    def _build_candidate_parameters(
        self,
        prompt: str,
        function_name: str,
        decoder_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge decoder parameters with local extraction from the prompt."""
        function_schema = self.function_map[function_name]
        parameter_types = self._extract_parameter_types(function_schema)
        extracted_parameters = self.parameter_extractor.extract_parameters(
            prompt=prompt,
            function_name=function_name,
            parameters=parameter_types,
        )

        merged_parameters: dict[str, Any] = {}

        for key, expected_type in parameter_types.items():
            decoder_value = decoder_parameters.get(key)
            extracted_value = extracted_parameters.get(key)

            merged_value = self._merge_parameter_values(
                expected_type=expected_type,
                decoder_value=decoder_value,
                extracted_value=extracted_value,
            )
            if merged_value is not None:
                merged_parameters[key] = merged_value

        return merged_parameters

    def select_and_extract(self, prompt: str) -> SelectionResult:
        """Run the decoder, validate the chosen function, and clean parameters."""  # noqa
        try:
            self._debug(f"prompt={prompt!r}")
            raw_result = self._call_decoder(prompt)
            self._debug(f"raw_decoder_output={raw_result!r}")

            decoder_name, decoder_parameters, normalize_error = (
                self._normalize_decoder_output(raw_result)
            )

            if normalize_error is not None:
                self._debug(f"normalize_error={normalize_error}")
                decoder_name = None
                decoder_parameters = {}

            candidate_name = decoder_name
            if candidate_name is not None and candidate_name not in self.function_map:  # noqa
                self._debug(f"unknown_function={candidate_name!r}")
                candidate_name = None

            if candidate_name is None:
                candidate_name = self._select_fallback_function(prompt)
                self._debug(f"fallback_function={candidate_name!r}")

            if candidate_name is None:
                return SelectionResult(
                    prompt=prompt,
                    name=None,
                    parameters={},
                    error=normalize_error,
                )

            candidate_parameters = self._build_candidate_parameters(
                prompt=prompt,
                function_name=candidate_name,
                decoder_parameters=decoder_parameters,
            )

            validated_parameters, validation_error = self._validate_parameters(
                self.function_map[candidate_name],
                candidate_parameters,
            )

            if validation_error is not None:
                self._debug(
                    "validation_error="
                    f"{validation_error}; name={candidate_name!r}; "
                    f"parameters={candidate_parameters!r}"
                )
                return SelectionResult(
                    prompt=prompt,
                    name=None,
                    parameters={},
                    error=validation_error,
                )

            self._debug(
                f"ok name={candidate_name!r}; "
                f"validated_parameters={validated_parameters!r}"
            )
            return SelectionResult(
                prompt=prompt,
                name=candidate_name,
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
