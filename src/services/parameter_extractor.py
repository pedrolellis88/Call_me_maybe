"""Utilities to extract function arguments from natural-language prompts."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field


class ParameterExtractor(BaseModel):
    """Extract argument values from natural-language prompts."""

    model_config = ConfigDict(extra="forbid")

    number_pattern: str = Field(
        default=r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
    )

    def extract_parameter(
        self,
        prompt: str,
        function_name: str,
        parameter_name: str,
        parameter_type: str,
    ) -> Any:
        """Extract a single parameter value from a prompt."""
        clean_prompt = prompt.strip()
        clean_name = parameter_name.strip()
        clean_type = parameter_type.strip().lower()

        if not clean_prompt or not clean_name:
            return None

        semantic_context = self._build_semantic_context(clean_prompt)

        if clean_type in {"number", "float", "int", "integer"}:
            return self._extract_number_parameter(
                prompt=clean_prompt,
                parameter_name=clean_name,
                parameter_type=clean_type,
            )

        if clean_type in {"boolean", "bool"}:
            return self._extract_bool_parameter(
                prompt=clean_prompt,
                parameter_name=clean_name,
            )

        if clean_type in {"string", "str", "text"}:
            return self._extract_string_parameter(
                prompt=clean_prompt,
                function_name=function_name,
                parameter_name=clean_name,
                context=semantic_context,
            )

        return None

    def extract_parameters(
        self,
        prompt: str,
        function_name: str,
        parameters: Dict[str, str],
    ) -> Dict[str, Any]:
        """Extract all parameters for a selected function."""
        extracted: Dict[str, Any] = {}

        for parameter_name, parameter_type in parameters.items():
            value = self.extract_parameter(
                prompt=prompt,
                function_name=function_name,
                parameter_name=parameter_name,
                parameter_type=parameter_type,
            )
            if value is not None:
                extracted[parameter_name] = value

        return extracted

    def can_fallback_to_llm(
        self,
        prompt: str,
        function_name: str,
        parameter_name: str,
        parameter_type: str,
    ) -> bool:
        """Return whether LLM fallback is acceptable."""
        _ = function_name
        lowered_type = parameter_type.strip().lower()
        lowered_name = parameter_name.strip().lower()
        clean_prompt = prompt.strip()

        if not clean_prompt:
            return False

        if lowered_type in {"string", "str", "text"}:
            return True

        if lowered_type in {"number", "float", "int", "integer"}:
            numbers = self._extract_numbers(clean_prompt)
            if not numbers:
                return False

            if self._is_second_like_parameter(lowered_name):
                return len(numbers) >= 2

            return True

        if lowered_type in {"boolean", "bool"}:
            lowered_prompt = clean_prompt.lower()

            patterns = [
                rf"{re.escape(lowered_name)}\s*(?:=|is|to)?\s*"
                r"(true|false|yes|no|on|off)",
                r"\b(true|false|yes|no|on|off)\b",
            ]
            return any(
                re.search(pattern, lowered_prompt) is not None
                for pattern in patterns
            )

        return False

    def _extract_number_parameter(
        self,
        prompt: str,
        parameter_name: str,
        parameter_type: str,
    ) -> Optional[float | int]:
        """Extract numeric parameters from the prompt."""
        numbers = self._extract_numbers(prompt)
        if not numbers:
            return None

        lowered_name = parameter_name.lower()

        indexed_value = self._pick_number_by_parameter_name(
            parameter_name=lowered_name,
            numbers=numbers,
        )

        if indexed_value is None:
            indexed_value = self._pick_number_by_prompt_semantics(
                prompt=prompt,
                parameter_name=lowered_name,
                numbers=numbers,
            )

        if indexed_value is None:
            if self._is_second_like_parameter(lowered_name):
                return None
            indexed_value = numbers[0]

        if parameter_type in {"int", "integer"}:
            return int(indexed_value)

        return float(indexed_value)

    def _extract_bool_parameter(
        self,
        prompt: str,
        parameter_name: str,
    ) -> Optional[bool]:
        """Extract boolean-like values safely."""
        lowered_prompt = prompt.lower()
        lowered_name = parameter_name.lower()

        explicit_patterns = [
            rf"{re.escape(lowered_name)}\s*(?:=|is|should be|to)?\s*"
            r"(true|false)",
            rf"{re.escape(lowered_name)}\s*(?:=|is|should be|to)?\s*"
            r"(yes|no)",
            rf"{re.escape(lowered_name)}\s*(?:=|is|should be|to)?\s*"
            r"(on|off)",
        ]

        for pattern in explicit_patterns:
            match = re.search(pattern, lowered_prompt)
            if match is not None:
                return self._coerce_bool(match.group(1))

        trailing_match = re.search(
            r"\b(true|false|yes|no|on|off)\b",
            lowered_prompt,
        )
        if trailing_match is not None:
            return self._coerce_bool(trailing_match.group(1))

        return None

    def _extract_string_parameter(
        self,
        prompt: str,
        function_name: str,
        parameter_name: str,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Extract string/text parameters from the prompt."""
        lowered_name = parameter_name.lower()
        lowered_function = function_name.lower()

        regex_role_value = self._extract_from_regex_context(
            parameter_name=lowered_name,
            context=context,
        )
        if regex_role_value is not None:
            return regex_role_value

        if self._looks_like_template_parameter(lowered_name, lowered_function):
            template_value = self._extract_template_text(prompt)
            if template_value is not None:
                return template_value

        if self._looks_like_database_parameter(lowered_name):
            database_value = self._extract_database_name(prompt)
            if database_value is not None:
                return database_value

        if self._looks_like_query_parameter(lowered_name, lowered_function):
            query_value = self._extract_sql_query(prompt)
            if query_value is not None:
                return query_value

        if self._looks_like_encoding_parameter(lowered_name):
            encoding_value = self._extract_encoding(prompt)
            if encoding_value is not None:
                return encoding_value

        if self._looks_like_path_parameter(lowered_name):
            path_value = self._extract_file_path(prompt)
            if path_value is not None:
                return path_value

        if self._looks_like_person_parameter(lowered_name):
            person_value = self._extract_person_text(prompt)
            if person_value is not None:
                return person_value

        quoted_strings = self._extract_quoted_strings(prompt)

        direct_quoted = self._pick_quoted_by_parameter_name(
            parameter_name=lowered_name,
            quoted_strings=quoted_strings,
        )
        if direct_quoted is not None:
            return direct_quoted

        reverse_value = self._extract_reverse_text(
            prompt=prompt,
            parameter_name=lowered_name,
            function_name=function_name,
        )
        if reverse_value is not None:
            return reverse_value

        tail_value = self._extract_tail_text(
            prompt=prompt,
            parameter_name=lowered_name,
        )
        if tail_value is not None:
            if self._looks_like_person_parameter(lowered_name):
                return self._clean_person_candidate(tail_value)
            return tail_value

        return None

    def _build_semantic_context(self, prompt: str) -> Dict[str, Any]:
        """Build semantic clues used by multiple extractors."""
        quoted_strings = self._extract_quoted_strings(prompt)
        lowered_prompt = prompt.lower()

        context: Dict[str, Any] = {
            "quoted_strings": quoted_strings,
            "mentions_reverse": bool(
                re.search(r"\b(reverse|flip)\b", lowered_prompt)
            ),
            "mentions_replace": bool(
                re.search(r"\b(replace|substitute)\b", lowered_prompt)
            ),
            "mentions_numbers_word": bool(
                re.search(r"\b(numbers?|digits?)\b", lowered_prompt)
            ),
            "mentions_vowels_word": bool(
                re.search(r"\bvowels?\b", lowered_prompt)
            ),
        }

        context.update(self._parse_replace_prompt(prompt))
        return context

    def _parse_replace_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse replace/substitute prompt structures."""
        result: Dict[str, Any] = {
            "replacement_input_text": None,
            "replacement_pattern": None,
            "replacement_substitute": None,
        }

        quoted_strings = self._extract_quoted_strings(prompt)
        lowered_prompt = prompt.lower()

        if not re.search(r"\b(replace|substitute)\b", lowered_prompt):
            return result

        if re.search(r"\b(numbers?|digits?)\b", lowered_prompt):
            result["replacement_pattern"] = r"\d"

        if re.search(r"\bvowels?\b", lowered_prompt):
            result["replacement_pattern"] = r"[AEIOUaeiou]"

        word_with_in_match = re.search(
            (
                r"""substitute\s+the\s+word\s+"""
                r"""(?P<q1>["'])(?P<pattern>.*?)(?P=q1)\s+with\s+"""
                r"""(?P<q2>["'])(?P<substitute>.*?)(?P=q2)\s+in\s+"""
                r"""(?P<q3>["'])(?P<input_text>.*?)(?P=q3)"""
            ),
            prompt,
            flags=re.IGNORECASE,
        )
        if word_with_in_match is not None:
            result["replacement_pattern"] = word_with_in_match.group(
                "pattern"
            )
            result["replacement_substitute"] = word_with_in_match.group(
                "substitute"
            )
            result["replacement_input_text"] = word_with_in_match.group(
                "input_text"
            )
            return result

        all_with_match = re.search(
            r"\b(?:replace|substitute)\b.*?\bwith\s+([^\s,.;]+)",
            prompt,
            flags=re.IGNORECASE,
        )
        if all_with_match is not None:
            substitute = all_with_match.group(1).strip()
            substitute = self._strip_wrapping_quotes(substitute)
            if substitute.lower() in {"asterisk", "asterisks"}:
                substitute = "*"
            result["replacement_substitute"] = substitute

        in_match = re.search(
            r"""\bin\s+(?P<quote>["'])(?P<value>.*?)(?P=quote)""",
            prompt,
            flags=re.IGNORECASE,
        )
        if in_match is not None:
            result["replacement_input_text"] = in_match.group("value")

        if len(quoted_strings) >= 3:
            if result["replacement_pattern"] is None:
                result["replacement_pattern"] = quoted_strings[0]
            if result["replacement_substitute"] is None:
                result["replacement_substitute"] = quoted_strings[1]
            if result["replacement_input_text"] is None:
                result["replacement_input_text"] = quoted_strings[2]
            return result

        if result["replacement_input_text"] is None and quoted_strings:
            result["replacement_input_text"] = quoted_strings[0]

        return result

    def _extract_from_regex_context(
        self,
        parameter_name: str,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Map semantic regex slots to schema parameter names."""
        input_text = context.get("replacement_input_text")
        pattern = context.get("replacement_pattern")
        substitute = context.get("replacement_substitute")

        if self._is_input_text_parameter(parameter_name):
            return input_text

        if self._is_pattern_parameter(parameter_name):
            return pattern

        if self._is_substitute_parameter(parameter_name):
            return substitute

        return None

    def _extract_person_text(self, prompt: str) -> Optional[str]:
        """Extract person/name-like values from greeting prompts."""
        patterns = [
            r"\b(?:can\s+you\s+)?greet\s+(.+?)(?:\s+for\s+me)?[?.!,:; ]*$",
            r"\b(?:can\s+you\s+)?say\s+hello\s+to\s+(.+?)"
            r"(?:\s+for\s+me)?[?.!,:; ]*$",
            r"\b(?:can\s+you\s+)?say\s+hi\s+to\s+(.+?)"
            r"(?:\s+for\s+me)?[?.!,:; ]*$",
            r"\bhello\s+(.+?)(?:\s+for\s+me)?[?.!,:; ]*$",
            r"\bhi\s+(.+?)(?:\s+for\s+me)?[?.!,:; ]*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match is None:
                continue

            candidate = self._clean_person_candidate(match.group(1))
            if candidate:
                return candidate

        return None

    def _extract_reverse_text(
        self,
        prompt: str,
        parameter_name: str,
        function_name: str,
    ) -> Optional[str]:
        """Handle reverse/flip prompts, including unquoted text."""
        lowered_prompt = prompt.lower()
        lowered_function = function_name.lower()

        if not re.search(r"\b(reverse|flip)\b", lowered_prompt):
            return None

        if not (
            self._parameter_looks_like_free_text(parameter_name)
            or "reverse" in lowered_function
        ):
            return None

        quoted_strings = self._extract_quoted_strings(prompt)
        if quoted_strings:
            return quoted_strings[0]

        patterns = [
            r"^\s*(?:reverse|flip)\s+the\s+string\s+(.+?)\s*$",
            r"^\s*(?:reverse|flip)\s+the\s+text\s+(.+?)\s*$",
            r"^\s*(?:reverse|flip)\s+(.+?)\s*$",
        ]

        generic_placeholders = {
            "string",
            "text",
            "the string",
            "the text",
        }

        for pattern in patterns:
            match = re.match(pattern, prompt, flags=re.IGNORECASE)
            if match is None:
                continue

            raw_value = match.group(1).strip()
            if raw_value.lower() in generic_placeholders:
                continue

            value = self._clean_free_text_candidate(raw_value)
            if not value or value.lower() in generic_placeholders:
                continue

            return value

        return None

    def _extract_tail_text(
        self,
        prompt: str,
        parameter_name: str,
    ) -> Optional[str]:
        """Fallback extraction for plain trailing text."""
        if not self._parameter_looks_like_free_text(parameter_name):
            return None

        patterns = [
            r"\b(?:say hello to|greet)\s+(.+?)\s*$",
            r"\b(?:reverse|flip)\s+(.+?)\s*$",
        ]

        generic_placeholders = {
            "string",
            "text",
            "the string",
            "the text",
        }

        for pattern in patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match is None:
                continue

            raw_value = match.group(1).strip()
            if raw_value.lower() in generic_placeholders:
                continue

            value = self._clean_free_text_candidate(raw_value)
            if value and value.lower() not in generic_placeholders:
                return value

        return None

    def _extract_template_text(self, prompt: str) -> Optional[str]:
        """Extract full template text after a 'Format template:' prefix."""
        match = re.search(
            r"^\s*format\s+template:\s*(?P<value>.+?)\s*$",
            prompt,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None

        value = match.group("value")
        if not value.strip():
            return None

        return value

    def _extract_sql_query(self, prompt: str) -> Optional[str]:
        """Extract SQL query text from prompts that mention query execution."""
        patterns = [
            r"\bsql\s+query\s+(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
            r"\bquery\s+(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match is not None:
                return match.group("value")

        return None

    def _extract_database_name(self, prompt: str) -> Optional[str]:
        """Extract database name from phrases like 'on the production database'."""  # noqa
        patterns = [
            r"\b(?:on|in)\s+the\s+([A-Za-z0-9_-]+)\s+database\b",
            r"\b(?:on|in)\s+([A-Za-z0-9_-]+)\s+database\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match is not None:
                return match.group(1)

        return None

    def _extract_encoding(self, prompt: str) -> Optional[str]:
        """Extract encoding name from phrases like 'with utf-8 encoding'."""
        match = re.search(
            r"\bwith\s+([A-Za-z0-9._-]+)\s+encoding\b",
            prompt,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None

        return match.group(1)

    def _extract_file_path(self, prompt: str) -> Optional[str]:
        """Extract file paths for Linux and Windows style prompts."""
        patterns = [
            (
                r"\bfile\s+at\s+(?P<value>.+?)\s+with\s+"
                r"[A-Za-z0-9._-]+\s+encoding\b"
            ),
            (
                r"^\s*read\s+(?P<value>.+?)\s+with\s+"
                r"[A-Za-z0-9._-]+\s+encoding\s*$"
            ),
            r"\bfile\s+at\s+(?P<value>.+?)\s*$",
            r"^\s*read\s+(?P<value>.+?)\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match is None:
                continue

            value = match.group("value").strip()
            value = self._strip_wrapping_quotes(value)
            value = value.strip(" \t\r\n,;")
            value = re.sub(r"[?.!]+$", "", value)
            if value:
                return value

        return None

    def _pick_quoted_by_parameter_name(
        self,
        parameter_name: str,
        quoted_strings: Sequence[str],
    ) -> Optional[str]:
        """Pick a quoted string using parameter-name semantics."""
        if not quoted_strings:
            return None

        if self._is_pattern_parameter(parameter_name):
            return quoted_strings[0]

        if self._is_substitute_parameter(parameter_name):
            if len(quoted_strings) >= 2:
                return quoted_strings[1]
            return quoted_strings[-1]

        if self._is_input_text_parameter(parameter_name):
            return quoted_strings[-1]

        if self._parameter_looks_like_free_text(parameter_name):
            return quoted_strings[0]

        return None

    def _pick_number_by_parameter_name(
        self,
        parameter_name: str,
        numbers: Sequence[float],
    ) -> Optional[float]:
        """Pick a number based on parameter naming conventions."""
        if self._is_first_like_parameter(parameter_name) and len(numbers) >= 1:
            return numbers[0]

        if self._is_second_like_parameter(parameter_name) and len(numbers) >= 2:  # noqa
            return numbers[1]

        if any(
            token in parameter_name
            for token in ("second", "right", "end")
        ):
            if len(numbers) >= 2:
                return numbers[1]

        if any(
            token in parameter_name
            for token in ("first", "left", "start")
        ):
            return numbers[0]

        return None

    def _pick_number_by_prompt_semantics(
        self,
        prompt: str,
        parameter_name: str,
        numbers: Sequence[float],
    ) -> Optional[float]:
        """Pick a number using prompt semantics when names are weak."""
        lowered_prompt = prompt.lower()

        if re.search(r"\b(square root|sqrt|root)\b", lowered_prompt):
            return numbers[0]

        if re.search(r"\b(add|sum|plus|product|multiply)\b", lowered_prompt):
            if len(numbers) >= 2:
                if parameter_name in {"x", "a"}:
                    return numbers[0]
                if parameter_name in {"y", "b"}:
                    return numbers[1]

        return None

    def _extract_numbers(self, prompt: str) -> List[float]:
        """Extract all numeric literals from a prompt."""
        matches = re.findall(self.number_pattern, prompt)
        numbers: List[float] = []

        for raw in matches:
            try:
                numbers.append(float(raw))
            except ValueError:
                continue

        return numbers

    def _extract_quoted_strings(self, prompt: str) -> List[str]:
        """Extract text wrapped in single or double quotes."""
        matches = re.findall(r'(["\'])(.*?)\1', prompt)
        strings: List[str] = []

        for _, value in matches:
            strings.append(value)

        return strings

    def _remove_conversational_suffixes(self, value: str) -> str:
        """Remove common conversational tails from extracted text."""
        cleaned = value.strip()

        suffix_patterns = [
            r"\s+for\s+me[?.!,:; ]*$",
            r"\s+please[?.!,:; ]*$",
            r"\s+right\s+now[?.!,:; ]*$",
            r"\s+now[?.!,:; ]*$",
            r"\s+thanks[?.!,:; ]*$",
        ]

        for pattern in suffix_patterns:
            cleaned = re.sub(
                pattern,
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

        return cleaned.strip()

    def _clean_person_candidate(self, value: str) -> str:
        """Clean person/name values extracted from casual prompts."""
        cleaned = self._clean_free_text_candidate(value)
        cleaned = self._remove_conversational_suffixes(cleaned)
        cleaned = cleaned.strip("?.!,:; ")
        return cleaned

    def _clean_free_text_candidate(self, value: str) -> str:
        """Normalize extracted free-text candidates."""
        cleaned = value.strip()
        cleaned = self._strip_articles(cleaned)
        cleaned = self._strip_wrapping_quotes(cleaned)
        cleaned = cleaned.strip(" \t\r\n")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _strip_wrapping_quotes(self, value: str) -> str:
        """Remove wrapping quotes from a value."""
        stripped = value.strip()
        if len(stripped) >= 2:
            if stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
                return stripped[1:-1].strip()
        return stripped

    def _strip_articles(self, value: str) -> str:
        """Remove common leading markers from extracted text."""
        stripped = value.strip()
        stripped = re.sub(
            r"^(?:the\s+string\s+|the\s+text\s+|string\s+|text\s+)",
            "",
            stripped,
            flags=re.IGNORECASE,
        )
        return stripped.strip()

    def _coerce_bool(self, raw: str) -> Optional[bool]:
        """Convert common textual booleans safely."""
        lowered = raw.strip().lower()

        if lowered in {"true", "yes", "on", "1"}:
            return True

        if lowered in {"false", "no", "off", "0"}:
            return False

        return None

    def _parameter_looks_like_free_text(
        self,
        parameter_name: str,
    ) -> bool:
        """Heuristic for parameters that should receive plain text."""
        if parameter_name == "s":
            return True

        return any(
            token in parameter_name
            for token in (
                "text",
                "string",
                "message",
                "input",
                "value",
                "person",
                "name",
                "word",
            )
        )

    def _looks_like_person_parameter(self, parameter_name: str) -> bool:
        """Whether the parameter likely stores a person/name value."""
        return any(
            token in parameter_name
            for token in (
                "person",
                "name",
                "recipient",
                "user",
                "contact",
            )
        )

    def _looks_like_path_parameter(self, parameter_name: str) -> bool:
        """Whether the parameter likely stores a file path."""
        return any(
            token in parameter_name
            for token in (
                "path",
                "file",
                "filepath",
                "filename",
            )
        )

    def _looks_like_encoding_parameter(self, parameter_name: str) -> bool:
        """Whether the parameter likely stores an encoding value."""
        return any(
            token in parameter_name
            for token in (
                "encoding",
                "charset",
            )
        )

    def _looks_like_database_parameter(self, parameter_name: str) -> bool:
        """Whether the parameter likely stores a database name."""
        return parameter_name in {"database", "db"}

    def _looks_like_query_parameter(
        self,
        parameter_name: str,
        function_name: str,
    ) -> bool:
        """Whether the parameter likely stores a SQL/query string."""
        _ = function_name
        return parameter_name in {"query", "sql", "statement"}

    def _looks_like_template_parameter(
        self,
        parameter_name: str,
        function_name: str,
    ) -> bool:
        """Whether the parameter likely stores a template string."""
        return (
            parameter_name == "template"
            or "template" in parameter_name
            or "template" in function_name
        )

    def _is_pattern_parameter(self, parameter_name: str) -> bool:
        """Whether the parameter name likely represents a pattern."""
        return any(
            token in parameter_name
            for token in (
                "pattern",
                "regex",
                "match",
                "target",
                "search",
                "find",
            )
        )

    def _is_substitute_parameter(self, parameter_name: str) -> bool:
        """Whether the parameter name likely represents replacement text."""
        return any(
            token in parameter_name
            for token in (
                "substitute",
                "replacement",
                "replace",
                "repl",
                "new",
                "with",
            )
        )

    def _is_input_text_parameter(self, parameter_name: str) -> bool:
        """Whether the parameter name likely represents source text."""
        return any(
            token in parameter_name
            for token in (
                "input",
                "text",
                "source",
                "content",
                "body",
                "string",
            )
        )

    def _is_first_like_parameter(self, parameter_name: str) -> bool:
        """Whether the parameter name behaves like a first operand."""
        return parameter_name in {
            "a",
            "x",
            "left",
            "start",
            "min",
            "minimum",
            "first",
            "value1",
            "number1",
            "num1",
            "operand1",
        }

    def _is_second_like_parameter(self, parameter_name: str) -> bool:
        """Whether the parameter name behaves like a second operand."""
        return parameter_name in {
            "b",
            "y",
            "right",
            "end",
            "max",
            "maximum",
            "second",
            "value2",
            "number2",
            "num2",
            "operand2",
        }
