import re
from typing import Any, List, Optional, Tuple


NUMBER_IN_TEXT = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
SINGLE_QUOTED_TEXT = re.compile(r"'([^']*)'")
DOUBLE_QUOTED_TEXT = re.compile(r'"([^"]*)"')

INVALID_NAME_WORDS = {
    "me",
    "you",
    "him",
    "her",
    "them",
    "us",
    "please",
    "now",
    "for",
    "to",
}


class ParameterExtractor:
    """Extract parameter values directly from the raw prompt when possible."""

    def extract_parameter(
        self,
        prompt: str,
        function_name: str,
        parameter_name: str,
        parameter_type: str,
    ) -> Optional[Any]:
        """Try to extract one parameter value deterministically from the prompt."""  # noqa: E501
        lower_prompt = prompt.lower()

        if parameter_type == "number":
            return self._extract_number_parameter(  # noqa: E501
                prompt, function_name, parameter_name)

        if parameter_type == "boolean":
            return self._extract_boolean_parameter(prompt)

        if parameter_type == "string":
            return self._extract_string_parameter(
                prompt=prompt,
                lower_prompt=lower_prompt,
                function_name=function_name,
                parameter_name=parameter_name,
            )

        return None

    def can_fallback_to_llm(
        self,
        prompt: str,
        function_name: str,
        parameter_name: str,
        parameter_type: str,
    ) -> bool:
        """
        Allow LLM fallback only when the prompt contains minimum evidence
        that this parameter really exists in the user request.
        """
        lower_prompt = prompt.lower()
        quoted_values = self._extract_quoted_strings(prompt)  # noqa: E501
        numeric_values = [match.group(0)
                          for match in NUMBER_IN_TEXT.finditer(prompt)]

        if parameter_type == "number":
            if function_name == "fn_add_numbers":
                if parameter_name == "a":
                    return len(numeric_values) >= 1
                if parameter_name == "b":
                    return len(numeric_values) >= 2
                return len(numeric_values) >= 2

            if function_name == "fn_get_square_root":
                return len(numeric_values) >= 1

            return len(numeric_values) >= 1

        if parameter_type == "boolean":
            return bool(re.search(r"\btrue\b|\bfalse\b", lower_prompt))

        if parameter_type == "string":
            if function_name == "fn_greet" and parameter_name == "name":
                return (
                    self._extract_greet_name(prompt) is not None
                    or len(quoted_values) >= 1
                )

            if function_name == "fn_reverse_string" and parameter_name == "s":
                if quoted_values:
                    return True
                return (  # noqa: E501
                    re.search(
                        r"(?i)\breverse(?:\s+the\s+string)?\s+([A-Za-z0-9_\-]+)\s*$",  # noqa: E501
                        prompt,
                    ) is not None)

            if function_name == "fn_substitute_string_with_regex":
                return self._can_fallback_substitute(
                    prompt=prompt,
                    lower_prompt=lower_prompt,
                    parameter_name=parameter_name,
                    quoted_values=quoted_values,
                )

            return len(quoted_values) >= 1

        return False

    def _extract_number_parameter(
        self,
        prompt: str,
        function_name: str,
        parameter_name: str,
    ) -> Optional[float]:
        values = [float(match.group(0))  # noqa: E501
                  for match in NUMBER_IN_TEXT.finditer(prompt)]
        if not values:
            return None

        if function_name == "fn_add_numbers":
            if parameter_name == "a" and len(values) >= 1:
                return values[0]
            if parameter_name == "b" and len(values) >= 2:
                return values[1]
            return None

        if function_name == "fn_get_square_root" and parameter_name == "a":
            return values[0]

        if len(values) == 1:
            return values[0]

        return None

    def _extract_boolean_parameter(self, prompt: str) -> Optional[bool]:
        lower_prompt = prompt.lower()

        if re.search(r"\btrue\b", lower_prompt):
            return True
        if re.search(r"\bfalse\b", lower_prompt):
            return False

        return None

    def _extract_string_parameter(
        self,
        prompt: str,
        lower_prompt: str,
        function_name: str,
        parameter_name: str,
    ) -> Optional[str]:
        quoted_values = self._extract_quoted_strings(prompt)

        if function_name == "fn_greet" and parameter_name == "name":
            direct_name = self._extract_greet_name(prompt)
            if direct_name is not None:
                return direct_name

            if quoted_values:
                return quoted_values[0]

        if function_name == "fn_reverse_string" and parameter_name == "s":
            if quoted_values:
                return quoted_values[0]

            reverse_match = re.search(
                r"(?i)\breverse(?:\s+the\s+string)?\s+([A-Za-z0-9_\-]+)\s*$",
                prompt,
            )
            if reverse_match:
                return reverse_match.group(1)

        if function_name == "fn_substitute_string_with_regex":
            return self._extract_substitute_parameter(
                prompt=prompt,
                lower_prompt=lower_prompt,
                parameter_name=parameter_name,
                quoted_values=quoted_values,
            )

        if len(quoted_values) == 1:
            return quoted_values[0]

        return None

    def _extract_greet_name(self, prompt: str) -> Optional[str]:
        """Extract a short literal name for greeting-style prompts."""
        patterns = [
            r"(?i)^\s*greet\s+([A-Za-z][A-Za-z0-9_\-]*)\s*$",
            r"(?i)^\s*say\s+hello\s+to\s+([A-Za-z][A-Za-z0-9_\-]*)\s*$",
            r"(?i)^\s*say\s+hi\s+to\s+([A-Za-z][A-Za-z0-9_\-]*)\s*$",
            r"(?i)^\s*hello\s+([A-Za-z][A-Za-z0-9_\-]*)\s*$",
            r"(?i)^\s*hi\s+([A-Za-z][A-Za-z0-9_\-]*)\s*$",
        ]

        for pattern in patterns:
            match = re.fullmatch(pattern, prompt)
            if match:
                value = match.group(1).strip()
                if value.lower() in INVALID_NAME_WORDS:
                    return None
                return value

        loose_patterns = [
            r"(?i)\bgreet\s+([A-Za-z][A-Za-z0-9_\-]*)\b",
            r"(?i)\bsay\s+hello\s+to\s+([A-Za-z][A-Za-z0-9_\-]*)\b",
            r"(?i)\bsay\s+hi\s+to\s+([A-Za-z][A-Za-z0-9_\-]*)\b",
        ]

        for pattern in loose_patterns:
            match = re.search(pattern, prompt)
            if match:
                value = match.group(1).strip()
                if value.lower() in INVALID_NAME_WORDS:
                    return None
                return value

        return None

    def _extract_substitute_parameter(
        self,
        prompt: str,
        lower_prompt: str,
        parameter_name: str,
        quoted_values: List[str],
    ) -> Optional[str]:
        if "replace all numbers in" in lower_prompt:
            source_match = DOUBLE_QUOTED_TEXT.search(prompt)
            if parameter_name == "source_string" and source_match:
                return source_match.group(1)

            if parameter_name == "regex":
                return r"\d+"
    # noqa: E501
            replacement_match = re.search(
                r"\bwith\s+([A-Za-z0-9_\-*]+)\s*$", prompt)
            if parameter_name == "replacement" and replacement_match:
                return replacement_match.group(1)

        if "replace all vowels in" in lower_prompt:
            if parameter_name == "source_string" and quoted_values:
                return quoted_values[0]
            if parameter_name == "regex":
                return r"[AEIOUaeiou]"
            if parameter_name == "replacement" and "asterisk" in lower_prompt:
                return "*"

        if "substitute the word" in lower_prompt:
            if parameter_name == "regex" and len(quoted_values) >= 1:
                return quoted_values[0]
            if parameter_name == "replacement" and len(quoted_values) >= 2:
                return quoted_values[1]
            if parameter_name == "source_string" and len(quoted_values) >= 3:
                return quoted_values[2]

        if len(quoted_values) >= 3:
            if parameter_name == "regex":
                return quoted_values[0]
            if parameter_name == "replacement":
                return quoted_values[1]
            if parameter_name == "source_string":
                return quoted_values[2]

        return None

    def _can_fallback_substitute(
        self,
        prompt: str,
        lower_prompt: str,
        parameter_name: str,
        quoted_values: List[str],
    ) -> bool:
        if "replace all numbers in" in lower_prompt:  # noqa: E501
            source_match = DOUBLE_QUOTED_TEXT.search(prompt)
            replacement_match = re.search(
                r"\bwith\s+([A-Za-z0-9_\-*]+)\s*$", prompt)

            if parameter_name == "source_string":
                return source_match is not None
            if parameter_name == "regex":
                return source_match is not None
            if parameter_name == "replacement":
                return replacement_match is not None
            return source_match is not None and replacement_match is not None

        if "replace all vowels in" in lower_prompt:
            if parameter_name == "source_string":
                return len(quoted_values) >= 1
            if parameter_name == "regex":
                return len(quoted_values) >= 1
            if parameter_name == "replacement":
                return "asterisk" in lower_prompt
            return len(quoted_values) >= 1 and "asterisk" in lower_prompt

        if "substitute the word" in lower_prompt:
            if parameter_name == "regex":
                return len(quoted_values) >= 1
            if parameter_name == "replacement":
                return len(quoted_values) >= 2
            if parameter_name == "source_string":
                return len(quoted_values) >= 3
            return len(quoted_values) >= 3

        if parameter_name == "regex":
            return len(quoted_values) >= 1
        if parameter_name == "replacement":
            return len(quoted_values) >= 2
        if parameter_name == "source_string":
            return len(quoted_values) >= 3

        return len(quoted_values) >= 3  # noqa: E501

    def _extract_quoted_strings(self, prompt: str) -> List[str]:
        single = [match.group(1)
                  for match in SINGLE_QUOTED_TEXT.finditer(prompt)]
        double = [match.group(1)
                  for match in DOUBLE_QUOTED_TEXT.finditer(prompt)]

        if single and not double:
            return single

        if double and not single:
            return double

        if single and double:
            ordered: List[Tuple[int, str]] = []
            for match in SINGLE_QUOTED_TEXT.finditer(prompt):
                ordered.append((match.start(), match.group(1)))
            for match in DOUBLE_QUOTED_TEXT.finditer(prompt):
                ordered.append((match.start(), match.group(1)))
            ordered.sort(key=lambda item: item[0])
            return [value for _, value in ordered]

        return []
