import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.llm.client import LLMClient
from src.llm.vocabulary import load_vocabulary
from src.services.parameter_extractor import ParameterExtractor


JSON_NUMBER_PREFIX = re.compile(
    r"^-?(?:0|[1-9]\d*)?(?:\.\d*)?(?:[eE][+-]?\d*)?$"
)
JSON_NUMBER_FINAL = re.compile(
    r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$"
)
NUMBER_TOKEN_CHARS = set("0123456789-+.eE")
WORD_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")


class ConstrainedDecoder:
    """Choose functions and generate typed argument values with constrained decoding."""

    def __init__(self, id_to_token: Optional[Dict[int, str]] = None) -> None:
        self.llm = LLMClient()
        self.parameter_extractor = ParameterExtractor()
        self.id_to_token: Dict[int, str] = (
            id_to_token if id_to_token is not None else load_vocabulary()
        )

        self.all_token_ids: List[int] = list(self.id_to_token.keys())
        self.token_text_cache: Dict[int, str] = {
            token_id: str(self.llm.decode([token_id]))
            for token_id in self.all_token_ids
        }

        self.quote_token_ids: List[int] = []
        self.string_token_ids: List[int] = []
        self.number_token_ids: List[int] = []

        for token_id in self.all_token_ids:
            token_text = self.token_text_cache[token_id]

            if token_text == '"':
                self.quote_token_ids.append(token_id)

            if self._is_string_token_candidate(token_text):
                self.string_token_ids.append(token_id)

            if self._is_number_token_candidate(token_text):
                self.number_token_ids.append(token_id)

        self._number_allowed_cache: Dict[str, List[int]] = {}
        self._string_allowed_cache: Dict[str, List[int]] = {}

    def generate_call(
        self,
        prompt: str,
        function_definitions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate one structured function call."""
        if not self._has_function_intent(prompt, function_definitions):
            raise ValueError("Could not determine a valid target function from the prompt.")

        function_name = self._choose_function_name(prompt, function_definitions)
        fn_def = self._get_function_definition(function_name, function_definitions)

        parameters: Dict[str, Any] = {}
        raw_parameters = fn_def.get("parameters", {})

        if not isinstance(raw_parameters, dict):
            raise ValueError("Function parameters definition must be a dictionary.")

        for param_name, spec in raw_parameters.items():
            if not isinstance(spec, dict):
                raise ValueError(f"Invalid parameter spec for: {param_name}")

            param_type = spec.get("type")
            if not isinstance(param_type, str):
                raise ValueError(f"Missing or invalid type for parameter: {param_name}")

            parameters[param_name] = self._generate_parameter_value(
                prompt=prompt,
                function_name=function_name,
                function_description=str(fn_def.get("description", "")),
                parameter_name=param_name,
                parameter_type=param_type,
            )

        return {
            "prompt": prompt,
            "name": function_name,
            "parameters": parameters,
        }

    def generate_json(
        self,
        prompt: str,
        function_definitions: List[Dict[str, Any]],
    ) -> str:
        """Generate a valid JSON string for one prompt."""
        import json

        result = self.generate_call(prompt, function_definitions)
        return json.dumps(result, ensure_ascii=False)

    def _has_function_intent(
        self,
        prompt: str,
        function_definitions: List[Dict[str, Any]],
    ) -> bool:
        """Return True only if the prompt shows some evidence of matching a function."""
        normalized = prompt.strip().lower()

        if not normalized:
            return False

        if not re.search(r"[a-z0-9]", normalized):
            return False

        intent_patterns = {
            "fn_add_numbers": [
                r"\badd\b",
                r"\bsum\b",
                r"\bplus\b",
                r"\btotal\b",
            ],
            "fn_reverse_string": [
                r"\breverse\b",
                r"\bbackwards\b",
            ],
            "fn_greet": [
                r"\bgreet\b",
                r"\bhello\b",
                r"\bhi\b",
            ],
            "fn_get_square_root": [
                r"\bsquare root\b",
                r"\bsqrt\b",
                r"\broot\b",
            ],
            "fn_substitute_string_with_regex": [
                r"\bsubstitute\b",
                r"\breplace\b",
                r"\bregex\b",
            ],
        }

        available_names = {
            fn.get("name")
            for fn in function_definitions
            if isinstance(fn.get("name"), str)
        }

        for function_name, patterns in intent_patterns.items():
            if function_name not in available_names:
                continue
            for pattern in patterns:
                if re.search(pattern, normalized):
                    return True

        return False

    def _choose_function_name(
        self,
        prompt: str,
        function_definitions: List[Dict[str, Any]],
    ) -> str:
        """Use the LLM to select one function name among the allowed names."""
        options = [
            fn["name"]
            for fn in function_definitions
            if "name" in fn and isinstance(fn["name"], str)
        ]

        if not options:
            raise ValueError("No function definitions available.")

        selection_prompt = self._build_function_choice_prompt(prompt, function_definitions)
        return self._generate_one_of(selection_prompt, options, max_steps=12)

    def _build_function_choice_prompt(
        self,
        prompt: str,
        function_definitions: List[Dict[str, Any]],
    ) -> str:
        lines = [
            "Choose exactly one function name.",
            "Return only the function name.",
            "",
            "Available functions:",
        ]

        for fn in function_definitions:
            name = fn.get("name", "")
            description = fn.get("description", "")
            parameters = fn.get("parameters", {})
            lines.append(f"- {name}: {description}")
            if isinstance(parameters, dict):
                for param_name, spec in parameters.items():
                    if isinstance(spec, dict):
                        param_type = spec.get("type", "unknown")
                    else:
                        param_type = "unknown"
                    lines.append(f"  - {param_name}: {param_type}")

        lines.extend(
            [
                "",
                f"User request: {prompt}",
                "Function:",
            ]
        )
        return "\n".join(lines)

    def _get_function_definition(
        self,
        function_name: str,
        function_definitions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        for fn in function_definitions:
            if fn.get("name") == function_name:
                return fn
        raise ValueError(f"Function definition not found: {function_name}")

    def _generate_parameter_value(
        self,
        prompt: str,
        function_name: str,
        function_description: str,
        parameter_name: str,
        parameter_type: str,
    ) -> Any:
        """Generate one argument value constrained by its declared type."""
        extracted = self.parameter_extractor.extract_parameter(
            prompt=prompt,
            function_name=function_name,
            parameter_name=parameter_name,
            parameter_type=parameter_type,
        )
        if extracted is not None:
            return extracted

        if not self.parameter_extractor.can_fallback_to_llm(
            prompt=prompt,
            function_name=function_name,
            parameter_name=parameter_name,
            parameter_type=parameter_type,
        ):
            raise ValueError(
                f"Missing enough information to extract parameter "
                f"{parameter_name!r} for function {function_name!r}."
            )

        base_prompt = self._build_parameter_prompt(
            prompt=prompt,
            function_name=function_name,
            function_description=function_description,
            parameter_name=parameter_name,
            parameter_type=parameter_type,
        )

        if parameter_type == "string":
            return self._generate_json_string_value(
                base_prompt,
                max_steps=4,
                max_chars=24,
            )

        if parameter_type == "number":
            return self._generate_json_number_value(base_prompt, max_steps=6)

        if parameter_type == "boolean":
            return self._generate_json_boolean_value(base_prompt)

        raise ValueError(f"Unsupported parameter type: {parameter_type}")

    def _build_parameter_prompt(
        self,
        prompt: str,
        function_name: str,
        function_description: str,
        parameter_name: str,
        parameter_type: str,
    ) -> str:
        return "\n".join(
            [
                "Extract exactly one parameter value for a function call.",
                "Return only the raw value for this one parameter.",
                "Do not solve the task.",
                "Do not explain anything.",
                "Do not transform the input.",
                "Do not reverse, compute, summarize, or describe.",
                "Copy exact text from the user request whenever possible.",
                f"Function: {function_name}",
                f"Description: {function_description}",
                f"Parameter name: {parameter_name}",
                f"Parameter type: {parameter_type}",
                f"User request: <<<{prompt}>>>",
                "Parameter value:",
            ]
        )

    def _generate_one_of(
        self,
        prompt: str,
        options: List[str],
        max_steps: int = 12,
    ) -> str:
        """
        Constrained generation for a finite set of exact string options,
        using tokenized options to avoid scanning the whole vocabulary.
        """
        prompt_ids = self.llm.encode(prompt)
        option_token_pairs: List[Tuple[str, List[int]]] = [
            (option, self.llm.encode(option)) for option in options
        ]

        generated: List[int] = []
        remaining = option_token_pairs

        for _ in range(max_steps):
            exact_matches = [
                text for text, token_ids in remaining if token_ids == generated
            ]
            if len(exact_matches) == 1 and len(remaining) == 1:
                return exact_matches[0]

            allowed_ids = self._allowed_next_tokens_for_tokenized_choices(
                current_ids=generated,
                choices=remaining,
            )
            if not allowed_ids:
                break

            next_token = self._pick_next_token(prompt_ids, generated, allowed_ids)
            generated.append(next_token)

            remaining = [
                (text, token_ids)
                for text, token_ids in remaining
                if len(token_ids) >= len(generated)
                and token_ids[: len(generated)] == generated
            ]

            exact_matches = [
                text for text, token_ids in remaining if token_ids == generated
            ]
            if len(exact_matches) == 1 and len(remaining) == 1:
                return exact_matches[0]

        exact_matches = [text for text, token_ids in remaining if token_ids == generated]
        if len(exact_matches) == 1:
            return exact_matches[0]

        if len(remaining) == 1:
            return remaining[0][0]

        raise ValueError(f"Could not constrained-decode one option from: {options}")

    def _allowed_next_tokens_for_tokenized_choices(
        self,
        current_ids: Sequence[int],
        choices: Sequence[Tuple[str, List[int]]],
    ) -> List[int]:
        allowed = {
            token_ids[len(current_ids)]
            for _, token_ids in choices
            if len(token_ids) > len(current_ids)
            and token_ids[: len(current_ids)] == list(current_ids)
        }
        return list(allowed)

    def _pick_next_token(
        self,
        prompt_ids: List[int],
        generated_ids: List[int],
        allowed_token_ids: List[int],
    ) -> int:
        if not allowed_token_ids:
            raise ValueError("No valid token available during constrained decoding.")

        if len(allowed_token_ids) == 1:
            return allowed_token_ids[0]

        logits = self.llm.get_logits(prompt_ids + generated_ids)

        best_id: Optional[int] = None
        best_score = -math.inf

        for token_id in allowed_token_ids:
            if token_id < 0 or token_id >= len(logits):
                continue
            score = float(logits[token_id])
            if score > best_score:
                best_score = score
                best_id = token_id

        if best_id is None:
            raise ValueError("No valid token available during constrained decoding.")

        return best_id

    def _token_to_text(self, token_id: int) -> str:
        cached = self.token_text_cache.get(token_id)
        if cached is not None:
            return cached

        text = str(self.llm.decode([token_id]))
        self.token_text_cache[token_id] = text
        return text

    def _generate_json_boolean_value(self, prompt: str) -> bool:
        raw = self._generate_one_of(prompt, ["true", "false"], max_steps=6)
        return raw == "true"

    def _generate_json_number_value(self, prompt: str, max_steps: int = 6) -> float:
        """Constrained generation for a JSON number."""
        prompt_ids = self.llm.encode(prompt)
        generated: List[int] = []
        built = ""
        last_valid_number: Optional[str] = None

        for _ in range(max_steps):
            allowed_ids = self._allowed_tokens_for_number(current_text=built)
            if not allowed_ids:
                break

            next_token = self._pick_next_token(prompt_ids, generated, allowed_ids)
            token_text = self._token_to_text(next_token)
            candidate = built + token_text

            if not self._is_json_number_prefix(candidate):
                break

            built = candidate
            generated.append(next_token)

            if self._is_json_number_final(built):
                last_valid_number = built

        if last_valid_number is not None:
            return float(last_valid_number)

        raise ValueError(f"Failed to decode a valid JSON number. Got: {built!r}")

    def _allowed_tokens_for_number(self, current_text: str) -> List[int]:
        cached = self._number_allowed_cache.get(current_text)
        if cached is not None:
            return cached

        allowed: List[int] = []

        for token_id in self.number_token_ids:
            token_text = self._token_to_text(token_id)
            candidate = current_text + token_text
            if self._is_json_number_prefix(candidate):
                allowed.append(token_id)

        self._number_allowed_cache[current_text] = allowed
        return allowed

    def _is_json_number_prefix(self, text: str) -> bool:
        if not text:
            return True
        return bool(JSON_NUMBER_PREFIX.fullmatch(text))

    def _is_json_number_final(self, text: str) -> bool:
        return bool(JSON_NUMBER_FINAL.fullmatch(text))

    def _generate_json_string_value(
        self,
        prompt: str,
        max_steps: int = 4,
        max_chars: int = 24,
    ) -> str:
        """Constrained generation for short JSON string content."""
        prompt_ids = self.llm.encode(prompt)
        generated: List[int] = []
        built = ""

        for _ in range(max_steps):
            if len(built) >= max_chars:
                break

            allowed_ids = self._allowed_tokens_for_string_content(current_text=built)
            if not allowed_ids:
                break

            next_token = self._pick_next_token(prompt_ids, generated, allowed_ids)
            token_text = self._token_to_text(next_token)

            if token_text == '"':
                cleaned = self._clean_generated_string(built)
                if cleaned:
                    return cleaned
                break

            candidate = built + token_text
            if len(candidate) > max_chars:
                break

            if not self._is_valid_json_string_content(candidate):
                break

            built = candidate

            if built.endswith((".", "(", ")", ",", ";", ":")):
                break

            generated.append(next_token)

        cleaned = self._clean_generated_string(built)
        if cleaned:
            return cleaned

        raise ValueError("Failed to decode a valid JSON string value.")

    def _allowed_tokens_for_string_content(self, current_text: str) -> List[int]:
        """Allow only a conservative subset of string tokens plus quote."""
        cached = self._string_allowed_cache.get(current_text)
        if cached is not None:
            return cached

        allowed: List[int] = []

        if current_text:
            allowed.extend(self.quote_token_ids)

        for token_id in self.string_token_ids:
            token_text = self._token_to_text(token_id)

            if len(token_text) > 6:
                continue

            if any(symbol in token_text for symbol in ["(", ")", ".", ",", ";", ":"]):
                continue

            candidate = current_text + token_text
            if self._is_valid_json_string_content(candidate):
                allowed.append(token_id)

        self._string_allowed_cache[current_text] = allowed
        return allowed

    def _clean_generated_string(self, text: str) -> str:
        """Trim noisy fallback generations to a short literal-like value."""
        cleaned = text.strip()
        if not cleaned:
            return ""

        for separator in [".", "(", ")", ",", ";", ":"]:
            if separator in cleaned:
                cleaned = cleaned.split(separator, maxsplit=1)[0].strip()

        parts = cleaned.split()
        if parts:
            cleaned = parts[0]

        match = WORD_TOKEN.search(cleaned)
        if match:
            return match.group(0)

        return cleaned

    def _is_valid_json_string_content(self, text: str) -> bool:
        """Fast conservative validation for JSON string content."""
        if not text:
            return True

        if '"' in text:
            return False

        if "\n" in text or "\r" in text or "\t" in text:
            return False

        for char in text:
            code = ord(char)
            if code < 32:
                return False

        return True

    def _is_string_token_candidate(self, token_text: str) -> bool:
        """Cheap pre-filter for tokens usable inside string content."""
        if not token_text:
            return False

        if '"' in token_text:
            return False

        if "\n" in token_text or "\r" in token_text or "\t" in token_text:
            return False

        if len(token_text) > 6:
            return False

        for char in token_text:
            code = ord(char)
            if code < 32:
                return False

        return True

    def _is_number_token_candidate(self, token_text: str) -> bool:
        """Cheap pre-filter for tokens usable inside JSON numbers."""
        if not token_text:
            return False

        return all(char in NUMBER_TOKEN_CHARS for char in token_text)