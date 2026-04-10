from __future__ import annotations

import os
import re
from collections.abc import Sequence as SequenceABC
from typing import Any, Protocol, Sequence, cast

try:
    from src.llm.client import LLMClient
except ImportError:  # pragma: no cover
    from llm.client import LLMClient  # type: ignore


class SupportsLLM(Protocol):
    """Public protocol used from the LLM wrapper."""

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token ids into text."""

    def get_logits(
        self,
        input_ids: Sequence[int],
    ) -> Sequence[float] | Sequence[Sequence[float]]:
        """Return next-token logits for the given input ids."""


class _TokenTrieNode:
    """Trie node used for constrained choice decoding."""

    def __init__(self) -> None:
        self.children: dict[int, _TokenTrieNode] = {}
        self.value: str | None = None


class _TokenTrie:
    """Trie over token-id sequences."""

    def __init__(self) -> None:
        self.root = _TokenTrieNode()

    def insert(self, token_ids: Sequence[int], value: str) -> None:
        """Insert a token sequence mapped to a final value."""
        node = self.root
        for token_id in token_ids:
            if token_id not in node.children:
                node.children[token_id] = _TokenTrieNode()
            node = node.children[token_id]
        node.value = value


class ConstrainedDecoder:
    """Select a function and extract schema-compliant arguments."""

    NO_FUNCTION = "NO_FUNCTION"

    _FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
    _QUOTED_RE = re.compile(r'"([^"]*)"|\'([^\']*)\'')
    _WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_]+")

    def __init__(
        self,
        llm_client: SupportsLLM | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Create a decoder compatible with the project wiring."""
        del args, kwargs
        self._llm = llm_client or self._try_build_default_llm()
        self._debug_enabled = self._is_debug_enabled()

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _is_debug_enabled(self) -> bool:
        """Return whether decoder debug logging is enabled."""
        raw_value = os.environ.get("CALL_ME_MAYBE_DEBUG_DECODER", "0")
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}

    def _debug(self, message: str) -> None:
        """Print a debug message when enabled."""
        if self._debug_enabled:
            print(f"[CONSTRAINED_DECODER] {message}")

    # ------------------------------------------------------------------
    # Public compatibility surface
    # ------------------------------------------------------------------

    def generate_call(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del args, kwargs
        return self._decode_impl(prompt, functions)

    def decode(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del args, kwargs
        return self._decode_impl(prompt, functions)

    def generate(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del args, kwargs
        return self._decode_impl(prompt, functions)

    def decode_function_call(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del args, kwargs
        return self._decode_impl(prompt, functions)

    def select_and_extract(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del args, kwargs
        return self._decode_impl(prompt, functions)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _decode_impl(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        clean_prompt = prompt.strip()
        normalized_functions = self._normalize_functions(functions)

        self._debug(f"prompt={clean_prompt!r}")

        if not clean_prompt or not normalized_functions:
            self._debug("empty prompt or no functions")
            return self._no_function_payload()

        selected = self._select_function(clean_prompt, normalized_functions)
        self._debug(
            f"selected_function={selected['name']!r}"
            if selected
            else "selected_function=None"
        )

        if selected is None:
            return self._no_function_payload()

        args = self._extract_parameters(clean_prompt, selected)
        self._debug(f"extracted_args={args!r}")

        return {
            "fn_name": str(selected["name"]),
            "args": args,
        }

    def _no_function_payload(self) -> dict[str, Any]:
        """Return the canonical no-function payload."""
        return {"fn_name": None, "args": {}}

    # ------------------------------------------------------------------
    # Function normalization
    # ------------------------------------------------------------------

    def _normalize_functions(
        self,
        functions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize raw function definitions from JSON."""
        normalized: list[dict[str, Any]] = []

        for raw_function in functions:
            if not isinstance(raw_function, dict):
                continue

            name = raw_function.get("name")
            description = raw_function.get("description", "")
            parameters = raw_function.get("parameters", {})
            returns = raw_function.get("returns", {})

            if not isinstance(name, str) or not name.strip():
                continue
            if not isinstance(description, str):
                description = ""
            if not isinstance(parameters, dict):
                parameters = {}
            if not isinstance(returns, dict):
                returns = {}

            clean_parameters: dict[str, dict[str, Any]] = {}
            for parameter_name, parameter_spec in parameters.items():
                if not isinstance(parameter_name, str):
                    continue
                if not isinstance(parameter_spec, dict):
                    continue

                clean_parameters[parameter_name] = {
                    **parameter_spec,
                    "type": self._normalize_type(
                        str(parameter_spec.get("type", "string"))
                    ),
                    "description": str(
                        parameter_spec.get("description", "")
                    ).strip(),
                }

            normalized.append(
                {
                    "name": name.strip(),
                    "description": description.strip(),
                    "parameters": clean_parameters,
                    "returns": returns,
                }
            )

        return normalized

    # ------------------------------------------------------------------
    # Function selection
    # ------------------------------------------------------------------

    def _select_function(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Select the best function for the prompt."""
        ranked = self._rank_functions(prompt, functions)
        if not ranked:
            return None

        self._debug(
            "ranking="
            + ", ".join(
                f"{fn['name']}:{score:.2f}" for score, fn in ranked[:5]
            )
        )

        best_score, best_function = ranked[0]
        if best_score <= 0.0:
            return None

        second_score = ranked[1][0] if len(ranked) > 1 else float("-inf")

        if best_score >= 3.0 and (best_score - second_score) >= 1.0:
            self._debug("using clear lexical winner")
            return best_function

        if self._llm is None:
            self._debug("no llm available, using lexical winner")
            return best_function

        llm_candidates = self._pick_llm_candidates(ranked)
        llm_function = self._select_function_with_llm(prompt, llm_candidates)

        if llm_function is not None:
            llm_score = self._score_function_lexically(
                prompt,
                llm_function,
            )
            self._debug(
                f"llm_candidate={llm_function['name']!r}, "
                f"llm_score={llm_score:.2f}"
            )

            if llm_score >= max(best_score - 0.5, 0.0):
                self._debug("using llm tie-break result")
                return llm_function

        self._debug("falling back to lexical winner")
        return best_function

    def _rank_functions(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
    ) -> list[tuple[float, dict[str, Any]]]:
        """Rank functions by lexical compatibility."""
        scored: list[tuple[float, dict[str, Any]]] = []

        for function_schema in functions:
            score = self._score_function_lexically(
                prompt,
                function_schema,
            )
            scored.append((score, function_schema))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _pick_llm_candidates(
        self,
        ranked: list[tuple[float, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Choose a small candidate set for LLM tie-breaking."""
        if not ranked:
            return []

        top_score = ranked[0][0]
        min_score = max(top_score - 1.0, 0.5)

        candidates: list[dict[str, Any]] = []
        for score, function_schema in ranked:
            if score >= min_score:
                candidates.append(function_schema)
            if len(candidates) >= 5:
                break

        if not candidates and ranked:
            candidates.append(ranked[0][1])

        return candidates

    def _select_function_with_llm(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Use constrained choice over candidate function names."""
        if self._llm is None or not functions:
            return None

        choices = [self.NO_FUNCTION]
        name_to_function: dict[str, dict[str, Any]] = {}

        for function_schema in functions:
            name = str(function_schema["name"])
            choices.append(name)
            name_to_function[name] = function_schema

        selection_prompt = self._build_function_selection_prompt(
            prompt,
            functions,
        )

        try:
            selected_name = self._decode_choice_from_options(
                selection_prompt,
                choices,
            )
        except Exception as exc:
            self._debug(f"llm selection failed: {exc!r}")
            return None

        if selected_name == self.NO_FUNCTION:
            self._debug("llm returned NO_FUNCTION")
            return None

        return name_to_function.get(selected_name)

    def _score_function_lexically(
        self,
        prompt: str,
        function_schema: dict[str, Any],
    ) -> float:
        """Return a lexical compatibility score between prompt and schema."""
        prompt_tokens = set(self._content_tokens(prompt))
        schema_tokens = self._schema_tokens(function_schema)

        overlap = len(prompt_tokens & schema_tokens)
        score = float(overlap) * 2.0

        parameter_types = [
            str(spec.get("type", "string"))
            for spec in function_schema["parameters"].values()
        ]
        number_count = len(self._extract_number_candidates(prompt))
        quoted_count = len(self._extract_quoted_strings(prompt))

        if "number" in parameter_types and number_count > 0:
            score += 0.25
        if "string" in parameter_types and quoted_count > 0:
            score += 0.25

        return score

    def _build_function_selection_prompt(
        self,
        user_prompt: str,
        functions: list[dict[str, Any]],
    ) -> str:
        """Build the function-selection prompt for the model."""
        lines = [
            "You are a function router.",
            "Choose the single best function for the user request.",
            f"If none match, return {self.NO_FUNCTION}.",
            "Return only the function name.",
            "",
            f"User request: {user_prompt}",
            "",
            "Available functions:",
        ]

        for function_schema in functions:
            name = function_schema["name"]
            description = function_schema["description"]
            parameters = function_schema["parameters"]

            lines.append(f"- {name}: {description}")
            if parameters:
                for param_name, param_schema in parameters.items():
                    param_type = param_schema.get("type", "string")
                    param_desc = param_schema.get("description", "")
                    lines.append(
                        f"  - {param_name} ({param_type}): {param_desc}"
                    )

        lines.append("")
        lines.append("Answer:")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Parameter extraction
    # ------------------------------------------------------------------

    def _extract_parameters(
        self,
        prompt: str,
        function_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract parameters dynamically from the schema."""
        parameters = function_schema.get("parameters", {})
        if not isinstance(parameters, dict):
            return {}

        result: dict[str, Any] = {}
        used_number_indexes: set[int] = set()
        used_string_values: set[str] = set()

        piecewise_string_params = [
            parameter_name
            for parameter_name, parameter_schema in parameters.items()
            if self._normalize_type(
                str(parameter_schema.get("type", "string"))
            ) == "string"
            and self._is_piecewise_string_parameter(parameter_name)
        ]

        non_piecewise_params = [
            parameter_name
            for parameter_name in parameters.keys()
            if parameter_name not in piecewise_string_params
        ]

        for parameter_name in non_piecewise_params:
            parameter_schema = parameters[parameter_name]

            value = self._extract_single_parameter(
                prompt=prompt,
                parameter_name=parameter_name,
                parameter_schema=parameter_schema,
                used_number_indexes=used_number_indexes,
                used_string_values=used_string_values,
            )

            if value is not None:
                result[parameter_name] = value
                if isinstance(value, str):
                    used_string_values.add(value)

        if piecewise_string_params:
            compound_candidates = [
                candidate
                for candidate in self._collect_compound_string_candidates(prompt) # noqa
                if candidate not in used_string_values
            ]

            ordered_piecewise_params = self._order_piecewise_string_parameters(
                piecewise_string_params
            )

            for compound_candidate in compound_candidates:
                split_values = self._split_compound_string_value(
                    compound_candidate,
                    len(ordered_piecewise_params),
                )

                if len(split_values) != len(ordered_piecewise_params):
                    continue

                for parameter_name, piece in zip(
                    ordered_piecewise_params,
                    split_values,
                ):
                    if piece:
                        result[parameter_name] = piece
                        used_string_values.add(piece)

                break

        for parameter_name in piecewise_string_params:
            if parameter_name in result:
                continue

            parameter_schema = parameters[parameter_name]
            value = self._extract_single_parameter(
                prompt=prompt,
                parameter_name=parameter_name,
                parameter_schema=parameter_schema,
                used_number_indexes=used_number_indexes,
                used_string_values=used_string_values,
            )

            if value is not None:
                result[parameter_name] = value
                if isinstance(value, str):
                    used_string_values.add(value)

        return result

    def _extract_single_parameter(
        self,
        prompt: str,
        parameter_name: str,
        parameter_schema: dict[str, Any],
        used_number_indexes: set[int],
        used_string_values: set[str],
    ) -> Any | None:
        """Extract one parameter using schema and prompt-local candidates."""
        param_type = self._normalize_type(
            str(parameter_schema.get("type", "string"))
        )

        enum_values = parameter_schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            enum_match = self._match_enum_option(prompt, enum_values)
            if enum_match is not None:
                return enum_match

        if param_type == "number":
            number_candidates = self._collect_number_candidates(
                prompt,
                used_number_indexes,
            )
            if not number_candidates:
                return None

            chosen_index, chosen_number_value = number_candidates[0]
            used_number_indexes.add(chosen_index)
            return chosen_number_value

        if param_type == "boolean":
            lowered = prompt.lower()

            if re.search(r"\b(enable|enabled|true|yes|on)\b", lowered):
                return True

            if re.search(r"\b(disable|disabled|false|no|off)\b", lowered):
                return False

            return None

        if param_type == "string":
            string_candidates = self._collect_string_candidates(
                prompt=prompt,
                parameter_name=parameter_name,
            )
            filtered_string_candidates = [
                candidate
                for candidate in string_candidates
                if candidate not in used_string_values
            ]
            if not filtered_string_candidates:
                return None

            chosen_string_value = filtered_string_candidates[0]
            used_string_values.add(chosen_string_value)
            return chosen_string_value

        return None

    def _collect_number_candidates(
        self,
        prompt: str,
        used_number_indexes: set[int],
    ) -> list[tuple[int, float]]:
        """Return numeric candidates found explicitly in the prompt."""
        candidates: list[tuple[int, float]] = []

        for index, item in enumerate(self._extract_number_candidates(prompt)):
            if index in used_number_indexes:
                continue
            candidates.append((index, float(item["value"])))

        return candidates

    def _collect_string_candidates(
        self,
        prompt: str,
        parameter_name: str,
    ) -> list[str]:
        """Return ordered string candidates for a parameter."""
        parameter_lower = parameter_name.lower()
        candidates: list[str] = []

        quoted = self._extract_quoted_strings(prompt)
        signature_fields = self._extract_signature_fields(prompt)

        if "scale" in parameter_lower or "label" in parameter_lower:
            lowered = prompt.lower()
            for candidate in ("cold", "mild", "hot"):
                if re.search(rf"\b{candidate}\b", lowered):
                    if candidate not in candidates:
                        candidates.append(candidate)

        if "source" in parameter_lower:
            source_string = self._extract_source_string(prompt)
            if source_string is not None:
                candidates.append(source_string)

        if "old_value" in parameter_lower or (
            parameter_lower.startswith("old")
            and "value" in parameter_lower
        ):
            old_value = self._extract_old_value_string(prompt)
            if old_value is not None:
                candidates.append(old_value)

        if "replacement" in parameter_lower or "new_value" in parameter_lower:
            replacement = self._extract_replacement_string(prompt)
            if replacement is not None:
                candidates.append(replacement)

        if "regex" in parameter_lower or "pattern" in parameter_lower:
            regex_value = self._infer_regex_from_prompt(prompt)
            if regex_value is not None:
                candidates.append(regex_value)

        if "display_name" in parameter_lower:
            display_name = self._extract_handle_display_name(prompt)
            if display_name is not None:
                candidates.append(display_name)

        if "person_name" in parameter_lower and signature_fields is not None:
            candidates.append(signature_fields[0])

        if ("role" in parameter_lower or "title" in parameter_lower) and (
            signature_fields is not None
        ):
            candidates.append(signature_fields[1])

        if (
            "organization" in parameter_lower
            or "company" in parameter_lower
            or "org" in parameter_lower
        ) and signature_fields is not None:
            candidates.append(signature_fields[2])

        if (
            "name" in parameter_lower
            and "display_name" not in parameter_lower
            and not self._is_piecewise_string_parameter(parameter_name)
        ):
            name_value = self._extract_name_from_prompt(prompt)
            if name_value is not None:
                candidates.append(name_value)

        if (
            parameter_lower == "s"
            or "string" in parameter_lower
            or "text" in parameter_lower
            or "headline" in parameter_lower
            or "title" in parameter_lower
        ):
            direct_string = self._extract_main_string_from_prompt(prompt)
            if direct_string is not None:
                candidates.append(direct_string)

        for item in quoted:
            if item not in candidates:
                candidates.append(item)

        inferred = self._extract_string_candidate_from_prompt(
            prompt,
            parameter_name,
        )
        if inferred is not None and inferred not in candidates:
            candidates.append(inferred)

        return candidates

    def _dedupe_keep_order(self, values: list[str]) -> list[str]:
        """Deduplicate string values while preserving order."""
        unique_values: list[str] = []
        seen: set[str] = set()

        for value in values:
            clean_value = self._clean_candidate_text(value)
            if not clean_value:
                continue

            key = clean_value.casefold()
            if key in seen:
                continue

            seen.add(key)
            unique_values.append(clean_value)

        return unique_values

    def _collect_compound_string_candidates(
        self,
        prompt: str,
    ) -> list[str]:
        """Collect multi-word string candidates from the prompt."""
        candidates: list[str] = []

        extracted_candidates = [
            self._extract_handle_display_name(prompt),
            self._extract_name_from_prompt(prompt),
            self._extract_main_string_from_prompt(prompt),
            self._extract_source_string(prompt),
        ]

        for candidate in extracted_candidates:
            if candidate is not None and self._looks_like_compound_candidate(
                candidate
            ):
                candidates.append(candidate)

        for quoted_value in self._extract_quoted_strings(prompt):
            if self._looks_like_compound_candidate(quoted_value):
                candidates.append(quoted_value)

        generic_patterns = (
            r"\bfor\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_.-]*"
            r"(?:\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_.-]*)+)\s*$",
            r"\bfrom\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_.-]*"
            r"(?:\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_.-]*)+)\s*$",
            r"\bnamed\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_.-]*"
            r"(?:\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_.-]*)+)\s*$",
            r"\bcalled\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_.-]*"
            r"(?:\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_.-]*)+)\s*$",
        )

        for pattern in generic_patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match is not None:
                candidates.append(match.group(1))

        return self._dedupe_keep_order(candidates)

    def _looks_like_compound_candidate(self, value: str) -> bool:
        """Return whether a candidate looks like a multi-part string."""
        tokens = [token for token in value.split() if token]
        return len(tokens) >= 2

    def _is_piecewise_string_parameter(self, parameter_name: str) -> bool:
        """Return whether a string parameter likely represents one piece."""
        lowered = parameter_name.lower()
        piece_keywords = (
            "first",
            "last",
            "middle",
            "given",
            "family",
            "surname",
            "forename",
            "prefix",
            "suffix",
        )
        return any(keyword in lowered for keyword in piece_keywords)

    def _order_piecewise_string_parameters(
        self,
        parameter_names: list[str],
    ) -> list[str]:
        """Order piecewise string parameters semantically."""
        indexed_names = list(enumerate(parameter_names))
        indexed_names.sort(
            key=lambda item: self._semantic_string_parameter_rank(
                item[1],
                item[0],
            )
        )
        return [item[1] for item in indexed_names]

    def _semantic_string_parameter_rank(
        self,
        parameter_name: str,
        position: int,
    ) -> tuple[int, int]:
        """Rank piecewise string fields in a stable semantic order."""
        lowered = parameter_name.lower()

        if any(token in lowered for token in ("full", "display", "complete")):
            return (0, position)
        if any(token in lowered for token in ("first", "given", "forename")):
            return (1, position)
        if "middle" in lowered:
            return (2, position)
        if any(token in lowered for token in ("last", "family", "surname")):
            return (3, position)
        if any(token in lowered for token in ("prefix", "start", "head")):
            return (4, position)
        if any(token in lowered for token in ("suffix", "end", "tail")):
            return (5, position)

        return (6, position)

    def _split_compound_string_value(
        self,
        value: str,
        expected_parts: int,
    ) -> list[str]:
        """Split a multi-word value across multiple string parameters."""
        tokens = [token for token in value.split() if token]

        if expected_parts <= 1:
            return [value]

        if len(tokens) < expected_parts:
            return []

        if len(tokens) == expected_parts:
            return tokens

        parts: list[str] = []
        remaining_tokens = tokens[:]

        for index in range(expected_parts):
            remaining_slots = expected_parts - index
            if remaining_slots == 1:
                parts.append(" ".join(remaining_tokens))
                break
            parts.append(remaining_tokens.pop(0))

        return [part.strip() for part in parts if part.strip()]

    # ------------------------------------------------------------------
    # Constrained decoding helpers
    # ------------------------------------------------------------------

    def _decode_choice_from_options(
        self,
        prompt: str,
        options: list[str],
    ) -> str:
        """Choose one exact string from a closed set."""
        if self._llm is None:
            raise RuntimeError("LLM client is not available.")

        trie = _TokenTrie()
        for option in options:
            token_ids = self._llm.encode(option)
            trie.insert(token_ids, option)

        input_ids = list(self._llm.encode(prompt))
        node = trie.root
        generated: list[int] = []

        for _ in range(64):
            if node.value is not None:
                return node.value

            logits = self._get_last_logits(input_ids + generated)
            allowed_token_ids = list(node.children.keys())
            if not allowed_token_ids:
                break

            best_token = self._pick_best_allowed_token(
                logits,
                allowed_token_ids,
            )
            generated.append(best_token)
            node = node.children[best_token]

        if node.value is not None:
            return node.value

        raise RuntimeError("Could not decode a valid option.")

    def _pick_best_allowed_token(
        self,
        logits: Sequence[float],
        allowed_token_ids: Sequence[int],
    ) -> int:
        """Pick the highest-logit token among the allowed token ids."""
        best_token = int(allowed_token_ids[0])
        best_score = float(logits[best_token])

        for token_id in allowed_token_ids[1:]:
            score = float(logits[token_id])
            if score > best_score:
                best_score = score
                best_token = int(token_id)

        return best_token

    # ------------------------------------------------------------------
    # Primitive extraction helpers
    # ------------------------------------------------------------------

    def _extract_number_candidates(
        self,
        prompt: str,
    ) -> list[dict[str, Any]]:
        """Extract numeric candidates from the prompt."""
        candidates: list[dict[str, Any]] = []
        lowered = prompt.lower()

        quoted_spans: list[tuple[int, int]] = [
            (match.start(), match.end())
            for match in self._QUOTED_RE.finditer(prompt)
        ]

        def _is_inside_quotes(index: int) -> bool:
            for start, end in quoted_spans:
                if start <= index < end:
                    return True
            return False

        visible_index = 0
        for match in self._FLOAT_RE.finditer(prompt):
            if _is_inside_quotes(match.start()):
                continue

            raw_value = match.group(0)
            try:
                value = float(raw_value)
            except ValueError:
                continue

            start = match.start()
            end = match.end()
            left = lowered[max(0, start - 32): start]
            right = lowered[end: min(len(lowered), end + 32)]
            tokens = set(self._simple_tokens(left + " " + right))

            candidates.append(
                {
                    "index": visible_index,
                    "value": value,
                    "context_tokens": tokens,
                }
            )
            visible_index += 1

        return candidates

    def _extract_quoted_strings(self, prompt: str) -> list[str]:
        """Extract quoted string candidates from the prompt."""
        values: list[str] = []
        for match in self._QUOTED_RE.finditer(prompt):
            value = (
                match.group(1)
                if match.group(1) is not None
                else match.group(2)
            )
            if value is None:
                continue

            clean_value = value.strip()
            if clean_value:
                values.append(clean_value)

        return values

    def _extract_source_string(self, prompt: str) -> str | None:
        """Extract the source string in replacement/substitution prompts."""
        match = re.search(
            r"\bin\s+['\"]([^'\"]+)['\"]\s*$",
            prompt,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return self._clean_candidate_text(match.group(1))

        quoted = self._extract_quoted_strings(prompt)
        if quoted:
            return quoted[-1]

        return None

    def _extract_old_value_string(self, prompt: str) -> str | None:
        """Extract the old value in replacement/substitution prompts."""
        quoted = self._extract_quoted_strings(prompt)
        if len(quoted) >= 2:
            return self._clean_candidate_text(quoted[0])

        match = re.search(
            r"\breplace\s+([A-Za-z0-9_*.-]+)\s+with\b",
            prompt,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return self._clean_candidate_text(match.group(1))

        match = re.search(
            r"\bsubstitute\s+([A-Za-z0-9_*.-]+)\s+with\b",
            prompt,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return self._clean_candidate_text(match.group(1))

        return None

    def _extract_replacement_string(self, prompt: str) -> str | None:
        """Extract the replacement value from prompts like 'with X'."""
        quoted = self._extract_quoted_strings(prompt)
        if len(quoted) >= 2:
            return self._clean_candidate_text(quoted[1])

        match = re.search(
            r"\bwith\s+['\"]([^'\"]+)['\"]",
            prompt,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return self._clean_candidate_text(match.group(1))

        match = re.search(
            r"\bwith\s+([A-Za-z0-9_*.-]+)",
            prompt,
            flags=re.IGNORECASE,
        )
        if match is not None:
            value = self._clean_candidate_text(match.group(1))
            if value.lower() in {"asterisk", "asterisks"}:
                return "*"
            return value

        return None

    def _extract_name_from_prompt(self, prompt: str) -> str | None:
        """Extract a person/name-like string from the prompt."""
        patterns = (
            r"\bgreet\s+(.+)$",
            r"\bfor\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_.-]*"
            r"(?:\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_.-]*)+)\s*$",
            r"\bnamed\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_.-]*"
            r"(?:\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_.-]*)+)\s*$",
            r"\bcalled\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_.-]*"
            r"(?:\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_.-]*)+)\s*$",
        )

        for pattern in patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match is not None:
                return self._clean_name_like_text(match.group(1))

        quoted = self._extract_quoted_strings(prompt)
        if quoted:
            return self._clean_name_like_text(quoted[0])

        return None

    def _extract_handle_display_name(self, prompt: str) -> str | None:
        """Extract a display name from username/handle prompts."""
        quoted = self._extract_quoted_strings(prompt)
        if quoted:
            return quoted[0]

        patterns = (
            r"\bgenerate\s+a\s+username\s+for\s+(.+)$",
            r"\bmake\s+a\s+username\s+for\s+(.+)$",
            r"\bi\s+need\s+a\s+handle\s+for\s+(.+)$",
            r"\bcreate\s+a\s+handle\s+from\s+(.+)$",
            r"\bcreate\s+a\s+username\s+for\s+(.+)$",
            r"\bturn\s+the\s+name\s+(.+?)\s+into\s+a\s+username$",
            r"\bturn\s+(.+?)\s+into\s+a\s+username$",
            r"\b(?:username|handle)\s+for\s+(.+)$",
            r"\b(?:username|handle)\s+from\s+(.+)$",
        )

        for pattern in patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match is not None:
                return self._clean_name_like_text(match.group(1))

        return None

    def _extract_signature_fields(
        self,
        prompt: str,
    ) -> tuple[str, str, str] | None:
        """Extract person name, role, and organization from signature prompts.""" # noqa
        patterns = (
            r"\b(?:signature|footer|signature block|email signature|"
            r"business signature).*?for\s+(.+?),\s*(.+?)\s+at\s+(.+)$",
            r"\b(?:write|build|make|create|generate).*?for\s+(.+?),\s*"
            r"(.+?)\s+at\s+(.+)$",
            r"\b(.+?),\s*(.+?)\s+at\s+(.+)$",
        )

        for pattern in patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match is None:
                continue

            person_name = self._clean_name_like_text(match.group(1))
            role_label = self._clean_candidate_text(match.group(2))
            organization_name = self._clean_candidate_text(match.group(3))

            if person_name and role_label and organization_name:
                return (person_name, role_label, organization_name)

        return None

    def _extract_main_string_from_prompt(self, prompt: str) -> str | None:
        """Extract the main user-provided string argument."""
        quoted = self._extract_quoted_strings(prompt)
        if quoted:
            return quoted[0]

        match = re.search(
            r"\breverse\s+the\s+string\s+(.+)$",
            prompt,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return self._clean_candidate_text(match.group(1))

        match = re.search(
            r"\breverse\s+(.+)$",
            prompt,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return self._clean_candidate_text(match.group(1))

        return None

    def _extract_string_candidate_from_prompt(
        self,
        prompt: str,
        parameter_name: str,
    ) -> str | None:
        """Infer a bare string candidate when the prompt has no quotes."""
        parameter_lower = parameter_name.lower()

        if "scale" in parameter_lower or "label" in parameter_lower:
            lowered = prompt.lower()
            for candidate in ("cold", "mild", "hot"):
                if re.search(rf"\b{candidate}\b", lowered):
                    return candidate

        if "display_name" in parameter_lower:
            return self._extract_handle_display_name(prompt)

        if "person_name" in parameter_lower:
            fields = self._extract_signature_fields(prompt)
            if fields is not None:
                return fields[0]

        if "role" in parameter_lower or "title" in parameter_lower:
            fields = self._extract_signature_fields(prompt)
            if fields is not None:
                return fields[1]

        if (
            "organization" in parameter_lower
            or "company" in parameter_lower
            or "org" in parameter_lower
        ):
            fields = self._extract_signature_fields(prompt)
            if fields is not None:
                return fields[2]

        if "name" in parameter_lower and not self._is_piecewise_string_parameter( # noqa
            parameter_name
        ):
            return self._extract_name_from_prompt(prompt)

        if (
            parameter_lower == "s"
            or "string" in parameter_lower
            or "text" in parameter_lower
            or "headline" in parameter_lower
            or "title" in parameter_lower
        ):
            return self._extract_main_string_from_prompt(prompt)

        if "source" in parameter_lower:
            return self._extract_source_string(prompt)

        if "old_value" in parameter_lower or (
            parameter_lower.startswith("old")
            and "value" in parameter_lower
        ):
            return self._extract_old_value_string(prompt)

        if "replacement" in parameter_lower or "new_value" in parameter_lower:
            return self._extract_replacement_string(prompt)

        if "regex" in parameter_lower or "pattern" in parameter_lower:
            return self._infer_regex_from_prompt(prompt)

        return None

    def _infer_regex_from_prompt(self, prompt: str) -> str | None:
        """Infer a regex string from natural-language replacement prompts."""
        lowered = prompt.lower()

        if "numbers" in lowered or "digits" in lowered:
            return r"\d"
        if "vowels" in lowered:
            return r"[AEIOUaeiou]"

        match = re.search(
            r"\bword\s+['\"]([^'\"]+)['\"]",
            prompt,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return re.escape(match.group(1))

        return None

    def _match_enum_option(
        self,
        prompt: str,
        options: list[Any],
    ) -> Any | None:
        """Match an enum option lexically."""
        lowered = prompt.lower()

        normalized_prompt = re.sub(r"[^a-z0-9\s]+", " ", lowered)
        normalized_prompt = re.sub(r"\s+", " ", normalized_prompt).strip()

        for option in options:
            if not isinstance(option, str):
                continue

            normalized_option = option.strip().lower()
            if not normalized_option:
                continue

            if re.search(
                rf"\b{re.escape(normalized_option)}\b",
                normalized_prompt,
            ):
                return option

        return None

    def _clean_candidate_text(self, value: str) -> str:
        """Normalize a generic extracted text span."""
        cleaned = value.strip().strip(" \t\n\r\"'")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip(" \t\n\r\"'")

    def _clean_name_like_text(self, value: str) -> str:
        """Normalize name/entity-like text and remove helper words."""
        cleaned = self._clean_candidate_text(value)
        cleaned = re.sub(
            r"^(?:for|from|to|a|an|the)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip(" \t\n\r\"'")

    def _schema_tokens(self, function_schema: dict[str, Any]) -> set[str]:
        """Tokenize relevant schema fields for lexical fallback."""
        tokens: set[str] = set()
        tokens.update(
            self._content_tokens(str(function_schema.get("name", "")))
        )
        tokens.update(
            self._content_tokens(str(function_schema.get("description", "")))
        )

        parameters = function_schema.get("parameters", {})
        if isinstance(parameters, dict):
            for parameter_name, parameter_spec in parameters.items():
                tokens.update(self._content_tokens(parameter_name))

                if isinstance(parameter_spec, dict):
                    tokens.update(
                        self._content_tokens(
                            str(parameter_spec.get("description", ""))
                        )
                    )
                    tokens.add(
                        self._normalize_type(
                            str(parameter_spec.get("type", "string"))
                        )
                    )

                    enum_values = parameter_spec.get("enum")
                    if isinstance(enum_values, list):
                        for enum_value in enum_values:
                            if isinstance(enum_value, str):
                                tokens.update(self._content_tokens(enum_value))

        return tokens

    def _simple_tokens(self, text: str) -> list[str]:
        """Basic tokenizer for lexical comparisons."""
        return [
            match.group(0).lower()
            for match in self._WORD_RE.finditer(text.lower())
        ]

    def _content_tokens(self, text: str) -> list[str]:
        """Tokenizer for content-bearing tokens with light normalization."""
        stopwords = {
            "a",
            "an",
            "the",
            "this",
            "that",
            "these",
            "those",
            "is",
            "are",
            "am",
            "be",
            "to",
            "of",
            "in",
            "on",
            "at",
            "for",
            "from",
            "with",
            "and",
            "or",
            "by",
            "my",
            "me",
            "i",
            "it",
            "its",
            "do",
            "does",
            "please",
            "can",
            "could",
            "would",
            "will",
            "what",
            "as",
        }

        synonym_map = {
            "footer": "signature",
            "username": "handle",
            "headline": "title",
            "slugify": "slug",
            "substitute": "replace",
            "label": "classify",
            "mark": "classify",
            "categorized": "classify",
            "categorize": "classify",
            "c": "celsius",
        }

        tokens = self._simple_tokens(text.replace("_", " ").replace("-", " "))
        normalized_tokens: list[str] = []

        for token in tokens:
            if token in stopwords:
                continue
            normalized_tokens.append(synonym_map.get(token, token))

        return normalized_tokens

    def _normalize_type(self, raw_type: str) -> str:
        """Normalize aliased type names."""
        normalized = raw_type.strip().lower()
        aliases = {
            "str": "string",
            "text": "string",
            "float": "number",
            "int": "number",
            "integer": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }
        return aliases.get(normalized, normalized or "string")

    def _get_last_logits(
        self,
        input_ids: Sequence[int],
    ) -> Sequence[float]:
        """Get the logits vector for the next token."""
        if self._llm is None:
            raise RuntimeError("LLM client is not available.")

        logits = self._llm.get_logits(input_ids)
        if len(logits) == 0:
            return []

        first_item = logits[0]

        if isinstance(first_item, SequenceABC) and not isinstance(
            first_item,
            (str, bytes),
        ):
            last_item = logits[-1]
            if not isinstance(last_item, SequenceABC) or isinstance(
                last_item,
                (str, bytes),
            ):
                raise RuntimeError("LLM returned inconsistent logits.")

            return [
                float(value)
                for value in cast(Sequence[float], last_item)
            ]

        return [float(value) for value in cast(Sequence[float], logits)]

    def _try_build_default_llm(self) -> SupportsLLM | None:
        """Try to construct the default LLM client."""
        try:
            return LLMClient()
        except Exception:
            return None
