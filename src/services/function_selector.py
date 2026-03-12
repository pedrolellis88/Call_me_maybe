import json
from typing import Any, Dict, List

from src.llm.client import LLMClient
from src.llm.prompt_builder import build_full_prompt, build_system_prompt
from src.models.function_definition import FunctionDefinition


class FunctionSelector:
    """Select a function call candidate from the user prompt."""

    def __init__(self) -> None:
        self.llm = LLMClient()

    def select(
        self,
        prompt: str,
        functions: List[FunctionDefinition],
    ) -> Dict[str, Any]:
        """Generate a raw function call prediction for one prompt."""
        system_prompt = build_system_prompt(functions)
        full_prompt = build_full_prompt(system_prompt, prompt)

        input_ids = self.llm.encode(full_prompt)
        logits = self.llm.get_logits(input_ids)

        # Temporary placeholder:
        # this does not generate a real answer yet.
        _ = logits
        output_tokens = input_ids
        text = self.llm.decode(output_tokens)

        try:
            parsed = json.loads(text)
            return parsed
        except Exception:
            return {
                "name": functions[0].name if functions else "undefined_function", # noqa
                "parameters": {},
            }
