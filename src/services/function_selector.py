from typing import Any, Dict, List

from src.llm.constrained_decoder import ConstrainedDecoder
from src.models.function_definition import FunctionDefinition


class FunctionSelector:
    """Select a function call candidate from the user prompt."""

    def __init__(self) -> None:
        self.decoder = ConstrainedDecoder()

    def select(
        self,
        prompt: str,
        functions: List[FunctionDefinition],
    ) -> Dict[str, Any]:
        """Generate a structured function call prediction for one prompt."""
        function_definitions = [function.model_dump() for function in functions]

        try:
            result = self.decoder.generate_call(prompt, function_definitions)
            return {
                "name": result["name"],
                "parameters": result["parameters"],
                "error": None,
            }
        except ValueError as exc:
            return {
                "name": None,
                "parameters": {},
                "error": str(exc),
            }
