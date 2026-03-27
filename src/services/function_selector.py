from typing import Any, Optional


class FunctionSelector:
    def __init__(
        self,
        functions: list[dict[str, Any]],
        decoder: Optional[Any] = None,
    ) -> None:
        self.functions = functions

        if decoder is not None:
            self.decoder = decoder
        else:
            from src.llm.constrained_decoder import ConstrainedDecoder

            self.decoder = ConstrainedDecoder()

    def select_and_extract(self, prompt: str) -> dict[str, Any]:
        try:
            result = self.decoder.decode(prompt)

            return {
                "prompt": prompt,
                "name": result.get("name"),
                "parameters": result.get("parameters", {}),
            }

        except Exception as exc:
            return {
                "prompt": prompt,
                "name": None,
                "parameters": {},
                "error": str(exc),
            }
