from typing import Any

from src.models.selection_result import SelectionResult


class FunctionSelector:
    def __init__(
        self,
        functions: list[dict[str, Any]],
        decoder: Any | None = None,
    ) -> None:
        self.functions = functions

        if decoder is not None:
            self.decoder = decoder
        else:
            from src.llm.constrained_decoder import ConstrainedDecoder

            self.decoder = ConstrainedDecoder()

    def select_and_extract(self, prompt: str) -> SelectionResult:
        try:
            result = self.decoder.generate_call(prompt, self.functions)

            if not isinstance(result, dict):
                return SelectionResult(
                    prompt=prompt,
                    name=None,
                    parameters={},
                    error="Decoder returned a non-dict result",
                )

            name = result.get("name")
            parameters = result.get("parameters", {})

            if name is not None and not isinstance(name, str):
                return SelectionResult(
                    prompt=prompt,
                    name=None,
                    parameters={},
                    error="Decoder returned an invalid function name",
                )

            if not isinstance(parameters, dict):
                return SelectionResult(
                    prompt=prompt,
                    name=None,
                    parameters={},
                    error="Decoder returned invalid parameters",
                )

            return SelectionResult(
                prompt=prompt,
                name=name,
                parameters=parameters if name is not None else {},
            )

        except Exception as exc:
            return SelectionResult(
                prompt=prompt,
                name=None,
                parameters={},
                error=str(exc),
            )
