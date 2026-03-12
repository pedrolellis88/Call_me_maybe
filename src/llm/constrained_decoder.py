from src.llm.client import LLMClient


class ConstrainedDecoder:
    """Generate JSON text using constrained decoding logic."""

    def __init__(self) -> None:
        self.llm = LLMClient()

    def generate_json(self, prompt: str) -> str:
        """Generate a JSON string from a prompt."""
        input_ids = self.llm.encode(prompt)
        generated = []

        for _ in range(200):
            logits = self.llm.get_logits(input_ids + generated)
            next_token = logits.argmax().item()
            generated.append(next_token)

            partial_text = self.llm.decode(generated)
            if partial_text.endswith("}"):
                break

        return self.llm.decode(generated)
