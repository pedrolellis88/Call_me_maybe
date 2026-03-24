from typing import List, Sequence

from llm_sdk import Small_LLM_Model


class LLMClient:
    """Wrapper around the provided LLM SDK."""

    def __init__(self) -> None:
        self.model = Small_LLM_Model()
        self._encode_cache: dict[str, List[int]] = {}

    def encode(self, text: str) -> List[int]:
        """Encode text into a flat list of token ids."""
        cached = self._encode_cache.get(text)
        if cached is not None:
            return cached.copy()

        encoded = self.model.encode(text)

        if hasattr(encoded, "tolist"):
            data = encoded.tolist()
            if isinstance(data, list) and data and isinstance(data[0], list):
                result = [int(token_id) for token_id in data[0]]
                self._encode_cache[text] = result
                return result.copy()
            if isinstance(data, list):
                result = [int(token_id) for token_id in data]
                self._encode_cache[text] = result
                return result.copy()

        raise TypeError("Unexpected encode() return type from llm_sdk.")

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token ids into text."""
        return str(self.model.decode(list(token_ids)))

    def get_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        """Return next-token logits for the given input ids."""
        return self.model.get_logits_from_input_ids(list(input_ids))

    def get_vocab_file_path(self) -> str:
        """Return the path to the vocab file."""
        return str(self.model.get_path_to_vocab_file())

    def get_merges_file_path(self) -> str:
        """Return the path to the merges file."""
        return str(self.model.get_path_to_merges_file())

    def get_tokenizer_file_path(self) -> str:
        """Return the path to the tokenizer file."""
        return str(self.model.get_path_to_tokenizer_file())
