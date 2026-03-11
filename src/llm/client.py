class LLMClient:
    """Thin wrapper around the provided SDK."""

    def __init__(self) -> None:
        from llm_sdk import Small_LLM_Model
        self.model = Small_LLM_Model()

    def encode(self, text: str) -> Any:
        """Encode text into tokens."""
        return self.model.encode(text)

    def decode(self, ids)-> Any:
        """Decode tokens into text."""
        return self.model.decode(ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]: # noqa
        """Return next-token logits."""
        return self.model.get_logits_from_input_ids(input_ids)

    def get_vocab_file_path(self) -> str:
        """Return path to the vocab file."""
        return self.model.get_path_to_vocab_file()

    def get_merges_file_path(self) -> str:
        """Return path to the merges file."""
        return self.model.get_path_to_merges_file()

    def get_tokenizer_file_path(self) -> str:
        """Return path to the tokenizer file."""
        return self.model.get_path_to_tokenizer_file()
