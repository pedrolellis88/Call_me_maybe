from pathlib import Path

from src.llm.client import LLMClient


def get_tokenizer_paths() -> dict[str, Path]:
    """Collect tokenizer-related file paths from the SDK."""
    client = LLMClient()
    return {
        "vocab": Path(client.get_vocab_file_path()),
        "merges": Path(client.get_merges_file_path()),
        "tokenizer": Path(client.get_tokenizer_file_path()),
    }
