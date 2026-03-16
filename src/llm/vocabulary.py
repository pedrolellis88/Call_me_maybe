import json
from typing import Dict

from huggingface_hub import hf_hub_download


MODEL_NAME = "Qwen/Qwen3-0.6B"


def load_vocabulary() -> Dict[int, str]:
    """Load token_id -> token_string mapping without loading the full model."""
    vocab_path = hf_hub_download(
        repo_id=MODEL_NAME,
        filename="vocab.json",
    )

    with open(vocab_path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    id_to_token: Dict[int, str] = {}

    if isinstance(raw, dict):
        for token, idx in raw.items():
            try:
                id_to_token[int(idx)] = str(token)
            except (ValueError, TypeError):
                continue
        return id_to_token

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                idx = item.get("id")
                tok = item.get("token")
                if isinstance(idx, int) and isinstance(tok, str):
                    id_to_token[idx] = tok
        return id_to_token

    raise ValueError("Unsupported vocabulary format.")
