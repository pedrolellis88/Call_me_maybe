from pathlib import Path
import getpass
import os
from typing import List, Sequence

from src.errors import ProjectError

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _pick_shared_hf_cache() -> Path:
    """Pick the best Hugging Face cache location."""
    user = os.environ.get("USER") or getpass.getuser() or "student"
    goinfre_dir = Path("/goinfre") / user

    if goinfre_dir.exists() and os.access(goinfre_dir, os.W_OK):
        return goinfre_dir / ".cache" / "huggingface"

    return Path.home() / ".cache" / "huggingface"


def _configure_hf_cache() -> None:
    """Configure cache paths and create a safe symlink for HF cache."""
    target_cache = _pick_shared_hf_cache()
    home_cache = Path.home() / ".cache" / "huggingface"
    tmp_dir = target_cache / "tmp"

    target_cache.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    home_cache.parent.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(target_cache)
    os.environ["HF_HUB_CACHE"] = str(target_cache / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(target_cache / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(target_cache / "transformers")
    os.environ["HF_ASSETS_CACHE"] = str(target_cache / "assets")
    os.environ["XDG_CACHE_HOME"] = str(target_cache.parent)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    if home_cache == target_cache:
        return

    if home_cache.is_symlink():
        try:
            if home_cache.resolve() == target_cache.resolve():
                return
        except OSError:
            pass
        home_cache.unlink()
        home_cache.symlink_to(target_cache, target_is_directory=True)
        return

    if not home_cache.exists():
        home_cache.symlink_to(target_cache, target_is_directory=True)


class LLMClient:
    """Wrapper around the provided LLM SDK."""

    def __init__(self) -> None:
        _configure_hf_cache()

        try:
            from llm_sdk import Small_LLM_Model

            self.model = Small_LLM_Model()
        except Exception as exc:
            raise ProjectError(
                "Could not initialize the Qwen/Qwen3-0.6B model. "
                "Check disk space, cache permissions, and local environment."
            ) from exc

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
        token_list = list(input_ids)

        if torch is not None:
            with torch.inference_mode():
                return self.model.get_logits_from_input_ids(token_list)

        return self.model.get_logits_from_input_ids(token_list)

    def get_vocab_file_path(self) -> str:
        """Return the path to the vocab file."""
        return str(self.model.get_path_to_vocab_file())

    def get_merges_file_path(self) -> str:
        """Return the path to the merges file."""
        return str(self.model.get_path_to_merges_file())

    def get_tokenizer_file_path(self) -> str:
        """Return the path to the tokenizer file."""
        return str(self.model.get_path_to_tokenizer_file())
