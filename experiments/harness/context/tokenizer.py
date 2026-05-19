"""Provider-agnostic token counting utilities.

1. HuggingFaceTokenizer — exact counts via AutoTokenizer (any HF model)
2. TiktokenTokenizer    — exact counts via tiktoken (OpenAI BPE models)
3. SimpleTokenizer      — character-ratio estimate, zero dependencies

Use get_tokenizer() to pick the right one automatically, or instantiate
directly if you know your model family.
"""

from __future__ import annotations

import functools
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Chars-per-token reference (for SimpleTokenizer fallback)
#   BPE (OpenAI / GPT):              ~4.0
#   SentencePiece (Mistral, Llama,
#     Gemini, Cohere, Qwen, …):      ~3.5  ← default
#   WordPiece / code-heavy:          ~3.0
# ---------------------------------------------------------------------------


class Tokenizer(ABC):
    """Abstract base — all tokenizers share this interface."""

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

    def count_message_tokens(self, message: Dict[str, Any]) -> int:
        """Tokens for a single chat message dict (text + tool_calls)."""
        content = message.get("content") or ""
        if isinstance(content, list):
            # Multimodal: only count text parts
            tokens = sum(
                self.count_tokens(p.get("text", ""))
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        else:
            tokens = self.count_tokens(str(content))

        tokens += 4  # per-message role/framing overhead

        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            tokens += self.count_tokens(func.get("name", ""))
            args = func.get("arguments", "")
            tokens += self.count_tokens(
                json.dumps(args) if not isinstance(args, str) else args
            )
        return tokens

    def count_messages_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Total tokens across all messages plus a 3-token conversation primer."""
        return sum(self.count_message_tokens(m) for m in messages) + 3

    def fits_in_window(
        self,
        messages: List[Dict[str, Any]],
        context_window: int,
        reserve_output_tokens: int = 1_024,
    ) -> Tuple[bool, int, int]:
        """Return (fits, used_tokens, available_tokens)."""
        used = self.count_messages_tokens(messages)
        available = context_window - reserve_output_tokens
        return used <= available, used, available


class SimpleTokenizer(Tokenizer):
    """Character-ratio estimator — no dependencies required.

    Args:
        chars_per_token: Tune to match the model family:
            4.0 for BPE (OpenAI), 3.5 for SentencePiece (most others),
            3.0 for WordPiece / code-heavy models.
    """

    def __init__(self, chars_per_token: float = 3.5):
        self.chars_per_token = chars_per_token

    @functools.lru_cache(maxsize=1024)
    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token))


class TiktokenTokenizer(Tokenizer):
    """Exact BPE token counts via tiktoken (OpenAI models).

    Encoding selection:
      o200k_base  → gpt-4o, o1, o3, o4-*
      cl100k_base → gpt-4, gpt-3.5-turbo, text-embedding-3-*, ada-002

    Falls back to SimpleTokenizer(4.0) if tiktoken is not installed.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._enc = None
        self._fallback = SimpleTokenizer(chars_per_token=4.0)

        try:
            import tiktoken
            try:
                self._enc = tiktoken.encoding_for_model(model)
            except KeyError:
                name = (
                    "o200k_base"
                    if any(x in model.lower() for x in ("o1", "o3", "o4", "gpt-4o"))
                    else "cl100k_base"
                )
                self._enc = tiktoken.get_encoding(name)
        except ImportError:
            pass

    @functools.lru_cache(maxsize=1024)
    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text))
        return self._fallback.count_tokens(text)


class HuggingFaceTokenizer(Tokenizer):
    """Exact token counts via HuggingFace AutoTokenizer.

    Works for any model on the HuggingFace Hub or a local path:
      Mistral, Llama, Qwen, DeepSeek, Phi, Gemma, BGE, E5, Nomic,
      Jina, Voyage (many use HF tokenizers), and more.

    Args:
        model_name_or_path: HF model ID or local directory,
            e.g. "mistralai/Mistral-7B-Instruct-v0.2" or "BAAI/bge-m3".
        chars_per_token_fallback: Ratio used when the tokenizer cannot load.

    Falls back to SimpleTokenizer(chars_per_token_fallback) if
    `transformers` is not installed or the model is unavailable.
    """

    def __init__(
        self,
        model_name_or_path: str,
        chars_per_token_fallback: float = 3.5,
    ):
        self.model_name_or_path = model_name_or_path
        self._tok = None
        self._fallback = SimpleTokenizer(chars_per_token_fallback)

        try:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=False,
            )
        except Exception:
            pass  # offline, missing sentencepiece, private repo, etc.

    @functools.lru_cache(maxsize=512)
    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False))
        return self._fallback.count_tokens(text)


def get_tokenizer(model: Optional[str] = None, **kwargs: Any) -> Tokenizer:
    """Return the best tokenizer for *model*.

    Routing:
      - HuggingFace path ("org/name" or known open-source name) → HuggingFaceTokenizer
      - OpenAI / tiktoken models → TiktokenTokenizer
      - None or unknown → SimpleTokenizer

    Pass extra kwargs (e.g. chars_per_token_fallback=3.0) to the constructor.

    Examples::

        get_tokenizer("BAAI/bge-m3")
        get_tokenizer("mistralai/Mistral-7B-Instruct-v0.2")
        get_tokenizer("gpt-4o")
        get_tokenizer("text-embedding-3-small")
        get_tokenizer()                          # zero-dep fallback
    """
    if model is None:
        return SimpleTokenizer(**kwargs)

    # Explicit HuggingFace path
    if "/" in model:
        return HuggingFaceTokenizer(model, **kwargs)

    m = model.lower()

    # tiktoken / OpenAI BPE family
    if any(x in m for x in ("gpt-", "o1", "o3", "o4", "text-embedding", "davinci")):
        return TiktokenTokenizer(model)

    # Well-known open-source families — route to HuggingFace with canonical org prefix
    _HF_PREFIXES = {
        "mistral": "mistralai", "mixtral": "mistralai",
        "codestral": "mistralai", "magistral": "mistralai",
        "llama":   "meta-llama",
        "qwen":    "Qwen",
        "deepseek": "deepseek-ai",
        "phi":     "microsoft",
        "gemma":   "google",
        "bge":     "BAAI",       "e5":    "intfloat",
        "nomic":   "nomic-ai",   "jina":  "jinaai",
    }
    for prefix, org in _HF_PREFIXES.items():
        if m.startswith(prefix):
            return HuggingFaceTokenizer(f"{org}/{model}", **kwargs)

    # Unknown — generic fallback
    return SimpleTokenizer(**kwargs)