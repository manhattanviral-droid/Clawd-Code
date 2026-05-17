"""DeepSeek provider implementation.

DeepSeek exposes an OpenAI-compatible Chat Completions API at
https://api.deepseek.com, so we reuse the OpenAI SDK with a custom base URL.

Supported models:
  - deepseek-chat      (V3.1, hybrid reasoning, tool/function calling)
  - deepseek-reasoner  (R1, chain-of-thought; does NOT support tool calling)
"""

from __future__ import annotations

from typing import Optional

from .openai_provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek provider — OpenAI-compatible endpoint at api.deepseek.com."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url or "https://api.deepseek.com",
            model=model or "deepseek-chat",
        )

    def get_available_models(self) -> list[str]:
        return [
            "deepseek-chat",       # V3.1 hybrid, supports tool calling
            "deepseek-reasoner",   # R1 chain-of-thought, NO tool calling
        ]
