"""Moonshot / Kimi provider implementation.

Moonshot AI (the company behind Kimi) exposes an OpenAI-compatible Chat
Completions API. International users hit https://api.moonshot.ai/v1;
mainland China users hit https://api.moonshot.cn/v1.

Supported models (international):
  - kimi-k2-0905-preview     (Kimi K2, recommended for coding/agent work)
  - kimi-k2-turbo-preview    (faster, cheaper)
  - moonshot-v1-8k           (older, small context)
  - moonshot-v1-32k          (older, mid context)
  - moonshot-v1-128k         (older, large context)
"""

from __future__ import annotations

from typing import Optional

from .openai_provider import OpenAIProvider


class MoonshotProvider(OpenAIProvider):
    """Moonshot/Kimi provider — OpenAI-compatible endpoint at api.moonshot.ai."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url or "https://api.moonshot.ai/v1",
            model=model or "kimi-k2-0905-preview",
        )

    def get_available_models(self) -> list[str]:
        return [
            "kimi-k2-0905-preview",
            "kimi-k2-turbo-preview",
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
        ]
