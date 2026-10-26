"""Provider construction; credentials are read only by the chosen official SDK."""

from pathlib import Path

from dataforge.config import ModelConfig
from dataforge.providers.base import ModelProvider


def create_provider(config: ModelConfig, media_root: Path) -> ModelProvider:
    if config.provider == "openai":
        from dataforge.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(config, media_root)
    if config.provider == "anthropic":
        from dataforge.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(config, media_root)
    from dataforge.providers.mock_provider import MockProvider

    return MockProvider(final_answer="42" if config.model == "offline-rlvr-demo" else None)
