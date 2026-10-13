"""Anthropic Messages API adapter."""

from pathlib import Path

from anthropic import AsyncAnthropic

from dataforge.config import ModelConfig
from dataforge.datasets.media import image_data
from dataforge.schemas import GenerationRequest, GenerationResult


class AnthropicProvider:
    def __init__(
        self,
        config: ModelConfig,
        media_root: Path,
        *,
        client: AsyncAnthropic | None = None,
    ) -> None:
        if config.temperature is not None:
            raise ValueError("the supported Anthropic SDK does not accept temperature")
        self.config = config
        self.media_root = media_root
        self.client = client or AsyncAnthropic(
            timeout=config.timeout_seconds, max_retries=config.max_retries
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        system_parts = [request.instructions] if request.instructions else []
        messages = []
        for message in request.messages:
            if message.role == "system":
                system_parts.extend(block.text for block in message.content)
                continue
            content = []
            for block in message.content:
                if block.type == "text":
                    content.append({"type": "text", "text": block.text})
                else:
                    if not self.config.vision:
                        raise ValueError("image input requires model.vision=true")
                    mime, encoded = image_data(block.uri, self.media_root)
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": encoded,
                            },
                        }
                    )
            messages.append({"role": message.role, "content": content})

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_output_tokens,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        response = await self.client.messages.create(**kwargs)
        status = "incomplete"
        if response.stop_reason == "refusal":
            status = "refused"
        elif response.stop_reason in {"end_turn", "stop_sequence"} and all(
            block.type in {"text", "thinking", "redacted_thinking"} for block in response.content
        ):
            status = "completed"
        return GenerationResult(
            text="".join(block.text for block in response.content if block.type == "text"),
            status=status,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            request_id=response._request_id,
            raw=response.model_dump(mode="json"),
        )

    async def aclose(self) -> None:
        await self.client.close()
