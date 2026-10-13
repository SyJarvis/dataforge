"""OpenAI Responses API adapter."""

from pathlib import Path

from openai import AsyncOpenAI

from dataforge.config import ModelConfig
from dataforge.datasets.media import image_data
from dataforge.schemas import GenerationRequest, GenerationResult


class OpenAIProvider:
    def __init__(
        self,
        config: ModelConfig,
        media_root: Path,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.config = config
        self.media_root = media_root
        self.client = client or AsyncOpenAI(
            timeout=config.timeout_seconds, max_retries=config.max_retries
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        messages = []
        for message in request.messages:
            if message.role == "assistant":
                # EasyInputMessage accepts prior assistant text as a string.
                content = "".join(block.text for block in message.content)
            else:
                content = []
                for block in message.content:
                    if block.type == "text":
                        content.append({"type": "input_text", "text": block.text})
                    else:
                        if not self.config.vision:
                            raise ValueError("image input requires model.vision=true")
                        mime, encoded = image_data(block.uri, self.media_root)
                        content.append(
                            {
                                "type": "input_image",
                                "image_url": f"data:{mime};base64,{encoded}",
                                "detail": "auto",
                            }
                        )
            messages.append({"role": message.role, "content": content})

        kwargs = {
            "model": self.config.model,
            "input": messages,
            "max_output_tokens": self.config.max_output_tokens,
            "store": False,
        }
        if request.instructions:
            kwargs["instructions"] = request.instructions
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        response = await self.client.responses.create(**kwargs)

        text_parts = []
        refused = False
        complete = response.status == "completed"
        for item in response.output:
            if item.type == "message":
                complete = complete and item.status == "completed"
                for block in item.content:
                    if block.type == "output_text":
                        text_parts.append(block.text)
                    elif block.type == "refusal":
                        refused = True
            elif item.type != "reasoning":
                # Tool calls and other non-text outputs are not SFT completions.
                complete = False
        status = "refused" if refused else "completed" if complete else "incomplete"
        usage = response.usage
        return GenerationResult(
            text="".join(text_parts),
            status=status,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            request_id=response._request_id,
            raw=response.model_dump(mode="json"),
        )

    async def aclose(self) -> None:
        await self.client.close()
