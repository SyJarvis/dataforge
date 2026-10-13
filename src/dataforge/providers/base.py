"""The narrow async interface used by synthesis recipes."""

from typing import Protocol

from dataforge.schemas import GenerationRequest, GenerationResult


class ModelProvider(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...

    async def aclose(self) -> None: ...
