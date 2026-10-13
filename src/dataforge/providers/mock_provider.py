"""Deterministic, network-free fixtures for exercising the pipeline."""

import hashlib

from dataforge.schemas import GenerationRequest, GenerationResult


class MockProvider:
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        digest = hashlib.sha256(request.model_dump_json().encode()).hexdigest()[:12]
        return GenerationResult(
            text=f"[MOCK 演示数据 {digest}] 这是用于验证流水线的占位回答，不可用于模型训练。",
            status="completed",
            request_id=f"mock-{digest}",
            raw={"provider": "mock", "synthetic_demo": True},
        )

    async def aclose(self) -> None:
        pass
