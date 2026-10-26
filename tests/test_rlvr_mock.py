import hashlib
from pathlib import Path

from dataforge.config import ModelConfig
from dataforge.providers import create_provider
from dataforge.providers.mock_provider import MockProvider
from dataforge.schemas import GenerationRequest, Message, TextBlock


def request(text):
    return GenerationRequest(messages=[Message(role="user", content=[TextBlock(text=text)])])


async def test_default_mock_preserves_existing_output():
    prompt = request("计算 6 × 7。")
    digest = hashlib.sha256(prompt.model_dump_json().encode()).hexdigest()[:12]
    provider = MockProvider()
    result = await provider.generate(prompt)
    assert result.text == (
        f"[MOCK 演示数据 {digest}] 这是用于验证流水线的占位回答，不可用于模型训练。"
    )
    assert result.request_id == f"mock-{digest}"
    assert result.raw == {"provider": "mock", "synthetic_demo": True}
    assert await provider.generate(prompt) == result
    await provider.aclose()


async def test_rlvr_demo_answer_is_fixed_without_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = create_provider(ModelConfig(provider="mock", model="offline-rlvr-demo"), Path("."))
    try:
        first = await provider.generate(request("计算 6 × 7。"))
        second = await provider.generate(request("计算 2 + 3。"))
    finally:
        await provider.aclose()
    for result in (first, second):
        assert result.text.splitlines()[-1] == "#### 42"
        assert result.text.count("####") == 1
        assert result.status == "completed"
        assert result.raw["synthetic_demo"] is True
    assert first.request_id != second.request_id


async def test_ordinary_mock_factory_does_not_add_answer():
    provider = create_provider(ModelConfig(provider="mock", model="offline-demo"), Path("."))
    try:
        result = await provider.generate(request("计算 6 × 7。"))
    finally:
        await provider.aclose()
    assert "####" not in result.text
