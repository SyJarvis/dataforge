import base64
import json
from pathlib import Path

import httpx2
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from PIL import Image

from dataforge.config import ModelConfig
from dataforge.providers import create_provider
from dataforge.providers.anthropic_provider import AnthropicProvider
from dataforge.providers.openai_provider import OpenAIProvider
from dataforge.schemas import GenerationRequest, Message


def request_with_history(image_uri=None):
    content = [{"type": "text", "text": "请继续回答。"}]
    if image_uri:
        content += [
            {"type": "image", "uri": image_uri},
            {"type": "text", "text": "图中是什么颜色？"},
        ]
    return GenerationRequest(
        instructions="合成指令",
        messages=[
            Message(role="system", content=[{"type": "text", "text": "原始系统指令"}]),
            Message(role="user", content=[{"type": "text", "text": "你好"}]),
            Message(role="assistant", content=[{"type": "text", "text": "你好。"}]),
            Message(role="user", content=content),
        ],
    )


def openai_response(status="completed", content=None, output=None):
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "model": "test-model",
        "status": status,
        "output": output
        if output is not None
        else [
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": content
                or [{"type": "output_text", "text": "正确回答", "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 12,
            "output_tokens": 5,
            "total_tokens": 17,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def anthropic_response(stop_reason="end_turn", content=None):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": content or [{"type": "text", "text": "正确回答"}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 5},
    }


def sdk_client(provider, response, captured):
    def handle(request):
        captured.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json=response,
            headers={"x-request-id": "req_test", "request-id": "req_test"},
        )

    client_cls = AsyncOpenAI if provider == "openai" else AsyncAnthropic
    return client_cls(
        api_key="test-placeholder-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),
    )


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("with_image", [False, True])
async def test_real_sdk_request_conversion_and_response(provider, with_image, tmp_path):
    captured = []
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), "red").save(image_path)
    config = ModelConfig(
        provider=provider,
        model="test-model",
        vision=with_image,
        max_output_tokens=123,
    )
    response = openai_response() if provider == "openai" else anthropic_response()
    adapter_cls = OpenAIProvider if provider == "openai" else AnthropicProvider
    client = sdk_client(provider, response, captured)
    adapter = adapter_cls(config, tmp_path, client=client)
    try:
        result = await adapter.generate(request_with_history("image.png" if with_image else None))
    finally:
        await adapter.aclose()

    assert client.is_closed()
    assert result.text == "正确回答"
    assert result.status == "completed"
    assert (result.input_tokens, result.output_tokens) == (12, 5)
    assert result.request_id == "req_test"
    assert result.raw["id"] == response["id"]
    body = captured[0]
    assert body["model"] == "test-model"
    assert "temperature" not in body
    if provider == "openai":
        assert body["max_output_tokens"] == 123
        assert body["instructions"] == "合成指令"
        assert body["store"] is False
        assert body["input"][0]["content"][0]["text"] == "原始系统指令"
        assert body["input"][2] == {"role": "assistant", "content": "你好。"}
        content = body["input"][-1]["content"]
        assert content[0] == {"type": "input_text", "text": "请继续回答。"}
        if with_image:
            assert content[1]["type"] == "input_image"
            assert content[1]["image_url"].startswith("data:image/png;base64,")
            encoded = content[1]["image_url"].split(",", 1)[1]
            assert base64.b64decode(encoded) == image_path.read_bytes()
            assert content[2]["text"] == "图中是什么颜色？"
    else:
        assert body["max_tokens"] == 123
        assert body["system"] == "合成指令\n\n原始系统指令"
        assert [message["role"] for message in body["messages"]] == [
            "user",
            "assistant",
            "user",
        ]
        content = body["messages"][-1]["content"]
        assert content[0] == {"type": "text", "text": "请继续回答。"}
        if with_image:
            assert content[1]["type"] == "image"
            source = content[1]["source"]
            assert source["type"] == "base64"
            assert source["media_type"] == "image/png"
            assert base64.b64decode(source["data"]) == image_path.read_bytes()
            assert content[2]["text"] == "图中是什么颜色？"


@pytest.mark.parametrize(
    ("status", "content", "expected"),
    [
        ("incomplete", None, "incomplete"),
        ("in_progress", None, "incomplete"),
        ("failed", None, "incomplete"),
        ("completed", [{"type": "refusal", "refusal": "拒绝"}], "refused"),
    ],
)
async def test_openai_completion_status(status, content, expected, tmp_path):
    response = openai_response(status, content)
    adapter = OpenAIProvider(
        ModelConfig(provider="openai", model="test"),
        tmp_path,
        client=sdk_client("openai", response, []),
    )
    try:
        result = await adapter.generate(request_with_history())
        assert result.status == expected
        if expected == "refused":
            assert result.text == ""
    finally:
        await adapter.aclose()


async def test_openai_tool_call_is_not_completed_text(tmp_path):
    response = openai_response(
        output=[
            {
                "id": "call_1",
                "type": "function_call",
                "call_id": "a",
                "name": "f",
                "arguments": "{}",
            }
        ]
    )
    adapter = OpenAIProvider(
        ModelConfig(provider="openai", model="test"),
        tmp_path,
        client=sdk_client("openai", response, []),
    )
    try:
        result = await adapter.generate(request_with_history())
        assert result.status == "incomplete"
    finally:
        await adapter.aclose()


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("end_turn", "completed"),
        ("stop_sequence", "completed"),
        ("max_tokens", "incomplete"),
        ("tool_use", "incomplete"),
        ("pause_turn", "incomplete"),
        ("model_context_window_exceeded", "incomplete"),
        (None, "incomplete"),
        ("refusal", "refused"),
    ],
)
async def test_anthropic_completion_status(reason, expected, tmp_path):
    adapter = AnthropicProvider(
        ModelConfig(provider="anthropic", model="test"),
        tmp_path,
        client=sdk_client("anthropic", anthropic_response(reason), []),
    )
    try:
        assert (await adapter.generate(request_with_history())).status == expected
    finally:
        await adapter.aclose()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_vision_capability_rejects_before_request(provider, tmp_path):
    captured = []
    response = openai_response() if provider == "openai" else anthropic_response()
    cls = OpenAIProvider if provider == "openai" else AnthropicProvider
    adapter = cls(
        ModelConfig(provider=provider, model="test", vision=False),
        tmp_path,
        client=sdk_client(provider, response, captured),
    )
    try:
        with pytest.raises(ValueError, match="vision"):
            await adapter.generate(request_with_history("not-read.png"))
        assert captured == []
    finally:
        await adapter.aclose()


async def test_openai_temperature_is_explicitly_forwarded(tmp_path):
    captured = []
    adapter = OpenAIProvider(
        ModelConfig(provider="openai", model="test", temperature=0.2),
        tmp_path,
        client=sdk_client("openai", openai_response(), captured),
    )
    try:
        await adapter.generate(request_with_history())
        assert captured[0]["temperature"] == 0.2
    finally:
        await adapter.aclose()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_text_blocks_are_preserved_without_added_characters(provider, tmp_path):
    captured = []
    if provider == "openai":
        response = openai_response(
            content=[
                {"type": "output_text", "text": "第一", "annotations": []},
                {"type": "output_text", "text": "段\n第二段", "annotations": []},
            ]
        )
        cls = OpenAIProvider
    else:
        response = anthropic_response(
            content=[
                {"type": "text", "text": "第一"},
                {"type": "text", "text": "段\n第二段"},
            ]
        )
        cls = AnthropicProvider
    adapter = cls(
        ModelConfig(provider=provider, model="test"),
        tmp_path,
        client=sdk_client(provider, response, captured),
    )
    request = request_with_history()
    request.messages[2] = Message(
        role="assistant",
        content=[{"type": "text", "text": "你"}, {"type": "text", "text": "好。"}],
    )
    try:
        assert (await adapter.generate(request)).text == "第一段\n第二段"
        if provider == "openai":
            assert captured[0]["input"][2]["content"] == "你好。"
    finally:
        await adapter.aclose()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_factory_configures_sdk_timeout_and_retries(provider, monkeypatch, tmp_path):
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "test-placeholder-key")
    adapter = create_provider(
        ModelConfig(provider=provider, model="test", timeout_seconds=7, max_retries=0),
        tmp_path,
    )
    try:
        assert adapter.client.timeout == 7
        assert adapter.client.max_retries == 0
    finally:
        await adapter.aclose()


async def test_mock_is_deterministic_and_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = create_provider(ModelConfig(provider="mock", model="demo"), Path("."))
    first = await provider.generate(request_with_history())
    second = await provider.generate(request_with_history())
    assert first == second
    assert "MOCK" in first.text
    assert first.raw["synthetic_demo"] is True
    await provider.aclose()
