import httpx
import pytest

from src.nlq.llm_client import OpenAICompatibleClient, OpenAIConnectionError


def test_openai_compatible_client_sends_structured_output_request(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": '{"answer":"ok"}'}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OpenAICompatibleClient(
        model="test-model",
        api_key="secret",
        base_url="https://llm.example/v1/",
    )
    result = client.generate(
        system="system",
        user="user",
        json_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )

    assert result == '{"answer":"ok"}'
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["json"]["messages"][0]["role"] == "developer"
    assert captured["json"]["response_format"]["type"] == "json_schema"
    assert captured["json"]["response_format"]["json_schema"]["strict"] is True


def test_openai_compatible_client_requires_api_key():
    client = OpenAICompatibleClient(model="test-model", api_key="", base_url="https://llm.example/v1")

    with pytest.raises(OpenAIConnectionError, match="OPENAI_API_KEY"):
        client.check_connection()
