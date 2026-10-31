"""Pluggable LLM providers. Only already-redacted messages are ever passed in."""

from __future__ import annotations

import re

import httpx

from promptshield.config import settings


class Provider:
    name = "base"

    async def chat(self, messages: list[dict], model: str | None = None) -> str:
        raise NotImplementedError


class MockProvider(Provider):
    """Deterministic stand-in for demos/tests: echoes placeholders back like a real LLM would."""

    name = "mock"

    async def chat(self, messages, model=None):
        last = messages[-1]["content"]
        tokens = sorted(set(re.findall(r"\[[A-Z_]+_\d+\]", last)))
        if tokens:
            refs = ", ".join(tokens)
            return f"[mock LLM] I received your request. It mentions {refs}; I can help with that."
        return f"[mock LLM] I received your request ({len(last)} characters)."


class OllamaProvider(Provider):
    name = "ollama"

    async def chat(self, messages, model=None):
        async with httpx.AsyncClient(base_url=settings.llm_base_url, timeout=120) as client:
            r = await client.post(
                "/api/chat", json={"model": model or settings.llm_model, "messages": messages, "stream": False}
            )
            r.raise_for_status()
            return r.json()["message"]["content"]


class OpenAICompatibleProvider(Provider):
    """Any endpoint that speaks the /v1/chat/completions format."""

    name = "openai_compatible"

    async def chat(self, messages, model=None):
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
        async with httpx.AsyncClient(base_url=settings.llm_base_url, timeout=120, headers=headers) as client:
            r = await client.post(
                "/v1/chat/completions", json={"model": model or settings.llm_model, "messages": messages}
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]


def get_provider(name: str | None = None) -> Provider:
    name = name or settings.llm_provider
    return {"mock": MockProvider, "ollama": OllamaProvider, "openai_compatible": OpenAICompatibleProvider}[name]()
