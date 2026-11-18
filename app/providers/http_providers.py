"""HTTP integrations for hosted and local model providers (no vendor SDKs required)."""

from __future__ import annotations

import time
from typing import Any

import httpx

from .base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderError,
    ProviderProfile,
    estimate_tokens,
    is_retryable_status,
)


class _HTTPProvider(LLMProvider):
    def __init__(self, profile: ProviderProfile, api_key: str, base_url: str, *, timeout: float = 60.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.profile = profile
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    async def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                resp = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"network error: {exc}", retryable=True) from exc
        if resp.status_code >= 400:
            raise ProviderError(self.name, f"HTTP {resp.status_code}: {resp.text[:300]}",
                                retryable=is_retryable_status(resp.status_code), status=resp.status_code)
        return resp.json()

    def _response(self, text: str, request: LLMRequest, started: float, usage_in: int | None,
                  usage_out: int | None, raw: dict[str, Any]) -> LLMResponse:
        tin = usage_in or estimate_tokens(request.system + request.prompt)
        tout = usage_out or estimate_tokens(text)
        return LLMResponse(text=text, provider=self.name, model=self.profile.model, input_tokens=tin,
                           output_tokens=tout, latency_ms=(time.perf_counter() - started) * 1000,
                           cost_usd=self.cost(tin, tout), raw=raw)


class OpenAICompatibleProvider(_HTTPProvider):
    """OpenAI Chat Completions; also serves Groq, OpenRouter and Ollama (OpenAI-compatible APIs)."""

    def __init__(self, profile: ProviderProfile, api_key: str, base_url: str = "https://api.openai.com/v1",
                 *, require_key: bool = True, **kw: Any):
        super().__init__(profile, api_key, base_url, **kw)
        self.require_key = require_key

    @property
    def available(self) -> bool:
        return bool(self.api_key) if self.require_key else bool(self.base_url)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = await self._post(f"{self.base_url}/chat/completions", payload, headers)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ProviderError(self.name, f"unexpected response shape: {data}") from exc
        usage = data.get("usage") or {}
        return self._response(text, request, started, usage.get("prompt_tokens"), usage.get("completion_tokens"), data)


class AnthropicProvider(_HTTPProvider):
    def __init__(self, profile: ProviderProfile, api_key: str, base_url: str = "https://api.anthropic.com/v1", **kw: Any):
        super().__init__(profile, api_key, base_url, **kw)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        system = request.system
        if request.json_mode:
            system += "\n\nRespond with a single valid JSON object and nothing else."
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": m.role, "content": m.content} for m in request.chat],
        }
        if system:
            payload["system"] = system
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = await self._post(f"{self.base_url}/messages", payload, headers)
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        usage = data.get("usage") or {}
        return self._response(text, request, started, usage.get("input_tokens"), usage.get("output_tokens"), data)


class GeminiProvider(_HTTPProvider):
    def __init__(self, profile: ProviderProfile, api_key: str,
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta", **kw: Any):
        super().__init__(profile, api_key, base_url, **kw)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in request.chat
        ]
        generation: dict[str, Any] = {"temperature": request.temperature, "maxOutputTokens": request.max_tokens}
        if request.json_mode:
            generation["responseMimeType"] = "application/json"
        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation}
        if request.system:
            payload["systemInstruction"] = {"parts": [{"text": request.system}]}
        url = f"{self.base_url}/models/{self.profile.model}:generateContent"
        data = await self._post(url, payload, {"x-goog-api-key": self.api_key, "Content-Type": "application/json"})
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(self.name, f"no candidates returned: {str(data)[:200]}") from exc
        text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata") or {}
        return self._response(text, request, started, usage.get("promptTokenCount"),
                              usage.get("candidatesTokenCount"), data)
