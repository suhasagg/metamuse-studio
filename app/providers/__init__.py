"""Provider registry: builds every integration from settings."""

from __future__ import annotations

from ..config import Settings
from .base import LLMProvider, LLMRequest, LLMResponse, Message, ProviderError, ProviderProfile
from .http_providers import AnthropicProvider, GeminiProvider, OpenAICompatibleProvider
from .mock_provider import MuseLocalProvider
from .router import STRATEGIES, ModelRouter, RouteTrace

__all__ = ["LLMProvider", "LLMRequest", "LLMResponse", "Message", "ModelRouter", "ProviderError",
           "RouteTrace", "STRATEGIES", "build_providers", "build_router"]


def build_providers(settings: Settings) -> list[LLMProvider]:
    t = settings.request_timeout_s
    return [
        OpenAICompatibleProvider(
            ProviderProfile("openai", "OpenAI", settings.openai_model, 9.2, 7.5, 0.15, 0.60, ("code", "tools", "json")),
            settings.openai_api_key, timeout=t),
        AnthropicProvider(
            ProviderProfile("anthropic", "Anthropic", settings.anthropic_model, 9.5, 6.5, 3.0, 15.0,
                            ("planning", "writing", "reasoning")),
            settings.anthropic_api_key, timeout=t),
        GeminiProvider(
            ProviderProfile("gemini", "Google Gemini", settings.gemini_model, 8.8, 8.5, 0.075, 0.30,
                            ("long context", "research", "judging")),
            settings.gemini_api_key, timeout=t),
        OpenAICompatibleProvider(
            ProviderProfile("groq", "Groq", settings.groq_model, 7.6, 9.8, 0.05, 0.08, ("ultra-low latency", "triage")),
            settings.groq_api_key, "https://api.groq.com/openai/v1", timeout=t),
        OpenAICompatibleProvider(
            ProviderProfile("openrouter", "OpenRouter", settings.openrouter_model, 8.6, 6.8, 0.52, 0.75,
                            ("model marketplace", "open weights")),
            settings.openrouter_api_key, "https://openrouter.ai/api/v1", timeout=t),
        OpenAICompatibleProvider(
            ProviderProfile("ollama", "Ollama (local)", settings.ollama_model, 7.4, 7.0, 0.0, 0.0,
                            ("private", "on-device")),
            "", settings.ollama_base_url.rstrip("/") + "/v1" if settings.ollama_base_url else "",
            require_key=False, timeout=t),
        MuseLocalProvider(simulate_latency=settings.mock_latency),
    ]


def build_router(settings: Settings) -> ModelRouter:
    return ModelRouter(build_providers(settings))
