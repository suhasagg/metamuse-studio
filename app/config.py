"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("MUSE_DATA_DIR", ".muse")))
    default_strategy: str = field(default_factory=lambda: _env("MUSE_ROUTING_STRATEGY", "balanced"))
    quality_threshold: float = field(default_factory=lambda: float(_env("MUSE_QUALITY_THRESHOLD", "8.0")))
    max_revisions: int = field(default_factory=lambda: int(_env("MUSE_MAX_REVISIONS", "1")))
    mock_latency: bool = field(default_factory=lambda: _env("MUSE_MOCK_LATENCY", "1") == "1")
    search_backend: str = field(default_factory=lambda: _env("MUSE_SEARCH_BACKEND", "auto"))
    sandbox_timeout_s: float = field(default_factory=lambda: float(_env("MUSE_SANDBOX_TIMEOUT", "8")))
    request_timeout_s: float = field(default_factory=lambda: float(_env("MUSE_LLM_TIMEOUT", "60")))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-1.5-flash"))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "llama-3.1-8b-instant"))
    openrouter_api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))
    openrouter_model: str = field(default_factory=lambda: _env("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct"))
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.1"))
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY"))

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def memory_path(self) -> Path:
        return self.data_dir / "memory.sqlite"


def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
