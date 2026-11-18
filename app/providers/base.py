"""Provider abstraction shared by every LLM integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class LLMRequest:
    messages: list[Message]
    agent: str = "general"
    task: str = "general"
    json_mode: bool = False
    temperature: float = 0.3
    max_tokens: int = 1200
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def system(self) -> str:
        return "\n\n".join(m.content for m in self.messages if m.role == "system")

    @property
    def chat(self) -> list[Message]:
        return [m for m in self.messages if m.role != "system"]

    @property
    def prompt(self) -> str:
        return "\n\n".join(m.content for m in self.chat)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    raw: dict[str, Any] | None = None


class ProviderError(RuntimeError):
    def __init__(self, provider: str, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable
        self.status = status


@dataclass(frozen=True)
class ProviderProfile:
    """Static routing metadata: relative quality, speed and price per 1M tokens."""

    name: str
    label: str
    model: str
    quality: float
    speed: float
    input_cost_per_m: float
    output_cost_per_m: float
    strengths: tuple[str, ...] = ()


class LLMProvider:
    profile: ProviderProfile

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def available(self) -> bool:
        return True

    async def complete(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        p = self.profile
        return round(input_tokens / 1e6 * p.input_cost_per_m + output_tokens / 1e6 * p.output_cost_per_m, 6)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def is_retryable_status(status: int) -> bool:
    return status in (408, 409, 425, 429) or status >= 500
