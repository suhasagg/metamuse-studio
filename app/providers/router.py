"""Policy-driven model router with fallback chains, retries and circuit breakers."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .base import LLMProvider, LLMRequest, LLMResponse, ProviderError

STRATEGIES = ("balanced", "quality", "cost", "speed", "offline")

# Each agent gets a preferred provider order. Critics deliberately prefer a different
# model family than the authors they review (cross-model verification).
BALANCED_PREFERENCES: dict[str, tuple[str, ...]] = {
    "sentinel": ("groq", "gemini", "openai", "anthropic", "openrouter", "ollama"),
    "planner": ("anthropic", "openai", "gemini", "openrouter", "groq", "ollama"),
    "research": ("gemini", "openai", "anthropic", "openrouter", "groq", "ollama"),
    "creative": ("anthropic", "openai", "gemini", "openrouter", "groq", "ollama"),
    "code": ("openai", "anthropic", "groq", "openrouter", "gemini", "ollama"),
    "action": ("openai", "anthropic", "gemini", "groq", "openrouter", "ollama"),
    "critic": ("gemini", "openai", "anthropic", "openrouter", "groq", "ollama"),
    "composer": ("anthropic", "openai", "gemini", "openrouter", "groq", "ollama"),
    "memory": ("groq", "gemini", "openai", "anthropic", "openrouter", "ollama"),
}


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    cooldown_s: float = 30.0
    failures: int = 0
    opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at >= self.cooldown_s:
            self.opened_at, self.failures = None, 0  # half-open: allow a trial request
            return False
        return True

    def record_success(self) -> None:
        self.failures, self.opened_at = 0, None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.monotonic()


@dataclass
class RouteTrace:
    agent: str
    strategy: str
    preferred: str
    used: str = ""
    model: str = ""
    skipped: list[dict[str, str]] = field(default_factory=list)

    @property
    def fell_back(self) -> bool:
        return self.used != self.preferred

    def as_dict(self) -> dict[str, Any]:
        return {"agent": self.agent, "strategy": self.strategy, "preferred": self.preferred, "used": self.used,
                "model": self.model, "fallback": self.fell_back, "skipped": self.skipped}


@dataclass
class ProviderStats:
    calls: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    total_latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        avg = self.total_latency_ms / self.calls if self.calls else 0.0
        return {"calls": self.calls, "failures": self.failures, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "cost_usd": round(self.cost_usd, 6),
                "avg_latency_ms": round(avg, 1)}


class ModelRouter:
    def __init__(self, providers: list[LLMProvider], *, fallback: str = "muse-local", max_retries: int = 2,
                 backoff_s: float = 0.4):
        self.providers: dict[str, LLMProvider] = {p.name: p for p in providers}
        self.fallback = fallback
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.breakers: dict[str, CircuitBreaker] = defaultdict(CircuitBreaker)
        self.stats: dict[str, ProviderStats] = defaultdict(ProviderStats)

    def candidates(self, agent: str, strategy: str) -> list[str]:
        hosted = [n for n in self.providers if n != self.fallback]
        if strategy == "offline":
            order = ["ollama"]
        elif strategy == "quality":
            order = sorted(hosted, key=lambda n: -self.providers[n].profile.quality)
        elif strategy == "cost":
            order = sorted(hosted, key=lambda n: self.providers[n].profile.input_cost_per_m
                           + self.providers[n].profile.output_cost_per_m)
        elif strategy == "speed":
            order = sorted(hosted, key=lambda n: -self.providers[n].profile.speed)
        else:
            order = list(BALANCED_PREFERENCES.get(agent, BALANCED_PREFERENCES["planner"]))
        chain = [n for n in order if n in self.providers]
        if self.fallback in self.providers:
            chain.append(self.fallback)
        return chain

    async def complete(self, request: LLMRequest, strategy: str = "balanced") -> tuple[LLMResponse, RouteTrace]:
        strategy = strategy if strategy in STRATEGIES else "balanced"
        chain = self.candidates(request.agent, strategy)
        trace = RouteTrace(agent=request.agent, strategy=strategy, preferred=chain[0] if chain else "")
        last_error: Exception | None = None
        for name in chain:
            provider = self.providers[name]
            if not provider.available:
                trace.skipped.append({"provider": name, "reason": "not configured"})
                continue
            breaker = self.breakers[name]
            if breaker.is_open:
                trace.skipped.append({"provider": name, "reason": "circuit open"})
                continue
            for attempt in range(self.max_retries + 1):
                try:
                    response = await provider.complete(request)
                except ProviderError as exc:
                    last_error = exc
                    self.stats[name].failures += 1
                    if exc.retryable and attempt < self.max_retries:
                        await asyncio.sleep(self.backoff_s * (2 ** attempt))
                        continue
                    breaker.record_failure()
                    trace.skipped.append({"provider": name, "reason": str(exc)[:160]})
                    break
                breaker.record_success()
                self._record(name, response)
                trace.used, trace.model = name, response.model
                return response, trace
        raise ProviderError("router", f"all providers failed for agent '{request.agent}': {last_error}")

    def _record(self, name: str, response: LLMResponse) -> None:
        s = self.stats[name]
        s.calls += 1
        s.input_tokens += response.input_tokens
        s.output_tokens += response.output_tokens
        s.cost_usd += response.cost_usd
        s.total_latency_ms += response.latency_ms

    def describe(self) -> list[dict[str, Any]]:
        out = []
        for name, p in self.providers.items():
            out.append({
                "name": name, "label": p.profile.label, "model": p.profile.model, "available": p.available,
                "quality": p.profile.quality, "speed": p.profile.speed,
                "input_cost_per_m": p.profile.input_cost_per_m, "output_cost_per_m": p.profile.output_cost_per_m,
                "strengths": list(p.profile.strengths), "circuit_open": self.breakers[name].is_open,
                "stats": self.stats[name].as_dict(),
            })
        return out
