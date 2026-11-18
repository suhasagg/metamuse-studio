"""Shared helpers every agent uses to call the model router and stream telemetry."""

from __future__ import annotations

import json
import re
from typing import Any

from langgraph.config import get_config, get_stream_writer

from ..providers import LLMRequest, Message, ModelRouter

_router: ModelRouter | None = None


def set_router(router: ModelRouter) -> None:
    global _router
    _router = router


def get_router() -> ModelRouter:
    if _router is None:
        from ..config import get_settings
        from ..providers import build_router

        set_router(build_router(get_settings()))
    assert _router is not None
    return _router


def emit(event: dict[str, Any]) -> None:
    """Push a custom event onto the LangGraph stream (no-op outside a graph run)."""
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - not running inside a graph
        return
    if writer:
        writer(event)


def current_strategy() -> str:
    try:
        return get_config().get("configurable", {}).get("strategy", "balanced")
    except Exception:  # noqa: BLE001
        return "balanced"


def parse_json(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


async def ask(agent: str, task: str, system: str, prompt: str, *, context: dict[str, Any] | None = None,
              json_mode: bool = False, fallback: Any = None, temperature: float = 0.3,
              max_tokens: int = 1400) -> Any:
    request = LLMRequest(messages=[Message("system", system), Message("user", prompt)], agent=agent, task=task,
                         json_mode=json_mode, temperature=temperature, max_tokens=max_tokens,
                         context=context or {})
    response, trace = await get_router().complete(request, current_strategy())
    emit({"type": "llm", "agent": agent, "task": task, "provider": response.provider, "model": response.model,
          "latency_ms": round(response.latency_ms, 1), "input_tokens": response.input_tokens,
          "output_tokens": response.output_tokens, "cost_usd": response.cost_usd, "route": trace.as_dict(),
          "preview": response.text[:280]})
    if not json_mode:
        return response.text
    try:
        return parse_json(response.text)
    except (json.JSONDecodeError, ValueError):
        emit({"type": "log", "level": "warn", "agent": agent, "message": f"{task}: invalid JSON, using fallback"})
        return fallback


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return (max(blocks, key=len) if blocks else text).strip() + "\n"
