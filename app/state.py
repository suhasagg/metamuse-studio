"""Typed LangGraph state for the supervisor graph and its specialist subgraphs."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

AGENTS = ("research", "creative", "code", "action")


class AgentResult(TypedDict, total=False):
    agent: str
    task_id: str
    title: str
    content: str
    artifacts: dict[str, Any]
    signals: dict[str, Any]


class MuseState(TypedDict, total=False):
    goal: str
    workflow: str
    safe_goal: str
    memories: list[dict[str, Any]]
    memory_context: dict[str, list[dict[str, Any]]]
    security: dict[str, Any]
    plan: list[dict[str, Any]]
    plan_rationale: str
    results: Annotated[list[AgentResult], operator.add]
    critique: dict[str, Any]
    muse_score: dict[str, Any]
    iteration: int
    final: str


class TaskInput(TypedDict):
    """Payload each subgraph receives through a LangGraph `Send`."""

    task: dict[str, Any]
    goal: str


class SubgraphOutput(TypedDict):
    results: Annotated[list[AgentResult], operator.add]
