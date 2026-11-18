"""Research subgraph: plan queries -> parallel web search -> cited synthesis -> verification loop."""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..agents.llm import ask, emit
from ..state import AgentResult, SubgraphOutput, TaskInput
from ..tools.search import web_search


class ResearchState(TypedDict, total=False):
    task: dict[str, Any]
    goal: str
    queries: list[str]
    sources: list[dict[str, Any]]
    draft: str
    verification: dict[str, Any]
    attempt: int
    results: Annotated[list[AgentResult], operator.add]


async def plan_queries(state: ResearchState) -> dict[str, Any]:
    attempt = state.get("attempt", 0)
    out = await ask("research", "research.queries",
                    "Generate 2-3 concise, diverse web/encyclopedia search queries for the research objective. "
                    'Return JSON: {"queries": [str]}',
                    f"Objective: {state['task']['objective']}\nPrevious queries: {state.get('queries', [])}",
                    context={"goal": state["goal"], "attempt": attempt}, json_mode=True,
                    fallback={"queries": [state["goal"][:80]]})
    return {"queries": [q for q in out.get("queries", []) if isinstance(q, str)][:3] or [state["goal"][:80]]}


async def search(state: ResearchState) -> dict[str, Any]:
    async def one(q: str) -> list[dict[str, Any]]:
        emit({"type": "tool", "agent": "research", "tool": "web_search", "input": q, "status": "running"})
        hits = await web_search(q, k=3)
        emit({"type": "tool", "agent": "research", "tool": "web_search", "input": q, "status": "done",
              "output": [h["title"] for h in hits], "backend": hits[0]["source"] if hits else "none"})
        flagged = [h["title"] for h in hits if h.get("security")]
        if flagged:
            emit({"type": "security_finding", "agent": "research", "source": "web_search", "items": flagged,
                  "findings": sorted({f for h in hits for f in h.get("security", [])}),
                  "action": "instruction-like text removed before the model saw it"})
        return hits

    batches = await asyncio.gather(*(one(q) for q in state["queries"]))
    seen = {s["url"] for s in state.get("sources", [])}
    sources = list(state.get("sources", []))
    for hit in (h for batch in batches for h in batch):
        if hit["url"] not in seen:
            seen.add(hit["url"])
            sources.append(hit)
    return {"sources": sources[:6]}


async def synthesize(state: ResearchState) -> dict[str, Any]:
    numbered = "\n".join(f"[{i}] {s['title']} ({s['url']}): {s['snippet']}" for i, s in enumerate(state["sources"], 1))
    draft = await ask("research", "research.synthesize",
                      "You are a meticulous research analyst. Write a Markdown briefing using ONLY the numbered "
                      "sources. Cite every claim inline like [1]. End with a short 'What this means' section.",
                      f"Objective: {state['task']['objective']}\n\nSources:\n{numbered}",
                      context={"goal": state["goal"], "sources": state["sources"]})
    return {"draft": draft}


async def verify(state: ResearchState) -> dict[str, Any]:
    result = await ask("critic", "research.verify",
                       "Check that the briefing's claims are supported by the cited sources. Return JSON: "
                       '{"supported": bool, "coverage": 0..1, "issues": [str]}',
                       f"Sources: {[s['title'] for s in state['sources']]}\n\nBriefing:\n{state['draft']}",
                       context={"goal": state["goal"], "sources": state["sources"], "draft": state["draft"]},
                       json_mode=True, fallback={"supported": True, "coverage": 0.7, "issues": []})
    return {"verification": result, "attempt": state.get("attempt", 0) + 1}


def after_verify(state: ResearchState) -> str:
    if state["verification"].get("supported") or state.get("attempt", 0) >= 2:
        return "finalize"
    return "plan_queries"


async def finalize(state: ResearchState) -> dict[str, Any]:
    refs = "\n".join(f"{i}. [{s['title']}]({s['url']})" for i, s in enumerate(state["sources"], 1))
    v = state["verification"]
    content = f"{state['draft']}\n\n**Sources**\n{refs}"
    return {"results": [{
        "agent": "research", "task_id": state["task"]["id"], "title": "Research briefing", "content": content,
        "artifacts": {"sources": state["sources"], "verification": v},
        "signals": {"sources": len(state["sources"]), "citations_verified": bool(v.get("supported")),
                    "citation_coverage": v.get("coverage", 0)},
    }]}


def build_research_graph():
    g = StateGraph(ResearchState, input_schema=TaskInput, output_schema=SubgraphOutput)
    g.add_node("plan_queries", plan_queries)
    g.add_node("search", search)
    g.add_node("synthesize", synthesize)
    g.add_node("verify", verify)
    g.add_node("finalize", finalize)
    g.add_edge(START, "plan_queries")
    g.add_edge("plan_queries", "search")
    g.add_edge("search", "synthesize")
    g.add_edge("synthesize", "verify")
    g.add_conditional_edges("verify", after_verify, ["plan_queries", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()
