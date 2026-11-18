"""Creative subgraph: ideate -> select -> draft -> cross-model critique -> reflexion revise loop."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..agents.llm import ask
from ..state import AgentResult, SubgraphOutput, TaskInput

TARGET_SCORE = 8.5
MAX_ROUNDS = 2


class CreativeState(TypedDict, total=False):
    task: dict[str, Any]
    goal: str
    concepts: list[dict[str, Any]]
    concept: dict[str, Any]
    draft: str
    scores: list[float]
    feedback: list[str]
    round: int
    results: Annotated[list[AgentResult], operator.add]


async def ideate(state: CreativeState) -> dict[str, Any]:
    out = await ask("creative", "creative.ideate",
                    "You are an award-winning creative director. Propose 3 distinct campaign concepts. Return JSON: "
                    '{"concepts": [{"name": str, "hook": str, "angle": str}]}',
                    f"Brief: {state['task']['objective']}", context={"goal": state["goal"]}, json_mode=True,
                    temperature=0.9, fallback={"concepts": [{"name": "Muse", "hook": state["goal"], "angle": "Direct"}]})
    concepts = out.get("concepts") or [{"name": "Muse", "hook": state["goal"], "angle": "Direct"}]
    return {"concepts": concepts[:3]}


def select(state: CreativeState) -> dict[str, Any]:
    # Deterministic selection: prefer the concept with the most concrete angle (longest), ties -> first.
    best = max(state["concepts"], key=lambda c: (len(str(c.get("angle", ""))), -state["concepts"].index(c)))
    return {"concept": best, "round": 0, "scores": [], "feedback": []}


async def draft(state: CreativeState) -> dict[str, Any]:
    text = await ask("creative", "creative.draft",
                     "Write the campaign: tagline, positioning, launch narrative and channel plan in Markdown.",
                     f"Brief: {state['task']['objective']}\nConcept: {state['concept']}",
                     context={"goal": state["goal"], "concept": state["concept"]}, temperature=0.8)
    return {"draft": text}


async def critique(state: CreativeState) -> dict[str, Any]:
    out = await ask("critic", "creative.critique",
                    "You are a demanding brand strategist from a different model family than the author. Score the "
                    'draft 0-10 and list concrete improvements. Return JSON: {"score": float, "feedback": [str]}',
                    f"Brief: {state['task']['objective']}\n\nDraft:\n{state['draft']}",
                    context={"goal": state["goal"], "round": state.get("round", 0)}, json_mode=True,
                    fallback={"score": TARGET_SCORE, "feedback": []})
    return {"scores": state.get("scores", []) + [float(out.get("score", 0))], "feedback": out.get("feedback", [])}


def after_critique(state: CreativeState) -> str:
    if state["scores"][-1] >= TARGET_SCORE or state.get("round", 0) >= MAX_ROUNDS:
        return "finalize"
    return "revise"


async def revise(state: CreativeState) -> dict[str, Any]:
    text = await ask("creative", "creative.revise",
                     "Revise the campaign to address every piece of feedback. Keep it specific, vivid and "
                     "measurable. Include a phased launch table with KPIs and 3 social posts. Markdown only.",
                     f"Draft:\n{state['draft']}\n\nFeedback:\n" + "\n".join(f"- {f}" for f in state["feedback"]),
                     context={"goal": state["goal"], "concept": state["concept"], "feedback": state["feedback"]},
                     temperature=0.7, max_tokens=2000)
    return {"draft": text, "round": state.get("round", 0) + 1}


def finalize(state: CreativeState) -> dict[str, Any]:
    scores = state["scores"]
    journey = " -> ".join(f"{s:.1f}" for s in scores)
    return {"results": [{
        "agent": "creative", "task_id": state["task"]["id"], "title": "Creative campaign",
        "content": f"{state['draft']}\n\n*Reflection loop: critic score {journey} over {len(scores)} review(s).*",
        "artifacts": {"concepts": state["concepts"], "selected": state["concept"], "scores": scores},
        "signals": {"creative_first_score": scores[0], "creative_final_score": scores[-1]},
    }]}


def build_creative_graph():
    g = StateGraph(CreativeState, input_schema=TaskInput, output_schema=SubgraphOutput)
    for name, fn in (("ideate", ideate), ("select", select), ("draft", draft), ("critique", critique),
                     ("revise", revise), ("finalize", finalize)):
        g.add_node(name, fn)
    g.add_edge(START, "ideate")
    g.add_edge("ideate", "select")
    g.add_edge("select", "draft")
    g.add_edge("draft", "critique")
    g.add_conditional_edges("critique", after_critique, ["revise", "finalize"])
    g.add_edge("revise", "critique")
    g.add_edge("finalize", END)
    return g.compile()
