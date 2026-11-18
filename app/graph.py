"""Supervisor graph: memory -> Sentinel -> planner -> parallel subgraphs (Send) -> critic loop -> composer."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents.sentinel import blocked_node, sentinel_node
from .agents.supervisor import (
    composer_node,
    critic_node,
    dispatch,
    planner_node,
    recall_node,
    remember_node,
    revise_node,
    route_after_critic,
    route_after_sentinel,
)
from .state import MuseState
from .subgraphs.action import build_action_graph
from .subgraphs.code import build_code_graph
from .subgraphs.creative import build_creative_graph
from .subgraphs.research import build_research_graph

SUBGRAPHS = {
    "research": build_research_graph,
    "creative": build_creative_graph,
    "code": build_code_graph,
    "action": build_action_graph,
}


def build_graph(checkpointer=None):
    g = StateGraph(MuseState)
    g.add_node("recall", recall_node)
    g.add_node("sentinel", sentinel_node)
    g.add_node("blocked", blocked_node)
    g.add_node("planner", planner_node)
    for name, factory in SUBGRAPHS.items():
        g.add_node(name, factory())
    g.add_node("critic", critic_node)
    g.add_node("revise", revise_node)
    g.add_node("composer", composer_node)
    g.add_node("remember", remember_node)

    g.add_edge(START, "recall")
    g.add_edge("recall", "sentinel")
    g.add_conditional_edges("sentinel", route_after_sentinel, ["blocked", "planner"])
    g.add_edge("blocked", END)
    g.add_conditional_edges("planner", dispatch, list(SUBGRAPHS))
    for name in SUBGRAPHS:
        g.add_edge(name, "critic")
    g.add_conditional_edges("critic", route_after_critic, ["revise", "composer"])
    g.add_edge("revise", "planner")
    g.add_edge("composer", "remember")
    g.add_edge("remember", END)
    return g.compile(checkpointer=checkpointer)
