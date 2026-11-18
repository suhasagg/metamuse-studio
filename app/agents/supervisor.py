"""Supervisor-level agents: memory recall, planner, critic (LLM-as-judge), composer, memory write."""

from __future__ import annotations

import re
from typing import Any

from langgraph.config import get_config
from langgraph.types import Send

from ..config import get_settings
from ..memory import get_memory
from ..state import AGENTS, MuseState
from .llm import ask, emit

RESEARCHABLE = re.compile(r"\b(buy|order|gift|shop|purchase|trip|flight|hotel|restaurant|dinner|research|find the best|"
                         r"compare|recommend)\b", re.I)
WORKFLOW_AGENTS: dict[str, list[str]] = {
    "launch": ["research", "creative"],
    "code": ["code"],
    "trip": ["research", "action"],
    "errands": ["research", "action"],
    "full": ["research", "creative", "code", "action"],
}


def _suggest_agents(goal: str) -> list[str]:
    g = goal.lower()
    picks = []
    if any(w in g for w in ("research", "market", "compare", "analy", "trend", "launch", "trip", "travel", "buy", "gift")):
        picks.append("research")
    if any(w in g for w in ("launch", "campaign", "story", "brand", "copy", "slogan", "pitch", "marketing")):
        picks.append("creative")
    if any(w in g for w in ("code", "function", "python", "implement", "algorithm", "script", "bug")):
        picks.append("code")
    if any(w in g for w in ("book", "trip", "travel", "schedule", "calendar", "email", "flight", "hotel", "buy", "order",
                            "pay", "split", "refund", "remind", "dinner", "slack", "checkout", "invite")):
        picks.append("action")
    return picks or ["research", "creative"]


def latest_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for r in results:
        latest[r["agent"]] = r
    return [latest[a] for a in AGENTS if a in latest]


async def recall_node(state: MuseState) -> dict[str, Any]:
    cfg = get_config().get("configurable", {})
    layers = get_memory().recall_all(state["goal"], cfg.get("session_id", "default"), state.get("workflow", "auto"))
    emit({"type": "memory", "action": "recall", "count": sum(len(v) for v in layers.values()),
          "layers": {k: len(v) for k, v in layers.items()}})
    return {"memories": layers["episodic"], "memory_context": layers, "iteration": state.get("iteration", 0)}


async def planner_node(state: MuseState) -> dict[str, Any]:
    goal = state["safe_goal"]
    workflow = state.get("workflow", "auto")
    iteration = state.get("iteration", 0)
    critique = state.get("critique") or {}
    if iteration and critique.get("weak_agents"):
        allowed = [a for a in critique["weak_agents"] if a != "action"] or ["research"]
    elif workflow in WORKFLOW_AGENTS:
        allowed = WORKFLOW_AGENTS[workflow]
    else:
        allowed = _suggest_agents(goal)
    ctx = state.get("memory_context") or {}
    notes = [f"- past mission: {m['summary']}" for m in state.get("memories", [])]
    notes += [f"- earlier this session: {t['goal'][:120]}" for t in ctx.get("session", [])[:3]]
    notes += [f"- proven plan ({p['wins']}/{p['uses']} wins): {p['name']}" for p in ctx.get("procedural", [])
              if p["kind"] == "plan"]
    notes += [f"- user {p['kind']}: {p['value']}" for p in ctx.get("profile", []) if p["kind"] in ("budget", "contact")][:4]
    if workflow == "errands" and not iteration and not RESEARCHABLE.search(goal):
        allowed = ["action"]  # nothing to look up (e.g. pay, email, schedule): do not pad the brief with research
    memory_note = "\n".join(notes) or "none"
    plan = await ask(
        "planner", "planner.plan",
        "You are the Meta Muse supervisor. Decompose the goal into tasks for specialist agents that run in "
        f"parallel. Allowed agents: {allowed}. research=cited web research, creative=positioning/copy with "
        "self-critique, code=implementation verified in a sandbox, action=real-world tool actions with human "
        'approval. Return JSON: {"rationale": str, "tasks": [{"agent": str, "objective": str}]}',
        f"Goal: {goal}\nRelevant memories:\n{memory_note}\nReviewer feedback: {critique.get('feedback', 'none')}",
        context={"goal": goal, "suggested_agents": allowed,
                 "feedback": critique.get("feedback") if iteration else None},
        json_mode=True, fallback={"rationale": "Heuristic plan", "tasks": [{"agent": a, "objective": goal} for a in allowed]})
    tasks, seen = [], set()
    for i, t in enumerate(plan.get("tasks", []) if isinstance(plan, dict) else []):
        agent = str(t.get("agent", "")).lower()
        if agent in allowed and agent not in seen:
            seen.add(agent)
            tasks.append({"id": f"t{iteration}-{i + 1}", "agent": agent, "objective": t.get("objective") or goal})
    for agent in allowed:  # guarantee coverage even if the model omitted an agent
        if agent not in seen:
            tasks.append({"id": f"t{iteration}-{len(tasks) + 1}", "agent": agent, "objective": goal})
    emit({"type": "plan", "tasks": tasks, "rationale": plan.get("rationale", "") if isinstance(plan, dict) else ""})
    return {"plan": tasks, "plan_rationale": plan.get("rationale", "") if isinstance(plan, dict) else ""}


def dispatch(state: MuseState) -> list[Send]:
    """Fan out: one `Send` per planned task, each into its specialist subgraph (runs in parallel)."""
    return [Send(t["agent"], {"task": t, "goal": state["safe_goal"]}) for t in state["plan"]]


def route_after_sentinel(state: MuseState) -> str:
    return "blocked" if state["security"]["blocked"] else "planner"


def _signals(results: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for r in results:
        merged.update(r.get("signals", {}))
    return merged


def muse_score(overall: float, results: list[dict[str, Any]], security: dict[str, Any]) -> dict[str, Any]:
    """Quality (0-100 from the judge) plus an excellence bonus (0-10) earned only from verified run signals."""
    s = _signals(results)
    agents = {r["agent"] for r in results}
    checks: list[tuple[str, bool]] = [("Sentinel cleared with no high-risk findings",
                                       not any(f["severity"] == "high" for f in security.get("findings", [])))]
    if "research" in agents:
        checks.append((f"{s.get('sources', 0)} sources retrieved, citations verified", bool(s.get("citations_verified"))))
    if "creative" in agents:
        checks.append(("Cross-model reflection improved creative score",
                       s.get("creative_final_score", 0) > s.get("creative_first_score", 99)))
    if "code" in agents:
        checks.append((f"All {s.get('tests_total', 0)} sandbox tests pass "
                       f"(self-healed in {s.get('code_attempts', 0)} attempt(s))", bool(s.get("tests_passed"))))
    if "action" in agents:
        checks.append(("Only human-approved actions executed (idempotent)", bool(s.get("approval_honored"))))
    earned = sum(1 for _, ok in checks if ok)
    bonus = round(10 * earned / len(checks), 1)
    quality = round(overall * 10, 1)
    return {"quality": quality, "bonus": bonus, "total": round(quality + bonus, 1),
            "checks": [{"label": label, "ok": ok} for label, ok in checks]}


async def critic_node(state: MuseState) -> dict[str, Any]:
    settings = get_settings()
    results = latest_results(state.get("results", []))
    signals = _signals(results)
    digest = "\n\n".join(f"### {r['agent']}\n{r['content'][:1500]}" for r in results)
    judged = await ask(
        "critic", "critic.judge",
        "You are an exacting reviewer (LLM-as-judge) from a different model family than the authors. Score the "
        "combined work 0-10 on relevance, grounding, correctness, safety, clarity. Return JSON: "
        '{"scores": {...}, "overall": float, "feedback": str, "pass": bool, "weak_agents": [str]}',
        f"Goal: {state['safe_goal']}\nVerified signals: {signals}\n\nWork:\n{digest}",
        context={"goal": state["safe_goal"], "signals": signals, "threshold": settings.quality_threshold},
        json_mode=True, fallback={"scores": {}, "overall": 8.0, "feedback": "Judge unavailable", "pass": True})
    overall = float(judged.get("overall") or 0) or (
        sum(judged.get("scores", {}).values()) / max(1, len(judged.get("scores", {}))))
    if signals.get("tests_passed") is False:
        overall = min(overall, 6.5)
    passed = overall >= settings.quality_threshold
    critique = {**judged, "overall": round(overall, 2), "pass": passed,
                "weak_agents": judged.get("weak_agents") or ([] if passed else [r["agent"] for r in results])}
    score = muse_score(overall, results, state.get("security", {}))
    emit({"type": "critique", **critique, "muse_score": score})
    return {"critique": critique, "muse_score": score}


def route_after_critic(state: MuseState) -> str:
    critique = state.get("critique", {})
    if critique.get("pass") or state.get("iteration", 0) >= get_settings().max_revisions:
        return "composer"
    return "revise"


async def revise_node(state: MuseState) -> dict[str, Any]:
    emit({"type": "log", "level": "info", "agent": "critic",
          "message": f"Quality {state['critique'].get('overall')} below threshold; re-planning weak agents."})
    return {"iteration": state.get("iteration", 0) + 1}


async def composer_node(state: MuseState) -> dict[str, Any]:
    results = latest_results(state.get("results", []))
    body = "\n\n".join(f"## {r['title']}\n{r['content']}" for r in results)
    final = await ask(
        "composer", "composer.final",
        "You are the Meta Muse composer. Merge specialist outputs into one polished Markdown brief: title, "
        "executive summary, one section per specialist (keep code blocks, tables and citations intact), next steps.",
        f"Goal: {state['safe_goal']}\n\nSpecialist outputs:\n{body}",
        context={"goal": state["safe_goal"], "results": results}, max_tokens=3000)
    return {"final": final}


async def remember_node(state: MuseState) -> dict[str, Any]:
    results = latest_results(state.get("results", []))
    summary = await ask("memory", "memory.summarize", "Summarize this completed mission in one sentence for long-term memory.",
                        f"Goal: {state['safe_goal']}\nAgents: {[r['agent'] for r in results]}",
                        context={"goal": state["safe_goal"], "results": results}, max_tokens=120)
    cfg = get_config().get("configurable", {})
    written = get_memory().consolidate(
        run_id=cfg.get("thread_id", ""), session_id=cfg.get("session_id", "default"),
        workflow=state.get("workflow", "auto"), strategy=cfg.get("strategy", "balanced"), goal=state["safe_goal"],
        summary=summary.strip(), score=state.get("muse_score", {}).get("total", 0), plan=state.get("plan", []),
        results=results)
    emit({"type": "memory", "action": "write", "summary": summary.strip(), "layers": written})
    return {}
