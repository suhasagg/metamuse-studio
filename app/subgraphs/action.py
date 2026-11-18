"""Action subgraph: plan tool calls -> risk policy -> human approval (interrupt) -> idempotent execution."""

from __future__ import annotations

import math
import operator
import re
from typing import Annotated, Any, TypedDict

from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..agents.llm import ask, emit
from ..providers.mock_provider import MuseLocalProvider
from ..state import AgentResult, SubgraphOutput, TaskInput
from ..tools import connectors

PAYMENT_LIMIT_USD = 500  # per-action hard limit
RUN_BUDGET_USD = 1000  # cumulative spend cap per mission
TRUSTED_EMAIL_DOMAINS = ("example.com", "muse.local")


class ActionState(TypedDict, total=False):
    task: dict[str, Any]
    goal: str
    summary: str
    actions: list[dict[str, Any]]
    decision: dict[str, Any]
    executed: list[dict[str, Any]]
    results: Annotated[list[AgentResult], operator.add]


def _unique_ids(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Model-written plans can repeat or omit ids. Approval is granted per id, so every action gets its own;
    a dependency on an ambiguous (repeated) id is dropped to an unsatisfiable one rather than guessed."""
    raw = [str(a.get("id") or "") for a in actions]
    out = []
    for n, a in enumerate(actions, 1):
        deps = a.get("depends_on") if isinstance(a.get("depends_on"), list) else []
        out.append({**a, "id": f"a{n}", "args": a.get("args") if isinstance(a.get("args"), dict) else {},
                    "title": str(a.get("title") or a.get("tool")),
                    "depends_on": [f"a{raw.index(str(d)) + 1}" if raw.count(str(d)) == 1 else f"unknown:{d}" for d in deps]})
    return out


async def plan_actions(state: ActionState) -> dict[str, Any]:
    tools = "; ".join(f"{c.name} [{c.domain}/{c.risk}]: {c.description}" for c in connectors.REGISTRY.values())
    default = MuseLocalProvider._action_plan(None, state["goal"], {})  # type: ignore[arg-type]
    out = await ask("action", "action.plan",
                    f"You plan real-world actions using ONLY these connectors: {tools}. Return JSON: "
                    '{"summary": str, "actions": [{"id": str, "tool": str, "title": str, "args": object, '
                    '"depends_on": [str]}]}. Put money amounts in args.amount_usd',
                    f"Objective: {state['task']['objective']}", context={"goal": state["goal"]}, json_mode=True,
                    fallback=default)
    out = out if isinstance(out, dict) else default
    actions = [a for a in (out.get("actions") or []) if a.get("tool") in connectors.REGISTRY] or default["actions"]
    return {"summary": out.get("summary", ""), "actions": _unique_ids(actions)}


def _amount(action: dict[str, Any]) -> float | None:
    """Dollar amount of an action; None when it is not a finite, non-negative number."""
    try:
        value = float((action.get("args") or {}).get("amount_usd", 0) or 0)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _run_id(state: ActionState) -> str:
    try:
        return get_config().get("configurable", {}).get("thread_id", "local")
    except RuntimeError:  # executed outside a graph run (unit tests, scripts)
        return state.get("run_id", "local")


def _bind(action: dict[str, Any], by_id: dict[str, dict[str, Any]], outputs: dict[str, Any]) -> None:
    """Resolve an action's arguments from the results of the read-only steps it depends on, so the human
    approves the real slot, product and price instead of placeholders."""
    args = action["args"] = dict(action.get("args") or {})
    for dep in action.get("depends_on", []):
        if dep not in outputs:
            parent = by_id.get(dep, {})
            if action["tool"] == "checkout_cart" and parent.get("tool") == "add_to_cart":
                if parent.get("unresolved"):
                    action["unresolved"] = parent["unresolved"]
                elif "price_usd" in parent["args"]:
                    args["amount_usd"] = round(float(parent["args"]["price_usd"]) * int(parent["args"].get("qty", 1)), 2)
                    action["title"] = f"Checkout and pay ${args['amount_usd']:,.2f}"
            continue
        out, source = outputs[dep] or {}, by_id.get(dep, {}).get("tool")
        if action["tool"] == "create_calendar_event" and source == "find_meeting_slots" and not args.get("start"):
            if out.get("slots"):
                # The slot was checked free for this duration, so the event must not run longer than that.
                args.update(start=out["slots"][0], duration_min=out.get("duration_min", 60))
            else:
                action["unresolved"] = "no free slot was found"
        elif action["tool"] == "add_to_cart" and source == "search_products":
            priced = [r for r in out.get("results", []) if r.get("sku") and r.get("price_usd") is not None]
            if priced:
                args.update(sku=priced[0]["sku"], price_usd=priced[0]["price_usd"])
                action["title"] = f"Add {priced[0].get('title') or priced[0]['sku']} to cart"
            else:
                action["unresolved"] = "the product search returned nothing to buy"


async def discover(state: ActionState) -> dict[str, Any]:
    """Run the read-only steps that need no prerequisite, then bind their results into dependent actions."""
    run_id = _run_id(state)
    actions = [dict(a) for a in state["actions"]]
    by_id = {a["id"]: a for a in actions}
    outputs: dict[str, Any] = {}
    for a in actions:
        if connectors.risk_of(a["tool"]) != connectors.RISK_READ or a.get("depends_on"):
            continue
        res = await _call(a, run_id)
        if res["ok"]:
            outputs[a["id"]] = res["output"]
    for a in actions:
        _bind(a, by_id, outputs)
    return {"actions": actions}


async def _call(a: dict[str, Any], run_id: str) -> dict[str, Any]:
    emit({"type": "tool", "agent": "action", "tool": a["tool"], "input": a["title"], "status": "running"})
    res = await connectors.execute(a["tool"], a.get("args") or {}, idempotency_key=f"{run_id}:{a['id']}")
    emit({"type": "tool", "agent": "action", "tool": a["tool"], "input": a["title"], "status": "done", "backend": res["mode"],
          "output": _fmt_output(res["output"]) if res["ok"] else "FAILED: " + res["error"]})
    if res.get("security"):
        emit({"type": "security_finding", "agent": "action", "source": a["tool"], "findings": res["security"],
              "items": [a["title"]], "action": "instruction-like text removed; it cannot trigger any action"})
    return res


def apply_policy(state: ActionState) -> dict[str, Any]:
    """Deterministic guardrails applied *before* any human sees the plan."""
    governed, committed = [], 0.0
    for a in state["actions"]:
        risk = connectors.risk_of(a["tool"])
        amount = _amount(a)
        notes: list[str] = []
        policy = "auto-approved (read-only)" if risk == connectors.RISK_READ else "needs human approval"
        if a.get("unresolved") and risk != connectors.RISK_READ:
            policy = f"blocked: {a['unresolved']}"
        elif risk == connectors.RISK_PAYMENT:
            if amount is None or amount <= 0:
                policy = "blocked: payment needs a positive, finite amount"
            elif amount > PAYMENT_LIMIT_USD:
                policy = f"blocked: ${amount:,.2f} exceeds the ${PAYMENT_LIMIT_USD} per-action limit"
            elif committed + amount > RUN_BUDGET_USD:
                policy = f"blocked: would exceed the ${RUN_BUDGET_USD} mission budget"
            else:
                committed += amount
                notes.append(f"spend ${amount:,.2f}")
        recipients = (a.get("args") or {}).get("to")
        if a["tool"] == "send_email" and recipients:
            to = recipients if isinstance(recipients, list) else [recipients]
            external = [r for r in to if not str(r).endswith(TRUSTED_EMAIL_DOMAINS)]
            if external:
                notes.append("external recipient: " + ", ".join(map(str, external)))
        governed.append({**a, "risk": risk, "amount_usd": amount or 0.0, "requires_approval": risk != connectors.RISK_READ,
                         "domain": connectors.REGISTRY[a["tool"]].domain if a["tool"] in connectors.REGISTRY else "?",
                         "policy": policy + (f" · {'; '.join(notes)}" if notes and not policy.startswith("blocked") else "")})
    emit({"type": "policy", "actions": governed, "committed_usd": round(committed, 2),
          "limits": {"per_action_usd": PAYMENT_LIMIT_USD, "mission_budget_usd": RUN_BUDGET_USD}})
    return {"actions": governed}


def human_approval(state: ActionState) -> dict[str, Any]:
    pending = [a for a in state["actions"] if a["requires_approval"] and not a["policy"].startswith("blocked")]
    if not pending:
        return {"decision": {"decision": "approve", "approved": []}}
    decision = interrupt({"type": "approval", "summary": state.get("summary", ""), "actions": state["actions"],
                          "pending": [a["id"] for a in pending]})
    return {"decision": decision if isinstance(decision, dict) else {"decision": str(decision), "approved": []}}


async def execute(state: ActionState) -> dict[str, Any]:
    run_id = _run_id(state)
    decision = state["decision"]
    approved = set(decision.get("approved", [])) if decision.get("decision") == "approve" else set()
    executed: list[dict[str, Any]] = []
    done: set[str] = set()
    for a in state["actions"]:
        allowed = not a["policy"].startswith("blocked") and (not a["requires_approval"] or a["id"] in approved)
        missing = [d for d in a.get("depends_on", []) if d not in done]
        if not allowed or missing:
            reason = (a["policy"].removeprefix("blocked: ") if a["policy"].startswith("blocked") else
                      "not approved" if not allowed else f"dependency {', '.join(missing)} did not run")
            executed.append({**a, "status": "skipped", "reason": reason, "output": None})
            continue
        key = f"{run_id}:{a['id']}"
        # Read-only steps already ran in `discover`; their recorded result is reused, not fetched again.
        res = connectors.recorded(key) if a["risk"] == connectors.RISK_READ else None
        res = res or await _call(a, run_id)
        if not res["ok"]:
            executed.append({**a, "status": "failed", "reason": res["error"][:80], "mode": res["mode"], "output": None})
            continue
        executed.append({**a, "status": "executed", "mode": res["mode"], "output": res["output"]})
        done.add(a["id"])
    return {"executed": executed}


def _amounts_covered(goal: str, actions: list[dict[str, Any]]) -> bool:
    """Goal-following check: every dollar amount the user stated must appear in the planned actions
    (executed, declined or blocked by policy). A plan that silently drops or changes an amount fails."""
    asked = {float(m.replace(",", "")) for m in re.findall(r"\$\s?(\d[\d,]*(?:\.\d+)?)", goal)}
    planned = {float(v) for a in actions for v in (a.get("args") or {}).values()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return asked <= planned


def _fmt_output(out: dict[str, Any] | None) -> str:
    if not out:
        return "-"
    if "options" in out:
        if not out["options"]:
            return "0 flight options"
        best = min(out["options"], key=lambda o: o["price_usd"])
        return f"{len(out['options'])} options, best {best['carrier']} ${best['price_usd']}"
    if "results" in out:
        if not out["results"]:
            return "0 products"
        top = out["results"][0]
        stars = f" ({top['rating']}★)" if top.get("rating") else ""
        return f"{len(out['results'])} products, top pick {top['title']} ${top['price_usd']}{stars}"
    if "slots" in out:
        return "free: " + ", ".join(out["slots"])
    if "threads" in out:
        return f"{len(out['threads'])} threads, {sum(t['unread'] for t in out['threads'])} unread"
    if "requests" in out:
        return ", ".join(f"{r['to']} owes ${r['amount_usd']}" for r in out["requests"])
    return ", ".join(f"{k}: {v}" for k, v in out.items() if not isinstance(v, (list, dict)))[:140]


def finalize(state: ActionState) -> dict[str, Any]:
    ex = state["executed"]
    rows = "\n".join(f"| {a['title']} | `{a['tool']}` | {a['risk']} | {a['status']}"
                     f"{' (' + a['reason'] + ')' if a.get('reason') else ''} | {_fmt_output(a['output'])} |" for a in ex)
    spent = sum(a.get("amount_usd", 0) for a in ex if a["status"] == "executed" and a["risk"] == connectors.RISK_PAYMENT)
    decision = state["decision"].get("decision", "approve")
    ok_ids = set(state["decision"].get("approved", []))
    honored = all(a["status"] in ("skipped", "failed") or not a["requires_approval"] or a["id"] in ok_ids
                  for a in ex)
    content = (f"{state.get('summary', '')}\n\n**Human decision:** {decision} · **Spend executed:** ${spent:,.2f} "
               f"(limit ${PAYMENT_LIMIT_USD}/action, ${RUN_BUDGET_USD}/mission)\n\n| Action | Connector | Risk | Status | "
               f"Result |\n|---|---|---|---|---|\n{rows}")
    return {"results": [{
        "agent": "action", "task_id": state["task"]["id"], "title": "Actions & approvals", "content": content,
        "artifacts": {"actions": ex, "decision": state["decision"]},
        "signals": {"approval_honored": honored, "goal_amounts_covered": _amounts_covered(state["goal"], ex),
                    "actions_executed": sum(a["status"] == "executed" for a in ex),
                    "spend_usd": round(spent, 2)},
    }]}


def build_action_graph():
    g = StateGraph(ActionState, input_schema=TaskInput, output_schema=SubgraphOutput)
    for name, fn in (("plan_actions", plan_actions), ("discover", discover), ("apply_policy", apply_policy),
                     ("human_approval", human_approval), ("execute", execute), ("finalize", finalize)):
        g.add_node(name, fn)
    g.add_edge(START, "plan_actions")
    g.add_edge("plan_actions", "discover")
    g.add_edge("discover", "apply_policy")
    g.add_edge("apply_policy", "human_approval")
    g.add_edge("human_approval", "execute")
    g.add_edge("execute", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
