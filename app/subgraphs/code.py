"""Code subgraph: spec + tests -> implement -> sandbox execution -> self-healing debug loop."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .. import security
from ..agents.llm import ask, emit, extract_code
from ..config import get_settings
from ..providers.mock_provider import pick_code_task
from ..state import AgentResult, SubgraphOutput, TaskInput
from ..tools.sandbox import run_python_tests

MAX_ATTEMPTS = 3


class CodeState(TypedDict, total=False):
    task: dict[str, Any]
    goal: str
    spec: dict[str, Any]
    code: str
    report: dict[str, Any]
    attempts: int
    history: list[dict[str, Any]]
    results: Annotated[list[AgentResult], operator.add]


async def specify(state: CodeState) -> dict[str, Any]:
    default = {k: v for k, v in pick_code_task(state["goal"]).items() if k in ("function_name", "signature",
                                                                                 "description", "tests")}
    spec = await ask("code", "code.spec",
                     "You are a staff engineer. Turn the request into a precise spec for ONE pure Python function "
                     "plus executable assert-based tests (no imports beyond the standard library 're', "
                     "'collections', 'math', 'itertools'). Return JSON: {\"function_name\": str, \"signature\": str, "
                     "\"description\": str, \"tests\": str}",
                     f"Request: {state['task']['objective']}", context={"goal": state["goal"]}, json_mode=True,
                     fallback=default)
    if not isinstance(spec, dict) or not spec.get("tests"):
        spec = default
    return {"spec": spec, "attempts": 0, "history": []}


async def implement(state: CodeState) -> dict[str, Any]:
    spec = state["spec"]
    text = await ask("code", "code.implement",
                     "Write the Python implementation only, in one ```python block. Pure function, type hints, "
                     "docstring, no I/O.", f"Spec: {spec['signature']}\n{spec['description']}\nTests:\n{spec['tests']}",
                     context={"goal": state["goal"], "spec": spec})
    return {"code": extract_code(text)}


async def run_tests(state: CodeState) -> dict[str, Any]:
    emit({"type": "tool", "agent": "code", "tool": "python_sandbox", "status": "running",
          "input": f"attempt {state.get('attempts', 0) + 1}"})
    security.authorize("code", "python_sandbox")
    report = await run_python_tests(state["code"], state["spec"]["tests"], get_settings().sandbox_timeout_s)
    emit({"type": "tool", "agent": "code", "tool": "python_sandbox", "status": "done",
          "output": f"{report['total'] - report['failed']}/{report['total']} tests passed"
                    + (f"; policy: {report['policy_violations']}" if report["policy_violations"] else ""),
          "passed": report["passed"]})
    attempts = state.get("attempts", 0) + 1
    history = state.get("history", []) + [{"attempt": attempts, "passed": report["passed"],
                                           "failed": report["failed"], "total": report["total"]}]
    return {"report": report, "attempts": attempts, "history": history}


def after_tests(state: CodeState) -> str:
    if state["report"]["passed"] or state["attempts"] >= MAX_ATTEMPTS:
        return "finalize"
    return "debug"


async def debug(state: CodeState) -> dict[str, Any]:
    r = state["report"]
    failures = "\n".join(f"- {x['test']} -> {x.get('error')}" for x in r["results"] if not x["ok"])
    problems = failures or "\n".join(r["policy_violations"]) or r["stderr"]
    text = await ask("code", "code.debug",
                     "You are debugging. Explain the root cause in one line starting 'Diagnosis:', then give the "
                     "complete corrected implementation in one ```python block.",
                     f"Spec: {state['spec']['signature']}\nCode:\n{state['code']}\nFailures:\n{problems}",
                     context={"goal": state["goal"], "failures": problems})
    diagnosis = next((ln for ln in text.splitlines() if ln.lower().startswith("diagnosis")), "")
    emit({"type": "log", "level": "info", "agent": "code", "message": diagnosis or "Applied fix"})
    return {"code": extract_code(text)}


def finalize(state: CodeState) -> dict[str, Any]:
    r, spec = state["report"], state["spec"]
    status = "PASSED" if r["passed"] else "FAILED"
    journey = " -> ".join(f"{h['total'] - h['failed']}/{h['total']}" for h in state["history"])
    content = (f"**Function:** `{spec['signature']}`  \n{spec['description']}\n\n```python\n{state['code']}```\n\n"
               f"**Sandbox result:** {status}, {r['total'] - r['failed']}/{r['total']} tests in "
               f"{state['attempts']} attempt(s) (progress {journey}).\n\n<details><summary>Tests</summary>\n\n"
               f"```python\n{spec['tests']}```\n</details>")
    return {"results": [{
        "agent": "code", "task_id": state["task"]["id"], "title": "Verified implementation", "content": content,
        "artifacts": {"code": state["code"], "tests": spec["tests"], "report": r, "history": state["history"]},
        "signals": {"tests_passed": r["passed"], "tests_total": r["total"], "code_attempts": state["attempts"]},
    }]}


def build_code_graph():
    g = StateGraph(CodeState, input_schema=TaskInput, output_schema=SubgraphOutput)
    for name, fn in (("specify", specify), ("implement", implement), ("run_tests", run_tests),
                     ("debug", debug), ("finalize", finalize)):
        g.add_node(name, fn)
    g.add_edge(START, "specify")
    g.add_edge("specify", "implement")
    g.add_edge("implement", "run_tests")
    g.add_conditional_edges("run_tests", after_tests, ["debug", "finalize"])
    g.add_edge("debug", "run_tests")
    g.add_edge("finalize", END)
    return g.compile()
