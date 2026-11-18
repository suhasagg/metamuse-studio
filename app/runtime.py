"""Run manager: executes graph runs in the background, fans events out to SSE subscribers,
aggregates telemetry, and resumes runs paused on human-approval interrupts."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import Command

from .providers import STRATEGIES
from .security import get_audit, redact_secrets

TERMINAL = {"completed", "failed", "blocked"}
AUDITED = {"status", "security", "security_finding", "policy", "approval_required", "approval_decision", "recovered"}
PERSISTED = ("goal", "workflow", "strategy", "session_id", "status", "created_at", "finished_at", "events",
             "pending_approval", "final", "muse_score", "critique", "results", "plan", "security", "error", "llm_calls",
             "tool_calls", "tokens", "cost_usd", "resume_decision")


class RunStore:
    """Durable run registry so history, telemetry and pending approvals survive a restart."""

    def __init__(self, path: Any):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, updated REAL, data TEXT)")

    def save(self, run: RunRecord) -> None:
        data = redact_secrets({k: getattr(run, k) for k in PERSISTED} | {"providers": dict(run.providers)})
        with self.conn as c:
            c.execute("INSERT OR REPLACE INTO runs (id, updated, data) VALUES (?, ?, ?)",
                      (run.id, time.time(), json.dumps(data, default=str)))

    def load(self) -> list[RunRecord]:
        runs = []
        for rid, data in self.conn.execute("SELECT id, data FROM runs ORDER BY updated"):
            d = json.loads(data)
            providers = Counter(d.pop("providers", {}))
            runs.append(RunRecord(id=rid, providers=providers, **{k: d[k] for k in PERSISTED if k in d}))
        return runs


@dataclass
class RunRecord:
    id: str
    goal: str
    workflow: str
    strategy: str
    session_id: str = "default"
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    pending_approval: dict[str, Any] | None = None
    resume_decision: dict[str, Any] | None = None  # accepted approval not yet applied to a checkpoint
    final: str = ""
    muse_score: dict[str, Any] = field(default_factory=dict)
    critique: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    plan: list[dict[str, Any]] = field(default_factory=list)
    security: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    providers: Counter = field(default_factory=Counter)

    def metrics(self) -> dict[str, Any]:
        end = self.finished_at or time.time()
        return {"llm_calls": self.llm_calls, "tool_calls": self.tool_calls, "tokens": self.tokens,
                "cost_usd": round(self.cost_usd, 6), "duration_s": round(end - self.created_at, 2),
                "providers": dict(self.providers)}

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "goal": self.goal, "workflow": self.workflow, "strategy": self.strategy,
                "session_id": self.session_id, "status": self.status, "created_at": self.created_at,
                "muse_score": self.muse_score, "metrics": self.metrics()}

    def detail(self) -> dict[str, Any]:
        return redact_secrets(self._detail())

    def _detail(self) -> dict[str, Any]:
        return {**self.summary(), "final": self.final, "critique": self.critique, "results": self.results,
                "plan": self.plan, "security": self.security, "pending_approval": self.pending_approval,
                "error": self.error, "events": self.events}


class RunManager:
    def __init__(self, graph: Any, store: RunStore | None = None):
        self.graph = graph
        self.store = store
        self.runs: dict[str, RunRecord] = {}
        self._tasks: set[asyncio.Task] = set()

    async def restore(self) -> dict[str, int]:
        """Reload persisted runs. Runs paused on approval stay resumable; runs that were mid-flight when the
        process died continue from their last LangGraph checkpoint instead of starting over."""
        stats = {"loaded": 0, "recovered": 0}
        for run in self.store.load() if self.store else []:
            self.runs[run.id] = run
            stats["loaded"] += 1
            if run.status in TERMINAL or (run.status == "awaiting_approval" and not run.resume_decision):
                continue
            snapshot = await self.graph.aget_state(self._config(run), subgraphs=True)
            stats["recovered"] += 1
            if run.resume_decision and _collect_interrupts(snapshot):
                # The approval was accepted but the process died before the graph consumed it: replay it.
                self._publish(run, {"type": "recovered", "from": "accepted approval", "next": ["execute"]})
                self._spawn(self._execute(run, Command(resume=run.resume_decision)))
            elif snapshot.next:
                self._publish(run, {"type": "recovered", "from": "checkpoint", "next": list(snapshot.next)})
                self._spawn(self._execute(run, None))
            elif snapshot.values:
                await self._after_stream(run)
            else:
                self._publish(run, {"type": "recovered", "from": "start", "next": ["recall"]})
                self._spawn(self._execute(run, {"goal": run.goal, "workflow": run.workflow, "iteration": 0}))
        return stats

    # ---- public API ------------------------------------------------------------------------
    def start(self, goal: str, workflow: str = "auto", strategy: str = "balanced",
              session_id: str = "default") -> RunRecord:
        run = RunRecord(id=uuid.uuid4().hex[:12], goal=goal.strip(), workflow=workflow, session_id=session_id,
                        strategy=strategy if strategy in STRATEGIES else "balanced")
        self.runs[run.id] = run
        self._spawn(self._execute(run, {"goal": run.goal, "workflow": workflow, "iteration": 0}))
        return run

    def resume(self, run_id: str, decision: dict[str, Any]) -> RunRecord:
        run = self.runs[run_id]
        if run.status != "awaiting_approval":
            raise ValueError(f"run {run_id} is not awaiting approval (status={run.status})")
        run.pending_approval = None
        run.resume_decision = decision
        self._publish(run, {"type": "approval_decision", **decision})
        self._spawn(self._execute(run, Command(resume=decision)))
        return run

    async def wait(self, run_id: str, timeout: float = 60.0) -> RunRecord:
        """Block until the run is terminal or paused (used by tests and the CLI)."""
        run = self.runs[run_id]
        deadline = time.monotonic() + timeout
        while run.status not in TERMINAL | {"awaiting_approval"} and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        return run

    def subscribe(self, run_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.runs[run_id].subscribers.add(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        self.runs[run_id].subscribers.discard(q)

    # ---- internals -------------------------------------------------------------------------
    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _publish(self, run: RunRecord, event: dict[str, Any]) -> None:
        event = redact_secrets({"ts": round(time.time(), 3), "seq": len(run.events), **event})
        run.events.append(event)
        for q in list(run.subscribers):
            q.put_nowait(event)
        kind = event.get("type")
        if kind in AUDITED or (kind == "tool" and event.get("status") == "done"):
            slim = {k: v for k, v in event.items() if k in ("status", "tool", "input", "backend", "decision", "approved",
                                                            "blocked", "risk", "findings", "source", "pending",
                                                            "committed_usd", "error", "from", "agent")}
            get_audit().append(run.id, str(kind), slim)
        if self.store and kind in ("status", "approval_required", "approval_decision"):
            self.store.save(run)

    def _config(self, run: RunRecord) -> dict[str, Any]:
        return {"configurable": {"thread_id": run.id, "strategy": run.strategy, "session_id": run.session_id},
                "recursion_limit": 80}

    async def _execute(self, run: RunRecord, payload: Any) -> None:
        run.status = "running"
        self._publish(run, {"type": "status", "status": "running", "goal": run.goal, "workflow": run.workflow,
                            "strategy": run.strategy})
        try:
            async for ns, mode, chunk in self.graph.astream(payload, self._config(run), stream_mode=["tasks", "custom"],
                                                            subgraphs=True):
                graph = ns[0].split(":")[0] if ns else "main"
                if mode == "tasks":
                    if chunk.get("name", "").startswith("__"):
                        continue
                    status = "start" if "input" in chunk else ("error" if chunk.get("error") else "end")
                    self._publish(run, {"type": "node", "graph": graph, "node": chunk["name"], "status": status,
                                        "error": str(chunk.get("error") or "") or None})
                else:
                    self._track(run, chunk)
                    self._publish(run, {**chunk, "graph": graph})
            await self._after_stream(run)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            run.status, run.error, run.finished_at = "failed", f"{type(exc).__name__}: {exc}", time.time()
            self._publish(run, {"type": "status", "status": "failed", "error": run.error, "metrics": run.metrics()})

    def _track(self, run: RunRecord, chunk: dict[str, Any]) -> None:
        kind = chunk.get("type")
        if kind == "llm":
            run.llm_calls += 1
            run.tokens += chunk.get("input_tokens", 0) + chunk.get("output_tokens", 0)
            run.cost_usd += chunk.get("cost_usd", 0.0)
            run.providers[chunk.get("provider", "?")] += 1
        elif kind == "tool" and chunk.get("status") == "done":
            run.tool_calls += 1
        elif kind == "plan":
            run.plan = chunk.get("tasks", [])
        elif kind == "security":
            run.security = {k: v for k, v in chunk.items() if k != "type"}

    async def _after_stream(self, run: RunRecord) -> None:
        cfg = self._config(run)
        snapshot = await self.graph.aget_state(cfg, subgraphs=True)
        interrupts = _collect_interrupts(snapshot)
        values = snapshot.values
        run.resume_decision = None
        if interrupts:
            run.status = "awaiting_approval"
            run.pending_approval = interrupts[0]
            self._publish(run, {**interrupts[0], "type": "approval_required"})
            self._publish(run, {"type": "status", "status": "awaiting_approval", "metrics": run.metrics()})
            return
        run.final = values.get("final", "")
        run.muse_score = values.get("muse_score", {})
        run.critique = values.get("critique", {})
        from .agents.supervisor import latest_results

        run.results = latest_results(values.get("results", []))
        run.finished_at = time.time()
        run.status = "blocked" if values.get("security", {}).get("blocked") else "completed"
        self._publish(run, {"type": "status", "status": run.status, "final": run.final, "muse_score": run.muse_score,
                            "critique": run.critique, "results": run.results, "metrics": run.metrics()})


def _collect_interrupts(snapshot: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for task in getattr(snapshot, "tasks", ()) or ():
        for intr in getattr(task, "interrupts", ()) or ():
            if isinstance(intr.value, dict):
                found.append(intr.value)
        child = getattr(task, "state", None)
        if child is not None and hasattr(child, "tasks"):
            for value in _collect_interrupts(child):
                if value not in found:
                    found.append(value)
    return found
