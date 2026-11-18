"""Six-layer agent memory, all persisted in SQLite.

| # | Layer      | Scope            | Holds                                              |
|---|------------|------------------|----------------------------------------------------|
| 1 | working    | one run (thread) | LangGraph state, checkpointed after every step     |
| 2 | session    | one conversation | recent turns of the current session (short-term)   |
| 3 | episodic   | all time         | what happened in past missions and how they scored |
| 4 | semantic   | all time         | verified facts with their sources                  |
| 5 | procedural | all time         | plans and code skills that worked, with win rates  |
| 6 | profile    | the user         | contacts, budgets, approval habits, preferences    |

Layer 1 lives in the LangGraph checkpointer (`checkpoints.sqlite`); layers 2-6 live in `memory.sqlite`.
Everything written passes the memory-poisoning guard in `app.security`.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from .security import redact_secrets, safe_to_store

LAYERS = [
    ("working", "Working memory", "this task", "What I am doing right now, saved after every step"),
    ("session", "Conversation memory", "this conversation", "What we did earlier in this conversation"),
    ("episodic", "Past tasks", "all time", "Tasks I finished before and how they went"),
    ("semantic", "Knowledge", "all time", "Facts I verified, with their sources"),
    ("procedural", "Skills", "all time", "Plans and code that worked before"),
    ("profile", "About you", "you", "Your contacts, budgets and what you usually approve"),
]
SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, run_id TEXT, goal TEXT,
  summary TEXT, score REAL);
CREATE TABLE IF NOT EXISTS session_turns (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, session_id TEXT, run_id TEXT,
  workflow TEXT, goal TEXT, outcome TEXT, score REAL);
CREATE INDEX IF NOT EXISTS idx_session ON session_turns (session_id, id);
CREATE TABLE IF NOT EXISTS facts (url TEXT PRIMARY KEY, ts REAL, title TEXT, snippet TEXT, source TEXT,
  run_id TEXT, hits INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS procedures (key TEXT PRIMARY KEY, ts REAL, kind TEXT, name TEXT, body TEXT,
  uses INTEGER DEFAULT 0, wins INTEGER DEFAULT 0, best_score REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS profile (key TEXT PRIMARY KEY, ts REAL, kind TEXT, value TEXT, weight INTEGER DEFAULT 1);
"""
WIN_SCORE = 100.0


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", text.lower())}


def _rank(query: str, rows: list[dict[str, Any]], fields: tuple[str, ...], k: int) -> list[dict[str, Any]]:
    q = _terms(query)
    scored = [(len(q & _terms(" ".join(str(r.get(f, "")) for f in fields))), r) for r in rows]
    return [r for s, r in sorted(scored, key=lambda x: -x[0]) if s][:k]


class MemoryStore:
    def __init__(self, path: Path, checkpoint_path: Path | None = None):
        self.path = path
        self.checkpoint_path = checkpoint_path
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _rows(self, sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    # ---- layer 3: episodic (kept API-compatible with the original single-layer store) -------
    def add(self, run_id: str, goal: str, summary: str, score: float) -> bool:
        if not safe_to_store(goal + " " + summary):
            return False
        with self._conn() as c:
            c.execute("INSERT INTO memories (ts, run_id, goal, summary, score) VALUES (?, ?, ?, ?, ?)",
                      (time.time(), run_id, goal, summary, score))
        return True

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._rows("SELECT ts, run_id, goal, summary, score FROM memories ORDER BY id DESC LIMIT ?", (limit,))

    def recall(self, goal: str, k: int = 3) -> list[dict[str, Any]]:
        return _rank(goal, self.recent(200), ("goal", "summary"), k)

    # ---- layer 2: session --------------------------------------------------------------------
    def session_turns(self, session_id: str, limit: int = 5) -> list[dict[str, Any]]:
        return self._rows("SELECT ts, run_id, workflow, goal, outcome, score FROM session_turns WHERE session_id = ? "
                          "ORDER BY id DESC LIMIT ?", (session_id, limit))

    # ---- layer 4: semantic -------------------------------------------------------------------
    def add_facts(self, run_id: str, sources: list[dict[str, Any]]) -> int:
        n = 0
        with self._conn() as c:
            for s in sources:
                title, snippet, url = str(s.get("title", "")), str(s.get("snippet", "")), str(s.get("url", ""))
                if not url or not snippet or not safe_to_store(title + " " + snippet):
                    continue
                c.execute("INSERT INTO facts (url, ts, title, snippet, source, run_id) VALUES (?, ?, ?, ?, ?, ?) "
                          "ON CONFLICT(url) DO UPDATE SET hits = hits + 1, ts = excluded.ts",
                          (url, time.time(), title, snippet[:400], str(s.get("source", "")), run_id))
                n += 1
        return n

    def recall_facts(self, goal: str, k: int = 4) -> list[dict[str, Any]]:
        rows = self._rows("SELECT url, title, snippet, source, hits FROM facts ORDER BY ts DESC LIMIT 400")
        return _rank(goal, rows, ("title", "snippet"), k)

    # ---- layer 5: procedural -----------------------------------------------------------------
    def _learn(self, key: str, kind: str, name: str, body: Any, score: float) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO procedures (key, ts, kind, name, body, uses, wins, best_score) "
                      "VALUES (?, ?, ?, ?, ?, 1, ?, ?) "
                      "ON CONFLICT(key) DO UPDATE SET uses = uses + 1, wins = wins + excluded.wins, ts = excluded.ts, "
                      "best_score = MAX(best_score, excluded.best_score), "
                      "body = CASE WHEN excluded.best_score >= best_score THEN excluded.body ELSE body END",
                      (key, time.time(), kind, name, json.dumps(body), int(score >= WIN_SCORE), score))

    def recall_procedures(self, workflow: str, goal: str, k: int = 3) -> list[dict[str, Any]]:
        rows = self._rows("SELECT key, kind, name, body, uses, wins, best_score FROM procedures ORDER BY wins DESC, ts DESC")
        for r in rows:
            r["body"] = json.loads(r["body"])
        plans = [r for r in rows if r["kind"] == "plan" and r["key"].startswith(f"plan:{workflow}:")][:1]
        return plans + _rank(goal, [r for r in rows if r["kind"] == "code_skill"], ("name",), k - len(plans))

    # ---- layer 6: profile --------------------------------------------------------------------
    def _note(self, c: sqlite3.Connection, kind: str, name: str, value: str) -> None:
        c.execute("INSERT INTO profile (key, ts, kind, value) VALUES (?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
                  "weight = weight + 1, ts = excluded.ts, value = excluded.value", (f"{kind}:{name}", time.time(), kind, value))

    def profile(self, limit: int = 40) -> list[dict[str, Any]]:
        return self._rows("SELECT key, kind, value, weight FROM profile ORDER BY weight DESC, ts DESC LIMIT ?", (limit,))

    # ---- read path: one call that assembles context from every layer -----------------------
    def recall_all(self, goal: str, session_id: str, workflow: str) -> dict[str, list[dict[str, Any]]]:
        return {"session": self.session_turns(session_id), "episodic": self.recall(goal),
                "semantic": self.recall_facts(goal), "procedural": self.recall_procedures(workflow, goal),
                "profile": self.profile(12)}

    # ---- write path: consolidate a finished run into layers 2-6 -----------------------------
    def consolidate(self, *, run_id: str, session_id: str, workflow: str, strategy: str, goal: str, summary: str,
                    score: float, plan: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, int]:
        goal, summary = redact_secrets(goal), redact_secrets(summary)
        written = dict.fromkeys(("session", "episodic", "semantic", "procedural", "profile"), 0)
        if not safe_to_store(goal + " " + summary):
            return written
        with self._conn() as c:
            c.execute("INSERT INTO session_turns (ts, session_id, run_id, workflow, goal, outcome, score) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?)", (time.time(), session_id, run_id, workflow, goal, summary, score))
        written["session"] = 1
        written["episodic"] = int(self.add(run_id, goal, summary, score))
        agents = sorted({t["agent"] for t in plan})
        if agents:
            self._learn(f"plan:{workflow}:{'+'.join(agents)}", "plan", f"{workflow}: {' + '.join(agents)}",
                        [{"agent": t["agent"], "objective": t.get("objective", "")[:160]} for t in plan], score)
            written["procedural"] += 1
        with self._conn() as c:
            self._note(c, "workflow", workflow, workflow)
            self._note(c, "strategy", strategy, strategy)
            written["profile"] += 2
            for m in re.finditer(r"under \$\s?(\d[\d,]*)", goal):
                self._note(c, "budget", "typical_cap_usd", m.group(1).replace(",", ""))
                written["profile"] += 1
            for r in results:
                art = r.get("artifacts") or {}
                for a in art.get("actions", []):
                    verdict = "approved" if a.get("status") == "executed" else "declined"
                    if a.get("requires_approval"):
                        self._note(c, "approval", f"{a['tool']}:{verdict}", verdict)
                        written["profile"] += 1
                    for who in (a.get("args") or {}).get("to", []) if isinstance((a.get("args") or {}).get("to"), list) else []:
                        self._note(c, "contact", str(who), str(who))
                        written["profile"] += 1
        for r in results:
            art, sig = r.get("artifacts") or {}, r.get("signals") or {}
            if r.get("agent") == "research":
                written["semantic"] += self.add_facts(run_id, art.get("sources", []))
            if r.get("agent") == "code" and sig.get("tests_total") and sig.get("tests_passed") == sig["tests_total"]:
                code = str(art.get("code", ""))
                fn = re.search(r"def (\w+)", code)
                if fn and safe_to_store(code):
                    self._learn(f"code:{fn.group(1)}", "code_skill", fn.group(1), {"code": code[:2000]}, score)
                    written["procedural"] += 1
        return written

    # ---- introspection for the API / UI ------------------------------------------------------
    def _checkpoints(self, thread_id: str | None) -> int:
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            return 0
        try:
            with sqlite3.connect(f"file:{self.checkpoint_path}?mode=ro", uri=True) as c:
                if thread_id:
                    return c.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread_id,)).fetchone()[0]
                return c.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        except sqlite3.Error:
            return 0

    def overview(self, session_id: str = "default", run_id: str | None = None) -> list[dict[str, Any]]:
        def count(table: str, where: str = "", args: tuple[Any, ...] = ()) -> int:
            return self._rows(f"SELECT COUNT(*) AS n FROM {table} {where}", args)[0]["n"]  # noqa: S608 - fixed table names

        procs = self._rows("SELECT kind, name, uses, wins, best_score FROM procedures ORDER BY wins DESC, ts DESC LIMIT 6")
        data = {
            "working": (self._checkpoints(run_id), [{"text": f"{self._checkpoints(run_id)} saved steps "
                                                    + ("in the current task" if run_id else "across all tasks")}]),
            "session": (count("session_turns", "WHERE session_id = ?", (session_id,)),
                        [{"text": t["goal"][:90], "meta": time.strftime("%b %d, %H:%M", time.localtime(t["ts"]))}
                         for t in self.session_turns(session_id, 4)]),
            "episodic": (count("memories"), [{"text": m["summary"][:110], "meta": time.strftime("%b %d", time.localtime(m["ts"]))}
                                             for m in self.recent(4)]),
            "semantic": (count("facts"), [{"text": f["title"], "meta": f"{f['source']} · {f['hits']}×"} for f in self._rows(
                "SELECT title, source, hits FROM facts ORDER BY hits DESC, ts DESC LIMIT 5")]),
            "procedural": (count("procedures"), [{"text": p["name"], "meta": f"{p['kind'].replace('_', ' ')} · "
                                                  f"{p['wins']}/{p['uses']} wins"} for p in procs]),
            "profile": (count("profile"), [{"text": p["key"].replace(":", " · "), "meta": f"{p['weight']}×"}
                                           for p in self.profile(6)]),
        }
        return [{"id": i, "layer": n + 1, "name": name, "scope": scope, "description": desc, "count": data[i][0],
                 "items": data[i][1]} for n, (i, name, scope, desc) in enumerate(LAYERS)]


_store: MemoryStore | None = None


def get_memory() -> MemoryStore:
    global _store
    if _store is None:
        from .config import get_settings

        s = get_settings()
        _store = MemoryStore(s.memory_path, s.checkpoint_path)
    return _store


def set_memory(store: MemoryStore) -> None:
    global _store
    _store = store
