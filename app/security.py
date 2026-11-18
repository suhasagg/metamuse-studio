"""Agent security controls shared by every layer.

1. Untrusted-content firewall: web results, inbox threads and other tool output are scanned for
   instruction-like text (indirect prompt injection) and neutralised before a model or memory sees them.
2. Least privilege: each agent may only call the tools on its allowlist.
3. Secret hygiene: credentials are redacted from every event, log line and stored record.
4. Tamper-evident audit log: hash-chained SQLite ledger of security-relevant events.
5. Memory-poisoning guard: instruction-like or secret-bearing text is never persisted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

UNTRUSTED_PATTERNS = [
    (r"(ignore|disregard|forget) (all |any |the )?(previous|prior|above|earlier|your) (instructions|rules|prompts?)",
     "instruction override"),
    (r"(you are|act as|pretend to be) (now )?(an? )?(dan|unrestricted|jailbroken|developer mode)", "persona hijack"),
    (r"(system|developer|assistant) ?(prompt|message)\s*:", "role spoofing"),
    (r"<\s*/?\s*(system|instructions?|tool_call|function_call)\b[^>]*>", "markup-based role spoofing"),
    (r"(send|forward|wire|transfer|pay|email|post|upload).{0,60}(password|api[_ ]?key|secret|token|credentials|\$\s?\d{3,})",
     "exfiltration or payment instruction"),
    (r"(do not|don't|never) (tell|inform|ask|alert|notify) the (user|human|owner)", "concealment instruction"),
    (r"(approve|confirm|authori[sz]e) (this|all|the|any).{0,30}(without|automatically|silently)", "approval bypass"),
    (r"(call|invoke|run|execute) the (tool|function|connector)\b", "tool-invocation instruction"),
]
SECRET_PATTERNS = [
    r"sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{16,}", r"sk_(?:live|test)_[A-Za-z0-9]{10,}", r"xox[abprs]-[A-Za-z0-9\-]{10,}",
    r"AKIA[0-9A-Z]{16}", r"AIza[0-9A-Za-z_\-]{30,}", r"gsk_[A-Za-z0-9]{20,}", r"ya29\.[A-Za-z0-9_\-.]{20,}",
    r"(?i)bearer\s+[A-Za-z0-9_\-.=]{16,}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
]
SECRET_ENV_HINTS = ("KEY", "SECRET", "TOKEN", "PASSWORD")
REMOVED = "[removed: instruction-like content]"

# Least privilege: the only tools each agent is allowed to call.
AGENT_TOOLS: dict[str, frozenset[str]] = {
    "research": frozenset({"web_search"}),
    "creative": frozenset(),
    "code": frozenset({"python_sandbox"}),
    "action": frozenset(),  # filled with the connector registry at import time by tools.connectors
}


class SecurityError(PermissionError):
    """Raised when an agent attempts something outside its privileges."""


def authorize(agent: str, tool: str) -> None:
    if tool not in AGENT_TOOLS.get(agent, frozenset()):
        raise SecurityError(f"agent '{agent}' is not allowed to call tool '{tool}'")


def scan_untrusted(text: str) -> list[str]:
    lowered = text.lower()
    return [label for pattern, label in UNTRUSTED_PATTERNS if re.search(pattern, lowered)]


def sanitize_untrusted(text: str) -> tuple[str, list[str]]:
    """Return (text with instruction-like spans removed, labels of what was found)."""
    findings = scan_untrusted(text)
    if not findings:
        return text, []
    clean = text
    for pattern, _ in UNTRUSTED_PATTERNS:
        clean = re.sub(pattern + r"[^.!?\n]*[.!?]?", REMOVED, clean, flags=re.I)
    return clean, findings


def sanitize_tree(value: Any) -> tuple[Any, list[str]]:
    """Sanitise every string inside a JSON-like tool result."""
    found: list[str] = []
    if isinstance(value, str):
        clean, f = sanitize_untrusted(value)
        return clean, f
    if isinstance(value, list):
        out = []
        for v in value:
            c, f = sanitize_tree(v)
            out.append(c)
            found += f
        return out, found
    if isinstance(value, dict):
        res = {}
        for k, v in value.items():
            c, f = sanitize_tree(v)
            res[k] = c
            found += f
        return res, found
    return value, found


def _env_secrets() -> list[str]:
    return [v for k, v in os.environ.items()
            if len(v) >= 12 and any(h in k.upper() for h in SECRET_ENV_HINTS) and not k.startswith("MUSE_")]


def redact_secrets(value: Any) -> Any:
    if isinstance(value, str):
        for secret in _env_secrets():
            value = value.replace(secret, "[SECRET]")
        for pattern in SECRET_PATTERNS:
            value = re.sub(pattern, "[SECRET]", value)
        return value
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    if isinstance(value, dict):
        return {k: redact_secrets(v) for k, v in value.items()}
    return value


def safe_to_store(text: str) -> bool:
    """Memory-poisoning guard: refuse instruction-like or secret-bearing text."""
    return not scan_untrusted(text) and redact_secrets(text) == text


class AuditLog:
    """Append-only, hash-chained ledger: each row commits to the previous row's hash."""

    def __init__(self, path: Path):
        self.path = path
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, run_id TEXT, "
                      "kind TEXT, payload TEXT, prev TEXT, hash TEXT)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    @staticmethod
    def _digest(prev: str, ts: float, run_id: str, kind: str, payload: str) -> str:
        return hashlib.sha256(f"{prev}|{ts!r}|{run_id}|{kind}|{payload}".encode()).hexdigest()

    def append(self, run_id: str, kind: str, payload: dict[str, Any]) -> str:
        body = json.dumps(redact_secrets(payload), sort_keys=True, default=str)[:4000]
        ts = time.time()
        with self._conn() as c:
            row = c.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            prev = row[0] if row else "genesis"
            digest = self._digest(prev, ts, run_id, kind, body)
            c.execute("INSERT INTO audit (ts, run_id, kind, payload, prev, hash) VALUES (?, ?, ?, ?, ?, ?)",
                      (ts, run_id, kind, body, prev, digest))
        return digest

    def verify(self) -> dict[str, Any]:
        prev, n = "genesis", 0
        with self._conn() as c:
            for rid, ts, run_id, kind, payload, p, h in c.execute(
                    "SELECT id, ts, run_id, kind, payload, prev, hash FROM audit ORDER BY id"):
                if p != prev or self._digest(prev, ts, run_id, kind, payload) != h:
                    return {"valid": False, "entries": n, "broken_at": rid}
                prev, n = h, n + 1
        return {"valid": True, "entries": n, "head": prev}

    def tail(self, limit: int = 50, run_id: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT id, ts, run_id, kind, payload, hash FROM audit"
        args: tuple[Any, ...] = ()
        if run_id:
            q, args = q + " WHERE run_id = ?", (run_id,)
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [{"id": r[0], "ts": r[1], "run_id": r[2], "kind": r[3], "payload": json.loads(r[4]), "hash": r[5][:16]}
                for r in rows]


_audit: AuditLog | None = None


def get_audit() -> AuditLog:
    global _audit
    if _audit is None:
        from .config import get_settings

        _audit = AuditLog(get_settings().data_dir / "audit.sqlite")
    return _audit


CONTROLS = [
    {"id": "input_guard", "name": "Request screening", "detail": "Rules plus model classifier block injection, "
     "jailbreaks, exfiltration and harmful requests before any agent runs."},
    {"id": "pii", "name": "PII redaction", "detail": "Emails, phone numbers and card numbers are masked before prompts."},
    {"id": "untrusted", "name": "Untrusted-content firewall", "detail": "Web, inbox and tool output is scanned for "
     "indirect prompt injection and neutralised before reaching a model or memory."},
    {"id": "least_privilege", "name": "Least-privilege agents", "detail": "Each agent can only call its own tools; "
     "unknown tools get the strictest risk tier."},
    {"id": "hitl", "name": "Human approval and spend limits", "detail": "Sending, booking and paying wait for your approval; "
     "$500 per payment and $1,000 per mission caps are enforced in code."},
    {"id": "sandbox", "name": "Code sandbox", "detail": "AST policy on code and tests, separate process, CPU/memory/time caps."},
    {"id": "secrets", "name": "Secret redaction", "detail": "Credentials are stripped from events, logs, memory and audit."},
    {"id": "memory_guard", "name": "Memory-poisoning guard", "detail": "Instruction-like or secret-bearing text is "
     "never written to long-term memory."},
    {"id": "audit", "name": "Tamper-evident audit log", "detail": "Hash-chained SQLite ledger of approvals, tool calls "
     "and security findings."},
    {"id": "api", "name": "API hardening", "detail": "Optional bearer token, per-client rate limit, security headers."},
]
