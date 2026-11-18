"""Policy-checked Python sandbox used by the Code subgraph to execute generated code and tests."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

BLOCKED_MODULES = {"os", "sys", "subprocess", "socket", "shutil", "ctypes", "multiprocessing", "pathlib",
                   "requests", "httpx", "urllib", "http", "importlib", "signal", "threading", "asyncio", "pickle",
                   "io", "builtins", "tempfile", "glob", "fnmatch", "sqlite3", "ssl", "ftplib", "smtplib", "telnetlib",
                   "webbrowser", "runpy", "code", "codeop", "pty", "fcntl", "mmap", "resource", "gc", "inspect",
                   "marshal", "shelve", "dbm", "zipfile", "tarfile", "selectors", "select", "posix", "_io", "_thread"}
BLOCKED_CALLS = {"eval", "exec", "open", "__import__", "compile", "input", "breakpoint", "globals", "vars", "locals",
                 "getattr", "setattr", "delattr", "memoryview"}

RUNNER = r'''
import ast, json
NS = {}
exec(compile(SOLUTION, "solution.py", "exec"), NS)
results = []
for stmt in ast.parse(TESTS).body:
    src = ast.get_source_segment(TESTS, stmt)
    try:
        exec(compile(ast.Module([stmt], []), "tests.py", "exec"), NS)
        results.append({"test": src, "ok": True})
    except Exception as exc:  # noqa
        detail = f"{type(exc).__name__}: {exc}".strip()
        if isinstance(stmt, ast.Assert) and isinstance(stmt.test, ast.Compare):
            try:
                got = eval(compile(ast.Expression(stmt.test.left), "tests.py", "eval"), NS)
                detail = f"{type(exc).__name__}: got {got!r}"
            except Exception as inner:  # noqa
                detail = f"{type(inner).__name__}: {inner}"
        results.append({"test": src, "ok": False, "error": detail})
print("__MUSE_RESULTS__" + json.dumps(results))
'''


def policy_violations(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc.msg} (line {exc.lineno})"]
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                if name.split(".")[0] in BLOCKED_MODULES:
                    problems.append(f"import of '{name}' is not allowed in the sandbox")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_CALLS:
            problems.append(f"use of '{node.id}' is not allowed in the sandbox")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__init__":
            problems.append(f"dunder attribute access '{node.attr}' is not allowed")
    return problems


def _limits() -> None:  # pragma: no cover - runs in the child process
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    os.setsid()


async def run_python_tests(code: str, tests: str, timeout_s: float = 8.0) -> dict[str, Any]:
    started = time.perf_counter()
    # Generated tests are untrusted too, so they pass the same policy as the solution.
    violations = policy_violations(code) + [f"tests: {v}" for v in policy_violations(tests)]
    if violations:
        return {"passed": False, "total": 0, "failed": 0, "results": [], "policy_violations": violations,
                "stderr": "", "duration_ms": 0.0}
    with tempfile.TemporaryDirectory(prefix="muse-sbx-") as tmp:
        script = Path(tmp) / "runner.py"
        script.write_text(f"SOLUTION = {code!r}\nTESTS = {tests!r}\n{RUNNER}")
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", str(script), cwd=tmp, env={"PATH": "/usr/bin:/bin"},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, preexec_fn=_limits)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            proc.kill()
            return {"passed": False, "total": 0, "failed": 0, "results": [], "policy_violations": [],
                    "stderr": f"Timed out after {timeout_s}s", "duration_ms": timeout_s * 1000}
    stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
    results: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if line.startswith("__MUSE_RESULTS__"):
            results = json.loads(line[len("__MUSE_RESULTS__"):])
    failed = sum(1 for r in results if not r["ok"])
    return {"passed": bool(results) and failed == 0, "total": len(results), "failed": failed, "results": results,
            "policy_violations": [], "stderr": stderr[-1500:], "duration_ms": round((time.perf_counter() - started) * 1000, 1)}
