"""muse-local: a deterministic, offline provider.

It lets every workflow run end-to-end with no API keys (CI, demos, air-gapped installs). Output
depends only on the request, so tests are reproducible. Behaviour is task-aware: it returns
structured JSON for planner/critic tasks and writes grounded prose from the real search results
in `request.context`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import date
from typing import Any

from .base import LLMProvider, LLMRequest, LLMResponse, ProviderProfile, estimate_tokens

STOPWORDS = set("""a an the and or for to of in on with by from into about please write build create make plan
design give me my our we i you it this that python function which can should will help us need want an app
using use some new best how what is are be as at""".split())


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def topic_of(goal: str, words: int = 6) -> str:
    tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-']+", goal)]
    keep = [t for t in tokens if t.lower() not in STOPWORDS]
    return " ".join(keep[:words]) or goal[:40]


GENERIC = set("day days week weeks trip plan launch campaign function app product team".split())
CALENDAR_WORDS = set("""january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday""".split())
INTENT_SUFFIX = (("trip", "tourism"), ("travel", "tourism"), ("visit", "tourism"), ("launch", "market"),
                 ("campaign", "marketing"), ("startup", "industry"), ("code", "algorithm"), ("function", "algorithm"),
                 ("buy", "shopping"), ("gift", "shopping"), ("order", "shopping"))


def proper_nouns(goal: str) -> list[str]:
    words = re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", goal)
    return list(dict.fromkeys(w for w in words if w.lower() not in STOPWORDS))


def subject_of(goal: str) -> str:
    m = re.search(r"\bfor (?:an?|the|my|our) ([^.,;:]+)", goal, re.I)
    if not m:
        return topic_of(goal, 4)
    return re.split(r"\s+(?:called|named|with|and|that|to|in)\s+", m.group(1).strip())[0]


def brand_of(goal: str) -> str:
    named = re.search(r"\b(?:called|named)\s+([A-Z][\w-]+)", goal)
    if named:
        return named.group(1)
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z-]{2,}", subject_of(goal)) if w.lower() not in STOPWORDS | GENERIC]
    if words:
        return max(words, key=len).split("-")[0].capitalize()
    nouns = proper_nouns(goal)
    return nouns[0] if nouns else "Muse"


def keywords(goal: str, n: int = 4) -> list[str]:
    seen: list[str] = []
    for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", goal.lower()):
        if t not in STOPWORDS and len(t) > 2 and t not in seen:
            seen.append(t)
    return seen[:n] or ["innovation"]


# --------------------------------------------------------------------------------------------
# Code catalog: each entry has a deliberately flawed first draft so the self-healing loop
# (implement -> sandbox -> debug) is exercised deterministically.
# --------------------------------------------------------------------------------------------
CODE_CATALOG: dict[str, dict[str, str]] = {
    "roman": {
        "function_name": "to_roman",
        "signature": "def to_roman(n: int) -> str",
        "description": "Convert an integer in [1, 3999] to a Roman numeral, raising ValueError outside the range.",
        "tests": (
            "assert to_roman(1) == 'I'\nassert to_roman(4) == 'IV'\nassert to_roman(9) == 'IX'\n"
            "assert to_roman(14) == 'XIV'\nassert to_roman(40) == 'XL'\nassert to_roman(1994) == 'MCMXCIV'\n"
            "assert to_roman(3999) == 'MMMCMXCIX'\n"
            "try:\n    to_roman(0)\n    raise AssertionError('expected ValueError')\nexcept ValueError:\n    pass\n"
        ),
        "buggy": (
            "def to_roman(n: int) -> str:\n"
            "    \"\"\"Convert an integer to a Roman numeral.\"\"\"\n"
            "    symbols = [(1000, 'M'), (500, 'D'), (100, 'C'), (50, 'L'), (10, 'X'), (5, 'V'), (1, 'I')]\n"
            "    out = []\n"
            "    for value, sym in symbols:\n"
            "        while n >= value:\n"
            "            out.append(sym)\n"
            "            n -= value\n"
            "    return ''.join(out)\n"
        ),
        "fixed": (
            "def to_roman(n: int) -> str:\n"
            "    \"\"\"Convert an integer in [1, 3999] to a Roman numeral.\"\"\"\n"
            "    if not isinstance(n, int) or not 1 <= n <= 3999:\n"
            "        raise ValueError('n must be an integer in [1, 3999]')\n"
            "    symbols = [\n"
            "        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),\n"
            "        (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'),\n"
            "    ]\n"
            "    out = []\n"
            "    for value, sym in symbols:\n"
            "        count, n = divmod(n, value)\n"
            "        out.append(sym * count)\n"
            "    return ''.join(out)\n"
        ),
        "diagnosis": "Subtractive pairs (CM, CD, XC, XL, IX, IV) were missing, so 4 rendered as 'IIII', "
                     "and out-of-range input was not validated.",
    },
    "palindrome": {
        "function_name": "is_palindrome",
        "signature": "def is_palindrome(text: str) -> bool",
        "description": "Return True when text reads the same forwards and backwards, ignoring case, spaces and punctuation.",
        "tests": (
            "assert is_palindrome('racecar')\nassert is_palindrome('A man, a plan, a canal: Panama')\n"
            "assert not is_palindrome('meta muse')\nassert is_palindrome('')\nassert is_palindrome('No lemon, no melon')\n"
        ),
        "buggy": "def is_palindrome(text: str) -> bool:\n    return text == text[::-1]\n",
        "fixed": (
            "def is_palindrome(text: str) -> bool:\n"
            "    \"\"\"Case- and punctuation-insensitive palindrome check.\"\"\"\n"
            "    cleaned = [c.lower() for c in text if c.isalnum()]\n"
            "    return cleaned == cleaned[::-1]\n"
        ),
        "diagnosis": "The draft compared raw text, so case, whitespace and punctuation broke the check.",
    },
    "fibonacci": {
        "function_name": "fibonacci",
        "signature": "def fibonacci(n: int) -> list[int]",
        "description": "Return the first n Fibonacci numbers starting from 0; n < 0 raises ValueError.",
        "tests": (
            "assert fibonacci(0) == []\nassert fibonacci(1) == [0]\nassert fibonacci(2) == [0, 1]\n"
            "assert fibonacci(10) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n"
            "try:\n    fibonacci(-1)\n    raise AssertionError('expected ValueError')\nexcept ValueError:\n    pass\n"
        ),
        "buggy": (
            "def fibonacci(n: int) -> list[int]:\n    seq = [0, 1]\n    while len(seq) < n:\n"
            "        seq.append(seq[-1] + seq[-2])\n    return seq\n"
        ),
        "fixed": (
            "def fibonacci(n: int) -> list[int]:\n"
            "    \"\"\"Return the first n Fibonacci numbers.\"\"\"\n"
            "    if n < 0:\n        raise ValueError('n must be non-negative')\n"
            "    seq: list[int] = []\n    a, b = 0, 1\n    for _ in range(n):\n"
            "        seq.append(a)\n        a, b = b, a + b\n    return seq\n"
        ),
        "diagnosis": "The draft always seeded [0, 1], returning two items for n < 2, and accepted negative n.",
    },
    "word_frequency": {
        "function_name": "top_words",
        "signature": "def top_words(text: str, k: int = 3) -> list[tuple[str, int]]",
        "description": "Return the k most frequent words (case-insensitive, punctuation stripped), ties broken alphabetically.",
        "tests": (
            "assert top_words('the cat and the hat', 1) == [('the', 2)]\n"
            "assert top_words('Muse muse MUSE agent agent graph', 2) == [('muse', 3), ('agent', 2)]\n"
            "assert top_words('b a c', 3) == [('a', 1), ('b', 1), ('c', 1)]\n"
            "assert top_words('', 3) == []\nassert top_words('Hello, hello! world.', 2) == [('hello', 2), ('world', 1)]\n"
        ),
        "buggy": (
            "def top_words(text: str, k: int = 3) -> list[tuple[str, int]]:\n    counts = {}\n"
            "    for w in text.split():\n        counts[w] = counts.get(w, 0) + 1\n"
            "    return sorted(counts.items(), key=lambda kv: -kv[1])[:k]\n"
        ),
        "fixed": (
            "import re\nfrom collections import Counter\n\n\n"
            "def top_words(text: str, k: int = 3) -> list[tuple[str, int]]:\n"
            "    \"\"\"Most frequent words, case-insensitive, ties broken alphabetically.\"\"\"\n"
            "    counts = Counter(re.findall(r\"[a-z0-9']+\", text.lower()))\n"
            "    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]\n"
        ),
        "diagnosis": "Words were not normalised (case/punctuation) and ties were not broken alphabetically.",
    },
}


def pick_code_task(goal: str) -> dict[str, str]:
    g = goal.lower()
    if "roman" in g:
        return CODE_CATALOG["roman"]
    if "palindrom" in g:
        return CODE_CATALOG["palindrome"]
    if "fibonacci" in g:
        return CODE_CATALOG["fibonacci"]
    return CODE_CATALOG["word_frequency"]


class MuseLocalProvider(LLMProvider):
    profile = ProviderProfile("muse-local", "Muse Local (offline)", "muse-local-sim-1", 7.0, 9.5, 0.0, 0.0,
                              ("deterministic", "offline", "zero cost"))

    def __init__(self, simulate_latency: bool = True):
        self.simulate_latency = simulate_latency

    async def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        ctx = request.context
        goal = str(ctx.get("goal", request.prompt))
        handler = getattr(self, "_" + request.task.replace(".", "_"), None)
        result: Any = handler(goal, ctx) if handler else self._general(goal, ctx)
        text = json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
        if self.simulate_latency:
            await asyncio.sleep(0.25 + (_seed(request.task, goal) % 450) / 1000)
        tin, tout = estimate_tokens(request.system + request.prompt), estimate_tokens(text)
        return LLMResponse(text=text, provider=self.name, model=self.profile.model, input_tokens=tin,
                           output_tokens=tout, latency_ms=(time.perf_counter() - started) * 1000, cost_usd=0.0)

    # ---- security ------------------------------------------------------------------------
    def _sentinel_classify(self, goal: str, ctx: dict[str, Any]) -> dict[str, Any]:
        flagged = ctx.get("rule_findings") or []
        risky = any(f.get("severity") == "high" for f in flagged)
        return {"verdict": "block" if risky else "allow", "risk": 0.92 if risky else 0.04,
                "reasons": [f["message"] for f in flagged] or ["No injection, exfiltration or abuse patterns detected."]}

    # ---- planning ------------------------------------------------------------------------
    def _planner_plan(self, goal: str, ctx: dict[str, Any]) -> dict[str, Any]:
        agents = ctx.get("suggested_agents") or ["research"]
        t = topic_of(goal)
        objectives = {
            "research": f"Gather and verify current, cited facts about {t}.",
            "creative": f"Produce a distinctive narrative, positioning and launch copy for {t}.",
            "code": f"Implement and test production-quality code for: {goal}",
            "action": f"Execute real-world actions (email, calendar, shopping, payments, travel, reminders) for {t}, "
                      "within spend limits and gated by human approval.",
        }
        feedback = ctx.get("feedback")
        tasks = [{"agent": a, "objective": objectives[a] + (f" Address reviewer feedback: {feedback}" if feedback else "")}
                 for a in agents]
        return {"rationale": f"Decomposed the goal into {len(tasks)} specialist task(s) that can run in parallel: "
                             + ", ".join(agents) + ".", "tasks": tasks}

    # ---- research ------------------------------------------------------------------------
    def _research_queries(self, goal: str, ctx: dict[str, Any]) -> dict[str, Any]:
        g = goal.lower()
        kws = [k for k in keywords(goal, 8) if k not in GENERIC and not k[0].isdigit()] or keywords(goal, 3)
        suffix = next((sfx for word, sfx in INTENT_SUFFIX if word in g), "overview")
        subj = subject_of(goal)
        if suffix in ("market", "marketing", "industry"):
            # The product name is usually invented, so research the category it competes in instead.
            brand = brand_of(goal).lower()
            kinds = "app|platform|tool|software|product|service|device|bottle"
            product = re.search(rf"\b(?:an?|the)\s+([\w\- ]{{2,40}}?\b(?:{kinds}))\b", goal, re.I)
            subj = product.group(1) if product else subj
            words = [w for w in subj.split() if w.lower() not in GENERIC | {brand, "go-to-market", "a", "an"}]
            category = " ".join(words) or subj
            core = re.sub(r"\b(ai|app|apps|tool|platform)\b", "", category, flags=re.I).strip() or category
            core = re.sub(r"\bnotes\b", "note-taking", core)
            techy = re.search(r"\b(app|software|ai|platform|tool|saas)\b", category, re.I)
            queries = [f"{core} software" if techy else core, category, "product launch marketing"]
        elif suffix == "shopping":
            queries = [_product(goal), "online shopping"]
        elif suffix == "tourism":
            places = [p for p in proper_nouns(goal) if p.lower() not in CALENDAR_WORDS][:2] or kws[:1]
            queries = [f"{p} tourism" for p in places] + [places[0], "travel itinerary"]
        else:
            queries = [f"{p} {suffix}" for p in proper_nouns(goal)[:2]]
            if subj and subj.lower() not in " ".join(queries).lower():
                queries.append(f"{subj} {suffix}")
            queries.append(" ".join(kws[:2]))
        if ctx.get("attempt", 0):
            queries = [kws[0], " ".join(kws[1:3])]
        return {"queries": list(dict.fromkeys(q.strip() for q in queries if q.strip()))[:3]}

    def _research_synthesize(self, goal: str, ctx: dict[str, Any]) -> str:
        sources = ctx.get("sources", [])
        t = topic_of(goal)
        if not sources:
            return f"## Findings on {t}\n\nNo external sources were reachable; findings are limited."
        lines = [f"## Key findings: {t}", ""]
        for i, s in enumerate(sources[:5], 1):
            snippet = re.sub(r"\s+", " ", s.get("snippet", "")).strip()
            first = re.split(r"(?<=[.!?])\s", snippet)[0][:260] if snippet else s.get("title", "")
            lines.append(f"- **{s.get('title', 'Source')}**: {first} [{i}]")
        lines += ["", "### What this means",
                  f"- The landscape around **{t}** is active and well documented, with {len(sources)} independent "
                  "sources corroborating the core facts above.",
                  "- The strongest opportunities sit where these sources overlap: proven demand, clear user pain, "
                  "and room for a differentiated, trustworthy experience.",
                  "- Risks to monitor: incumbents moving fast, regulatory/privacy expectations, and the cost of "
                  "earning user trust."]
        return "\n".join(lines)

    def _research_verify(self, goal: str, ctx: dict[str, Any]) -> dict[str, Any]:
        n = len(ctx.get("sources", []))
        cited = len(set(re.findall(r"\[(\d+)\]", ctx.get("draft", ""))))
        coverage = round(min(1.0, cited / max(1, min(n, 5))), 2)
        ok = n >= 2 and coverage >= 0.6
        return {"supported": ok, "coverage": coverage,
                "issues": [] if ok else ["Too few independent sources; broaden the search."]}

    # ---- creative ------------------------------------------------------------------------
    def _creative_ideate(self, goal: str, ctx: dict[str, Any]) -> dict[str, Any]:
        t = subject_of(goal)
        name = brand_of(goal)
        return {"concepts": [
            {"name": f"{name}, Everywhere", "hook": f"Your {t}, woven into every moment of your day.",
             "angle": "Ubiquity and calm: the product disappears into habits."},
            {"name": f"The {name} Effect", "hook": "Small sparks of insight, compounding into big outcomes.",
             "angle": "Outcome-led storytelling with measurable before/after proof and pilot metrics."},
            {"name": f"Made with {name}", "hook": f"A community building the future of {t} together.",
             "angle": "Creator-first movement with ambassadors and open showcases."},
        ]}

    def _creative_draft(self, goal: str, ctx: dict[str, Any]) -> str:
        c = ctx.get("concept", {})
        t = subject_of(goal)
        return (f"## Campaign concept: {c.get('name', 'Muse')}\n\n**Tagline:** {c.get('hook', t)}\n\n"
                f"**Positioning:** For people who want {t} without the friction, {c.get('name', 'Muse')} is the "
                "companion that turns intent into results.\n\n### Launch narrative\nWe open with a single, vivid "
                "moment of transformation, then show the system behind it.\n\n### Channel plan\n"
                "- Launch film (60s) and three cut-downs\n- Founder thread and live demo\n- Creator early-access program")

    def _creative_critique(self, goal: str, ctx: dict[str, Any]) -> dict[str, Any]:
        rnd = int(ctx.get("round", 0))
        if rnd == 0:
            return {"score": 7.6, "feedback": [
                "Add a concrete proof point or metric to the positioning.",
                "Name the target persona explicitly.",
                "Give a phased launch timeline with success KPIs."]}
        return {"score": 9.2, "feedback": ["Strong, specific and launch-ready. Minor polish only."]}

    def _creative_revise(self, goal: str, ctx: dict[str, Any]) -> str:
        c = ctx.get("concept", {})
        name = c.get("name", "Muse")
        t = subject_of(goal)
        return (f"## Campaign concept: {name}\n\n**Tagline:** {c.get('hook', t)}\n\n"
                f"**Persona:** *Ambitious operators*, time-poor professionals who want {t} to just work.\n\n"
                f"**Positioning:** {name} turns intent into outcomes. Early pilots target **40% less busywork** "
                "in the first 30 days.\n\n### Phased launch\n| Phase | Weeks | Moves | KPI |\n|---|---|---|---|\n"
                "| Tease | 1-2 | Waitlist, founder story, creator seeding | 10k waitlist |\n"
                "| Launch | 3 | Launch film, live demo, press embargo lift | 25% waitlist activation |\n"
                "| Amplify | 4-8 | Community showcases, referrals, case studies | 1.4 viral coefficient |\n\n"
                "### Hero copy\n> *Stop managing the work about the work. Let your Muse handle it, and "
                "get back to what only you can do.*\n\n### Social launch posts\n"
                f"1. \"We built {name} because your best ideas deserve more than a to-do list.\"\n"
                "2. \"30 days. 40% less busywork. Here's how our pilot teams did it.\"\n"
                "3. \"Show us what you made with your Muse. Best builds get featured.\"")

    # ---- code ----------------------------------------------------------------------------
    def _code_spec(self, goal: str, ctx: dict[str, Any]) -> dict[str, Any]:
        task = pick_code_task(goal)
        return {k: task[k] for k in ("function_name", "signature", "description", "tests")}

    def _code_implement(self, goal: str, ctx: dict[str, Any]) -> str:
        return "```python\n" + pick_code_task(goal)["buggy"] + "```"

    def _code_debug(self, goal: str, ctx: dict[str, Any]) -> str:
        task = pick_code_task(goal)
        return f"Diagnosis: {task['diagnosis']}\n\n```python\n{task['fixed']}```"

    # ---- actions -------------------------------------------------------------------------
    def _action_plan(self, goal: str, ctx: dict[str, Any]) -> dict[str, Any]:
        return plan_actions_for(goal)

    # ---- judge & compose -----------------------------------------------------------------
    def _critic_judge(self, goal: str, ctx: dict[str, Any]) -> dict[str, Any]:
        """Offline judge = deterministic rubric over *verified* run signals (a live LLM judge replaces it
        when a provider key is configured). Each dimension only reaches 10 when its evidence checks out."""
        s = ctx.get("signals", {})
        sources = s.get("sources")
        grounding = 10.0 if sources is None else (10.0 if s.get("citations_verified") and sources >= 3 else
                                                    8.5 if sources else 6.0)
        correctness = 10.0 if s.get("tests_passed", True) else 5.0
        safety = 10.0 if s.get("approval_honored", True) else 4.0
        creative_gain = s.get("creative_final_score", 9.0) - s.get("creative_first_score", 9.0)
        clarity = 10.0 if creative_gain >= 0 and s.get("creative_final_score", 9.5) >= 9.0 else 8.8
        relevance = 10.0 if all(v is not None and v is not False for v in s.values()) else 9.0
        scores = {"relevance": relevance, "grounding": grounding, "correctness": correctness,
                  "safety": safety, "clarity": clarity}
        overall = round(sum(scores.values()) / len(scores), 2)
        weak = [k for k, v in scores.items() if v < 9.5]
        feedback = ("All rubric dimensions verified: grounded, tested, safe and consistent across specialists."
                    if not weak else f"Improve: {', '.join(weak)}.")
        return {"scores": scores, "overall": overall, "feedback": feedback,
                "pass": overall >= float(ctx.get("threshold", 8.0))}

    def _composer_final(self, goal: str, ctx: dict[str, Any]) -> str:
        results = ctx.get("results", [])
        done = [a for r in results for a in (r.get("artifacts") or {}).get("actions", [])]
        ran = sum(a.get("status") == "executed" for a in done)
        held = len(done) - ran
        summary = "Here is what I did for you."
        if done:
            extra = f" and held back {held} that were declined or not allowed" if held else ""
            summary = f"I completed {ran} step(s){extra}."
        parts = [f"# {_headline(goal)}", "", f"> **You asked:** {goal}", "", summary, ""]
        for r in sorted(results, key=lambda r: r.get("agent") != "action"):  # what was done comes first
            parts += [f"## {r.get('title', r.get('agent', 'Result'))}", r.get("content", ""), ""]
        return "\n".join(parts)

    def _memory_summarize(self, goal: str, ctx: dict[str, Any]) -> str:
        agents = ", ".join(r.get("agent", "") for r in ctx.get("results", []))
        return f"{_headline(goal)} (done with {agents or 'no'} agent(s))."

    def _general(self, goal: str, ctx: dict[str, Any]) -> str:
        return f"Acknowledged: {topic_of(goal, 8)}."


# --------------------------------------------------------------------------------------------
# Intent-driven action planner used offline (live models plan with the same connector catalog).
# --------------------------------------------------------------------------------------------
CITIES = ("tokyo", "lisbon", "paris", "bangalore", "goa", "london", "new york", "singapore", "kyoto", "bali")
ACTION_INTENTS: dict[str, tuple[str, ...]] = {
    "travel": ("trip", "flight", "hotel", "travel", "visit", "logistics", "offsite"),
    "ride": ("ride", "uber", "cab", "taxi", "airport transfer"),
    "shopping": ("buy", "order", "gift", "shop", "cart", "purchase", "checkout"),
    "dining": ("dinner", "restaurant", "lunch", "table for", "reservation"),
    "calendar": ("calendar", "meeting", "schedule", "invite", "block", "sync"),
    "email": ("email", "mail", "inbox", "invite"),
    "message": ("slack", "message", "text ", "sms", "notify"),
    "payment": ("pay", "split", "deposit", "refund", "charge", "invoice", "venmo"),
    "reminder": ("remind", "reminder", "todo", "to-do", "follow up"),
}


def _money(goal: str, default: float) -> float:
    m = re.search(r"\$\s?(\d[\d,]*(?:\.\d+)?)", goal)
    return float(m.group(1).replace(",", "")) if m else default


def _product(goal: str) -> str:
    m = re.search(r"\b(?:buy|order|purchase|get)\s+(?:an?\s+|some\s+|the\s+)?([a-z][a-z\- ]{2,40}?)"
                  r"(?=\s+(?:under|for|as|below|within|to|and|,)|[,.]|$)", goal, re.I)
    return m.group(1).strip() if m else "gift box"


def _people(goal: str) -> list[str]:
    names = [n for n in proper_nouns(goal) if n.lower() not in CALENDAR_WORDS | set(CITIES)
             and n.lower() not in ("slack", "lumen", "muse", "uber")]
    if not names and re.search(r"\bmy (partner|wife|husband|manager|team)\b", goal, re.I):
        return [re.search(r"\bmy (partner|wife|husband|manager|team)\b", goal, re.I).group(1).lower() + "@example.com"]
    return [f"{n.lower()}@example.com" for n in names[:3]] or ["team@example.com"]


def _headline(goal: str) -> str:
    first = re.split(r"[,:;.]", goal)[0].strip()
    return first if len(first) <= 60 else first[:57] + "..."


def plan_actions_for(goal: str) -> dict[str, Any]:
    g = goal.lower()
    intents = [k for k, words in ACTION_INTENTS.items() if any(w in g for w in words)] or ["travel", "calendar", "email"]
    dest = next((c.title() for c in CITIES if c in g), "Lisbon")
    people = _people(goal)
    actions: list[dict[str, Any]] = []

    def add(tool: str, title: str, args: dict[str, Any], depends: list[str] | None = None) -> str:
        aid = f"a{len(actions) + 1}"
        actions.append({"id": aid, "tool": tool, "title": title, "args": args, "depends_on": depends or []})
        return aid

    if "travel" in intents:
        add("search_flights", f"Search flights to {dest}", {"destination": dest, "passengers": 2, "cabin": "economy"})
        add("hold_hotel", f"Hold a boutique hotel in {dest} (3 nights)",
            {"city": dest, "nights": 3, "budget_per_night_usd": 180})
    if "ride" in intents:
        add("book_ride", "Book an airport ride", {"pickup": "Home", "dropoff": f"{dest} airport", "amount_usd": 42})
    if "shopping" in intents:
        item, budget = _product(goal), _money(goal, 120)
        price = round(budget * 0.92, 2)
        add("wallet_balance", "Check wallet balance before spending", {})
        s = add("search_products", f"Find the best {item} under ${budget:,.0f}", {"query": item, "max_price_usd": budget})
        c = add("add_to_cart", f"Add top-rated {item} to cart",
                {"sku": "SKU-" + hashlib.sha256(item.encode()).hexdigest()[:6].upper(), "qty": 1, "price_usd": price}, [s])
        add("checkout_cart", f"Checkout and pay ${price:,.2f}", {"amount_usd": price, "ship_to": "Home"}, [c])
    if "dining" in intents:
        add("reserve_restaurant", "Reserve dinner for the group",
            {"restaurant": "Muse Bistro", "party_size": len(people) + 1, "time": "Friday 19:30"})
    if "calendar" in intents or "dining" in intents or "travel" in intents:
        minutes = re.search(r"(\d+)[- ]?(?:minute|min)\b", g)
        hours = re.search(r"(\d+(?:\.\d+)?)[- ]?hour", g)
        duration = int(minutes.group(1)) if minutes else int(float(hours.group(1)) * 60) if hours else 90
        within = 21 if "travel" in intents else 7 if "next week" in g else 1 if "tomorrow" in g else 3
        window: dict[str, Any] = {"attendees": people, "duration_min": duration, "within_days": within}
        if "next week" in g:  # nothing before next Monday
            window["after_days"] = 7 - date.today().weekday()
        f = add("find_meeting_slots", "Find a time that works for everyone", window)
        meeting = re.search(r"\b((?:[a-z0-9]+ ){0,2}(?:meeting|sync|review|call|workshop|standup))\b", g)
        title = (f"{dest} trip" if "travel" in intents else "Dinner" if "dining" in intents
                 else re.sub(r"^(a|an|the|\d+|minute|hour)\s+", "", meeting.group(1)).strip().capitalize() if meeting
                 else "Planning sync")
        title = re.sub(r"^(minute|hour)\s+", "", title, flags=re.I).capitalize()
        add("create_calendar_event", f"Create calendar event: {title}",
            {"title": title, "attendees": people, "days": 3 if "travel" in intents else 0}, [f])
    if "email" in intents:
        add("search_inbox", "Check inbox for related threads", {"query": dest if "travel" in intents else "plans"})
        d = add("draft_email", "Draft a summary email", {"to": people, "subject": f"Plan: {_headline(goal)}"})
        add("send_email", "Send the summary email", {"to": people, "subject": f"Plan: {_headline(goal)}"}, [d])
    if "message" in intents:
        add("send_message", "Post an update to Slack", {"channel": "#team", "text": topic_of(goal, 8)})
    if "payment" in intents:
        clauses = list(re.finditer(
            r"\b(pay|charge|refund|split)\b([^$,;.]{0,60}?)\$\s?(\d[\d,]*(?:\.\d+)?)(?=([^,;.$]{0,40}))", goal, re.I))

        def subject(text: str) -> str:
            text = re.split(r"\band\b", text)[0]
            return re.sub(r"^\s*((the|a|an|for|to|of)\s+)+|\s+(for|to|of|the)\s*$", "", text.strip() + " ", flags=re.I).strip()

        for m in clauses:
            verb, amount = m.group(1).lower(), float(m.group(3).replace(",", ""))
            what = subject(m.group(2)) or subject(m.group(4)) or "merchant"
            if verb == "refund":
                add("refund_payment", f"Refund ${amount:,.2f} to {what}", {"to": what, "amount_usd": amount})
            elif verb == "split":
                add("split_payment", f"Split the ${amount:,.0f} {what} with friends", {"amount_usd": amount, "people": people})
            else:
                add("charge_card", f"Pay ${amount:,.2f} for {what}", {"amount_usd": amount, "payee": what})
        if not clauses:
            total = _money(goal, 240) if "shopping" not in intents else 240.0
            if "split" in g or "deposit" in g:
                add("split_payment", f"Split ${total:,.0f} with friends", {"amount_usd": total, "people": people})
            elif "refund" in g:
                add("refund_payment", "Refund the previous order", {"order_id": "ord_prev01", "amount_usd": total})
            else:
                add("charge_card", f"Pay ${total:,.2f}", {"amount_usd": total, "payee": "merchant"})
    if "reminder" in intents:
        todo = re.search(r"remind (?:me|us) to ([^,;.]+)", goal, re.I)
        text = todo.group(1).strip().capitalize() if todo else f"Follow up: {topic_of(goal, 6)}"
        add("create_reminder", f"Remind me: {text}", {"text": text, "due": "tomorrow 09:00"})
    domains = sorted({ACTION_DOMAIN.get(a["tool"], "other") for a in actions})
    return {"summary": f"{len(actions)} actions across {', '.join(domains)}; read-only steps run automatically, "
                       "side effects wait for your approval.", "actions": actions}


ACTION_DOMAIN = {"search_flights": "travel", "hold_hotel": "travel", "book_ride": "travel", "find_meeting_slots": "calendar",
                 "create_calendar_event": "calendar", "search_inbox": "email", "draft_email": "email",
                 "send_email": "email", "send_message": "messaging", "create_reminder": "reminders",
                 "reserve_restaurant": "dining", "search_products": "shopping", "add_to_cart": "shopping",
                 "checkout_cart": "shopping", "wallet_balance": "payments", "charge_card": "payments",
                 "split_payment": "payments", "refund_payment": "payments"}
