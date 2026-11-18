"""Web research tool: Tavily (if keyed) -> Google Programmable Search (if keyed) -> Wikipedia (live,
keyless) -> bundled offline corpus."""

from __future__ import annotations

import re
from typing import Any

import httpx

from .. import integrations, security
from ..config import get_settings
from ..integrations import google as integrations_google

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "MetaMuseStudio/1.0 (https://github.com/suhasagg/meta-muse-langgraph-studio)"

OFFLINE_CORPUS: list[dict[str, str]] = [
    {"title": "Intelligent agent", "url": "https://en.wikipedia.org/wiki/Intelligent_agent",
     "snippet": "An intelligent agent perceives its environment, takes actions autonomously to achieve goals, "
                "and may improve its performance through learning."},
    {"title": "Large language model", "url": "https://en.wikipedia.org/wiki/Large_language_model",
     "snippet": "A large language model is a model trained with self-supervised learning on vast text, "
                "designed for natural language processing tasks such as generation."},
    {"title": "Product launch", "url": "https://en.wikipedia.org/wiki/Product_launch",
     "snippet": "A product launch is the coordinated effort to introduce a new product to the market, "
                "spanning positioning, messaging, channels and timing."},
    {"title": "Travel itinerary", "url": "https://en.wikipedia.org/wiki/Itinerary",
     "snippet": "An itinerary is a schedule of events relating to planned travel, generally including "
                "destinations, transport and accommodation."},
    {"title": "Software testing", "url": "https://en.wikipedia.org/wiki/Software_testing",
     "snippet": "Software testing is the act of checking whether software satisfies expectations, "
                "using techniques such as unit tests and assertions."},
]


def _clean(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html)).strip()


async def _wikipedia(query: str, k: int, transport: httpx.AsyncBaseTransport | None = None) -> list[dict[str, Any]]:
    params = {"action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": k, "prop": "extracts|info",
              "exintro": 1, "explaintext": 1, "exsentences": 3, "inprop": "url", "format": "json"}
    async with httpx.AsyncClient(timeout=8, headers={"User-Agent": USER_AGENT}, transport=transport) as client:
        resp = await client.get(WIKI_API, params=params)
        resp.raise_for_status()
    pages = sorted((resp.json().get("query", {}).get("pages", {}) or {}).values(), key=lambda p: p.get("index", 0))
    return [{"title": p.get("title", ""), "url": p.get("fullurl", ""), "snippet": _clean(p.get("extract", ""))[:600],
             "source": "wikipedia"} for p in pages
            if p.get("extract") and "disambiguation" not in p.get("title", "").lower()
            and "may refer to" not in p.get("extract", "")]


async def _tavily(query: str, k: int, key: str) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=12) as client:
        resp = await client.post("https://api.tavily.com/search",
                                 json={"api_key": key, "query": query, "max_results": k})
        resp.raise_for_status()
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")[:600],
             "source": "tavily"} for r in resp.json().get("results", [])]


def _offline(query: str, k: int) -> list[dict[str, Any]]:
    words = set(query.lower().split())
    ranked = sorted(OFFLINE_CORPUS, key=lambda d: -len(words & set((d["title"] + " " + d["snippet"]).lower().split())))
    return [dict(d, source="offline") for d in ranked[:k]]


async def _search(query: str, k: int = 4) -> list[dict[str, Any]]:
    settings = get_settings()
    backend = settings.search_backend
    if backend in ("auto", "tavily") and settings.tavily_api_key:
        try:
            return await _tavily(query, k, settings.tavily_api_key)
        except httpx.HTTPError:
            pass
    if backend in ("auto", "google") and integrations.configured("google_search"):
        try:
            results = await integrations_google.search(query, k)
            if results:
                return results
        except (httpx.HTTPError, integrations.IntegrationError):
            pass
    if backend in ("auto", "wikipedia", "tavily", "google"):
        try:
            results = await _wikipedia(query, k)
            if results:
                return results
        except (httpx.HTTPError, ValueError):
            pass
    return _offline(query, k)


async def web_search(query: str, k: int = 4, agent: str = "research") -> list[dict[str, Any]]:
    """Search, then pass every result through the untrusted-content firewall (indirect prompt injection)."""
    security.authorize(agent, "web_search")
    hits = []
    for hit in await _search(query, k):
        clean, findings = security.sanitize_tree({"title": hit.get("title", ""), "snippet": hit.get("snippet", "")})
        hits.append({**hit, **clean, **({"security": sorted(set(findings))} if findings else {})})
    return hits
