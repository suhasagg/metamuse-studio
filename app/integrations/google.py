"""Google APIs: Programmable Search (Custom Search JSON API), Gmail, Calendar and Tasks (OAuth 2.0)."""

from __future__ import annotations

import base64
import re
import time
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from typing import Any

import httpx

from . import IntegrationError, env

_token: dict[str, Any] = {}
TIMEOUT = 15


async def search(query: str, k: int = 5) -> list[dict[str, Any]]:
    params = {"key": env("GOOGLE_API_KEY"), "cx": env("GOOGLE_CSE_ID"), "q": query, "num": min(k, 10)}
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.get("https://www.googleapis.com/customsearch/v1", params=params)
    if r.status_code != 200:
        raise IntegrationError(f"google search {r.status_code}: {r.text[:200]}")
    return [{"title": i.get("title", ""), "url": i.get("link", ""), "snippet": i.get("snippet", ""),
             "source": "google"} for i in r.json().get("items", [])[:k]]


async def access_token() -> str:
    if env("GOOGLE_ACCESS_TOKEN"):
        return env("GOOGLE_ACCESS_TOKEN")
    if _token.get("value") and _token.get("exp", 0) > time.time() + 60:
        return _token["value"]
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.post("https://oauth2.googleapis.com/token", data={
            "client_id": env("GOOGLE_CLIENT_ID"), "client_secret": env("GOOGLE_CLIENT_SECRET"),
            "refresh_token": env("GOOGLE_REFRESH_TOKEN"), "grant_type": "refresh_token"})
    if r.status_code != 200:
        raise IntegrationError(f"google oauth refresh failed {r.status_code}: {r.text[:200]}")
    data = r.json()
    _token.update(value=data["access_token"], exp=time.time() + int(data.get("expires_in", 3600)))
    return _token["value"]


async def _call(method: str, url: str, **kw: Any) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {await access_token()}"}
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.request(method, url, headers=headers, **kw)
    if r.status_code >= 300:
        raise IntegrationError(f"{method} {url.split('?')[0]} -> {r.status_code}: {r.text[:200]}")
    return r.json() if r.content else {}


def _recipients(to: Any) -> str:
    return ", ".join(to) if isinstance(to, list) else str(to or "")


def _raw(args: dict[str, Any]) -> str:
    msg = EmailMessage()
    msg["To"] = _recipients(args.get("to"))
    msg["Subject"] = args.get("subject", "Update from Meta Muse")
    msg.set_content(args.get("body") or f"{args.get('subject', '')}\n\nSent by Meta Muse after your approval.")
    return base64.urlsafe_b64encode(msg.as_bytes()).decode()


GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me"


async def gmail_search(args: dict[str, Any]) -> dict[str, Any]:
    q = args.get("query", "")
    data = await _call("GET", f"{GMAIL}/messages", params={"q": q, "maxResults": 5})
    threads = []
    for m in data.get("messages", [])[:5]:
        meta = await _call("GET", f"{GMAIL}/messages/{m['id']}",
                           params={"format": "metadata", "metadataHeaders": ["From", "Subject"]})
        hdrs = {h["name"]: h["value"] for h in meta.get("payload", {}).get("headers", [])}
        threads.append({"from": hdrs.get("From", ""), "subject": hdrs.get("Subject", ""),
                        "unread": "UNREAD" in meta.get("labelIds", [])})
    return {"query": q, "threads": threads}


async def gmail_draft(args: dict[str, Any]) -> dict[str, Any]:
    data = await _call("POST", f"{GMAIL}/drafts", json={"message": {"raw": _raw(args)}})
    return {"draft_id": data.get("id"), "to": args.get("to"), "subject": args.get("subject"), "status": "draft saved"}


async def gmail_send(args: dict[str, Any]) -> dict[str, Any]:
    data = await _call("POST", f"{GMAIL}/messages/send", json={"raw": _raw(args)})
    return {"message_id": data.get("id"), "to": args.get("to"), "subject": args.get("subject"), "status": "sent"}


CAL = "https://www.googleapis.com/calendar/v3"


async def calendar_free_slots(args: dict[str, Any]) -> dict[str, Any]:
    start = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    if args.get("after_days"):  # e.g. "next week": do not offer anything earlier than that day
        start = (start + timedelta(days=int(args["after_days"]))).replace(hour=9)
    end = start + timedelta(days=int(args.get("within_days", 3)))
    data = await _call("POST", f"{CAL}/freeBusy", json={
        "timeMin": start.isoformat(), "timeMax": end.isoformat(), "items": [{"id": "primary"}]})
    busy = [(datetime.fromisoformat(b["start"].replace("Z", "+00:00")), datetime.fromisoformat(b["end"].replace("Z", "+00:00")))
            for b in data.get("calendars", {}).get("primary", {}).get("busy", [])]
    dur = timedelta(minutes=int(args.get("duration_min", 60)))
    slots, t = [], start
    while t < end and len(slots) < 3:
        if 9 <= t.hour <= 19 and all(not (t < e and t + dur > s) for s, e in busy):
            slots.append(t.isoformat(timespec="minutes"))
            t += timedelta(hours=3)
        else:
            t += timedelta(minutes=30)
    return {"slots": slots, "attendees": args.get("attendees", []), "duration_min": int(dur.total_seconds() // 60)}


def _when(value: Any, default_days: int) -> datetime:
    """Parse an approved date/time. Accepts ISO datetimes, `YYYY-MM-DD[ HH:MM]`, and `today|tomorrow [HH:MM]`.
    Only a missing value gets the default; an unparseable one is rejected rather than silently replaced."""
    now = datetime.now(UTC)
    if not value:
        return (now + timedelta(days=default_days)).replace(hour=9, minute=0, second=0, microsecond=0)
    text = str(value).strip()
    rel = re.fullmatch(r"(today|tomorrow)(?:\s+(\d{1,2}):(\d{2}))?", text, re.I)
    if rel:
        day = now + timedelta(days=rel.group(1).lower() == "tomorrow")
        return day.replace(hour=int(rel.group(2) or 9), minute=int(rel.group(3) or 0), second=0, microsecond=0)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IntegrationError(f"unsupported date/time '{text}'") from exc
    if len(text) <= 10:
        parsed = parsed.replace(hour=9)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


async def calendar_create(args: dict[str, Any]) -> dict[str, Any]:
    begin = _when(args.get("start"), default_days=7)
    end = begin + (timedelta(days=int(args["days"])) if args.get("days")
                   else timedelta(minutes=int(args.get("duration_min") or 60)))
    attendees = args.get("attendees") or []
    body = {"summary": args.get("title", "Meta Muse event"), "start": {"dateTime": begin.isoformat()},
            "end": {"dateTime": end.isoformat()},
            "attendees": [{"email": a} for a in (attendees if isinstance(attendees, list) else [attendees])]}
    data = await _call("POST", f"{CAL}/calendars/primary/events", params={"sendUpdates": "all"}, json=body)
    return {"event_id": data.get("id"), "title": data.get("summary"), "start": begin.isoformat(),
            "end": end.isoformat(), "link": data.get("htmlLink")}


async def tasks_create(args: dict[str, Any]) -> dict[str, Any]:
    due = _when(args.get("due"), default_days=1)
    data = await _call("POST", "https://tasks.googleapis.com/tasks/v1/lists/@default/tasks",
                       json={"title": args.get("text", "Follow up"), "due": due.isoformat().replace("+00:00", "Z")})
    return {"reminder_id": data.get("id"), "text": data.get("title"), "due": data.get("due")}
