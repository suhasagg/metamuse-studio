"""Slack Web API chat.postMessage."""

from __future__ import annotations

from typing import Any

import httpx

from . import IntegrationError, env


async def post_message(args: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post("https://slack.com/api/chat.postMessage",
                         headers={"Authorization": f"Bearer {env('SLACK_BOT_TOKEN')}"},
                         json={"channel": args.get("channel", env("SLACK_DEFAULT_CHANNEL", "#general")),
                               "text": args.get("text", "Update from Meta Muse")})
    data = r.json()
    if not data.get("ok"):
        raise IntegrationError(f"slack: {data.get('error', r.status_code)}")
    return {"message_ts": data.get("ts"), "channel": data.get("channel"), "status": "posted"}
