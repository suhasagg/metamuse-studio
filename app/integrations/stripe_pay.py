"""Stripe: PaymentIntents (charges), Refunds and Payment Links (bill splitting).
Live secret keys are refused unless MUSE_ALLOW_LIVE_PAYMENTS=1, so demos only ever touch test mode."""

from __future__ import annotations

import math
from typing import Any

import httpx

from . import IntegrationError, env

API = "https://api.stripe.com/v1"


def _key() -> str:
    key = env("STRIPE_API_KEY")
    if "_live_" in key[:10] and env("MUSE_ALLOW_LIVE_PAYMENTS") != "1":
        raise IntegrationError("refusing a live Stripe key; set MUSE_ALLOW_LIVE_PAYMENTS=1 to allow real charges")
    return key


async def _post(path: str, data: dict[str, Any], idem: str | None = None) -> dict[str, Any]:
    headers = {"Idempotency-Key": idem} if idem else {}
    async with httpx.AsyncClient(timeout=15, auth=(_key(), "")) as c:
        r = await c.post(f"{API}{path}", data=data, headers=headers)
    if r.status_code >= 300:
        raise IntegrationError(f"stripe {path} {r.status_code}: {r.text[:200]}")
    return r.json()


def _cents(amount: Any) -> int:
    value = float(amount or 0)
    if not math.isfinite(value) or value < 0:
        raise IntegrationError(f"invalid amount: {amount!r}")
    return int(round(value * 100))


async def charge(args: dict[str, Any], idem: str | None = None) -> dict[str, Any]:
    pi = await _post("/payment_intents", {
        "amount": _cents(args.get("amount_usd")), "currency": "usd", "confirm": "true",
        "payment_method": args.get("payment_method", "pm_card_visa"),
        "automatic_payment_methods[enabled]": "true", "automatic_payment_methods[allow_redirects]": "never",
        "description": args.get("payee") or args.get("description") or "Meta Muse approved payment"}, idem)
    if pi["status"] != "succeeded":
        # requires_action / processing / requires_payment_method: money has not moved, so this is not a success.
        raise IntegrationError(f"payment {pi['id']} not completed (status: {pi['status']})")
    return {"payment_id": pi["id"], "amount_usd": pi["amount"] / 100, "status": pi["status"], "livemode": pi["livemode"]}


async def refund(args: dict[str, Any], idem: str | None = None) -> dict[str, Any]:
    if _cents(args.get("amount_usd")) <= 0:
        raise IntegrationError("refund needs an explicit positive amount_usd; refusing an implicit full refund")
    if not (args.get("payment_intent") or args.get("charge")):
        raise IntegrationError("refund needs the payment_intent or charge id of the original payment")
    data = {"payment_intent": args["payment_intent"]} if args.get("payment_intent") else {"charge": args.get("charge", "")}
    data["amount"] = _cents(args["amount_usd"])
    rf = await _post("/refunds", data, idem)
    return {"refund_id": rf["id"], "amount_usd": rf["amount"] / 100, "status": rf["status"]}


async def split(args: dict[str, Any], idem: str | None = None) -> dict[str, Any]:
    people = args.get("people") or []
    share = round(float(args.get("amount_usd", 0)) / (len(people) + 1), 2)
    price = await _post("/prices", {"unit_amount": _cents(share), "currency": "usd",
                                    "product_data[name]": args.get("title", "Shared deposit")}, idem and idem + ":price")
    link = await _post("/payment_links", {"line_items[0][price]": price["id"], "line_items[0][quantity]": 1},
                       idem and idem + ":link")
    return {"request_id": link["id"], "total_usd": float(args.get("amount_usd", 0)),
            "requests": [{"to": p, "amount_usd": share, "pay_url": link["url"]} for p in people]}
