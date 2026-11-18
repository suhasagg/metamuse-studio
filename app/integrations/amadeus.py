"""Amadeus Self-Service Flight Offers Search (test environment by default)."""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

import httpx

from . import IntegrationError, env

_tok: dict[str, Any] = {}


def _base() -> str:
    return env("AMADEUS_BASE_URL", "https://test.api.amadeus.com")


async def _token(c: httpx.AsyncClient) -> str:
    if _tok.get("v") and _tok["exp"] > time.time() + 30:
        return _tok["v"]
    r = await c.post(f"{_base()}/v1/security/oauth2/token", data={
        "grant_type": "client_credentials", "client_id": env("AMADEUS_CLIENT_ID"),
        "client_secret": env("AMADEUS_CLIENT_SECRET")})
    if r.status_code != 200:
        raise IntegrationError(f"amadeus auth {r.status_code}: {r.text[:200]}")
    _tok.update(v=r.json()["access_token"], exp=time.time() + int(r.json().get("expires_in", 1700)))
    return _tok["v"]


async def search_flights(args: dict[str, Any]) -> dict[str, Any]:
    dest = args.get("destination", "Lisbon")
    async with httpx.AsyncClient(timeout=20) as c:
        h = {"Authorization": f"Bearer {await _token(c)}"}
        loc = await c.get(f"{_base()}/v1/reference-data/locations", headers=h,
                          params={"subType": "CITY,AIRPORT", "keyword": dest[:20]})
        code = next((x.get("iataCode") for x in loc.json().get("data", []) if x.get("iataCode")), None)
        if not code:
            raise IntegrationError(f"amadeus: no IATA code for {dest}")
        r = await c.get(f"{_base()}/v2/shopping/flight-offers", headers=h, params={
            "originLocationCode": env("MUSE_HOME_AIRPORT", "SFO"), "destinationLocationCode": code,
            "departureDate": str(date.today() + timedelta(days=21)), "adults": int(args.get("passengers", 1)),
            "max": 3, "currencyCode": "USD"})
    if r.status_code != 200:
        raise IntegrationError(f"amadeus offers {r.status_code}: {r.text[:200]}")
    opts = []
    for o in r.json().get("data", []):
        seg = o["itineraries"][0]["segments"]
        opts.append({"carrier": seg[0]["carrierCode"], "depart": seg[0]["departure"]["at"][11:16],
                     "stops": len(seg) - 1, "price_usd": float(o["price"]["grandTotal"])})
    return {"destination": dest, "options": opts}
