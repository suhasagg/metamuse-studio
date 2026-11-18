"""Amazon Product Advertising API 5.0 (SearchItems) with AWS SigV4 signing, plus the official
add-to-cart form URL. Amazon offers no public checkout API, so purchase is handed back to the user."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

import httpx

from . import IntegrationError, env

HOST = "webservices.amazon.com"
REGION = "us-east-1"
SERVICE = "ProductAdvertisingAPI"
TARGET = "com.amazon.paapi5.v1.ProductAdvertisingAPIv1.SearchItems"


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def signed_headers(payload: str, now: datetime | None = None) -> dict[str, str]:
    now = now or datetime.now(UTC)
    amz_date, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    headers = {"content-encoding": "amz-1.0", "content-type": "application/json; charset=utf-8", "host": HOST,
               "x-amz-date": amz_date, "x-amz-target": TARGET}
    names = ";".join(sorted(headers))
    canonical = "\n".join(["POST", "/paapi5/searchitems", "", *(f"{k}:{headers[k]}" for k in sorted(headers)), "",
                           names, hashlib.sha256(payload.encode()).hexdigest()])
    scope = f"{day}/{REGION}/{SERVICE}/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    k = _sign(_sign(_sign(_sign(("AWS4" + env("AMAZON_PAAPI_SECRET_KEY")).encode(), day), REGION), SERVICE), "aws4_request")
    signature = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    headers["Authorization"] = (f"AWS4-HMAC-SHA256 Credential={env('AMAZON_PAAPI_ACCESS_KEY')}/{scope}, "
                                f"SignedHeaders={names}, Signature={signature}")
    return headers


async def search_products(args: dict[str, Any]) -> dict[str, Any]:
    q, budget = args.get("query", "gift"), float(args.get("max_price_usd", 0) or 0)
    body: dict[str, Any] = {"Keywords": q, "PartnerTag": env("AMAZON_PARTNER_TAG"), "PartnerType": "Associates",
                            "Marketplace": "www.amazon.com", "SearchIndex": "All", "ItemCount": 5,
                            "Resources": ["ItemInfo.Title", "Offers.Listings.Price", "Images.Primary.Small"]}
    if budget:
        body["MaxPrice"] = int(budget * 100)
    payload = json.dumps(body)
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(f"https://{HOST}/paapi5/searchitems", content=payload, headers=signed_headers(payload))
    if r.status_code != 200:
        raise IntegrationError(f"amazon paapi {r.status_code}: {r.text[:200]}")
    items = []
    for it in r.json().get("SearchResult", {}).get("Items", []):
        listing = (it.get("Offers", {}).get("Listings") or [{}])[0]
        items.append({"sku": it.get("ASIN"), "title": it.get("ItemInfo", {}).get("Title", {}).get("DisplayValue", ""),
                      "price_usd": listing.get("Price", {}).get("Amount"), "rating": None, "url": it.get("DetailPageURL")})
    return {"query": q, "results": items}


def cart_url(asin: str, qty: int = 1) -> str:
    return "https://www.amazon.com/gp/aws/cart/add.html?" + urlencode(
        {"ASIN.1": asin, "Quantity.1": qty, "AssociateTag": env("AMAZON_PARTNER_TAG")})


async def add_to_cart(args: dict[str, Any]) -> dict[str, Any]:
    qty = int(args.get("qty", 1))
    return {"sku": args.get("sku"), "qty": qty, "cart_url": cart_url(str(args.get("sku")), qty),
            "subtotal_usd": args.get("price_usd"), "note": "Amazon checkout is completed by you via cart_url"}
