"""Real third-party adapters (Google Search, Gmail, Google Calendar, Google Tasks, Amazon PA-API, Stripe,
Slack, Amadeus). Each adapter turns on only when its credentials are in the environment; otherwise
the matching connector stays in simulated mode. See README "Real integrations"."""

from __future__ import annotations

import os

INTEGRATIONS: dict[str, dict[str, object]] = {
    "google_search": {"label": "Google Programmable Search", "env": ("GOOGLE_API_KEY", "GOOGLE_CSE_ID")},
    "google_workspace": {"label": "Gmail · Google Calendar · Google Tasks",
                         "env": ("GOOGLE_ACCESS_TOKEN",),
                         "alt_env": ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN")},
    "amazon": {"label": "Amazon Product Advertising API 5.0",
               "env": ("AMAZON_PAAPI_ACCESS_KEY", "AMAZON_PAAPI_SECRET_KEY", "AMAZON_PARTNER_TAG")},
    "stripe": {"label": "Stripe Payments", "env": ("STRIPE_API_KEY",)},
    "slack": {"label": "Slack", "env": ("SLACK_BOT_TOKEN",)},
    "amadeus": {"label": "Amadeus Flight Offers", "env": ("AMADEUS_CLIENT_ID", "AMADEUS_CLIENT_SECRET")},
    "tavily": {"label": "Tavily Search", "env": ("TAVILY_API_KEY",)},
}


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def configured(integration: str) -> bool:
    spec = INTEGRATIONS[integration]
    if all(env(k) for k in spec["env"]):  # type: ignore[union-attr]
        return True
    alt = spec.get("alt_env")
    return bool(alt) and all(env(k) for k in alt)  # type: ignore[union-attr]


def status() -> list[dict[str, object]]:
    return [{"id": k, "label": v["label"], "live": configured(k), "env": list(v["env"])}  # type: ignore[arg-type]
            for k, v in INTEGRATIONS.items()]


class IntegrationError(RuntimeError):
    """Raised when a live adapter call fails; side-effecting actions surface it instead of faking success."""
