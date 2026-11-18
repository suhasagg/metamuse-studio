import asyncio

from app.integrations import IntegrationError, amazon, stripe_pay
from app.providers.mock_provider import plan_actions_for
from app.subgraphs import action
from app.tools import connectors


def test_catalog_covers_domains_and_defaults_to_simulated():
    cat = connectors.catalog()
    assert len(cat) == 18
    assert {c["domain"] for c in cat} >= {"email", "calendar", "shopping", "payments", "travel", "messaging"}
    assert all(c["mode"] == "simulated" for c in cat)


def test_unknown_tool_gets_strictest_risk():
    assert connectors.risk_of("wire_money") == connectors.RISK_PAYMENT


def test_execute_is_idempotent():
    first = asyncio.run(connectors.execute("send_email", {"to": "a@example.com", "subject": "hi"}, "k-1"))
    again = asyncio.run(connectors.execute("send_email", {"to": "a@example.com", "subject": "hi"}, "k-1"))
    assert first["ok"] and first["mode"] == "simulated"
    assert again["idempotent_replay"] and again["output"] == first["output"]


def test_live_write_failure_is_reported_not_faked(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    async def boom(args):
        raise IntegrationError("channel_not_found")

    monkeypatch.setattr("app.integrations.slack.post_message", boom)
    res = asyncio.run(connectors.execute("send_message", {"channel": "#x", "text": "hi"}, "k-live"))
    assert res["mode"] == "live" and not res["ok"] and "channel_not_found" in res["error"]


def test_policy_limits_and_budget():
    acts = [
        {"id": "a1", "tool": "charge_card", "args": {"amount_usd": 900}, "depends_on": []},
        {"id": "a2", "tool": "charge_card", "args": {"amount_usd": 450}, "depends_on": []},
        {"id": "a3", "tool": "split_payment", "args": {"amount_usd": 400}, "depends_on": []},
        {"id": "a4", "tool": "charge_card", "args": {"amount_usd": 300}, "depends_on": []},
        {"id": "a5", "tool": "search_inbox", "args": {}, "depends_on": []},
    ]
    out = {a["id"]: a for a in action.apply_policy({"actions": acts})["actions"]}
    assert out["a1"]["policy"].startswith("blocked") and "per-action" in out["a1"]["policy"]
    assert out["a2"]["policy"].startswith("needs human approval")
    assert out["a3"]["policy"].startswith("needs human approval")
    assert "mission budget" in out["a4"]["policy"]
    assert not out["a5"]["requires_approval"]


def test_dependencies_skip_when_parent_not_approved():
    plan = action.apply_policy({"actions": [
        {"id": "a1", "tool": "draft_email", "args": {"to": "x@example.com"}, "depends_on": []},
        {"id": "a2", "tool": "send_email", "args": {"to": "x@example.com"}, "depends_on": ["a1"]},
    ]})["actions"]
    state = {"actions": plan, "decision": {"decision": "approve", "approved": ["a2"]}, "run_id": "dep"}
    res = {a["id"]: a for a in asyncio.run(action.execute(state))["executed"]}
    assert res["a1"]["status"] == "skipped" and res["a2"]["status"] == "skipped"
    assert "dependency" in res["a2"]["reason"]


def test_offline_planner_builds_shopping_and_payment_chain():
    plan = plan_actions_for("Buy headphones under $250, email Maya the invite, split the $180 deposit, remind me Friday")
    tools = [a["tool"] for a in plan["actions"]]
    for t in ("search_products", "add_to_cart", "checkout_cart", "draft_email", "send_email", "split_payment",
              "create_reminder"):
        assert t in tools


def test_amazon_sigv4_headers(monkeypatch):
    monkeypatch.setenv("AMAZON_PAAPI_ACCESS_KEY", "AKIDEXAMPLE")
    monkeypatch.setenv("AMAZON_PAAPI_SECRET_KEY", "secret")
    h = amazon.signed_headers('{"Keywords":"x"}')
    assert h["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
    assert "SignedHeaders=content-encoding;content-type;host;x-amz-date;x-amz-target" in h["Authorization"]


def test_stripe_refuses_live_keys(monkeypatch):
    monkeypatch.setenv("STRIPE_API_KEY", "sk_live_123")
    monkeypatch.delenv("MUSE_ALLOW_LIVE_PAYMENTS", raising=False)
    try:
        stripe_pay._key()
    except IntegrationError as exc:
        assert "live Stripe key" in str(exc)
    else:
        raise AssertionError("live key accepted")


def test_planner_follows_payment_goal_exactly():
    plan = plan_actions_for("Pay the $900 venue invoice, pay $120 for catering and refund $40 to Leo")["actions"]
    got = [(a["tool"], a["args"]["amount_usd"]) for a in plan]
    assert got == [("charge_card", 900.0), ("charge_card", 120.0), ("refund_payment", 40.0)]
    governed = action.apply_policy({"actions": plan})["actions"]
    assert governed[0]["policy"].startswith("blocked") and not governed[1]["policy"].startswith("blocked")


def test_planner_follows_calendar_goal_without_extras():
    plan = plan_actions_for("Schedule a 90 minute planning meeting with Maya and Leo next week and remind me to "
                            "prepare slides")["actions"]
    assert [a["tool"] for a in plan] == ["find_meeting_slots", "create_calendar_event", "create_reminder"]
    window = plan[0]["args"]
    assert window["duration_min"] == 90 and window["within_days"] == 7 and 1 <= window["after_days"] <= 7
    assert plan[2]["args"]["text"] == "Prepare slides"


def test_calendar_event_uses_an_offered_slot():
    found = asyncio.run(action.discover({"actions": plan_actions_for("Schedule a planning meeting with Maya next week")
                                         ["actions"], "run_id": "slot"}))
    plan = action.apply_policy(found)
    assert plan["actions"][1]["args"]["start"]  # the human approves the real slot
    state = {"actions": plan["actions"], "run_id": "slot", "goal": "",
             "decision": {"decision": "approve", "approved": [a["id"] for a in plan["actions"]]}}
    done = {a["tool"]: a for a in asyncio.run(action.execute(state))["executed"]}
    assert done["create_calendar_event"]["output"]["start"] == done["find_meeting_slots"]["output"]["slots"][0]


def test_score_signal_fails_when_a_requested_amount_is_dropped():
    goal = "Pay the $900 venue invoice and refund $40 to Leo"
    assert action._amounts_covered(goal, plan_actions_for(goal)["actions"])
    assert not action._amounts_covered(goal, [{"args": {"amount_usd": 59.0}}])


def test_bound_event_keeps_checked_duration_and_ignores_unrelated_dependencies():
    acts = [{"id": "a1", "tool": "wallet_balance", "title": "w", "args": {}, "depends_on": []},
            {"id": "a2", "tool": "search_products", "title": "s", "args": {}, "depends_on": []},
            {"id": "a3", "tool": "add_to_cart", "title": "c", "args": {}, "depends_on": ["a1", "a2"]},
            {"id": "a4", "tool": "find_meeting_slots", "title": "f", "args": {}, "depends_on": []},
            {"id": "a5", "tool": "create_calendar_event", "title": "e", "args": {}, "depends_on": ["a4"]}]
    outputs = {"a1": {"balance_usd": 10}, "a2": {"results": [{"sku": "B01", "price_usd": 20.0, "title": "Mug"}]},
               "a4": {"slots": ["2026-12-01 10:00"], "duration_min": 30}}
    by_id = {a["id"]: a for a in acts}
    for a in acts:
        action._bind(a, by_id, outputs)
    assert "unresolved" not in acts[2] and acts[2]["args"]["sku"] == "B01"
    assert acts[4]["args"] == {"start": "2026-12-01 10:00", "duration_min": 30}
