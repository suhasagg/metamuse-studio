import asyncio
from datetime import UTC, datetime

import pytest

from app.integrations import IntegrationError, google, stripe_pay
from app.providers.mock_provider import plan_actions_for
from app.runtime import RunRecord, RunStore
from app.subgraphs import action
from app.tools import connectors
from app.tools.sandbox import policy_violations, run_python_tests


@pytest.mark.parametrize("amount", [-900, float("nan"), float("inf"), 0, "abc"])
def test_invalid_payment_amounts_are_blocked_and_never_free_budget(amount):
    acts = [{"id": "a1", "tool": "charge_card", "args": {"amount_usd": amount}, "depends_on": []},
            {"id": "a2", "tool": "charge_card", "args": {"amount_usd": 450}, "depends_on": []},
            {"id": "a3", "tool": "charge_card", "args": {"amount_usd": 450}, "depends_on": []},
            {"id": "a4", "tool": "charge_card", "args": {"amount_usd": 450}, "depends_on": []}]
    out = action.apply_policy({"actions": acts})["actions"]
    assert out[0]["policy"].startswith("blocked")
    assert "mission budget" in out[3]["policy"]


def test_duplicate_model_ids_cannot_widen_an_approval():
    acts = action._unique_ids([{"id": "a1", "tool": "draft_email", "args": {}},
                               {"id": "a1", "tool": "charge_card", "args": {"amount_usd": 400}, "depends_on": ["a1"]}])
    assert [a["id"] for a in acts] == ["a1", "a2"]
    assert acts[1]["depends_on"] == ["unknown:a1"]


def test_shopping_binds_the_searched_product_before_approval():
    plan = plan_actions_for("Buy headphones under $200")["actions"]
    found = asyncio.run(action.discover({"actions": plan, "run_id": "bind"}))["actions"]
    by_tool = {a["tool"]: a for a in found}
    top = connectors.recorded(f"bind:{by_tool['search_products']['id']}")["output"]["results"][0]
    assert by_tool["add_to_cart"]["args"]["sku"] == top["sku"]
    assert by_tool["checkout_cart"]["args"]["amount_usd"] == top["price_usd"]


def test_empty_search_blocks_the_dependent_purchase():
    acts = [{"id": "a1", "tool": "search_products", "title": "s", "args": {}, "depends_on": []},
            {"id": "a2", "tool": "add_to_cart", "title": "c", "args": {}, "depends_on": ["a1"]},
            {"id": "a3", "tool": "checkout_cart", "title": "p", "args": {"amount_usd": 10}, "depends_on": ["a2"]}]
    by_id = {a["id"]: a for a in acts}
    for a in acts:
        action._bind(a, by_id, {"a1": {"results": []}})
    out = action.apply_policy({"actions": acts})["actions"]
    assert out[1]["policy"].startswith("blocked") and out[2]["policy"].startswith("blocked")
    assert action._fmt_output({"results": []}) == "0 products" and action._fmt_output({"options": []}) == "0 flight options"


def test_sandbox_blocks_io_and_untrusted_tests():
    assert policy_violations("import io\nio.open('/etc/passwd')")
    assert policy_violations("f = open\nf('/etc/passwd')")
    report = asyncio.run(run_python_tests("def f():\n    return 1", "import os\nassert f() == 1"))
    assert not report["passed"] and any("tests:" in v for v in report["policy_violations"])


def test_stripe_guards(monkeypatch):
    monkeypatch.delenv("MUSE_ALLOW_LIVE_PAYMENTS", raising=False)
    monkeypatch.setenv("STRIPE_API_KEY", "rk_live_abc")
    with pytest.raises(IntegrationError):
        stripe_pay._key()
    monkeypatch.setenv("STRIPE_API_KEY", "sk_test_abc")
    with pytest.raises(IntegrationError, match="positive amount"):
        asyncio.run(stripe_pay.refund({"payment_intent": "pi_1"}))

    async def pending(path, data, idem=None):
        return {"id": "pi_1", "amount": 500, "status": "requires_action", "livemode": False}

    monkeypatch.setattr(stripe_pay, "_post", pending)
    with pytest.raises(IntegrationError, match="not completed"):
        asyncio.run(stripe_pay.charge({"amount_usd": 5}))


def test_google_dates_are_preserved_or_rejected():
    assert google._when("2026-12-25", 7).date().isoformat() == "2026-12-25"
    assert google._when("2026-12-01 10:00", 7) == datetime(2026, 12, 1, 10, 0, tzinfo=UTC)
    assert google._when("tomorrow 09:00", 1) > datetime.now(UTC)
    with pytest.raises(IntegrationError):
        google._when("sometime soon", 1)


def test_interrupted_live_write_is_not_repeated(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    connectors._ledger_put("crash-key", {"pending": True, "tool": "send_message"})  # died mid-dispatch
    called = []

    async def post(args):
        called.append(args)
        return {"status": "posted"}

    monkeypatch.setattr("app.integrations.slack.post_message", post)
    res = asyncio.run(connectors.execute("send_message", {"text": "hi"}, "crash-key"))
    assert not res["ok"] and "outcome unknown" in res["error"] and not called


def test_run_history_is_redacted_and_keeps_an_accepted_approval(tmp_path):
    store = RunStore(tmp_path / "r.sqlite")
    run = RunRecord(id="x", goal="use key sk-proj-abcdefghijklmnopqrstuv", workflow="auto", strategy="balanced",
                    status="awaiting_approval", resume_decision={"decision": "approve", "approved": ["a1"]})
    store.save(run)
    loaded = store.load()[0]
    assert "sk-proj" not in loaded.goal and "sk-proj" not in run.detail()["goal"]
    assert loaded.resume_decision == {"decision": "approve", "approved": ["a1"]}


def test_thousands_separators_in_amounts():
    plan = plan_actions_for("Pay $1,200 for the venue")["actions"]
    assert plan[0]["args"]["amount_usd"] == 1200.0
    assert action.apply_policy({"actions": plan})["actions"][0]["policy"].startswith("blocked")
