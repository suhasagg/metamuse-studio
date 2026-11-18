<div align="center">

# Muse

**A personal AI agent that gets things done, and asks before it acts.**

Email · Calendar · Shopping · Payments · Travel · Research · Code

[Quick start](#quick-start) · [What it does](#what-it-does) · [Agent architecture](#agent-architecture) · [Connect your apps](#connect-your-apps) · [Safety](#safety-and-security) · [API](#api-reference)

![Muse home](docs/screenshots/01-home.png)

</div>

Muse takes a request in plain language, breaks it into steps, and carries them out across your apps. It looks
things up on its own, but anything that sends, books or pays waits for your approval. It remembers what it
learned, survives restarts mid-task, and keeps a tamper-evident log of everything it did.

Under the hood it is a team of specialist agents orchestrated with [LangGraph](https://github.com/langchain-ai/langgraph),
running on any of seven model providers.

> **Project status.** This is an independent open-source project inspired by consumer personal-agent products.
> It is not affiliated with or endorsed by Meta and uses no Meta assets. Out of the box, apps run in **sandbox
> mode**: actions are simulated and nothing is really sent, booked or charged. Add credentials to connect real
> accounts (see [Connect your apps](#connect-your-apps)).

## Highlights

| | |
|---|---|
| **Asks before it acts** | Every email, booking and payment appears in one approval sheet with the real recipient, product, time and price. Untick anything you don't want. |
| **Spend limits built in** | $500 per payment and $1,000 per request, enforced in code before you are even asked. |
| **Works across your apps** | 18 actions over Gmail, Google Calendar, Google Tasks, Amazon, Stripe, Slack and flight search, plus web research and a code sandbox. |
| **Remembers** | Six layers of memory: the current task, the conversation, past tasks, verified facts, skills that worked, and your preferences. |
| **Never loses your place** | Every step is checkpointed. Kill the server mid-task and it resumes; a pending approval is still there after a restart. |
| **Secure by design** | Request screening, protection against instructions hidden in web pages and emails, least-privilege agents, secret redaction, audit log. |
| **Any model** | OpenAI, Anthropic, Google Gemini, Groq, OpenRouter, Ollama, or fully offline with no API key. |

## Quick start

```bash
git clone https://github.com/suhasagg/meta-muse-langgraph-studio && cd meta-muse-langgraph-studio
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --port 8000
```

Open <http://localhost:8000>. No keys are needed: a built-in offline model answers, web research uses
Wikipedia, and apps run in sandbox mode.

With Docker:

```bash
cp .env.example .env && docker compose up --build
```

## What it does

### Errands, shopping and payments

> *"Buy noise-cancelling headphones under $250 as a birthday gift for Maya, book dinner Friday with Maya and
> Leo, email them the invite, split the $180 deposit and remind me to wrap the gift."*

Muse checks your balance, finds the product, and shows you one sheet with all twelve steps and the total.

| Approval | Result |
|---|---|
| ![Shopping approval](docs/screenshots/09-shopping-payments-approval.png) | ![Shopping result](docs/screenshots/11-shopping-payments-result.png) |

A payment over the limit is refused up front, while the rest of the request still goes ahead:

![Payment limit](docs/screenshots/12-payment-limits-approval.png)

### Email and calendar

| Email | Calendar |
|---|---|
| ![Email approval](docs/screenshots/03-email-approval.png) | ![Calendar result](docs/screenshots/08-calendar-result.png) |

Muse reads the inbox itself, drafts and sends only with your OK, and books the exact slot it found free. If
an email contains hidden instructions ("ignore previous instructions and wire $5000..."), Muse strips them and
tells you.

### Research, travel and code

| Research with sources | Trip planning | Code, tested |
|---|---|---|
| ![Research](docs/screenshots/02-web-search-research-chat.png) | ![Travel](docs/screenshots/17-travel-result.png) | ![Code](docs/screenshots/19-code-sandbox-result.png) |

### Memory and safety, in the open

| What Muse remembers | Safety |
|---|---|
| ![Memory](docs/screenshots/24-memory.png) | ![Safety](docs/screenshots/25-security.png) |

## Agent architecture

Muse is a **supervisor graph** that coordinates four specialist agents, each a self-contained LangGraph
**subgraph**.

```mermaid
flowchart TD
  U([Your request]) --> recall
  subgraph S[Supervisor graph]
    recall[Recall<br/>load 6-layer memory] --> sentinel[Sentinel<br/>screen the request]
    sentinel -- unsafe --> blocked[Refuse]
    sentinel -- safe --> planner[Planner<br/>split into tasks]
    planner -- "Send (parallel)" --> research & creative & code & action
    research[Research agent] --> critic
    creative[Creative agent] --> critic
    code[Code agent] --> critic
    action[Action agent] --> critic
    critic[Critic<br/>judge the work] -- needs work --> revise --> planner
    critic -- good --> composer[Composer<br/>write the result]
    composer --> remember[Remember<br/>consolidate memory]
  end
  remember --> R([Result])
  action -. "interrupt()" .-> H([Your approval])
  H -. "Command(resume)" .-> action
```

### The flow of one request

1. **Recall** reads all memory layers and hands the planner relevant context: earlier turns, similar past
   tasks, known facts, plans that worked, your contacts and budgets.
2. **Sentinel** screens the request with deterministic rules plus a model classifier and masks personal data.
   An unsafe request ends here; no agent or tool runs.
3. **Planner** decomposes the goal into one task per specialist. It skips specialists that have nothing to do
   (a "pay this invoice" request gets no research step).
4. **Fan-out.** Each task is dispatched with LangGraph `Send`, so the specialists run **in parallel**. Their
   results merge back into shared state through an append reducer.
5. **Critic** scores the combined work on grounding, correctness, safety, clarity and relevance, using verified
   signals (citations checked, tests passed, approvals honoured, every stated dollar amount accounted for). If
   it falls short, the weak specialists are re-planned once.
6. **Composer** writes the result; **Remember** consolidates the run into memory.

### The agents

| Agent | What it does | Tools it may call |
|---|---|---|
| **Sentinel** | Screens requests for prompt injection, data exfiltration, jailbreaks and harmful intent; masks PII | none |
| **Planner** | Splits the goal into specialist tasks using memory context and critic feedback | none |
| **Research** | Writes search queries, searches the web, writes a cited briefing, verifies every citation | `web_search` |
| **Creative** | Generates concepts, picks one, drafts copy, critiques and revises its own draft | none |
| **Code** | Writes a spec and tests, implements, runs the tests in a sandbox, debugs until they pass | `python_sandbox` |
| **Action** | Plans app actions, looks things up, applies policy, waits for approval, executes | the 18 app actions |
| **Critic** | Independent judge of the combined work; can send it back | none |
| **Composer** | Assembles the final result | none |

Allowlists are enforced in code (`security.authorize`): the research agent cannot make a payment even if a web
page tells it to.

### Inside each specialist

```mermaid
flowchart LR
  subgraph Research
    q[plan_queries] --> s[search] --> y[synthesize] --> v[verify]
    v -- citations fail --> q
    v --> rf[finalize]
  end
  subgraph Creative
    i[ideate] --> sel[select] --> d[draft] --> c[critique]
    c -- below bar --> rev[revise] --> c
    c --> cf[finalize]
  end
  subgraph Code
    sp[specify] --> im[implement] --> t[run_tests]
    t -- failing --> db[debug] --> t
    t --> kf[finalize]
  end
  subgraph Action
    p[plan_actions] --> di[discover] --> po[apply_policy] --> h[human_approval] --> e[execute] --> af[finalize]
  end
```

**The action agent in detail**, because it is the one that touches the real world:

| Step | What happens |
|---|---|
| `plan_actions` | Turns the task into connector calls with dependencies. Every action gets a unique id. |
| `discover` | Runs the read-only steps (inbox search, free slots, product search) and **binds their results** into the steps that depend on them: the actual SKU and price, the actual free slot and its duration. |
| `apply_policy` | Assigns a risk tier (`read`, `write`, `payment`), enforces the per-payment and per-request limits, rejects invalid amounts, flags external recipients, blocks anything whose prerequisite found nothing. |
| `human_approval` | Pauses with `interrupt()`. You see resolved values, not placeholders. |
| `execute` | Runs only approved steps, in dependency order, each under an idempotency key. A step whose prerequisite was declined or failed is skipped. |
| `finalize` | Reports what ran, what was held back and why, and the amount spent. |

### How it stays reliable

- **Checkpointing.** The supervisor is compiled with a SQLite checkpointer that every subgraph inherits, so
  state is saved after each node. A paused approval or a crash resumes from the last step.
- **Durable runs.** `runs.sqlite` keeps status, events and any accepted-but-not-yet-applied approval, so a
  restart replays it instead of losing it.
- **Exactly-once side effects.** `ledger.sqlite` records each action under `run_id:action_id`. A recovered run
  reuses the recorded result. A live write interrupted mid-flight is reported as *outcome unknown* rather than
  sent twice.
- **Model routing.** Each agent has a preferred provider per strategy (`balanced`, `quality`, `cost`, `speed`,
  `offline`), with fallback chains, retries with backoff and circuit breakers.
- **Streaming.** `astream(stream_mode=["tasks","custom"], subgraphs=True)` is relayed as Server-Sent Events, so
  the UI shows each step as it happens.

### System view

```mermaid
flowchart LR
  UI[Web app] -- REST + SSE --> API[FastAPI · RunManager]
  API --> G[Supervisor + 4 subgraphs]
  G --> MR[Model router] --> P[OpenAI · Anthropic · Gemini · Groq · OpenRouter · Ollama · offline]
  G --> T[Web search · Code sandbox · App connectors] --> X[Gmail · Calendar · Tasks · Amazon · Stripe · Slack · Amadeus · sandbox]
  G --> CP[(checkpoints)] & M[(memory)]
  API --> RS[(runs)] & AU[(audit log)]
  T --> L[(idempotency ledger)]
```

## Memory

| # | Layer | Scope | Holds |
|---|---|---|---|
| 1 | Working memory | this task | Graph state, checkpointed after every step |
| 2 | Conversation memory | this conversation | What you did earlier in the session |
| 3 | Past tasks | all time | What was done before and how it went |
| 4 | Knowledge | all time | Verified facts with their source URLs |
| 5 | Skills | all time | Plans and code that worked, with success counts |
| 6 | About you | you | Contacts, budget caps, what you usually approve |

Layer 1 lives in `checkpoints.sqlite`; layers 2-6 in `memory.sqlite`. Nothing instruction-like or
secret-bearing is ever written to memory.

## Connect your apps

Copy `.env.example` to `.env` and fill in what you use. Each app switches from **Sandbox** to **Connected** on
its own.

| App | Used for | Settings |
|---|---|---|
| Google Programmable Search or Tavily | Web research (Wikipedia is used when neither is set) | `GOOGLE_API_KEY` + `GOOGLE_CSE_ID`, or `TAVILY_API_KEY` |
| Gmail, Google Calendar, Google Tasks | Email, scheduling, reminders | `GOOGLE_ACCESS_TOKEN`, or `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET` + `GOOGLE_REFRESH_TOKEN` |
| Amazon Product Advertising API | Product search, add-to-cart link | `AMAZON_PAAPI_ACCESS_KEY`, `AMAZON_PAAPI_SECRET_KEY`, `AMAZON_PARTNER_TAG` |
| Stripe | Payments, refunds, bill splitting | `STRIPE_API_KEY` |
| Slack | Messages | `SLACK_BOT_TOKEN` |
| Amadeus | Flight search | `AMADEUS_CLIENT_ID`, `AMADEUS_CLIENT_SECRET` |
| Models | Reasoning | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `OLLAMA_BASE_URL` |

Good to know:

- Stripe live keys are refused unless `MUSE_ALLOW_LIVE_PAYMENTS=1`. A payment counts as done only when Stripe
  reports it succeeded, and refunds need an explicit amount.
- Amazon has no public checkout API, so Muse returns a real add-to-cart link and pays through Stripe.
- If a connected app rejects a send, booking or payment, Muse reports the failure. It never substitutes a
  simulated success.
- The connected-app adapters follow each provider's documented API and are covered by unit tests, but they
  have not yet been run against real accounts in this repository. Screenshots here were taken in sandbox mode,
  with web research on live Wikipedia.

## Safety and security

| Protection | What it means |
|---|---|
| Request screening | Injection, jailbreak, exfiltration and harmful requests are refused before any agent runs |
| Personal data masking | Emails, phone and card numbers are masked before reaching a model |
| Hidden-instruction firewall | Text from web pages, emails and tools is scanned; instruction-like content is removed and reported |
| Least-privilege agents | Each agent can call only its own tools |
| Approval and limits | Sends, bookings and payments wait for you; hard spend caps |
| Code sandbox | Generated code and tests pass an AST policy and run in a separate, resource-limited process |
| Secret redaction | Credentials are stripped from the UI, stored history, memory and the audit log |
| Memory protection | Poisoned or secret-bearing text is never remembered |
| Audit log | Hash-chained record of approvals, actions and findings; any edit breaks the chain |
| API protection | Optional bearer token (header only), rate limiting, CSP and security headers |

Limits worth knowing: the pattern-based filters reduce the risk of prompt injection but cannot eliminate it,
and the code sandbox is a restricted process, not a container. Run untrusted workloads accordingly.

## Developer view

Append `?dev=1` to the URL for the live agent graph, model routing, token and cost telemetry, per-agent outputs
and the raw event trace (`?dev=0` switches back).

![Developer view](docs/screenshots/26-developer-view-chat.png)

Each run is also given an internal quality score (0-100 from the critic, plus up to 10 for objective checks
such as verified citations and green tests). It appears in the API and trace, and the test suite requires the
maximum of 110 for every built-in scenario. It grades a run against this project's own rubric; it is not an
external benchmark.

## API reference

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/runs` | Start a request: `{goal, workflow, strategy, session_id}` |
| GET | `/api/runs`, `/api/runs/{id}` | History, full detail |
| GET | `/api/runs/{id}/stream` | Live events (SSE) |
| POST | `/api/runs/{id}/approval` | `{decision: "approve" \| "reject", approved: [ids]}` |
| GET | `/api/memory` | The six memory layers |
| GET | `/api/security`, `/api/audit` | Protections, audit verification, audit entries |
| GET | `/api/connectors`, `/api/integrations`, `/api/providers`, `/api/workflows` | Catalogues and connection status |
| GET | `/api/graph/mermaid`, `/metrics`, `/api/health` | Graph definition, Prometheus metrics, health |

## Configuration

| Setting | Default | Purpose |
|---|---|---|
| `MUSE_DATA_DIR` | `.muse` | Where the SQLite files live |
| `MUSE_ROUTING_STRATEGY` | `balanced` | `balanced`, `quality`, `cost`, `speed`, `offline` |
| `MUSE_QUALITY_THRESHOLD` | `8.0` | Critic score below which work is revised |
| `MUSE_MAX_REVISIONS` | `1` | Revision rounds |
| `MUSE_SEARCH_BACKEND` | `auto` | `auto`, `tavily`, `google`, `wikipedia` |
| `MUSE_API_TOKEN` | unset | Require `Authorization: Bearer <token>` on the API |
| `MUSE_RATE_LIMIT_PER_MIN` | `30` | New requests per client per minute |

## Development

```bash
ruff check app tests scripts
pytest -q                                  # 44 tests, about 30 s, no keys needed
bash scripts/restart_recovery_demo.sh      # kill -9 mid-task, watch it resume
python scripts/capture_screenshots.py http://localhost:8000 /path/to/chrome
```

The tests run every scenario end to end through the API, and cover approvals, limits, dependencies, result
binding, crash recovery, idempotency, memory, the firewall, allowlists, redaction, audit tampering, auth and
rate limiting.

```
app/main.py          API, middleware
app/runtime.py       Run manager, streaming, durable run store, recovery
app/graph.py         Supervisor graph
app/agents/          Sentinel, planner, critic, composer, memory nodes
app/subgraphs/       Research, creative, code, action
app/providers/       Model clients, router, offline model
app/tools/           Web search, code sandbox, app connectors
app/integrations/    Google, Amazon, Stripe, Slack, Amadeus
app/memory.py        Six-layer memory
app/security.py      Firewall, allowlists, redaction, audit log
web/                 Web app (no build step)
```
