import os
import tempfile

os.environ.setdefault("MUSE_DATA_DIR", tempfile.mkdtemp(prefix="muse-test-"))
for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
          "STRIPE_API_KEY", "SLACK_BOT_TOKEN", "GOOGLE_ACCESS_TOKEN", "AMAZON_PAAPI_ACCESS_KEY", "AMADEUS_CLIENT_ID"):
    os.environ.pop(k, None)
