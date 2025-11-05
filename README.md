## GrowthSec Slack KB Bot

A simple Slack bot (Slack Bolt for Python) that listens for `app_mention` and DMs, then queries your Knowledge Base API and posts top evidence back into Slack.

### Prerequisites

- Python 3.9+
- A Slack app with a Bot Token (`chat:write`, `app_mentions:read`). For DMs, add `im:history` and subscribe to `message.im`.
- Choose one connection method:
  - Socket Mode (simplest locally): enable, create an App-Level Token with `connections:write`, set `SLACK_APP_TOKEN`.
  - Events API via public URL: expose port 3000 (e.g. ngrok/cloudflared) and set the Request URL to `/slack/events`.

### Setup

1. Create a virtual environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables (copy `.env.example` and set values):
```bash
export SLACK_BOT_TOKEN="xoxb-..."
export SLACK_SIGNING_SECRET="..."
export KB_BASE_URL="https://api.your-domain.com"
export KB_NAME="internal-help"
export KB_TOKEN="..."
export PORT=3000
export LOG_LEVEL=INFO
# Optional for Socket Mode:
# export SLACK_APP_TOKEN="xapp-..."
# Retrieval tuning (optional):
# export KB_LIMIT=5
# export KB_CACHE_TTL_S=300
# export KB_FILTERS_JSON='{"project":"growthsec"}'
# MCP adapter (optional):
# export KB_USE_MCP=true
# export KB_MCP_SERVER_COMMAND="/absolute/path/to/nu/nucli/nucli.d/llm.d/mcp.d/kb.d/run"
# export KB_MCP_SERVER_ARGS=""
# export KB_MCP_SERVER_ENV_JSON='{"PATH":"$PATH"}'
# export KB_MCP_EXPAND_WINDOW=4000
# export KB_INDEX_NAME="knowledge-base-index"
```

4. Run the app:
```bash
python app.py
```

### Connection options

#### A) Socket Mode (recommended locally)
- Enable Socket Mode in your Slack app.
- Create an App-Level Token with scope `connections:write` and set `SLACK_APP_TOKEN`.
- Run the app (no public URL required).

#### B) Events API via public URL
- Start the app, then expose port `3000`:
  - ngrok: `ngrok http 3000`
  - cloudflared: `cloudflared tunnel --url http://localhost:3000`
- In Slack app: Event Subscriptions → set Request URL to `https://<your-tunnel>/slack/events`.
- Subscribe to `app_mention` (and `message.im` for DMs). Reinstall the app after changing scopes/events.

### How it works

- The bot normalizes the query text (removes Slack formatting, code blocks, extra whitespace) and guards against too-short queries.
- It validates the configured knowledge base once, then performs a search:
  - If `KB_USE_MCP=true`, the bot spawns the `nu-knowledge-base-mcp-nu-cli` stdio server and calls the `searchKnowledgeBase` tool.
  - Otherwise it uses HTTP / CLI fallbacks:
    - Per-KB: `POST {KB_BASE_URL}/api/v1/knowledge-bases/{KB_NAME}/search`
    - Global: `POST {KB_BASE_URL}/api/v1/knowledge-bases/search` (via `nu-cli` when `KB_USE_NUCLI=true`)
- It uses a small in-memory cache with TTL and gentle retry/backoff on `429` responses.
- Results are posted back using Slack blocks (title, source link, snippet).

### Notes

- Mentions: the handler trims the `<@BOTID>` mention before searching.
- DMs: messages in IMs are treated as full queries.
- Tuning: `KB_LIMIT`, `KB_CACHE_TTL_S`, and optional `KB_FILTERS_JSON` can narrow/optimize retrieval.
- Ingestion/maintenance endpoints (indexes/items) are not used by the bot and should run in a separate pipeline.

### MCP integration (optional)

- Set `KB_USE_MCP=true` to route queries through the MCP server shipped in `nu-knowledge-base-mcp-nu-cli`.
- Provide `KB_MCP_SERVER_COMMAND` (usually `~/dev/nu/nucli/nucli.d/llm.d/mcp.d/kb.d/run`) and optional `KB_MCP_SERVER_ARGS` if you need to wrap the invocation (e.g. `-lc "..."`).
- The bot passes your current environment to the subprocess; use `KB_MCP_SERVER_ENV_JSON` to add or override specific variables (for example to inject a custom `PATH`).
- Optional knobs:
  - `KB_INDEX_NAME` to scope searches to a specific index.
  - `KB_MCP_EXPAND_WINDOW` to override the server’s expand window (default 4000 tokens).
- Slack will continue to use the same block formatting—the MCP server simply feeds the evidence list while centralising query expansion and re-ranking logic.

### Testing

1) Invite the bot to a channel and mention it: `@YourBot what's the data retention policy?`
2) DM the bot: e.g. `Help with VPN access`.
3) Optional: mock the KB by pointing `KB_BASE_URL` to a local test server if your backend isn’t ready.

### Troubleshooting

- Bot not responding in channels: confirm it’s invited and subscribed to `app_mention`; reinstall after scope/event changes.
- DMs not triggering: add `im:history` scope and subscribe to `message.im`.
- Events URL not verified: ensure the Request URL ends with `/slack/events` and your tunnel is running.
- KB errors: verify `KB_*` env vars, token validity, and that `/search` returns a JSON object with a `results` array.

