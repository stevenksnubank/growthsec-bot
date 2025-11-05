import os
import re
import time
import json
import logging
from urllib.parse import quote, urlparse
import subprocess
import requests
import boto3
from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler
from dotenv import load_dotenv

from kb_mcp_client import KnowledgeBaseMCPClient, parse_server_args
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_resolved_level = getattr(logging, LOG_LEVEL, None)
if not isinstance(_resolved_level, int):
    _resolved_level = logging.INFO
logging.basicConfig(level=_resolved_level)
logging.getLogger(__name__).info(
    "Logging configured",
    extra={"LOG_LEVEL": LOG_LEVEL, "effective_level": logging.getLevelName(_resolved_level)},
)

def kb_search_url():
    base = KB_BASE_URL.rstrip("/")
    if KB_NAME:
        return f"{base}/api/v1/knowledge-bases/{quote(KB_NAME, safe='')}/search"
    return f"{base}/api/v1/knowledge-bases/search"

# Initialize the Bolt app
# NOTE: If running on Lambda, use SlackRequestHandler(app=app)
APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")

if APP_TOKEN:
    # Socket Mode: App needs only the bot token; SocketModeHandler uses the app token when starting
    app = App(token=BOT_TOKEN)
else:
    # Events API over HTTP
    app = App(token=BOT_TOKEN, signing_secret=SIGNING_SECRET)

# KB API configuration (Set these as environment variables)
KB_BASE_URL = os.getenv("KB_BASE_URL", "https://api.your-domain.com")
KB_NAME = os.getenv("KB_NAME", "default-kb") 
KB_TOKEN = os.getenv("KB_TOKEN") 

# Search behavior
KB_LIMIT = int(os.getenv("KB_LIMIT", "5"))
KB_CACHE_TTL_S = int(os.getenv("KB_CACHE_TTL_S", "300"))
S3_PRESIGN_EXPIRES = int(os.getenv("S3_PRESIGN_EXPIRES", "3600"))

# Optional: use org CLI (nu-ist) to call KB (bypasses direct TLS/auth)
KB_USE_NUCLI = os.getenv("KB_USE_NUCLI", "0").lower() in ("1", "true", "yes")
NU_CLI_BIN = os.getenv("NU_CLI_BIN", "nu-ist")
NU_ENV = os.getenv("NU_ENV", "prod")
NU_SHARD = os.getenv("NU_SHARD", "global")
NU_SERVICE = os.getenv("NU_SERVICE", "llm-data-source")
NU_ACCOUNT = os.getenv("NU_ACCOUNT")  # optional, e.g., "ist"

# Optional filters for narrowing results
_filters_env = os.getenv("KB_FILTERS_JSON")
try:
    KB_FILTERS = json.loads(_filters_env) if _filters_env else None
except Exception:
    KB_FILTERS = None


def _split_env_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _parse_env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


KB_TITLE_BOOST_KEYWORDS = _split_env_list(os.getenv("KB_TITLE_BOOST_KEYWORDS"))
KB_URI_BOOST_SUBSTRINGS = _split_env_list(os.getenv("KB_URI_BOOST_SUBSTRINGS"))
KB_TITLE_BOOST_WEIGHT = _parse_env_float("KB_TITLE_BOOST_WEIGHT", "5")
KB_URI_BOOST_WEIGHT = _parse_env_float("KB_URI_BOOST_WEIGHT", "3")

KB_INDEX_NAME = os.getenv("KB_INDEX_NAME")


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


KB_USE_MCP = os.getenv("KB_USE_MCP", "0").lower() in ("1", "true", "yes")
KB_MCP_SERVER_COMMAND = os.getenv("KB_MCP_SERVER_COMMAND")
if not KB_MCP_SERVER_COMMAND:
    _nucli_home = os.environ.get("NUCLI_HOME")
    if _nucli_home:
        _default_mcp_run = os.path.join(
            _nucli_home,
            "nucli.d",
            "llm.d",
            "mcp.d",
            "kb.d",
            "run",
        )
        if os.path.isfile(_default_mcp_run):
            KB_MCP_SERVER_COMMAND = _default_mcp_run

KB_MCP_SERVER_ARGS = parse_server_args(os.getenv("KB_MCP_SERVER_ARGS"))
KB_MCP_SERVER_ENV_JSON = os.getenv("KB_MCP_SERVER_ENV_JSON")
KB_MCP_EXPAND_WINDOW = _parse_optional_int(os.getenv("KB_MCP_EXPAND_WINDOW"))

# Simple in-memory cache: query -> (expires_at, results)
_kb_cache = {}

def get_cached_results(q):
    entry = _kb_cache.get(q)
    if not entry:
        return None
    expires_at, results = entry
    if expires_at < time.time():
        _kb_cache.pop(q, None)
        return None
    return results

def set_cached_results(q, results):
    _kb_cache[q] = (time.time() + KB_CACHE_TTL_S, results)


def _build_mcp_env() -> dict[str, str]:
    env = dict(os.environ)
    if KB_MCP_SERVER_ENV_JSON:
        try:
            extra = json.loads(KB_MCP_SERVER_ENV_JSON)
            if isinstance(extra, dict):
                for key, value in extra.items():
                    if value is not None:
                        env[str(key)] = str(value)
        except Exception:
            pass
    return env


_kb_mcp_client: KnowledgeBaseMCPClient | None = None


def _get_mcp_client(logger=None) -> KnowledgeBaseMCPClient | None:
    global _kb_mcp_client
    if _kb_mcp_client is not None:
        return _kb_mcp_client

    if not KB_MCP_SERVER_COMMAND:
        if logger:
            logger.error(
                "KB_MCP_SERVER_COMMAND is not configured; cannot use MCP knowledge base search."
            )
        return None

    try:
        client = KnowledgeBaseMCPClient(
            command=KB_MCP_SERVER_COMMAND,
            args=KB_MCP_SERVER_ARGS,
            env=_build_mcp_env(),
            default_kb=KB_NAME,
        )
        if logger:
            logger.info(
                "Initialized KB MCP client",
                extra={
                    "command": KB_MCP_SERVER_COMMAND,
                    "args_list": KB_MCP_SERVER_ARGS,
                },
            )
        _kb_mcp_client = client
        return client
    except Exception as exc:
        if logger:
            logger.error(f"Failed to initialize Knowledge Base MCP client: {exc}")
        return None


def normalize_query(text: str) -> str:
    # Convert Slack <url|label> to label
    text = re.sub(r"<[^|>]+\|([^>]+)>", r"\1", text)
    # Remove inline/blocked code spans
    text = re.sub(r"`{1,3}[\s\S]*?`{1,3}", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Truncate to a reasonable length
    return text[:500]


_s3_client = None

def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def _extract_s3_from_result(result: dict):
    """Return (bucket, key) if an S3 object can be inferred from the result."""
    raw_url = (result.get("url") or "").strip()
    # s3://bucket/key
    m = re.match(r"^s3://([^/]+)/(.+)$", raw_url)
    if m:
        return m.group(1), m.group(2)
    # https://bucket.s3.<region>.amazonaws.com/key OR https://s3.<region>.amazonaws.com/bucket/key
    try:
        if raw_url:
            u = urlparse(raw_url)
            if u.scheme in ("http", "https") and "amazonaws.com" in u.netloc:
                path = u.path.lstrip("/")
                if u.netloc.startswith("s3."):
                    parts = path.split("/", 1)
                    if len(parts) == 2:
                        return parts[0], parts[1]
                else:
                    # virtual-hosted-style
                    bucket = u.netloc.split(".")[0]
                    if path:
                        return bucket, path
    except Exception:
        pass
    # Fallback to metadata
    md = result.get("metadata") or {}
    bucket, key = md.get("s3_bucket"), md.get("s3_key")
    if bucket and key:
        return bucket, key
    return None


def _presign_s3_object(bucket: str, key: str) -> str | None:
    try:
        return _get_s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=S3_PRESIGN_EXPIRES,
        )
    except Exception:
        return None


_kb_validated = False


def _rerank_results(results: list[dict], query: str) -> list[dict]:
    if not results:
        return results

    q_lower = (query or "").lower()
    query_terms = {term for term in re.findall(r"\w+", q_lower) if len(term) > 2}

    def compute_score(result: dict) -> float:
        base = result.get("score")
        try:
            score = float(base)
        except (TypeError, ValueError):
            score = 0.0

        title = (result.get("title") or "").lower()
        uri = (result.get("uri") or "").lower()
        bonus = 0.0

        if q_lower and q_lower in title:
            bonus += KB_TITLE_BOOST_WEIGHT
        elif query_terms:
            matched_terms = sum(1 for term in query_terms if term in title)
            if matched_terms:
                bonus += KB_TITLE_BOOST_WEIGHT * (matched_terms / len(query_terms))

        for kw in KB_TITLE_BOOST_KEYWORDS:
            if kw and kw in title:
                bonus += KB_TITLE_BOOST_WEIGHT
                break

        for pattern in KB_URI_BOOST_SUBSTRINGS:
            if pattern and pattern in uri:
                bonus += KB_URI_BOOST_WEIGHT
                break

        return score + bonus

    return sorted(results, key=compute_score, reverse=True)

def validate_kb(logger=None) -> bool:
    global _kb_validated
    if _kb_validated:
        return True
    if not KB_NAME:
        _kb_validated = True
        return True
    if KB_USE_MCP:
        _kb_validated = True
        return True
    if KB_USE_NUCLI:
        # Assume CLI config/auth handles routing; skip HTTP validation
        _kb_validated = True
        return True
    url = f"{KB_BASE_URL.rstrip('/')}/api/v1/knowledge-bases/{quote(KB_NAME, safe='')}"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {KB_TOKEN}"}, timeout=10)
        r.raise_for_status()
        info = r.json() or {}
        if logger:
            logger.info(f"KB '{KB_NAME}' accessible; current_index={info.get('current_index')}")
        _kb_validated = True
        return True
    except Exception as e:
        if logger:
            logger.error(f"KB '{KB_NAME}' not accessible: {e}")
        return False


def search_and_respond(query, channel_id, thread_ts, client, logger):
    """
    Core function to execute RAG search and post the formatted response.
    """
    if not query:
        client.chat.postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text='Hi! Please provide a question, e.g., "What is the data retention policy?"'
        )
        return

    # Normalize and guard against too-short queries
    query = normalize_query(query)
    if len(query) < 3:
        client.chat.postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text="Could you add a bit more detail?",
        )
        return

    # 1. Send immediate feedback (optional, but good UX)
    client.chat_postMessage(
        channel=channel_id,
        thread_ts=thread_ts,
        text=f':mag: Searching knowledge base for: “{query}”…'
    )

    # 2. Build and execute the KB search request
    try:
        results = get_cached_results(query)
        if results is None and KB_USE_MCP:
            mcp_client = _get_mcp_client(logger)
            if mcp_client:
                try:
                    mcp_response = mcp_client.search(
                        query,
                        kb_name=KB_NAME,
                        index_name=KB_INDEX_NAME,
                        num_results=KB_LIMIT,
                        expand_window=KB_MCP_EXPAND_WINDOW,
                        filters=KB_FILTERS,
                        logger=logger,
                    )
                    results = mcp_response.get("results", [])
                    if logger:
                        logger.info(
                            "KB MCP search completed",
                            extra={
                                "query": query,
                                "results": len(results or []),
                            },
                        )
                except Exception as mcp_error:
                    if logger:
                        logger.error(f"MCP knowledge base search failed: {mcp_error}")
                    results = None

        if results is None and KB_USE_NUCLI:
            if logger:
                logger.info(
                    "Falling back to nu-cli KB search",
                    extra={"query": query},
                )
            kb_payload = {
                "query": query,
                "knowledge-bases": [{"name": KB_NAME}],
                "num-results": KB_LIMIT,
            }
            if KB_FILTERS:
                kb_payload["filter"] = KB_FILTERS
            cmd = [NU_CLI_BIN, "ser", "curl", "--env", NU_ENV]
            if NU_ACCOUNT:
                cmd += ["--account", NU_ACCOUNT]
            cmd += [
                "POST",
                NU_SHARD,
                NU_SERVICE,
                "/api/v1/knowledge-bases/search",
                "--data",
                json.dumps(kb_payload),
            ]
            try:
                completed = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                data = json.loads(completed.stdout) if completed.stdout else {}
                results = data.get("results", []) if isinstance(data, dict) else []
            except subprocess.CalledProcessError as e:
                if logger:
                    logger.error(f"nu-cli global search failed (rc={e.returncode}): {e.stderr}")
                kb_payload_pk = {"query": query, "num-results": KB_LIMIT}
                if KB_FILTERS:
                    kb_payload_pk["filter"] = KB_FILTERS
                cmd_pk = [NU_CLI_BIN, "ser", "curl", "--env", NU_ENV]
                if NU_ACCOUNT:
                    cmd_pk += ["--account", NU_ACCOUNT]
                cmd_pk += [
                    "POST",
                    NU_SHARD,
                    NU_SERVICE,
                    f"/api/v1/knowledge-bases/{quote(KB_NAME, safe='')}/search",
                    "--data",
                    json.dumps(kb_payload_pk),
                ]
                try:
                    completed_pk = subprocess.run(
                        cmd_pk,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    data = json.loads(completed_pk.stdout) if completed_pk.stdout else {}
                    results = data.get("results", []) if isinstance(data, dict) else []
                except subprocess.CalledProcessError as cli_error:
                    if logger:
                        logger.error(f"nu-cli per-KB search failed (rc={cli_error.returncode}): {cli_error.stderr}")
                    results = None

        if results is None:
            if not validate_kb(logger):
                client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text="Knowledge base is not accessible right now. Please try again later.",
                )
                return

            url = kb_search_url()
            headers = {
                "Authorization": f"Bearer {KB_TOKEN}",
                "Content-Type": "application/json",
            }
            payload = {
                "query": query,
                "limit": KB_LIMIT,
            }
            if KB_FILTERS:
                payload["filters"] = KB_FILTERS

            for attempt in range(3):
                if logger:
                    logger.info(
                        "Calling KB HTTP search",
                        extra={"query": query, "attempt": attempt + 1},
                    )
                resp = requests.post(url, headers=headers, json=payload, timeout=10)
                if resp.status_code == 429 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", []) if isinstance(data, dict) else []
                break

        if results is not None:
            results = _rerank_results(results, query)
            set_cached_results(query, results)
            if logger:
                logger.info(
                    "KB search finished",
                    extra={
                        "query": query,
                        "result_count": len(results or []),
                    },
                )

        if not results:
            client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=f'No relevant evidence found in the Nullm Knowledge Base for “{query}”.',
            )
            return

        # 3. Build Slack blocks for RAG response
        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Top evidence results for: “{query}”*"}
            },
            {"type": "divider"}
        ]
        
        # NOTE: If your external KB provides the LLM-generated answer text,
        # you would put that text here as the first block instead of this header.
        
        for i, r in enumerate(results[:5]):
            title = r.get("title") or "Unknown Document"
            link = r.get("url")
            # Truncate snippet to fit well in Slack block
            snippet = (r.get("snippet") or r.get("content") or "")[:250]
            if snippet and not snippet.endswith("..."):
                snippet += "..."

            # Prefer a presigned S3 link when possible
            doc_link = link
            s3_info = _extract_s3_from_result(r)
            if s3_info:
                presigned = _presign_s3_object(*s3_info)
                if presigned:
                    doc_link = presigned

            text_block = f"*{title}*"
            if doc_link:
                text_block += f"  ( <{doc_link}|:page_facing_up: View Document> )"
            if snippet:
                text_block += f"\n> {snippet}"

            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": text_block},
                }
            )

        # 4. Post the final results
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=f'Top results for “{query}”:',
            blocks=blocks,
        )
        
    except Exception as e:
        logger.error(
            f"KB search error: {getattr(e, 'response', None) and e.response.text or e}"
        )
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text="Sorry, I ran into an issue while searching the knowledge base. Please try again later.",
        )


# --- SLACK EVENT HANDLERS ---

@app.event("app_mention")
def handle_mention(event, client, logger):
    """Handles mentions in channels (@GrowthSec Assistant what's the policy?)."""
    text = event.get("text", "")
    # Remove the <@BOTID> mention to get the clean query
    query = re.sub(r"<@[^>]+>\s*", "", text).strip()
    
    # Use the client to call the core search function
    search_and_respond(
        query=query, 
        channel_id=event["channel"], 
        thread_ts=event.get("ts"),
        client=client, 
        logger=logger
    )


@app.message(re.compile(".*"))
def handle_direct_message(message, client, logger):
    """Handles messages sent directly to the bot."""
    # Only handle DMs here; ignore channel messages
    if message.get("channel_type") != "im":
        return
    # In a DM, the message text is the query
    query = message.get("text", "").strip()
    
    # Use the client to call the core search function
    search_and_respond(
        query=query, 
        channel_id=message["channel"], 
        thread_ts=message.get("ts"),
        client=client, 
        logger=logger
    )


# --- MAIN EXECUTION ---
# This part is for local development or traditional server deployment
if __name__ == "__main__":
    if APP_TOKEN:
        # Start Socket Mode (no web server required)
        from slack_bolt.adapter.socket_mode import SocketModeHandler
        print("Starting Slack Bolt App in Socket Mode...")
        SocketModeHandler(app, APP_TOKEN).start()
    else:
        print("Starting Slack Bolt App (HTTP)...")
        app.start(port=int(os.environ.get("PORT", 3000)))

# If deploying to AWS Lambda with API Gateway:
# handler = SlackRequestHandler(app)
# def lambda_handler(event, context):
#     return handler.handle(event, context)
