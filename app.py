import os
import re
import time
import json
from urllib.parse import quote, urlparse
import subprocess
import requests
import boto3
from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler
from dotenv import load_dotenv
load_dotenv()

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

# Optional filters for narrowing results
_filters_env = os.getenv("KB_FILTERS_JSON")
try:
    KB_FILTERS = json.loads(_filters_env) if _filters_env else None
except Exception:
    KB_FILTERS = None

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

def validate_kb(logger=None) -> bool:
    global _kb_validated
    if _kb_validated:
        return True
    if not KB_NAME:
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

    try:
        # Use cache first
        results = get_cached_results(query)
        if results is None:
            if KB_USE_NUCLI:
                # Use global multi-KB search via CLI for reliability behind corp network
                kb_payload = {
                    "query": query,
                    "knowledge-bases": [{"name": KB_NAME}],
                    "num-results": KB_LIMIT,
                }
                # Map filters if present (API uses 'filter')
                if KB_FILTERS:
                    kb_payload["filter"] = KB_FILTERS
                cmd = [
                    NU_CLI_BIN,
                    "ser",
                    "curl",
                    "--env",
                    NU_ENV,
                    "POST",
                    NU_SHARD,
                    NU_SERVICE,
                    "/api/v1/knowledge-bases/search",
                    "--data",
                    json.dumps(kb_payload),
                ]
                completed = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                data = json.loads(completed.stdout) if completed.stdout else {}
                results = data.get("results", []) if isinstance(data, dict) else []
                set_cached_results(query, results)
            else:
                for attempt in range(3):
                    resp = requests.post(url, headers=headers, json=payload, timeout=10)
                    if resp.status_code == 429 and attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    results = data.get("results", []) if isinstance(data, dict) else []
                    set_cached_results(query, results)
                    break

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
        
    except requests.RequestException as e:
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
