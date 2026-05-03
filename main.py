"""
zai_proxy.py  —  chat.z.ai → OpenAI-compatible Flask API Proxy
===============================================================
Auth logic  : from zai_client.py  (token cached to .zai_token, auto-reauth on 401)
Proxy logic : from zai-proxy.js   (request building, SSE parsing, streaming, tool-calls)

Dependencies:
    pip install flask requests

PythonAnywhere setup:
    1. Upload this file.
    2. Create a Web app → Manual configuration → Python 3.10+
    3. In the WSGI file add:
           import sys
           sys.path.insert(0, '/home/<you>/<folder>')
           from zai_proxy import app as application
    4. pip install --user flask requests
    5. Reload the web app.

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions

Extra POST body params (beyond standard OpenAI):
    web_search      bool  – enable web search          (default: true)
    enable_thinking bool  – enable reasoning/thinking  (default: true)
    tools           list  – OpenAI function-calling tools
    tool_choice     any   – OpenAI tool_choice value
"""

import json
import os
import re
import uuid
import time
import threading
import logging

import requests
from flask import Flask, request, Response, jsonify, stream_with_context

# ── Config ────────────────────────────────────────────────────────────────────

ZAI_BASE      = "https://chat.z.ai"
AUTH_URL      = f"{ZAI_BASE}/api/v1/auths/"
CHAT_URL      = f"{ZAI_BASE}/api/v2/chat/completions"
MODELS_URL    = f"{ZAI_BASE}/api/models"
DEFAULT_MODEL = "GLM-5-Turbo"
FE_VERSION    = "prod-fe-1.1.21"

# Fallback model list used when /api/models is unreachable.
# Sources: https://docs.z.ai (confirmed May 2026)
KNOWN_MODELS = [
    # ── Language models ───────────────────────────────────────────
    "GLM-5.1",           # latest flagship, best agentic/coding
    "GLM-5",             # previous flagship, strong coding
    "GLM-5-Turbo",       # fast/cheap daily-use variant
    "GLM-4.7",
    "GLM-4.6",
    "GLM-4.5",
    "GLM-4-32B-0414-128K",
    # ── Vision-language models ────────────────────────────────────
    "GLM-5V-Turbo",      # vision + text
    "GLM-4.6V",
    "GLM-4.5V",
    "GLM-OCR",
    "AutoGLM-Phone-Multilingual",
]

# Case-normalisation map: any casing the user sends → exact upstream ID.
# Entries are lowercased keys so lookup is always .lower().
MODEL_ALIASES = {m.lower(): m for m in KNOWN_MODELS}

_DIR       = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(_DIR, ".zai_token")

BASE_HEADERS = {
    "User-Agent"     : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "x-fe-version"   : FE_VERSION,
    "x-region"       : "overseas",
    "Content-Type"   : "application/json",
    "accept"         : "*/*",
    "accept-language": "en-US",
    "Origin"         : ZAI_BASE,
    "Referer"        : ZAI_BASE + "/",
}

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_token_lock = threading.Lock()

app = Flask(__name__)

# ── Token management (from zai_client.py) ────────────────────────────────────

def _load_token():
    if os.path.isfile(TOKEN_FILE):
        t = open(TOKEN_FILE).read().strip()
        return t or None
    return None


def _save_token(t: str):
    with open(TOKEN_FILE, "w") as f:
        f.write(t)


def _clear_token():
    if os.path.isfile(TOKEN_FILE):
        os.remove(TOKEN_FILE)


def _fetch_fresh_token(session: requests.Session) -> str:
    log.info("Fetching fresh guest token from %s", AUTH_URL)
    r = session.get(AUTH_URL, timeout=15)
    r.raise_for_status()
    d = r.json()
    t = d.get("token")
    if not t:
        raise RuntimeError(f"No token in auth response: {d}")
    _save_token(t)
    log.info("Token obtained and cached.")
    return t


# Module-level shared session — reuses connections across Flask requests.
_session = requests.Session()
_session.headers.update(BASE_HEADERS)

_token = _load_token()
if _token:
    _session.headers["Authorization"] = f"Bearer {_token}"


def ensure_auth():
    global _token
    with _token_lock:
        if not _token:
            _token = _fetch_fresh_token(_session)
            _session.headers["Authorization"] = f"Bearer {_token}"


def reauth():
    global _token
    with _token_lock:
        log.info("Re-authenticating (token likely expired)…")
        _clear_token()
        _token = None
        _token = _fetch_fresh_token(_session)
        _session.headers["Authorization"] = f"Bearer {_token}"


# ── Upstream request builder (mirrors generateSignedRequest from zai-proxy.js) ──

def build_upstream_payload(model, messages, opts: dict):
    """
    Returns (url, body_dict).

    The JS proxy uses window.__zaiYM / __zaiMM to compute an X-Signature from
    a CDN bundle. The Python client (zai_client.py) uses ?version=0.0.1&platform=web
    without that header and the server accepts it — we follow the same approach.
    """
    enable_thinking = opts.get("enable_thinking", True)
    web_search      = opts.get("web_search", True)
    tools           = opts.get("tools")
    tool_choice     = opts.get("tool_choice")

    msg_id  = str(uuid.uuid4())
    chat_id = str(uuid.uuid4())
    sess_id = str(uuid.uuid4())

    body = {
        "model"                         : model,
        "messages"                      : messages,
        "stream"                        : True,   # always stream from upstream
        "signature_prompt"              : (messages[-1].get("content", "") if messages else ""),
        "chat_id"                       : chat_id,
        "id"                            : msg_id,
        "session_id"                    : sess_id,
        "current_user_message_id"       : msg_id,
        "current_user_message_parent_id": None,
        "params"                        : {},
        "extra"                         : {},
        "features": {
            "image_generation": False,
            "web_search"      : web_search,
            "auto_web_search" : True,   # always on per user request
            "preview_mode"    : True,
            "flags"           : [],
            "enable_thinking" : enable_thinking,
        },
        "variables": {},
    }

    if tools:
        body["tools"] = tools
    if tool_choice:
        body["tool_choice"] = tool_choice

    url = CHAT_URL + "?version=0.0.1&platform=web"
    return url, body


# ── SSE parser (mirrors inner loop of handleChat from zai-proxy.js) ──────────

def _iter_upstream_sse(response: requests.Response):
    """Yields parsed data-dicts from the upstream SSE stream."""
    for raw in response.iter_lines():
        line = (raw.decode("utf-8") if isinstance(raw, bytes) else raw).strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except Exception:
            continue

        # Upstream wraps events as {"type":"chat:completion","data":{…}}
        d = parsed.get("data") if parsed.get("type") == "chat:completion" else parsed
        if not d or isinstance(d, str):
            continue
        yield d


# ── Response helpers (mirrors makeChunk / makeCompletion from zai-proxy.js) ──

def _make_chunk(chat_id, model, content=None, reasoning=None,
                finish_reason=None, tool_calls=None) -> dict:
    delta = {}
    if content    is not None: delta["content"]           = content
    if reasoning  is not None: delta["reasoning_content"] = reasoning
    if tool_calls is not None: delta["tool_calls"]        = tool_calls
    return {
        "id"     : chat_id,
        "object" : "chat.completion.chunk",
        "created": int(time.time()),
        "model"  : model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _make_completion(chat_id, model, content, reasoning, usage, tool_calls) -> dict:
    msg = {"role": "assistant", "content": content}
    if reasoning :  msg["reasoning_content"] = reasoning
    if tool_calls:  msg["tool_calls"]        = tool_calls
    return {
        "id"     : chat_id,
        "object" : "chat.completion",
        "created": int(time.time()),
        "model"  : model,
        "choices": [{
            "index"        : 0,
            "message"      : msg,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ── Small helpers ─────────────────────────────────────────────────────────────

def _extract_edit_content(text: str) -> str:
    """Strip <details> wrapper that sometimes wraps thinking inside edit_content."""
    if not text:
        return ""
    marker = "</details>\n"
    idx = text.find(marker)
    return text[idx + len(marker):] if idx != -1 else text


def _to_oai_tool_call(call: dict, fallback_index: int) -> dict:
    return {
        "index"   : call.get("index", fallback_index),
        "id"      : call.get("id") or "call_" + uuid.uuid4().hex[:24],
        "type"    : "function",
        "function": {
            "name"     : call.get("function", {}).get("name", ""),
            "arguments": call.get("function", {}).get("arguments", ""),
        },
    }


def _format_citations(d: dict) -> str:
    sources = d.get("results") or d.get("sources") or d.get("citations")
    if not sources or not isinstance(sources, list):
        return ""
    return "\n\n---\n" + "\n".join(
        f"[{i+1}] [{s.get('title') or s.get('name', '')}]"
        f"({s.get('url') or s.get('link', '')})"
        for i, s in enumerate(sources)
    )


# ── Core chat handler (mirrors handleChat from zai-proxy.js) ─────────────────

def _do_chat(body: dict):
    model           = body.get("model", DEFAULT_MODEL)
    model           = MODEL_ALIASES.get(model.lower(), model)  # normalise casing
    messages        = body.get("messages", [])
    want_stream     = body.get("stream", False)
    enable_thinking = body.get("enable_thinking", True)
    web_search      = body.get("web_search", True)
    tools           = body.get("tools") or None
    tool_choice     = body.get("tool_choice") or None
    chat_id         = "chatcmpl-" + uuid.uuid4().hex[:24]

    log.info("model=%s msgs=%d stream=%s search=%s tools=%s",
             model, len(messages), want_stream, web_search,
             len(tools) if tools else 0)

    ensure_auth()
    url, upstream_body = build_upstream_payload(
        model, messages,
        {"enable_thinking": enable_thinking, "web_search": web_search,
         "tools": tools, "tool_choice": tool_choice},
    )

    def call_upstream(retry=True):
        resp = _session.post(url, json=upstream_body, stream=True, timeout=90)
        if resp.status_code == 401 and retry:
            reauth()
            return call_upstream(retry=False)
        if not resp.ok:
            err = resp.text[:300]
            log.error("Upstream %s: %s", resp.status_code, err)
            return None, resp.status_code, err
        return resp, 200, None

    upstream, status, err_text = call_upstream()
    if upstream is None:
        return jsonify({"error": {"message": err_text, "type": "upstream_error",
                                  "code": status}}), status

    # ── Streaming response ────────────────────────────────────────────────────
    if want_stream:
        def generate():
            # Role preamble chunk (mirrors JS proxy)
            preamble = {
                "id": chat_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model,
                "choices": [{"index": 0,
                             "delta": {"role": "assistant", "content": ""},
                             "finish_reason": None}],
            }
            yield f"data: {json.dumps(preamble)}\n\n"

            tool_calls_accum = []

            for d in _iter_upstream_sse(upstream):
                if d.get("done"):
                    fr = "tool_calls" if tool_calls_accum else "stop"
                    yield f"data: {json.dumps(_make_chunk(chat_id, model, finish_reason=fr))}\n\n"
                    yield "data: [DONE]\n\n"
                    continue

                phase = d.get("phase", "")

                # Thinking tokens
                if phase == "thinking" and d.get("delta_content"):
                    yield f"data: {json.dumps(_make_chunk(chat_id, model, reasoning=d['delta_content']))}\n\n"

                # Answer tokens
                elif phase == "answer":
                    text = d.get("delta_content") or _extract_edit_content(d.get("edit_content", ""))
                    if text:
                        yield f"data: {json.dumps(_make_chunk(chat_id, model, content=text))}\n\n"

                # Other phase (edit_content / usage)
                elif phase == "other":
                    text = d.get("edit_content", "")
                    if text:
                        yield f"data: {json.dumps(_make_chunk(chat_id, model, content=text))}\n\n"

                # Tool calls
                if d.get("tool_calls"):
                    calls = d["tool_calls"] if isinstance(d["tool_calls"], list) else [d["tool_calls"]]
                    for call in calls:
                        oai = _to_oai_tool_call(call, len(tool_calls_accum))
                        tool_calls_accum.append(oai)
                        yield f"data: {json.dumps(_make_chunk(chat_id, model, tool_calls=[oai]))}\n\n"

                # Web-search citations appended as content
                if phase == "search" or d.get("type") == "web_search":
                    citation = _format_citations(d)
                    if citation:
                        yield f"data: {json.dumps(_make_chunk(chat_id, model, content=citation))}\n\n"

        return Response(
            stream_with_context(generate()),
            status=200,
            mimetype="text/event-stream",
            headers={
                "Cache-Control"           : "no-cache",
                "X-Accel-Buffering"       : "no",   # disable nginx buffering on PythonAnywhere
                "Access-Control-Allow-Origin": "*",
            },
        )

    # ── Non-streaming response ────────────────────────────────────────────────
    full_content    = ""
    full_reasoning  = ""
    usage           = None
    tool_calls_accum = []

    for d in _iter_upstream_sse(upstream):
        if d.get("done"):
            break

        phase = d.get("phase", "")

        if phase == "thinking" and d.get("delta_content"):
            full_reasoning += d["delta_content"]

        elif phase == "answer":
            text = d.get("delta_content") or _extract_edit_content(d.get("edit_content", ""))
            if text:
                full_content += text

        elif phase == "other":
            if d.get("usage"):
                u = d["usage"]
                usage = {
                    "prompt_tokens"    : u.get("prompt_tokens", 0),
                    "completion_tokens": u.get("completion_tokens", 0),
                    "total_tokens"     : u.get("total_tokens", 0),
                }
            if d.get("edit_content"):
                full_content += d["edit_content"]

        if d.get("tool_calls"):
            calls = d["tool_calls"] if isinstance(d["tool_calls"], list) else [d["tool_calls"]]
            for call in calls:
                tool_calls_accum.append(_to_oai_tool_call(call, len(tool_calls_accum)))

        if phase == "search" or d.get("type") == "web_search":
            full_content += _format_citations(d)

    # Clean up <details> wrapper tags that sometimes surround thinking content
    clean_reasoning = re.sub(r"</?details[^>]*>\n?", "", full_reasoning)
    clean_reasoning = re.sub(r"^> ", "", clean_reasoning, flags=re.MULTILINE).strip()

    log.info("Done: %d chars content, %d chars reasoning, %d tool_calls",
             len(full_content), len(full_reasoning), len(tool_calls_accum))

    return jsonify(_make_completion(
        chat_id, model, full_content,
        clean_reasoning or None,
        usage,
        tool_calls_accum or None,
    ))


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.after_request
def _add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/v1/models", methods=["GET"])
@app.route("/models",    methods=["GET"])
def route_models():
    ensure_auth()
    try:
        r = _session.get(MODELS_URL, timeout=15)
        r.raise_for_status()
        data = r.json()
        ids  = [m["id"] for m in (data if isinstance(data, list) else data.get("data", []))]
    except Exception as e:
        log.warning("Could not fetch model list (%s). Returning default.", e)
        ids = KNOWN_MODELS

    return jsonify({
        "object": "list",
        "data"  : [{"id": i, "object": "model", "created": 1_700_000_000,
                    "owned_by": "zhipu"} for i in ids],
    })


@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
@app.route("/chat/completions",    methods=["POST", "OPTIONS"])
def route_chat():
    if request.method == "OPTIONS":
        return "", 200
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": {"message": "Invalid JSON body"}}), 400
    return _do_chat(body)


@app.route("/", methods=["GET"])
def route_index():
    return jsonify({
        "service"  : "chat.z.ai → OpenAI-compatible proxy",
        "base_url" : "/v1",
        "endpoints": ["GET /v1/models", "POST /v1/chat/completions"],
        "extra_params": {
            "web_search"     : "bool (default true, auto_web_search always true)",
            "enable_thinking": "bool (default true)",
            "tools"          : "list – OpenAI function-calling",
            "tool_choice"    : "any",
        },
    })


# ── Entrypoint (local dev) ────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9876))
    print("=" * 55)
    print("  chat.z.ai  →  OpenAI API Proxy  (Flask)")
    print("=" * 55)
    print(f"  Base URL  : http://localhost:{port}/v1")
    print("  Endpoints :")
    print("    GET  /v1/models")
    print("    POST /v1/chat/completions")
    print("  Extra params:")
    print("    web_search: true       – enable web search")
    print("    tools: [...]           – OpenAI function calling")
    print("    enable_thinking: bool  – thinking/reasoning (default: true)")
    print("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=False)
