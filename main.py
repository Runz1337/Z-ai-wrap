"""
flask_app.py
────────────
Flask-based OpenAI-compatible API wrapper for chat.z.ai.
Perfect for PythonAnywhere's native WSGI environment.
"""

import uuid
import time
import hashlib
import json
import os
import sys
import requests
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

# ── Config ──────────────────────────────────────────────────────────────────
TOKEN_FILE    = os.path.join(os.path.dirname(__file__), ".zai_token")
AUTH_URL      = "https://chat.z.ai/api/v1/auths/"
CHAT_URL      = "https://chat.z.ai/api/v2/chat/completions"
DEFAULT_MODEL = "GLM-5-Turbo"

KNOWN_MODELS = [
    "GLM-5.1", "GLM-5", "GLM-5-Turbo",
    "GLM-4.7", "GLM-4.6", "GLM-4.5", "GLM-4-32B-0414-128K",
    "GLM-5V-Turbo", "GLM-4.6V", "GLM-4.5V", "GLM-OCR",
    "AutoGLM-Phone-Multilingual",
]
MODEL_ALIASES = {m.lower(): m for m in KNOWN_MODELS}

BASE_HEADERS = {
    "User-Agent"   : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "x-fe-version" : "prod-fe-1.1.21",
    "x-region"     : "overseas",
    "Content-Type" : "application/json",
}

# ── Z.ai Client ─────────────────────────────────────────────────────────────

class SyncZAIClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(BASE_HEADERS)
        self.token = self._load_token()

    def _load_token(self):
        if os.path.isfile(TOKEN_FILE):
            with open(TOKEN_FILE, "r") as f:
                token = f.read().strip()
                return token if token else None
        return None

    def _save_token(self, token: str):
        with open(TOKEN_FILE, "w") as f:
            f.write(token)

    def authenticate(self) -> str:
        print("[ZAI] Fetching new authentication token...", file=sys.stderr)
        resp = self.session.get(AUTH_URL, timeout=15)
        resp.raise_for_status()
        token = resp.json().get("token")
        if not token:
            raise RuntimeError("Failed to get token from Z.ai")
        self._save_token(token)
        self.token = token
        return token

    def get_token(self) -> str:
        if not self.token:
            return self.authenticate()
        return self.token

    def extract_last_prompt(self, messages: list) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str): return content
                elif isinstance(content, list):
                    return " ".join([p.get("text", "") for p in content if p.get("type") == "text"])
        return ""

    def stream_zai_deltas(self, messages: list, model: str = DEFAULT_MODEL):
        token = self.get_token()
        
        chat_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        timestamp_ms = str(int(time.time() * 1000))
        request_id = str(uuid.uuid4())

        payload_dict = {
            "stream": True,
            "model": model,
            "messages": messages,
            "signature_prompt": self.extract_last_prompt(messages),
            "params": {}, "extra": {},
            "features": {
                "image_generation": False, "web_search": False, "auto_web_search": True,
                "preview_mode": True, "flags": [], "vlm_tools_enable": False,
                "vlm_web_search_enable": False, "vlm_website_mode": False, "enable_thinking": True
            },
            "variables": {
                "{{USER_NAME}}": "Z.ai Proxy Client",
                "{{CURRENT_TIMEZONE}}": "Asia/Calcutta",
                "{{USER_LANGUAGE}}": "en-US"
            },
            "chat_id": chat_id, "id": message_id,
            "background_tasks": {"title_generation": False, "tags_generation": False}
        }

        payload_str = json.dumps(payload_dict, separators=(',', ':'))
        signature = hashlib.sha256(payload_str.encode('utf-8')).hexdigest()

        url_params = {
            "version": "0.0.1", "platform": "web",
            "timestamp": timestamp_ms, "requestId": request_id, "signature_timestamp": timestamp_ms
        }

        headers = self.session.headers.copy()
        headers["Authorization"] = f"Bearer {token}"
        headers["x-signature"] = signature

        resp = self.session.post(CHAT_URL, data=payload_str, headers=headers, params=url_params, stream=True, timeout=60)

        # Retry once on 401
        if resp.status_code == 401:
            token = self.authenticate()
            headers["Authorization"] = f"Bearer {token}"
            resp = self.session.post(CHAT_URL, data=payload_str, headers=headers, params=url_params, stream=True, timeout=60)

        resp.raise_for_status()

        # ── SSE parser — mirrors handleChat() from zai-proxy.js exactly ──────
        # Yields dicts: {content}, {reasoning_content}, {tool_call}, {usage}, {done}
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue

            json_str = line[5:].strip()
            if not json_str or json_str == "[DONE]":
                break

            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            # Upstream wraps as {"type":"chat:completion","data":{...}}
            d = parsed.get("data") if parsed.get("type") == "chat:completion" else parsed
            if not d or isinstance(d, str):
                continue

            # d.done -> stream finished
            if d.get("done"):
                yield {"done": True}
                continue

            # phase === 'thinking'  (reasoning tokens)
            if d.get("phase") == "thinking" and d.get("delta_content"):
                yield {"reasoning_content": d["delta_content"]}

            # phase === 'answer'
            elif d.get("phase") == "answer":
                if d.get("delta_content"):
                    yield {"content": d["delta_content"]}
                elif d.get("edit_content"):
                    # strip <details>...</details> wrapper (JS: const marker = '</details>\n')
                    text = d["edit_content"]
                    marker = "</details>\n"
                    idx = text.find(marker)
                    if idx != -1:
                        text = text[idx + len(marker):]
                    if text:
                        yield {"content": text}

            # phase === 'other'  (usage + trailing edit_content)
            elif d.get("phase") == "other":
                if d.get("usage"):
                    yield {"usage": {
                        "prompt_tokens"    : d["usage"].get("prompt_tokens", 0),
                        "completion_tokens": d["usage"].get("completion_tokens", 0),
                        "total_tokens"     : d["usage"].get("total_tokens", 0),
                    }}
                if d.get("edit_content"):
                    yield {"content": d["edit_content"]}

            # tool_calls
            if d.get("tool_calls"):
                tc = d["tool_calls"] if isinstance(d["tool_calls"], list) else [d["tool_calls"]]
                for call in tc:
                    yield {"tool_call": {
                        "id"      : call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "type"    : "function",
                        "function": {
                            "name"     : (call.get("function") or {}).get("name", ""),
                            "arguments": (call.get("function") or {}).get("arguments", ""),
                        },
                    }}

            # web search citations
            if d.get("phase") == "search" or d.get("type") == "web_search":
                sources = d.get("results") or d.get("sources") or d.get("citations")
                if isinstance(sources, list) and sources:
                    citation = "\n\n---\n" + "\n".join(
                        f"[{i+1}] [{s.get('title') or s.get('name','')}]"
                        f"({s.get('url') or s.get('link','')})"
                        for i, s in enumerate(sources)
                    )
                    yield {"content": citation}

            # server-side error frame
            if isinstance(d.get("error"), dict):
                yield {"content": f"\n[API Error: {d['error'].get('detail', 'Unknown')}]"}

# ── Flask App ───────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)
zai_client = SyncZAIClient()

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "message": "Z.ai Flask Wrapper running on PythonAnywhere!"})

@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    body = request.get_json(silent=True) or {}
    messages = body.get("messages", [])
    stream_to_client = body.get("stream", False)

    model   = body.get("model", DEFAULT_MODEL)
    model   = MODEL_ALIASES.get(model.lower(), model)
    chat_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())

    if stream_to_client:
        def generate():
            tool_calls_accum = []
            try:
                # Role preamble chunk (mirrors JS proxy)
                preamble = {
                    "id": chat_id, "object": "chat.completion.chunk", "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(preamble)}\n\n"

                for event in zai_client.stream_zai_deltas(messages, model):
                    if event.get("done"):
                        fr = "tool_calls" if tool_calls_accum else "stop"
                        fin = {"id": chat_id, "object": "chat.completion.chunk", "created": created,
                               "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": fr}]}
                        yield f"data: {json.dumps(fin)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    delta = {}
                    if "content" in event:
                        delta["content"] = event["content"]
                    if "reasoning_content" in event:
                        delta["reasoning_content"] = event["reasoning_content"]
                    if "tool_call" in event:
                        tc = event["tool_call"]
                        tc["index"] = len(tool_calls_accum)
                        tool_calls_accum.append(tc)
                        delta["tool_calls"] = [tc]

                    if delta:
                        chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": created,
                                 "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk)}\n\n"

                # fallback finish if upstream closed without d.done
                fr = "tool_calls" if tool_calls_accum else "stop"
                fin = {"id": chat_id, "object": "chat.completion.chunk", "created": created,
                       "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": fr}]}
                yield f"data: {json.dumps(fin)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    else:
        try:
            full_content   = ""
            full_reasoning = ""
            usage          = None
            tool_calls     = []

            for event in zai_client.stream_zai_deltas(messages, model):
                if event.get("done"):
                    break
                if "content" in event:
                    full_content += event["content"]
                if "reasoning_content" in event:
                    full_reasoning += event["reasoning_content"]
                if "usage" in event:
                    usage = event["usage"]
                if "tool_call" in event:
                    tc = event["tool_call"]
                    tc["index"] = len(tool_calls)
                    tool_calls.append(tc)

            # clean up <details> tags from reasoning (mirrors JS cleanReasoning)
            import re
            clean_reasoning = re.sub(r"</?details[^>]*>\n?", "", full_reasoning)
            clean_reasoning = re.sub(r"^> ", "", clean_reasoning, flags=re.MULTILINE).strip()

            msg = {"role": "assistant", "content": full_content}
            if clean_reasoning:
                msg["reasoning_content"] = clean_reasoning
            if tool_calls:
                msg["tool_calls"] = tool_calls

            return jsonify({
                "id"     : chat_id,
                "object" : "chat.completion",
                "created": created,
                "model"  : model,
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": "tool_calls" if tool_calls else "stop"}],
                "usage"  : usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            })
        except Exception as e:
            return jsonify({"error": {"message": str(e)}}), 500

@app.route("/v1/models", methods=["GET"])
@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{"id": m, "object": "model", "created": 1700000000, "owned_by": "z-ai"}
                 for m in KNOWN_MODELS]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
