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

    def stream_zai_deltas(self, messages: list):
        """
        Yields (delta_type, delta_text) tuples where delta_type is one of:
          "role"              – initial role announcement (content="", role="assistant")
          "content"           – tool-use / answer content delta
          "reasoning_content" – thinking/reasoning delta
        """
        token = self.get_token()

        chat_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        timestamp_ms = str(int(time.time() * 1000))
        request_id = str(uuid.uuid4())

        payload_dict = {
            "stream": True,
            "model": DEFAULT_MODEL,
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

        role_announced = False

        for raw_line in resp.iter_lines():
            if not raw_line: continue
            line = raw_line.decode('utf-8').strip()
            if not line.startswith("data:"): continue

            json_str = line[5:].strip()
            if json_str == "[DONE]": break

            try:
                data_obj = json.loads(json_str)
                if data_obj.get("type") != "chat:completion":
                    continue

                inner = data_obj.get("data", {})
                phase = inner.get("phase", "")
                delta = inner.get("delta_content", "")

                if "error" in inner:
                    yield ("content", f"\n[API Error: {inner['error'].get('detail', 'Unknown')}]")
                    continue

                # Announce role once at the very start (mirrors OpenAI behaviour)
                if not role_announced:
                    yield ("role", "")
                    role_announced = True

                if phase == "thinking":
                    # Reasoning / chain-of-thought → reasoning_content delta
                    if delta:
                        yield ("reasoning_content", delta)

                elif phase in ("mcp", "tool", "browser"):
                    # Tool-use / web-search result blob → content delta
                    if delta:
                        yield ("content", delta)

                elif phase == "answer":
                    # Final answer → content delta
                    if delta:
                        yield ("content", delta)

                else:
                    # Unknown phase: if there is a delta, forward it as content
                    if delta:
                        yield ("content", delta)

            except json.JSONDecodeError:
                continue

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
    # Always stream — Z.ai is a streaming-only backend.
    # Clients that send stream=false will still receive SSE; wrap with a
    # buffering proxy if you need a single-shot JSON response downstream.

    def generate():
        chunk_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())
        try:
            for delta_type, delta_text in zai_client.stream_zai_deltas(messages):
                if delta_type == "role":
                    delta_obj = {"role": "assistant", "content": ""}
                elif delta_type == "reasoning_content":
                    delta_obj = {"reasoning_content": delta_text}
                else:  # "content"
                    delta_obj = {"content": delta_text}

                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": DEFAULT_MODEL,
                    "choices": [{"index": 0, "delta": delta_obj, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            stop_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": DEFAULT_MODEL,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            err_chunk = {"error": str(e)}
            yield f"data: {json.dumps(err_chunk)}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx buffering on PythonAnywhere
        }
    )

@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{"id": DEFAULT_MODEL, "object": "model", "created": int(time.time()), "owned_by": "z-ai"}]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
