"""
flask_app.py
────────────
Flask-based transparent proxy for chat.z.ai.
Zero modification — raw SSE bytes from Z.ai are forwarded as-is to the client.
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

    def raw_stream(self, messages: list):
        """Yields raw bytes chunks exactly as received from Z.ai — no parsing."""
        token = self.get_token()

        chat_id      = str(uuid.uuid4())
        message_id   = str(uuid.uuid4())
        timestamp_ms = str(int(time.time() * 1000))
        request_id   = str(uuid.uuid4())

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
        signature   = hashlib.sha256(payload_str.encode('utf-8')).hexdigest()

        url_params = {
            "version": "0.0.1", "platform": "web",
            "timestamp": timestamp_ms, "requestId": request_id,
            "signature_timestamp": timestamp_ms
        }

        headers = self.session.headers.copy()
        headers["Authorization"] = f"Bearer {token}"
        headers["x-signature"]   = signature

        resp = self.session.post(
            CHAT_URL, data=payload_str, headers=headers,
            params=url_params, stream=True, timeout=60
        )

        # Retry once on 401
        if resp.status_code == 401:
            token = self.authenticate()
            headers["Authorization"] = f"Bearer {token}"
            resp = self.session.post(
                CHAT_URL, data=payload_str, headers=headers,
                params=url_params, stream=True, timeout=60
            )

        resp.raise_for_status()

        # Yield raw bytes exactly as they arrive — no decode, no parse, no touch
        for chunk in resp.iter_content(chunk_size=None):
            if chunk:
                yield chunk


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
    body     = request.get_json(silent=True) or {}
    messages = body.get("messages", [])

    def generate():
        try:
            for raw_chunk in zai_client.raw_stream(messages):
                yield raw_chunk
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control"    : "no-cache",
            "X-Accel-Buffering": "no",
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
