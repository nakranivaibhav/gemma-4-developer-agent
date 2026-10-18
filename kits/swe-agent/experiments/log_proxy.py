"""Pass-through proxy that records the settings of every request the harness sends to the model server.

usage: python -I kits/swe-agent/experiments/log_proxy.py <out.jsonl> [listen_port=8001] [upstream_port=8000]
Only request settings and response sizes are written to out.jsonl, never message text.
env LOG_PROXY_FULL=<file.jsonl.gz>: also keep every chat request and response in full (messages, tools, settings,
the returned message and usage), one gzip member per call, for training data. Read with gzip.open(...).
"""
import gzip
import json
import os
import sys
import time
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

out_path = sys.argv[1]
listen_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8001
upstream = f"http://127.0.0.1:{int(sys.argv[3]) if len(sys.argv) > 3 else 8000}"
lock = threading.Lock()
SKIP = {"messages", "tools"}
FULL = os.environ.get("LOG_PROXY_FULL")


def record(row):
    with lock, open(out_path, "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _forward(self, body):
        req = urllib.request.Request(upstream + self.path, data=body, method=self.command)
        for key, value in self.headers.items():
            if key.lower() not in ("host", "content-length", "accept-encoding", "connection"):
                req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                return resp.status, resp.headers.get("Content-Type", "application/json"), resp.read()
        except urllib.error.HTTPError as err:
            return err.code, err.headers.get("Content-Type", "application/json"), err.read()

    def _reply(self, status, ctype, payload):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._reply(*self._forward(None))

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        row = {"path": self.path}
        try:
            req = json.loads(body)
            row.update({k: v for k, v in req.items() if k not in SKIP})
            msgs = req.get("messages") or []
            row["n_messages"] = len(msgs)
            row["n_tools"] = len(req.get("tools") or [])
            row["roles"] = "".join((m.get("role") or "?")[0] for m in msgs)[-40:]
            row["msgs_with_reasoning"] = sum(1 for m in msgs if m.get("reasoning") or m.get("reasoning_content"))
            first = msgs[0].get("content") if msgs else ""
            row["first_msg_head"] = (first if isinstance(first, str) else json.dumps(first))[:60]
        except Exception as exc:  # non-JSON body: still forward it
            row["parse_error"] = repr(exc)
        status, ctype, payload = self._forward(body)
        row["status"] = status
        try:
            resp = json.loads(payload)
            choice = (resp.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            row["usage"] = resp.get("usage")
            row["finish_reason"] = choice.get("finish_reason")
            row["reasoning_chars"] = len(msg.get("reasoning") or msg.get("reasoning_content") or "")
            row["content_chars"] = len(msg.get("content") or "")
            row["n_tool_calls"] = len(msg.get("tool_calls") or [])
        except Exception:
            row["resp_head"] = payload[:120].decode("utf-8", "replace")
        record(row)
        if FULL and self.path.endswith("/chat/completions"):
            try:
                full = {"utc": time.time(), "status": status, "request": json.loads(body), "response": json.loads(payload)}
            except Exception as exc:
                full = {"utc": time.time(), "status": status, "error": repr(exc)}
            with lock, gzip.open(FULL, "at") as fh:
                fh.write(json.dumps(full) + "\n")
        self._reply(status, ctype, payload)


ThreadingHTTPServer(("127.0.0.1", listen_port), Handler).serve_forever()
