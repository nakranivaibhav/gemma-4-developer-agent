"""Rollout tag + TITO capture through the real ADK client and the real proxy, against a fake vLLM.
Runs the harness side in envs/harness (its own Python) as subprocesses."""
import gzip
import json
import socket
import subprocess
import sys
import textwrap
import time

import pytest

from training.rl.tests.conftest import ROOT

PY = ROOT / "envs/harness/bin/python"
pytestmark = pytest.mark.skipif(not PY.exists(), reason="envs/harness missing")

FAKE_VLLM = textwrap.dedent('''
    import json, sys, threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    seen = open(sys.argv[2], "a"); lock = threading.Lock()
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def do_GET(self):
            out = b'{"data": []}'
            self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                seen.write(json.dumps({"hdr": self.headers.get("X-Rollout-Id"), "tito": bool(body.get("return_token_ids")),
                                       "last": body["messages"][-1]["content"]}) + "\\n"); seen.flush()
            r = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                 "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                 "prompt_token_ids": None,
                 "choices": [{"index": 0, "finish_reason": "stop", "logprobs": None, "token_ids": None,
                              "message": {"role": "assistant", "content": "ok"}}]}
            if body.get("return_token_ids"):
                r["prompt_token_ids"] = [1, 2, 3]; r["choices"][0]["token_ids"] = [7, 106]
            if body.get("logprobs"):
                r["choices"][0]["logprobs"] = {"content": [{"token": "a", "logprob": -0.5}, {"token": "b", "logprob": -0.1}]}
            out = json.dumps(r).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
''')

CLIENT = textwrap.dedent('''
    import asyncio, sys
    sys.path.insert(0, sys.argv[2])
    import rollout_tag
    rollout_tag.install(); rollout_tag.install()          # idempotent
    from google.adk.models.lite_llm import LiteLLMClient
    base = f"http://127.0.0.1:{sys.argv[1]}/v1"
    client = LiteLLMClient()
    async def ask(text):
        await client.acompletion(model="openai/m", messages=[{"role": "user", "content": text}], tools=None,
                                 api_base=base, api_key="x")
    async def one_try(rid, n):
        rollout_tag.ROLLOUT_ID.set(rid)
        for k in range(n):
            await ask(f"{rid}#{k}")
            await asyncio.sleep(0.01)
        await asyncio.create_task(ask(f"{rid}#child"))                     # nested task copies the context
        await asyncio.to_thread(lambda: asyncio.run(ask(f"{rid}#thread")))  # thread too
    async def main():
        await ask("untagged")
        await asyncio.gather(*(one_try(f"run_p{i}/task{i % 2}", 3) for i in range(6)))
    asyncio.run(main())
''')


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_port(port, t=20):
    end = time.time() + t
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(port)


@pytest.fixture
def stack(tmp_path):
    up, px = free_port(), free_port()
    (tmp_path / "fake.py").write_text(FAKE_VLLM)
    env = {"PATH": "/usr/bin:/bin", "LOG_PROXY_TITO": "1", "LOG_PROXY_FULL": str(tmp_path / "full.jsonl.gz")}
    procs = [subprocess.Popen([str(PY), "-I", str(tmp_path / "fake.py"), str(up), str(tmp_path / "seen.jsonl")]),
             subprocess.Popen([str(PY), "-I", str(ROOT / "kits/swe-agent/experiments/log_proxy.py"),
                               str(tmp_path / "settings.jsonl"), str(px), str(up)], env=env)]
    wait_port(up)
    wait_port(px)
    yield tmp_path, px
    for p in procs:
        p.kill()
        p.wait()


def test_every_call_tagged_with_its_own_try_and_header_not_forwarded(stack):
    tmp, px = stack
    (tmp / "client.py").write_text(CLIENT)
    subprocess.run([str(PY), "-I", str(tmp / "client.py"), str(px), str(ROOT / "kits/swe-agent/run")],
                   check=True, timeout=120, capture_output=True)
    rows = [json.loads(line) for line in gzip.open(tmp / "full.jsonl.gz", "rt")]
    assert len(rows) == 1 + 6 * 5
    for r in rows:
        text = r["request"]["messages"][-1]["content"]
        if text == "untagged":
            assert "rollout_id" not in r
        else:
            assert r["rollout_id"] == text.split("#")[0]          # never another try's id
            assert r["tito"]["completion_ids"] == [7, 106] and r["tito"]["logprobs"] == [-0.5, -0.1]
        assert r["response"]["choices"][0]["token_ids"] is None    # the harness still gets the old reply shape
    seen = [json.loads(line) for line in open(tmp / "seen.jsonl")]
    assert all(s["hdr"] is None and s["tito"] for s in seen)    # vLLM never sees the tag; always the TITO fields


def test_capture_reads_back_into_tries(stack):
    from training.rl import rollouts
    tmp, px = stack
    (tmp / "client.py").write_text(CLIENT)
    subprocess.run([str(PY), "-I", str(tmp / "client.py"), str(px), str(ROOT / "kits/swe-agent/run")],
                   check=True, timeout=120, capture_output=True)
    cap, stats = rollouts.read_capture(tmp / "full.jsonl.gz")
    assert stats == {"calls": 31, "untagged": 1}
    assert sorted(cap) == sorted(f"run_p{i}/task{i % 2}" for i in range(6))
    assert all(len(v) == 5 for v in cap.values())
