"""Does vLLM load our LoRA on the int4 base and actually use it?

usage: envs/harness/bin/python -I training/adapter_check.py <agent_dir> <model_dir> <requests.jsonl.gz> [n=6]

Starts the server the way the runs do (kits/swe-agent/serve/serve_model.py, plain decoding, adapters
discovered from <agent_dir>/adapters), then replays n captured agent requests at temperature 0, once on the
base model and once on each adapter, and compares the outputs. Two checks:
  load:   server comes up; its log has no "unexpected"/"not found"/"missing" lines about LoRA modules
  effect: at temperature 0 the adapter output differs from the base output on most requests
          (identical everywhere = the adapter is not attached, e.g. module names did not match)
Prints one CHECK line per adapter and exits 1 if any check fails.
"""
import gzip
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
agent_dir, model_dir, req_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
n = int(sys.argv[4]) if len(sys.argv) > 4 else 6
log_path = ROOT / "runs/logs/adapter_check_server.log"
URL = "http://127.0.0.1:8000"


def post(path, body, timeout=600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def up():
    try:
        with urllib.request.urlopen(URL + "/v1/models", timeout=2) as r:
            return [m["id"] for m in json.load(r)["data"]]
    except Exception:
        return None


# pick n requests spread over the capture (later ones have longer contexts)
rows = [json.loads(l) for l in gzip.open(req_path, "rt")]
rows = [r for r in rows if r.get("status") == 200 or r.get("response")]
picks = [rows[i * (len(rows) - 1) // max(n - 1, 1)]["request"] for i in range(n)]

env = dict(os.environ, SPEC="", MAX_NUM_SEQS="4")
with open(log_path, "w") as log:
    srv = subprocess.Popen([str(ROOT / "envs/harness/bin/python"), "-I", str(ROOT / "kits/swe-agent/serve/serve_model.py"),
                            str(model_dir), str(agent_dir)], stdout=log, stderr=subprocess.STDOUT, env=env,
                           start_new_session=True)
try:
    for _ in range(240):
        if (models := up()) or srv.poll() is not None:
            break
        time.sleep(5)
    if not models:
        print("CHECK load FAIL: server did not come up, see", log_path)
        sys.exit(1)
    base = str(model_dir.resolve())
    adapters = [m for m in models if m != base]
    print("models served:", models)
    text = log_path.read_text(errors="replace")
    warn = [l for l in text.splitlines() if re.search(r"lora", l, re.I) and re.search(r"unexpected|not found|missing|ignor|unsupported", l, re.I)]
    for l in warn[:5]:
        print("  LOG", l[:200])

    def gen(model, q):
        body = dict(q, model=model, temperature=0.0, top_p=1.0, max_completion_tokens=256)
        body.pop("thinking_token_budget", None)
        out = post("/v1/chat/completions", body)["choices"][0]["message"]
        return json.dumps({k: out.get(k) for k in ("content", "reasoning_content", "reasoning", "tool_calls")}, sort_keys=True)

    base_out = [gen(base, q) for q in picks]
    ok_all = bool(adapters) and not warn
    for a in adapters:
        outs = [gen(a, q) for q in picks]
        diff = sum(x != y for x, y in zip(base_out, outs))
        ok = diff >= max(1, n // 2) and not warn
        ok_all &= ok
        print(f"CHECK adapter={a} load={'ok' if not warn else 'WARN'} effect: {diff}/{n} outputs differ from base -> {'PASS' if ok else 'FAIL'}")
        print("  base   :", base_out[0][:240])
        print("  adapter:", outs[0][:240])
    if not adapters:
        print("CHECK FAIL: no adapter served")
finally:
    os.killpg(srv.pid, 2)
    try:
        srv.wait(120)
    except subprocess.TimeoutExpired:
        os.killpg(srv.pid, 9)
sys.exit(0 if ok_all else 1)
