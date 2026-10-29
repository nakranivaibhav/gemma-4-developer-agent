"""Live probe: how long does a vLLM server keep a finished request's prompt cached?

usage: envs/harness/bin/python -I analysis/cache_probe.py <requests.jsonl.gz> <server_port> [--n 16] [--waits 0,5,15,30]
Sends captured prompts (1 output token) straight to the server (use the server port, NOT the capture proxy, so the
probe stays out of the training capture), waits W seconds, sends the same prompt again. A second send of ~1 s is a
cache hit; a second send as slow as the first is a miss. Several prompts per wait, because shared-GPU load makes
single timings noisy. Measured 2026-10-23 on the 31B pod (32 agents at once): hits 4/4 at 0 s, 1/4 at 5, 15 and 30 s.
"""
import argparse
import gzip
import json
import random
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("capture")
ap.add_argument("port", type=int)
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--waits", default="0,5,15,30")
ap.add_argument("--min-tokens", type=int, default=7000)
ap.add_argument("--max-tokens", type=int, default=14000)
a = ap.parse_args()
waits = [int(w) for w in a.waits.split(",")]

idx = []
for i, line in enumerate(gzip.open(a.capture, "rt")):
    try:
        r = json.loads(line)
    except Exception:  # the live run may be mid-write on the last line
        break
    pt = ((r.get("response") or {}).get("usage") or {}).get("prompt_tokens", 0)
    if r.get("status") == 200 and a.min_tokens < pt < a.max_tokens:
        idx.append(i)
random.seed(7)
pick = set(random.sample(idx, min(a.n, len(idx))))
reqs = [json.loads(line)["request"] for i, line in enumerate(gzip.open(a.capture, "rt")) if i in pick]


def send(q):
    q = dict(q, max_completion_tokens=1)
    q.pop("thinking_token_budget", None)
    t = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(q).encode(),
                                 {"Content-Type": "application/json"})
    u = json.loads(urllib.request.urlopen(req, timeout=300).read())["usage"]
    return time.time() - t, u["prompt_tokens"]


res = {}
for k, q in enumerate(reqs):
    w = waits[k % len(waits)]
    first = send(q)
    time.sleep(w)
    second = send(q)
    res.setdefault(w, []).append(round(second[0], 2))
    print(f"wait {w:2d}s: first {first[0]:5.2f}s  second {second[0]:5.2f}s  ({first[1]} tok)", flush=True)
for w in sorted(res):
    print("wait", w, "second-send seconds", sorted(res[w]))
