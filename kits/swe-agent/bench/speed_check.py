"""Time a short generation (base + one adapter) and a long-prompt prefill on the local vLLM server."""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000/v1"


def post(body):
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    return json.load(urllib.request.urlopen(req, timeout=600)), time.time() - t


ids = [m["id"] for m in json.load(urllib.request.urlopen(BASE + "/models"))["data"]]
print("served models:", [i.split("/")[-1] for i in ids])
q = "Write a Python function that returns the n-th Fibonacci number iteratively, then explain it in three sentences."
for name in (sys.argv[1], "main_lora"):
    d, dt = post({"model": name, "messages": [{"role": "user", "content": q}], "max_tokens": 300, "temperature": 0.2})
    u, m = d["usage"], d["choices"][0]["message"]
    print(f"{name.split('/')[-1]} | prompt {u['prompt_tokens']} completion {u['completion_tokens']} | {dt:.1f}s -> {u['completion_tokens'] / dt:.1f} tok/s")
    print("   text:", repr((m.get("content") or "")[:140]), "| reasoning:", repr((m.get("reasoning_content") or m.get("reasoning") or "")[:60]))
filler = "def f(x):\n    return x + 1\n" * 2500
d, dt = post({"model": sys.argv[1], "messages": [{"role": "user", "content": filler + "\nHow many functions are defined above? Answer with a number."}], "max_tokens": 20, "temperature": 0})
u = d["usage"]
print(f"long prompt | prompt {u['prompt_tokens']} completion {u['completion_tokens']} | {dt:.1f}s")
