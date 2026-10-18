"""Does a conversation whose saved history was pushed out of the GPU cache still get the same answers?

usage: python -I kits/swe-agent/bench/spill_check.py <served_model> <label>
One conversation X (about 9k tokens of code with 40 planted "secret" functions) is asked the same open question
three times at temperature 0 with nothing else running: cold, warm, and after a flood of 7 other long
conversations (about 77k tokens, more than the GPU cache holds) has pushed X out of the GPU cache. Then six planted-number questions are asked after another flood.
Run it against the server with and without the system-RAM spill and compare.
"""
import json
import random
import re
import sys
import threading
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_multi import BASE, code_text  # noqa: E402

MODEL, LABEL = sys.argv[1], sys.argv[2]


def ask(msgs, max_tokens):
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "temperature": 0, "seed": 1,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    return (d["choices"][0]["message"].get("content") or ""), time.time() - start, d["usage"]["prompt_tokens"]


def metric_lines(pattern):
    text = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(vllm:[a-z_]+)\{[^}]*\} ([0-9.e+]+)$", line)
        if m and re.search(pattern, m.group(1)) and not m.group(1).endswith(("_bucket", "_created")):
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def flood(tag):
    def one(i):
        rng = random.Random(f"flood-{tag}-{i}-{time.time()}")
        ask([{"role": "user", "content": f"Flood {tag} {i}.\n" + code_text(rng, 30000) + "\nSay ok."}], 20)
    threads = [threading.Thread(target=one, args=(i,)) for i in range(7)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


rng = random.Random(4242)
secrets, parts = {}, []
for k in range(40):
    name, value = f"secret_{rng.randrange(10**6)}", rng.randrange(10000, 99999)
    secrets[name] = value
    parts.append(code_text(rng, 560) + f"def {name}():\n    return {value}\n")
doc = "".join(parts)
names = list(secrets)
base = [{"role": "system", "content": "You answer questions about the code you are given, briefly and exactly."},
        {"role": "user", "content": "Here is a module.\n" + doc + "\nReply READY."},
        {"role": "assistant", "content": "READY"}]
open_q = base + [{"role": "user", "content": "List the first eight functions whose names start with secret_, each with the number it returns, one per line."}]

before = metric_lines("prefix_cache|preempt|kv|offload|connector")
cold, t_cold, n_prompt = ask(open_q, 160)
warm, t_warm, _ = ask(open_q, 160)
flood("a")
after, t_after, _ = ask(open_q, 160)
print(f"{LABEL} prompt_tokens={n_prompt} seconds cold={t_cold:.2f} warm={t_warm:.2f} after_flood={t_after:.2f}")
print(f"{LABEL} same_text warm_vs_cold={warm == cold} after_flood_vs_cold={after == cold}")
truth8 = sum(str(secrets[n]) in cold for n in names[:8])
print(f"{LABEL} open answer lists {truth8}/8 correct numbers (cold), {sum(str(secrets[n]) in after for n in names[:8])}/8 (after flood)")
flood("c")
right = 0
msgs = list(base)
for name in (names[3], names[11], names[19], names[27], names[33], names[39]):
    msgs = msgs + [{"role": "user", "content": f"What number does {name} return? Reply with the number only."}]
    text, took, _ = ask(msgs, 12)
    ok = str(secrets[name]) in text
    right += ok
    msgs = msgs + [{"role": "assistant", "content": text or "?"}]
    flood(name)
print(f"{LABEL} planted-number questions right after floods: {right}/6")
delta = {k.split(":")[1]: round(v - before.get(k, 0.0)) for k, v in metric_lines("prefix_cache|preempt|kv|offload|connector").items()}
print(LABEL, "metrics", json.dumps(delta)[:600])
