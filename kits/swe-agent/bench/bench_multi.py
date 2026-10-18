"""Throughput of the local server under N simulated agents running at once.

usage: python -I kits/swe-agent/bench/bench_multi.py <served_model> <out.json> <label> [N ...]
Each simulated agent is a 12-turn conversation shaped like our recorded runs: the prompt starts near 4k tokens
and grows about 800 tokens a turn (reply + tool result), every reply is forced to exactly REPLY tokens, and the
agent pauses GAP seconds between turns as if a tool were running. Sizes come from runs/logs/requests_census.jsonl.
"""
import json
import random
import re
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
TURNS, REPLY, GAP = 12, 230, 1.5
WORDS = ("request session cookie header adapter response stream timeout retry proxy auth render console style "
         "segment table column route param schema field model depends router scope buffer encode decode merge "
         "resolve prepare redirect history status payload chunk border padding align theme layout").split()


def code_text(rng, chars):
    out = []
    size = 0
    while size < chars:
        a, b, c, d = (rng.choice(WORDS) for _ in range(4))
        line = f"def {a}_{b}_{rng.randrange(9999)}(self, {c}):\n    return self.{d}.{a}({c}, {rng.randrange(999)})\n"
        out.append(line)
        size += len(line)
    return "".join(out)


def metrics():
    text = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    want = ("vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:prefix_cache_queries_total",
            "vllm:prefix_cache_hits_total", "vllm:num_preemptions_total", "vllm:external_prefix_cache_queries_total",
            "vllm:external_prefix_cache_hits_total", "vllm:request_prefill_kv_computed_tokens_sum")
    got = dict.fromkeys(want, 0.0)
    for line in text.splitlines():
        m = re.match(r"^(vllm:[a-z_]+)\{[^}]*\} ([0-9.e+]+)$", line)
        if m and m.group(1) in got:
            got[m.group(1)] += float(m.group(2))
    return got


def agent(model, seed, shared, rows, errors, recalls):
    rng = random.Random(seed)
    secret = rng.randrange(100000, 999999)  # asked back after the last turn: is the saved history still intact?
    msgs = [{"role": "system", "content": shared},
            {"role": "user", "content": f"Task {seed}. The ticket number is {secret}. Study this module.\n" + code_text(rng, 3400)}]
    for turn in range(TURNS):
        body = {"model": model, "messages": msgs, "max_tokens": REPLY, "min_tokens": REPLY, "ignore_eos": True,
                "temperature": 0.2, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        start = time.time()
        try:
            d = json.load(urllib.request.urlopen(req, timeout=1800))
        except Exception as exc:
            errors.append(repr(exc)[:200])
            return
        took = time.time() - start
        u = d["usage"]
        rows.append({"turn": turn, "s": took, "prompt": u["prompt_tokens"], "completion": u["completion_tokens"]})
        reply = d["choices"][0]["message"].get("content") or "ok"
        msgs.append({"role": "assistant", "content": reply})
        msgs.append({"role": "user", "content": f"Tool result {turn}:\n" + code_text(rng, 1750)})
        time.sleep(GAP)
    msgs[-1] = {"role": "user", "content": "What is the ticket number from the start of this task? Reply with the number only."}
    body = {"model": model, "messages": msgs, "max_tokens": 12, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        text = json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]["message"].get("content") or ""
        recalls.append(str(secret) in text)
    except Exception as exc:
        errors.append(repr(exc)[:200])


def level(model, n, nonce):
    shared = "You are a careful software engineer. Shared reference code follows.\n" + code_text(random.Random(7), 5000)
    rows, errors, recalls = [], [], []
    before = metrics()
    start = time.time()
    threads = [threading.Thread(target=agent, args=(model, f"{nonce}-{n}-{i}", shared, rows, errors, recalls)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - start
    after = metrics()
    delta = {k.split(":")[1]: after[k] - before[k] for k in after}
    lat = sorted(r["s"] for r in rows) or [0.0]
    prompts = sorted(r["prompt"] for r in rows) or [0]
    return {
        "agents": n, "wall_s": round(wall, 1), "turns": len(rows), "errors": errors[:3],
        "turns_per_min": round(60 * len(rows) / wall, 2),
        "reply_tok_per_s": round(sum(r["completion"] for r in rows) / wall, 1),
        "latency_median_s": round(lat[len(lat) // 2], 2), "latency_p90_s": round(lat[int(0.9 * (len(lat) - 1))], 2),
        "prompt_median": prompts[len(prompts) // 2], "prompt_max": prompts[-1],
        "cache_hit_share": round(delta["prefix_cache_hits_total"] / max(1.0, delta["prefix_cache_queries_total"]), 3),
        "preemptions": int(delta["num_preemptions_total"]),
        "ram_hit_share": round(delta["external_prefix_cache_hits_total"] / max(1.0, delta["prefix_cache_queries_total"]), 3),
        "prompt_tokens_recomputed_share": round(delta["request_prefill_kv_computed_tokens_sum"] / max(1.0, delta["prompt_tokens_total"]), 3),
        "recall_ok": f"{sum(recalls)}/{len(recalls)}",
    }


if __name__ == "__main__":
    model, out_path, label = sys.argv[1], sys.argv[2], sys.argv[3]
    levels = [int(x) for x in sys.argv[4:]] or [1, 2, 4]
    nonce = int(time.time())
    results = []
    for n in levels:
        res = level(model, n, nonce)
        results.append(res)
        print(label, json.dumps(res), flush=True)
        json.dump({"label": label, "turns": TURNS, "reply_tokens": REPLY, "gap_s": GAP, "levels": results},
                  open(out_path, "w"), indent=1)
