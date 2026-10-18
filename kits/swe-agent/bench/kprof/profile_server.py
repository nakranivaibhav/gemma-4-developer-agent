"""Time and profile decode steps against the LIVE local server (same flags, adapters and request path as the agent).

usage: python -I kits/swe-agent/bench/kprof/profile_server.py <out_dir> <num_seqs> <prompt_tokens> <model_name> [ngram]
The server must already run on localhost:8000 with VLLM_TORCH_PROFILER_DIR set, so /start_profile writes a trace.
Writes <out_dir>/timing.json; the trace lands in the server's profiler dir (run_all.sh points it at <out_dir>/trace).
Prompts are fixed random token ids so prefill is exact and the second call on the same ids hits the prefix cache.
"""
import json
import os
import random
import sys
import threading
import time
import urllib.request

out_dir, num_seqs, prompt_tokens, model_name = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ngram = len(sys.argv) > 5 and sys.argv[5] == "ngram"
os.makedirs(out_dir, exist_ok=True)
URL = "http://127.0.0.1:8000"
VOCAB = 262144
STEPS = 64


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read() or b"{}")


def prompt_ids(seed, i):
    # avoid the lowest ids (special tokens) so a random prompt never contains <bos>/<eos>
    r = random.Random(seed * 100 + i)
    return [r.randrange(1000, VOCAB - 1000) for _ in range(prompt_tokens)]


def run(seed, max_tokens):
    """All sequences at once, as the agents are; returns wall seconds and tokens generated."""
    done, lat = [None] * num_seqs, [None] * num_seqs

    def one(i):
        t0 = time.perf_counter()
        out = post("/v1/completions", {"model": model_name, "prompt": prompt_ids(seed, i), "max_tokens": max_tokens,
                                       "temperature": 0, "ignore_eos": True})
        done[i] = out["usage"]["completion_tokens"]
        lat[i] = time.perf_counter() - t0

    ts = [threading.Thread(target=one, args=(i,)) for i in range(num_seqs)]
    t = time.perf_counter()
    for th in ts:
        th.start()
    for th in ts:
        th.join()
    run.last_latencies = lat
    return time.perf_counter() - t, sum(done)


def counters():
    want = ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:num_preemptions_total")
    out = {}
    with urllib.request.urlopen(URL + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            key = line.split("{")[0].split(" ")[0]
            if key in want:
                out[key] = out.get(key, 0) + float(line.rsplit(" ", 1)[1])
    return out


timing = {"num_seqs": num_seqs, "prompt_tokens": prompt_tokens, "model": model_name, "ngram": ngram}
run(1, 4)                                                       # warm-up
pre = [run(10 + i, 1)[0] for i in range(3)]                     # fresh prompts each time: prefill, no cache help
timing["prefill_s_per_prompt_set"] = sorted(pre)[1]
run(10, 1)                                                      # prime the cache for seed 10
dec, c0 = [], counters()
for _ in range(3):
    dt, gen = run(10, STEPS + 1)
    dec.append(dt / STEPS)
c1 = counters()
timing["decode_phase_prefix_hit"] = round((c1["vllm:prefix_cache_hits_total"] - c0["vllm:prefix_cache_hits_total"])
                                          / max(c1["vllm:prefix_cache_queries_total"] - c0["vllm:prefix_cache_queries_total"], 1), 3)
timing["decode_phase_preemptions"] = c1["vllm:num_preemptions_total"] - c0["vllm:num_preemptions_total"]
# if every request took about the whole wall, they ran together; if they add up to the wall, they ran one after another
timing["last_call_wall_s"] = round(dt, 2)
timing["last_call_request_latencies_s"] = [round(x, 2) for x in run.last_latencies]
timing["decode_s_per_step"] = sorted(dec)[1]
timing["decode_tokens_per_s"] = num_seqs / timing["decode_s_per_step"]
timing["generated_per_call"] = gen
json.dump(timing, open(os.path.join(out_dir, "timing.json"), "w"), indent=1)
print("TIMING", json.dumps(timing), flush=True)

run(10, 1)
post("/start_profile", {})
run(10, 33)                                                     # 32 decode steps under the profiler
post("/stop_profile", {})
time.sleep(8)                                                   # the trace is written asynchronously
print("PROFILE_DONE", flush=True)
