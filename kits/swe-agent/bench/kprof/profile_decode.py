"""Profile one vLLM decode workload at agent-like shapes and time it without the profiler.

usage: python -I kits/swe-agent/bench/kprof/profile_decode.py <model_dir> <out_dir> <num_seqs> <prompt_tokens> [ngram]
Writes <out_dir>/timing.json (un-profiled wall per decode step and per prefill) and a torch-profiler
trace under <out_dir>/trace/ covering 32 decode steps. Prompts are random token ids so the length is exact.
"""
import json
import os
import random
import sys
import time

model_dir, out_dir, num_seqs, prompt_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ngram = len(sys.argv) > 5 and sys.argv[5] == "ngram"
os.makedirs(os.path.join(out_dir, "trace"), exist_ok=True)
os.environ["VLLM_TORCH_PROFILER_DIR"] = os.path.join(out_dir, "trace")

from vllm import LLM, SamplingParams  # noqa: E402  (env var must be set before import)
from vllm.inputs import TokensPrompt  # noqa: E402

kw = {}
if ngram:
    kw["speculative_config"] = {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4, "prompt_lookup_min": 2}
llm = LLM(model=model_dir, gpu_memory_utilization=0.88, max_model_len=32768, max_num_seqs=8, **kw)
vocab = llm.get_tokenizer().vocab_size
rng = random.Random(0)


def prompts(seed):
    r = random.Random(seed)
    # avoid the lowest ids (special tokens) so a random prompt never contains <bos>/<eos>
    return [TokensPrompt(prompt_token_ids=[r.randrange(1000, vocab - 1000) for _ in range(prompt_tokens)]) for _ in range(num_seqs)]


def run(seed, max_tokens):
    t = time.perf_counter()
    out = llm.generate(prompts(seed), SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0), use_tqdm=False)
    dt = time.perf_counter() - t
    gen = sum(len(o.outputs[0].token_ids) for o in out)
    return dt, gen


timing = {"num_seqs": num_seqs, "prompt_tokens": prompt_tokens, "ngram": ngram}
run(1, 4)                                        # warm-up, compiles anything lazy
# prefill: fresh prompts, one token out, repeated on new seeds so the prefix cache cannot help
pre = [run(10 + i, 1)[0] for i in range(3)]
timing["prefill_s_per_prompt_set"] = sorted(pre)[1]
# decode: same prompts again (prefix cached, so prefill is ~free) and 64 new tokens
run(10, 1)
dec = []
for _ in range(3):
    dt, gen = run(10, 65)
    dec.append(dt / 64)                          # seconds per decode step (all seqs advance together)
timing["decode_s_per_step"] = sorted(dec)[1]
timing["decode_tokens_per_s"] = num_seqs / timing["decode_s_per_step"]
timing["generated_per_call"] = gen
json.dump(timing, open(os.path.join(out_dir, "timing.json"), "w"), indent=1)
print("TIMING", json.dumps(timing))

# profiled pass: 32 decode steps on cached prompts
run(10, 1)
llm.start_profile()
run(10, 33)
llm.stop_profile()
time.sleep(5)                                    # the trace is written asynchronously
print("PROFILE_DONE", os.listdir(os.path.join(out_dir, "trace")))
