"""Turn a torch-profiler chrome trace from profile_decode.py into a per-kernel-family table.

usage: python -I kits/swe-agent/bench/kprof/summarize_trace.py <out_dir> [...]
Reads <out_dir>/trace/*.json(.gz) and <out_dir>/timing.json; prints kernel families by share of GPU time,
GPU-busy fraction, and the Marlin matmul time per step against the weight-read floor.
"""
import glob
import gzip
import json
import os
import re
import sys

FAMILIES = [
    ("LoRA adapter", r"lora"),
    ("4-bit matmul (Marlin)", r"marlin"),
    ("bf16 matmul (cuBLAS/CUTLASS)", r"gemm|cutlass|nvjet|cublas|matmul"),
    ("attention", r"flash|fmha|attn|attention|paged|flashinfer|decode_kernel|prefill_kernel|merge_state|reshape_and_cache"),
    ("norm", r"rms|norm"),
    ("activation / elementwise", r"gelu|silu|act|elementwise|vectorized|mul_|add_|fill|copy|cat|index|gather|scatter|where|clamp|rotary|rope"),
    ("sampling / logits", r"sample|softmax|topk|top_k|argmax|sort|cumsum|multinomial|penalt|logit|ngram|spec|rejection"),
    ("reductions", r"reduce|sum|mean|max"),
]
WEIGHT_BYTES_4BIT = 19.85 * 2**30          # whole model as loaded (bf16 output layer included)
HBM_BYTES_PER_S = 1.79e12                  # RTX 5090


def family(name):
    low = name.lower()
    for fam, pat in FAMILIES:
        if re.search(pat, low):
            return fam
    return "other"


for out_dir in sys.argv[1:]:
    files = glob.glob(os.path.join(out_dir, "trace", "*.json*"))
    if not files:
        print(out_dir, "NO TRACE")
        continue
    kernels = []
    for f in files:
        with (gzip.open if f.endswith(".gz") else open)(f, "rt") as fh:
            ev = json.load(fh)["traceEvents"]
        ks = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
        if len(ks) > len(kernels):
            kernels = ks
    if not kernels:
        print(out_dir, "NO KERNEL EVENTS in", len(files), "trace files")
        continue
    timing = json.load(open(os.path.join(out_dir, "timing.json")))
    by_fam, by_name = {}, {}
    # one decode step launches 60 layers x 4 Marlin matmuls (qkv, o, gate_up, down); infer the step count from that
    steps = max(1, round(sum(1 for e in kernels if "marlin" in e["name"].lower()) / 240))
    for e in kernels:
        f = family(e["name"])
        by_fam[f] = by_fam.get(f, 0) + e["dur"]
        by_name[e["name"]] = by_name.get(e["name"], 0) + e["dur"]
    total = sum(by_fam.values())
    # GPU-busy: union of kernel intervals over the span of the trace
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    busy, cur_s, cur_e = 0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    span = iv[-1][1] - iv[0][0]
    step_ms = timing["decode_s_per_step"] * 1000
    floor_ms = WEIGHT_BYTES_4BIT / HBM_BYTES_PER_S * 1000
    print(f"== {os.path.basename(out_dir)}: {timing['num_seqs']} seq x {timing['prompt_tokens']} prompt tokens, "
          f"ngram={timing['ngram']}  | un-profiled decode step {step_ms:.1f} ms  ({timing['decode_tokens_per_s']:.1f} tok/s), "
          f"prefill of the prompt set {timing['prefill_s_per_prompt_set']:.2f} s")
    print(f"   trace covers {steps} steps ({len(kernels)} kernel launches)")
    print(f"   profiled span {span / 1000 / steps:.1f} ms/step, GPU busy {busy / span:.0%}, kernel launches per step {len(kernels) / steps:.0f}")
    print(f"   weight-read floor {floor_ms:.1f} ms/step; Marlin alone {by_fam.get('4-bit matmul (Marlin)', 0) / 1000 / steps:.1f} ms/step")
    for f, d in sorted(by_fam.items(), key=lambda kv: -kv[1]):
        print(f"   {d / total:5.1%}  {d / 1000 / steps:6.2f} ms/step  {f}")
    print("   top of 'other':", ", ".join(n[:40] for n, _ in sorted(((n, d) for n, d in by_name.items() if family(n) == "other"), key=lambda kv: -kv[1])[:4]))
    print("   top kernels:")
    for n, d in sorted(by_name.items(), key=lambda kv: -kv[1])[:8]:
        print(f"     {d / total:5.1%}  {n[:90]}")
