"""Render training rows through the model's own chat template and compare with what the run really saw.

For each row of collect_sft.py output: apply the tokenizer chat template (with the harness tool schemas from
runs/sft/tools_schema.json, thinking on) and count tokens. A faithful rebuild lands close to the run's
peak context (the last request's prompt + its output). Prints the ratio spread and the rows over the
32k model limit, and writes the rendered token count back as `rendered_tokens` (--write).

usage: sft_render_check.py [--rows runs/sft/solved_v0.jsonl] [--tokenizer data/external/model] [--write]
"""
import argparse
import json
import statistics as st
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]  # the repo root


def to_template(messages):
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):  # the template wants arguments as a dict, the OpenAI wire format has a JSON string
            m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])}}
                               for c in m["tool_calls"]]
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="runs/sft/solved_v0.jsonl")
    ap.add_argument("--tokenizer", default="data/external/model")
    ap.add_argument("--tools", default="runs/sft/tools_schema.json")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(ROOT / a.tokenizer)
    tools = json.load(open(ROOT / a.tools))
    rows = [json.loads(line) for line in open(ROOT / a.rows)]
    ratios, over = [], 0
    for r in rows:
        txt = tok.apply_chat_template(to_template(r["messages"]), tools=tools, tokenize=False, enable_thinking=True)
        n = len(tok(txt, add_special_tokens=False)["input_ids"])
        r["rendered_tokens"] = n
        over += n > 32768
        if r.get("peak_context_tokens"):
            ratios.append(n / r["peak_context_tokens"])
    q = st.quantiles(ratios, n=20)
    print(f"{len(rows)} rows | rendered / run peak context: p5 {q[0]:.3f}  median {st.median(ratios):.3f}  p95 {q[-1]:.3f}"
          f" | over 32768 tokens: {over}")
    if a.write:
        with open(ROOT / a.rows, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"rendered_tokens written to {a.rows}")


if __name__ == "__main__":
    main()
