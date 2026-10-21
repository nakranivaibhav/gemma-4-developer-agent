"""Reproduce the vLLM gemma4 tool-call parser hang (vLLM 0.19.1, the version in the competition wheels).

`_parse_gemma4_array` loops forever when a list argument holds a stray `]` outside a string: its bare-value branch
stops on `]` without moving past it, appends an empty item and tries again, so one CPU core spins and memory grows
without bound. The parse runs on the API server's event loop, so every other request waits: the whole server freezes
with no error. Found 2026-10-20 when E2B wrote `args:["upload", file"]` (a string missing its opening quote) for
run_skill_script and froze a 128-task run (journal 2026-10-20).

Any tool argument the model writes as a list can trigger it (run_skill_script `args`, get_code_subgraph `nodes`).
A dict whose values are strings did not hang in 480k fuzzed inputs; a dict value that is itself a malformed list does.

usage: envs/harness/bin/python -I analysis/gemma4_parser_hang.py   (each case runs in a child with a 5 s timeout)
"""
import multiprocessing as mp

Q = '<|"|>'  # gemma4 string delimiter
CASES = [
    f'args:[{Q}upload{Q},{Q}file{Q}]',      # well formed list
    f'args:{{query:{Q}upload file{Q}}}',    # dict with a string value
    f'args:[{Q}upload{Q}, file{Q}]',        # list item missing its opening quote: hangs
    f'args:[upload, file{Q}]',              # bare words with a stray quote: hangs
    f'nodes:[a[]',                          # unbalanced bracket in a list: hangs
]


def parse(s, q):
    from vllm.tool_parsers.gemma4_tool_parser import _parse_gemma4_args
    q.put(repr(_parse_gemma4_args(s))[:80])


if __name__ == "__main__":
    for s in CASES:
        q = mp.Queue()
        p = mp.Process(target=parse, args=(s, q))
        p.start()
        p.join(5)
        shown = s.replace(Q, '"')
        if p.is_alive():
            p.kill()
            print(f"HANG  {shown}")
        else:
            print(f"ok    {shown}  ->  {q.get() if not q.empty() else 'error'}")
