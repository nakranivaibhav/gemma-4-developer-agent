"""Kaggle-time calculator: how long would these agent runs take on the scorer's 4x L4 machine?

Works on traces from ANY machine, because it never uses the recorded wall time. It replays each
trace's token counts (prompt, output, per LLM call) through a speed PROFILE of the target machine:

    call time = per_call_s + new_prompt_tokens / prefill_tok_s + output_tokens / decode_tok_s(context)
    task time = sum of call times + tool_s_per_call * tool calls + setup_min

RULE OF THUMB (measured on Kaggle 4x L4, 2026-10-15, results/kaggle_speed/): for the 31B through a LoRA adapter,

    task seconds ~ output tokens / 32 + 2 * tool calls + new prompt tokens / 1300 + ~30 s setup

Use / 36 for the 31B with no adapter, / 60 for the 12B, / 86 for the E4B, / 119 for the E2B (30k context, no adapter;
profiles l4x4_*_nolora). Prompt reading is cheap because vLLM caches the earlier
conversation. Without the prompt term it was within 15% of every one of 7 real Kaggle tasks, 5% low in total.
Budget: 12 h / 120 tasks = 6 min per task, about 8.5k output tokens per task on average. Still unmeasured: tool
time under the scorer's Docker 2-vCPU cap (the Kaggle notebook sandbox has no cap) and the per-task setup time.

Profiles live in kaggle_time_profiles.json. `forum_31b` / `forum_12b` are the forum numbers
(35 / 67 tok/s, 0.6 s per call). `fit` turns a speed_sweep.csv from the Kaggle speed-test
notebook into a measured profile.

  predict  --profile P --ledger results/task_runs.jsonl --runs l100_lean_p1,l100_lean_p2 [--tasks F]
           [--n-tasks 120 --cap-min 6 --budget-h 12 --sims 20000]
  predict  --profile P --traces DIR_OR_FILES...
  fit      --sweep speed_sweep.csv --setting no_lora --requests-to "base model" --name l4x4_31b_nolora
"""
import argparse
import csv
import glob
import json
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILES = HERE / 'kaggle_time_profiles.json'
DEFAULT_PROFILES = {
    'forum_31b': {'per_call_s': 0.6, 'prefill_tok_s': None, 'decode': [[0, 35.0]], 'tool_s_per_call': 0.0,
                  'setup_min': 0.5, 'source': 'forum #748287 (35 tok/s) + #747857 (8.7 s/call)'},
    'forum_12b': {'per_call_s': 0.6, 'prefill_tok_s': None, 'decode': [[0, 67.0]], 'tool_s_per_call': 0.0,
                  'setup_min': 0.5, 'source': 'forum #748287 (67 tok/s)'},
}


def load_profiles():
    prof = dict(DEFAULT_PROFILES)
    if PROFILES.exists():
        prof.update(json.loads(PROFILES.read_text()))
    return prof


def decode_rate(profile, context):
    """Output tokens per second at this context length (linear between measured points)."""
    pts = sorted(profile['decode'])
    if context <= pts[0][0]:
        return pts[0][1]
    for (c0, r0), (c1, r1) in zip(pts, pts[1:]):
        if context <= c1:
            return r0 + (r1 - r0) * (context - c0) / (c1 - c0)
    return pts[-1][1]


def trace_calls(path):
    """[(prompt_tokens, new_prompt_tokens, output_tokens)] per LLM call, plus the tool-call count."""
    t = json.loads(Path(path).read_text())
    calls, tools, last = [], 0, {}
    for s in t.get('steps', []):
        m = s.get('metrics')
        if s.get('tool_calls'):
            tc = s['tool_calls']
            tools += tc.count('function_name') if isinstance(tc, str) else len(tc)
        if not m or not m.get('prompt_tokens'):
            continue
        who = (s.get('extra') or {}).get('author', '')
        p, o = m['prompt_tokens'], m.get('completion_tokens') or 0
        prev = last.get(who)
        # new tokens since this agent's previous call; a shrink means history was compacted (cold again)
        new = p if prev is None or p < prev[0] else max(0, p - prev[0] - prev[1])
        last[who] = (p, o)
        calls.append((p, new, o))
    return calls, tools


def task_minutes(profile, calls, tools):
    s = 0.0
    for p, new, o in calls:
        s += profile['per_call_s'] + o / decode_rate(profile, p)
        if profile.get('prefill_tok_s'):
            s += new / profile['prefill_tok_s']
    s += profile.get('tool_s_per_call', 0.0) * tools
    return s / 60 + profile.get('setup_min', 0.0)


def runs_from_ledger(ledger, run_ids, tasks):
    out = []
    root = Path(ledger).resolve().parent.parent
    for r in map(json.loads, open(ledger)):
        if r.get('run_id') not in run_ids or (tasks and r['id'] not in tasks) or r.get('harness_error'):
            continue
        tp = root / r['trace'] if r.get('trace') else None
        if tp and tp.exists():
            out.append(dict(id=r['id'], run=r['run_id'], trace=tp, resolved=bool(r.get('resolved'))))
    return out


def predict(a):
    profile = load_profiles()[a.profile]
    if a.ledger:
        sys.path.insert(0, str(HERE))
        import manifest  # a manifest selector or a file of ids
        tasks = set(manifest.ids(a.tasks)) if a.tasks else None
        runs = runs_from_ledger(a.ledger, set(a.runs.split(',')), tasks)
    else:
        files = [f for x in a.traces for f in (glob.glob(f'{x}/**/trace_*.json', recursive=True) if Path(x).is_dir() else [x])]
        runs = [dict(id=Path(f).stem.removeprefix('trace_'), run='-', trace=Path(f), resolved=None) for f in files]
    for r in runs:
        r['min'] = task_minutes(profile, *trace_calls(r['trace']))
    mins = sorted(r['min'] for r in runs)
    q = lambda p: mins[min(len(mins) - 1, int(p * len(mins)))]
    print(f"profile {a.profile}: {profile.get('source', '')}")
    print(f'{len(runs)} runs | minutes per task: median {q(.5):.1f}  mean {st.mean(mins):.1f}  p90 {q(.9):.1f}  max {mins[-1]:.1f}')
    if a.per_task:
        for r in sorted(runs, key=lambda r: r['min']):
            print(f"  {r['id']:18s} {r['run']:14s} {r['min']:6.1f} min  resolved={r['resolved']}")
    caps = [None] + [float(c) for c in a.cap_min.split(',')] if a.cap_min else [None]
    rng = random.Random(0)
    print(f'\n{a.n_tasks} tasks, {a.budget_h} h budget, {a.sims} simulated submissions (tasks drawn from these runs):')
    for cap in caps:
        capped = [min(r['min'], cap + profile.get('setup_min', 0.0)) if cap else r['min'] for r in runs]
        totals = sorted(sum(rng.choice(capped) for _ in range(a.n_tasks)) / 60 for _ in range(a.sims))
        over = sum(t > a.budget_h for t in totals) / len(totals)
        hit = [r for r in runs if cap and r['min'] > cap + profile.get('setup_min', 0.0)]
        lost = sum(1 for r in hit if r['resolved'])
        label = f'cap {cap:g} min' if cap else 'no cap'
        extra = f' | runs cut by cap {len(hit)}/{len(runs)}, of them solved {lost}' if cap else ''
        print(f'  {label:12s} typical total {totals[len(totals) // 2]:5.1f} h  (90% range {totals[int(.05 * len(totals))]:.1f}-{totals[int(.95 * len(totals))]:.1f})'
              f'  chance over {a.budget_h:g} h: {100 * over:4.1f}%{extra}')


def fit(a):
    rows = [r for r in csv.DictReader(open(a.sweep)) if r['setting'] == a.setting and r['requests_to'] == a.requests_to]
    if not rows:
        raise SystemExit('no sweep rows match --setting/--requests-to')
    by = lambda kind, key: {c: st.median(float(r[key]) for r in rows if r['kind'] == kind and int(r['context']) == c)
                            for c in sorted({int(r['context']) for r in rows})}
    decode = by('warm', 'decode_tok_s')
    cold, warm = by('cold', 'first_token_s'), by('warm', 'first_token_s')
    cs = sorted(cold)
    slope = (cold[cs[-1]] - cold[cs[0]]) / (cs[-1] - cs[0])  # seconds per prompt token, processed cold
    prefill = 1 / slope if slope > 0 else None
    per_call = max(0.0, warm[cs[0]] - (300 / prefill if prefill else 0))
    prof = {'per_call_s': round(per_call, 3), 'prefill_tok_s': round(prefill, 1) if prefill else None,
            'decode': [[c, round(v, 2)] for c, v in decode.items()], 'tool_s_per_call': a.tool_s,
            'setup_min': a.setup_min, 'source': f'{a.sweep} setting={a.setting} requests_to={a.requests_to}'}
    allp = json.loads(PROFILES.read_text()) if PROFILES.exists() else {}
    allp[a.name] = prof
    PROFILES.write_text(json.dumps(allp, indent=2))
    print(json.dumps({a.name: prof}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('predict')
    p.add_argument('--profile', default='forum_31b')
    p.add_argument('--ledger')
    p.add_argument('--runs', default='')
    p.add_argument('--tasks')
    p.add_argument('--traces', nargs='*', default=[])
    p.add_argument('--n-tasks', type=int, default=120)
    p.add_argument('--budget-h', type=float, default=12.0)
    p.add_argument('--cap-min', default='')
    p.add_argument('--sims', type=int, default=20000)
    p.add_argument('--per-task', action='store_true')
    f = sub.add_parser('fit')
    f.add_argument('--sweep', required=True)
    f.add_argument('--setting', default='no_lora')
    f.add_argument('--requests-to', default='base model')
    f.add_argument('--name', required=True)
    f.add_argument('--tool-s', type=float, default=0.0)
    f.add_argument('--setup-min', type=float, default=0.5)
    a = ap.parse_args()
    predict(a) if a.cmd == 'predict' else fit(a)
