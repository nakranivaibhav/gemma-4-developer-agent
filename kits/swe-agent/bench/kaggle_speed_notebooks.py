"""Build the Kaggle 4xL4 speed-test notebooks (31B, 12B, E4B, E2B) from one template.

usage: python kits/swe-agent/bench/kaggle_speed_notebooks.py <out_dir> <tasks_json> [variant,variant]
Writes <out_dir>/<variant>/{speed_test.ipynb,kernel-metadata.json,timeout_s}.
A variant's `parts` picks the notebook sections: 'ABC' = sweep LoRA off, sweep LoRA on, real tasks;
'A' = the LoRA-off sweep only (the E4B/E2B runs: no adapters exist for them, and the weekly GPU quota was short);
'BC' = LoRA-on sweep + real tasks (e2b_sub03: the sub03 zip as submitted, adapter kept).
A variant's `agent` names the private dataset's folder/archive (default lean_agent); `keep_adapters` keeps the agent's
own model and LoRA unchanged instead of switching to the variant's model without adapters.
"""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
TASKS = json.loads(Path(sys.argv[2]).read_text())

VARIANTS = {
    '31b': dict(model='gemma-4-31b-it-qat-w4a16-ct', budget_min=80, reserve_min=12, timeout_s=5700),
    '12b': dict(model='gemma-4-12b-it-qat-w4a16-ct', budget_min=50, reserve_min=7, timeout_s=3900),
    'e4b': dict(model='gemma-4-e4b-it-qat-w4a16-ct', budget_min=14, reserve_min=99, timeout_s=1000, parts='A'),
    'e2b': dict(model='gemma-4-e2b-it-qat-w4a16-ct', budget_min=14, reserve_min=99, timeout_s=1000, parts='A'),
    # 2026-10-29: sub03 (E2B + e2b_sft_v1 LoRA r64) scored 0.00 on the LB; check the adapter loads and time real tasks.
    'e2b_sub03': dict(model='gemma-4-e2b-it-qat-w4a16-ct', budget_min=80, reserve_min=7, timeout_s=5400, parts='BC',
                      agent='sub03_agent', dataset='vaibhavnakrani/gemma4-sub03-agent', keep_adapters=True),
}
ONLY = sys.argv[3].split(',') if len(sys.argv) > 3 else list(VARIANTS)
STARTER_IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')}


def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n')}


def cells(v):
    size = v['model'].split('-')[2].upper()
    return [
md(f"""
# Gemma 4 Developer Agent — speed test on 4× L4 ({size})

The competition scores every submission on **4× NVIDIA L4** GPUs, with a **12-hour limit** for about 120 tasks,
one task at a time. This notebook measures how fast the **{size}** model really is on that hardware, so we can
predict whether an agent finishes in time.

It has three parts:

| Part | What it measures | Why |
|---|---|---|
| **A. Speed sweep, no LoRA** | time to first token and tokens per second, as the prompt grows from 1k to 30k tokens | the raw speed of the model |
| **B. Speed sweep, LoRA on** | the same, with the server set up the way the competition's `HARNESS_README` describes (LoRA always on, rank up to 128) | the README and the starter notebook disagree on this setting, so we time both |
| **C. Real tasks, end to end** | a real agent solving real competition tasks, timed per task | checks that the sweep numbers add up to real task times |

**Settings that must match the scorer:** accelerator `GPU L4 x4`, the competition attached as an input, internet off.
The model server is started with the same `VllmServer` settings as the official Getting Started notebook.

> Kaggle notebooks have no Docker, so part C runs the agent's commands in the harness's `subprocess` sandbox.
> The scorer uses Docker containers capped at 2 CPUs each. Model time carries over; command time is approximate.
"""),
md("""
## 0. Run settings

`RUN_BUDGET_MIN` is a time box for the whole notebook. Part C stops starting new tasks when the time is nearly used up.
"""),
code(f"""
import time
T0 = time.time()

MODEL_NAME = '{v['model']}'
RUN_BUDGET_MIN = {v['budget_min']}     # whole notebook, minutes
RESERVE_MIN = {v['reserve_min']}       # stop starting new tasks when less than this is left

# Real tasks for part C, ordered so that a cut at any point still mixes short and long tasks
E2E_TASKS = {json.dumps(TASKS)}

AGENT_NAME = '{v.get('agent', 'lean_agent')}'   # folder or <name>.tar.gz in the private agent dataset
KEEP_ADAPTERS = {bool(v.get('keep_adapters'))}       # True: run the agent exactly as submitted (its model + LoRA)

def minutes_used():
    return (time.time() - T0) / 60
"""),
md("""
## 1. Install the competition's packages

Copied from the official Getting Started notebook: install the pinned `vllm`, `swegemma` and `adk-submission`
wheels from the competition's wheelhouse dataset (no internet needed).
"""),
code("""
import glob, importlib, json, logging, os, re, shutil, subprocess, sys
from pathlib import Path

os.environ.update({
    'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'TRANSFORMERS_OFFLINE': '1',
    'VLLM_NO_USAGE_STATS': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
    'VLLM_ENGINE_READY_TIMEOUT_S': '1200', 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
    'NCCL_DEBUG': 'WARN', 'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'LITELLM_TELEMETRY': 'False',
    'LITELLM_LOG': 'WARNING', 'OPENAI_API_KEY': 'EMPTY', 'LOCAL_API_KEY': 'EMPTY',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'OTEL_SDK_DISABLED': 'true', 'OTEL_PYTHON_DISABLED': 'true',
})
for name in ('LiteLLM', 'LiteLLM Router', 'LiteLLM Proxy', 'litellm', 'httpx', 'httpcore', 'openai', 'urllib3'):
    logging.getLogger(name).setLevel(logging.WARNING)

WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
for pth in glob.glob('/usr/local/lib/python*/*-packages/*cutlass*.pth'):
    os.unlink(pth)

# Kaggle strips the '+' from '+cu128' wheel names; restore it
tmp_whl = Path('/tmp/wheelhouse')
tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob('*.whl'):
    if 'cutlass' in w.name.lower():
        continue
    name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
    if not (tmp_whl / name).exists():
        os.symlink(w, tmp_whl / name)

t = time.time()
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall',
                *sorted(str(w) for w in tmp_whl.glob('*.whl'))], check=True)
importlib.invalidate_caches()
print(f'Installed wheels in {time.time() - t:.0f} s')
"""),
md("""
## 2. Inputs: model, competition data and the agent

The agent comes from a private dataset (`AGENT_NAME` above). Unless `KEEP_ADAPTERS` is set,
for a model other than the 31B the agent files are switched to that model and the 31B-shaped LoRA adapters are removed,
the same way the Getting Started notebook does it.
"""),
code("""
import tarfile

DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
WORK = Path('/kaggle/working')
MODEL_PATH = Path(f'/kaggle/input/models/google/gemma-4/other/{MODEL_NAME}/2')
assert MODEL_PATH.exists(), f'model not attached: {MODEL_PATH}'

# Find the agent: either an unpacked folder or the uploaded archive
AGENT_DIR = WORK / 'agent'
shutil.rmtree(AGENT_DIR, ignore_errors=True)
found = sorted(glob.glob(f'/kaggle/input/**/{AGENT_NAME}/agent.yaml', recursive=True))
if found:
    shutil.copytree(Path(found[0]).parent, AGENT_DIR)
else:
    archive = sorted(glob.glob(f'/kaggle/input/**/{AGENT_NAME}.tar.gz', recursive=True))[0]
    with tarfile.open(archive) as tf:
        tf.extractall(WORK)
    (WORK / AGENT_NAME).rename(AGENT_DIR)

if MODEL_NAME != 'gemma-4-31b-it-qat-w4a16-ct' and not KEEP_ADAPTERS:
    for y in [AGENT_DIR / 'agent.yaml', *sorted((AGENT_DIR / 'sub_agents').glob('*.yaml'))]:
        text = y.read_text()
        text = re.sub(r'^(\\s*model:\\s*).*$', rf'\\g<1>{MODEL_NAME}', text, flags=re.MULTILINE)
        text = re.sub(r'^\\s*adapter:\\s*.*\\n?', '', text, flags=re.MULTILINE)
        y.write_text(text)
    shutil.rmtree(AGENT_DIR / 'adapters', ignore_errors=True)

print('model :', MODEL_PATH)
print('agent :', sorted(str(p.relative_to(AGENT_DIR)) for p in AGENT_DIR.rglob('*') if p.is_file()))
"""),
md("""
## 3. Helpers: start the model server, time one request

* `start_server(setting)` starts vLLM with one of two settings:
  * **`no_lora`** — the Getting Started notebook's settings for an agent without adapters (90% GPU memory, LoRA off).
  * **`readme_lora`** — the `HARNESS_README` scorer settings (80% GPU memory, LoRA always on, up to 8 adapters of rank up to 128).
* `timed_request(...)` sends a prompt made of random token ids and asks for an exact number of output tokens
  (`ignore_eos`), streaming the answer so we can time the **first token** and the **tokens per second** after it.
  Random tokens are fine here: speed depends on how many tokens there are, not what they say.
"""),
code("""
import random
import requests
import torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

BASE_URL = 'http://127.0.0.1:8000'
EXTRA_ARGS = {
    'gemma-4-12b-it-qat-w4a16-ct': ('--hf-overrides', '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'),
}
ADAPTERS = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
ADAPTER_NAMES = list(getattr(ADAPTERS, 'adapters', None) or {})
print('adapters in the agent:', ADAPTER_NAMES or 'none')

SETTINGS = {
    'no_lora':     dict(gpu_memory_utilization=0.90, enable_lora=False, max_loras=1, max_lora_rank=128),
    'readme_lora': dict(gpu_memory_utilization=0.80, enable_lora=True, max_loras=8, max_lora_rank=128,
                        default_chat_template_kwargs={'enable_thinking': True}),
}

def start_server(setting):
    cfg = VllmConfig(
        model=str(MODEL_PATH), port=8000, host='127.0.0.1',
        tool_call_parser='gemma4', reasoning_parser='gemma4',
        max_model_len=32768, dtype='bfloat16', enable_auto_tool_choice=True,
        scheduler_reserve_full_isl=False, tensor_parallel_size=torch.cuda.device_count(),
        startup_timeout=60 * 20, extra_args=EXTRA_ARGS.get(MODEL_NAME, ()),
        **SETTINGS[setting],
    )
    use_adapters = ADAPTERS if SETTINGS[setting]['enable_lora'] else None
    t = time.time()
    server = VllmServer(cfg, adapter_manifest=use_adapters)
    server.start()
    print(f'[{setting}] server ready in {time.time() - t:.0f} s')
    return server, round(time.time() - t, 1)

def stop_server(server):
    server.stop()
    time.sleep(15)  # let the GPU memory drain before the next start

def served_models():
    # (base model name, [LoRA adapter names]) as the server lists them
    data = requests.get(f'{BASE_URL}/v1/models', timeout=30).json()['data']
    base = [m['id'] for m in data if not m.get('parent')]
    loras = [m['id'] for m in data if m.get('parent')]
    return base[0], loras

_rng = random.Random(0)
def random_ids(n):
    return [_rng.randrange(1000, 150000) for _ in range(n)]

def timed_request(model, prompt_ids, out_tokens):
    body = {'model': model, 'prompt': prompt_ids, 'max_tokens': out_tokens, 'min_tokens': out_tokens,
            'ignore_eos': True, 'temperature': 0.0, 'stream': True, 'stream_options': {'include_usage': True}}
    t0 = time.perf_counter(); first = last = None; usage = {}
    with requests.post(f'{BASE_URL}/v1/completions', json=body, stream=True, timeout=900) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b'data: ') or line == b'data: [DONE]':
                continue
            msg = json.loads(line[6:])
            if msg.get('usage'):
                usage = msg['usage']
            if msg.get('choices'):
                last = time.perf_counter(); first = first or last
    n_out = usage.get('completion_tokens') or 0
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    return dict(prompt_tokens=usage.get('prompt_tokens'), cached_tokens=cached, out_tokens=n_out,
                first_token_s=round(first - t0, 3), total_s=round(last - t0, 3),
                decode_tok_s=round((n_out - 1) / (last - first), 2) if last > first else None)
"""),
md("""
## 4. The speed sweep

For each prompt length we send two requests:

* **cold** — a brand-new prompt, so the whole prompt is processed from scratch;
* **warm** — the same prompt plus 300 new tokens. The server reuses the cached prefix, which is what happens
  on every turn of a real agent conversation.

Each asks for exactly 256 output tokens. `decode_tok_s` is the writing speed after the first token.
"""),
code("""
import pandas as pd

CONTEXTS = [1000, 4000, 8000, 16000, 24000, 30000]
OUT_TOKENS = 256
SWEEP_ROWS = []

def sweep(setting, model, label, reps=2):
    timed_request(model, random_ids(200), 16)  # warm-up, not recorded
    for ctx in CONTEXTS:
        for rep in range(reps):
            prefix = random_ids(ctx)
            for kind, prompt in (('cold', prefix), ('warm', prefix + random_ids(300))):
                row = dict(setting=setting, requests_to=label, context=ctx, kind=kind, rep=rep,
                           **timed_request(model, prompt, OUT_TOKENS))
                SWEEP_ROWS.append(row)
    pd.DataFrame(SWEEP_ROWS).to_csv(WORK / 'speed_sweep.csv', index=False)
    print(f'[{setting} / {label}] sweep done at {minutes_used():.1f} min')

RUN_LOG = {'model': MODEL_NAME, 'gpus': torch.cuda.get_device_name(0), 'gpu_count': torch.cuda.device_count(),
           'server_start_s': {}, 'errors': []}
"""),
md("""
## 5. Part A — speed with LoRA off
"""),
code("""
try:
    server, RUN_LOG['server_start_s']['no_lora'] = start_server('no_lora')
    sweep('no_lora', served_models()[0], 'base model')
    stop_server(server)
except Exception as exc:
    RUN_LOG['errors'].append(f'part A: {exc!r}')
    print('Part A failed:', repr(exc))
"""),
md("""
## 6. Part B — speed with LoRA on (the `HARNESS_README` setting)

Requests go to the base model, and (if the agent has one) to its LoRA adapter.
The server stays up afterwards for part C.
"""),
code("""
SERVER_SETTING = 'readme_lora'
try:
    server, RUN_LOG['server_start_s']['readme_lora'] = start_server('readme_lora')
except Exception as exc:
    RUN_LOG['errors'].append(f'part B start: {exc!r}')
    print('LoRA-on server failed, part C will use the starter settings instead:', repr(exc))
    SERVER_SETTING = 'starter'
    SETTINGS['starter'] = dict(gpu_memory_utilization=0.90, enable_lora=bool(ADAPTER_NAMES),
                               max_loras=max(1, len(ADAPTER_NAMES)), max_lora_rank=128)
    server, RUN_LOG['server_start_s']['starter'] = start_server('starter')

base_name, lora_names = served_models()
print('served:', base_name, lora_names)
RUN_LOG['served'] = dict(base=base_name, loras=lora_names)
if SERVER_SETTING == 'readme_lora':
    try:
        sweep('readme_lora', base_name, 'base model')
        if lora_names:
            sweep('readme_lora', lora_names[0], f'adapter {lora_names[0]}', reps=1)
    except Exception as exc:
        RUN_LOG['errors'].append(f'part B sweep: {exc!r}')
        print('Part B sweep failed:', repr(exc))
"""),
md("""
## 7. Part C — real tasks, end to end

The same `Evaluator` the Getting Started notebook uses: phase 1 the agent writes a patch, phase 2 the hidden tests run.
Per-task limits come from the agent's own `eval_config.yaml`. If the notebook's time box is nearly used up,
the next task gets a shorter time cap (recorded in `time_cap_min`) or is skipped.
"""),
code("""
import asyncio, concurrent.futures, dataclasses, yaml
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks

def run_sync(fn, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(fn(*args, **kwargs))).result()

def part_c():
    models = server.create_model_registry(aliases=[MODEL_NAME], model_prefix='openai/', api_key='EMPTY')
    ev = yaml.safe_load((AGENT_DIR / 'eval_config.yaml').read_text())
    ev = ev.get('evaluation', ev)
    limits, gen_constraints = build_submission_limits()
    evaluator = Evaluator(EvalConfig(
        tasks_path=DATA_DIR / 'tasks.jsonl', snapshots_dir=DATA_DIR / 'snapshots',
        results_dir=WORK / 'results', submission_dir=AGENT_DIR, models=models, sandbox='subprocess',
        timeout_seconds=int(ev.get('timeout_seconds', 300)),
        verification_timeout_seconds=int(ev.get('verification_timeout_seconds', 300)),
        max_time_minutes=float(ev.get('max_time_minutes', 60.0)), max_tool_calls=int(ev.get('max_tool_calls', 100)),
        max_turns=int(ev['max_turns']) if ev.get('max_turns') is not None else None,
        limits=limits, generation_constraints=gen_constraints,
        adapter_manifest=ADAPTERS if SETTINGS[SERVER_SETTING]['enable_lora'] else None,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                        token_threshold=14336, event_retention_size=5),
        graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'),
        wheels_dir=DATA_DIR / 'wheels', verbose=False,
    ))
    all_tasks = {t.instance_id: t for t in load_tasks(DATA_DIR / 'tasks.jsonl')}
    default_cap = evaluator.config.budget.time_minutes

    for i, tid in enumerate(E2E_TASKS, start=1):
        left = RUN_BUDGET_MIN - minutes_used()
        if left < RESERVE_MIN:
            print(f'time box reached ({minutes_used():.0f} min used); skipping the remaining {len(E2E_TASKS) - i + 1} tasks')
            break
        cap = min(default_cap, left - 4)
        try:
            evaluator.config.budget = dataclasses.replace(evaluator.config.budget, time_minutes=cap)
        except Exception:
            cap = default_cap
        t = time.time()
        try:
            r = run_sync(evaluator.evaluate_task, task=all_tasks[tid], task_index=i, total_tasks=len(E2E_TASKS))
            row = dict(task=tid, resolved=r.resolved, agent_s=round(r.duration_seconds, 1), llm_calls=r.total_llm_calls,
                       total_tokens=r.total_tokens, tool_calls=r.tool_calls, test_exit_code=r.test_exit_code,
                       status=str(r.status), error=(r.error_message or '')[:300])
        except Exception as exc:
            row = dict(task=tid, error=repr(exc)[:300])
        row.update(wall_s=round(time.time() - t, 1), time_cap_min=round(cap, 1), server_setting=SERVER_SETTING)
        E2E_ROWS.append(row)
        pd.DataFrame(E2E_ROWS).to_csv(WORK / 'e2e_tasks.csv', index=False)
        print(f"[{i}/{len(E2E_TASKS)}] {tid}: resolved={row.get('resolved')} wall={row['wall_s']:.0f}s "
              f"calls={row.get('llm_calls')} at {minutes_used():.1f} min")

E2E_ROWS = []
try:
    part_c()
except Exception as exc:
    RUN_LOG['errors'].append(f'part C: {exc!r}')
    print('Part C failed:', repr(exc))
stop_server(server)
"""),
md("""
## 8. Results

Speed by prompt length (median over repeats), then the per-task table.
All raw numbers are saved in `/kaggle/working`: `speed_sweep.csv`, `e2e_tasks.csv`, `run_log.json`,
and the full agent traces under `results/traces/`.
"""),
code("""
RUN_LOG['minutes_used'] = round(minutes_used(), 1)
(WORK / 'run_log.json').write_text(json.dumps(RUN_LOG, indent=2))
print(json.dumps(RUN_LOG, indent=2))

if SWEEP_ROWS:
    df = pd.DataFrame(SWEEP_ROWS)
    print('\\nTokens per second after the first token:')
    display(df.pivot_table(index='context', columns=['setting', 'requests_to', 'kind'], values='decode_tok_s', aggfunc='median'))
    print('\\nSeconds to the first token:')
    display(df.pivot_table(index='context', columns=['setting', 'requests_to', 'kind'], values='first_token_s', aggfunc='median'))
if E2E_ROWS:
    display(pd.DataFrame(E2E_ROWS))
"""),
    ]


def trim_parts(cs, parts):
    """Sweep-only runs: replace the part B / part C code with a one-line note, say so in the intro."""
    if parts == 'ABC':
        return cs
    if parts == 'BC':
        cs[0]['source'] += ('\n\n**This run does parts B and C only.** Part A (LoRA off) was measured for this model '
                            'on 2026-10-15 (results/kaggle_speed/).')
        for c in cs:
            if c['cell_type'] == 'code' and "start_server('no_lora')" in c['source']:
                c['source'] = "print('Part A skipped: measured earlier')"
        return cs
    cs[0]['source'] += ('\n\n**This run does part A only** (the LoRA-off speed sweep). No LoRA adapter exists for this '
                        'model size, and LoRA-on with no adapter measured the same speed as LoRA-off on the 31B and 12B.')
    for c in cs:
        if c['cell_type'] != 'code':
            continue
        if 'B' not in parts and "SERVER_SETTING = 'readme_lora'" in c['source']:
            c['source'] = "SERVER_SETTING = 'no_lora'\nprint('Part B skipped: sweep-only run')"
        if 'C' not in parts and 'from swegemma.evaluate import Evaluator' in c['source']:
            c['source'] = "E2E_ROWS = []\nprint('Part C skipped: sweep-only run')"
    return cs


for key, v in VARIANTS.items():
    if key not in ONLY:
        continue
    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)
    nb = {'cells': trim_parts(cells(v), v.get('parts', 'ABC')), 'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}}}
    for i, c in enumerate(nb['cells']):
        c['id'] = f'cell-{i:02d}'
    (d / 'speed_test.ipynb').write_text(json.dumps(nb, indent=1))
    meta = {
        'id': f"vaibhavnakrani/gemma4-l4x4-speed-test-{key.replace('_', '-')}",
        'title': f"Gemma4 L4x4 speed test {key.replace('_', '-')}",
        'code_file': 'speed_test.ipynb', 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
        'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse', v.get('dataset', 'vaibhavnakrani/gemma4-lean-agent')],
        'kernel_sources': [], 'competition_sources': ['gemma-4-developer-agent'],
        'model_sources': [f"google/gemma-4/Other/{v['model']}/2"],
        'docker_image': STARTER_IMAGE, 'machine_shape': 'NvidiaL4',
    }
    (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    (d / 'timeout_s').write_text(str(v['timeout_s']))
    print('wrote', d)
