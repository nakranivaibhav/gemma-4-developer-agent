"""Drive the 9 harness tools directly (no model) on one real task and record
exact inputs/outputs to refs/tool_samples.md. Run with the harness env:
  envs/harness/bin/python -I kits/swe-agent/check/tool_samples.py [instance_id]
"""
import asyncio, inspect, json, sys
from pathlib import Path

from swegemma.config import EvalConfig
from swegemma.context import SwegemmaContext
from swegemma.harness.container_setup import (
    extract_snapshot, install_editable_package, install_test_dependencies,
    setup_baseline_commit, setup_container_wheels, setup_git_exclude,
    setup_workspace_test_config,
)
from swegemma.models import load_tasks
from swegemma.sandbox import ContainerConfig, ContainerManager
from swegemma.sandbox.base import sandbox_start, sandbox_stop

ROOT = Path(__file__).resolve().parents[3]  # the repo root
DATA = ROOT / 'data'
TASK_ID = sys.argv[1] if len(sys.argv) > 1 else None
CLIP = 1400


def clip(s: str) -> str:
    return s if len(s) <= CLIP else s[:CLIP] + f'\n… [clipped, {len(s)} chars total]'


async def call(tools, out, name, **kw):
    fn = tools[name]
    res = fn(**kw)
    if inspect.isawaitable(res):
        res = await res
    try:
        pretty = json.dumps(json.loads(res), indent=2, ensure_ascii=False)
    except Exception:
        pretty = str(res)
    out.append(f'### `{name}`\n\n**Input**\n```json\n{json.dumps(kw, indent=2)}\n```\n'
               f'**Output**\n```json\n{clip(pretty)}\n```\n')
    try:
        return json.loads(res)
    except Exception:
        return {}


async def main():
    tasks = load_tasks(DATA / 'tasks.jsonl')
    task = next(t for t in tasks if (t.instance_id == TASK_ID if TASK_ID else t.repo == 'psf/requests'))
    cfg = EvalConfig(
        tasks_path=DATA / 'tasks.jsonl', snapshots_dir=DATA / 'snapshots',
        results_dir=ROOT / 'runs' / 'tool_samples', submission_dir=DATA / 'sample_submission',
        models=None, wheels_dir=DATA / 'wheels',
        graph_dir=str(DATA / 'graphs'), embeddings_dir=str(DATA / 'embeddings'),
        max_tool_calls=100, max_time_minutes=60,
    )
    docker = ContainerManager(ContainerConfig(image=cfg.image, timeout_seconds=300))
    sid = await sandbox_start(docker)
    out = [f'# Tool input/output samples\n\nReal calls against task `{task.instance_id}` '
           f'(`{task.repo}`), made directly through the harness tool functions in a Docker '
           f'sandbox, no model involved. Outputs longer than {CLIP} chars are clipped here '
           f'(the harness itself already truncates as documented).\n']
    try:
        setup_container_wheels(docker, sid, cfg)
        extract_snapshot(docker, sid, DATA / 'snapshots' / f'{task.instance_id}.tgz')
        setup_git_exclude(docker, sid)
        install_editable_package(docker, sid)
        install_test_dependencies(docker, sid, task.repo, config=cfg)
        setup_workspace_test_config(docker, sid, repo=task.repo)
        setup_baseline_commit(docker, sid, 'baseline')
        ctx = SwegemmaContext(
            docker_manager=docker, container_id=sid, task=task, repo=task.repo,
            problem_statement=task.problem_statement,
            graph_dir=cfg.graph_dir, embeddings_dir=cfg.embeddings_dir,
            budget=cfg.budget, harness=cfg.harness,
        )
        ctx.start_agent_session()
        tools = ctx.create_tools()
        out.append(f'Tools returned by `create_tools()`: {sorted(tools)}\n')
        for n, fn in sorted(tools.items()):
            out.append(f'- `{n}{inspect.signature(fn)}`')
        out.append('')

        await call(tools, out, 'get_status')
        await call(tools, out, 'run_command', command='git log --oneline | head -3; ls | head -20')
        await call(tools, out, 'run_command', command='grep -rn "def copy" src/requests/ | head -5')
        await call(tools, out, 'run_command', command='python -c "import nonexistent_mod"')
        await call(tools, out, 'read_file', filepath='src/requests/models.py', start_line=1, end_line=12)
        await call(tools, out, 'read_file', filepath='src/requests/models.py')
        await call(tools, out, 'read_file', filepath='requests/models.py')
        await call(tools, out, 'search_similar_code', query='PreparedRequest.copy', k=3)
        await call(tools, out, 'search_similar_code', query='how are cookies merged into a session', k=3)
        await call(tools, out, 'get_code_neighbors', node='PreparedRequest.copy', max_neighbors=10)
        await call(tools, out, 'get_code_neighbors', node='copy', max_neighbors=10)
        await call(tools, out, 'get_code_neighbors', node='requests.models.PreparedRequest', edge_type='IMPORTS')
        await call(tools, out, 'get_code_subgraph', nodes=[
            'requests.models.PreparedRequest.copy', 'requests.models.PreparedRequest',
            'requests.models.PreparedRequest.__init__', 'requests.cookies._copy_cookie_jar'])
        await call(tools, out, 'write_file', filepath='scratch_probe.py', content='x = 1\nprint(x)\n')
        await call(tools, out, 'edit_file', filepath='scratch_probe.py', old_string='x = 1', new_string='x = 2')
        await call(tools, out, 'edit_file', filepath='scratch_probe.py', old_string='does not exist', new_string='y')
        await call(tools, out, 'get_status')
        await call(tools, out, 'submit_patch')
        await call(tools, out, 'get_status')
    finally:
        await sandbox_stop(docker, sid)
        docker.cleanup_all()
    dest = ROOT / 'refs' / 'tool_samples.md'
    dest.write_text('\n'.join(out), encoding='utf-8')
    print('wrote', dest, sum(len(x) for x in out), 'chars')


asyncio.run(main())
