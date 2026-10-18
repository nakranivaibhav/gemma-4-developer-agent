"""Rebuild every official code graph with graph/code_graph.py and compare it with data/graphs/.

usage: envs/harness/bin/python -I graph/graph_match.py <scratch_dir> [--only <graph stem substring>] [--jobs 10]
For each graph: extract the *.py files of one matching task snapshot into <scratch_dir>, build, diff, delete.
The build is given the official file order (read off the official node list; on Kaggle it is the builder's hashed
directory order, not reproducible), so the JSON can also be compared byte for byte.
Writes results/graph_match.csv (one row per graph) and prints a summary plus example misses per graph.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repo root
sys.path.insert(0, str(ROOT / 'graph'))
import code_graph  # noqa: E402


def one(job):
    stem, task_id, scratch = job
    work = Path(scratch) / stem
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        subprocess.run(['tar', '-xzf', str(ROOT / 'data' / 'snapshots' / f'{task_id}.tgz'), '-C', str(work),
                        '--wildcards', '*.py', '--exclude', './.git'], check=True, capture_output=True)
        off_raw = (ROOT / 'data' / 'graphs' / f'{stem}.json').read_text()
        off = json.loads(off_raw)
        # file order as the official builder happened to walk it: order of first appearance of each module
        files, _ = code_graph.parse_repo(work)
        mods = sorted({f[0] for f in files}, key=len, reverse=True)
        order = []
        for x in off['nodes']:
            m = next((m for m in mods if x['id'].startswith(m + '.')), None)
            if m is not None and m not in order:
                order.append(m)
        nodes, links, skipped = code_graph.build(work, module_order=order)
        same_bytes = code_graph.to_json(code_graph.to_graph(stem, nodes, links)) == off_raw
        edges = code_graph.pairs(links)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    off_nodes = {x['id']: x['text'] for x in off['nodes']}
    off_edges = {frozenset((e['source'], e['target'])) for e in off['edges']}
    both = set(nodes) & set(off_nodes)
    text_bad = sorted(q for q in both if nodes[q] != off_nodes[q])
    order_same = [q for q in nodes if q in off_nodes] == [q for q in off_nodes if q in nodes]
    row = dict(graph=stem, task=task_id, off_nodes=len(off_nodes), my_nodes=len(nodes),
               only_off=len(set(off_nodes) - set(nodes)), only_mine=len(set(nodes) - set(off_nodes)),
               text_diff=len(text_bad), node_order_same=order_same, json_identical=same_bytes,
               off_pairs=len(off_edges), my_pairs=len(edges), pair_tp=len(edges & off_edges),
               pair_fp=len(edges - off_edges), pair_fn=len(off_edges - edges), skipped_files=len(skipped))
    ex = dict(only_off=sorted(set(off_nodes) - set(nodes))[:5], only_mine=sorted(set(nodes) - set(off_nodes))[:5],
              text_bad=text_bad[:3], fp=[sorted(p) for p in sorted(edges - off_edges, key=sorted)[:5]],
              fn=[sorted(p) for p in sorted(off_edges - edges, key=sorted)[:5]], skipped=skipped[:3])
    return row, ex


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scratch')
    ap.add_argument('--only', default='')
    ap.add_argument('--jobs', type=int, default=10)
    x = ap.parse_args()
    stem_task = {}
    for t in map(json.loads, open(ROOT / 'data' / 'tasks.jsonl')):
        stem_task.setdefault(f"{t['repo'].split('/')[1]}_{t['base_commit']}", t['instance_id'])
    stems = sorted(p.stem for p in (ROOT / 'data' / 'graphs').glob('*.json') if x.only in p.stem)
    with Pool(x.jobs) as pool:
        results = pool.map(one, [(s, stem_task[s], x.scratch) for s in stems])
    rows = [r for r, _ in results]
    out = ROOT / 'results' / 'graph_match.csv'
    if not x.only:
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    tot = lambda k: sum(r[k] for r in rows)
    exact = sum(r['only_off'] == r['only_mine'] == r['text_diff'] == r['pair_fp'] == r['pair_fn'] == 0 for r in rows)
    print(f"graphs {len(rows)}, exact {exact} | nodes official {tot('off_nodes')} mine {tot('my_nodes')} "
          f"only-official {tot('only_off')} only-mine {tot('only_mine')} text-diff {tot('text_diff')} | "
          f"pairs official {tot('off_pairs')} tp {tot('pair_tp')} fp {tot('pair_fp')} fn {tot('pair_fn')} | "
          f"node order same {sum(r['node_order_same'] for r in rows)} | byte-identical JSON {sum(r['json_identical'] for r in rows)} | "
          f"skipped files {tot('skipped_files')}")
    for (r, ex) in results:
        if r['only_off'] or r['only_mine'] or r['text_diff'] or r['pair_fp'] or r['pair_fn']:
            print(f"- {r['graph'][:20]} only_off {r['only_off']} only_mine {r['only_mine']} text {r['text_diff']} "
                  f"fp {r['pair_fp']} fn {r['pair_fn']} :: {json.dumps(ex)[:600]}")
    if not x.only:
        print(f'wrote {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
