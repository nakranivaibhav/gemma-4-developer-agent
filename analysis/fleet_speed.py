"""Per-GPU, per-step speed table for a fleet run (kits/swe-agent/run/fleet.py): one row per (gpu, tasks-at-once step).

usage: python -I analysis/fleet_speed.py <fleet_dir> [ledger.jsonl] [--price gpu=$/h ...]

Steps come from the `target` events. Server numbers come from the minute-by-minute /metrics snapshots in
<fleet_dir>/metrics/<gpu>_<port>/ (counter deltas across the step; gauges averaged). Run numbers come from the ledger
rows of that GPU, joined to the `done` events for their end time; a run belongs to the step it STARTED in.
"""
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _t(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def _snap(path):
    out = defaultdict(float)
    for line in path.read_text().splitlines():
        if line.startswith("vllm:") and " " in line:
            name = line.split("{")[0].split(" ")[0]
            try:
                out[name] += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


def main():
    fd = Path(sys.argv[1])
    args = sys.argv[2:]
    ledger = Path(args.pop(0)) if args and not args[0].startswith("--") else Path(__file__).resolve().parents[1] / "results" / "task_runs.jsonl"
    price = {}
    if "--price" in args:
        for kv in args[args.index("--price") + 1:]:
            g, p = kv.split("=")
            price[g] = float(p)
    ev = [json.loads(l) for l in open(fd / "events.jsonl")]
    m = json.loads((fd / "manifest.json").read_text())
    end_of_run = {(e["gpu"], e["task"], e["run"]): _t(e["t"]) for e in ev if e["ev"] == "done"}
    rows = [r for r in map(json.loads, open(ledger)) if r.get("fleet_gpu") and r.get("run_id") in m["runs"]]
    print("gpu    step  from   to    | runs  solved  avg_min  >=55min  100calls | req/min  out_tok/s  ms/tok  "
          "running  waiting  kv%  preempt  cached%  | runs/h  $/run")
    for gpu in m["gpus"]:
        steps = [(_t(e["t"]), e["target"]) for e in ev if e["ev"] == "target" and e["gpu"] == gpu]
        if not steps:
            continue
        last_end = max([v for (g, _, _), v in end_of_run.items() if g == gpu] + [steps[-1][0]])
        mdirs = sorted((fd / "metrics").glob(f"{gpu}_*"))
        snaps = []
        for d in mdirs:
            for f in sorted(d.glob("*.txt")):
                ts = datetime.strptime(f.stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
                snaps.append((ts, f))
        for i, (t0, target) in enumerate(steps):
            t1 = steps[i + 1][0] if i + 1 < len(steps) else last_end
            label = f"{target}" if target else "drain"
            mine = []
            for r in rows:
                if r["fleet_gpu"] != gpu:
                    continue
                te = end_of_run.get((gpu, r["id"], r["run_id"]))
                if te and t0 <= te - r["wall_seconds"] < t1:
                    mine.append(r)
            fin = sum(1 for (g, _, _), te in end_of_run.items() if g == gpu and t0 <= te < t1)
            hrs = (t1 - t0) / 3600
            s_in = [(ts, f) for ts, f in snaps if t0 <= ts < t1]
            srv = ""
            if len(s_in) >= 2:
                a, b = _snap(s_in[0][1]), _snap(s_in[-1][1])
                mins = (s_in[-1][0] - s_in[0][0]) / 60
                d = lambda k: b[k] - a[k]  # noqa: E731
                g_ = [_snap(f) for _, f in s_in]
                avg = lambda k: sum(x[k] for x in g_) / len(g_)  # noqa: E731
                itl = 1000 * d("vllm:inter_token_latency_seconds_sum") / max(1, d("vllm:inter_token_latency_seconds_count"))
                cached = 100 * d("vllm:prompt_tokens_cached_total") / max(1, d("vllm:prompt_tokens_total"))
                srv = (f"{d('vllm:request_inference_time_seconds_count') / mins:7.1f}  "
                       f"{d('vllm:generation_tokens_total') / mins / 60:9.0f}  {itl:6.1f}  "
                       f"{avg('vllm:num_requests_running'):7.1f}  {avg('vllm:num_requests_waiting'):7.1f}  "
                       f"{100 * avg('vllm:kv_cache_usage_perc'):3.0f}  {d('vllm:num_preemptions_total'):7.0f}  {cached:7.0f}")
            n = len(mine)
            wm = [r["wall_seconds"] / 60 for r in mine]
            runs_h = fin / hrs if hrs else 0
            cost = f"{price[gpu] / runs_h:5.2f}" if gpu in price and runs_h else "  -  "
            print(f"{gpu:6s} {label:>5s}  {datetime.fromtimestamp(t0, timezone.utc):%H:%M}  "
                  f"{datetime.fromtimestamp(t1, timezone.utc):%H:%M} | {n:4d}  {sum(bool(r['resolved']) for r in mine):6d}  "
                  f"{(sum(wm) / n if n else 0):7.1f}  {sum(w >= 55 for w in wm):7d}  "
                  f"{sum((r.get('tool_calls') or 0) >= 99 for r in mine):8d} | {srv or ' (no metrics)':>68s} | "
                  f"{runs_h:6.1f}  {cost}")


if __name__ == "__main__":
    main()
