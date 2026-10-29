"""The RL loop: for each step k, with policy vk (init adapter for k = 0, else step k-1's adapter_out):

  agent    copy the base agent with vk as adapters/main_lora                  -> step_k/agent/, policy sha
  eval     (every eval.every steps) vk on the holdout tasks                   -> step_k/eval.done
  rollout  rollout.tries tries of every task, TITO capture, own ledger        -> runs/rl_<name>_s<k>_p*/, ledger
  batch    reward, advantage (verl), segments                                 -> step_k/batch.json.gz
  train    one update in a fresh process                                      -> step_k/adapter_out/ = v(k+1)

usage: uv run python -m training.rl.loop --config training/configs/rl_e2b_v1.yaml [--set k=v ...] [--max-steps N]
Launch detached under tools/run_with_watchdog.py (CLAUDE.md "Long runs"); stop gracefully: touch <run_dir>/STOP.

Every phase writes its artifacts, then step_k/<phase>.done (artifact-then-mark), so a killed loop resumes at the
first missing marker. A rollout relaunched with the same run ids skips the tries already in the ledger; a try
re-run after a kill is detected in the capture and excluded (rollouts.build: "restarted").
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import batch as batch_mod, config as C, gpu, publish, reward, rollout, rollouts

PHASES = ("agent", "eval", "rollout", "batch", "train")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())


class Loop:
    def __init__(self, cfg: C.RLConfig, log=print, impl: dict | None = None):
        self.cfg, self.log = cfg, log
        self.dir = cfg.run_dir
        self.ledger = self.dir / "ledger.jsonl"
        self.impl = {p: getattr(self, f"_{p}") for p in PHASES}
        self.impl.update(impl or {})
        self._tok = None

    # ------------------------------------------------------------------ paths
    def step_dir(self, k: int) -> Path:
        return self.dir / f"step_{k:04d}"

    def policy_in(self, k: int) -> Path:
        return C.ROOT / self.cfg.init_adapter if k == 0 else self.step_dir(k - 1) / "adapter_out"

    def optim_in(self, k: int) -> Path | None:
        p = self.step_dir(k - 1) / "optim.pt"
        return p if k > 0 and p.exists() else None

    def eval_due(self, k: int) -> bool:
        return bool(self.cfg.eval.every) and k % self.cfg.eval.every == 0

    # ------------------------------------------------------------------ driver
    def prepare(self, force: bool = False):
        self.dir.mkdir(parents=True, exist_ok=True)
        saved = self.dir / "config.yaml"
        now = yaml.safe_dump(self.cfg.to_dict(), sort_keys=True)
        if saved.exists() and saved.read_text() != now and not force:
            raise SystemExit(f"{saved} differs from this config; resume with the same config or pass --force")
        saved.write_text(now)

    def run(self, max_steps: int | None = None) -> int:
        done = 0
        for k in range(self.cfg.steps):
            sd = self.step_dir(k)
            if (sd / "done").exists():
                continue
            if (self.dir / "STOP").exists():
                self.log(f"STOP file found before step {k}")
                break
            if max_steps is not None and done >= max_steps:
                break
            self.run_step(k)
            done += 1
        return done

    def run_step(self, k: int):
        sd = self.step_dir(k)
        sd.mkdir(parents=True, exist_ok=True)
        summary = {"step": k}
        for phase in PHASES:
            if phase == "eval" and not self.eval_due(k):
                continue
            marker = sd / f"{phase}.done"
            if marker.exists():
                summary[phase] = json.loads(marker.read_text())
                continue
            self.log(f"STEP {k} PHASE {phase} start {utc()}")
            t0 = time.time()
            out = self.impl[phase](k, sd) or {}
            out["seconds"] = round(time.time() - t0, 1)
            marker.write_text(json.dumps(out))       # artifact first (inside the phase), then the mark
            summary[phase] = out
            self.log(f"STEP {k} PHASE {phase} done {json.dumps(out)[:300]}")
        with open(self.dir / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps({"utc": utc(), **summary}) + "\n")
        (sd / "done").write_text(utc())
        self._prune(k)
        self._wandb(k, summary)

    # ------------------------------------------------------------------ phases
    def _agent(self, k, sd):
        sha = publish.make_agent(C.ROOT / self.cfg.base_agent, self.policy_in(k), sd / "agent")
        return {"policy_sha": sha, "policy_in": str(self.policy_in(k))}

    def _policy_sha(self, sd) -> str:
        return json.loads((sd / "agent.done").read_text())["policy_sha"]

    def _rollout_tasks(self, prefix, sd, ids, tries, label):
        gpu.wait_free(self.cfg.rollout.need_gpu_mb, log=self.log)
        return rollout.run(self.cfg, sd / "agent", prefix, ids, tries, self.ledger, sd / f"{prefix}.tasks", label)

    def _eval(self, k, sd):
        ids = rollout.task_ids(self.cfg.eval.tasks)
        rr = self._rollout_tasks(f"rl_{self.cfg.name}_e{k:04d}", sd, ids, self.cfg.eval.tries,
                                 f"rl={self.cfg.name} step={k} eval")
        rows = rollouts.read_ledger(self.ledger, set(rr.run_ids))
        oc = [reward.classify(r) for r in rows]
        ok = [o for o in oc if not o.excluded]
        return {"tries": len(rows), "expected": rr.expected, "excluded": len(rows) - len(ok),
                "solved_in_clock": sum(o.reward for o in ok), "rate": round(sum(o.reward for o in ok) / len(ok), 4)
                if ok else None, "run_ids": rr.run_ids}

    def _rollout(self, k, sd):
        ids = rollout.task_ids(self.cfg.rollout.tasks)
        rollout.check_train_only(ids)
        rr = self._rollout_tasks(f"rl_{self.cfg.name}_s{k:04d}", sd, ids, self.cfg.rollout.tries,
                                 f"rl={self.cfg.name} step={k}")
        n = rollout.ledger_count(self.ledger, rr.run_ids)
        if n == 0:
            raise RuntimeError(f"step {k}: rollout produced no ledger rows (see runs/logs/serve_{rr.prefix}.log)")
        return {"run_ids": rr.run_ids, "capture": str(rr.capture), "rows": n, "expected": rr.expected}

    def _batch(self, k, sd):
        r = json.loads((sd / "rollout.done").read_text())
        cap, stats = rollouts.read_capture(r["capture"])
        rows = rollouts.read_ledger(self.ledger, set(r["run_ids"]))
        ro = rollouts.build(rows, cap, self._policy_sha(sd), capture_stats=stats)
        nl, ch = self.tokenizer_ids()
        b = batch_mod.make(ro, self.cfg.algo, self.cfg.model.max_train_tokens, self._policy_sha(sd), nl, ch)
        with gzip.open(sd / "batch.json.gz", "wt") as fh:
            json.dump(b.to_dict(), fh)
        return {**b.stats, "n_tries": b.n_tries, "n_tokens": b.n_tokens, "capture_calls": stats["calls"]}

    def _train(self, k, sd):
        b = json.loads((sd / "batch.done").read_text())
        out = sd / "adapter_out"
        if not b.get("segments"):                      # nothing to learn this step: the policy carries over
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir()
            for f in publish.ADAPTER_FILES:
                shutil.copy2(self.policy_in(k) / f, out / f)
            if self.optim_in(k):
                shutil.copy2(self.optim_in(k), sd / "optim.pt")
            return {"skipped": "no_signal", "adapter_out": publish.sha16(out / "adapter_model.safetensors")}
        gpu.wait_free(self.cfg.rollout.need_gpu_mb, log=self.log)
        cmd = [sys.executable, "-m", "training.rl.train_step", "--config", str(self.dir / "config.yaml"),
               "--step-dir", str(sd), "--adapter-in", str(self.policy_in(k)), "--step", str(k)]
        if self.optim_in(k):
            cmd += ["--optim-in", str(self.optim_in(k))]
        subprocess.run(cmd, cwd=C.ROOT, check=True)
        m = json.loads((sd / "train_metrics.json").read_text())
        return {k2: v for k2, v in m.items() if k2 != "epochs"} | {"epoch0": m["epochs"][0]}

    # ------------------------------------------------------------------ helpers
    def tokenizer_ids(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = batch_mod.tokenizer_ids(AutoTokenizer.from_pretrained(C.ROOT / self.cfg.model.model_dir))
        return self._tok

    def _prune(self, k):
        """Keep the adapters (every policy is a checkpoint); drop what only the next step needed."""
        shutil.rmtree(self.step_dir(k) / "agent" / "adapters", ignore_errors=True)   # = step k-1's adapter_out
        if k > 0:
            (self.step_dir(k - 1) / "optim.pt").unlink(missing_ok=True)

    def _wandb(self, k, summary):
        if not self.cfg.wandb:
            return
        try:
            import wandb
            if wandb.run is None:
                wandb.init(entity=self.cfg.wandb["entity"], project=self.cfg.wandb["project"], name=self.cfg.name,
                           config=self.cfg.to_dict(), resume="allow", id=f"rl_{self.cfg.name}")
            flat = {}
            for ph, d in summary.items():
                if isinstance(d, dict):
                    for key, v in d.items():
                        if isinstance(v, (int, float)):
                            flat[f"{ph}/{key}"] = v
                        elif isinstance(v, dict):
                            flat.update({f"{ph}/{key}/{a}": b for a, b in v.items() if isinstance(b, (int, float))})
            wandb.log(flat, step=k)
        except Exception as exc:                     # metrics.jsonl is the record; W&B is a view
            self.log(f"WANDB_ERROR {type(exc).__name__}: {exc}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--force", action="store_true", help="resume although the config changed")
    a = ap.parse_args(argv)
    cfg = C.load(a.config, a.set)
    lp = Loop(cfg, log=lambda s: print(s, flush=True))
    lp.prepare(force=a.force)
    print(f"RL_START {cfg.name} {utc()} run_dir={cfg.run_dir}", flush=True)
    with gpu.hold(cfg.gpu_hold):
        n = lp.run(a.max_steps)
    print(f"RL_DONE {cfg.name} steps_run={n} {utc()}", flush=True)


if __name__ == "__main__":
    main()
