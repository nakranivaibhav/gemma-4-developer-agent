"""config, publish, rollout command building, GPU sharing, and the loop's resume logic (stubbed phases)."""
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from training.rl import config as C, gpu, loop as L, publish, rollout
from training.rl.tests.conftest import ROOT

BASE = {"name": "t", "init_adapter": "runs/sft/e2b_v1/adapter", "base_agent": "agents/r09_e2b_8tool_sft",
        "rollout": {"tasks": "fastapi_14463", "tries": 4, "concurrency": 8}}


def cfg(**over):
    d = json.loads(json.dumps(BASE))
    for k, v in over.items():
        d[k] = {**d.get(k, {}), **v} if isinstance(v, dict) else v
    return C.from_dict(d)


# ---------------------------------------------------------------- config
def test_config_defaults_and_temperature_from_agent():
    c = cfg()
    assert c.algo.adv_estimator == "rloo" and c.algo.loss_agg_mode == "seq-mean-token-sum-norm"
    assert c.temperature == 0.7


@pytest.mark.parametrize("bad", [{"typo": 1}, {"algo": {"lr": 1}}, {"rollout": {"tasks": "x", "triez": 2}}])
def test_unknown_keys_fail(bad):
    d = {**BASE, **bad} if "rollout" not in bad else {**BASE, **bad}
    with pytest.raises(ValueError, match="unknown"):
        C.from_dict(d)


@pytest.mark.parametrize("over", [{"rollout": {"tries": 1}}, {"rollout": {"concurrency": 10}},
                                  {"algo": {"adv_estimator": "gae"}}, {"algo": {"loss_agg_mode": "seq-mean-token-mean"}},
                                  {"algo": {"ppo_epochs": 0}}, {"optim": {"lr": 0}}])
def test_validation(over):
    with pytest.raises(ValueError):
        cfg(**over)


def test_load_with_overrides_and_dump_round_trip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(BASE))
    c = C.load(p, ["algo.loss_scale_factor=2048", "optim.betas=[0.9, 0.99]"])
    assert c.algo.loss_scale_factor == 2048 and c.optim.betas == (0.9, 0.99)
    p2 = tmp_path / "d.yaml"
    p2.write_text(yaml.safe_dump(c.to_dict()))
    assert C.load(p2) == c


# ---------------------------------------------------------------- publish
def fake_adapter(d: Path, payload=b"weights"):
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(payload)
    return d


def fake_agent(d: Path):
    (d / "prompts").mkdir(parents=True)
    (d / "agent.yaml").write_text("name: x\nadapter: main_lora\n")
    (d / "prompts" / "system.md").write_text("hi")
    fake_adapter(d / "adapters" / "main_lora", b"OLD")
    return d


def test_make_agent_replaces_adapter_and_returns_its_sha(tmp_path):
    base, ad = fake_agent(tmp_path / "base"), fake_adapter(tmp_path / "ad", b"NEW")
    sha = publish.make_agent(base, ad, tmp_path / "out")
    lora = tmp_path / "out" / "adapters" / "main_lora"
    assert (lora / "adapter_model.safetensors").read_bytes() == b"NEW"
    assert sha == publish.sha16(ad / "adapter_model.safetensors")
    assert (tmp_path / "out" / "prompts" / "system.md").read_text() == "hi"
    assert sorted(p.name for p in lora.iterdir()) == sorted(publish.ADAPTER_FILES)
    assert publish.make_agent(base, ad, tmp_path / "out") == sha          # idempotent
    assert not (tmp_path / "out.tmp").exists()


def test_make_agent_missing_files(tmp_path):
    base = fake_agent(tmp_path / "base")
    (tmp_path / "ad").mkdir()
    with pytest.raises(FileNotFoundError):
        publish.make_agent(base, tmp_path / "ad", tmp_path / "out")


def test_sha16_matches_run_tasks_pin():
    # the ledger pin of tito_smoke1 (agents/r09_e2b_8tool_sft) is 67ea0f492938acfc
    p = ROOT / "agents/r09_e2b_8tool_sft/adapters/main_lora/adapter_model.safetensors"
    if not p.exists():
        pytest.skip("adapter not present")
    assert publish.sha16(p) == "67ea0f492938acfc"


# ---------------------------------------------------------------- rollout
class FakeRunner:
    """Records commands; each hist.sh call writes `per_call` ledger rows."""

    def __init__(self, ledger, run_ids, tasks, per_call):
        self.calls, self.ledger, self.rows, self.per_call = [], ledger, [(r, t) for r in run_ids for t in tasks], per_call

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw.get("env", {})))
        if cmd[0] == "bash":
            with open(self.ledger, "a") as fh:
                for r, t in self.rows[:self.per_call]:
                    fh.write(json.dumps({"run_id": r, "id": t}) + "\n")
            self.rows = self.rows[self.per_call:]
        return subprocess.CompletedProcess(cmd, 0, stdout="")


def test_rollout_env_command_and_resume(tmp_path):
    c = cfg(rollout={"spec": "", "docker_host": "unix:///tmp/x.sock", "concurrency": 64})
    ids = ["a", "b", "c"]
    run_ids = [f"pre_p{k}" for k in range(1, 5)]
    fr = FakeRunner(tmp_path / "l.jsonl", run_ids, ids, per_call=7)          # 12 expected: needs 2 attempts
    rr = rollout.run(c, tmp_path / "agent", "pre", ids, 4, tmp_path / "l.jsonl", tmp_path / "t.tasks", "lbl",
                     runner=fr)
    hist = [x for x in fr.calls if x[0][0] == "bash"]
    assert len(hist) == 2 and rr.run_ids == run_ids and rr.expected == 12
    cmd, env = hist[0]
    assert cmd[1:] == ["kits/swe-agent/experiments/hist.sh", str(tmp_path / "agent"), "pre", str(tmp_path / "t.tasks"),
                       "4", "12"]                                            # total capped at expected, multiple of 4
    assert env["LOG_PROXY_TITO"] == "1" and env["ROLLOUT_TAG"] == "1" and env["SPEC"] == ""
    assert env["DOCKER_HOST"] == "unix:///tmp/x.sock" and env["SERVE_AGENT"] == str(tmp_path / "agent")
    assert env["LEDGER"] == str(tmp_path / "l.jsonl") and env["SAMPLE_TAG"] == "p"
    assert (tmp_path / "t.tasks").read_text().split() == ids
    assert any(x[0][:2] == ["docker", "ps"] for x in fr.calls)               # leftover sandboxes checked


def test_rollout_skips_when_complete_and_ngram_unsets_spec(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEC", "stale")
    c = cfg(rollout={"spec": "ngram"})
    assert "SPEC" not in rollout.hist_env(c, tmp_path, tmp_path / "l", "x")
    fr = FakeRunner(tmp_path / "l.jsonl", ["pre_p1", "pre_p2", "pre_p3", "pre_p4"], ["a"], per_call=4)
    rollout.run(c, tmp_path, "pre", ["a"], 4, tmp_path / "l.jsonl", tmp_path / "t", "x", runner=fr)
    n = len(fr.calls)
    rollout.run(c, tmp_path, "pre", ["a"], 4, tmp_path / "l.jsonl", tmp_path / "t", "x", runner=fr)
    assert len(fr.calls) == n                                                # nothing re-run


def test_train_only_guard():
    import sys
    sys.path.insert(0, str(ROOT / "kits/swe-agent/run"))
    import manifest
    split = {r["id"]: r["split"] for r in manifest.rows()}
    hold = next(t for t, s in split.items() if s == "holdout")
    tune = next(t for t, s in split.items() if s == "tune")
    rollout.check_train_only([tune])
    with pytest.raises(ValueError, match="tune tasks only"):
        rollout.check_train_only([tune, hold])


# ---------------------------------------------------------------- gpu sharing
def test_wait_free_needs_two_quiet_checks():
    seq = iter([(True, 30000), (False, 1000), (False, 30000), (True, 30000), (False, 30000), (False, 30000)])
    state = {}

    def nxt():
        state["v"] = next(seq)

    logs = []
    gpu.wait_free(28000, poll_s=0, log=logs.append, running=lambda: (nxt(), state["v"][0])[1],
                  free=lambda: state["v"][1], sleep=lambda s: None)
    assert len(logs) == 5                       # quiet at polls 5 and 6: returns on the 6th, logs polls 1-5


def test_hold_is_removed_only_if_ours(tmp_path, monkeypatch):
    h = tmp_path / "hold"
    monkeypatch.setattr(gpu, "HOLD", h)
    with gpu.hold():
        assert h.exists()
    assert not h.exists()
    h.write_text("someone else")
    with gpu.hold():
        pass
    assert h.read_text() == "someone else"


# ---------------------------------------------------------------- loop
def stub_loop(tmp_path, fail=None, **over):
    c = cfg(out_root=str(tmp_path), steps=3, eval={"every": 2}, **over)
    calls = []

    def mk(phase):
        def f(k, sd):
            calls.append((k, phase))
            if fail and fail.pop((k, phase), None):
                raise RuntimeError(f"boom {k} {phase}")
            if phase == "train":
                fake_adapter(sd / "adapter_out", f"v{k + 1}".encode())
            return {"ok": phase}
        return f

    lp = L.Loop(c, log=lambda s: None, impl={p: mk(p) for p in L.PHASES})
    lp.prepare()
    return lp, calls


def test_loop_runs_phases_in_order_with_eval_schedule(tmp_path):
    lp, calls = stub_loop(tmp_path)
    assert lp.run() == 3
    assert calls == [(0, "agent"), (0, "eval"), (0, "rollout"), (0, "batch"), (0, "train"),
                     (1, "agent"), (1, "rollout"), (1, "batch"), (1, "train"),
                     (2, "agent"), (2, "eval"), (2, "rollout"), (2, "batch"), (2, "train")]
    m = [json.loads(line) for line in open(lp.dir / "metrics.jsonl")]
    assert [x["step"] for x in m] == [0, 1, 2]
    assert lp.policy_in(2) == lp.step_dir(1) / "adapter_out"
    assert lp.policy_in(0) == ROOT / "runs/sft/e2b_v1/adapter"


def test_loop_resumes_at_the_failed_phase(tmp_path):
    lp, calls = stub_loop(tmp_path, fail={(1, "batch"): True})
    with pytest.raises(RuntimeError):
        lp.run()
    calls.clear()
    lp.run()
    assert calls[:3] == [(1, "batch"), (1, "train"), (2, "agent")]      # step 0 and step 1's agent/rollout not redone


def test_loop_stop_file_and_max_steps(tmp_path):
    lp, calls = stub_loop(tmp_path)
    assert lp.run(max_steps=1) == 1
    (lp.dir / "STOP").write_text("")
    assert lp.run() == 0


def test_loop_refuses_changed_config(tmp_path):
    lp, _ = stub_loop(tmp_path)
    lp2 = L.Loop(cfg(out_root=str(tmp_path), steps=4, eval={"every": 2}), log=lambda s: None)
    with pytest.raises(SystemExit):
        lp2.prepare()
    lp2.prepare(force=True)


def test_train_without_signal_carries_the_policy_over(tmp_path):
    lp, _ = stub_loop(tmp_path)
    sd = lp.step_dir(1)
    sd.mkdir(parents=True)
    fake_adapter(lp.step_dir(0) / "adapter_out", b"v1")
    (lp.step_dir(0) / "optim.pt").write_bytes(b"opt")
    (sd / "batch.done").write_text(json.dumps({"segments": 0}))
    out = L.Loop._train(lp, 1, sd)
    assert out["skipped"] == "no_signal"
    assert (sd / "adapter_out" / "adapter_model.safetensors").read_bytes() == b"v1"
    assert (sd / "optim.pt").read_bytes() == b"opt"
