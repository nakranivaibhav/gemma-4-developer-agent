"""RL run configuration: one YAML file, nested sections, unknown keys are an error (a typo must not silently
fall back to a default). The resolved config is written into every step folder, so a step can be re-run or audited
exactly as it ran."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Rollout:
    tasks: str                        # manifest selector, task id(s), or a file of ids (kits/swe-agent/run/manifest.py)
    tries: int = 8                    # tries per task = group size for the advantage
    concurrency: int = 32             # tries at once on the server (hist.sh TOTAL); must be a multiple of tries
    spec: str = ""                    # vLLM speculative decoding: "" = plain decoding; "ngram" = serve_fast.sh default
    docker_host: str | None = None    # sandbox Docker host (e.g. unix socket of an SSH tunnel); None = local Docker
    need_gpu_mb: int = 28500          # free GPU memory to wait for before starting the server


@dataclass(frozen=True)
class Algo:
    adv_estimator: str = "rloo"       # verl estimator name; see advantage.SUPPORTED
    norm_adv_by_std: bool = False     # GRPO only: divide by the group std (Dr. GRPO says no)
    min_group_size: int = 2           # groups with fewer valid tries get advantage 0 (no baseline)
    policy_loss: str = "vanilla"      # verl policy loss name (vanilla = PPO clip; = REINFORCE when ratio is 1)
    loss_agg_mode: str = "seq-mean-token-sum-norm"   # verl agg_loss mode; see losses.ALLOWED_AGG
    loss_scale_factor: float = 4096.0  # C: fixed divisor of the summed token loss
    count_zero_adv_tries: bool = True  # B (and token counts) include tries with advantage 0
    ppo_epochs: int = 1               # >1: reuse the batch; old log-probs come from the first pass
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    clip_ratio_c: float = 3.0


@dataclass(frozen=True)
class Optim:
    lr: float = 2.0e-6
    warmup_steps: int = 0
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    eps: float = 1.0e-8
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class Model:
    model_dir: str = "data/external/models/gemma-4-e2b-it-qat-w4a16-ct"
    logit_softcap: float = 30.0       # Gemma 4 final_logit_softcapping
    max_train_tokens: int = 32768     # a try with a longer segment is not trained (still counted in B)
    log_mismatch: bool = True         # trainer vs vLLM log-prob gap at temperature 1 (vLLM logprobs are raw)
    offload_ple: bool = False         # E4B: per-layer embeddings on CPU (see sft_train.py)


@dataclass(frozen=True)
class Eval:
    every: int = 5                    # holdout eval every N steps (0 = never)
    tasks: str = "holdout"
    tries: int = 3


@dataclass(frozen=True)
class RLConfig:
    name: str
    init_adapter: str                 # adapter folder of the step-0 policy (peft format)
    base_agent: str                   # agent folder copied for each step; its adapters/main_lora is replaced
    rollout: Rollout
    algo: Algo = field(default_factory=Algo)
    optim: Optim = field(default_factory=Optim)
    model: Model = field(default_factory=Model)
    eval: Eval = field(default_factory=Eval)
    steps: int = 30
    out_root: str = "runs/rl"
    gpu_hold: bool = True             # touch /tmp/gpu_hold while running (the screen pipelines wait on it)
    wandb: dict | None = None         # {"entity": ..., "project": ...}; None = no W&B

    @property
    def run_dir(self) -> Path:
        return ROOT / self.out_root / self.name

    @property
    def temperature(self) -> float:
        """The rollout temperature, read from the agent's own sampling config so the two can never disagree."""
        return float(yaml.safe_load(open(ROOT / self.base_agent / "configs" / "sampling.yaml"))["temperature"])

    def to_dict(self) -> dict:
        import json
        return json.loads(json.dumps(dataclasses.asdict(self)))   # plain lists/dicts: safe to dump and reload


_SECTIONS = {"rollout": Rollout, "algo": Algo, "optim": Optim, "model": Model, "eval": Eval}


def _build(cls, d: dict, where: str):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {sorted(unknown)}")
    kw = {}
    for k, v in d.items():
        if k in _SECTIONS and cls is RLConfig:
            v = _build(_SECTIONS[k], v or {}, f"{where}.{k}")
        elif isinstance(v, list):
            v = tuple(v)
        kw[k] = v
    return cls(**kw)


def from_dict(d: dict) -> RLConfig:
    cfg = _build(RLConfig, d, "config")
    validate(cfg)
    return cfg


def load(path: str | Path, overrides: list[str] = ()) -> RLConfig:
    d = yaml.safe_load(open(path))
    for kv in overrides:                         # dotted key=value, value parsed as YAML
        key, val = kv.split("=", 1)
        node = d
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return from_dict(d)


def validate(cfg: RLConfig) -> None:
    from . import advantage, losses
    r, a = cfg.rollout, cfg.algo
    if r.tries < 2:
        raise ValueError("rollout.tries must be >= 2: the advantage needs a group baseline")
    if r.concurrency % r.tries:
        raise ValueError(f"rollout.concurrency ({r.concurrency}) must be a multiple of tries ({r.tries})")
    if a.adv_estimator not in advantage.SUPPORTED:
        raise ValueError(f"algo.adv_estimator {a.adv_estimator!r} not in {sorted(advantage.SUPPORTED)}")
    if a.loss_agg_mode not in losses.ALLOWED_AGG:
        raise ValueError(f"algo.loss_agg_mode {a.loss_agg_mode!r} not in {sorted(losses.ALLOWED_AGG)}")
    if a.ppo_epochs < 1 or a.min_group_size < 2 or a.loss_scale_factor <= 0:
        raise ValueError("algo: ppo_epochs >= 1, min_group_size >= 2, loss_scale_factor > 0")
    if cfg.optim.lr <= 0 or cfg.steps < 1:
        raise ValueError("optim.lr > 0 and steps >= 1")
