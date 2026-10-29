<!-- training/rl/README.md contract
WHAT: how the RL code is laid out, the file contracts between stages, and how to run, test and extend it.
Facts and paths only; measured numbers carry their date. Results of runs go to journal.md and docs/, not here.
-->
# RL for the Gemma 4 agent (`training/rl/`)

On-policy RL with a LoRA adapter, rolled out on our own harness and the prod serving stack (vLLM 0.19.1, int4
base + adapter, gemma4 parsers, Kaggle clock). The RL maths (advantage estimators, policy losses, loss aggregation)
comes from verl (`verl/trainer/ppo/core_algos.py`, pinned commit in `pyproject.toml`); we write the plumbing.

## One step

| phase | module | reads | writes |
|---|---|---|---|
| agent | `publish.py` | base agent folder, policy adapter vk | `step_k/agent/` (adapter vk as `main_lora`), its sha |
| eval (every `eval.every`) | `rollout.py` | holdout tasks | ledger rows `rl_<name>_e<k>_p*` |
| rollout | `rollout.py` -> `hist.sh` | pool tasks x `tries` | TITO capture `runs/rl_<name>_s<k>/requests.jsonl.gz`, ledger rows |
| batch | `rollouts.py`, `reward.py`, `advantage.py`, `segments.py`, `batch.py` | capture + ledger | `step_k/batch.json.gz` |
| train | `train_step.py` (own process) | batch, vk, optimizer state | `step_k/adapter_out/` = v(k+1), `optim.pt`, `train_metrics.json` |

`loop.py` runs the phases; each one writes its artifacts and then `step_k/<phase>.done`, so a killed loop resumes
at the first missing marker. Run folder: `runs/rl/<name>/` (`config.yaml`, `ledger.jsonl`, `metrics.jsonl`, steps).

## The pieces and their rules

- **Capture (token in, token out).** `kits/swe-agent/experiments/log_proxy.py` with `LOG_PROXY_TITO=1` asks vLLM
  for `return_token_ids` + `logprobs` and keeps the exact prompt ids, sampled ids (stop token included) and vLLM's
  raw log-probs (temperature 1) under `"tito"`; the harness gets the same reply as without it. Nothing is
  re-tokenized.
- **Try tag.** `run_tasks.py` with `ROLLOUT_TAG=1` sends `X-Rollout-Id: <run_id>/<task>` on every model call
  (`kits/swe-agent/run/rollout_tag.py`); the proxy records it and does not forward it.
- **Reward** (`reward.py`): 1 if solved inside the Kaggle clock, 0 for a fail, cut or policy ending (no submit,
  budgets, context overflow); infra endings and unknown error texts are excluded and counted.
- **Advantage** (`advantage.py`): verl estimators that give one number per try: `rloo`, `grpo` (+ `_vectorized`).
  A group with fewer than `min_group_size` valid tries gets 0.
- **Segments** (`segments.py`): a try is cut where the history really changed (nudge: template drops old thoughts).
  The template's extra newline before `<channel|>` is spliced (sampled id kept). Measured on tito_smoke1
  (2026-10-26): trainer vs vLLM mean |dlogp| 0.018 on the first call, 0.006 on later calls with 9 spliced newlines.
- **Loss** (`losses.py`, `train_step.py`): verl policy loss per segment with the step's global counts
  (`global_batch_info`), gradient accumulation, one optimizer step per epoch. B counts tries, not segments.
  `logprob.py` gets log pi from the final hidden state with liger (softcap 30, temperature folded in).
- **On-policy guard**: every ledger row pins the served adapter's sha; the batch refuses rows from another adapter,
  and `train_step` refuses a batch whose sha is not the adapter it loads.
- **GPU sharing** (`gpu.py`): the loop touches `/tmp/gpu_hold` (the screen pipelines wait on it) and, before
  starting a server, waits until no vLLM runs and memory is free (`hist.sh` kills any server on port 8000).

## Switching algorithm (config only)

| want | set |
|---|---|
| GRPO advantages | `algo.adv_estimator: grpo`, `algo.norm_adv_by_std: true/false` |
| reuse each batch (PPO) | `algo.ppo_epochs: 2-4`; old log-probs come from the first pass, clipping now bites |
| GSPO / other verl losses | `algo.policy_loss: gspo` (sequence ratio is per segment) |
| token-mean aggregation | `algo.loss_agg_mode: token-mean` (seq-mean-token-mean is refused: segments are not tries) |

Not built yet: mini-batches inside an epoch, async rollouts / truncated IS against vLLM log-probs (they are raw,
temperature 1; vLLM's processed mode would switch its top-k/top-p sampler from FlashInfer to PyTorch), runtime
adapter reload instead of a server restart per step, prefix sharing across segments.

## Machines

The loop, vLLM and the trainer share one GPU machine and take turns (the server stops after each rollout).
Sandboxes go wherever `rollout.docker_host` points, e.g. an SSH-tunnelled Docker socket on a sandbox VM; that host
needs the sandbox image `swebench-sandbox-local` and the task snapshots the tune tasks use.

## Run and test

```bash
uv run python -m training.rl.loop --config training/configs/rl_e2b_v1.yaml [--set k=v ...] [--max-steps N]
touch runs/rl/<name>/STOP                     # stop after the current step
uv run pytest -q training/rl/tests            # CPU: reward, advantage, segments, batch, losses, train core, loop
RL_GPU_TESTS=1 uv run pytest -q training/rl/tests/test_gpu.py   # liger grads, scorer vs fp32, drift, 32k memory
```
CPU tests use synthetic ids and fake ledger rows only. `test_harness_tag.py` runs the real ADK client and proxy
(envs/harness) against a fake vLLM.

## Parked: common industry additions (noted 2026-10-27; decision: run vanilla first)

The core (group tries, RLOO/GRPO baseline, binary test reward, collect-then-update, Dr. GRPO token normalisation,
loss on model tokens only) is the standard recipe. Not built, kept for later, in rough value order:

| addition | what | why later |
|---|---|---|
| pool by pass rate (curriculum) | keep tasks solved sometimes (0 < p < 1), refresh as the model improves | needs a multi-try screen; pass 1 is 1 try/task |
| sample tasks per step | draw e.g. 32 tasks x 8 tries per step from a large pool (today: whole pool every step) | needed once the pool is >> 24 tasks |
| dynamic sampling (DAPO) | drop all-0 / all-1 groups and resample to keep the batch full | today they get advantage 0: wasted compute, no harm |
| PPO mini-batches | 2-4 updates per batch with clipping (`algo.ppo_epochs` exists; mini-batches inside an epoch do not) | sample efficiency |
| async rollouts + truncated IS | generate and train at the same time on separate GPUs | one GPU today |
| runtime adapter reload | push the new LoRA into the running vLLM instead of a restart per step | saves minutes per step |
| KL penalty to a reference | keep the policy near a fixed reference model (details below) | tests are the reward (hard to game); LoRA + lr 5e-6 + 20 steps; holdout eval every 5 steps watches drift |
| trainer-vs-vLLM correction | weight each token by capped p_vLLM / p_trainer (truncated IS); today only logged (`model.log_mismatch`) | add if the logged gap is large |

### KL penalty: how to add it (noted 2026-10-28, not built)

The maths is verl's: `kl_penalty(logprob, ref_logprob, kl_penalty="low_var_kl")` (k3) and `agg_loss` in
`verl/trainer/ppo/core_algos.py` (verl 0.10). Our part (~half a day with tests):
1. Reference log-probs: per segment, one extra forward through the same liger path (`logprob.py`) under
   `with model.disable_adapter(), torch.no_grad():`. No extra model copy. Adds ~25% to the train step, 0 to rollouts.
2. Loss: in `losses.segment_loss`, `loss += kl_coef * agg_loss(kl_penalty(lp, ref_lp, "low_var_kl"), mask, ...)` with
   the same `loss_agg_mode` and global batch counts as the policy loss. Gradient flows through `lp` only.
3. Config `algo.use_kl_loss`, `algo.kl_coef` (GRPO used 0.04; 0.001-0.01 is common for agents), `algo.kl_type`
   (default `"k3+"`, see below); log mean KL per step.
4. Test: from the zero-init adapter (`init_adapter.py`) the policy equals the reference, so step-0 KL must be 0.
- KL in the loss, not in the reward: our advantages are computed at batch build, before the trainer runs, so a
  reward-side KL would need a scoring pass before advantages.
- Estimator: as a loss on tokens sampled from the policy, k3's expected value is KL(pi || ref) but its expected
  gradient is the gradient of the forward KL(ref || pi) (= pi - ref per logit); k2's expected gradient is the exact
  reverse-KL gradient and k1's is 0 (checked with autograd on a 6-token example, 2026-10-28). verl's
  `kl_penalty(..., "k3+")` returns the k3 value with k2's gradient: use it, so the logged KL and the trained KL match.
- Reference when starting from SFT: `disable_adapter()` gives the base model, not SFT. Load a second frozen copy of
  the SFT adapter (`model.load_adapter(path, adapter_name="ref")`, switch with `set_adapter`) and score with it.
