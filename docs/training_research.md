<!-- contract: research notes, written 2026-10-15 by a research subagent. Sources are linked inline.
"(unverified)" marks claims that no primary source confirmed. Figures under "est." are my arithmetic, not
measurements. Repo counts come from read-only queries to the HF datasets-server API (statistics/size/filter);
no dataset or model weights were downloaded. -->
# Training research: SWE task data, trajectories, Gemma 4 LoRA stack, agentic RL (as of 2026-10-15)

## 1. Executable SWE task datasets

| dataset | tasks | repos | docker images | tests | license | lang |
|---|---|---|---|---|---|---|
| [SWE-Gym/SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) ([arXiv 2412.21139](https://arxiv.org/abs/2412.21139), ICML'25) | 2,438 (Lite: 230) | 11 (pandas 737, moto 343, MONAI 374, mypy 257, dvc 225, dask 145, modin 107, pydantic 83, conan 75, hydra 66, bokeh 26) | per-instance `xingyaoww/sweb.eval.x86_64.*` on Docker Hub ([README](https://github.com/SWE-Gym/SWE-Gym)) | FAIL_TO_PASS / PASS_TO_PASS | MIT (card) | Python |
| [SWE-bench/SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith) ([arXiv 2504.21798](https://arxiv.org/abs/2504.21798), 2025-04-30) | 59,136 rows (card says 50k) | 222 repo@commit (card says 128 repos) | one image per repo snapshot (`image_name`, e.g. `jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536`) | F2P / P2P | MIT | Python. **Synthetic bugs** (LLM-injected / function rewrite / PR mirror), not real issues |
| [SWE-bench/SWE-smith-py](https://huggingface.co/datasets/SWE-bench/SWE-smith-py) (updated 2025-12-18) | 50,908 | 131 | same scheme | F2P / P2P | (per card, not checked) | Python. The card says the old mono-repo set is no longer updated (note dated 2025-12-14). There are sibling sets -go/-rs/-java/-ts/-cpp |
| [nebius/SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) ([arXiv 2505.20411](https://arxiv.org/abs/2505.20411)) | 21,336 (`test`) + 6,542 (`filtered`) | 3,400+ | ~7,500 prebuilt on Docker Hub under `swerebench` (announced 2025-08-05) | F2P / P2P (+F2F, P2F), `install_config`, `docker_image` | CC-BY-4.0 + each repo's license per row | Python. Real issues 2014-04 to 2025-04 |
| [nebius/SWE-rebench-V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2) ([arXiv 2602.23866](https://arxiv.org/abs/2602.23866), v1 2026-02-27, ICML'26; [blog 2026-03-03](https://nebius.com/blog/posts/meet-swe-rebench-v2)) | 32,079 (HF last modified 2026-10-01) | 3,617 | every task: `docker.io/swerebenchv2/<repo>:<PR>-<sha>` | F2P / P2P, `image_name` | CC-BY-4.0 | **20 languages**. Python count not stated; filter on the language field |
| [nebius/SWE-rebench-V2-PRs](https://huggingface.co/datasets/nebius/SWE-rebench-V2-PRs) | 126,300 | n/a | **none** (the PRs that could not be containerized) | n/a | CC-BY-4.0 | multi |
| [nebius/SWE-rebench-leaderboard](https://huggingface.co/datasets/nebius/SWE-rebench-leaderboard) | 860 in monthly splits (2025-01 to 2026-03) | n/a | `docker_image` + harbor resource fields | F2P / P2P | (not checked) | Python. A fresh, decontaminated **eval** source |
| [nebius/SWE-bench-extra](https://huggingface.co/datasets/nebius/SWE-bench-extra) | 6,376 | n/a | not stated (unverified) | F2P / P2P | (not checked) | Python. Predecessor of SWE-rebench |
| [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) ([arXiv 2504.07164](https://arxiv.org/abs/2504.07164), COLM'25) | Subset/Lite 4,578; [R2E-Gym-V1](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-V1) 8,101 | 13 | `namanjain12/<repo>_final:<commit>` (300-500 MB each) | **no F2P column**: `expected_output_json` maps test name to status | Apache-2.0 | Python. Commit-derived, backtranslated problem statements |
| [AweAI-Team/Scale-SWE](https://huggingface.co/datasets/AweAI-Team/Scale-SWE) ([arXiv 2602.09892](https://arxiv.org/abs/2602.09892), HF 2026-02-26) | 20,181 | 5,200 | `image_url` per instance (registry not named) | F2P / P2P, `f2p_script`, `pre_commands` | CC-BY-4.0 | has a `language` column (mix not checked) |
| AweAI-Team/DeNovoSWE (2026-06-11, [repo](https://github.com/AweAI-Team)) | size not shown by the API | n/a | n/a | n/a | n/a | "long-horizon SWE for SFT & RL" (unverified details) |
| [GAIR/OpenSWE](https://huggingface.co/datasets/GAIR/OpenSWE) (daVinci-Env, [arXiv 2603.13023](https://arxiv.org/abs/2603.13023), 2026-03-13) | 45,320 envs, ~9k quality-filtered | 12.8k | Dockerfiles + eval scripts; images only for permissive repos | not stated | mixed: card says "permissive + CC-BY-4.0", text says AGPL-3.0 | Python. **Gated** |
| SWE-Lego ([arXiv 2601.01426](https://arxiv.org/abs/2601.01426), [repo](https://github.com/SWE-Lego/SWE-Lego)) | 32,119 (18,409 real + 13,710 synthetic) | n/a | n/a | n/a | Apache-2.0 | HF dataset names have changed (renamed); look them up again |
| [princeton-nlp/SWE-bench](https://huggingface.co/datasets/princeton-nlp/SWE-bench) `train` | 19,008 | 35 | **none**. Not execution-validated per the SWE-bench paper; whether F2P is populated is unverified | n/a | n/a | Python. Use only as issue/patch text |

**Contamination with our local holdout (fastapi / rich / requests / httpx).** Checked 2026-10-15 via the HF statistics/filter API:
- **SWE-bench `train`: `tiangolo/fastapi` has 28 instances.** Exclude them.
- **SWE-bench `test`: `psf/requests` has 44 instances.** This is the source of SWE-bench Verified/Lite requests tasks. Any model or trajectory set tuned on SWE-bench test, and any "SWE-bench-Verified" trajectories, may contain requests fixes.
- **Clean by repo name:** SWE-Gym and Lite (the 11 repos above); SWE-smith (222 repo snapshots; the only "rich" substring hit is `kyoh86/richgo`, which is unrelated); SWE-smith-py (131 repos).
- **SWE-rebench `test`:** an exact-name filter returned 0 rows for `tiangolo/fastapi` and 0 for `Textualize/rich`. `fastapi/fastapi`, `psf/requests` and `encode/httpx` are **unverified**: the datasets-server index for that table timed out repeatedly.
- **Not checked** (repo is a free-text column and the filter API kept timing out): SWE-rebench V2, Scale-SWE, R2E-Gym, nebius OpenHands trajectories, SWE-Hero. Before training, run a repo/instance_id filter locally on whatever subset is pulled, and drop any repo matching `fastapi|rich|requests|httpx` plus forks. Note that the hidden test repo is private, so this only protects the local holdout's validity.

## 2. Open SWE agent trajectory datasets

| dataset | size | producer / scaffold | format | license |
|---|---|---|---|---|
| [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | 76,002 rows in splits `tool` 24.1k / `xml` 26.1k / `ticks` 25.8k; 5,017 resolved used for SWE-agent-LM-32B | Claude 3.7 Sonnet + SWE-agent | `messages` (JSON string), `resolved`, `patch`. The `tool` split uses function calls (bash, str_replace_editor, submit) | MIT |
| [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) (Dec 2025) | 67,074 traj, 32,161 resolved, 3,792 resolved issues, 1,823 repos; avg ~64 turns, up to 131k ctx | Qwen3-Coder-480B-A35B-Instruct + OpenHands v0.54.0 | OpenAI-style messages with `tool_calls` (arguments as a JSON string), a `tools` column, `resolved`, joinable to SWE-rebench docker images | CC-BY-4.0 |
| [nvidia/SWE-Hero-openhands-trajectories](https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories) ([arXiv 2604.01496](https://arxiv.org/abs/2604.01496), 2026-04) | 34,269 traj / 11,766 issues (execution-verified). SWE-Zero sibling: ~300k execution-free | Qwen3-Coder-480B + OpenHands | system/user/assistant/tool roles; issues from SWE-Gym, R2E-Gym-Subset, SWE-rebench | CC-BY-4.0 |
| [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) (2024-12) | 80,036 (13,389 resolved, ~17%) | SWE-agent + Llama-based models (e.g. `swe-agent-llama-70b`) | trajectory + eval_logs; old SWE-agent 100-line windowed editor | CC-BY-4.0 **plus the Llama 3.1 license** |
| [AweAI-Team/Scale-SWE-Distilled](https://huggingface.co/datasets/AweAI-Team/Scale-SWE-Distilled) | 71,498 (DeepSeek v3.2); also a 41k DeepSeek-v4-Pro-High set (2026-06-11) | AweAgent | (not inspected) | CC-BY-4.0 (repo) |
| [SWE-Gym/OpenHands-Sampled-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories) / [-SFT-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories) | 6,055 raw (with `resolved`, `tools`) / 491 successful | GPT-4o + Claude 3.5 Sonnet, OpenHands | messages + tools | (not checked) |
| [R2E-Gym/R2EGym-SFT-Trajectories](https://huggingface.co/datasets/R2E-Gym/R2EGym-SFT-Trajectories) | 3,231 | Claude 3.5 Sonnet, R2E-Gym tools | messages | Apache-2.0 (repo) |
| SWE-Lego trajectories / OpenSWE (~13k, gated) / DeNovoSWE-Trajectory-{Raw,Filtered} (2026-06-14) | 18k validated (SWE-Lego) | OpenHands 0.53 (SWE-Lego) | not inspected | see §1 |

**Conversion to our tools (run_command, read_file, edit_file, write_file, submit_patch, ...): yes, every set needs it.**
- The OpenHands and SWE-agent tool sets map fairly directly:
  - `execute_bash`/`bash` → run_command
  - `str_replace_editor view` → read_file
  - `str_replace_editor create` → write_file
  - `str_replace_editor str_replace`/`insert` → edit_file
  - `finish`/`submit` → submit_patch
  - `think` → drop it, or fold it into the thinking channel
- Hard filters:
  - (a) ≤32,768 tokens **under the Gemma 4 tokenizer**. Nebius trajectories average 64 turns and reach 131k, so most long ones drop out.
  - (b) A turn count and output length that fit the ~6 min/task budget (~30 tok/s on 31B; see memory note "Kaggle time formula").
  - (c) Re-render tool observations in our own tools' output format, or the model learns observation shapes it will never see.
- Off-policy distillation from Qwen3-Coder or Claude teaches *their* habits (long explorations). Our own on-policy rejection-sampled Gemma 4 trajectories are the most faithful SFT data; use converted sets to widen task coverage.
- License flags: nebius SWE-agent-trajectories carries the Llama 3.1 license; check the competition's external-data rules and each generator's terms before training on teacher outputs.

## 3. Gemma 4 LoRA training stack

**Facts about the served model** (local `data/external/model*/config.json` + [HF card](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct)):

| | 31B | 12B |
|---|---|---|
| architecture | `Gemma4ForConditionalGeneration` | **`Gemma4UnifiedForConditionalGeneration` (`gemma4_unified`)**, a different class (forum 2026-10-11/12: vLLM 0.19.1 needed a wheelhouse fix) |
| hidden size | 5376 | 3840 |
| layers | 60 (50 sliding / 10 full) | 48 (40 / 8) |
| heads | 32 heads, 16 KV heads (head_dim 256) | 16 heads, 8 KV heads |
| global layers | head_dim 512, 4 KV heads | 1 KV head |
| MLP intermediate | 21504 | 15360 |
| vocab | 262,144 | 262,144 |
| sliding window | 1024 | 1024 |

- `attention_k_eq_v: true` on both.
- Quantization: compressed-tensors `pack-quantized`, int4 symmetric, **group_size 32**, targets `Linear`, vision tower ignored (the 12B also ignores `lm_head`).
- The scorer uses vLLM **0.19.1** (forum). Sampling defaults: T=1.0, top_p=0.95, top_k=64.

**Library support:**
- **Unsloth**: [Gemma 4 guide](https://unsloth.ai/docs/models/gemma-4/train) (screenshots dated 2026-04-07).
  - Lists E2B, E4B, 26B-A4B and 31B; 12B is mentioned once (support for `gemma4_unified` is unverified).
  - "31B QLoRA works on 22GB". Its example uses `max_seq_length=8192`.
  - Templates are `gemma-4` and `gemma-4-thinking`; the guide recommends the thinking template for 26B/31B.
  - Known bugs: 31B inference IndexError (`num_kv_shared_layers=0`); E2B/E4B produce garbage with `use_cache=False`.
- **TRL**:
  - HF ships a `tiny-Gemma4ForConditionalGeneration` test model, and community Gemma 4 fine-tunes cite TRL 1.3 to 1.7 with transformers 5.7 to 5.13. No official TRL release note was found (unverified).
  - GRPO multi-turn via `environment_factory` (per [PyTorch Conf EU 2026 slides](https://hosted-files.sched.co/pytorchconferenceeu2026/c3/From%20Responses%20To%20Trajectories_%20Multi-Turn%20and%20Multi-Environment%20Reinforcement%20Learning.pdf); exact API unverified).
- **Axolotl**: README news "2026/04: Gemma 4 support" ([docs](https://docs.axolotl.ai/docs/models/gemma4.html)). Later news adds NVFP4 LoRA (2026/09) and Tinker-compatible remote training (2026/06).
- **LLaMA-Factory**: [v0.9.5](https://github.com/hiyouga/LlamaFactory/releases/tag/v0.9.5) "Qwen3.5/3.6, Gemma 4, Transformers v5". The 2026-05-30 date comes from third parties.
- **torchtune**: [README](https://github.com/pytorch/torchtune) says "no longer actively maintained… wound down in 2025". It has no Gemma 3/4 support. Do not use it.

**Training a bf16 LoRA vs. serving on the int4 QAT checkpoint:**
- Google publishes `google/gemma-4-{E2B,E4B,12B,31B}-it-qat-q4_0-unquantized`: bf16 "half-precision weights extracted from the QAT pipeline" ([docs](https://ai.google.dev/gemma/docs/core)).
- Asked "which base weights should adapters be trained against", the host replied (2026-10-11): "The unquantized variants are on the same page: Gemma 4". That means the QAT-unquantized weights, though it is ambiguous whether he meant those or plain `-it`.
- vLLM applies bf16 LoRA deltas on top of the quantized base layer, as do all mainstream engines. No source tests compressed-tensors W4A16 + LoRA for Gemma 4 specifically. The competition does support adapters, so it should load (still verify).
- My ranking, by reasoning rather than measurement:
  1. **bf16 dequantization of the exact `-qat-w4a16-ct` weights.** The adapter then sees exactly the frozen W the scorer uses. Transformers can decompress compressed-tensors (`run_compressed=False`); for Gemma 4 this is unverified.
  2. **`-qat-q4_0-unquantized` bf16.** Very close if round(W_unq) to an int4 grid with group 32 equals the CT weights (unverified; one tensor diff would settle it).
  3. **QLoRA (bnb NF4) on (2).** Cheapest in memory, but trains against NF4 error that is absent at serve time.
  4. Avoid the plain `gemma-4-*-it` bf16 weights: QAT weights are a further-trained variant.
- Whatever the choice, A/B the adapter on the real stack (vLLM 0.19.1, w4a16, `--enable-lora`) and check the startup log for skipped modules.
- vLLM attaches LoRA **only to the language model** of multimodal models.
- Because of `k_eq_v`, the global layers may not have a separate `v_proj`, so check `target_modules` against the checkpoint's module names.
- **Adapter size vs. the 3 GiB archive** (est., all 7 linear projections): 31B ≈ 7.7M params per rank, so r=64 ≈ 0.99 GB, r=128 ≈ 1.98 GB, r=256 ≈ 3.96 GB (**does not fit**). 12B ≈ 4.1M per rank, so r=256 ≈ 2.1 GB. Multiple adapters add up.

**Chat template for SFT** (local `chat_template.jinja`; [Google format doc](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)):
- Turns are `<|turn>system|user|model … <turn|>`.
- Tool declarations use `<|tool>declaration:…<tool|>`. A call is `<|tool_call>call:NAME{key:<|"|>str<|"|>,…}<tool_call|>`.
- The result is `<|tool_response>response:NAME{…}<tool_response|>` and is rendered **inside the same model turn**: there is no separate tool turn, and `<|tool_response>` is a stop sequence.
- Thinking: `<|think|>` goes in the system turn; reasoning sits in `<|channel>thought … <channel|>`. When thinking is off, the generation prompt appends an empty `<|channel>thought\n<channel|>`.
- In history, earlier thoughts are stripped, except between tool calls inside one model turn (the template keeps reasoning only after the last user message when there are tool_calls).
- **The template has no `{% generation %}` tags**, so TRL `assistant_only_loss` will not mask correctly. Write a custom mask that trains on call/answer tokens and masks the `<|tool_response>…<tool_response|>` spans that sit inside model turns.
- Render with the **same template vLLM 0.19.1 uses**, so the `gemma4` tool parser sees identical syntax.

**Memory on one 32 GB RTX 5090** (est. from the configs; unverified until run):
- Base weights (Google table, +20% overhead):
  - 31B: bf16 69.9 GB, Q4_0 17.5 GB
  - 12B: bf16 26.7 GB, Q4_0 6.7 GB
- Gradient-checkpoint activations (seq × hidden × 2 B × layers):
  - 31B: 10.6 GB at 16k, 21 GB at 32k
  - 12B: 6 GB at 16k, 12 GB at 32k
  - Unsloth offloads these to CPU RAM.
- Logits: 262k vocab × seq × 2 B = 8.6 GB (16k) or 17 GB (32k). This needs a fused or chunked cross-entropy (Unsloth or Liger).
- Global layers use **head_dim 512**, which exceeds FlashAttention-2's 256 limit. Long-sequence training therefore depends on SDPA mem-efficient/cuDNN or FA3/FA4 support. Eager attention at 32k would need ~68 GB for one layer's scores.
- Verdicts:
  - **12B QLoRA at 16k–32k:** plausible on the 5090.
  - **12B bf16 LoRA:** tight at 16k, unlikely at 32k.
  - **31B QLoRA at 16k:** borderline; it needs offloaded checkpointing plus fused CE.
  - **31B at 32k, or any 31B bf16 LoRA:** rent hardware.
- Rental: 1× H100 80 GB is enough for 31B QLoRA at 32k; 1× H200 141 GB or B200 for 31B bf16 LoRA. RunPod pricing was not checked.
- The 5090 (sm_120) needs CUDA ≥12.8 builds of torch, bitsandbytes and flash-attn (version specifics unverified).

## 4. Agentic RL frameworks for SWE with docker sandboxes

| framework | multi-turn tools | SWE/docker | LoRA-only RL | Gemma | reported hardware |
|---|---|---|---|---|---|
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) (NovaSky/Berkeley; v0.3.0 2026-07-16) | yes (SkyRL-Agent 2025-11-26, [arXiv 2511.16108](https://arxiv.org/abs/2511.16108)) | SkyRL-v0 (2025-05-06): VeRL + OpenHands on SWE-Gym, ~300 tasks; [Harbor](https://github.com/NovaSky-AI/SkyRL) integration 2026-02-17 (container tasks) | via its Tinker-API backend skyrl-tx (LoRA + multi-LoRA); LoRA in the core FSDP path unverified | not listed (unverified) | SkyRL-Agent-14B-v0: 20 h on 8×H200 (third-party summary, unverified). SA-SWE-32B (Qwen3-32B) 39.4% Verified |
| [rLLM / DeepSWE](https://github.com/rllm-org/rllm) | yes | DeepSWE-Preview: Qwen3-32B, pure RL on 4,500 R2E-Gym tasks; sandboxes docker/Daytona/Modal | backends verl / tinker / fireworks; LoRA not stated | not listed | **64×H100 for 6 days** ([Together blog, 2025-07-02](https://together.ai/blog/deepswe)); 42.2% Pass@1 |
| [verl](https://verl.readthedocs.io/en/latest/advance/agent_loop.html) | `AgentLoopBase` (since 0.4.2, alpha): you implement `run()` and the tools yourself; vLLM/SGLang rollouts | bring your own sandbox | LoRA RL supported per docs and skill summaries (details unverified) | follows its transformers/vLLM pins (unverified for Gemma 4) | multi-GPU |
| [verl-agent](https://github.com/langfengQ/verl-agent) | yes (GiGPO/GRPO etc.) | no SWE env shipped | yes | n/a | n/a |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | `--train.agent_func_path` (`AgentInstanceBase` reset/step), async, partial rollout; FlashREINFORCE (2026-09) | no sandbox | LoRA/QLoRA (DeepSpeed); adapters must be merged to deploy | not mentioned | examples assume 8 GPUs (8×A100 80 GB tip); vLLM 0.29.0 recommended |
| [ART (OpenPipe)](https://github.com/OpenPipe/ART) | yes (trajectory-level GRPO, RULER judge) | none built in (pair with e.g. AgentBox) | **LoRA-only by design** (vLLM serves the latest LoRA; trains via Unsloth) | README: "Gemma 3 does not appear to be supported"; Gemma 4 unverified | 1 GPU server + client anywhere |
| TRL GRPOTrainer | `environment_factory` / `rollout_func` (see §3) | bring your own | PEFT LoRA + vLLM | Gemma 4 SFT works per community; GRPO on Gemma 4 unverified | 1 GPU possible for small models |
| [Modal SWE-Gym example](https://modal.com/docs/examples/swe_gym) | bash-tool agent | Modal Sandboxes, reward = hidden tests pass | rank-32 LoRA per job, Tinker-compatible API | Qwen3.5-9B used | 31 SWE-Gym issues, 32 groups × 8 attempts per batch. A **good template for a small LoRA SWE-RL loop** |

Also relevant:
- NVIDIA ProRL Agent ([arXiv 2603.18815](https://arxiv.org/abs/2603.18815)): rollout-as-a-service with sandboxes, trainer-agnostic.
- ms-swift: multi-turn GRPO.
- SWE-MiniSandbox ([arXiv 2602.11210](https://arxiv.org/abs/2602.11210)): container-free isolation, which cuts the docker cost.
- SWE-World ([arXiv 2602.03419](https://arxiv.org/abs/2602.03419)): a learned execution surrogate.

None of the frameworks above document Gemma 4. For on-policy RL, the rollout engine should match the scorer: Gemma 4 w4a16 + LoRA under vLLM 0.19.1 with the gemma4 tool parser. At ~30 tok/s per 31B stream and minutes per rollout, RL compute is dominated by rollouts. 12B (about 1.9× faster) is the cheaper RL target.

## Recommended first steps
1. **Decontaminate first.** Drop any repo or instance matching `fastapi|rich|requests|httpx` (including SWE-bench-train's 28 fastapi tasks) from every pool. Re-check the unverified SWE-rebench/V2 counts locally on the rows you actually pull.
2. **Task pool v1 (Python, prebuilt images, F2P/P2P):**
   - SWE-rebench `filtered` (6.5k) and `test` (21k) via the `swerebench` Docker Hub images
   - SWE-Gym (2.4k)
   - SWE-rebench-V2 Python rows
   - SWE-smith-py (51k, synthetic) as cheap filler
   - Prefer small pure-Python library repos with fast pytest suites (closest to the hidden private repo's shape: issue → patch → pytest).
3. **Rejection-sample with our own harness** (same tools and prompts as the submission; Gemma 4 w4a16 on vLLM 0.19.1) over a few hundred to a few thousand of those tasks. Keep the resolved, ≤32k-token trajectories that fit the time budget. This becomes the core SFT set. Add converted nebius/SWE-Hero OpenHands trajectories only after an A/B shows they help.
4. **De-risk serving before any big training.**
   - Train a throwaway r=8 adapter for a few steps against the chosen base: dequantized CT, or `-qat-q4_0-unquantized` after diffing one tensor against the CT weights.
   - Package it the way the competition does and serve it under vLLM 0.19.1 + w4a16 with `--enable-lora`.
   - Confirm outputs change, no modules are skipped, and the gemma4 tool parser still parses.
5. **SFT on the 5090:**
   - Unsloth QLoRA on 12B (or 31B at ≤16k) with a custom loss mask for inline `<|tool_response>` spans.
   - Rent 1× H100/H200 for 31B at 32k.
   - Pick the LoRA rank with the 3 GiB archive cap in mind (31B r ≤ 128 for a single adapter).
6. **RL later:** start from the Modal SWE-Gym pattern (GRPO, LoRA r=32, 8 attempts/task, binary test reward) on TRL or ART-style LoRA-only loops. Move to SkyRL + Harbor or rLLM if scale is needed. Budget for rented multi-GPU nodes; public SWE-RL runs used 8×H200 to 64×H100.
