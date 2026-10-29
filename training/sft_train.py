"""LoRA SFT on a Gemma 4 int4 (compressed-tensors) checkpoint, from packed sequences.

usage:
  uv run training/sft_train.py --config training/configs/sft_e2b_v1.yaml
  uv run training/sft_train.py --config ... --max-steps 2 --no-wandb      # smoke test
  uv run training/sft_train.py --config ... --set train.lr=5e-5 run_name=foo

Input: packed.jsonl from training/sft_data.py, one row per sequence: {"text", "spans": [[a, b], ...]}.
Loss is on tokens [a, b) of each span only (the model's own thought + call + stop token).

What is non-standard, and why:
- Base weights: the int4 checkpoint is unpacked to bf16 on load (run_compressed=False). These are the
  exact values the scorer's int4 kernel computes with, so the adapter trains against the served base.
  Ship the adapter separately; never merge it into the int4 weights.
- Attention: E2B's global layers have head_dim 512 and 1 KV head. With GQA on, SDPA has no fused
  kernel for that and falls back to "math", which builds a 31 GB score matrix at 32k. We turn GQA off
  (transformers repeats the KV head) and force the memory-efficient SDPA kernel. Flex attention does
  not compile on the 5090 (needs 200 KB shared memory per block, card has 101 KB).
- Loss: the 262k vocab makes full logits ~34 GB at 32k, so liger's fused linear cross-entropy
  computes it in chunks from the final hidden state (checked: 0.7685 vs the model's own 0.7691).
- Trainer.model_accepts_loss_kwargs = False: our loss is a per-sequence mean, so the Trainer must
  divide by grad_accum itself; otherwise gradients are grad_accum x too large.
Peak memory on the longest sequence (32,748 tokens): 30.0 GB of 31.3 GB.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # before torch import

import argparse
import json
import subprocess
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- config
def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(open(path))
    for kv in overrides:                       # dotted key=value, value parsed as YAML
        key, val = kv.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


# ----------------------------------------------------------------------------- data
class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, tok):
        self.rows = []
        for line in open(path):
            s = json.loads(line)
            ids = tok(s["text"], add_special_tokens=False)["input_ids"]
            assert len(ids) == s.get("n_tokens", s["spans"][-1][1]), "packed text does not match this tokenizer"
            labels = [-100] * len(ids)
            for a, b in s["spans"]:
                labels[a:b] = ids[a:b]
            self.rows.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

    def summary(self) -> str:
        n_tok = sum(len(r["input_ids"]) for r in self.rows)
        n_loss = sum(l != -100 for r in self.rows for l in r["labels"])
        longest = max(len(r["input_ids"]) for r in self.rows)
        return f"{len(self.rows)} sequences | {n_tok} tokens | {n_loss} loss tokens | longest {longest}"


# ----------------------------------------------------------------------------- model
def load_model(cfg: dict):
    from peft import LoraConfig, get_peft_model

    from gemma_lora import load_base   # int4 -> bf16 base, attention fix, E4B offload (shared with training/rl)

    model = load_base(ROOT / cfg["model_dir"], offload_ple=bool(cfg.get("offload_ple")))

    lc = cfg["lora"]
    targets = []
    for name, mod in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if "language_model" not in name or leaf not in lc["modules"]:
            continue
        if lc.get("skip_kv_shared") and leaf in ("k_proj", "v_proj"):
            attn = model.get_submodule(name.rsplit(".", 1)[0])
            if getattr(attn, "is_kv_shared_layer", False):
                continue
        targets.append(name)

    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=targets))
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    print(f"LoRA on {len(targets)} modules | {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return model


def make_trainer_cls(softcap: float, offload_activations: bool = False):
    import contextlib

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers import Trainer

    from gemma_lora import final_hidden

    fused_ce = LigerFusedLinearCrossEntropyLoss(softcap=softcap)
    # E4B: the tensors autograd keeps for backward (with checkpointing: each layer's input) wait in
    # pinned CPU memory instead of on the GPU; only the model forward, not the loss
    saved = (lambda: torch.autograd.graph.save_on_cpu(pin_memory=True)) if offload_activations else contextlib.nullcontext

    class FusedCETrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # final-norm output via a hook: output_hidden_states=True would keep every layer's output on the GPU
            with final_hidden(model) as grab, saved():
                out = model(input_ids=inputs["input_ids"], logits_to_keep=1)
            h = grab["h"][:, :-1]                         # position t predicts t+1
            y = inputs["labels"][:, 1:]
            W = model.get_output_embeddings().weight      # tied to the input embedding
            h = h.to(W.dtype)                              # Trainer autocast leaves h in fp32
            loss = fused_ce(W, h.reshape(-1, h.shape[-1]), y.reshape(-1))
            return (loss, out) if return_outputs else loss

    return FusedCETrainer


# ----------------------------------------------------------------------------- main
def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N optimizer steps (smoke test)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config keys")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    smoke = args.max_steps > 0
    out_dir = ROOT / cfg["output_dir"] / ("smoke" if smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments

    report = "none" if args.no_wandb else "wandb"
    if report == "wandb":
        os.environ["WANDB_ENTITY"] = cfg["wandb"]["entity"]
        os.environ["WANDB_PROJECT"] = cfg["wandb"]["project"]

    tok = AutoTokenizer.from_pretrained(ROOT / cfg["model_dir"])
    data = PackedDataset(ROOT / cfg["data"], tok)
    print("data:", data.summary())

    meta = {"config": cfg, "git": git_sha(), "data": data.summary(), "smoke": smoke}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    model = load_model(cfg)
    t = cfg["train"]
    targs = TrainingArguments(
        output_dir=str(out_dir), run_name=cfg["run_name"] + ("_smoke" if smoke else ""), report_to=report,
        per_device_train_batch_size=1, gradient_accumulation_steps=t["grad_accum"],
        num_train_epochs=t["epochs"], max_steps=args.max_steps,
        learning_rate=t["lr"], lr_scheduler_type=t["scheduler"], warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"], max_grad_norm=t["max_grad_norm"], seed=t["seed"],
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="no" if smoke else "epoch", save_only_model=True,
        dataloader_num_workers=0, remove_unused_columns=False,
    )
    trainer = make_trainer_cls(t["logit_softcap"], cfg.get("offload_activations", False))(
        model=model, args=targs, train_dataset=data,
        data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100),
    )
    trainer.model_accepts_loss_kwargs = False      # see module docstring: loss

    torch.cuda.reset_peak_memory_stats()
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):   # never fall back to the math kernel
        result = trainer.train()
    peak = torch.cuda.max_memory_allocated() / 1e9

    model.save_pretrained(out_dir / "adapter")
    tok.save_pretrained(out_dir / "adapter")
    meta.update(train_loss=result.training_loss, steps=result.global_step, peak_gb=round(peak, 1))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"DONE steps={result.global_step} train_loss={result.training_loss:.4f} "
          f"peak={peak:.1f}GB adapter={out_dir / 'adapter'}")


if __name__ == "__main__":
    main()
