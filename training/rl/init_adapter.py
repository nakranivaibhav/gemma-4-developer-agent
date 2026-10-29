"""Write a fresh LoRA adapter that leaves the base model unchanged, to start RL from the base model (2026-10-27).

usage: uv run python -m training.rl.init_adapter <sft_config.yaml> <out_dir>
The LoRA shape (rank, alpha, target modules, skip_kv_shared) comes from the SFT config's `lora:` block, so the
adapter matches what train_step.py and vLLM already load. peft initialises lora_B to zeros, so B @ A = 0 and the
served policy is exactly the base model; the check below refuses to save otherwise.
"""
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sft_train import load_model  # noqa: E402  (same LoRA construction as SFT)


def main():
    cfg_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = yaml.safe_load(open(cfg_path))
    model = load_model(cfg)
    nonzero = [n for n, p in model.named_parameters() if "lora_B" in n and torch.count_nonzero(p) > 0]
    if nonzero:
        sys.exit(f"lora_B not zero in {len(nonzero)} tensors (e.g. {nonzero[0]}): adapter would change the base")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    print(f"INIT_ADAPTER zero LoRA (r={cfg['lora']['r']}) -> {out}")


if __name__ == "__main__":
    main()
