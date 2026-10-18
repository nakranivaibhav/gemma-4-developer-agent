#!/usr/bin/env bash
# Runs ON the rented pod (image vllm/vllm-openai:v0.19.1): check the GPU, list package versions, fetch the model weights.
# The model sits at the same absolute path as on the local machine, because the harness asks for the model by that path.
# usage: LOCAL_REPO=<repo path on your machine> bash pod_setup.sh
# LoRA adapters: the stock vLLM 0.19.1 image refuses LoRA for Gemma 4 ("Gemma4ForConditionalGeneration does not support
# LoRA yet"); the competition's own vLLM build accepts it. Copy its 3 changed files from your local harness env onto the
# pod's vLLM (run on your machine; find the pod's dir with: python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))"):
#   for f in lora/model_manager.py model_executor/models/gemma4.py model_executor/models/gemma4_mm.py; do
#     scp -P <port> envs/harness/lib/python3.12/site-packages/vllm/$f root@<pod>:<pod vllm dir>/$f; done
set -u
source /root/container_env.sh 2>/dev/null
P=${LOCAL_REPO:?set LOCAL_REPO to the absolute path of this repo on your own machine: the harness asks the server for the model by that path}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c "
from importlib.metadata import distributions
for n, v in sorted({(d.metadata['Name'] or '').lower().replace('_', '-'): d.version for d in distributions()}.items()): print(f'{n}=={v}')" > /root/pod_versions.txt
wc -l /root/pod_versions.txt
mkdir -p "$P/data/external/model" "$P/agents/out8k/adapters" && cd "$P/data/external/model"
T0=$(date +%s)
[ -f model.safetensors ] || curl -sS -L --fail --retry 3 -o model.safetensors https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/resolve/main/model.safetensors
echo "DOWNLOAD_SECONDS $(( $(date +%s) - T0 )) bytes $(stat -c %s model.safetensors)"
sha256sum model.safetensors | cut -c1-64
echo SETUP_DONE
