# The swe-agent kit: run a coding agent on SWE-style tasks on one GPU machine.
# Run every target from the repo root (the root Makefile includes this file). `make help` lists them.
#
#   make setup                         data + Python env + sandbox image (first time only)
#   make serve SIZE=e2b                 start the model server in the background
#   make run AGENT=agents/e2b_t07 TASKS=smoke6
#   make gold-check                     is every task solvable in the sandbox? (fills tasks/manifest.csv)
#   make down                           stop the server and leftover sandboxes

SHELL := /bin/bash
ENV_DIR ?= envs/harness
PY    := $(ENV_DIR)/bin/python
KIT   := kits/swe-agent
SLUG  := gemma-4-developer-agent

AGENT       ?= agents/e2b_t07
TASKS       ?= smoke6
SIZE        ?= e2b
CONCURRENCY ?= 4
NGRAM       ?= 1
TIER        ?= adhoc
NOTE        ?=
STAMP       := $(shell date -u +%Y%m%dT%H%MZ)
# run name: agent_tasks_time; a list of task ids becomes "<n>tasks" so the name has no spaces
RUN         ?= $(notdir $(AGENT))_$(if $(word 2,$(TASKS)),$(words $(TASKS))tasks,$(notdir $(TASKS)))_$(STAMP)
PORT        := 8000
BASE_IMAGE  := swebench-sandbox:latest
IMAGE       := swebench-sandbox-local:latest
NETWORK     := gemma-tarpit

# model weights (Hugging Face, public). The 31B sits at data/external/model, the harness's default path.
HF_31b := google/gemma-4-31B-it-qat-w4a16-ct
HF_12b := google/gemma-4-12B-it-qat-w4a16-ct
HF_e4b := google/gemma-4-E4B-it-qat-w4a16-ct
HF_e2b := google/gemma-4-E2B-it-qat-w4a16-ct
MODEL  ?= $(if $(filter 31b,$(SIZE)),data/external/model,data/external/models/gemma-4-$(SIZE)-it-qat-w4a16-ct)
# the 12B checkpoint declares an architecture vLLM 0.19.1 does not register: serve it as the text-only class
# (kept in its own variable: the commas in it would split the arguments of $(if ...))
EXTRA_12b := --hf-overrides '{"architectures":["Gemma4ForCausalLM"],"model_type":"gemma4"}'
MODEL_EXTRA ?= $(if $(filter 12b,$(SIZE)),$(EXTRA_12b))
MAX_NUM_SEQS ?= 16
export MODEL_EXTRA MAX_NUM_SEQS

.PHONY: help setup data model env sandbox serve run gold-check down

help: ## list the targets
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-11s %s\n", $$1, $$2}'

setup: data model env sandbox ## everything, first time (needs Kaggle API keys in KAGGLE_USERNAME / KAGGLE_KEY)
	$(PY) -I $(KIT)/run/manifest.py check

data: ## competition data -> data/, competition wheelhouse -> data/external/wheelhouse
	uv run tools/kaggle_io.py download $(SLUG) --dest data
	uv run kaggle datasets download metric/gemma-4-developer-agent-wheelhouse -p data/external/wheelhouse --unzip

model: ## model weights for SIZE (e2b | e4b | 12b | 31b) from Hugging Face -> $(MODEL)
	uvx --from huggingface_hub hf download $(HF_$(SIZE)) --local-dir $(MODEL)

env: ## the harness Python env ($(ENV_DIR)), exact versions from requirements.txt
	[ -x $(PY) ] || uv venv --python 3.12 $(ENV_DIR)
	# --no-deps: the list is a complete freeze. A few wheelhouse packages are pinned on purpose against their
	# declared dependencies (e.g. opentelemetry-exporter-otlp 1.44.0 next to ...-proto-http 1.41.1), so re-resolving fails.
	uv pip install --python $(PY) --no-deps --index-strategy unsafe-best-match -r $(KIT)/requirements.txt

sandbox: ## the competition sandbox image, the local repaired image on top, the offline network, the setup hook
	docker build -q -t $(BASE_IMAGE) -f data/docker/Dockerfile.sandbox data/docker
	uv run $(KIT)/sandbox/fetch_wheels.py
	docker build -q -t $(IMAGE) $(KIT)/sandbox
	docker network inspect $(NETWORK) >/dev/null 2>&1 || docker network create --internal --subnet 10.255.0.0/16 \
	  --gateway 10.255.0.1 -o com.docker.network.bridge.enable_icc=false $(NETWORK)
	$(PY) -I $(KIT)/sandbox/build_hook.py

serve: ## start vLLM in the background for SIZE (NGRAM=0 for plain decoding); log runs/logs/server.log
	@mkdir -p runs/logs
	@if curl -s -m 2 127.0.0.1:$(PORT)/v1/models >/dev/null; then echo "server already up on :$(PORT)"; exit 0; fi; \
	$(if $(filter 0,$(NGRAM)),SPEC=,) MODEL_DIR=$(MODEL) setsid nohup bash $(KIT)/serve/serve_fast.sh $(AGENT) \
	  > runs/logs/server.log 2>&1 < /dev/null & echo $$! > runs/logs/server.pid; \
	for i in $$(seq 1 120); do \
	  curl -s -m 2 127.0.0.1:$(PORT)/v1/models >/dev/null && { echo "server up: $(MODEL), log runs/logs/server.log"; exit 0; }; \
	  kill -0 $$(cat runs/logs/server.pid) 2>/dev/null || { echo "server died: see runs/logs/server.log"; exit 1; }; \
	  sleep 10; done; echo "server not up after 20 min: see runs/logs/server.log"; exit 1

run: ## run AGENT on TASKS (manifest selector or ids) -> runs/$$RUN + results/task_runs.jsonl
	MODEL_DIR=$(MODEL) RUN_ID=$(RUN) EVAL_TIER=$(TIER) EVAL_NOTE="$(NOTE)" CONCURRENCY=$(CONCURRENCY) \
	  $(PY) -I $(KIT)/run/run_tasks.py $(AGENT) runs/$(RUN) $(TASKS)

gold-check: ## real fix and empty patch on every task in the repaired sandbox; writes tasks/manifest.csv
	IMAGE=$(IMAGE) NETWORK=$(NETWORK) KAGGLE_SANDBOX_DIR=$(KIT)/sandbox bash $(KIT)/check/gold_run.sh runs/gold_$(STAMP)
	NO_PATCH=1 IMAGE=$(IMAGE) NETWORK=$(NETWORK) KAGGLE_SANDBOX_DIR=$(KIT)/sandbox bash $(KIT)/check/gold_run.sh runs/nopatch_$(STAMP)
	$(PY) -I $(KIT)/check/gold_to_manifest.py runs/gold_$(STAMP) runs/nopatch_$(STAMP)

down: ## stop the model server and remove stopped sandbox containers (running ones belong to a live run)
	@pid=$$(cat runs/logs/server.pid 2>/dev/null); [ -n "$$pid" ] && kill -INT $$pid 2>/dev/null && echo "stopped server $$pid"; rm -f runs/logs/server.pid; \
	left=$$(docker ps -aq --filter ancestor=$(IMAGE) --filter status=exited; docker ps -aq --filter ancestor=$(IMAGE) --filter status=created); \
	[ -n "$$left" ] && docker rm $$left >/dev/null && echo "removed $$(echo $$left | wc -w) stopped sandboxes"; true
