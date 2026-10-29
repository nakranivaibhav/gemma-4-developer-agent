<!-- contract: parked design note; append updates with a date, do not rewrite history -->
# Sandbox hosting options for outside-task runs (parked 2026-10-27)

Problem: pass 1 runs task containers on revideo (GCP Brussels, 16 vCPU). Its CPU (image unpacking + gold/empty
checks + test runs) is the bottleneck, ~300 tasks/h; the 5090 sits mostly idle.

| option | fit | est. cost per pass (~6,765 tasks, 1 try) | notes |
|---|---|---|---|
| revideo (current) | works today | already paid | ~1 day/pass, fragile at high load |
| **ephemeral big VM** (GCP n2-standard-64 spot, asia-south1, 1.5 TB disk), create -> run -> delete | same pipeline, `REMOTE=<vm>` | **~$10-30** (6-8 h) | preferred; Mumbai cuts per-command latency; needs Docker Hub login for pull limits; delete at end |
| Modal Sandboxes | any registry image per sandbox | ~$170-270 (1 core/4 GiB, ~10 min/task) | sandbox rates are 3x list: $0.142/core-h, $0.024/GiB-h |
| Cloudflare Sandboxes / Containers | only via Docker-in-Docker | ~$100 | 20 GB disk max, no iptables, state lost on sleep |

Estimates, not measured. Reopen when: a pass on revideo is too slow for the RL schedule. Next step then: user creates
the VM (or logs in gcloud locally), we set up Docker + SSH tunnel, run a 20-task pilot, then the pass, then delete.

## Update 2026-10-27: Vast.ai VM instances (Docker works inside)
Vast container instances (most offers) cannot run Docker; `vms_enabled` VM instances can. Then the whole stack
(vLLM + harness + local Docker, `DOCKER_HOST` unset) runs on one box: no SSH tunnel, no India-Europe latency per
command. Live search 2026-10-27 (prices move; filter >=32 cpu, >=60 GB RAM, >=500 GB disk, >=500 Mbps):

| use | offer | $/h | cpu | RAM | disk | down | reliability |
|---|---|---|---|---|---|---|---|
| all-in-one | 1x RTX 4090, Japan | 0.67 | 32 | 251 GB | 2.9 TB | 1.4 Gbps | 0.972 |
| all-in-one, big | 2x RTX 5090, Sweden | 1.39 | 96 | 126 GB | 909 GB | 0.6 Gbps | 0.979 |
| all-in-one, biggest | 4x RTX 5090, Texas | 5.68 | 192 | 377 GB | 12.6 TB | 4.8 Gbps | 0.999 |
| sandbox host only (GPU stays home) | 1x RTX 3070, Quebec | 0.27 | 44 | 126 GB | 1.46 TB | 7.3 Gbps | 0.993 |

Setup on a fresh VM (~1-2 h first time): Docker, uv envs (harness, vLLM), E2B weights, repo + task jsonl + graphs
(R2), build swebench-sandbox-local, create gemma-tarpit network, then the usual run with lower concurrency for RAM.
Account is prepaid (credit $0 on 2026-10-27). Rent only after the user's go on a quoted price; destroy when done.
