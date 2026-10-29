"""On-policy RL for the Gemma 4 agent (LoRA), on our own harness and the prod vLLM 0.19.1 server.

One step = rollout (hist.sh, TITO capture) -> batch (reward, advantage, segments) -> train (one update) ->
publish (next step's agent folder). Every stage reads files and writes files, so each one can run alone, on
another machine, or again after a crash. Map of the modules: training/rl/README.md.
"""
