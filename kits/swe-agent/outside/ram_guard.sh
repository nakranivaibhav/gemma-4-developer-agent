#!/usr/bin/env bash
# Hard RAM floor on the Docker host of an outside-task run (2026-10-26; human: keep 5 GB of revideo's RAM free).
# Every 10 s reads MemAvailable on the host; below LIMIT_GB (5.5) it removes our newest outside-task container (label
# outside_task), one per check, until the host is back above the limit. That try ends with a harness error and is
# re-run later. The admission gate in outside.py (OUTSIDE_RAM_GATE_GB=8) keeps this a rare last resort.
# usage: bash kits/swe-agent/outside/ram_guard.sh [host=revideo]      stop: touch /tmp/ram_guard.stop
set -u
HOST=${1:-revideo}; LIMIT_GB=${LIMIT_GB:-5.5}
export DOCKER_HOST=unix:///home/vaibhav/data/${HOST}-docker.sock
rm -f /tmp/ram_guard.stop
while [ ! -f /tmp/ram_guard.stop ]; do
  kb=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "awk '/MemAvailable/ {print \$2}' /proc/meminfo" 2>/dev/null)
  if [ -n "$kb" ] && awk -v k="$kb" -v l="$LIMIT_GB" 'BEGIN{exit !(k/1048576 < l)}'; then
    c=$(docker ps --filter label=outside_task --format '{{.CreatedAt}}|{{.ID}}|{{.Label "outside_task"}}' | sort -r | head -1)
    if [ -n "$c" ]; then
      docker rm -f "$(echo "$c" | cut -d'|' -f2)" > /dev/null 2>&1
      echo "$(date -u +%H:%M:%SZ) RAM_GUARD host avail $((kb / 1024)) MB < ${LIMIT_GB} GB: removed newest container ($(echo "$c" | cut -d'|' -f3))"
    fi
  fi
  sleep 10
done
echo "RAM_GUARD stopped"
