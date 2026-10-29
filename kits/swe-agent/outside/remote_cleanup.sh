#!/usr/bin/env bash
# Remove everything our outside-task runs left on a remote Docker host (2026-10-26, for revideo: keep that VM clean).
# Removes ONLY our objects: swerebench/* task images (also mirror.gcr.io/swerebench/*, pulled through Google's mirror), the pinned base image gemma/base-anchor:pinned, the offline
# network gemma-tarpit, then dangling layers. Never touches other images, containers, volumes or files on the host.
# Refuses while any of our task containers is running (a screen/RL batch is live) unless --force.
# usage: bash kits/swe-agent/outside/remote_cleanup.sh [host=revideo] [--keep-base] [--force]
#   --keep-base  keep the pinned base image + network (between runs; saves ~850 MB per image on the next run)
set -u
HOST=revideo; KEEP=""; FORCE=""
for a in "$@"; do case $a in --keep-base) KEEP=1 ;; --force) FORCE=1 ;; *) HOST=$a ;; esac; done
SOCK=/home/vaibhav/data/${HOST}-docker.sock
if ! DOCKER_HOST=unix://$SOCK docker info > /dev/null 2>&1; then
  rm -f "$SOCK"
  setsid nohup ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o StreamLocalBindUnlink=yes \
    -L "$SOCK":/var/run/docker.sock "$HOST" > /dev/null 2>&1 < /dev/null &
  sleep 5
fi
export DOCKER_HOST=unix://$SOCK
docker info > /dev/null 2>&1 || { echo "cannot reach Docker on $HOST"; exit 1; }
before=$(ssh -o BatchMode=yes "$HOST" "df -h / | tail -1" | awk '{print $4" free"}')
ours=$(docker ps -a --format '{{.ID}} {{.Image}}' | awk '$2 ~ /^((mirror\.gcr\.io\/)?swerebench\/|gemma\/base-anchor)/ {print $1}')
running=$(docker ps --format '{{.ID}} {{.Image}}' | awk '$2 ~ /^((mirror\.gcr\.io\/)?swerebench\/|gemma\/base-anchor)/' | wc -l)
if [ "$running" -gt 0 ] && [ -z "$FORCE" ]; then
  echo "$running of our task containers are running on $HOST (a batch is live); stop it first or pass --force"; exit 2
fi
[ -n "$ours" ] && echo "$ours" | xargs -r docker rm -f > /dev/null && echo "removed $(echo "$ours" | wc -l) task containers"
n=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -cE '^(mirror\.gcr\.io/)?swerebench/')
docker images --format '{{.Repository}}:{{.Tag}}' | grep -E '^(mirror\.gcr\.io/)?swerebench/' | xargs -r docker rmi -f > /dev/null 2>&1
echo "removed $n task images"
if [ -z "$KEEP" ]; then
  docker rmi gemma/base-anchor:pinned > /dev/null 2>&1 && echo "removed gemma/base-anchor:pinned"
  docker network rm gemma-tarpit > /dev/null 2>&1 && echo "removed network gemma-tarpit"
fi
docker image prune -f > /dev/null
echo "$HOST disk: $before -> $(ssh -o BatchMode=yes "$HOST" "df -h / | tail -1" | awk '{print $4" free"}')"
