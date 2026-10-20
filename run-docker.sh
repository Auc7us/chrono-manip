#!/usr/bin/env bash
set -euo pipefail
container=${CHRONO_CONTAINER:-chrono-orb}
docker_args=(-i)
if [ -t 0 ] && [ -t 1 ]; then
    docker_args+=(-t)
fi
exec docker exec "${docker_args[@]}" "$container" bash -lc 'cd "$HOME/mountdir/manip-sim2real/attempt-codex" && exec ./run.sh "$@"' bash "$@"
