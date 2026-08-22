#!/usr/bin/env bash
# Run a command inside the detectron2 image with GPU, mounts and W&B wired up.
# Exists so the exact invocation is version-controlled rather than retyped.
#
#   ./scripts/run.sh python scripts/train_balloon.py --smoke
#   ./scripts/run.sh wandb login
#   ./scripts/run.sh bash                      # interactive shell
set -euo pipefail

IMAGE="${IMAGE:-m2/detectron2:cu124-torch251}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Run as the invoking user, NOT root.
#
# Containers default to root, so everything written through the bind mount --
# checkpoints, metrics.json, wandb run dirs -- lands on the host owned by
# root:root. You then cannot delete or edit your own outputs without sudo, and
# Windows Explorer cannot touch them at all. Matching the host UID/GID keeps
# ownership correct on both sides.
USER_FLAGS="-u $(id -u):$(id -g)"

# A non-root UID has no entry in the container passwd file, so HOME is unset and
# anything expecting a home directory breaks. Point it at a mounted cache dir:
# this also persists the model-zoo downloads (~178 MB per checkpoint) and the
# matplotlib/fontconfig caches between runs instead of refetching every time.
CACHE="${REPO}/.cache"
mkdir -p "$CACHE"

# `wandb login` wrote the credential to the host ~/.netrc. Mount it where the
# container HOME expects to find it.
NETRC="${HOME}/.netrc"
[ -f "$NETRC" ] || touch "$NETRC"

# Allocate a TTY only when there actually is one. `-t` without a terminal
# (CI, piped output, non-interactive shell) makes docker refuse to start.
TTY_FLAGS="-i"
if [ -t 0 ] && [ -t 1 ]; then
  TTY_FLAGS="-it"
fi

# Entity is pinned to the personal handle rather than the auto-created
# benjbritton-geoai team, so published run URLs match the same username used on
# GitHub, Hugging Face and LinkedIn. Override per-invocation if a project
# genuinely needs the team (collaborators can only be added to a team entity).
exec docker run --rm ${TTY_FLAGS} ${USER_FLAGS} \
  --gpus all \
  --shm-size=8g \
  -v "${REPO}:/workspace" \
  -v "${CACHE}:/cache" \
  -v "${NETRC}:/cache/.netrc" \
  -e HOME=/cache \
  -e WANDB_PROJECT="${WANDB_PROJECT:-fa26-independent-study}" \
  -e WANDB_ENTITY="${WANDB_ENTITY:-benjbritton-geoai}" \
  -w /workspace \
  "$IMAGE" "$@"
