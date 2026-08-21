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

# Persist the W&B credential across containers. `wandb login` writes ~/.netrc,
# so the file must exist on the host before it can be bind-mounted.
NETRC="${HOME}/.netrc"
[ -f "$NETRC" ] || touch "$NETRC"

exec docker run --rm -it \
  --gpus all \
  --shm-size=8g \
  -v "${REPO}:/workspace" \
  -v "${NETRC}:/root/.netrc" \
  -e WANDB_PROJECT="${WANDB_PROJECT:-fa26-independent-study}" \
  -w /workspace \
  "$IMAGE" "$@"
