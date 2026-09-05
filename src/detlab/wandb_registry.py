#!/usr/bin/env python
"""Resolve which W&B project a run belongs to, from the experiment's identity.

WHY THIS EXISTS
---------------
Every training script used to fall back to the literal string
"benjbritton_FA26" when nothing else was set, and scripts/run.sh injected the
same name as an environment default. That name was chosen during the SpaceNet
milestone, when it was the only work in the repository. Chactun training was
added later, reused the same runner, and inherited the SpaceNet project
silently -- 40 of the 54 runs landed in a project named after a different
experiment, and nobody noticed for three weeks.

The bug was not the name. It was that a project identity could be inherited
rather than declared.

PRECEDENCE, and the reason for the order
----------------------------------------
    1. --project on the command line     explicit, deliberate, wins
    2. the registry entry for this key   the experiment knows what it is
    3. WANDB_PROJECT in the environment  a shell-level fallback
    4. fail

The registry beats the environment on purpose. A stale `export
WANDB_PROJECT=...` left in a shell is the same class of accident this module
exists to prevent, and it must not be able to redirect a script that already
knows its own identity. An override still exists, but it has to be typed at the
call site where a reader can see it.

An unregistered key is an error rather than a default, so a new experiment
cannot start logging until it has declared what it is.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
REGISTRY = os.path.join(ROOT, "configs", "wandb_projects.json")


def load_registry(path=REGISTRY):
    with open(path) as f:
        return json.load(f)["projects"]


def resolve(experiment_key, explicit=None, registry_path=REGISTRY):
    """Return the W&B project for this experiment, or raise.

    experiment_key identifies the experiment family, not the run: "chactun",
    "spacenet2", "balloon", "gliht".
    """
    if explicit:
        return explicit

    projects = load_registry(registry_path)
    if experiment_key in projects:
        return projects[experiment_key]

    env = os.environ.get("WANDB_PROJECT")
    if env:
        return env

    raise SystemExit(
        "W&B project unresolved for experiment key %r.\n"
        "  Known keys: %s\n"
        "  Either add %r to %s, or pass --project explicitly.\n"
        "  Refusing to guess: inheriting another experiment's project is the\n"
        "  bug this check exists to prevent."
        % (experiment_key, ", ".join(sorted(load_registry(registry_path))),
           experiment_key, registry_path))


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else None
    if key:
        print(resolve(key))
    else:
        for k, v in sorted(load_registry().items()):
            print("%-12s -> %s" % (k, v))
