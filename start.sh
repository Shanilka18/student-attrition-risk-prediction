#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

if [[ ! -f .venv/bin/activate ]]; then
  printf '%s\n' "The project virtual environment is missing. Follow the setup steps in README.md first." >&2
  exit 1
fi

if [[ ! -f artifacts/active_run.txt ]]; then
  printf '%s\n' "No active model run exists. Run 'python -m mei.cli all' first." >&2
  exit 1
fi

source .venv/bin/activate
exec python -m mei.cli serve "$@"
