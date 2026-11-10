#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
[[ -x .venv/bin/python ]] || { printf 'Run bash scripts/setup.sh first.\n' >&2; exit 1; }
docker info >/dev/null 2>&1 || { printf 'Docker is not available.\n' >&2; exit 1; }
exec .venv/bin/python -B app/main.py "$@"
