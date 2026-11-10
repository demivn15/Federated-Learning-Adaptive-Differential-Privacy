#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$(uname -s)" == "Linux" ]] || { printf 'Run this script in Ubuntu/WSL.\n' >&2; exit 1; }
target_root="${NEBULA_PROJECT_DIR:-$HOME/NEBULA}"
python3 "$repo_root/scripts/copy_workspace.py" --target "$target_root"
bash "$target_root/scripts/setup.sh"
printf '\nContinue in: %s\nRun: bash scripts/test.sh\nStart: bash scripts/run.sh\n' "$target_root"
