#!/usr/bin/env bash
# Apply the reviewed sources, run checks in Ubuntu, then start the platform.
set -euo pipefail
update_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="${1:-$HOME/crypto-project2-nebula}"
if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
    printf 'Expected the existing Python environment at %s/.venv\n' "$project_dir" >&2
    exit 1
fi
python3 -B "$update_dir/update_project2.py" --project "$project_dir" --apply
cd -- "$project_dir"
source .venv/bin/activate
PROJECT2_REPO="$project_dir" python -B "$update_dir/verify_privacy.py"
python -B "$update_dir/verify_adaptive_release.py"
printf '\nChecks passed. Starting NEBULA with fixed/adaptive controls.\n'
python app/main.py
