#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
test_python="${NEBULA_TEST_PYTHON:-$repo_root/.venv/bin/python}"
"$test_python" -B scripts/check_repository.py
"$test_python" -B tests/verify_privacy.py
"$test_python" -B tests/verify_adaptive_release.py
if command -v node >/dev/null 2>&1; then
    node --experimental-vm-modules tests/verify_frontend_adaptive.mjs
else
    printf 'Optional browser-script checks skipped. Install Node.js 22 to run them.\n'
fi
