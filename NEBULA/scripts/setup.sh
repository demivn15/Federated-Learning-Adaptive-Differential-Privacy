#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" != "Linux" ]]; then
    printf 'Run setup inside Linux or Ubuntu/WSL.\n' >&2
    exit 1
fi
if [[ "$repo_root" == /mnt/?/* ]]; then
    printf 'Install in the Linux filesystem. From this folder run: bash scripts/install_wsl.sh\n' >&2
    exit 1
fi
command -v python3 >/dev/null || { printf 'python3 is required.\n' >&2; exit 1; }
docker info >/dev/null 2>&1 || { printf 'Start Docker and enable Ubuntu WSL integration.\n' >&2; exit 1; }
cd -- "$repo_root"
mkdir -p .tools
if [[ ! -x .tools/uv ]]; then
    python3 scripts/install_uv_linux.py "$repo_root/.tools"
fi
export UV_CACHE_DIR="$repo_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$repo_root/.python"
export UV_PROJECT_ENVIRONMENT="$repo_root/.venv"
export UV_CONCURRENT_DOWNLOADS=2
export UV_CONCURRENT_BUILDS=1
python_version="$(cat .python-version)"
.tools/uv python install "$python_version"
.tools/uv sync --locked --python "$python_version" --group controller --group core --no-install-project --no-dev
.venv/bin/python -B app/main.py --version
for component in core controller frontend; do
    docker build --file Dockerfile.cpu --target "$component" --tag "nebula-project2-$component" .
done
printf '%s\n' '525b31c674644c653117126b5a1473a5321ac240' > .project2-managed
mkdir -p .local/installation
.tools/uv pip freeze --python .venv/bin/python > .local/installation/host-dependencies.txt
{
    date --iso-8601=seconds
    .venv/bin/python --version
    docker version --format 'Docker client={{.Client.Version}} server={{.Server.Version}}'
    docker image inspect nebula-project2-core --format 'CPU image ID={{.Id}}'
} > .local/installation/environment.txt
printf '\nSetup complete. Run checks: bash scripts/test.sh\nStart the platform: bash scripts/run.sh\n'
