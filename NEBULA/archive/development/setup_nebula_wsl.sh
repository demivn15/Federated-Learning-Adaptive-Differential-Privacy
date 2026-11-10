#!/usr/bin/env bash
set -euo pipefail
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target_dir="${NEBULA_PROJECT2_DIR:-$HOME/crypto-project2-nebula}"
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Run this script inside Ubuntu/WSL, not PowerShell." >&2; exit 1
fi
if ! command -v docker >/dev/null || ! docker info >/dev/null 2>&1; then
    echo "Docker is not available inside Ubuntu. Check Docker Desktop > Settings > Resources > WSL Integration > Ubuntu." >&2
    exit 1
fi
if [[ -e "$target_dir" && ! -f "$target_dir/.project2-managed" ]]; then
    echo "Target already exists without the Project 2 marker. Choose another NEBULA_PROJECT2_DIR." >&2; exit 1
fi
if [[ ! -d "$target_dir" ]]; then
    mkdir -p "$target_dir"
    cp -R "$source_dir/nebula/." "$target_dir/"
    cp "$source_dir/Dockerfile.cpu" "$target_dir/Dockerfile.cpu"
    printf '%s\n' '525b31c674644c653117126b5a1473a5321ac240' > "$target_dir/.project2-managed"
elif ! cmp --silent "$source_dir/nebula/.project2-revision" "$target_dir/.project2-revision" || ! cmp --silent "$source_dir/Dockerfile.cpu" "$target_dir/Dockerfile.cpu"; then
    echo "The prepared sources changed. Existing installation was preserved. Select a fresh NEBULA_PROJECT2_DIR for this revision." >&2
    exit 1
fi
cd "$target_dir"
if ! command -v python3 >/dev/null; then
    echo "python3 is missing inside Ubuntu. Install it, then rerun this script." >&2; exit 1
fi
mkdir -p .tools
if [[ ! -x .tools/uv ]]; then
    python3 "$source_dir/install_uv_linux.py" "$target_dir/.tools"
fi
export UV_CACHE_DIR="$target_dir/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$target_dir/.python"
export UV_PROJECT_ENVIRONMENT="$target_dir/.venv"
export UV_CONCURRENT_DOWNLOADS=2
export UV_CONCURRENT_BUILDS=1
.tools/uv python install 3.11
.tools/uv sync --python 3.11 --group controller --group core --no-install-project --no-dev
.venv/bin/python app/main.py --version
for component in core controller frontend; do
    docker build --file Dockerfile.cpu --target "$component" --tag "nebula-project2-$component" .
done
mkdir -p project2-evidence
.tools/uv pip freeze --python .venv/bin/python > project2-evidence/host-dependencies.txt
{
    printf '%s\n' "NEBULA upstream commit: $(cat .project2-managed)"
    date --iso-8601=seconds
    .venv/bin/python --version
    docker version --format 'Docker client={{.Client.Version}} server={{.Server.Version}}'
    docker image inspect nebula-project2-core --format 'CPU image ID={{.Id}}'
} > project2-evidence/installation.txt
printf '\nSetup completed. Start NEBULA with:\ncd %q\nsource .venv/bin/activate\npython app/main.py\n' "$target_dir"
printf '\nLocal interface: http://127.0.0.1:6060\nInitial local-demo login: admin / admin\n'
