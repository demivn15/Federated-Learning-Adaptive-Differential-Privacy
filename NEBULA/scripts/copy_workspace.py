"""Copy the distributable project to a fresh Linux working directory."""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIRS = {".git", ".venv", ".tools", ".python", ".cache", ".local", "__pycache__", "node_modules"}


def copy_workspace(target):
    target = target.expanduser().resolve()
    if target == ROOT:
        return
    if target.is_relative_to(ROOT):
        raise ValueError("The installation cannot be nested inside its source folder")
    if target.exists():
        marker = target / ".project2-revision"
        if marker.is_file() and marker.read_bytes() == (ROOT / ".project2-revision").read_bytes():
            print(f"Reusing the existing working directory without overwriting it: {target}")
            return
        raise ValueError("The target already exists. Select a fresh NEBULA_PROJECT_DIR; existing files were preserved.")
    def ignore(directory, names):
        directory = Path(directory)
        excluded = set(LOCAL_DIRS)
        if directory == ROOT / "app":
            excluded.update({".env", "config", "logs", "databases", "certs", "tmp"})
        if directory == ROOT / "results":
            excluded.add("generated")
        return [name for name in names if name in excluded or name.endswith((".pyc", ".pem", ".key"))]
    shutil.copytree(ROOT, target, ignore=ignore)
    print(f"Copied the project to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    copy_workspace(parser.parse_args().target)
