"""Check/apply the reviewed Project 2 sources to the existing managed WSL copy.

No datasets, logs, certificates, databases, environments or Docker images are
deleted. Changed sources are backed up. Unexpected edits stop the update.
"""
import argparse
import datetime
import getpass
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def update(project, apply=False):
    project = project.resolve()
    marker = project / ".project2-managed"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != "525b31c674644c653117126b5a1473a5321ac240":
        raise ValueError("The target is not the previously prepared Project 2 installation")
    manifest = json.loads((HERE / "update_manifest.json").read_text(encoding="utf-8"))
    pending = []
    for relative, hashes in manifest.items():
        src, dest = HERE / "nebula" / relative, project / relative
        if not dest.resolve().is_relative_to(project):
            raise ValueError("A destination escapes the selected project")
        if digest(src) != hashes["after"]:
            raise ValueError(f"Prepared source changed since verification: {relative}")
        actual = digest(dest)
        if actual == hashes["after"]:
            continue
        if actual not in [hashes["before"], *hashes.get("previous", [])]:
            raise ValueError(f"Unexpected local edits in {relative}. Existing files were preserved.")
        pending.append((relative, src, dest))
    print(f"Verified target: {project}")
    print(f"Files to update: {len(pending)}")
    if not apply:
        print("Check only. Add --apply after stopping NEBULA with Ctrl+C.")
        return
    if platform.system() != "Linux":
        raise ValueError("Apply this update from Ubuntu/WSL")
    prefix = getpass.getuser() + "_"
    running = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True).splitlines()
    affected = [name for name in running if name.startswith(prefix) and
                ("nebula-frontend" in name or "nebula-controller" in name or "-participant" in name)]
    if affected:
        raise ValueError("Stop NEBULA first. Running containers: " + ", ".join(affected))
    if not pending:
        print("This update is already installed.")
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = project / "project2-backups" / stamp
    installed = []
    try:
        for relative, src, dest in pending:
            if dest.exists():
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dest, saved)
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_name(dest.name + ".project2-new")
            shutil.copyfile(src, temp)
            os.replace(temp, dest)
            installed.append((relative, dest))
    except Exception:
        for relative, dest in reversed(installed):
            saved = backup / relative
            if saved.exists():
                shutil.copy2(saved, dest)
            elif dest.exists():
                dest.unlink()
        raise
    (project / ".project2-revision").write_text(
        (HERE / "nebula" / ".project2-revision").read_text(encoding="ascii"), encoding="ascii")
    print(f"Update applied. Source backup: {backup}")
    print("No Docker rebuild is required: these sources are bind-mounted into the existing images.")
    print("Start: source .venv/bin/activate && python app/main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    update(args.project, args.apply)
