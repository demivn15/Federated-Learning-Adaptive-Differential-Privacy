"""Refresh readable dependency exports from the existing uv lock."""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
binary = ROOT / ".tools" / ("uv.exe" if os.name == "nt" else "uv")
uv = str(binary) if binary.is_file() else shutil.which("uv")
if not uv:
    raise SystemExit("Run setup first to install the project-local uv tool.")
for name, groups in (("host", ["controller", "core"]), ("containers", ["controller", "core", "frontend"])):
    destination = ROOT / "requirements" / (name + ".txt")
    command = [uv, "export", "--quiet", "--locked", "--no-dev", "--no-emit-project", "--no-hashes",
               "--format", "requirements-txt", "--output-file", destination.relative_to(ROOT).as_posix()]
    for group in groups:
        command.extend(["--group", group])
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    content = destination.read_text(encoding="utf-8")
    directive = "--extra-index-url https://download.pytorch.org/whl/cpu\n"
    if directive not in content:
        destination.write_text(directive + content, encoding="utf-8", newline="\n")
    print(f"Exported requirements/{destination.name}")
