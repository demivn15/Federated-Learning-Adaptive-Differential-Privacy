"""Check distributable files, Python syntax, hashes and required resources."""
import ast
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".tools", ".python", ".cache", ".local", ".venv", "__pycache__", "node_modules"}


def check():
    required = ["README.md", "LICENSE", "NOTICE.md", "pyproject.toml", "uv.lock", "Dockerfile.cpu",
                "app/main.py", "app/deployer.py", "nebula/core/training/adaptive_clipping.py",
                "scripts/setup.sh", "scripts/run.sh", "scripts/test.sh",
                "provenance/update_manifest.json", "configs/scenarios/adaptive.json"]
    for relative in required:
        if not (ROOT / relative).is_file():
            raise ValueError(f"Missing repository file: {relative}")
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["requires-python"] == ">=3.11,<3.12"
    manifest = json.loads((ROOT / "provenance/update_manifest.json").read_text(encoding="utf-8"))
    for relative, row in manifest.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == row["after"], relative
    private_key = re.compile(rb"-----BEGIN(?: RSA| EC| OPENSSH| ENCRYPTED)? PRIVATE KEY-----\s+[A-Za-z0-9+/=\r\n]{80,}")
    count = 0
    for directory in (ROOT / "app", ROOT / "nebula", ROOT / "scripts", ROOT / "tests"):
        for path in directory.rglob("*"):
            if not path.is_file() or SKIP.intersection(path.relative_to(ROOT).parts) or "fixtures" in path.parts:
                continue
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path.relative_to(ROOT)))
                count += 1
            if path.suffix == ".sh" and b"\r\n" in path.read_bytes():
                raise ValueError(f"Shell script contains Windows line endings: {path.relative_to(ROOT)}")
            if private_key.search(path.read_bytes()):
                raise ValueError(f"Private key material in distributable source: {path.relative_to(ROOT)}")
    for relative in ("README.md", *[str(p.relative_to(ROOT)) for p in (ROOT / "scripts").glob("*") if p.is_file()]):
        text = (ROOT / relative).read_text(encoding="utf-8")
        patterns = (r"C:[\\/]Users[\\/]", r"/home/" + r"[A-Za-z0-9_.-]+/",
                    r"/mnt/" + r"[a-z]/Users/[A-Za-z0-9_.-]+/")
        if any(re.search(pattern, text) for pattern in patterns):
            raise ValueError(f"Machine-specific path in active instructions: {relative}")
    if (ROOT / ".git").exists():
        for relative in ("app/main.py", "uv.lock", "pyproject.toml", ".github/workflows/checks.yml"):
            result = subprocess.run(["git", "check-ignore", relative], cwd=ROOT, capture_output=True)
            if result.returncode == 0:
                raise ValueError(f"Required file is ignored by Git: {relative}")
    print(json.dumps({"python_files_parsed": count, "managed_sources_verified": len(manifest),
                      "required_files_present": len(required), "result": "PASS"}))


if __name__ == "__main__":
    check()
