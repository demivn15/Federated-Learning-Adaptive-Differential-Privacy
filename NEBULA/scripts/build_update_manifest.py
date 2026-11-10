"""Record reviewed managed files for the non-destructive WSL updater."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE
before = json.loads((HERE / "provenance/update_before.json").read_text(encoding="utf-8"))
previous_path = HERE / "provenance/update_manifest.json"
previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
new = ["nebula/core/training/privacy_policy.py", "nebula/core/training/research_results.py",
       "nebula/core/datasets/breast_cancer/research_data.py", "nebula/frontend/research_view.py",
       "nebula/frontend/templates/research.html", "nebula/frontend/static/js/research.js",
       "nebula/frontend/static/css/research.css", "nebula/core/training/adaptive_clipping.py",
       "nebula/controller/container_lifecycle.py"]
manifest = {}
revision = hashlib.sha256()
for relative in sorted(set(before) | set(new)):
    data = (ROOT / relative).read_bytes()
    revision.update(relative.encode())
    revision.update(data)
    after = hashlib.sha256(data).hexdigest()
    old = previous.get(relative, {})
    accepted = set(old.get("previous", []))
    if old.get("after") and old["after"] != after:
        accepted.add(old["after"])
    manifest[relative] = {"before": before.get(relative), "previous": sorted(accepted), "after": after}
revision.update((HERE / "Dockerfile.cpu").read_bytes())
(ROOT / ".project2-revision").write_text(revision.hexdigest() + "\n", encoding="ascii")
(HERE / "provenance/update_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Recorded {len(manifest)} reviewed update files")
