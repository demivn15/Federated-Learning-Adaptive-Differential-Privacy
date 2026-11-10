# Dependency files

`pyproject.toml` declares the direct dependencies and component groups. `uv.lock` is the authoritative locked resolution for Linux and Python 3.11. The CPU PyTorch index is explicit in the project configuration.

- `host.txt` exports the controller and core groups for the host launcher and local tests.
- `containers.txt` exports controller, core and frontend groups for the shared CPU image layer.

Setup and Docker builds install through uv with a lock check. These text files provide readable exports and a pip-compatible CPU index directive. Regenerate them after an intentional lock update:

```bash
python3 scripts/export_requirements.py
```

The lock prepared with this repository is distinct from a freeze of the historical November 4 environment. The preserved results retain their original provenance. New dependency installations require their own runtime validation.
