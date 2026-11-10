# Preparing the GitHub repository

The `NEBULA` directory is the repository root. The surrounding workspace contains unrelated projects.

## Review the local files

The local repository starts on branch `main` without a remote or a commit. From this directory:

```bash
git status --short
python3 scripts/check_repository.py
```

`app/main.py`, `app/deployer.py`, `nebula/`, `pyproject.toml`, `uv.lock`, the CPU Dockerfile, tests, documentation and recorded public-data evidence belong in the repository. Local virtual environments, downloaded tools, runtime certificates, application databases and generated logs do not.

The `.gitignore` protects generated paths while retaining application code. It does not remove files which were already committed elsewhere. Review the staged list before publishing:

```bash
git add .
git diff --cached --stat
git status --short
```

## Connect an empty repository

Create an empty GitHub repository for this project. Avoid initializing the remote with another README or license. Replace the example account and repository URL below with the actual destination:

```bash
git commit -m "Organize NEBULA federated privacy research project"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Configure Git author information locally if Git requests it. Authenticate through the normal GitHub credential flow. Do not put a token in the remote URL or in a tracked configuration file.

If the remote already contains project files, integrate its history normally rather than forcing a replacement. The preserved upstream license and attribution are part of this repository.

## After upload

Check that the root README renders, its local links work, and `app/` is present. The Repository checks workflow validates the portable source layout, migration fixtures, lifecycle logic and JavaScript behavior. It does not publish a site, start Docker experiments, or send results elsewhere.

Historical material under `archive/` is not the current experiment specification. The main README and `docs/project/` describe the active public-reference adaptive-clipping protocol.
