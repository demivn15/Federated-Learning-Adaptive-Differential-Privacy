# Operating the project

All active commands assume a terminal at the repository root in Linux or Ubuntu/WSL.

## Daily startup

Start Docker Desktop on Windows, then enter the Linux installation and run:

```bash
bash scripts/run.sh
```

Keep that terminal available for startup messages and normal Ctrl+C shutdown. The web interface is `http://localhost:6060`.

## Inspect a scenario

The results page reports the execution source, protocol, state, global evaluation and privacy journals. Complete training requires the planned final evaluation and, for private modes, complete accounting from all expected trainers. A Running state alone is insufficient.

For participant state, list the managed containers without assuming an individual container exists:

```bash
docker ps -a --format 'table {{.Names}}\t{{.Status}}'
```

New participants carry a `nebula.scenario` label. Replace the example ID with the actual scenario name:

```bash
docker ps -a --filter label=nebula.scenario=SCENARIO_ID --format 'table {{.Names}}\t{{.Status}}'
```

Read a listed container's logs with `docker logs --tail 100 CONTAINER_NAME`. The application can also export the scenario's logs and configuration through its dashboard.

## Result locations

| Location | Meaning |
| --- | --- |
| `app/config/SCENARIO_ID/` | Generated configuration, manifest and HDF5 partitions |
| `app/logs/SCENARIO_ID/` | Participant logs, privacy journals and server evaluations |
| `results/generated/` | Optional local exports and new smoke checks |
| `results/nebula_v3_20261104/` | Preserved completed research evidence |

The privacy journal has one entry per client per round. The global evaluation journal belongs to the server. The panel's maximum epsilon is the maximum accounted value among clients, not their sum under the stated disjoint-allocation assumptions.

## Container lifecycle

The queue waits for cleanup before reusing participant names. Cleanup matches the requested scenario through a label or the configuration path for legacy containers. A delayed cleanup request cannot remove participants from the next scenario. Startup errors propagate and receive an error journal and failed state.

The framework can report either `completed` or `finished` as terminal states. Scientific evidence is validated separately through final evaluations and complete privacy accounting. The excluded startup failure in the recorded results has no training metrics.

## Existing managed installation

The original working installation can remain separate from this Git checkout. The source-only updater is available for an installation carrying the expected project marker:

```bash
python3 scripts/update_project2.py --project "$HOME/crypto-project2-nebula"
```

This first command only checks the target. After normal platform shutdown, adding `--apply` copies reviewed application sources and creates backups. Unexpected local edits stop the update. It does not replace saved databases, certificates or results, and it does not rebuild dependency images.

For the locked environment described by this repository, install the new checkout with `scripts/setup.sh`. The source-only updater is not a dependency migration tool.

## Dependency maintenance

The existing lock fixes versions for fresh installations. Updating it changes the software environment and requires a separate validation. Keep the historical results labelled with their original environment.

```bash
.tools/uv lock --check
python3 scripts/export_requirements.py
```

Neither command launches a federated experiment. Full rebuilds and training checks remain explicit steps.
