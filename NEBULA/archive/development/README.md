# Federated Breast Tumor Classification with Record-Level Differential Privacy

## November 4 adaptive privacy update

See [README_PRIVACY_ADAPTIVE.md](README_PRIVACY_ADAPTIVE.md) for the current fixed/adaptive mechanism, verification and safe Ubuntu update command. Version 3 implements public-reference adaptive clipping. All eight privacy tests and nine release/lifecycle tests passed in Ubuntu. Three full NEBULA runs produced separate final evaluations: baseline, fixed DP and adaptive DP. Their measured evidence and the excluded Docker startup failure are preserved under `results/privacy_v3_nebula_20261104`. The live stop endpoint also returned HTTP 200 after the final correction. Version 2 experiments retain their original results. Earlier installation and pilot history below is retained as historical context.

This Project 2 extension evaluates classification of benign and malignant breast tumor records across three simulated medical centers. Its research question concerns the effect of record-level DP-SGD on classification performance and computational cost.

The application concerns numerical measurements of cell nuclei. It does not process mammograms or collect patient records. The public [Wisconsin Diagnostic Breast Cancer dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) contains 569 records and 30 input features. The three centers are synthetic partitions of one dataset, not data from three real institutions.

## Status on October 26, 2026

After installation in Ubuntu, the student's logs confirmed controller and frontend startup with Python 3.11.13 and PyTorch 2.4.1+cpu. A direct HTTP check from the chat confirmed status 200 at `http://127.0.0.1:6060/platform`. The first federated scenario is still pending.

| Component | Actual status |
| --- | --- |
| Docker Desktop | Linux engine confirmed working by the student's terminal output, version 4.97.0 / Engine 29.9.2 |
| Ubuntu | Registered as a WSL 2 distribution |
| NEBULA source | Downloaded and pinned to `525b31c674644c653117126b5a1473a5321ac240` |
| CPU/local NEBULA preparation | Source patches, TOML/JSON checks and Python syntax checks completed |
| NEBULA CPU Docker images | Build recipe supplied. Controller/frontend now started from the student's Ubuntu terminal; image inventory cannot be inspected from the chat |
| NEBULA scenario | Not executed yet |
| Independent CPU pilot | Baseline and DP-SGD runs completed, three rounds each |

The chat execution environment cannot access the WSL service or Docker's named pipe. Docker works in the student's normal terminal. Installation commands must run in that terminal until service access becomes available here. The Python pilot is separate from NEBULA and must retain that description in the progress presentation.

## Research scope

The proposed full experiment compares ordinary FedAvg, a clipping-only control, and three private settings. Both random and heterogeneous client allocations can follow after the first operational IID scenario. Privacy accounting must persist across all rounds.

The application-specific evaluation should report malignant-case recall, malignant F1, ROC-AUC, accuracy, elapsed time and peak memory. A private configuration with good overall accuracy can still have an unsuitable error distribution. The public test set must remain separate from hyperparameter selection. Repetitions and the clipping-only control are still pending.

The original proposal's MNIST dataset, 10 clients and 7,850-parameter model are no longer descriptions of this application pilot. The independent pilot uses three clients and a 30-256-128-2 MLP with 41,090 parameters, matching the dimensions of NEBULA's BreastCancer MLP. Its optimizer is Adam with learning rate 0.001. Its Adam state persists between rounds, so it is not an exact reproduction of NEBULA's optimizer lifecycle.

## Prepared NEBULA changes

The untouched upstream checkout remains under the chat's `work/project2-nebula/nebula-upstream` directory. The `nebula` directory beside this README contains a separate, locally modified copy.

- CPU PyTorch and torchvision wheels replace the upstream CUDA image path.
- One shared Python 3.11 dependency layer supports core, controller and frontend images.
- Separate image tags `nebula-project2-core`, `nebula-project2-controller` and `nebula-project2-frontend` avoid replacing upstream tags.
- Published web, statistics and controller ports bind to `127.0.0.1`.
- BreastCancer batch size is 8, workers default to zero, and participant CPU threading is limited.
- Validation and test loaders retain incomplete final batches.
- Default participant settings select a short CPU baseline. The web interface still requires explicit scenario selection.
- `.dockerignore` excludes local environments, caches and generated artifacts from image build inputs.

`prepare_nebula.py` applies these changes with checks against the pinned source. Docker execution remains the integration check. The Docker build includes an explicit CPU-wheel and NumPy interoperability smoke check.

## Ubuntu/WSL installation

From Windows PowerShell:

```powershell
wsl -d Ubuntu
```

Inside Ubuntu, confirm Docker is reachable:

```bash
docker version
```

Then run the prepared installer from its actual Windows location:

```bash
bash "/mnt/c/Users/USER/Documents/Codex/2026-09-28/ver-s-la-idea-que-yo/outputs/project2-nebula/setup_nebula_wsl.sh"
```

The script checks Docker first. It creates an isolated Linux working copy at `$HOME/crypto-project2-nebula`, installs uv and Python 3.11 inside that project, installs the host dependencies and builds the three CPU images sequentially. It does not install a second Docker daemon or change shell startup files. Existing unmarked directories are preserved. A changed prepared revision requires a fresh target directory rather than overwriting an existing installation.

The first build downloads dependencies and can take substantial time. Its actual duration and resource requirements have not been measured on this PC. NEBULA's documentation lists 8 GB RAM as the minimum, recommends 32 GB for virtualized participants, and requires at least 20 GB disk space plus datasets. The selected laptop has approximately 8 GB RAM, so the initial scenario stays small.

After successful installation:

```bash
cd "$HOME/crypto-project2-nebula"
source .venv/bin/activate
python app/main.py
```

The local interface is `http://127.0.0.1:6060`. The initial local-demo credentials are `admin` / `admin`. This configuration is for a local public-data experiment.

The installer saves the Python dependency list and image identifier under `project2-evidence`. A failed dependency/build check should be resolved before starting a scenario. No successful NEBULA deployment is claimed until its containers and scenario complete.

## First NEBULA scenario

### Existing installation: preserve the web port

The first HTTP check found a Dashboard link generated as `http://127.0.0.1/platform/dashboard`, omitting `:6060`. The Nginx template stripped the port from the forwarded Host header. Newly prepared sources include the correction. For the already installed Ubuntu copy, apply it without rebuilding images:

```bash
python "/mnt/c/Users/USER/Documents/Codex/2026-09-28/ver-s-la-idea-que-yo/outputs/project2-nebula/fix_proxy_port.py" --project "$HOME/crypto-project2-nebula" --reload
```

The helper backs up the existing Nginx configuration, preserves the external Host including its port, removes redundant forwarded headers, tests the configuration and reloads only Nginx. If validation or reload fails, it restores the previous file. The chat has verified the source transformation and its idempotence; the running Ubuntu reload still requires this command.

The startup warning from Requests 2.32.3 is caused by its check for `chardet<6`, while chardet 6 is installed. The warning alone does not justify downgrading urllib3. The pynvml deprecation concerns optional NVIDIA monitoring and does not require installing GPU drivers for this CPU experiment. Dependency constraints can be reconciled in the project environment after the first functional check.

| Field | First baseline |
| --- | --- |
| Federation | CFL |
| Deployment | Virtual devices / Docker |
| Total nodes | 4 |
| Node 0 | Server, start enabled |
| Nodes 1, 2, 3 | Trainers representing the three simulated medical centers |
| Topology | Star, server connected to each trainer |
| Dataset | BreastCancer |
| Distribution | IID |
| Model | MLP |
| Aggregation | FedAvg |
| Rounds | 3 |
| Local epochs | 1 |
| Accelerator | CPU |
| Differential privacy | Disabled for the initial operational baseline |
| Attacks, reputation, extra defenses, mobility | Disabled |

Four total nodes are required for three training centers plus the coordinator. The UI can reset to DFL/MNIST defaults, so the selected fields must be checked before deployment. `participant.json.example` is an internal participant template, not an importable web scenario file.

After the baseline completes, save its scenario configuration and logs. A separate DP-enabled run can then retain the same data allocation and training settings, with noise multiplier 1, clipping norm 1, delta 0.00001 and RDP accounting. NEBULA accepts noise settings and reports epsilon after training. It does not directly accept a target epsilon in this version.

## NEBULA issues identified before formal privacy claims

The audited upstream pipeline contains issues relevant to this research:

1. The BreastCancer standardizer fits the entire training pool before client allocation. A formal record-level analysis must account for this preprocessing or replace it with a fixed public calibration transform.
2. Local training and validation metrics are logged without an additional privacy mechanism. A formal transcript guarantee requires suppressing private-data metrics or providing a suitable protected/public evaluation path.
3. The bootstrap loader mixes its validation sampler with the training dataset. Reputation and extra defenses remain disabled for the initial experiment.
4. Balanced IID allocation can discard majority-class records. The CFL controller also creates partitions for all nodes, including the server. Actual used record counts must be checked before any comparison. The independent pilot's 364-record training allocation is not the NEBULA allocation.
5. Public deterministic scenario seeds and the DP randomness lifecycle require review before production-level claims.

These findings distinguish an operational noise-enabled demonstration from a complete privacy guarantee. The first NEBULA baseline can establish platform operation. Formal private experiments require the corresponding data-flow and accounting checks.

## Independent CPU pilot already executed

`pilot.py` is a small, readable feasibility experiment. It does not start NEBULA or exchange packets. It runs sequential simulated clients and aggregates weights in memory.

Its dataset split is fixed across the two runs:

- 364 training records, allocated as 122/121/121 records without discarding samples.
- 91 designated public calibration records. The scaler fits this split only.
- 114 separate public test records.
- Labels map explicitly to `0=benign`, `1=malignant`.

Both runs have three rounds, one local epoch per round, Poisson sampling at rate 1/16 and 48 optimizer steps per client. The private run retains one Opacus RDP accountant per client. Noise and sampling generators are independent and their seeds are not exported. Only public-split metrics are logged.

| Recorded test result | Baseline | DP-SGD, noise multiplier 1 |
| --- | ---: | ---: |
| Accuracy | 98.25% | 83.33% |
| Malignant F1 | 97.56% | 80.81% |
| Malignant recall | 95.24% | 95.24% |
| ROC-AUC | 0.9983 | 0.9524 |
| Federated-loop wall time | 0.3767 s | 1.9669 s |
| Accounted epsilon at delta 0.00001 | No DP bound | 3.7981 |

These are one run per setting, not averages. The measured interval includes public validation, accountant queries and console output. It excludes imports, setup and the final test. The difference combines clipping, noise and sampling variation. It does not isolate the noise effect. Secure RNG mode is disabled in this feasibility environment, so epsilon is reported as nominal Opacus training accounting, not a clinical or production privacy certification.

Raw summaries, the full public allocation manifest and validation curves remain under `results`. The plot `results/pilot_comparison.png` reads those saved measurements and does not rerun or select experiments.

To repeat the independent pilot from Windows PowerShell in this directory, select fresh output directories:

```powershell
.\.venv-pilot\Scripts\python.exe -B .\pilot.py --mode baseline --output .\results\new_baseline
.\.venv-pilot\Scripts\python.exe -B .\pilot.py --mode dp --output .\results\new_dp
```

The pilot environment reuses the existing local PyTorch installation and adds Opacus/scikit-learn locally. It has different Python/PyTorch versions from the prepared NEBULA container stack. Its recorded versions are part of each summary. This environment is not a replacement for NEBULA's Linux installation.

## Progress presentation on October 28

The defensible progress consists of a specific application, a public dataset, a checked local hardware inventory, a pinned framework revision, documented integration issues, CPU installation scripts, and two actual feasibility runs. The remaining steps are the NEBULA container build, a completed CFL baseline, a reviewed DP pipeline, clipping-only control and repeated full experiments.

## Sources

- [NEBULA source at the audited commit](https://github.com/CyberDataLab/nebula/tree/525b31c674644c653117126b5a1473a5321ac240).
- [NEBULA official installation guide](https://docs.nebula-dfl.com/installation/).
- [UCI dataset and attribution](https://doi.org/10.24432/C5DW2B): W. Wolberg, O. Mangasarian, N. Street and W. Street, 1993. CC BY 4.0.
- [Opacus scientific paper](https://arxiv.org/abs/2109.12298).
- [Original DP-SGD paper](https://arxiv.org/abs/1607.00133).

The upstream NEBULA license remains in `nebula/LICENSE`. No upstream modifications have been pushed or published.
