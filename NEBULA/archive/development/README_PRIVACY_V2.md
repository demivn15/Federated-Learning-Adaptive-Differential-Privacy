# Dynamic Differential Privacy and Readable Results

Historical v2 documentation. New experiments now follow [README_PRIVACY_ADAPTIVE.md](README_PRIVACY_ADAPTIVE.md). Existing v2 scheduled-noise results remain unchanged and must not be relabeled adaptive.

This update turns the Project 2 prototype into a controlled comparison of non-private federated learning, fixed-noise differential privacy and a public round-based noise schedule. It extends the local prepared NEBULA checkout, not the untouched upstream repository.

The completed October 28 NEBULA experiment remains a **version 1 functional test**. It completed three Non-IID rounds, but it did not enable DP. New experiments use protocol `project2-public-evaluation-v2` and must receive fresh scenario names. Version 1 and version 2 results are not interchangeable.

## What is implemented

- The NEBULA deployment panel exposes DP mode, maximum epsilon, delta, clipping norm and the relative noise schedule.
- Fixed mode keeps the same noise multiplier across rounds. Dynamic mode interpolates a public noise profile across rounds. A pre-training calibration scales either profile to the selected epsilon ceiling, accounting for each client's public sampling rate and planned optimizer steps.
- One persistent Opacus RDP accountant tracks all rounds for each client. The report includes the actual sigma, epsilon, delta, sampling rate and cumulative step count.
- The schedule never reads private accuracy, loss or gradient statistics. It is dynamic **by round**, not an accuracy-feedback controller.
- An unexpected optimizer step is rejected before the underlying optimizer updates the model. A changed sampling plan, a repeated round or a restarted process with an existing privacy journal is rejected. A failed private round produces a visible error record.
- Independent, non-exported noise and sampling generators avoid the published scenario seed. Experimental mode still employs an ordinary PyTorch RNG. The optional secure-mode checkbox requires `torchcsprng`. It does not silently fall back if that dependency is missing.
- A new “Privacidad y resultados” link on the NEBULA monitor opens a live results page with explanations, per-client budgets, noise schedules, global-model validation, final-test metrics, confusion matrix, synthetic client distributions, comparisons and JSON/CSV exports.

## Meaning of the privacy controls

| Control | Meaning |
| --- | --- |
| Maximum epsilon, epsilon | Ceiling for the cumulative record-level training bound per client in one run. Smaller values give a stricter bound for the same delta. This is not a percentage. |
| Delta | Additive term in the approximate-DP inequality. It is not the probability of disclosing a patient. |
| Clipping norm, C | Upper bound on the L2 norm of an individual example's gradient. |
| Applied noise, sigma | Gaussian noise multiplier relative to C. The optimizer adds noise to the clipped gradient sum and normalizes it. |
| Initial/final noise-profile weights | Relative shape before calibration. They are not the final sigma values. A decreasing profile changes the allocation of privacy cost over rounds. |
| Dirichlet alpha | Controls synthetic label heterogeneity. It is independent of epsilon and the DP noise schedule. |

For neighboring datasets D and D' differing in one record, the intended training mechanism satisfies a bound of the form `Pr[M(D) in S] <= exp(epsilon) * Pr[M(D') in S] + delta`, under the stated accountant assumptions. DP is a statistical privacy mechanism, not encryption. It limits information released by training, while encryption addresses access to messages or stored data. The research belongs to privacy-preserving computation. Adding DP does not claim a new cryptographic primitive.

The dynamic profile is not claimed to outperform fixed DP. That is an experimental question. Compare fixed and dynamic runs with the same epsilon ceiling, delta, C, training data, model, sampling procedure, rounds and epochs. Report the actual accounted epsilon too, since a conservative calibration can finish below the ceiling.

## Controlled public-data protocol

The source is the public WDBC dataset. Labels retain scikit-learn's mapping: **0 malignant, 1 benign**.

- 364 records form the synthetic private-training pool.
- 91 designated public records fit the StandardScaler and provide public validation.
- 114 disjoint public records form the final test set.
- A fixed seed prepares the split and synthetic center allocation. Three trainers receive disjoint partitions, with full coverage of the 364-record training pool. The server receives no private-training partition. Its placeholder dataset contains public calibration records and its role does not train.
- Non-IID mode accepts Dirichlet with alpha 0.5 as an initial setting. IID mode retains all records too. The previous Percentage/0.5 failure is avoided, and the inconsistent Percentage UI case is corrected.
- All three experiment modes sample Poisson batches. Private training loss/accuracy are suppressed. Validation reads only the public subset. Local test membership no longer depends on private training labels.
- The server evaluates the aggregated global model on public validation after each round. It evaluates the separate final test only after the final aggregation in the research result journal. NEBULA's existing diagnostic test logs also concern this designated public test, not private records.

The allocation is a **fixed public simulation design**, including public sizes and synthetic class-count displays. The privacy ledger conditions on it. It does not establish DP for selecting patients, constructing label-dependent hospital partitions, publishing real hospital class counts or revealing hospital membership. It is record-level training accounting, not hospital-level DP. Production claims additionally require secure randomness and a review of deployment access and all released outputs. Epsilon is accounted per run. Publishing multiple runs on the same real private records requires composition across those runs as well.

## Update the existing Ubuntu installation

Keep the existing project. Do not rerun the original installer over it. In the Ubuntu terminal running NEBULA, press Ctrl+C and let the platform stop. This closes its running services without deleting saved project data.

Open another Ubuntu terminal and check the reviewed update:

```bash
python3 "/mnt/c/Users/USER/Documents/Codex/2026-09-28/ver-s-la-idea-que-yo/outputs/project2-nebula/update_project2.py" --project "$HOME/crypto-project2-nebula"
```

Then apply it:

```bash
python3 "/mnt/c/Users/USER/Documents/Codex/2026-09-28/ver-s-la-idea-que-yo/outputs/project2-nebula/update_project2.py" --project "$HOME/crypto-project2-nebula" --apply
```

The updater checks the managed-installation marker and every old/new file hash. Unexpected local edits stop the update. It refuses to apply while the affected NEBULA containers are running, backs up changed source files under `project2-backups`, and copies only reviewed source files. It preserves environments, images, datasets, databases, previous scenario configurations and results. No Docker rebuild or new dependency is required for experimental RNG mode. CSPRNG mode is not installed by this updater.

Restart:

```bash
cd "$HOME/crypto-project2-nebula"
source .venv/bin/activate
PROJECT2_REPO="$HOME/crypto-project2-nebula" python -B "/mnt/c/Users/USER/Documents/Codex/2026-09-28/ver-s-la-idea-que-yo/outputs/project2-nebula/verify_privacy.py"
python app/main.py
```

The verification command includes the HDF5 writer and actual NEBULA Lightning-DP path when their dependencies are present. These two checks were skipped in the Windows pilot environment. Resolve any failing check before launching the scenarios.

Visit `http://localhost:6060`, log in and create fresh scenarios. If Docker retained stopped controller/frontend containers after an abrupt shutdown, resolve that name conflict before starting, as in the existing restart guide. Refresh the deployment page with Ctrl+F5 so the browser loads the updated DP controls.

## First three comparable scenarios

All runs: Docker, CFL, four total nodes, server node 0 with Start enabled, trainer nodes 1/2/3, Star, BreastCancer, Non-IID, Dirichlet alpha 0.5, MLP, FedAvg, CPU, three rounds and one local epoch. Disable attacks, reputation, adversarial training, feature squeezing and mobility for this controlled comparison.

| Scenario | DP | Mode | Epsilon ceiling | Delta | C | Profile start / end |
| --- | --- | --- | --- | --- | --- | --- |
| WDBC_v2_baseline_01 | Off | — | No bound | — | — | — |
| WDBC_v2_fixed_01 | On | Fixed | 8 | 0.00001 | 1 | 1 / ignored |
| WDBC_v2_dynamic_01 | On | Dynamic | 8 | 0.00001 | 1 | 1 / 0.7 |

These are initial debugging settings, not an optimized diagnostic model. The monitor displays sigma after calibration, not the profile weights. The cumulative epsilon must never reset between rounds. Each trainer can receive a different calibrated sigma because the public sampling rate differs. Under the fixed disjoint-record design and its stated assumptions, the displayed overall training bound is the maximum client bound, rather than their sum.

After each scenario finishes, open its monitor and select **Privacidad y resultados · Project 2**. Add the other two executions in the comparison section. The page checks the protocol, allocation, model, epochs and rounds, and identifies differing DP budgets. Download evidence JSON and results CSV. Do not select the old October 28 run as the v2 baseline.

## Verified evidence on November 4, 2026

Four automated tests passed with real Opacus operations: parameter validation, fixed/public schedules, independent public calibration and disjoint Non-IID splits, repeated private rounds with a persistent accountant, epsilon monotonicity, an epsilon ceiling, rejection of an extra optimizer step, and malignant-positive evaluation metrics. Python syntax checks passed across the prepared source tree and the three relevant Jinja templates parsed. The live results page loaded actual smoke-test journals in the browser.

The separate `research_smoke.py` completed baseline, fixed and dynamic **standalone CPU integration tests**, three rounds each. These tests exercise the new data preparation, policy, Opacus hooks, aggregation, evaluator, journals and result reader. They do not run NEBULA networking, Docker or Lightning. They have a different Python/PyTorch runtime from the Ubuntu images. Their source label is visible in the preview and saved evidence.

One measured run per mode produced:

| Metric | Baseline | Fixed DP | Dynamic DP |
| --- | ---: | ---: | ---: |
| Final test accuracy | 94.74% | 64.91% | 63.16% |
| Malignant recall | 97.62% | 4.76% | 0.00% |
| Malignant F1 | 93.18% | 9.09% | 0.00% |
| Actual maximum epsilon | No DP bound | 7.9414 | 7.9422 |

The poor private recall is an actual finding of these short smoke tests. A model predicting every record as benign achieves 63.16% accuracy on this test because it contains 72 benign and 42 malignant records. The results page displays recall and false negatives alongside accuracy so this behavior is visible. High ROC-AUC in a smoke run does not establish acceptable decisions at the fixed classification threshold. Repetitions and tuning must rely on the public validation split rather than selecting configurations on this test result.

The version 2 update has **not yet completed a NEBULA container scenario in the student's Ubuntu installation**. Its local controller was unavailable during this update. Full integration, HDF5 partition writing and the Lightning path still require the updated Ubuntu execution. The older completed NEBULA scenario cannot validate these new paths.

## Sources

- [Deep Learning with Differential Privacy, Abadi et al., ACM CCS 2016](https://arxiv.org/abs/1607.00133).
- [Opacus documentation and privacy accounting](https://opacus.ai/docs/faq).
- [Opacus PrivacyEngine source and secure mode](https://opacus.ai/api/_modules/opacus/privacy_engine.html).
- [Communication-Efficient Learning of Deep Networks from Decentralized Data, McMahan et al.](https://proceedings.mlr.press/v54/mcmahan17a.html).
- [UCI WDBC dataset](https://doi.org/10.24432/C5DW2B).
- [Audited NEBULA upstream revision](https://github.com/CyberDataLab/nebula/tree/525b31c674644c653117126b5a1473a5321ac240).
